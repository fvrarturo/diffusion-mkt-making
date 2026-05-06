"""Fine-tune the conditional diffusion generator on cleaned event-tape windows.

Composes data + generator + cluster configs via Hydra. Streams metrics to W&B
when enabled. Writes checkpoints under cluster.checkpoint_root, with a sidecar
manifest tying every checkpoint to a git SHA + config hash.

Usage (local smoke):
    python scripts/11_finetune_generator.py +smoke=true

Usage (cluster):
    sbatch slurms/ph2_finetune.slurm \\
        cluster=engaging \\
        data=intc_top \\
        generator.training.max_epochs=30
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from diffmm.data.dataset import (
    LOBWindowDataset, RandomWindowDataset, NormStats, N_FEATURES,
)
from diffmm.generator.sample import DDIMSchedule
from diffmm.generator.train import DDPMTrainer, CFGDropoutCurriculum
from diffmm.generator.trades_adapter import build_generator
from diffmm.io.manifest import write_meta
from diffmm.utils.logging import get_logger
from diffmm.utils.seeding import seed_all

log = get_logger("ph2_finetune")


def _gather_tape_paths(processed_dir: Path, split: str, ticker: str) -> list[Path]:
    return sorted((processed_dir / split).glob(f"{ticker}_*.parquet"))


def _build_schedule(cfg: DictConfig) -> DDIMSchedule:
    pred_type = cfg.generator.schedule.get("prediction_type", "eps")
    if cfg.generator.schedule.type == "cosine":
        return DDIMSchedule.cosine(T=cfg.generator.schedule.T, prediction_type=pred_type)
    if cfg.generator.schedule.type == "linear":
        return DDIMSchedule.linear(T=cfg.generator.schedule.T, prediction_type=pred_type)
    raise ValueError(f"unknown schedule: {cfg.generator.schedule.type}")


def _build_curriculum(cfg: DictConfig) -> CFGDropoutCurriculum | None:
    cur_cfg = cfg.generator.training.get("cfg_dropout_curriculum", None)
    if cur_cfg is None:
        return None
    return CFGDropoutCurriculum(
        enabled=bool(cur_cfg.get("enabled", True)),
        initial=float(cur_cfg.get("initial", 1.0)),
        final=float(cur_cfg.get("final", cfg.generator.training.cfg_dropout)),
        ramp_start_epoch=int(cur_cfg.get("ramp_start_epoch", 2)),
        ramp_end_epoch=int(cur_cfg.get("ramp_end_epoch", 5)),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_all(cfg.seed)
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    if cfg.get("smoke", False):
        log.info("SMOKE mode — using RandomWindowDataset, no real data")
        train_ds = RandomWindowDataset(n_windows=256, window_length=cfg.data.window_length)
        val_ds = RandomWindowDataset(n_windows=64, window_length=cfg.data.window_length, seed=1)
    else:
        import time

        processed_dir = Path(cfg.data.processed_dir)
        train_paths = _gather_tape_paths(processed_dir, "train", cfg.data.ticker)
        val_paths = _gather_tape_paths(processed_dir, "val", cfg.data.ticker)
        if not train_paths:
            raise FileNotFoundError(
                f"no training tapes found under {processed_dir}/train/ for {cfg.data.ticker} "
                "— run the Phase-1 cleaner first"
            )
        log.info("found %d train tapes, %d val tapes", len(train_paths), len(val_paths))

        copula_cfg = cfg.generator.get("copula", None)
        use_copula = bool(copula_cfg is not None and copula_cfg.get("enabled", False))

        if use_copula:
            # Phase E (v8) — train on Gaussianized data via empirical-CDF transform.
            from diffmm.data.copula_transform import CopulaTransform
            copula_path = Path(cfg.data.norm_stats_file).with_suffix(".copula.json")
            if copula_path.exists():
                norm = CopulaTransform.load(copula_path)
                log.info("loaded copula transform from %s", copula_path)
            else:
                log.info("fitting copula transform on training tapes (%d files)", len(train_paths))
                t0 = time.time()
                discrete = copula_cfg.get("discrete_features", None)
                if discrete is not None:
                    discrete = tuple(discrete)
                norm = CopulaTransform.fit(
                    train_paths,
                    discrete_features=discrete,
                    clip_quantile=float(copula_cfg.get("clip_quantile", 0.001)),
                    max_samples_per_feature=int(copula_cfg.get("max_samples_per_feature", 5_000_000)),
                )
                norm.save(copula_path)
                log.info(
                    "fit copula in %.1fs → %s  (discrete_features=%s, clip=%g)",
                    time.time() - t0, copula_path,
                    norm.discrete_features, norm.clip_quantile,
                )
        else:
            norm_path = Path(cfg.data.norm_stats_file)
            if norm_path.exists():
                norm = NormStats.load(norm_path)
                log.info("loaded norm stats from %s", norm_path)
            else:
                log.info("computing norm stats on training tapes (%d files)", len(train_paths))
                from diffmm.data.dataset import compute_norm_stats
                t0 = time.time()
                norm = compute_norm_stats(train_paths)
                norm.save(norm_path)
                log.info("computed norm stats in %.1fs → %s", time.time() - t0, norm_path)

        log.info("loading %d train tapes into memory", len(train_paths))
        t0 = time.time()
        train_ds = LOBWindowDataset(
            train_paths, window_length=cfg.data.window_length,
            stride=cfg.data.stride, norm_stats=norm,
        )
        log.info("train dataset ready (%d windows) in %.1fs", len(train_ds), time.time() - t0)

        t0 = time.time()
        val_ds = LOBWindowDataset(
            val_paths, window_length=cfg.data.window_length,
            stride=cfg.data.window_length, norm_stats=norm,  # non-overlapping
        )
        log.info("val dataset ready (%d windows) in %.1fs", len(val_ds), time.time() - t0)

    log.info("train windows: %d   val windows: %d", len(train_ds), len(val_ds))

    edm_cfg = cfg.generator.get("edm", None)
    use_edm = bool(edm_cfg is not None and edm_cfg.get("enabled", False))
    conditioning_type = cfg.generator.model.get("conditioning_type", "film")
    log.info("conditioning_type=%s  edm_enabled=%s", conditioning_type, use_edm)

    if use_edm:
        # Phase D path — EDM training. The DDIM schedule above is ignored.
        from diffmm.generator.edm import (
            EDMSchedule, EDMTrainer, build_edm_generator,
        )
        edm_schedule = EDMSchedule(
            sigma_min=float(edm_cfg.get("sigma_min", 0.002)),
            sigma_max=float(edm_cfg.get("sigma_max", 80.0)),
            sigma_data=float(edm_cfg.get("sigma_data", 0.5)),
            rho=float(edm_cfg.get("rho", 7.0)),
            P_mean=float(edm_cfg.get("P_mean", -1.2)),
            P_std=float(edm_cfg.get("P_std", 1.2)),
        )
        log.info(
            "EDM schedule: σ_min=%.4f σ_max=%.1f σ_data=%.3f ρ=%.1f P_mean=%.2f P_std=%.2f",
            edm_schedule.sigma_min, edm_schedule.sigma_max, edm_schedule.sigma_data,
            edm_schedule.rho, edm_schedule.P_mean, edm_schedule.P_std,
        )
        gen = build_edm_generator(
            n_features=N_FEATURES,
            d_model=cfg.generator.model.embed_dim * 2,
            num_heads=8,
            depth=8,
            max_seq_len=cfg.data.window_length + 8,
            embed_dim=cfg.generator.model.embed_dim,
            n_categories_per_axis=tuple(cfg.generator.model.n_categories_per_axis),
            conditioning_type=conditioning_type,
            sigma_data=edm_schedule.sigma_data,
        )
        pl_module = EDMTrainer(
            generator=gen,
            schedule=edm_schedule,
            learning_rate=cfg.generator.training.learning_rate,
            weight_decay=cfg.generator.training.weight_decay,
            cfg_dropout=cfg.generator.training.cfg_dropout,
            cfg_dropout_curriculum=_build_curriculum(cfg),
        )
    else:
        # DDPM/v-pred path (v2-v5).
        gen = build_generator(
            n_features=N_FEATURES,
            d_model=cfg.generator.model.embed_dim * 2,
            num_heads=8,
            depth=8,
            max_seq_len=cfg.data.window_length + 8,
            embed_dim=cfg.generator.model.embed_dim,
            num_diffusionsteps=cfg.generator.schedule.T,
            n_categories_per_axis=tuple(cfg.generator.model.n_categories_per_axis),
            conditioning_type=conditioning_type,
        )
        schedule = _build_schedule(cfg)
        pl_module = DDPMTrainer(
            generator=gen,
            schedule=schedule,
            learning_rate=cfg.generator.training.learning_rate,
            weight_decay=cfg.generator.training.weight_decay,
            cfg_dropout=cfg.generator.training.cfg_dropout,
            freeze_backbone=cfg.generator.training.freeze_backbone,
            min_snr_gamma=cfg.generator.training.get("min_snr_gamma", None),
            prediction_type=cfg.generator.schedule.get("prediction_type", "eps"),
            cfg_dropout_curriculum=_build_curriculum(cfg),
        )

    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

    # Prefer the training-block num_workers (CLI-override-friendly); fall back
    # to the cluster default if not set. This is the override that smoke runs
    # rely on (num_workers=0 disables multiprocessing entirely).
    n_workers = int(
        cfg.generator.training.get("num_workers", cfg.cluster.num_workers)
    )
    log.info("DataLoader num_workers=%d", n_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.generator.training.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(cfg.cluster.device == "cuda"),
        persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.generator.training.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=(cfg.cluster.device == "cuda"),
        persistent_workers=(n_workers > 0),
    )

    ckpt_dir = Path(cfg.cluster.checkpoint_root) / cfg.experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            # Filename without '=' — Hydra CLI treats '=' as an override delimiter,
            # so checkpoint paths that contain '=' can't be passed through.
            filename="ckpt-epoch{epoch:03d}-valloss{val/loss:.4f}",
            monitor=cfg.generator.checkpointing.monitor,
            mode=cfg.generator.checkpointing.mode,
            save_top_k=cfg.generator.checkpointing.save_top_k,
            every_n_epochs=cfg.generator.checkpointing.every_n_epochs,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor=cfg.generator.checkpointing.monitor,
            mode=cfg.generator.checkpointing.mode,
            patience=cfg.generator.training.early_stopping_patience,
        ),
    ]

    from pytorch_lightning.loggers import CSVLogger
    loggers: list = [CSVLogger(
        save_dir=Path(cfg.cluster.results_root) / "lightning_logs",
        name=cfg.experiment_name,
    )]
    if cfg.wandb.enabled:
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(WandbLogger(
            project=cfg.wandb.project, entity=cfg.wandb.entity,
            mode=cfg.wandb.mode, name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        ))

    trainer_kwargs = dict(
        max_epochs=cfg.generator.training.max_epochs,
        accelerator=("gpu" if cfg.cluster.device == "cuda" else "cpu"),
        devices=1,
        precision=cfg.generator.training.precision if cfg.cluster.device == "cuda" else 32,
        gradient_clip_val=cfg.generator.training.gradient_clip_val,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=20,
        enable_progress_bar=False,  # TQDM doesn't render well in SLURM logs
    )
    # Optional limits — set for smoke runs, leave null for full training.
    if cfg.generator.training.get("max_steps") is not None:
        trainer_kwargs["max_steps"] = int(cfg.generator.training.max_steps)
    if cfg.generator.training.get("limit_train_batches") is not None:
        trainer_kwargs["limit_train_batches"] = int(cfg.generator.training.limit_train_batches)
    if cfg.generator.training.get("limit_val_batches") is not None:
        trainer_kwargs["limit_val_batches"] = int(cfg.generator.training.limit_val_batches)
    trainer = pl.Trainer(**trainer_kwargs)
    # Resume from a prior checkpoint when +resume_from=<path> is set. Lightning
    # restores optimizer state, LR scheduler, epoch counter, and global_step
    # (not just model weights), so the continuation is indistinguishable from
    # one uninterrupted run — useful when SLURM time caps force splitting a
    # long training into 6h chunks.
    resume_path = cfg.get("resume_from", None)
    if resume_path:
        log.info("RESUMING from checkpoint: %s", resume_path)
    trainer.fit(pl_module, train_loader, val_loader, ckpt_path=resume_path)

    best_path = callbacks[0].best_model_path
    if best_path:
        write_meta(
            best_path, seed=cfg.seed, config=cfg,
            creator_script="scripts/11_finetune_generator.py",
            input_artifacts=[str(p) for p in (train_paths if not cfg.get("smoke") else [])],
        )
        log.info("best checkpoint: %s", best_path)


if __name__ == "__main__":
    main()
