"""Generate synthetic event-tape windows by reverse-DDIM with classifier-free guidance.

Two modes:
  1. Single-call:       python scripts/13_sample_synthetic.py +regime=toxic +seed=42
  2. SLURM-array:       --task-id 0..399 → unpacks (regime, seed) from a manifest

Output: data/synthetic/{ticker}/{regime}/day_{seed:04d}_{i:03d}.parquet (canonical schema).
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from diffmm.data.dataset import N_FEATURES, NormStats
from diffmm.generator.decode import decode_batch_to_parquet
from diffmm.generator.sample import DDIMSchedule, ddim_sample
from diffmm.generator.trades_adapter import build_generator
from diffmm.io.manifest import write_meta
from diffmm.utils.logging import get_logger
from diffmm.utils.seeding import seed_all

log = get_logger("ph2_sample")


def _load_generator_from_checkpoint(ckpt_path: str, cfg: DictConfig, device: str):
    gen = build_generator(
        n_features=N_FEATURES,
        d_model=cfg.generator.model.embed_dim * 2,
        num_heads=8,
        depth=8,
        max_seq_len=cfg.data.window_length + 8,
        embed_dim=cfg.generator.model.embed_dim,
        num_diffusionsteps=cfg.generator.schedule.T,
        n_categories_per_axis=tuple(cfg.generator.model.n_categories_per_axis),
    )
    state = torch.load(ckpt_path, map_location=device)
    sd = state.get("state_dict", state)
    sd = {k.removeprefix("generator."): v for k, v in sd.items() if k.startswith("generator.")}
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    if missing or unexpected:
        log.warning("checkpoint load: missing=%s unexpected=%s", missing, unexpected)
    gen.to(device).eval()
    return gen


def _resolve_task(cfg: DictConfig) -> tuple[str, int]:
    """Either (regime, seed) is given directly, or unpacked from a manifest."""
    if "regime" in cfg and "seed" in cfg:
        return str(cfg.regime), int(cfg.seed)
    task_id = cfg.get("task_id")
    manifest_path = cfg.get("manifest")
    if task_id is None or manifest_path is None:
        raise ValueError("must pass either +regime=... +seed=... or +task_id=... +manifest=...")
    manifest = json.loads(Path(manifest_path).read_text())
    entry = manifest[int(task_id)]
    return entry["regime"], int(entry["seed"])


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_all(cfg.seed)
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    sample_cfg = cfg.generator  # ddim_sample.yaml lives under generator group when selected
    if not sample_cfg.get("checkpoint_path"):
        raise ValueError("generator.checkpoint_path must be set (path to a finetune checkpoint)")
    if cfg.cluster.device == "cuda" and not torch.cuda.is_available():
        log.warning("cluster.device=cuda but CUDA not available; falling back to cpu")
        device = "cpu"
    else:
        device = cfg.cluster.device

    regime, seed = _resolve_task(cfg)
    log.info("task: regime=%s seed=%d device=%s", regime, seed, device)

    cond_vec = sample_cfg.regime_to_condition[regime]
    cond = torch.tensor([list(cond_vec)] * sample_cfg.batch_size, dtype=torch.long, device=device)

    gen = _load_generator_from_checkpoint(sample_cfg.checkpoint_path, cfg, device=device)
    null = gen.regime_embed.null_condition(sample_cfg.batch_size, device=device)

    pred_type = cfg.generator.schedule.get("prediction_type", "eps")
    schedule = (
        DDIMSchedule.cosine(T=cfg.generator.schedule.T, prediction_type=pred_type)
        if cfg.generator.schedule.type == "cosine"
        else DDIMSchedule.linear(T=cfg.generator.schedule.T, prediction_type=pred_type)
    )
    log.info("prediction_type=%s", pred_type)

    n_features = sample_cfg.get("n_features", N_FEATURES)
    L = sample_cfg.get("window_length", cfg.data.window_length)

    samples = ddim_sample(
        eps_theta=gen.predict_noise,
        shape=(sample_cfg.batch_size, L, n_features),
        schedule=schedule,
        condition=cond,
        null_condition=null,
        guidance_weight=sample_cfg.guidance_weight,
        n_steps=sample_cfg.n_steps,
        device=device,
        seed=seed,
        eta=sample_cfg.eta,
        x0_clip=sample_cfg.get("x0_clip"),
    )

    if not Path(cfg.data.norm_stats_file).exists():
        raise FileNotFoundError(
            f"norm stats not found at {cfg.data.norm_stats_file} — required for decode "
            "(provides anchor_mid). Run finetune once first to generate them."
        )
    norm = NormStats.load(cfg.data.norm_stats_file)
    # Allow decode-time override of the anchor price without retraining.
    override = sample_cfg.get("anchor_mid_override", None)
    if override is not None:
        from dataclasses import replace
        norm = replace(norm, anchor_mid=float(override))
        log.info("anchor_mid overridden: %.2f (was %.2f)", norm.anchor_mid, NormStats.load(cfg.data.norm_stats_file).anchor_mid)
    out_dir = Path(cfg.cluster.synthetic_root)

    paths = decode_batch_to_parquet(
        samples,
        cond,
        out_dir=out_dir,
        ticker=cfg.data.ticker,
        start_ts_ns=int(1_700_000_000_000_000_000) + seed * 86_400_000_000_000,
        seed=seed,
        regime_label=regime,
        norm_stats=norm,
    )
    for p in paths:
        write_meta(
            p, seed=seed, config=cfg,
            creator_script="scripts/13_sample_synthetic.py",
            input_artifacts=[sample_cfg.checkpoint_path],
            extra={"regime": regime, "guidance_weight": sample_cfg.guidance_weight},
        )
    log.info("wrote %d synthetic-window parquet files to %s", len(paths), out_dir)


if __name__ == "__main__":
    main()
