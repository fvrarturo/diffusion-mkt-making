"""Phase F Model A — cascade sampling: Stage 1 (book features) → Stage 2 (mid_return).

Stage 1 = a previously-trained generator that produces book features (typically v2,
i.e. an ε-prediction FiLM model trained on all 8 features). We use ONLY its bid/ask,
size, and trade outputs and discard its mid_return.

Stage 2 = a generator trained with `cascade.noise_mask=[0,0,0,0,0,0,0,1]` and the
matching `cascade.loss_mask`, so it learned to predict mid_return given clean book
features as conditioning context.

Usage example (cluster):
    python scripts/13_sample_cascade.py \
        cluster=engaging \
        data=intc_top \
        generator=ddim_sample \
        +cascade_stage1_ckpt=/path/to/intc_v2/ckpt-best.ckpt \
        +cascade_stage2_ckpt=/path/to/intc_v10_stage2/ckpt-best.ckpt \
        +regime=base +seed=0

Or via SLURM array (manifest-driven), pass `+task_id=$SLURM_ARRAY_TASK_ID +manifest=...`
exactly like 13_sample_synthetic.py. See `slurms/ph2_sample_cpu.slurm` — point it at
this script instead of 13_sample_synthetic.py.

Output: same canonical Parquet schema as 13_sample_synthetic.py, written to
`cluster.synthetic_root/{ticker}/{regime}/day_{seed:04d}_{i:03d}.parquet`.
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from diffmm.data.dataset import N_FEATURES, NormStats
from diffmm.generator.decode import decode_batch_to_parquet
from diffmm.generator.sample import DDIMSchedule, ddim_sample, ddim_sample_cascade
from diffmm.generator.trades_adapter import build_generator
from diffmm.io.manifest import write_meta
from diffmm.utils.logging import get_logger
from diffmm.utils.seeding import seed_all

log = get_logger("ph2_sample_cascade")


def _load_generator(ckpt_path: str, cfg: DictConfig, device: str):
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

    sample_cfg = cfg.generator
    s1_ckpt = cfg.get("cascade_stage1_ckpt")
    s2_ckpt = cfg.get("cascade_stage2_ckpt")
    if not s1_ckpt or not s2_ckpt:
        raise ValueError("must pass +cascade_stage1_ckpt=<path> AND +cascade_stage2_ckpt=<path>")

    device = cfg.cluster.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("cluster.device=cuda but CUDA not available; falling back to cpu")
        device = "cpu"

    regime, seed = _resolve_task(cfg)
    log.info("task: regime=%s seed=%d device=%s", regime, seed, device)

    cond_vec = sample_cfg.regime_to_condition[regime]
    cond = torch.tensor([list(cond_vec)] * sample_cfg.batch_size, dtype=torch.long, device=device)

    pred_type = cfg.generator.schedule.get("prediction_type", "eps")
    schedule = (
        DDIMSchedule.cosine(T=cfg.generator.schedule.T, prediction_type=pred_type)
        if cfg.generator.schedule.type == "cosine"
        else DDIMSchedule.linear(T=cfg.generator.schedule.T, prediction_type=pred_type)
    )
    log.info("prediction_type=%s", pred_type)

    n_features = sample_cfg.get("n_features", N_FEATURES)
    L = sample_cfg.get("window_length", cfg.data.window_length)

    # ─── Stage 1: full-feature sampling (v2-style) ───
    log.info("Stage 1: loading %s", s1_ckpt)
    gen1 = _load_generator(s1_ckpt, cfg, device=device)
    null1 = gen1.regime_embed.null_condition(sample_cfg.batch_size, device=device)
    stage1_x0 = ddim_sample(
        eps_theta=gen1.predict_noise,
        shape=(sample_cfg.batch_size, L, n_features),
        schedule=schedule,
        condition=cond,
        null_condition=null1,
        guidance_weight=sample_cfg.guidance_weight,
        n_steps=sample_cfg.n_steps,
        device=device,
        seed=seed,
        eta=sample_cfg.eta,
        x0_clip=sample_cfg.get("x0_clip"),
    )
    log.info("Stage 1 sample: shape=%s mid_return.std=%.4g (will be replaced by Stage 2)",
             tuple(stage1_x0.shape), stage1_x0[..., 7].std().item())

    # Force-free Stage 1 memory before loading Stage 2. Without this, PyTorch
    # modules retain references via hooks/buffers and Python's GC doesn't run
    # before Stage 2's allocation, OOM-killing tasks at 8G mem (observed
    # 2026-05-09 across cascade jobs 13648320 and 13636992 even after the
    # @torch.no_grad() fix). Explicit gc.collect() + del + (cuda) empty_cache
    # cuts Stage-1 memory before Stage 2 starts.
    import gc
    del gen1, null1
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ─── Stage 2: cascade refinement of mid_return ───
    log.info("Stage 2: loading %s", s2_ckpt)
    gen2 = _load_generator(s2_ckpt, cfg, device=device)
    null2 = gen2.regime_embed.null_condition(sample_cfg.batch_size, device=device)
    # Mask: 1 = diffuse this channel via Stage 2; 0 = keep clean from Stage 1.
    # Default: only mid_return (index 7) is diffused. Override via
    # +cascade_noise_mask=[0,0,0,0,0,0,0,1] if needed.
    mask_list = list(cfg.get("cascade_noise_mask", [0, 0, 0, 0, 0, 0, 0, 1]))
    noise_mask_t = torch.tensor(mask_list, dtype=torch.float32, device=device)
    log.info("cascade noise_mask = %s", mask_list)

    final_x0 = ddim_sample_cascade(
        eps_theta=gen2.predict_noise,
        clean_features=stage1_x0,
        noise_mask=noise_mask_t,
        schedule=schedule,
        condition=cond,
        null_condition=null2,
        guidance_weight=sample_cfg.guidance_weight,
        n_steps=sample_cfg.n_steps,
        device=device,
        seed=seed + 1,                  # different seed than Stage 1 noise
        eta=sample_cfg.eta,
        x0_clip=sample_cfg.get("x0_clip"),
    )
    log.info("Stage 2 sample: shape=%s mid_return.std=%.4g (post-cascade)",
             tuple(final_x0.shape), final_x0[..., 7].std().item())

    # ─── Decode (uses NormStats; cascade sampling is in normalized space) ───
    if not Path(cfg.data.norm_stats_file).exists():
        raise FileNotFoundError(
            f"norm stats not found at {cfg.data.norm_stats_file} — required for decode"
        )
    norm = NormStats.load(cfg.data.norm_stats_file)
    override = sample_cfg.get("anchor_mid_override", None)
    if override is not None:
        from dataclasses import replace
        prev = norm.anchor_mid
        norm = replace(norm, anchor_mid=float(override))
        log.info("anchor_mid overridden: %.2f (was %.2f)", norm.anchor_mid, prev)
    out_dir = Path(cfg.cluster.synthetic_root)

    paths = decode_batch_to_parquet(
        final_x0, cond,
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
            creator_script="scripts/13_sample_cascade.py",
            input_artifacts=[s1_ckpt, s2_ckpt],
            extra={"regime": regime, "guidance_weight": sample_cfg.guidance_weight},
        )
    log.info("wrote %d cascade-synth parquet files to %s", len(paths), out_dir)


if __name__ == "__main__":
    main()
