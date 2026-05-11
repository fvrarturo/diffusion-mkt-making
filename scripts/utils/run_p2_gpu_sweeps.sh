#!/bin/bash
# P2 — GPU-heavy ablation sweeps. Each one is independently optional but each
# adds a section/figure to the report. Run after P0 has settled.
#
#   P2.G1 — guidance-weight sweep on v2 (sample 50 tapes per w, validate)
#   P2.G2 — per-timestep denoising quality for v2 / v5 / v9
#   P2.G3 — regime-embedding t-SNE (CPU; small)
#   P2.G4 — predictive-score robustness to LSTM hidden-dim
#
# Run on the cluster:
#   tmux new -s p2
#   ./scripts/utils/run_p2_gpu_sweeps.sh \
#       2>&1 | tee logs/p2_$(date +%Y%m%d_%H%M%S).log
#   # Ctrl-b d
#
# Total wall time: ~4-5 h (most of it is P2.G1 sampling).

set -uo pipefail

CKPT_ROOT="$HOME/orcd/scratch/diffmm/checkpoints"
SYNTH_BASE="$HOME/orcd/scratch/diffmm"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p logs

# Resolve a "best" checkpoint by valloss filename
best_ckpt() {
    local d="$1"
    [[ -d "$d" ]] || return 0
    ls -1 "$d"/ckpt-epoch*-valloss*.ckpt 2>/dev/null \
        | awk -F'valloss' '{print $2, $0}' \
        | sort -n | head -1 | awk '{print $2}'
}

# ───────────────────────────────────────────────────────────────────────────
# P2.G1 — guidance-weight sweep on v2
# ───────────────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════════════════════════"
log "=== P2.G1 — guidance-weight sweep on v2 ==="
log "════════════════════════════════════════════════════════════════"

V2_CKPT=$(best_ckpt "$CKPT_ROOT/intc_v2")
if [[ -z "$V2_CKPT" ]]; then
    log "WARN: no v2 checkpoint at $CKPT_ROOT/intc_v2 — skipping P2.G1"
else
    GUIDANCE_WEIGHTS=(0 0.5 1.0 2.0 4.0 8.0)
    for w in "${GUIDANCE_WEIGHTS[@]}"; do
        SYNTH_OUT="$SYNTH_BASE/synthetic/intc_v2_guidance_${w}"
        if [[ -d "$SYNTH_OUT" ]] && [[ $(find "$SYNTH_OUT" -name '*.parquet' | wc -l) -ge 50 ]]; then
            log "  [w=$w] synth dir already has parquets — skipping sample"
            continue
        fi
        log "  [w=$w] EXEC: sbatch --array=0-49 ph2_sample_cpu.slurm guidance_weight=$w"
        sbatch --parsable --array=0-49 \
            slurms/ph2_sample_cpu.slurm \
            generator=ddim_sample data=intc_top \
            generator.checkpoint_path="$V2_CKPT" \
            generator.guidance_weight="$w" \
            cluster.synthetic_root="$SYNTH_OUT" \
            || log "  WARN: sbatch failed for w=$w"
    done

    log "  Note: validation step (G1 + predictive + PCA per w) deferred."
    log "        After all guidance-array jobs drain, run:"
    log "          for w in 0 0.5 1.0 2.0 4.0 8.0; do"
    log "              python scripts/50_validate_generator.py --parts B C D \\"
    log "                  --real-dir data/processed/val \\"
    log "                  --synth-dir $SYNTH_BASE/synthetic/intc_v2_guidance_\$w/INTC \\"
    log "                  --ticker INTC --out-dir results/_archive/v2_guidance_\$w/validation"
    log "          done"
fi

# ───────────────────────────────────────────────────────────────────────────
# P2.G2 — per-timestep denoising quality (small custom script)
# ───────────────────────────────────────────────────────────────────────────
log ""
log "════════════════════════════════════════════════════════════════"
log "=== P2.G2 — per-timestep denoising quality for v2/v5/v9 ==="
log "════════════════════════════════════════════════════════════════"

if [[ ! -f scripts/85_per_timestep_denoising.py ]]; then
    log "  scripts/85_per_timestep_denoising.py not found — skipping P2.G2"
    log "  (deferred: write a small script that loads ckpt, noises real windows"
    log "   to t in {10,50,100,200,500,800}, denoises one step, computes per-feature MSE)"
else
    log "  EXEC: sbatch ph6_validate_gpu.slurm (override with per-timestep mode)"
    # Submit one GPU job per checkpoint
    for model in v2 v5 v9; do
        ckpt=$(best_ckpt "$CKPT_ROOT/intc_$model")
        if [[ -z "$ckpt" ]]; then
            log "  [$model] no checkpoint — skipping"
            continue
        fi
        log "  [$model] EXEC: sbatch --gres=gpu:1 --mem=32G --time=01:00:00 -- python scripts/85_per_timestep_denoising.py --ckpt $ckpt --label $model --out results/per_timestep_$model"
        sbatch --parsable --gres=gpu:1 --mem=32G --time=01:00:00 \
            --wrap="python scripts/85_per_timestep_denoising.py \
                    --ckpt $ckpt --label $model \
                    --real-dir data/processed/val \
                    --out results/per_timestep_$model" \
            || log "  WARN: P2.G2 [$model] sbatch failed"
    done
fi

# ───────────────────────────────────────────────────────────────────────────
# P2.G3 — regime-embedding t-SNE (CPU only, runs on login node)
# ───────────────────────────────────────────────────────────────────────────
log ""
log "════════════════════════════════════════════════════════════════"
log "=== P2.G3 — regime-embedding t-SNE on v2/v5/v9 (login-node CPU) ==="
log "════════════════════════════════════════════════════════════════"

CKPT_ARGS=()
for model in v2 v5 v9; do
    c=$(best_ckpt "$CKPT_ROOT/intc_$model")
    if [[ -n "$c" ]]; then
        CKPT_ARGS+=("$c:$model")
    fi
done
if [[ ${#CKPT_ARGS[@]} -gt 0 ]]; then
    log "  EXEC: python scripts/84_regime_embedding_tsne.py --ckpts ${CKPT_ARGS[*]}"
    python scripts/84_regime_embedding_tsne.py --ckpts "${CKPT_ARGS[@]}" \
        || log "  WARN: P2.G3 failed (non-fatal)"
fi

# ───────────────────────────────────────────────────────────────────────────
# P2.G4 — predictive score sensitivity to LSTM hidden_dim
# ───────────────────────────────────────────────────────────────────────────
log ""
log "════════════════════════════════════════════════════════════════"
log "=== P2.G4 — predictive score LSTM-size sweep ==="
log "════════════════════════════════════════════════════════════════"

# Note: scripts/63_predictive_score.py currently bakes hidden_dim. Need to
# add --hidden-dim flag (1-line change). Sketch:
#   ap.add_argument("--hidden-dim", type=int, default=64)
#   ... use args.hidden_dim instead of the hard-coded 64
# Then:
#   for h in 32 64 128; do
#     python scripts/63_predictive_score.py \
#         --real-train-dir data/processed/train --real-test-dir data/processed/val \
#         --synth-dirs ... --labels ... --hidden-dim $h \
#         --out results/predictive_lstm_h$h
#   done
# Compare ranking stability across the three values.

log "  Patch needed: add --hidden-dim flag to scripts/63_predictive_score.py"
log "  Then sweep h ∈ {32, 64, 128} per the comment block above."
log "  Skipping execution until that flag exists."

log ""
log "════════════════════════════════════════════════════════════════"
log "=== P2 GPU sweeps DONE (sampling jobs may still be queued) ==="
log "════════════════════════════════════════════════════════════════"
log "Check  squeue -u \$USER  for pending/running jobs."
log "Pull results back to Mac after all complete:"
log "  rsync -avz favara@orcd-login.mit.edu:diff_mm/results/{_archive/v2_guidance_*,per_timestep_*,predictive_lstm_*}/  ~/Desktop/diff-market-making/results/"
