#!/bin/bash
# Cross-ticker v2 vs v5 pipeline: data prep → train → sample → validate → score.
# Two tickers (TSLA, SPY) × two architectures (v2 = eps+film, v5 = v+adaln_zero).
#
# Run on the cluster login node in tmux so it survives detach:
#
#   tmux new -s cross_ticker
#   ./scripts/utils/run_cross_ticker_v2_v5.sh \
#       2>&1 | tee logs/cross_ticker_$(date +%Y%m%d_%H%M%S).log
#   # Ctrl-b d to detach
#
# Total wall time (overnight, no resumes):
#   Stage 0 (data prep, both tickers parallel): ~3-5h
#   Stage 1 (train, 4 jobs parallel — queues if QOS-limited):  6h cap each
#   Stage 2 (sample, 4 jobs sequential):  ~30 min × 4 = 2h
#   Stage 3 (validate B/C/D, 4 jobs parallel/ticker):  ~45 min × 2 = 1.5h
#   Stage 4 (TRADES + predictive score, per ticker):  ~3h × 2 = 6h
#                                                                ─────
#   Worst-case total                                              ~18h
#   Realistic (cached data, training fits 6h): ~10h
#
# Why v2 and v5 only:
#   - v2 (eps + FiLM)        ↔ v5 (v-pred + AdaLN-Zero)
#   - These are the extremes of the architectural axis on INTC.
#   - 4 (ticker, model) data points test whether the three-axis Pareto tension
#     observed on INTC generalizes across microstructure regimes (small-tick TSLA,
#     ETF SPY) or is INTC-specific.
#
# ── Robustness notes ────────────────────────────────────────────────────────
# - `set -uo pipefail` (NOT -e): we want to continue past stage failures with
#   explicit log lines, since hours of upstream work would otherwise be lost.
# - All sbatch invocations use `--parsable` and we wait via squeue polling.
# - Each stage echoes `EXEC: ...` BEFORE running so failures are easy to
#   reproduce.
# - No nested tees — outer tmux `tee` (in the launch invocation) captures
#   everything. (Nested tees ate output silently in past runs.)
# - Idempotent: each stage skips if its outputs already exist, so re-running
#   after a partial completion just resumes from where it stopped.

set -uo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
TICKERS=(TSLA SPY)
MODELS=(v2 v5)

# Architecture overrides — must match the v2/v5 INTC runs documented in
# Work6_phaseE.md and the multi_axis_comparison.csv archive.
declare -A PRED_TYPE=( [v2]=eps [v5]=v          )
declare -A COND_TYPE=( [v2]=film [v5]=adaln_zero )

CKPT_ROOT="$HOME/orcd/scratch/diffmm/checkpoints"
SYNTH_ROOT="$HOME/orcd/scratch/diffmm/synthetic"
RAW_DIR="data/raw"
PROCESSED_DIR="data/processed"
MANIFEST_DIR="data/manifests"

# Sampling concurrency — array=0-399 with %20 cap (matches v9 orchestrator).
SAMPLE_ARRAY="--array=0-399%20"

# ─── Helpers ────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

submit() {
    # sbatch --parsable wrapper. Echos the command for reproducibility, returns
    # the JID on stdout.
    log "EXEC: sbatch --parsable $*"
    local jid
    if ! jid=$(sbatch --parsable "$@" 2>&1); then
        log "FAILED to submit: sbatch $*"
        log "  output: $jid"
        return 1
    fi
    echo "$jid"
}

wait_for_jobs() {
    # Poll squeue until none of the given JIDs are still in the queue.
    local jids="$*"
    [[ -z "$jids" ]] && return 0
    log "Waiting for jobs: $jids"
    while true; do
        local running=0
        for jid in $jids; do
            if squeue -u "$USER" -j "$jid" -h 2>/dev/null | grep -q .; then
                running=$((running + 1))
            fi
        done
        if [[ $running -eq 0 ]]; then
            log "All jobs finished: $jids"
            return 0
        fi
        sleep 60
    done
}

best_ckpt() {
    # Lowest valloss checkpoint in the given dir. Empty stdout if none.
    local run_dir="$1"
    [[ -d "$run_dir" ]] || return 0
    ls -1 "$run_dir"/ckpt-epoch*-valloss*.ckpt 2>/dev/null \
        | awk -F'valloss' '{print $2, $0}' \
        | sort -n \
        | head -1 \
        | awk '{print $2}'
}

count_parquets() {
    local dir="$1"
    [[ -d "$dir" ]] || { echo 0; return; }
    find "$dir" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' '
}

mkdir -p logs "$MANIFEST_DIR" results/_archive

# ─── Pre-flight checks ──────────────────────────────────────────────────────
log "════════════════════════════════════════════════════════════════"
log "=== Cross-ticker v2/v5 orchestrator starting ==="
log "════════════════════════════════════════════════════════════════"
log "User:        $USER"
log "Tickers:     ${TICKERS[*]}"
log "Models:      ${MODELS[*]}"
log "Ckpt root:   $CKPT_ROOT"
log "Synth root:  $SYNTH_ROOT"
log ""

# Verify raw data exists for each ticker
fatal=0
for ticker in "${TICKERS[@]}"; do
    for kind in order trade; do
        f="$RAW_DIR/${kind}_${ticker}.csv.gz"
        if [[ ! -f "$f" ]]; then
            log "MISSING: $f"
            fatal=1
        fi
    done
done
if [[ $fatal -eq 1 ]]; then
    log "FATAL: missing raw TAQ files (see above). Aborting."
    exit 1
fi
log "All raw TAQ files present."

# Verify ticker configs exist
for ticker in "${TICKERS[@]}"; do
    cfg="configs/data/$(echo "$ticker" | tr '[:upper:]' '[:lower:]')_top.yaml"
    if [[ ! -f "$cfg" ]]; then
        log "FATAL: $cfg missing — create it first (mirror tsla_top.yaml)."
        exit 1
    fi
done
log "All ticker data configs present."

# ─── STAGE 0: per-ticker data prep ──────────────────────────────────────────
log ""
log "════════════════════════════════════════════════════════════════"
log "=== STAGE 0: data prep (split → clean → finalize) per ticker ==="
log "════════════════════════════════════════════════════════════════"

# Stage 0a: split raw — submit all (ticker × kind) jobs in parallel
split_jids=()
for ticker in "${TICKERS[@]}"; do
    if [[ -d "data/raw_partitioned/$ticker/quotes" ]] && [[ -d "data/raw_partitioned/$ticker/trades" ]]; then
        nq=$(count_parquets "data/raw_partitioned/$ticker/quotes")
        nt=$(count_parquets "data/raw_partitioned/$ticker/trades")
        if [[ $nq -gt 50 && $nt -gt 50 ]]; then
            log "[$ticker] raw_partitioned/ already populated ($nq quotes, $nt trades) — skipping split"
            continue
        fi
    fi
    log "[$ticker] submitting split (quotes + trades parallel)"
    jq=$(submit slurms/ph1_split_raw.slurm \
        +gz="data/raw/order_${ticker}.csv.gz" \
        +kind=quotes \
        +ticker="$ticker") || exit 1
    jt=$(submit slurms/ph1_split_raw.slurm \
        +gz="data/raw/trade_${ticker}.csv.gz" \
        +kind=trades \
        +ticker="$ticker") || exit 1
    split_jids+=("$jq" "$jt")
    log "[$ticker] split JIDs: quotes=$jq trades=$jt"
done
wait_for_jobs "${split_jids[@]:-}"

# Stage 0b: clean (per-date array job) per ticker
for ticker in "${TICKERS[@]}"; do
    inter_dir="$PROCESSED_DIR/intermediate/$ticker"
    nfiles=$(count_parquets "$inter_dir")
    if [[ $nfiles -gt 50 ]]; then
        log "[$ticker] intermediate/ already populated ($nfiles parquets) — skipping clean"
        continue
    fi
    manifest="$MANIFEST_DIR/ph1_clean_$(echo "$ticker" | tr '[:upper:]' '[:lower:]').json"
    log "[$ticker] generating clean manifest → $manifest"
    python scripts/utils/make_manifest.py \
        --kind clean \
        --tickers "$ticker" \
        --out "$manifest" || { log "FATAL: manifest gen failed for $ticker"; exit 1; }
    n=$(python -c "import json; print(len(json.load(open('$manifest'))))")
    last=$((n - 1))
    log "[$ticker] manifest has $n entries; submitting clean array (0-$last)"
    j=$(MANIFEST="$manifest" submit \
        --array="0-${last}%50" \
        slurms/ph1_clean.slurm) || exit 1
    log "[$ticker] clean JID=$j"
    wait_for_jobs "$j"
done

# Stage 0c: finalize (splits + quantiles + canonical schema) per ticker
for ticker in "${TICKERS[@]}"; do
    cfg="$(echo "$ticker" | tr '[:upper:]' '[:lower:]')_top"
    if [[ -f "$PROCESSED_DIR/SPLITS_${ticker}.json" ]] \
       && [[ -f "$PROCESSED_DIR/quantiles_${ticker}.json" ]]; then
        log "[$ticker] SPLITS + quantiles JSONs exist — skipping finalize"
        continue
    fi
    log "[$ticker] submitting finalize (data=$cfg)"
    j=$(submit slurms/ph1_finalize.slurm data="$cfg") || exit 1
    log "[$ticker] finalize JID=$j"
    wait_for_jobs "$j"
done

# Sanity: train/val parquets must exist for each ticker
for ticker in "${TICKERS[@]}"; do
    n_train=$(find "$PROCESSED_DIR/train" -name "${ticker}_*.parquet" 2>/dev/null | wc -l | tr -d ' ')
    n_val=$(find "$PROCESSED_DIR/val" -name "${ticker}_*.parquet" 2>/dev/null | wc -l | tr -d ' ')
    log "[$ticker] processed: train=$n_train  val=$n_val"
    if [[ $n_train -lt 10 || $n_val -lt 3 ]]; then
        log "FATAL: insufficient processed data for $ticker — aborting"
        exit 1
    fi
done

# ─── STAGE 1: train all 4 (ticker, model) jobs in parallel ─────────────────
log ""
log "════════════════════════════════════════════════════════════════"
log "=== STAGE 1: training (4 jobs, submitted in parallel) ==="
log "════════════════════════════════════════════════════════════════"

declare -A TRAIN_JID
declare -A CKPT_DIR
all_train_jids=""
for ticker in "${TICKERS[@]}"; do
    ticker_lc=$(echo "$ticker" | tr '[:upper:]' '[:lower:]')
    cfg="${ticker_lc}_top"
    for model in "${MODELS[@]}"; do
        run_name="${ticker_lc}_${model}"
        ckpt_dir="$CKPT_ROOT/$run_name"
        CKPT_DIR[$run_name]="$ckpt_dir"

        existing=$(best_ckpt "$ckpt_dir")
        if [[ -n "$existing" ]]; then
            log "[$run_name] checkpoint exists ($existing) — skipping training"
            continue
        fi

        log "[$run_name] submitting training (data=$cfg, pred=${PRED_TYPE[$model]}, cond=${COND_TYPE[$model]})"
        jid=$(submit slurms/ph2_finetune.slurm \
            data="$cfg" \
            generator.schedule.prediction_type="${PRED_TYPE[$model]}" \
            generator.model.conditioning_type="${COND_TYPE[$model]}" \
            cluster.checkpoint_root="$ckpt_dir") || { log "[$run_name] submit failed"; continue; }
        TRAIN_JID[$run_name]=$jid
        all_train_jids="$all_train_jids $jid"
        log "[$run_name] training JID=$jid"
    done
done
wait_for_jobs $all_train_jids

# Verify each model produced a checkpoint
for ticker in "${TICKERS[@]}"; do
    ticker_lc=$(echo "$ticker" | tr '[:upper:]' '[:lower:]')
    for model in "${MODELS[@]}"; do
        run_name="${ticker_lc}_${model}"
        ckpt=$(best_ckpt "${CKPT_DIR[$run_name]}")
        if [[ -z "$ckpt" ]]; then
            log "[$run_name] WARN: no checkpoint found after training; downstream stages will skip"
        else
            log "[$run_name] best ckpt: $ckpt"
        fi
    done
done

# ─── STAGE 2: sample sequentially per (ticker, model) ─────────────────────
log ""
log "════════════════════════════════════════════════════════════════"
log "=== STAGE 2: sampling (4 jobs, sequential to avoid QOS contention) ==="
log "════════════════════════════════════════════════════════════════"

# Ensure sampling manifest exists
manifest="$MANIFEST_DIR/ph2_sample.json"
if [[ ! -f "$manifest" ]]; then
    log "Generating sampling manifest (100 seeds × 4 regimes = 400 tasks)"
    python scripts/utils/make_manifest.py \
        --kind sample --n-seeds 100 \
        --out "$manifest" || { log "FATAL: sample manifest gen failed"; exit 1; }
fi

for ticker in "${TICKERS[@]}"; do
    ticker_lc=$(echo "$ticker" | tr '[:upper:]' '[:lower:]')
    cfg="${ticker_lc}_top"
    for model in "${MODELS[@]}"; do
        run_name="${ticker_lc}_${model}"
        ckpt=$(best_ckpt "${CKPT_DIR[$run_name]}")
        if [[ -z "$ckpt" ]]; then
            log "[$run_name] no checkpoint — skipping sampling"
            continue
        fi
        synth_root="$SYNTH_ROOT/$run_name"
        synth_dir="$synth_root/$ticker"

        n_p=$(count_parquets "$synth_dir")
        if [[ $n_p -ge 1000 ]]; then
            log "[$run_name] $n_p parquets exist at $synth_dir — skipping sampling"
            continue
        fi

        log "[$run_name] sampling: ckpt=$ckpt → $synth_root"
        jid=$(submit $SAMPLE_ARRAY \
            slurms/ph2_sample_cpu.slurm \
            generator=ddim_sample \
            data="$cfg" \
            generator.checkpoint_path="$ckpt" \
            generator.schedule.prediction_type="${PRED_TYPE[$model]}" \
            generator.model.conditioning_type="${COND_TYPE[$model]}" \
            cluster.synthetic_root="$synth_root") || { log "[$run_name] sample submit failed"; continue; }
        log "[$run_name] sampling JID=$jid"
        wait_for_jobs "$jid"
        n_p=$(count_parquets "$synth_dir")
        log "[$run_name] sampling done — $n_p parquets at $synth_dir"
        if [[ $n_p -lt 1000 ]]; then
            log "[$run_name] WARN: < 1000 parquets — sampling largely failed"
        fi
    done
done

# ─── STAGE 3: validate (CPU parts B/C/D) — per-ticker, parallel models ────
log ""
log "════════════════════════════════════════════════════════════════"
log "=== STAGE 3: validation (B/C/D) — 2 jobs parallel per ticker ==="
log "════════════════════════════════════════════════════════════════"

for ticker in "${TICKERS[@]}"; do
    ticker_lc=$(echo "$ticker" | tr '[:upper:]' '[:lower:]')
    val_jids=""
    for model in "${MODELS[@]}"; do
        run_name="${ticker_lc}_${model}"
        synth_dir="$SYNTH_ROOT/$run_name/$ticker"
        out_dir="results/_archive/$run_name/validation"

        if [[ $(count_parquets "$synth_dir") -lt 100 ]]; then
            log "[$run_name] insufficient synth parquets — skipping validation"
            continue
        fi

        mkdir -p "$out_dir"
        log "[$run_name] submitting validation → $out_dir"
        jid=$(SYNTH_DIR="$synth_dir" OUT_DIR="$out_dir" TICKER="$ticker" N_SYNTH=250 \
            submit slurms/ph6_validate_cpu.slurm) || { log "[$run_name] validate submit failed"; continue; }
        log "[$run_name] validation JID=$jid"
        val_jids="$val_jids $jid"
    done
    wait_for_jobs $val_jids
done

# ─── STAGE 4: cross-model TRADES + predictive_score per ticker ────────────
log ""
log "════════════════════════════════════════════════════════════════"
log "=== STAGE 4: TRADES metrics + predictive score, per ticker ==="
log "════════════════════════════════════════════════════════════════"
log "Note: scripts 62/63 run on the login node (CPU). If your login-node memory"
log "      is constrained, wrap each python invocation in a SLURM job (see"
log "      slurms/ph7_trades_eval.slurm for a 128G template)."

# Scripts 62/63 don't take --ticker; they read all parquets in --real-dir.
# Since data/processed/{train,val}/ is shared across tickers (filename prefix
# distinguishes them), we symlink the per-ticker parquets into scoped dirs
# before invoking the eval scripts.
stage_real_for_ticker() {
    local ticker="$1" split="$2"
    local src="$PROCESSED_DIR/$split"
    local dst="$PROCESSED_DIR/${split}_${ticker}"
    mkdir -p "$dst"
    # Clean stale symlinks then re-link this ticker's files
    find "$dst" -maxdepth 1 -type l -delete 2>/dev/null || true
    local n=0
    for f in "$src"/${ticker}_*.parquet; do
        [[ -e "$f" ]] || continue
        ln -sf "$(realpath "$f")" "$dst/$(basename "$f")"
        n=$((n + 1))
    done
    log "[$ticker] staged $n $split parquets → $dst"
}

for ticker in "${TICKERS[@]}"; do
    ticker_lc=$(echo "$ticker" | tr '[:upper:]' '[:lower:]')

    # Collect (label, dir) pairs whose synth dirs are populated
    labels=()
    dirs=()
    for model in "${MODELS[@]}"; do
        run_name="${ticker_lc}_${model}"
        d="$SYNTH_ROOT/$run_name/$ticker"
        if [[ $(count_parquets "$d") -ge 100 ]]; then
            labels+=("$model")
            dirs+=("$d")
        else
            log "[$ticker] skipping $model from cross-eval (synth dir empty)"
        fi
    done
    if [[ ${#labels[@]} -lt 1 ]]; then
        log "[$ticker] no models with synth — skipping cross-eval"
        continue
    fi

    # Stage per-ticker real-data symlink dirs
    stage_real_for_ticker "$ticker" train
    stage_real_for_ticker "$ticker" val
    real_train="$PROCESSED_DIR/train_${ticker}"
    real_val="$PROCESSED_DIR/val_${ticker}"

    out_trades="results/trades_metrics_${ticker_lc}"
    out_pred="results/predictive_score_${ticker_lc}"
    mkdir -p "$out_trades" "$out_pred"

    log "[$ticker] EXEC: python scripts/62_trades_metrics.py --real-dir $real_val --synth-dirs ${dirs[*]} --labels ${labels[*]} --out $out_trades --max-tapes 100"
    python scripts/62_trades_metrics.py \
        --real-dir "$real_val" \
        --synth-dirs "${dirs[@]}" \
        --labels "${labels[@]}" \
        --out "$out_trades" \
        --max-tapes 100 \
        || log "[$ticker] WARN: TRADES metrics failed (non-fatal)"

    log "[$ticker] EXEC: python scripts/63_predictive_score.py --real-train-dir $real_train --real-test-dir $real_val --synth-dirs ${dirs[*]} --labels ${labels[*]} --out $out_pred ..."
    python scripts/63_predictive_score.py \
        --real-train-dir "$real_train" \
        --real-test-dir "$real_val" \
        --synth-dirs "${dirs[@]}" \
        --labels "${labels[@]}" \
        --out "$out_pred" \
        --window-size 64 \
        --stride 8 \
        --max-tapes-train 200 \
        --max-tapes-test 10 \
        --seed 42 \
        --device cpu \
        || log "[$ticker] WARN: predictive score failed (non-fatal)"
done

# ─── Summary ────────────────────────────────────────────────────────────────
log ""
log "════════════════════════════════════════════════════════════════"
log "=== ALL DONE ==="
log "════════════════════════════════════════════════════════════════"
log "Per-(ticker, model) artifacts:"
for ticker in "${TICKERS[@]}"; do
    ticker_lc=$(echo "$ticker" | tr '[:upper:]' '[:lower:]')
    for model in "${MODELS[@]}"; do
        run_name="${ticker_lc}_${model}"
        log "  results/_archive/$run_name/validation/  ($(count_parquets "$SYNTH_ROOT/$run_name/$ticker") synth parquets)"
    done
done
log ""
log "Per-ticker cross-model summaries:"
for ticker in "${TICKERS[@]}"; do
    ticker_lc=$(echo "$ticker" | tr '[:upper:]' '[:lower:]')
    log "  results/trades_metrics_${ticker_lc}/{trades_metrics_summary.txt,pca_coverage.csv,vol_volatility_correlation.csv,leverage_effect.csv}"
    log "  results/predictive_score_${ticker_lc}/predictive_score.csv"
done
log ""
log "Cross-ticker comparison (manual next step):"
log "  Append rows to results/_archive/multi_axis_comparison.csv keyed by"
log "  (ticker, model) and re-run scripts/utils/build_comparison_matrix.py if"
log "  you've extended it to be ticker-aware."
