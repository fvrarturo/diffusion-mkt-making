# Prompt 1: TRADES-Inspired Supplementary Metrics

---

## Context

I'm working on a course project (15.458, MIT Sloan) that uses diffusion models to generate synthetic Limit Order Book (LOB) data for stress-testing market-making agents. We have a validation suite that evaluates 16 stylized facts across 16 generator checkpoints. We need to add 4 additional metrics inspired by the TRADES paper (Berti et al. 2025, "Generating Realistic Market Simulations with Diffusion Models").

## What exists

- **Real validation data**: `data/processed/val/INTC_*.parquet` — 10-20 parquet files, each one day, ~2-4M events per day
- **Synthetic data**: directories like `$HOME/orcd/scratch/diffmm/synthetic_v2_stitched/INTC/` containing subdirectories `{base, high_vol, toxic, thin}/` with parquet files (2560 events each, 1600 per regime = 6400 total)
- **Existing diagnostic script**: `scripts/60_deep_diagnostic.py` which loads real/synth data, computes per-feature stats, cross-feature correlations, temporal ACF, window-level stats, and writes a text report + CSVs
- **Parquet schema** (23 columns): `event_idx, ts_ns, ticker, event_type, bid_px, ask_px, bid_sz, ask_sz, trade_px, trade_sz, trade_sign, is_lit, mid, spread, bid_dist, ask_dist, trade_dist, mid_return, c_vol, c_vpin, c_imb, c_tod, regime_label`
- Key columns for these metrics:
  - `mid_return`: log return of midprice, 89% zeros for INTC (large-tick stock)
  - `mid`: midprice level
  - `trade_sz`: trade size (0 for non-trade events)
  - `bid_sz`, `ask_sz`: best bid/ask queue sizes
  - `spread`: ask_px - bid_px
  - `ts_ns`: nanosecond timestamp

## Task

Create a NEW script `scripts/62_trades_metrics.py` that computes the following 4 metrics across real and multiple synthetic generators, producing both a text report and publication-quality PDF figures.

### CLI interface

```bash
python scripts/62_trades_metrics.py \
    --real-dir data/processed/val \
    --synth-dirs \
        $HOME/orcd/scratch/diffmm/synthetic_v2_stitched/INTC \
        $HOME/orcd/scratch/diffmm/synthetic/intc_v9/INTC \
    --labels v2 v9 \
    --out results/trades_metrics \
    --max-tapes 100
```

### Metric 1: Volume-Volatility Correlation (TRADES Fig 3.2)

**Stylized fact**: Trading volume and price volatility are positively correlated (Karpoff 1987).

**Computation**:
1. For each tape (real day or synthetic parquet), bucket events into 1-minute intervals using `ts_ns`.
2. Per bucket: compute `volume = sum(trade_sz)` and `volatility = std(mid_return)` (or `sum(|mid_return|)` if too few nonzero returns per bucket).
3. Compute Pearson correlation between the volume and volatility time series for that tape.
4. Collect one correlation value per tape → distribution of correlations.

**Output**:
- `vol_volatility_correlation.pdf`: Overlaid KDE density plots of the per-tape correlation distributions for real vs each synth model. X-axis: correlation coefficient [-1, 1]. Y-axis: density. Use distinct colors per model, include legend. Title: "Volume-Volatility Correlation Distribution". Real data should be a filled/shaded curve, synth models as lines.
- `vol_volatility_correlation.csv`: columns `model, tape_id, pearson_r`
- In the text report: per-model mean ± std of the correlation, plus real reference.

**Expected**: Real should show a positive-skewed distribution centered around +0.3 to +0.6. Models that get trade arrival right (v9 with 4.9% trade fraction) should match this better than models with inflated trade rates (v2 with 55%).

### Metric 2: Returns-Volatility Correlation / Leverage Effect (TRADES Fig 3.3)

**Stylized fact**: Returns and future volatility are negatively correlated (Black 1976, "leverage effect"). When prices drop, volatility tends to increase.

**Computation**:
1. Same 1-minute bucketing as Metric 1.
2. Per bucket: compute `return_bucket = sum(mid_return)` and `volatility_next = std(mid_return) in the NEXT bucket`.
3. Compute Pearson correlation between `return_bucket[:-1]` and `volatility_next[1:]` for that tape.
4. One correlation per tape → distribution.

**Output**:
- `leverage_effect.pdf`: Same KDE format as Metric 1. Title: "Returns-Volatility (Leverage Effect) Correlation Distribution".
- `leverage_effect.csv`: columns `model, tape_id, pearson_r`
- Text report: per-model mean ± std.

**Expected**: Real should show a distribution centered around -0.1 to -0.3 (negative = leverage effect). This is a cross-temporal asymmetry — harder for diffusion models to capture than contemporaneous correlations.

### Metric 3: PCA Coverage / Convex Hull Overlap (TRADES Fig 2)

**Stylized fact**: A good generator should cover the full support of the real data distribution, not just a subset (mode collapse detection).

**Computation**:
1. For each tape, compute a FEATURE VECTOR summarizing that tape:
   - `mean_spread`, `std_spread`
   - `mean_bid_sz`, `std_bid_sz`
   - `mean_ask_sz`, `std_ask_sz`
   - `trade_fraction` (fraction of events with trade_sz > 0)
   - `return_kurtosis` (scipy.stats.kurtosis of mid_return)
   - `return_std`
   - `abs_return_acf_lag1` (autocorrelation of |mid_return| at lag 1)
   - `ofi_proxy_std` (std of diff(bid_sz) - diff(ask_sz))
   This gives a ~10-dimensional feature vector per tape.
2. Standardize all features (z-score using real data's mean/std).
3. Fit PCA on REAL tape feature vectors (2 components).
4. Project both real and each synth model's tape vectors into PCA space.
5. Compute 2D convex hulls for real and for each synth model.
6. Compute overlap: `area(intersection(hull_real, hull_synth)) / area(hull_real)` as a fraction. Use `shapely.geometry.Polygon` for this (or implement a simple polygon intersection).

**Output**:
- `pca_coverage.pdf`: Scatter plot in PCA space. Real points as filled circles (gray/black). Each synth model as colored markers. Draw convex hull outlines for real (black dashed) and each synth model (colored dashed). Annotate each model's coverage percentage. Title: "PCA Coverage of Real Data Distribution".
- `pca_coverage.csv`: columns `model, coverage_pct, n_tapes, pc1_mean, pc1_std, pc2_mean, pc2_std`
- Text report: coverage % per model.

**Expected**: Models with better marginals should have higher coverage. v9 (copula+dequant with realistic trade fraction) should cover more than v2 (55% trade fraction = very different from real).

**Dependencies**: `shapely` for convex hull intersection. If not available, fall back to a simpler metric: fraction of real PCA points that fall within the synth convex hull (no intersection computation needed — just `hull_synth.contains(point)` for each real point). Report this as "containment fraction" instead.

### Metric 4: Mid-Price Trace Visualization (TRADES Fig 3.6)

**Purpose**: Visual comparison of generated price paths vs real. Not a quantitative metric — a figure for the paper.

**Computation**:
1. Select 5 synthetic tapes from the "base" regime (first 5 parquets, or random sample).
2. Select 1 real validation day.
3. For each tape, reconstruct the mid-price series: it's already in the `mid` column.
4. Normalize: divide each trace by its starting value to get relative price (start at 1.0).
5. Plot all 6 traces (5 synth + 1 real) on the same axes.

**Output**:
- `midprice_traces_{model}.pdf`: One figure per model. 5 colored synth traces (thin, alpha=0.6) + 1 real trace (thick black). X-axis: event index (or time if ts_ns is available). Y-axis: normalized mid-price. Title: "Mid-Price Traces: {model} vs Real". Include legend: "Real", "Synth seed 1-5".
- Generate one figure per model passed in `--labels`.

**Styling**: Use matplotlib with a clean style (seaborn-v0_8-whitegrid or similar). Figure size 10x4 inches. DPI 150. Colors: real=black thick, synth=tab10 colormap thin.

### General requirements

- **No GPU needed** — all metrics are computed on decoded parquet data.
- **Reuse the data loading pattern** from `60_deep_diagnostic.py`: iterate regime subdirs, load parquets, concatenate with `_tape_id` and `_regime` columns.
- **Handle edge cases**: tapes with zero trades (skip volume-volatility for those), tapes with constant mid_return (skip ACF), very short tapes (skip if < 100 events).
- **Text report**: Write a `trades_metrics_summary.txt` with all results, similar format to deep_diagnostic_summary.txt.
- **Print to stdout AND write to file** (same pattern as 60_deep_diagnostic.py).
- Use only standard scientific Python: numpy, scipy, pandas, matplotlib, polars (for parquet loading). shapely for PCA convex hull (with fallback if not installed).

---

# Prompt 2: Predictive Score (TRADES Table 1)
# Give this as a SEPARATE task to the coding agent

---

## Context

Same project as above. The TRADES paper's headline quantitative metric is the **predictive score**: train a mid-price prediction model on synthetic data, test on real data, measure MAE. This evaluates the "usefulness" of synthetic data — if it's realistic, a model trained on it should generalize to real data.

This is a separate script because it involves training a neural network, unlike the purely statistical metrics above.

## Task

Create `scripts/63_predictive_score.py` that:

1. Trains a simple mid-price prediction model on synthetic data from a given generator
2. Tests it on real validation data
3. Reports the predictive score (MAE) 
4. Repeats for multiple generators and for the "market replay" baseline (train on real train data, test on real val data)

### Model architecture

Use a SIMPLE model — the point is to measure data quality, not model sophistication:

```python
class MidPricePredictor(nn.Module):
    """1-layer LSTM that predicts next mid_return from a window of LOB features."""
    def __init__(self, input_dim=8, hidden_dim=64, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # predict next mid_return
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # predict from last hidden state
```

Input features (per event): `[mid_return, spread, bid_sz, ask_sz, bid_dist, ask_dist, trade_sz, trade_sign]` — the same 8 features the diffusion model uses. Normalize using the TRAINING set's mean/std (per-feature z-score).

### Data preparation

**Training set construction** (from synthetic or real data):
1. Load tapes from the specified directory
2. For each tape, create sliding windows of length `W=64` events
3. Input: features of events 1..W. Target: `mid_return` of event W+1
4. Collect all windows, shuffle, split 90/10 into train/val for early stopping

**Test set**: Always real validation data (`data/processed/val/INTC_*.parquet`), using the same windowing.

### Training

- Optimizer: Adam, lr=1e-3
- Batch size: 256
- Max epochs: 20
- Early stopping: patience 3 on validation MAE
- Loss: MSE (but report MAE at test time, following TRADES)
- Device: CPU is fine for this model size (or GPU if available)

### CLI interface

```bash
python scripts/63_predictive_score.py \
    --real-train-dir data/processed/train \
    --real-test-dir data/processed/val \
    --synth-dirs \
        $HOME/orcd/scratch/diffmm/synthetic_v2_stitched/INTC \
        $HOME/orcd/scratch/diffmm/synthetic/intc_v9/INTC \
        $HOME/orcd/scratch/diffmm/synthetic_v2_remapped/INTC \
    --labels v2 v9 v2_remapped \
    --out results/predictive_score \
    --window-size 64 \
    --max-tapes-train 200 \
    --max-tapes-test 10 \
    --seed 42
```

### Output

**Table** (printed + CSV):
```
Model               Train Source    Test MAE    Relative to Replay
market_replay       real train      0.XXX       1.00×
v2                  v2 synth        0.XXX       X.XX×
v9                  v9 synth        0.XXX       X.XX×
v2_remapped         remapped        0.XXX       X.XX×
```

**CSV**: `predictive_score.csv` with columns `model, train_source, test_mae, relative_to_replay, train_epochs, train_final_loss`

**Figure**: `predictive_score.pdf` — horizontal bar chart of test MAE per model. Market replay as a reference line. Lower is better. Include error bars if you run multiple seeds.

### Important details

- The "market replay" baseline trains on REAL training data (60 days in `data/processed/train/`). This is the best possible score — any synth-trained model should be worse. The ratio (synth MAE / replay MAE) is the predictive score as defined in TRADES.
- Normalize features using the TRAINING set statistics (whether real or synth). The test set is normalized with the SAME statistics (not its own). This is important — if the synthetic data has different scale, the normalization mismatch is part of what the metric measures.
- For real data loading: concatenate all parquets in the directory. For synth: concatenate across regime subdirs (or flat if no subdirs).
- Set `torch.manual_seed(seed)` and `np.random.seed(seed)` for reproducibility.
- If the model predicts `mid_return` and real `mid_return` is 89% zeros, MAE will be dominated by the zero-return events. This is fine — it's measuring whether the model learns the right baseline. A synth dataset with 55% trade fraction (v2) will train the LSTM to expect frequent nonzero returns, which will hurt its MAE on real data where returns are mostly zero. A synth dataset with 5% trade fraction (v9) will train the LSTM to correctly predict zero most of the time.

### Dependencies

- PyTorch (already in the diffmm environment)
- Standard scientific Python (numpy, scipy, polars/pandas, matplotlib)
- No additional packages needed

### Runtime estimate

- Data prep: ~2 min per model (loading + windowing)
- Training: ~5 min per model on CPU (small LSTM, 20 epochs max)
- Total: ~30 min for 4 models (market_replay + v2 + v9 + v2_remapped)