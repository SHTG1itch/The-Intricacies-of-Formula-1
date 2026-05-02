# IDP Final Project — Formula 1 Data Analysis

**Srivats Iyer & Aakash B.**

> This project investigates what drives success in Formula 1. We analyze driver skill vs. car performance, quantify the impact of the 2021 cost cap, and build a machine-learning ensemble to predict World Drivers' Championship outcomes. Data comes from the FastF1 API (2018–2025 seasons).

---

## Table of Contents

1. [Research Questions](#research-questions)
2. [Repository Structure](#repository-structure)
3. [Data Pipeline Overview](#data-pipeline-overview)
4. [Quick Start](#quick-start)
5. [Step-by-Step Instructions](#step-by-step-instructions)
6. [Script Reference](#script-reference)
7. [ML Model Architecture](#ml-model-architecture)
8. [Key Results](#key-results)
9. [Phase Timeline](#phase-timeline)

---

## Research Questions

| # | Question |
|---|---|
| **RQ1** | What factors in a race weekend lead to a win? Does driver skill really matter? |
| **RQ2** | Has the 2021 cost cap measurably altered race results? |
| **RQ2.1** | Did the cost cap function as the FIA intended? (lap-time convergence test) |
| **RQ4** | At what accuracy can we predict the WDC winner, and how much is skill vs. car? |

---

## Repository Structure

```
idp-final-project-25-26-srivatsi-aakashb/
│
├── cache_fastf1_data.py          # Step 1 — download raw FIA timing data
├── test_cache_sandbox.py         # Unit tests for the caching system (fully mocked)
│
├── discovery_raw_data/           # Raw FastF1 cache (produced by Step 1)
│   ├── cache_progress.json       # Tracks which sessions have been downloaded
│   ├── discovery.md              # Research context and dataset descriptions
│   └── YYYY/YYYY-MM-DD_Race/     # One folder per session (Practice 1-3, Quali, Race)
│
├── data_organized/               # Cleaned CSVs and trained models (produced by Steps 2-4)
│   ├── prepocess_data.py         # Step 2 — builds the 4 analysis CSVs
│   ├── qualifying_deltas.csv     # RQ1: teammate qualifying gaps
│   ├── race_results.csv          # RQ2: all finishes 2018-2025
│   ├── lap_time_spreads.csv      # RQ2.1: per-driver lap delta from fastest
│   ├── driver_season_profiles.csv# RQ4: season aggregates per driver
│   ├── ml_training_features.csv  # ML feature matrix (2022-2024 training)
│   ├── ml_2025_features.csv      # ML feature matrix (2025 backtest)
│   ├── 2026_training_data.csv    # ML feature matrix (2022-2025 for 2026 predictions)
│   ├── 2025_predictions.csv      # Model output vs. actual 2025 results
│   ├── 2025_race_metrics.csv     # Per-race accuracy metrics
│   ├── feature_importances.csv   # Feature importance weights
│   ├── best_hyperparams.txt      # Optimal hyperparameters found during tuning
│   ├── model_evaluation.txt      # LOSO cross-validation results
│   ├── wf_evaluation.txt         # Walk-forward validation metrics
│   ├── models/                   # Saved model artifacts (*.joblib)
│   └── test_data/                # Small test datasets for preprocessing validation
│
└── final_report/                 # All analysis code, plots, and the written report
    ├── feature_engineering.py    # Step 3 — builds the ML feature matrix
    ├── mach_learning.py          # Step 4 — trains the ensemble + Bayesian post-processing
    ├── predict_2025.py           # Step 5a — backtest predictions against 2025 season
    ├── predict_2026.py           # Step 5b — retrain and predict the 2026 season
    ├── walk_forward_model.py     # Step 5c — Plackett-Luce in-season updater
    ├── create_plots.py           # Step 6 — generates all report visualizations
    ├── final.md                  # Written research report (~20 pages)
    └── plots/                    # Generated PNG figures (18+ files)
```

---

## Data Pipeline Overview

```
cache_fastf1_data.py
  │  Downloads raw FIA timing data from the FastF1 API (2018-2025)
  ▼
discovery_raw_data/          (~149 race weekends × 5 sessions each)
  │
data_organized/prepocess_data.py
  │  Cleans raw sessions into 4 analysis-ready CSVs
  ▼
qualifying_deltas.csv  race_results.csv  lap_time_spreads.csv  driver_season_profiles.csv
  │
final_report/feature_engineering.py
  │  Extracts rolling form, circuit history, pace, consistency, telemetry features
  │  (Computes rolling stats BEFORE loading race results — no data leakage)
  ▼
ml_training_features.csv  +  ml_2025_features.csv
  │
final_report/mach_learning.py
  │  Trains 3-model ensemble with Bayesian post-processing
  │  Validates with Leave-One-Season-Out (LOSO) cross-validation
  ▼
data_organized/models/        (11 joblib model files + imputers + calibration)
  │
  ├── final_report/predict_2025.py   (backtest vs. known 2025 results)
  ├── final_report/predict_2026.py   (forward prediction: retrains on 2022-2025)
  └── final_report/walk_forward_model.py   (Plackett-Luce in-season updater)
  │
final_report/create_plots.py
  │  Reads the CSVs and generates all 18+ PNG visualizations
  ▼
final_report/plots/  +  final_report/final.md
```

---

## Quick Start

**Prerequisites:** Python 3.9+, pip

```bash
# 1. Clone
git clone <repo-url>
cd idp-final-project-25-26-srivatsi-aakashb

# 2. Install dependencies
pip install fastf1 pandas numpy matplotlib seaborn scipy xgboost scikit-learn joblib

# 3. Cache raw data  (one-time, ~several hours — see note below)
python cache_fastf1_data.py

# 4. Preprocess into analysis CSVs
python data_organized/prepocess_data.py

# 5. Build ML feature matrix
python final_report/feature_engineering.py

# 6. Train the ensemble
python final_report/mach_learning.py

# 7. Run backtests and predictions
python final_report/predict_2025.py
python final_report/predict_2026.py
python final_report/walk_forward_model.py

# 8. Generate all report plots
python final_report/create_plots.py
```

> **Note on caching:** `cache_fastf1_data.py` downloads from the FastF1 API and respects rate limits using exponential back-off (60s → 300s → 1200s → 3600s). By default it uses 1 worker to stay within API limits. Progress is saved to `discovery_raw_data/cache_progress.json` so the download is resumable — you can safely interrupt and restart it. Once cached, all subsequent steps run fully offline.

---

## Step-by-Step Instructions

### Step 1 — Cache Raw Data

```bash
python cache_fastf1_data.py
```

Downloads session data for every race weekend from 2018 to 2025 (Practice 1–3, Qualifying, Race) from the FastF1 API and saves them to `discovery_raw_data/`. The script tracks progress in `cache_progress.json` and skips sessions it has already downloaded.

**Optional:** To speed up downloads with a warm cache, set `FF1_WORKERS=2` (not recommended above 2):
```bash
FF1_WORKERS=2 python cache_fastf1_data.py   # macOS/Linux
$env:FF1_WORKERS=2; python cache_fastf1_data.py   # PowerShell
```

**To run the unit tests** (no network required, fully mocked):
```bash
python test_cache_sandbox.py
```

---

### Step 2 — Preprocess Data

```bash
python data_organized/prepocess_data.py
```

Reads the FastF1 cache and outputs four cleaned CSVs to `data_organized/`:

| Output file | Rows | Contents |
|---|---|---|
| `qualifying_deltas.csv` | ~488 | Teammate qualifying gaps per race (RQ1) |
| `race_results.csv` | ~2,977 | All race finishes with points and gap-to-winner (RQ2) |
| `lap_time_spreads.csv` | large | Per-driver lap delta from fastest lap per race (RQ2.1) |
| `driver_season_profiles.csv` | — | Season aggregate stats per driver (RQ4) |

---

### Step 3 — Build the ML Feature Matrix

```bash
python final_report/feature_engineering.py
```

Constructs one row per driver per race weekend from the FastF1 cache. Features include:

- **Pace signals:** FP1/FP2/FP3 lap times, qualifying position and gap to pole
- **Consistency metrics:** lap-time standard deviation across practice sessions
- **Sector performance:** sector split deltas vs. fastest driver
- **Rolling form:** weighted average finishing positions over last 3/5/10 races
- **Circuit history:** driver's historical performance at each specific track
- **Weather indicators:** session condition flags
- **Sprint weekend flags:** presence of sprint race format

Rolling statistics are computed strictly from races *before* the current round to prevent data leakage. A warmup period using 2019–2021 data populates the rolling windows before the 2022 training set begins.

Outputs:
- `data_organized/ml_training_features.csv` — 2022–2024 training features
- `data_organized/ml_2025_features.csv` — 2025 backtest features

---

### Step 4 — Train the Ensemble

```bash
python final_report/mach_learning.py
```

Trains the three-model ensemble with Bayesian post-processing and saves all artifacts to `data_organized/models/`.

**Training takes approximately 5–15 minutes** on a typical laptop depending on hardware.

See [ML Model Architecture](#ml-model-architecture) below for full details.

Saved artifacts:

| File | Description |
|---|---|
| `xgb_model.joblib` | XGBoost Regressor (predicts finishing position) |
| `xgb_ranker.joblib` | XGBoost Ranker (NDCG objective, top-3 focused) |
| `rf_win_model.joblib` | Random Forest win probability classifier |
| `rf_top10_model.joblib` | Random Forest top-10 ranker |
| `et_model.joblib` | Extra Trees ensemble |
| `meta_model.joblib` | Stacking meta-model |
| `feature_imputer.joblib` | Missing-value imputer (must reuse at inference — do not refit) |
| `bma_weights.joblib` | Bayesian Model Averaging blend weights |
| `cal_temperature.joblib` | Calibration temperature (Platt scaling) |
| `pl_temperature.joblib` | Plackett-Luce temperature parameter |

---

### Step 5a — Backtest Against 2025

```bash
python final_report/predict_2025.py
```

Loads the trained models and makes pre-race predictions for every 2025 race (using only FP/qualifying data available before Sunday). Race results are loaded only after predictions are made to prevent leakage.

Outputs to `data_organized/` and `final_report/plots/`:
- `2025_predictions.csv` — predicted vs. actual position per driver per race
- `2025_race_metrics.csv` — winner accuracy, podium overlap (Jaccard), position MAE by race
- `plots/season_accuracy_2025.png`, `plots/predicted_vs_actual.png`, `plots/top10_heatmap.png`

---

### Step 5b — Predict the 2026 Season

```bash
python final_report/predict_2026.py
```

Retrains the full ensemble from scratch on 2022–2025 data (adding one extra full season vs. the 2025 backtest) then generates race-by-race predictions for the 2026 season. New models are saved to `data_organized/models_2026/` to avoid overwriting the 2025 validation artifacts.

**Important caveat:** The 2026 season introduces entirely new hybrid power units and active aerodynamics. The model cannot account for regulation resets — its predictions assume relative team competitiveness follows historical trends (see `final_report/final.md` Discussion section for more detail).

---

### Step 5c — Walk-Forward Plackett-Luce Updater

```bash
python final_report/walk_forward_model.py
```

A complementary model that updates championship beliefs after each race. Unlike LOSO (which holds out an entire season at once), the walk-forward model incorporates in-season race results to refine driver quality estimates using a Plackett-Luce probabilistic framework.

See [ML Model Architecture](#ml-model-architecture) for details on the Plackett-Luce formulation.

---

### Step 6 — Generate All Plots

```bash
python final_report/create_plots.py
```

Reads all cleaned CSVs from `data_organized/` and outputs 18+ PNG files to `final_report/plots/`. Uses `matplotlib.use("Agg")` for headless (no display required) rendering.

Generated visualizations include swarm plots, KDE distributions, heatmaps, ridge (joy) plots, violin plots, radar charts, feature importance bar charts, and predicted-vs-actual scatter plots — all referenced by figure number in `final_report/final.md`.

---

## Script Reference

| Script | Inputs | Outputs | Runtime |
|---|---|---|---|
| `cache_fastf1_data.py` | FastF1 API (internet) | `discovery_raw_data/` | Hours (first run) |
| `test_cache_sandbox.py` | None (mocked) | Test results in terminal | <1 min |
| `data_organized/prepocess_data.py` | `discovery_raw_data/` | 4 CSVs in `data_organized/` | ~5 min |
| `final_report/feature_engineering.py` | `discovery_raw_data/` | `ml_training_features.csv`, `ml_2025_features.csv` | ~10 min |
| `final_report/mach_learning.py` | `ml_training_features.csv` | `data_organized/models/*.joblib` | 5–15 min |
| `final_report/predict_2025.py` | models + `ml_2025_features.csv` | prediction CSVs + plots | ~2 min |
| `final_report/predict_2026.py` | `2026_training_data.csv` | new models + prediction plots | 5–15 min |
| `final_report/walk_forward_model.py` | `ml_training_features.csv` | evaluation metrics + plots | ~5 min |
| `final_report/create_plots.py` | CSVs in `data_organized/` | `final_report/plots/*.png` | ~2 min |

---

## ML Model Architecture

### Base Ensemble (Three Models)

**Model 1 — XGBoost Regressor**
Predicts each driver's finishing position as a continuous number. Trained with gradient-boosted trees using `reg:squarederror` objective. Hyperparameters tuned via randomized search with LOSO cross-validation.

**Model 2 — XGBoost Ranker**
Treats each race as a ranking problem with NDCG objective, placing extra weight on correctly ordering the top-3. Uses `rank:ndcg` objective with query groups defined per race.

**Model 3 — Random Forest**
Binary classifier predicting win probability. A second Random Forest predicts top-10 probability. Outputs are probability scores that feed into the Bayesian calibration step.

### Bayesian Post-Processing (Three Layers)

**Layer A — Normal-Normal Hierarchical Shrinkage (Circuit History)**
Applies James-Stein shrinkage to each driver's per-circuit historical average. Drivers with few appearances at a track are shrunk toward the population mean; drivers with many appearances retain more of their individual signal.

**Layer B — Temperature Scaling (Probability Calibration)**
Applies Platt scaling (a single learned temperature parameter) to the Random Forest win probabilities to convert raw scores into well-calibrated probabilities. The temperature is fitted on a held-out validation fold.

**Layer C — Bayesian Model Averaging**
Blends the three base-model predictions using inverse-variance weighting. Models with lower variance on the validation set receive higher weight. Weights are saved to `bma_weights.joblib` and are not refitted at inference time.

### Validation Strategy

**Leave-One-Season-Out (LOSO):** The primary validation strategy. Each season from 2022 to 2024 is held out in turn — the model trains on all other seasons and predicts the held-out one. This simulates real deployment where an entire season's worth of new cars and driver transfers create distribution shift.

**Walk-Forward (Plackett-Luce):** The `walk_forward_model.py` adds a second validation track. The Plackett-Luce model represents the probability of a finishing order given latent driver quality scores θ:

```
P(ranking | θ) = ∏_i  exp(θ_πi) / Σ_j exp(θ_πj)
```

This softmax-over-remaining formulation (Luce 1959) treats each race result as a single draw from a complete ordering. After each race, driver quality estimates are updated via Bayesian inference, enabling real-time belief refinement as the season progresses. This is not ELO (which is pairwise); Plackett-Luce models the full finishing order as one event.

---

## Key Results

| Metric | Value |
|---|---|
| Winner prediction accuracy (2025 backtest) | ~59.2% |
| Podium prediction accuracy (LOSO avg) | ~50.3% |
| Field lap-time spread compression 2019→2025 | ~6–7 s → ~2–3 s behind fastest |
| Performance gap top-3 vs. bottom-3 teams (quali gap to pole) | 3.57% (2021) → 1.76% (2025) |
| Training data size | 2,977 driver-race entries, 149 rounds, 2018–2025 |

**Known limitations:**
- Missing/corrupted: 6 sessions from 2018–2019 (filled with rolling-average imputation)
- 2022 regulation change is confounded with cost-cap introduction (acknowledged in report)
- Wet-session qualifying times treated as missing (binary rain flag added instead)
- 2026 active aero and new PU regs are unmodeled — forward predictions assume historical relative pace holds

---

## Phase Timeline

| Phase | Due | Contents |
|---|---|---|
| Phase 1 | Mon, Feb 23, 2026 | Discovery document + raw data (`discovery_raw_data/`) |
| Phase 2 | Mon, Mar 16, 2026 | Data organization + plot sketches (`data_organized/`) |
| Phase 3 | Fri, May 1, 2026 | Final code + written report (`final_report/`) |
| Phase 4 | May 18 – Jun 1, 2026 | Presentations |

Full project rubric: `./rubric/`
Assignment instructions: `FP_instructions.md`
