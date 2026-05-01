# Final Report

This folder contains the final research document, plot-generation code, and generated figures for the Formula 1 IDP final project.

## Main Files

| File or folder | Purpose |
|---|---|
| `final.md` | Written final research report with plot explanations, conclusions, challenge goals, and references |
| `create_plots.py` | Generates the final report figures into `final_report/plots/` |
| `feature_engineering.py` | Builds model-ready race features from the FastF1 cache |
| `mach_learning.py` | Trains and evaluates the race prediction model |
| `walk_forward_model.py` | Adds in-season Plackett-Luce updating and walk-forward evaluation |
| `predict_2025.py` | Validates the trained model on the 2025 season |
| `predict_2026.py` | Produces directional 2026 predictions and WDC projection |
| `plots/` | Final PNG visualizations used by `final.md` |

## Setup

From the repository root:

```bash
python3 -m pip install -r requirements.txt
```

The slowest step is downloading FastF1 data. The raw cache is stored in `discovery_raw_data/`; once it is present, preprocessing and plot generation run offline.

```bash
python3 cache_fastf1_data.py
python3 data_organized/prepocess_data.py
python3 final_report/feature_engineering.py
python3 final_report/mach_learning.py
python3 final_report/walk_forward_model.py
python3 final_report/predict_2025.py
python3 final_report/create_plots.py
```

The model-training scripts require `xgboost`. If that dependency is missing, the report and existing generated plots can still be read, but the ML artifacts cannot be regenerated.

## Plot Index

| Plot | Research question | What it supports |
|---|---|---|
| `rq1a_teammate_quali_delta.png` | RQ1 | Teammate qualifying gaps as a driver-skill signal |
| `rq1b_throttle_brake_kde.png` | RQ1 | Telemetry evidence for braking/throttle technique differences |
| `rq2a_points_heatmap.png` | RQ2 | Whether points spread beyond the historic top teams |
| `rq2b_gap_to_winner_violin.png` | RQ2 | Whether race gaps compressed before vs. after the cost cap |
| `rq2_1a_field_spread_ridge.png` | RQ2.1 | Year-by-year field pace convergence |
| `rq2_1b_performance_tiers_step.png` | RQ2.1 | Top-three vs. bottom-three team performance spread |
| `rq3a_wdc_radar.png` | RQ4 | Champion feature profile and car/driver factor split |
| `rq3b_standings_confusion_matrix.png` | RQ4 | Predicted vs. actual rank structure |
| `wf_accuracy_by_race.png` | RQ4 | Walk-forward model accuracy by race stage |
| `wf_model_comparison.png` | RQ4 | Base, walk-forward, and oracle accuracy by season |
| `predicted_vs_actual.png` | RQ4 | 2025 out-of-sample prediction quality |
| `feature_importances.png` | RQ1/RQ4 | Model signal split between car, driver, form, and circuit features |

Additional supporting plots in `plots/` provide season fingerprints, team convergence, pole conversion, tyre degradation, and telemetry deep dives.

## Challenge Goals

- **API usage:** `cache_fastf1_data.py` downloads race weekends through the FastF1 API and stores a local resumable cache.
- **Machine learning:** the final model uses XGBoost, XGBRanker, Random Forest, ExtraTrees, leave-one-season-out validation, feature importance review, 2025 validation, and walk-forward updating.

## Presentation

Final presentation link: pending.
