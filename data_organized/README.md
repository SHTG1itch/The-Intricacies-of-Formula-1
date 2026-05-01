# Data Organized

This folder contains **cleaned, preprocessed data** ready for plotting and analysis. If the raw data from `discovery_raw_data/` is too large or messy for quick consumption, we reduce and organize it here.

## How to Generate the CSV Files

**From the project root**, run:

```bash
python data_organized/prepocess_data.py
```

This reads from the FastF1 cache in `discovery_raw_data/` and produces 4 CSV files:

| File | Research Question | Description | Key Columns |
|---|---|---|---|
| `qualifying_deltas.csv` | RQ1 — Skill vs. Car | Teammate qualifying time gaps per race (2018–2025) | Year, Team, Driver1, Driver2, Delta_s |
| `race_results.csv` | RQ2 — Cost Cap Impact | Every driver's race finish with gap to winner | Year, Driver, Position, Points, GapToWinner_s |
| `lap_time_spreads.csv` | RQ2.1 — Cost Cap Function | Each driver's median lap time delta from fastest driver per race | Year, Driver, DeltaFromFastest_s |
| `driver_season_profiles.csv` | RQ3 — WDC Prediction | Season-aggregate stats per driver | Year, Driver, AvgQualiPos, Wins, DNFs, TotalPoints |

## Prerequisites

1. **Cache the raw data first:** `python cache_fastf1_data.py`
2. **Install dependencies:** `python3 -m pip install -r requirements.txt`

## Subfolders

- **`sketches/`** — Plot sketch images ([see table](sketches/README.md))
- **`test_data/`** — Small test datasets for validating preprocessing

## Notes

- The preprocessing file is named `prepocess_data.py` because that was the original project filename; the README keeps the existing path so old links and commands continue to work.
- `race_results.csv` stores `GapToWinner_s` as a positive gap in seconds. Winners are `0.0`; lapped finishers and DNFs may be blank when FastF1 does not provide a comparable time.
