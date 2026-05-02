# The Formula of Formula 1

**Student Name 1:** Srivats Iyer  
**Student Name 2:** Aakash Bothra  
**Date:** January 27, 2026  
**Course:** Intermediate Data Programming

## Research Questions

1. **Driver vs. car:** Which measurable factors in a Formula 1 weekend are most associated with winning, and how much signal comes from the driver compared with the car?
2. **Cost cap impact:** After the FIA introduced the 2021 cost cap, did race results become less concentrated among the top teams?
3. **Lap-time convergence:** Did the gap between the fastest and slowest teams shrink after the cost cap and 2022 regulation reset?
4. **Predictability:** How accurately can a model predict race winners from pre-race data, and which features explain the model's predictions?

These questions are designed to produce measurable answers: finishing-position distributions, lap-time gaps, teammate qualifying deltas, model accuracy, and feature importance.

## Motivation

Formula 1 looks like a high-speed spectacle, but it is also a data and engineering competition. Teams make decisions from telemetry, tyre degradation, weather, qualifying pace, reliability history, and race strategy. That makes F1 a strong data-programming subject because small changes in a tenth of a second can decide a race.

The project also asks a question fans argue about constantly: does the driver win, or does the car win? We cannot answer that perfectly, but we can compare teammates in the same car, inspect telemetry, track team performance before and after the cost cap, and test whether machine-learning models rely more on car-pace or driver-specific features.

## Data Sources

The project uses the FastF1 Python API, which provides access to official Formula 1 timing data, session results, lap data, telemetry, weather, and event metadata.

- FastF1 documentation: <https://docs.fastf1.dev/>
- FastF1 GitHub repository: <https://github.com/theOehrly/Fast-F1>
- Underlying Ergast/F1 timing data reference: <https://docs.fastf1.dev/ergast.html>

The final dataset focuses on 2018-2025 because modern FastF1 telemetry coverage is strongest in that period.

## Data Pipeline

1. `cache_fastf1_data.py` downloads FastF1 sessions and saves them into `discovery_raw_data/`.
2. `data_organized/preprocess_data.py` reads the local FastF1 cache and produces cleaner CSVs for analysis.
3. `final_report/feature_engineering.py` turns practice, qualifying, weather, rolling form, circuit history, and race results into model-ready features.
4. `final_report/mach_learning.py`, `walk_forward_model.py`, and the prediction scripts train and evaluate the model.
5. `final_report/create_plots.py` generates the report plots in `final_report/plots/`.

The cache step is separated from preprocessing because FastF1 API calls are slow and rate-limited. Once the cache exists, most analysis can run offline.

## Organized Datasets

| Dataset | Main use | Example columns |
|---|---|---|
| Race results | Finishing order, points, gap to winner, model targets | Year, Round, Driver, Team, Position, Points, GapToWinner_s |
| Lap timing | Pace spread, consistency, tyre degradation, sector analysis | Driver, LapNumber, LapTime, Sector1Time, Sector2Time, Sector3Time, Compound |
| Event schedule | Chronological ordering and race metadata | RoundNumber, EventName, Location, Country, EventDate |
| Weather | Track conditions and wet-session indicators | AirTemp, TrackTemp, Rainfall, Humidity, WindSpeed |
| Telemetry | Throttle/brake behavior and corner-level analysis | Distance, Speed, Throttle, Brake |

## Data Challenges

- **Single-source dependency:** FastF1 is a rich source, but most raw data comes through the same library, so API limitations affect the whole project.
- **Large cache:** Telemetry data is too large to treat like a small CSV download, so the project uses a resumable local cache.
- **Missing sessions:** Some older sessions, wet sessions, sprint weekends, and lapped/DNF results have missing or inconsistent fields.
- **Causality limits:** The cost cap and 2022 aerodynamic regulation reset overlap, so the project can show convergence but cannot prove the exact cause.
- **Temporal leakage:** Machine-learning features must be computed using only information available before the race being predicted.

## Challenge Goals

1. **API usage:** The project uses FastF1 to collect race results, lap timing, weather, and telemetry data. The cache script handles rate limits, retries, and resumable progress.
2. **Machine learning:** The project trains multiple models, compares validation accuracy by season, studies feature importance, and validates predictions against known race outcomes.

## Schedule

Task list: <https://docs.google.com/spreadsheets/d/1zUgpSpIK_AiBIrSttxw661JSu8US9uLypAuj_rLHlhI/edit?gid=0#gid=0>
