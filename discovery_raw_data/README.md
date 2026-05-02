# Discovery Raw Data

This folder stores the local FastF1 cache used by the project. The cache is
created by `cache_fastf1_data.py` and contains raw FIA timing/session files
before any filtering or analysis-ready cleanup happens.

## Source

- FastF1 documentation: <https://docs.fastf1.dev/>
- FastF1 GitHub repository: <https://github.com/theOehrly/Fast-F1>

## How It Is Used

1. `cache_fastf1_data.py` downloads race-weekend sessions into this folder.
2. `data_organized/preprocess_data.py` reads the cached sessions and writes
   smaller CSV files to `data_organized/`.
3. The final report and plots use the organized CSVs, not these raw cache files.

The broader project discovery write-up is in `../discovery.md`.
