"""
feature_engineering.py — F1 Race Prediction Feature Builder
=============================================================
Extracts machine-learning-ready features from FastF1 cached sessions for
every race in the training window (2022–2024).

One row = one driver at one race weekend.

FEATURES EXTRACTED:
  Pre-race session pace:
    fp1_pace_delta, fp2_pace_delta, fp3_pace_delta
      → driver's best lap relative to session best (0 = fastest, positive = slower)
    sprint_quali_pos, sprint_pos
      → finishing position in sprint qualifying / sprint race (NaN if no sprint)
    grid_position
      → official qualifying/grid position

  Recent form (rolling last-N races BEFORE this race):
    rolling_avg_finish_5   → avg finishing position (lower = better)
    rolling_dnf_rate_5     → fraction of last 5 races that were DNFs
    rolling_points_5       → total points scored in last 5 races
    team_rolling_avg_5     → team's avg finish across both drivers, last 5 races

  Circuit-specific history:
    circuit_driver_avg_finish → driver's career avg finish at this track
    circuit_team_avg_finish   → team's career avg finish at this track

TARGET:
    finish_position  → actual race finishing position (1 = winner)
    is_winner        → 1 if finish_position == 1
    is_top10         → 1 if finish_position <= 10

HOW TO RUN (from project root):
    python final_report/feature_engineering.py

PREREQUISITES:
    1. Run cache_fastf1_data.py first to populate discovery_raw_data/
    2. pip install fastf1 pandas numpy

OUTPUT:
    data_organized/ml_training_features.csv   — training data (2022–2024 races)
    data_organized/ml_2025_features.csv       — 2025 pre-race features (for prediction)

REFERENCES:
    FastF1 docs          : https://docs.fastf1.dev/
    Session results      : https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
    Laps                 : https://docs.fastf1.dev/core.html#fastf1.core.Laps
    Event schedule       : https://docs.fastf1.dev/events.html
    RF + XGBoost for F1  : https://towardsdatascience.com/formula-1-race-predictor
"""

import os
import logging
import warnings
from collections import defaultdict

import fastf1
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

CACHE_DIR   = "discovery_raw_data"
OUTPUT_DIR  = "data_organized"

# Training seasons: use 2022-2024 (telemetry + FP data reliable from 2018,
# but 2022+ gives us post-regulation-change data which is more relevant)
TRAIN_YEARS = [2022, 2023, 2024]

# 2025: we gather FP/Quali/Sprint to predict race outcomes
PREDICT_YEAR = 2025

# Rolling window for recent-form features
ROLLING_N = 5

# Suppress noisy logs
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.Cache.offline_mode(enabled=True)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def load_session(year, round_num, session_type, laps=True, telemetry=False):
    """
    Safely load a FastF1 session from cache.
    Returns None on any failure.

    session_type: 'FP1','FP2','FP3','Q','SQ','S','R'
    Ref: https://docs.fastf1.dev/events.html#session-identifiers
    """
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(laps=laps, telemetry=telemetry,
                     weather=False, messages=False)
        return session
    except Exception as e:
        print(f"    [skip] {year} R{round_num:02d} {session_type}: {e}")
        return None


def best_lap_delta(session):
    """
    For each driver in a session, compute their best lap time in seconds
    relative to the overall session best (delta = 0 means fastest).

    Returns dict: {abbreviation: delta_seconds}

    We use session.laps.pick_quicklaps() to filter pit laps and outliers.
    Ref: https://docs.fastf1.dev/core.html#fastf1.core.Laps.pick_quicklaps
    """
    if session is None:
        return {}
    try:
        laps = session.laps
        if laps is None or laps.empty:
            return {}

        # pick_quicklaps filters to laps within 107% of fastest lap
        quick = laps.pick_quicklaps()
        if quick.empty:
            quick = laps  # fall back to all laps if no quicklaps

        # Best lap per driver (minimum LapTime)
        best = (
            quick
            .dropna(subset=["LapTime"])
            .groupby("Driver")["LapTime"]
            .min()
            .dt.total_seconds()
        )
        if best.empty:
            return {}
        session_best = best.min()
        deltas = (best - session_best).to_dict()
        return deltas
    except Exception as e:
        print(f"    [warn] best_lap_delta failed: {e}")
        return {}


def get_session_positions(session):
    """
    Return a dict of {abbreviation: position} from session.results.
    Handles NaN positions gracefully.
    Ref: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
    """
    if session is None:
        return {}
    try:
        results = session.results
        if results is None or results.empty:
            return {}
        pos_dict = {}
        for _, row in results.iterrows():
            abbrv = row.get("Abbreviation", None)
            pos   = row.get("Position", np.nan)
            if abbrv and pd.notna(pos):
                pos_dict[str(abbrv)] = float(pos)
        return pos_dict
    except Exception as e:
        print(f"    [warn] get_session_positions failed: {e}")
        return {}


def get_grid_positions(quali_session):
    """
    Return dict {abbreviation: grid_position} from qualifying results.
    Uses the Position column from qualifying results.
    """
    return get_session_positions(quali_session)


def get_race_results_dict(race_session):
    """
    Return dict {abbreviation: (finish_pos, points, status, team)} from race.
    """
    if race_session is None:
        return {}
    try:
        results = race_session.results
        out = {}
        for _, row in results.iterrows():
            abbrv = str(row.get("Abbreviation", ""))
            pos   = row.get("Position", np.nan)
            pts   = row.get("Points", 0.0)
            status = str(row.get("Status", ""))
            team  = str(row.get("TeamName", ""))
            grid  = row.get("GridPosition", np.nan)
            if abbrv:
                out[abbrv] = {
                    "finish_pos":  float(pos) if pd.notna(pos) else np.nan,
                    "points":      float(pts) if pd.notna(pts) else 0.0,
                    "status":      status,
                    "team":        team,
                    "grid_pos":    float(grid) if pd.notna(grid) else np.nan,
                }
        return out
    except Exception as e:
        print(f"    [warn] get_race_results_dict failed: {e}")
        return {}


def is_dnf(status: str) -> bool:
    """
    Returns True if the driver did not finish the race.
    Status is 'Finished' or '+X Laps' for classified finishers.
    """
    if not status:
        return False
    s = str(status).strip()
    return s != "Finished" and not s.startswith("+")


# ─────────────────────────────────────────────────────────────────
# ROLLING STATS BUILDER
# ─────────────────────────────────────────────────────────────────

class RollingStats:
    """
    Maintains a chronological race history per driver and team.
    Call .update() after each race to add results.
    Call .get_driver_rolling() to get features BEFORE a race.
    Call .get_team_rolling() to get team features BEFORE a race.
    """

    def __init__(self, window=5):
        self.window = window
        # {driver_abbr: [(finish_pos, points, is_dnf), ...]}
        self._driver_history = defaultdict(list)
        # {team_name: [(finish_pos_driver1, finish_pos_driver2, ...), ...]}
        # stored as flat list of finish positions per race (both drivers)
        self._team_history = defaultdict(list)

    def get_driver_rolling(self, driver: str):
        """
        Returns rolling stats for the last N races BEFORE the current one.
        """
        hist = self._driver_history[driver][-self.window:]
        if not hist:
            return {
                "rolling_avg_finish_5": np.nan,
                "rolling_dnf_rate_5":   np.nan,
                "rolling_points_5":     np.nan,
            }
        positions = [h[0] for h in hist if pd.notna(h[0])]
        dnfs      = [h[2] for h in hist]
        points    = [h[1] for h in hist if pd.notna(h[1])]
        return {
            "rolling_avg_finish_5": np.mean(positions) if positions else np.nan,
            "rolling_dnf_rate_5":   np.mean(dnfs) if dnfs else np.nan,
            "rolling_points_5":     np.sum(points) if points else np.nan,
        }

    def get_team_rolling(self, team: str):
        """
        Returns team's avg finish position over last N race appearances.
        """
        hist = self._team_history[team][-self.window * 2:]  # ~2 drivers × N races
        positions = [p for p in hist if pd.notna(p)]
        if not positions:
            return {"team_rolling_avg_5": np.nan}
        return {"team_rolling_avg_5": np.mean(positions)}

    def update(self, race_results: dict):
        """
        After a race, update histories.
        race_results: {abbr: {finish_pos, points, status, team, ...}}
        """
        for abbr, info in race_results.items():
            fp  = info["finish_pos"]
            pts = info["points"]
            dnf = is_dnf(info["status"])
            self._driver_history[abbr].append((fp, pts, dnf))

        # Update per team (collect all drivers' finish positions)
        team_positions = defaultdict(list)
        for abbr, info in race_results.items():
            team = info["team"]
            fp   = info["finish_pos"]
            if team and pd.notna(fp):
                team_positions[team].append(fp)
        for team, positions in team_positions.items():
            self._team_history[team].extend(positions)


# ─────────────────────────────────────────────────────────────────
# CIRCUIT HISTORY BUILDER
# ─────────────────────────────────────────────────────────────────

class CircuitHistory:
    """
    Tracks historical average finish per (driver, circuit) and (team, circuit).
    Only counts races BEFORE the current one (to avoid data leakage).
    """

    def __init__(self):
        # {(driver, circuit_name): [finish_positions]}
        self._driver_circuit = defaultdict(list)
        # {(team, circuit_name): [finish_positions]}
        self._team_circuit   = defaultdict(list)

    def get_driver_avg(self, driver: str, circuit: str):
        hist = self._driver_circuit[(driver, circuit)]
        return np.mean(hist) if hist else np.nan

    def get_team_avg(self, team: str, circuit: str):
        hist = self._team_circuit[(team, circuit)]
        return np.mean(hist) if hist else np.nan

    def update(self, race_results: dict, circuit: str):
        for abbr, info in race_results.items():
            fp   = info["finish_pos"]
            team = info["team"]
            if pd.notna(fp):
                self._driver_circuit[(abbr, circuit)].append(fp)
                if team:
                    self._team_circuit[(team, circuit)].append(fp)


# ─────────────────────────────────────────────────────────────────
# MAIN FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────

def build_features(years, include_race_target=True):
    """
    Iterates over all races in `years` and constructs the feature DataFrame.

    If include_race_target=True, appends race results as the target variable
    (used for training). Set to False for 2025 prediction rows.

    Returns a pandas DataFrame with one row per (race, driver).
    """
    rolling = RollingStats(window=ROLLING_N)
    circuit_hist = CircuitHistory()
    all_rows = []

    # We iterate over all years in order so rolling stats carry over
    # even if we later slice to a subset
    all_years = sorted(set([2019, 2020, 2021] + list(years)))

    for year in all_years:
        print(f"\n{'='*55}")
        print(f"  Year: {year}")
        print(f"{'='*55}")

        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"  [!] Could not load schedule for {year}: {e}")
            continue

        for _, event in schedule.iterrows():
            round_num   = int(event["RoundNumber"])
            event_name  = str(event["EventName"])
            circuit     = str(event.get("Location", event_name))
            is_sprint_weekend = bool(event.get("EventFormat", "").lower() == "sprint")

            print(f"\n  R{round_num:02d} {event_name} (sprint={is_sprint_weekend})")

            # ── 1. Load pre-race sessions ──────────────────────────────────

            fp1 = load_session(year, round_num, "FP1")
            fp2 = load_session(year, round_num, "FP2")
            fp3 = load_session(year, round_num, "FP3")

            # Sprint weekends have Sprint Qualifying (SQ) instead of FP3
            sprint_quali = None
            sprint_race  = None
            if is_sprint_weekend:
                sprint_quali = load_session(year, round_num, "SQ")
                sprint_race  = load_session(year, round_num, "S")

            quali = load_session(year, round_num, "Q")

            # ── 2. Extract pace deltas from FP sessions ────────────────────

            fp1_deltas = best_lap_delta(fp1)
            fp2_deltas = best_lap_delta(fp2)
            fp3_deltas = best_lap_delta(fp3)

            # ── 3. Sprint session results ──────────────────────────────────

            sprint_quali_pos = get_session_positions(sprint_quali)
            sprint_pos       = get_session_positions(sprint_race)

            # ── 4. Grid positions from qualifying ──────────────────────────

            grid_pos = get_grid_positions(quali)

            # ── 5. Load race results (target) ──────────────────────────────

            race_results = {}
            if include_race_target:
                race = load_session(year, round_num, "R", laps=False)
                if race is not None:
                    race_results = get_race_results_dict(race)

            # ── 6. Determine all drivers in this race weekend ──────────────

            # Collect drivers from qualifying results (most reliable)
            # Fall back to race results if quali unavailable
            if quali is not None and quali.results is not None and not quali.results.empty:
                drivers_df = quali.results[["Abbreviation", "TeamName"]].copy()
            elif race_results:
                drivers_df = pd.DataFrame([
                    {"Abbreviation": abbr, "TeamName": info["team"]}
                    for abbr, info in race_results.items()
                ])
            else:
                print(f"    [skip] No driver list available for R{round_num}")
                continue

            drivers_df = drivers_df.dropna(subset=["Abbreviation"])

            # ── 7. Build one row per driver ────────────────────────────────

            if year not in years:
                # We only build rows for actual requested years;
                # earlier years are just used to warm up rolling stats
                if race_results:
                    rolling.update(race_results)
                    circuit_hist.update(race_results, circuit)
                continue

            for _, drow in drivers_df.iterrows():
                abbr = str(drow["Abbreviation"])
                team = str(drow.get("TeamName", ""))

                # Rolling form BEFORE this race
                driver_rolling = rolling.get_driver_rolling(abbr)
                team_rolling   = rolling.get_team_rolling(team)

                # Circuit history BEFORE this race
                circ_driver_avg = circuit_hist.get_driver_avg(abbr, circuit)
                circ_team_avg   = circuit_hist.get_team_avg(team, circuit)

                row = {
                    # Identifiers (not used as ML features)
                    "Year":         year,
                    "Round":        round_num,
                    "EventName":    event_name,
                    "Circuit":      circuit,
                    "Driver":       abbr,
                    "Team":         team,
                    "IsSprintWeekend": int(is_sprint_weekend),

                    # Pre-race session pace features
                    "fp1_pace_delta":    fp1_deltas.get(abbr, np.nan),
                    "fp2_pace_delta":    fp2_deltas.get(abbr, np.nan),
                    "fp3_pace_delta":    fp3_deltas.get(abbr, np.nan),
                    "sprint_quali_pos":  sprint_quali_pos.get(abbr, np.nan),
                    "sprint_pos":        sprint_pos.get(abbr, np.nan),
                    "grid_position":     grid_pos.get(abbr, np.nan),

                    # Rolling form
                    **driver_rolling,
                    **team_rolling,

                    # Circuit-specific history
                    "circuit_driver_avg_finish": circ_driver_avg,
                    "circuit_team_avg_finish":   circ_team_avg,
                }

                # Race target (only for training years)
                if include_race_target and race_results:
                    rinfo = race_results.get(abbr, {})
                    fp  = rinfo.get("finish_pos", np.nan)
                    row["finish_position"] = fp
                    row["is_winner"]       = int(fp == 1) if pd.notna(fp) else np.nan
                    row["is_top10"]        = int(fp <= 10) if pd.notna(fp) else np.nan
                elif include_race_target:
                    row["finish_position"] = np.nan
                    row["is_winner"]       = np.nan
                    row["is_top10"]        = np.nan

                all_rows.append(row)

            # ── 8. Update rolling stats and circuit history AFTER this race ─
            if race_results:
                rolling.update(race_results)
                circuit_hist.update(race_results, circuit)

            print(f"    → {len(drivers_df)} drivers processed")

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  F1 Race Prediction — Feature Engineering")
    print(f"  Training years: {TRAIN_YEARS}")
    print(f"  Predict year:   {PREDICT_YEAR}")
    print(f"  Cache:          {CACHE_DIR}/")
    print(f"  Output:         {OUTPUT_DIR}/")
    print("=" * 55)

    # ── Build training features (2022–2024) ────────────────────────
    print("\n[1/2] Building TRAINING features (2022–2024)...")
    train_df = build_features(TRAIN_YEARS, include_race_target=True)

    if train_df.empty:
        print("[!] No training data built — check FastF1 cache.")
        return

    # Drop rows where we couldn't get a race result (incomplete season data)
    train_df = train_df.dropna(subset=["finish_position"])

    train_path = os.path.join(OUTPUT_DIR, "ml_training_features.csv")
    train_df.to_csv(train_path, index=False)
    print(f"\n  → Saved {train_path}  ({len(train_df)} rows, "
          f"{train_df['Driver'].nunique()} unique drivers, "
          f"{train_df['Round'].nunique()} unique rounds)")

    # ── Build 2025 prediction features ─────────────────────────────
    print("\n[2/2] Building 2025 PREDICTION features...")
    pred_df = build_features([PREDICT_YEAR], include_race_target=True)

    pred_path = os.path.join(OUTPUT_DIR, "ml_2025_features.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  → Saved {pred_path}  ({len(pred_df)} rows)")

    # ── Quick summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Feature Engineering Complete!")
    print("=" * 55)
    print(f"\nTraining set shape:   {train_df.shape}")
    print(f"Prediction set shape: {pred_df.shape}")
    print(f"\nFeature columns:")
    feature_cols = [c for c in train_df.columns
                    if c not in ("Year", "Round", "EventName", "Circuit",
                                 "Driver", "Team", "finish_position",
                                 "is_winner", "is_top10")]
    for col in feature_cols:
        pct_na = train_df[col].isna().mean() * 100
        print(f"  {col:<35} ({pct_na:.1f}% NaN)")

    print(f"\nTarget distribution (training):")
    print(f"  Winners (is_winner=1):  {int(train_df['is_winner'].sum())} rows")
    print(f"  Top-10 (is_top10=1):    {int(train_df['is_top10'].sum())} rows")
    print(f"  Total rows:             {len(train_df)}")


if __name__ == "__main__":
    main()
