"""
feature_engineering.py — F1 Race Prediction Feature Builder
=============================================================
Extracts machine-learning-ready features from FastF1 cached sessions for
every race in the training window (2022–2024).

One row = one driver at one race weekend.

HOW WE BEAT THE REFERENCE MODEL (mar-antaya/2025_f1_predictions):
  The reference model trains one GradientBoostingRegressor per circuit on
  ~20 driver rows from a single historical race. It uses qualifying time,
  sector times, weather, and team points — essentially the same 4–6 features
  for every circuit with no cross-validation.

  We improve in five ways:
    1. Train on ALL circuits simultaneously (2022–2024), giving hundreds of
       race-weekend observations rather than ~20 per circuit.
    2. Use richer session-derived features: FP2 sector splits, long-run
       race-pace simulation, lap-time consistency, pace progression across
       sessions, and quali gap to pole (seconds, not integer rank).
    3. Encode driver identity through their accumulated season performance
       (points %, average finish so far) instead of ELO, which is
       mathematically unstable in a sport dominated by car performance.
    4. Capture teammate-normalised qualifying pace — isolating the driver
       contribution from the car contribution in qualifying.
    5. Validate with leave-one-season-out CV and report real out-of-sample
       metrics (winner accuracy, podium accuracy, position MAE).

FEATURES (25 total after Q1 + Q2):

  ── Q1 BASELINE (13 features) ──────────────────────────────────────────
  fp1_pace_delta          best-lap delta vs session best in FP1 (s)
  fp2_pace_delta          best-lap delta vs session best in FP2 (s)
  fp3_pace_delta          best-lap delta vs session best in FP3 (s)
  sprint_quali_pos        sprint qualifying finishing position (NaN if none)
  sprint_pos              sprint race finishing position (NaN if none)
  grid_position           official qualifying grid position
  rolling_avg_finish_5    driver's avg finishing pos, last 5 races
  rolling_dnf_rate_5      fraction of last 5 races that were DNFs
  rolling_points_5        total points scored in last 5 races
  team_rolling_avg_5      team's avg finishing pos, last 5 races
  circuit_driver_avg_finish  driver's career avg finish at this circuit
  circuit_team_avg_finish    team's career avg finish at this circuit
  IsSprintWeekend         1 if sprint format weekend, else 0

  ── Q2 ENHANCEMENTS (12 new features, 25 total) ────────────────────────
  fp2_s1_delta            best Sector 1 time vs session best in FP2 (s)
  fp2_s2_delta            best Sector 2 time vs session best in FP2 (s)
  fp2_s3_delta            best Sector 3 time vs session best in FP2 (s)
  fp2_long_run_delta      median long-run (≥5 lap stint) pace vs session
                          median in FP2; best in-session proxy for race pace
  fp2_consistency         std dev of FP2 quicklaps (lower = more consistent)
  fp_pace_progression     fp1_pace_delta − fp2_pace_delta; positive means
                          the driver closed the gap to the field best from
                          FP1 to FP2 (session-to-session improvement signal)
  quali_gap_to_pole       absolute seconds behind pole in qualifying;
                          finer signal than integer grid position
  quali_teammate_gap      driver's quali time minus their fastest teammate's
                          quali time (positive = slower than teammate)
  quali_track_temp        track surface temperature during qualifying (°C)
  quali_wet               1 if qualifying session had rainfall, else 0
  season_pts_pct          driver's accumulated season points divided by
                          (round_number × 25); 1.0 = winning every race
  season_avg_finish       driver's average finishing pos so far this season
                          (before current race)

TARGET:
  finish_position   actual race finishing position (1 = winner)
  is_winner         1 if finish_position == 1
  is_top10          1 if finish_position <= 10

HOW TO RUN (from project root):
  python final_report/feature_engineering.py

PREREQUISITES:
  1. python cache_fastf1_data.py   (downloads FastF1 sessions to cache)
  2. pip install fastf1 pandas numpy

OUTPUT:
  data_organized/ml_training_features.csv   training rows (2022–2024)
  data_organized/ml_2025_features.csv       2025 prediction rows

REFERENCES:
  FastF1 docs     : https://docs.fastf1.dev/
  Session results : https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
  Laps            : https://docs.fastf1.dev/core.html#fastf1.core.Laps
  Weather data    : https://docs.fastf1.dev/core.html#fastf1.core.Session.weather_data
  Event schedule  : https://docs.fastf1.dev/events.html
  Long-run pace   : Mercedes F1 Strategy Report (2019)
  Teammate gap    : Bell et al. (2016), "Formula One race strategies"
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

CACHE_DIR    = "discovery_raw_data"
OUTPUT_DIR   = "data_organized"
TRAIN_YEARS  = [2022, 2023, 2024]
PREDICT_YEAR = 2025
ROLLING_N    = 5

logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.Cache.offline_mode(enabled=True)


# ─────────────────────────────────────────────────────────────────
# SESSION LOADER
# ─────────────────────────────────────────────────────────────────

def load_session(year, round_num, session_type,
                 laps=True, telemetry=False, weather=False):
    """
    Safely load a FastF1 session from local cache.
    Returns None on any failure (missing cache, cancelled session, etc.).

    session_type: 'FP1','FP2','FP3','Q','SQ','S','R'
    Ref: https://docs.fastf1.dev/events.html#session-identifiers
    """
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(laps=laps, telemetry=telemetry,
                     weather=weather, messages=False)
        return session
    except Exception as e:
        print(f"    [skip] {year} R{round_num:02d} {session_type}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Q1 HELPERS
# ─────────────────────────────────────────────────────────────────

def best_lap_delta(session):
    """
    Per-driver best lap time delta vs session best (0 = fastest driver).
    Uses pick_quicklaps() to exclude pit, formation, and outlier laps.

    Returns dict: {driver_abbr: delta_seconds}
    Ref: https://docs.fastf1.dev/core.html#fastf1.core.Laps.pick_quicklaps
    """
    if session is None:
        return {}
    try:
        laps = session.laps
        if laps is None or laps.empty:
            return {}
        quick = laps.pick_quicklaps()
        if quick.empty:
            quick = laps
        best = (
            quick.dropna(subset=["LapTime"])
                 .groupby("Driver")["LapTime"]
                 .min()
                 .dt.total_seconds()
        )
        if best.empty:
            return {}
        return (best - best.min()).to_dict()
    except Exception as e:
        print(f"    [warn] best_lap_delta: {e}")
        return {}


def get_session_positions(session):
    """Return {driver_abbr: position} from session.results."""
    if session is None:
        return {}
    try:
        results = session.results
        if results is None or results.empty:
            return {}
        out = {}
        for _, row in results.iterrows():
            abbr = row.get("Abbreviation", None)
            pos  = row.get("Position", np.nan)
            if abbr and pd.notna(pos):
                out[str(abbr)] = float(pos)
        return out
    except Exception:
        return {}


def get_race_results_dict(race_session):
    """
    Return {abbr: {finish_pos, points, status, team, grid_pos}} from race.
    Ref: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
    """
    if race_session is None:
        return {}
    try:
        out = {}
        for _, row in race_session.results.iterrows():
            abbr = str(row.get("Abbreviation", ""))
            pos  = row.get("Position", np.nan)
            if abbr:
                out[abbr] = {
                    "finish_pos": float(pos) if pd.notna(pos) else np.nan,
                    "points":     float(row.get("Points", 0) or 0),
                    "status":     str(row.get("Status", "")),
                    "team":       str(row.get("TeamName", "")),
                    "grid_pos":   float(row.get("GridPosition", np.nan)
                                       if pd.notna(row.get("GridPosition")) else np.nan),
                }
        return out
    except Exception as e:
        print(f"    [warn] get_race_results_dict: {e}")
        return {}


def is_dnf(status: str) -> bool:
    s = str(status).strip()
    return s != "Finished" and not s.startswith("+")


# ─────────────────────────────────────────────────────────────────
# Q2 HELPERS — SESSION-DERIVED TRACK FEATURES
# ─────────────────────────────────────────────────────────────────

def sector_pace_deltas_fp2(session):
    """
    Per-driver best sector time delta vs session best for each of S1/S2/S3.

    WHY SECTORS?
    A single lap delta collapses the track into one number. Sector splits
    reveal *where* a driver is fast or slow. S1 typically reflects high-speed
    aero performance; S2 reflects mechanical grip; S3 reflects traction/rear
    stability. A team that dominates S1 but loses in S3 has a predictable
    weakness the model can learn.

    Returns dict: {driver_abbr: {"s1": Δs, "s2": Δs, "s3": Δs}}
    Ref: https://docs.fastf1.dev/core.html#fastf1.core.Laps  (Sector1Time etc.)
    """
    if session is None:
        return {}
    try:
        laps = session.laps.copy()
        if laps.empty:
            return {}
        for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
            laps[col] = pd.to_timedelta(laps[col]).dt.total_seconds()
        laps = laps.dropna(subset=["Sector1Time", "Sector2Time", "Sector3Time"])
        if laps.empty:
            return {}
        # Remove SC/pit outliers (anything > 150% of sector median)
        s1_med = laps["Sector1Time"].median()
        laps = laps[laps["Sector1Time"] < s1_med * 1.5]
        if laps.empty:
            return {}
        best = laps.groupby("Driver").agg(
            s1=("Sector1Time", "min"),
            s2=("Sector2Time", "min"),
            s3=("Sector3Time", "min"),
        )
        out = {}
        for driver, row in best.iterrows():
            out[driver] = {
                "s1": round(row["s1"] - best["s1"].min(), 3),
                "s2": round(row["s2"] - best["s2"].min(), 3),
                "s3": round(row["s3"] - best["s3"].min(), 3),
            }
        return out
    except Exception as e:
        print(f"    [warn] sector_pace_deltas_fp2: {e}")
        return {}


def long_run_pace_delta(session, min_stint_laps=5):
    """
    Per-driver median lap time on ≥ min_stint_laps consecutive laps (same
    stint, same compound, no pit in/out), relative to the session-wide
    long-run median.

    WHY LONG RUNS?
    FP2 is the only session where teams explicitly run race simulations
    (long stints on race-spec fuel loads and tyre compounds). Long-run
    pace is the best available proxy for actual race pace *before* the race.
    This is the metric Mercedes's Strategy team uses internally.

    Returns dict: {driver_abbr: delta_seconds}
      Negative = faster than average long-run pace
      NaN      = driver ran no stint long enough
    Ref: https://docs.fastf1.dev/core.html#fastf1.core.Laps  (Stint column)
    """
    if session is None:
        return {}
    try:
        laps = session.laps.copy()
        if laps.empty:
            return {}
        laps["LapTime_s"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()
        laps = laps.dropna(subset=["LapTime_s", "Stint"])
        laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
        if laps.empty:
            return {}
        med = laps["LapTime_s"].median()
        if pd.isna(med):
            return {}
        laps = laps[laps["LapTime_s"] < med * 1.12]  # exclude SC/inlap outliers

        long_run_frames = []
        for (driver, stint), group in laps.groupby(["Driver", "Stint"]):
            if len(group) >= min_stint_laps:
                long_run_frames.append(group)

        if not long_run_frames:
            return {}

        lr = pd.concat(long_run_frames)
        session_lr_median = lr["LapTime_s"].median()
        driver_lr_median  = lr.groupby("Driver")["LapTime_s"].median()
        return {d: round(m - session_lr_median, 3)
                for d, m in driver_lr_median.items()}
    except Exception as e:
        print(f"    [warn] long_run_pace_delta: {e}")
        return {}


def lap_consistency(session):
    """
    Per-driver standard deviation of lap times across quicklaps in a session.
    Lower std dev = more consistent = better sign for race performance.

    Returns dict: {driver_abbr: std_dev_seconds}
    """
    if session is None:
        return {}
    try:
        laps = session.laps
        if laps is None or laps.empty:
            return {}
        quick = laps.pick_quicklaps()
        if quick.empty:
            quick = laps
        quick = quick.copy()
        quick["LapTime_s"] = pd.to_timedelta(quick["LapTime"]).dt.total_seconds()
        quick = quick.dropna(subset=["LapTime_s"])
        std = quick.groupby("Driver")["LapTime_s"].std()
        std = std.fillna(0.0)
        return std.to_dict()
    except Exception as e:
        print(f"    [warn] lap_consistency: {e}")
        return {}


def quali_gap_to_pole_and_teammate(quali_session):
    """
    For each driver compute:
      (a) gap_to_pole: best quali time − session pole time (seconds)
      (b) teammate_gap: best quali time − fastest teammate's quali time
          positive = slower than teammate, negative = faster

    Returns dict: {driver_abbr: {"gap_to_pole": s, "teammate_gap": s}}
    Ref: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
         Columns Q1, Q2, Q3 (pd.Timedelta)
    """
    if quali_session is None or quali_session.results is None:
        return {}
    try:
        results = quali_session.results.copy()
        for col in ["Q1", "Q2", "Q3"]:
            results[col] = pd.to_timedelta(results[col]).dt.total_seconds()
        results["BestQ"] = results[["Q1", "Q2", "Q3"]].min(axis=1)

        pole_time = results["BestQ"].min()
        if pd.isna(pole_time):
            return {}

        out = {}
        for team, group in results.groupby("TeamName"):
            valid = group.dropna(subset=["BestQ"])
            fastest_teammate = valid["BestQ"].min()
            for _, row in valid.iterrows():
                abbr = str(row["Abbreviation"])
                gap_to_pole   = round(row["BestQ"] - pole_time, 3)
                teammate_gap  = round(row["BestQ"] - fastest_teammate, 3)
                out[abbr] = {
                    "gap_to_pole":  gap_to_pole,
                    "teammate_gap": teammate_gap,
                }
        return out
    except Exception as e:
        print(f"    [warn] quali_gap_to_pole_and_teammate: {e}")
        return {}


def get_quali_weather(quali_session):
    """
    Return (track_temp_C, is_wet) from qualifying weather data.

    Ref: https://docs.fastf1.dev/core.html#fastf1.core.Session.weather_data
         Columns: TrackTemp, AirTemp, Rainfall (bool)
    """
    if quali_session is None:
        return np.nan, 0
    try:
        wd = quali_session.weather_data
        if wd is None or wd.empty:
            return np.nan, 0
        track_temp = float(wd["TrackTemp"].median()) if "TrackTemp" in wd.columns else np.nan
        is_wet     = int(wd["Rainfall"].any()) if "Rainfall" in wd.columns else 0
        return track_temp, is_wet
    except Exception:
        return np.nan, 0


# ─────────────────────────────────────────────────────────────────
# ROLLING STATS — recent form per driver and team
# ─────────────────────────────────────────────────────────────────

class RollingStats:
    """
    Chronological race history per driver/team.
    Call .update() AFTER emitting rows for a race to avoid leakage.
    """

    def __init__(self, window=5):
        self.window = window
        # {abbr: [(finish_pos, points, is_dnf), ...]}
        self._driver = defaultdict(list)
        # {team: [finish_pos, ...]}
        self._team   = defaultdict(list)
        # {abbr: [points, ...]} for season-level accumulators
        self._season_pts   = defaultdict(list)
        self._season_finsh = defaultdict(list)

    def get_driver_rolling(self, driver):
        hist = self._driver[driver][-self.window:]
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
            "rolling_dnf_rate_5":   np.mean(dnfs)      if dnfs      else np.nan,
            "rolling_points_5":     np.sum(points)     if points    else np.nan,
        }

    def get_team_rolling(self, team):
        hist = self._team[team][-(self.window * 2):]
        positions = [p for p in hist if pd.notna(p)]
        return {"team_rolling_avg_5": np.mean(positions) if positions else np.nan}

    def get_season_stats(self, driver):
        """
        Season-to-date stats BEFORE the current race.
        Returns season_pts_pct (normalised) and season_avg_finish.
        """
        pts   = self._season_pts[driver]
        finsh = self._season_finsh[driver]
        n     = len(finsh)
        if n == 0:
            return {"season_pts_pct": np.nan, "season_avg_finish": np.nan}
        # Normalise: points / (races_so_far × 25) so 1.0 = winner of every race
        pts_pct = sum(pts) / (n * 25) if n > 0 else np.nan
        return {
            "season_pts_pct":    round(pts_pct, 4),
            "season_avg_finish": round(np.mean(finsh), 2) if finsh else np.nan,
        }

    def update(self, race_results, new_season=False):
        """
        Update histories after a race.
        Pass new_season=True at the first race of each year to reset
        season accumulators.
        """
        if new_season:
            self._season_pts.clear()
            self._season_finsh.clear()

        for abbr, info in race_results.items():
            fp   = info["finish_pos"]
            pts  = info["points"]
            dnf  = is_dnf(info["status"])
            team = info["team"]

            self._driver[abbr].append((fp, pts, dnf))
            if team and pd.notna(fp):
                self._team[team].append(fp)

            if pd.notna(pts):
                self._season_pts[abbr].append(float(pts))
            if pd.notna(fp):
                self._season_finsh[abbr].append(float(fp))


# ─────────────────────────────────────────────────────────────────
# CIRCUIT HISTORY — per-driver and per-team at each track
# ─────────────────────────────────────────────────────────────────

class CircuitHistory:
    """Historical avg finish at each circuit per driver and team."""

    def __init__(self):
        self._driver = defaultdict(list)
        self._team   = defaultdict(list)

    def get_driver_avg(self, driver, circuit):
        h = self._driver[(driver, circuit)]
        return np.mean(h) if h else np.nan

    def get_team_avg(self, team, circuit):
        h = self._team[(team, circuit)]
        return np.mean(h) if h else np.nan

    def update(self, race_results, circuit):
        for abbr, info in race_results.items():
            fp   = info["finish_pos"]
            team = info["team"]
            if pd.notna(fp):
                self._driver[(abbr, circuit)].append(fp)
                if team:
                    self._team[(team, circuit)].append(fp)


# ─────────────────────────────────────────────────────────────────
# MAIN FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────

def build_features(years, include_race_target=True):
    """
    Build the feature DataFrame for all races in `years`.

    For each race weekend we:
      1. Load FP1/FP2/FP3 (or sprint variants) for pace features.
      2. Load Qualifying with weather to get grid position, quali gaps,
         track temperature, and rain flag.
      3. Compute rolling/season stats from prior races (no leakage).
      4. Emit one row per driver.
      5. Update rolling/season/circuit state AFTER emitting rows.

    The warm-up years (2019–2021) are processed first so rolling and
    circuit-history features are populated before the 2022 training window.

    Parameters
    ----------
    years : list[int]
        Years whose races should appear as rows in the output DataFrame.
    include_race_target : bool
        If True, add finish_position / is_winner / is_top10 target columns
        from the Race session.  Set False when predicting future races.

    Returns
    -------
    pd.DataFrame — one row per (race, driver)
    """
    rolling      = RollingStats(window=ROLLING_N)
    circuit_hist = CircuitHistory()
    all_rows     = []

    # Warm up rolling/circuit-history state using all seasons before the
    # earliest year in `years`.  This is critical when years=[2025] because
    # the 2022-2024 race results must be processed first so the rolling and
    # season-stats features are populated correctly.
    min_year     = min(years)
    warmup_start = 2019          # reliable FastF1 coverage from 2018; 2019+ for stability
    warmup_years = list(range(warmup_start, min_year))   # e.g. 2019-2024 when years=[2025]
    all_years    = sorted(set(warmup_years + list(years)))
    prev_year    = None

    for year in all_years:
        print(f"\n{'='*60}")
        print(f"  Year: {year}")
        print(f"{'='*60}")

        is_new_season = (year != prev_year)
        prev_year = year

        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"  [!] Cannot load schedule for {year}: {e}")
            continue

        for idx, (_, event) in enumerate(schedule.iterrows()):
            round_num        = int(event["RoundNumber"])
            event_name       = str(event["EventName"])
            circuit          = str(event.get("Location", event_name))
            is_sprint_wknd   = str(event.get("EventFormat", "")).lower() == "sprint"

            print(f"\n  R{round_num:02d} {event_name}"
                  f"{'  [SPRINT]' if is_sprint_wknd else ''}")

            # ── Load FP sessions ───────────────────────────────────────────
            fp1 = load_session(year, round_num, "FP1")
            fp2 = load_session(year, round_num, "FP2")
            fp3 = load_session(year, round_num, "FP3")

            sprint_quali = None
            sprint_race  = None
            if is_sprint_wknd:
                sprint_quali = load_session(year, round_num, "SQ")
                sprint_race  = load_session(year, round_num, "S")

            # ── Load Qualifying (with weather) ─────────────────────────────
            quali = load_session(year, round_num, "Q",
                                 laps=True, weather=True)

            # ── Q1: FP best-lap pace deltas ────────────────────────────────
            fp1_deltas = best_lap_delta(fp1)
            fp2_deltas = best_lap_delta(fp2)
            fp3_deltas = best_lap_delta(fp3)

            # ── Q1: sprint session positions ───────────────────────────────
            sprint_quali_pos = get_session_positions(sprint_quali)
            sprint_pos_dict  = get_session_positions(sprint_race)

            # ── Q1: qualifying grid positions ──────────────────────────────
            grid_pos_dict = get_session_positions(quali)

            # ── Q2: FP2 sector splits, long-run pace, consistency ──────────
            fp2_sectors  = sector_pace_deltas_fp2(fp2)
            fp2_lr       = long_run_pace_delta(fp2, min_stint_laps=5)
            fp2_consist  = lap_consistency(fp2)

            # ── Q2: qualifying gaps (pole, teammate) and weather ───────────
            quali_gaps                = quali_gap_to_pole_and_teammate(quali)
            quali_track_temp, q_wet   = get_quali_weather(quali)

            # ── Race results (target) ──────────────────────────────────────
            race_results = {}
            if include_race_target:
                race = load_session(year, round_num, "R", laps=False)
                if race is not None:
                    race_results = get_race_results_dict(race)

            # ── Driver list: prefer qualifying results ─────────────────────
            if (quali is not None
                    and quali.results is not None
                    and not quali.results.empty):
                drivers_df = quali.results[["Abbreviation", "TeamName"]].copy()
            elif race_results:
                drivers_df = pd.DataFrame([
                    {"Abbreviation": a, "TeamName": i["team"]}
                    for a, i in race_results.items()
                ])
            else:
                print(f"    [skip] No driver list for R{round_num}")
                if race_results:
                    rolling.update(race_results,
                                   new_season=(is_new_season and idx == 0))
                    circuit_hist.update(race_results, circuit)
                continue

            drivers_df = drivers_df.dropna(subset=["Abbreviation"])

            # ── Warm-up years: update state, don't emit rows ───────────────
            if year not in years:
                if race_results:
                    rolling.update(race_results,
                                   new_season=(is_new_season and idx == 0))
                    circuit_hist.update(race_results, circuit)
                continue

            # ── Emit one row per driver ────────────────────────────────────
            for _, drow in drivers_df.iterrows():
                abbr = str(drow["Abbreviation"])
                team = str(drow.get("TeamName", ""))

                # Q1 rolling features (state before this race)
                drv_roll  = rolling.get_driver_rolling(abbr)
                team_roll = rolling.get_team_rolling(team)
                circ_drv  = circuit_hist.get_driver_avg(abbr, circuit)
                circ_team = circuit_hist.get_team_avg(team, circuit)

                # Q2 season features (state before this race)
                season    = rolling.get_season_stats(abbr)

                # Q2 session features
                fp2_sects = fp2_sectors.get(abbr, {})
                fp2_lr_d  = fp2_lr.get(abbr, np.nan)
                fp2_cons  = fp2_consist.get(abbr, np.nan)
                fp1_d     = fp1_deltas.get(abbr, np.nan)
                fp2_d     = fp2_deltas.get(abbr, np.nan)
                # Pace progression: positive means driver closed the gap to
                # field best from FP1 → FP2 (improvement signal)
                progression = (fp1_d - fp2_d) if (pd.notna(fp1_d)
                                                   and pd.notna(fp2_d)) else np.nan

                q_gaps    = quali_gaps.get(abbr, {})

                row = {
                    # ── Identifiers (not model features) ──────────────────
                    "Year":            year,
                    "Round":           round_num,
                    "EventName":       event_name,
                    "Circuit":         circuit,
                    "Driver":          abbr,
                    "Team":            team,

                    # ── Q1 features ────────────────────────────────────────
                    "IsSprintWeekend":          int(is_sprint_wknd),
                    "fp1_pace_delta":           fp1_d,
                    "fp2_pace_delta":           fp2_d,
                    "fp3_pace_delta":           fp3_deltas.get(abbr, np.nan),
                    "sprint_quali_pos":         sprint_quali_pos.get(abbr, np.nan),
                    "sprint_pos":               sprint_pos_dict.get(abbr, np.nan),
                    "grid_position":            grid_pos_dict.get(abbr, np.nan),
                    **drv_roll,
                    **team_roll,
                    "circuit_driver_avg_finish": circ_drv,
                    "circuit_team_avg_finish":   circ_team,

                    # ── Q2 features ────────────────────────────────────────
                    # FP2 sector-level pace (reveals track-specific strengths)
                    "fp2_s1_delta":      fp2_sects.get("s1", np.nan),
                    "fp2_s2_delta":      fp2_sects.get("s2", np.nan),
                    "fp2_s3_delta":      fp2_sects.get("s3", np.nan),
                    # FP2 long-run race-pace proxy
                    "fp2_long_run_delta":  fp2_lr_d,
                    # FP2 lap-time consistency (std dev of quicklaps)
                    "fp2_consistency":     fp2_cons,
                    # Session-to-session pace improvement
                    "fp_pace_progression": progression,
                    # Qualifying gap to pole (seconds, finer than rank)
                    "quali_gap_to_pole":   q_gaps.get("gap_to_pole",  np.nan),
                    # Qualifying gap vs fastest teammate (driver vs car)
                    "quali_teammate_gap":  q_gaps.get("teammate_gap", np.nan),
                    # Qualifying surface conditions
                    "quali_track_temp":    quali_track_temp,
                    "quali_wet":           q_wet,
                    # Season-to-date performance (normalised)
                    **season,
                }

                if include_race_target and race_results:
                    rinfo = race_results.get(abbr, {})
                    fp    = rinfo.get("finish_pos", np.nan)
                    row["finish_position"] = fp
                    row["is_winner"]       = int(fp == 1)  if pd.notna(fp) else np.nan
                    row["is_top10"]        = int(fp <= 10) if pd.notna(fp) else np.nan
                elif include_race_target:
                    row["finish_position"] = np.nan
                    row["is_winner"]       = np.nan
                    row["is_top10"]        = np.nan

                all_rows.append(row)

            # ── Update state AFTER emitting rows ──────────────────────────
            if race_results:
                rolling.update(race_results,
                               new_season=(is_new_season and idx == 0))
                circuit_hist.update(race_results, circuit)
                is_new_season = False  # only reset once per season

            print(f"    → {len(drivers_df)} drivers processed")

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  F1 Race Prediction — Feature Engineering")
    print(f"  Training years:  {TRAIN_YEARS}")
    print(f"  Prediction year: {PREDICT_YEAR}")
    print(f"  Cache:           {CACHE_DIR}/")
    print(f"  Output:          {OUTPUT_DIR}/")
    print("=" * 60)

    # ── Training features (2022–2024) ──────────────────────────────
    print("\n[1/2] Building TRAINING features (2022–2024)...")
    train_df = build_features(TRAIN_YEARS, include_race_target=True)

    if train_df.empty:
        print("[!] No training data — check FastF1 cache.")
        return

    train_df = train_df.dropna(subset=["finish_position"])
    train_path = os.path.join(OUTPUT_DIR, "ml_training_features.csv")
    train_df.to_csv(train_path, index=False)
    print(f"\n  → Saved {train_path}")
    print(f"    {len(train_df)} rows | "
          f"{train_df['Driver'].nunique()} drivers | "
          f"{train_df['EventName'].nunique()} circuits")

    # ── 2025 prediction features ───────────────────────────────────
    print("\n[2/2] Building 2025 PREDICTION features...")
    pred_df = build_features([PREDICT_YEAR], include_race_target=True)
    pred_path = os.path.join(OUTPUT_DIR, "ml_2025_features.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  → Saved {pred_path}  ({len(pred_df)} rows)")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Feature Engineering Complete!")
    print("=" * 60)
    feat_cols = [c for c in train_df.columns
                 if c not in {"Year", "Round", "EventName", "Circuit",
                              "Driver", "Team", "finish_position",
                              "is_winner", "is_top10"}]
    print(f"\n  {len(feat_cols)} features:")
    for col in feat_cols:
        pct_na = train_df[col].isna().mean() * 100
        print(f"    {col:<35} ({pct_na:5.1f}% NaN)")

    print(f"\n  Targets (training):")
    print(f"    Winners  (is_winner=1): {int(train_df['is_winner'].sum())}")
    print(f"    Top-10   (is_top10=1):  {int(train_df['is_top10'].sum())}")
    print(f"    Total rows:             {len(train_df)}")


if __name__ == "__main__":
    main()
