# Build the ML feature matrix from FastF1 cached sessions.
# One row = one driver at one race weekend.
#
# The idea is simple: before a race we have FP1, FP2, FP3, qualifying, and
# sometimes sprint data. We want to extract as much useful signal as possible
# from these sessions - pace, consistency, sector performance, weather, recent
# form - and put it all into a flat CSV that the model can actually use.
#
# Took a while to figure out the right set of features. Started with just grid
# position and a couple of FP deltas, kept adding things as the model kept
# getting tripped up on specific scenarios (weather, rookies, first-timers at
# new circuits, etc.).
#
# One thing that bites you immediately if you're not careful: rolling stats and
# circuit history have to be computed BEFORE the current race result goes in.
# If you update the state then compute features you get beautiful training
# accuracy and garbage validation accuracy, it's classic leakage. Fixed this early
# but it's still important to understand.
#
# Warmup years (2019 through start of training window) are processed to
# populate rolling stats and circuit history correctly. You cannot skip the
# warmup - the first race of 2022 would have no history otherwise.
#
# run from project root: python final_report/feature_engineering.py
# needs: cache_fastf1_data.py to have been run first
#
# FastF1 docs: https://docs.fastf1.dev/
# Session results schema: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
# Laps API: https://docs.fastf1.dev/core.html#fastf1.core.Laps


import os
import logging
import warnings
from collections import defaultdict

import fastf1
import numpy as np
import pandas as pd

# config
CACHE_DIR    = "discovery_raw_data"
OUTPUT_DIR   = "data_organized"
TRAIN_YEARS  = [2022, 2023, 2024]
PREDICT_YEAR = 2025
ROLLING_N    = 5  # how many recent races to keep in the rolling window

# suppress the wall of fastf1 logs - they make the output unreadable
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.Cache.offline_mode(enabled=True)  # only use local cache, no network calls


# ── session loading ──────────────────────────────────────────────────────────

def load_session(year, round_num, session_type, laps=True, telemetry=False, weather=False):
    # wrapper around fastf1.get_session that swallows all errors
    # there are a lot of reasons a session might fail to load:
    #   - not in cache yet (future race or cache_fastf1_data.py not run)
    #   - sprint weekends don't have FP3, they have SQ and S instead
    #   - occasionally cancelled sessions (2021 Belgian GP quali was famously weird)
    # returning None lets callers handle missing sessions gracefully rather than crashing
    # ref: https://docs.fastf1.dev/events.html#session-identifiers
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(laps=laps, telemetry=telemetry, weather=weather, messages=False)
        return session
    except Exception as e:
        print(f"    [skip] {year} R{round_num:02d} {session_type}: {e}")
        return None


# ── basic session helpers ─────────────────────────────────────────────────────

def best_lap_delta(session):
    # per-driver best lap time minus the session best lap time (0 = fastest driver)
    # pick_quicklaps filters out pit laps, outlaps, safety car laps and other junk
    # ref: https://docs.fastf1.dev/core.html#fastf1.core.Laps.pick_quicklaps
    if session is None:
        return {}
    try:
        laps = session.laps
        if laps is None or laps.empty:
            return {}
        quick = laps.pick_quicklaps()
        if quick.empty:
            quick = laps  # fallback if no quicklaps (happens in very short sessions)
        best = (
            quick.dropna(subset=["LapTime"])
                 .groupby("Driver")["LapTime"]
                 .min()
                 .dt.total_seconds()
        )
        if best.empty:
            return {}
        return (best - best.min()).to_dict()  # delta from session minimum
    except Exception as e:
        print(f"    [warn] best_lap_delta: {e}")
        return {}


def get_session_positions(session):
    # {driver_abbr: position} from session.results
    # used for sprint qualifying pos, sprint race pos, qualifying grid pos
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
    # pull everything we need from race.results into a clean dict
    # status field is how we detect DNFs later (rolling_dnf_rate_5)
    # ref: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
    if race_session is None:
        return {}
    try:
        out = {}
        for _, row in race_session.results.iterrows():
            abbr = str(row.get("Abbreviation", ""))
            pos  = row.get("Position", np.nan)
            if not abbr:
                continue
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


def is_dnf(status):
    # "Finished" or "+X laps" means they completed the race; everything else is a DNF
    # ref: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
    s = str(status).strip()
    return s != "Finished" and not s.startswith("+")


# ── FP2 detailed features ─────────────────────────────────────────────────────

def sector_pace_deltas_fp2(session):
    # best sector time per driver vs session best, for each of S1/S2/S3
    # sectors tell you WHERE on track the car is quick or slow:
    #   S1 is typically high-speed aero-dependent corners
    #   S2 is medium-speed mechanical grip sections
    #   S3 covers the traction zones and slower final chicanes
    # knowing sector splits means you can see if a car is good in high-speed
    # corners but weak in slow zones - useful context the model can exploit
    # ref: https://docs.fastf1.dev/core.html#fastf1.core.Laps
    if session is None:
        return {}
    try:
        laps = session.laps.copy()
        if laps.empty:
            return {}
        # convert timedelta columns to seconds
        for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
            laps[col] = pd.to_timedelta(laps[col]).dt.total_seconds()
        laps = laps.dropna(subset=["Sector1Time", "Sector2Time", "Sector3Time"])
        if laps.empty:
            return {}
        # filter out laps that are obviously slow (SC, outlaps, etc.)
        # 1.5x the median S1 time is a reasonable cutoff for contaminated laps
        s1_med = laps["Sector1Time"].median()
        laps   = laps[laps["Sector1Time"] < s1_med * 1.5]
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
    # per-driver median long-stint pace vs session median
    # this is the best prediction for race pace you can get before the race
    # teams run fuel-loaded race simulations in FP2 - similar fuel load and
    # tyre compound to what they'll use at the start of the race
    # a driver with a better long run than their qualifying position suggests
    # is almost always a threat once things settle down in race trim
    # negative delta = faster than field median long-run pace
    # ref: https://docs.fastf1.dev/core.html#fastf1.core.Laps (Stint column)
    if session is None:
        return {}
    try:
        laps = session.laps.copy()
        if laps.empty:
            return {}
        laps["LapTime_s"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()
        laps = laps.dropna(subset=["LapTime_s", "Stint"])
        # remove pit in/out laps - those are always artificially slow
        laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
        if laps.empty:
            return {}
        med = laps["LapTime_s"].median()
        if pd.isna(med):
            return {}
        laps = laps[laps["LapTime_s"] < med * 1.12]  # drop SC laps and outlap contamination

        long_stints = []
        for (driver, stint), group in laps.groupby(["Driver", "Stint"]):
            if len(group) >= min_stint_laps:
                long_stints.append(group)

        if not long_stints:
            return {}

        lr                = pd.concat(long_stints)
        session_lr_median = lr["LapTime_s"].median()
        driver_lr_median  = lr.groupby("Driver")["LapTime_s"].median()
        return {d: round(m - session_lr_median, 3) for d, m in driver_lr_median.items()}
    except Exception as e:
        print(f"    [warn] long_run_pace_delta: {e}")
        return {}


def long_run_pace_delta_fp3(session, min_stint_laps=4):
    # same logic as the FP2 long run function but for FP3 on Saturday morning
    # FP3 is shorter so we drop the minimum stint threshold to 4 laps
    # it's a noisier signal than FP2 because teams often do qualifying simulations
    # in FP3 rather than race sims, but when they do run long stints it's very fresh
    # some teams save it for FP3 because they want Saturday track conditions
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
        laps = laps[laps["LapTime_s"] < med * 1.12]

        long_stints = []
        for (driver, stint), group in laps.groupby(["Driver", "Stint"]):
            if len(group) >= min_stint_laps:
                long_stints.append(group)

        if not long_stints:
            return {}

        lr = pd.concat(long_stints)
        session_lr_median = lr["LapTime_s"].median()
        driver_lr_median  = lr.groupby("Driver")["LapTime_s"].median()
        return {d: round(m - session_lr_median, 3) for d, m in driver_lr_median.items()}
    except Exception as e:
        print(f"    [warn] long_run_pace_delta_fp3: {e}")
        return {}


def lap_consistency(session):
    # std deviation of quicklap times per driver within a session
    # lower = more consistent driver or more stable car setup
    # a driver lapping within 0.1s of their own pace every lap is in better
    # shape than someone fluctuating by 0.5s - even if the mean is the same
    # ref: https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.medcouple.html
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
        std = quick.groupby("Driver")["LapTime_s"].std().fillna(0.0)
        return std.to_dict()
    except Exception as e:
        print(f"    [warn] lap_consistency: {e}")
        return {}


# ── qualifying-specific features ──────────────────────────────────────────────

def quali_gap_to_pole_and_teammate(quali_session):
    # two features from qualifying:
    #   gap_to_pole: how far behind pole were they in seconds? absolute pace benchmark
    #   teammate_gap: their time minus their fastest teammate's time
    #
    # the teammate gap idea is from Bell et al. 2016 - comparing teammates in the
    # same car lets you isolate driver skill by cancelling out the car contribution
    # it's one of the more principled features in the whole set
    # ref: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults (Q1/Q2/Q3 columns)
    if quali_session is None or quali_session.results is None:
        return {}
    try:
        results = quali_session.results.copy()
        for col in ["Q1", "Q2", "Q3"]:
            results[col] = pd.to_timedelta(results[col]).dt.total_seconds()
        results["BestQ"] = results[["Q1", "Q2", "Q3"]].min(axis=1)  # best of all three segments

        pole_time = results["BestQ"].min()
        if pd.isna(pole_time):
            return {}

        out = {}
        for team, group in results.groupby("TeamName"):
            valid            = group.dropna(subset=["BestQ"])
            fastest_teammate = valid["BestQ"].min()
            for _, row in valid.iterrows():
                abbr = str(row["Abbreviation"])
                gap_abs = row["BestQ"] - pole_time
                out[abbr] = {
                    "gap_to_pole":  round(gap_abs, 3),
                    "teammate_gap": round(row["BestQ"] - fastest_teammate, 3),  # 0 if fastest in team
                    "gap_pct":      round(gap_abs / pole_time, 6) if pole_time > 0 else np.nan,  # fractional gap - scale-invariant across circuits
                }
        return out
    except Exception as e:
        print(f"    [warn] quali_gap_to_pole_and_teammate: {e}")
        return {}


def quali_q3_appearances(quali_session):
    # did each driver set a Q3 time? returns {abbr: 1} or {abbr: 0}
    # this matters because Q3 appearance means the car was clearly in top-10 pace
    # a driver who qualifies P11 after being knocked out in Q2 is fundamentally
    # different from someone who qualifies P10 after safely making it through Q3
    # grid position alone doesn't capture whether they were Q2 or Q3 material
    # the model was confusing these cases at the P10/P11 boundary - this fixes it
    # Q3 column in fastf1 results is NaN if the driver didn't reach Q3
    # ref: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
    if quali_session is None or quali_session.results is None:
        return {}
    try:
        results = quali_session.results.copy()
        if "Q3" not in results.columns:
            return {}
        q3_times = pd.to_timedelta(results["Q3"]).dt.total_seconds()
        out = {}
        for i, row in results.iterrows():
            abbr = str(row.get("Abbreviation", ""))
            if abbr:
                q3_valid = q3_times.iloc[results.index.get_loc(i)] if i in results.index else np.nan
                out[abbr] = int(pd.notna(q3_valid))
        return out
    except Exception as e:
        print(f"    [warn] quali_q3_appearances: {e}")
        return {}


def get_session_weather(session):
    # return (track_temp, air_temp, is_wet) from any session's weather data
    # we take the median across the session to smooth transient spikes
    # FP2 Friday afternoon track temp is a good proxy for Sunday conditions -
    # same time of day, ambient temperature, track rubber is being laid down
    # qualifying air temp matters for tyre thermal behaviour in the race
    # ref: https://docs.fastf1.dev/core.html#fastf1.core.Session.weather_data
    if session is None:
        return np.nan, np.nan, 0
    try:
        wd = session.weather_data
        if wd is None or wd.empty:
            return np.nan, np.nan, 0
        track_temp = float(wd["TrackTemp"].median()) if "TrackTemp" in wd.columns else np.nan
        air_temp   = float(wd["AirTemp"].median())   if "AirTemp"   in wd.columns else np.nan
        is_wet     = int(wd["Rainfall"].any())        if "Rainfall"  in wd.columns else 0
        return track_temp, air_temp, is_wet
    except Exception:
        return np.nan, np.nan, 0


# ── rolling form tracker ──────────────────────────────────────────────────────

class RollingStats:
    # tracks recent race history per driver and per team
    # CRITICAL: call .update() AFTER emitting the feature rows for a race
    # if you call it before, the current race results contaminate the pre-race
    # features and you get leakage - the model looks artificially good on
    # training data and then falls apart completely on unseen races
    # https://scikit-learn.org/stable/common_pitfalls.html#data-leakage

    def __init__(self, window=5):
        self.window = window
        self._driver      = defaultdict(list)  # {abbr: [(finish_pos, points, is_dnf, is_win, is_podium), ...]}
        self._team        = defaultdict(list)  # {team: [finish_pos, ...]}
        self._season_pts  = defaultdict(list)  # for season-level accumulators
        self._season_finsh = defaultdict(list)
        self._season_wins  = defaultdict(int)  # win count this season - resets at start of each year

    def get_driver_rolling(self, driver):
        # returns 3-race, 5-race, and 10-race rolling windows
        # 3-race: hot streaks and recent slumps, noisier but more current
        # 5-race: stable mid-term baseline - the main reference window
        # 10-race: longer trend that smooths out peaks and valleys in the data
        # win rate and podium rate capture dominant-car phases that avg finish alone misses
        # e.g. VER in 2023 had avg finish ~1.3 but win_rate_5 was often 1.0 - a much cleaner signal
        hist10 = self._driver[driver][-10:]           # last 10 races
        hist5  = self._driver[driver][-self.window:]  # last 5 races
        hist3  = self._driver[driver][-3:]            # last 3 races

        if not hist5:
            # new driver or start of season - NaN here, imputer fills it later
            return {
                "rolling_avg_finish_3":  np.nan,
                "rolling_avg_finish_5":  np.nan,
                "rolling_dnf_rate_5":    np.nan,
                "rolling_points_5":      np.nan,
                "rolling_win_rate_5":    np.nan,
                "driver_podium_rate_5":  np.nan,
                "rolling_avg_finish_10": np.nan,
                "rolling_points_10":     np.nan,
            }

        pos3  = [h[0] for h in hist3  if pd.notna(h[0])]
        pos5  = [h[0] for h in hist5  if pd.notna(h[0])]
        pos10 = [h[0] for h in hist10 if pd.notna(h[0])]
        dnfs  = [h[2] for h in hist5]
        pts5  = [h[1] for h in hist5  if pd.notna(h[1])]
        pts10 = [h[1] for h in hist10 if pd.notna(h[1])]
        wins5 = [h[3] for h in hist5]   # 1 if won that race, 0 otherwise
        pods5 = [h[4] for h in hist5]   # 1 if finished P1/P2/P3

        return {
            "rolling_avg_finish_3":  np.mean(pos3)  if pos3  else np.nan,
            "rolling_avg_finish_5":  np.mean(pos5)  if pos5  else np.nan,
            "rolling_dnf_rate_5":    np.mean(dnfs)  if dnfs  else np.nan,
            "rolling_points_5":      np.sum(pts5)   if pts5  else np.nan,  # sum not mean for points
            "rolling_win_rate_5":    np.mean(wins5) if wins5 else np.nan,  # fraction of last 5 that were wins
            "driver_podium_rate_5":  np.mean(pods5) if pods5 else np.nan,  # fraction of last 5 on podium
            "rolling_avg_finish_10": np.mean(pos10) if pos10 else np.nan,
            "rolling_points_10":     np.sum(pts10)  if pts10 else np.nan,
        }

    def get_team_rolling(self, team):
        # team's average finishing position over recent races
        # using a slightly wider window per team (5 races x 2 drivers = ~10 data points)
        # this captures whether the underlying car is fast right now, independent of driver
        hist      = self._team[team][-(self.window * 2):]
        positions = [p for p in hist if pd.notna(p)]
        return {"team_rolling_avg_5": np.mean(positions) if positions else np.nan}

    def get_season_stats(self, driver):
        # cumulative season stats up to but not including this race
        # season_pts_pct: normalised so 1.0 means they won every race this season
        # season_avg_finish: mean finishing position so far (lower is better)
        # season_wins_pct: wins / races_so_far - captures whether a driver is in a dominant streak
        pts   = self._season_pts[driver]
        finsh = self._season_finsh[driver]
        wins  = self._season_wins[driver]
        n     = len(finsh)
        if n == 0:
            return {"season_pts_pct": np.nan, "season_avg_finish": np.nan, "season_wins_pct": np.nan}
        pts_pct  = sum(pts) / (n * 25) if n > 0 else np.nan  # 25 pts is the max per race
        wins_pct = wins / n if n > 0 else 0.0
        return {
            "season_pts_pct":    round(pts_pct, 4),
            "season_avg_finish": round(np.mean(finsh), 2) if finsh else np.nan,
            "season_wins_pct":   round(wins_pct, 4),
        }

    def update(self, race_results, new_season=False):
        # update histories after a race - this must always happen AFTER emitting rows
        if new_season:
            self._season_pts.clear()    # reset the season accumulators at the start of a new year
            self._season_finsh.clear()
            self._season_wins.clear()   # win count resets with the season

        for abbr, info in race_results.items():
            fp   = info["finish_pos"]
            pts  = info["points"]
            dnf  = is_dnf(info["status"])
            team = info["team"]
            # track wins and podiums in the tuple so the rolling stat methods can read them back
            is_win    = int(fp == 1) if pd.notna(fp) else 0
            is_podium = int(fp <= 3) if pd.notna(fp) else 0

            self._driver[abbr].append((fp, pts, dnf, is_win, is_podium))
            if team and pd.notna(fp):
                self._team[team].append(fp)

            if pd.notna(pts):
                self._season_pts[abbr].append(float(pts))
            if pd.notna(fp):
                self._season_finsh[abbr].append(float(fp))
            if is_win:
                self._season_wins[abbr] += 1


# ── circuit history tracker ───────────────────────────────────────────────────

class CircuitHistory:
    # per-driver and per-team historical average finish at each specific circuit
    # also tracks how many times they've raced there (circuit experience)
    #
    # first-timers at a circuit consistently underperform their qualifying position
    # unfamiliar braking points, different kerb profiles, track-specific setup knowledge
    # all accumulate over years and you can see it in the data pretty clearly
    # ref: https://en.wikipedia.org/wiki/Circuit_knowledge (general concept)
    # ref: similar observation in https://github.com/mar-antaya/F1-Race-Predictions
    #
    # warmup years (2019-2021) populate this correctly so by 2022 we have real history
    # particularly useful for tracks that were new in the 2022-2024 window
    # (Miami, Las Vegas, Qatar Sprint format all introduced during training range)

    def __init__(self):
        self._driver = defaultdict(list)  # {(abbr, circuit): [finish_positions]}
        self._team   = defaultdict(list)  # {(team, circuit): [finish_positions]}
        self._gains  = defaultdict(list)  # {(abbr, circuit): [grid_pos - finish_pos]; positive = gained places

    def get_driver_avg(self, driver, circuit):
        h = self._driver[(driver, circuit)]
        return np.mean(h) if h else np.nan  # NaN if first time here, imputer handles it

    def get_team_avg(self, team, circuit):
        h = self._team[(team, circuit)]
        return np.mean(h) if h else np.nan

    def get_driver_count(self, driver, circuit):
        # raw count of prior completed races at this circuit
        return len(self._driver[(driver, circuit)])

    def get_driver_circuit_gains(self, driver, circuit):
        # average positions gained (grid_pos - finish_pos) at this specific circuit
        # positive = driver typically moves forward in races at this track
        # useful for circuits where track knowledge enables overtaking (e.g. Hamilton at Silverstone)
        h = self._gains[(driver, circuit)]
        return round(np.mean(h), 3) if h else np.nan

    def update(self, race_results, circuit):
        for abbr, info in race_results.items():
            fp   = info["finish_pos"]
            team = info["team"]
            grid = info.get("grid_pos", np.nan)
            if pd.notna(fp):
                self._driver[(abbr, circuit)].append(fp)
                if team:
                    self._team[(team, circuit)].append(fp)
            # track grid-to-finish gains; grid must be valid (>0) to avoid pit lane start contamination
            if pd.notna(fp) and pd.notna(grid) and grid > 0:
                self._gains[(abbr, circuit)].append(float(grid) - float(fp))


# ── main feature extraction loop ─────────────────────────────────────────────

def build_features(years, include_race_target=True):
    # build feature rows for all races in `years`
    # warmup years (2019 through the start of the target window) are processed
    # first to populate rolling stats and circuit history without emitting rows
    #
    # per race the process is:
    #   1. load FP sessions -> pace deltas, sector splits, long runs, consistency
    #   2. load qualifying  -> grid position, quali gaps, q3 appearance, weather
    #   3. read the pre-computed rolling stats (state before this race)
    #   4. emit one row per driver into all_rows
    #   5. THEN update rolling stats and circuit history with race results
    #
    # step 5 must come after step 4 - this is the critical anti-leakage ordering
    # ref: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split

    rolling      = RollingStats(window=ROLLING_N)
    circuit_hist = CircuitHistory()
    all_rows     = []

    min_year     = min(years)
    warmup_start = 2019  # FastF1 coverage is reliable from 2018, 2019 is a bit safer
    warmup_years = list(range(warmup_start, min_year))
    all_years    = sorted(set(warmup_years + list(years)))
    prev_year    = None

    for year in all_years:
        print(f"\n{'='*60}")
        print(f"  Year: {year}  {'[WARMUP - not emitting rows]' if year not in years else ''}")
        print(f"{'='*60}")

        is_new_season = (year != prev_year)
        prev_year     = year

        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"  [!] Cannot load {year} schedule: {e}")
            continue

        for idx, (_, event) in enumerate(schedule.iterrows()):
            round_num      = int(event["RoundNumber"])
            event_name     = str(event["EventName"])
            circuit        = str(event.get("Location", event_name))
            is_sprint_wknd = str(event.get("EventFormat", "")).lower() == "sprint"

            print(f"\n  R{round_num:02d} {event_name}"
                  f"{'  [SPRINT]' if is_sprint_wknd else ''}")

            # load FP sessions - weather=True for FP1/FP2 so we get track and air temp
            # FP3 doesn't need weather data since we only use pace from it
            fp1 = load_session(year, round_num, "FP1", weather=True)
            fp2 = load_session(year, round_num, "FP2", weather=True)
            fp3 = load_session(year, round_num, "FP3")

            # sprint weekends replace FP3 with sprint qualifying (SQ) and the sprint (S)
            sprint_quali = None
            sprint_race  = None
            if is_sprint_wknd:
                sprint_quali = load_session(year, round_num, "SQ")
                sprint_race  = load_session(year, round_num, "S")

            # qualifying with weather=True for track/air temps
            quali = load_session(year, round_num, "Q", laps=True, weather=True)

            # compute FP features
            fp1_deltas = best_lap_delta(fp1)
            fp2_deltas = best_lap_delta(fp2)
            fp3_deltas = best_lap_delta(fp3)

            sprint_quali_pos = get_session_positions(sprint_quali)
            sprint_pos_dict  = get_session_positions(sprint_race)
            grid_pos_dict    = get_session_positions(quali)

            fp2_sectors = sector_pace_deltas_fp2(fp2)
            fp2_lr      = long_run_pace_delta(fp2, min_stint_laps=5)
            fp2_consist = lap_consistency(fp2)
            fp3_lr      = long_run_pace_delta_fp3(fp3, min_stint_laps=4)

            fp1_track_temp, _, fp1_wet             = get_session_weather(fp1)
            fp2_track_temp, _, fp2_wet             = get_session_weather(fp2)
            quali_track_temp, quali_air_temp, q_wet = get_session_weather(quali)

            quali_gaps  = quali_gap_to_pole_and_teammate(quali)
            q3_appeared = quali_q3_appearances(quali)  # which drivers made it to Q3?

            # load race results if we need target columns
            race_results = {}
            if include_race_target:
                race = load_session(year, round_num, "R", laps=False)
                if race is not None:
                    race_results = get_race_results_dict(race)

            # figure out the driver list for this weekend
            # qualifying results are the most complete; fall back to race results
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
                    rolling.update(race_results, new_season=(is_new_season and idx == 0))
                    circuit_hist.update(race_results, circuit)
                continue

            drivers_df = drivers_df.dropna(subset=["Abbreviation"])

            # warmup years: just update state, don't emit feature rows
            if year not in years:
                if race_results:
                    rolling.update(race_results, new_season=(is_new_season and idx == 0))
                    circuit_hist.update(race_results, circuit)
                continue

            # emit one row per driver for this race weekend
            for _, drow in drivers_df.iterrows():
                abbr = str(drow["Abbreviation"])
                team = str(drow.get("TeamName", ""))

                # get rolling stats BEFORE this race (from prior races only)
                drv_roll      = rolling.get_driver_rolling(abbr)
                team_roll     = rolling.get_team_rolling(team)
                season        = rolling.get_season_stats(abbr)

                # circuit history BEFORE this race
                circ_drv  = circuit_hist.get_driver_avg(abbr, circuit)
                circ_team = circuit_hist.get_team_avg(team, circuit)
                circ_exp  = circuit_hist.get_driver_count(abbr, circuit)

                fp2_sects  = fp2_sectors.get(abbr, {})
                fp1_d      = fp1_deltas.get(abbr, np.nan)
                fp2_d      = fp2_deltas.get(abbr, np.nan)
                q_gaps     = quali_gaps.get(abbr, {})

                # pace progression: positive means they got faster vs the field from FP1 to FP2
                # useful because teams run different programmes across sessions and this shows
                # whether they found pace as the weekend went on
                progression = (fp1_d - fp2_d) if (pd.notna(fp1_d) and pd.notna(fp2_d)) else np.nan

                # derived grid position features
                grid_pos     = grid_pos_dict.get(abbr, np.nan)
                n_drivers    = max(len(drivers_df), 1)
                is_front_row = int(grid_pos <= 2) if pd.notna(grid_pos) else 0
                grid_pos_pct = (grid_pos / n_drivers) if pd.notna(grid_pos) else np.nan  # normalised to [0,1]

                # how is this driver doing vs their own team's baseline?
                # negative means outperforming the car, positive means underperforming
                # useful for isolating driver skill from car quality in the rolling stats
                drv_avg5    = drv_roll.get("rolling_avg_finish_5", np.nan)
                team_avg5   = team_roll.get("team_rolling_avg_5", np.nan)
                drv_vs_team = (drv_avg5 - team_avg5) if (pd.notna(drv_avg5) and pd.notna(team_avg5)) else np.nan

                # circuit-specific position gains history - before this race
                circ_gains = circuit_hist.get_driver_circuit_gains(abbr, circuit)

                row = {
                    # identifiers - not used as ML features
                    "Year":      year,
                    "Round":     round_num,
                    "EventName": event_name,
                    "Circuit":   circuit,
                    "Driver":    abbr,
                    "Team":      team,

                    # sprint / qualifying grid
                    "IsSprintWeekend":   int(is_sprint_wknd),
                    "grid_position":     grid_pos,
                    "sprint_quali_pos":  sprint_quali_pos.get(abbr, np.nan),
                    "sprint_pos":        sprint_pos_dict.get(abbr, np.nan),
                    "is_front_row":      is_front_row,    # 1 if P1 or P2; front row starters win ~70% of races
                    "grid_position_pct": grid_pos_pct,    # grid_pos / field_size, normalised to [0,1]

                    # FP best-lap deltas (seconds behind session fastest)
                    "fp1_pace_delta": fp1_d,
                    "fp2_pace_delta": fp2_d,
                    "fp3_pace_delta": fp3_deltas.get(abbr, np.nan),

                    # FP2 detailed breakdown - sector splits and long run
                    "fp2_s1_delta":       fp2_sects.get("s1", np.nan),
                    "fp2_s2_delta":       fp2_sects.get("s2", np.nan),
                    "fp2_s3_delta":       fp2_sects.get("s3", np.nan),
                    "fp2_long_run_delta": fp2_lr.get(abbr, np.nan),
                    "fp2_consistency":    fp2_consist.get(abbr, np.nan),
                    "fp_pace_progression": progression,

                    # FP3 long-run (Saturday, fresher signal than FP2)
                    "fp3_long_run_delta": fp3_lr.get(abbr, np.nan),

                    # qualifying gaps in seconds (not positions - magnitude matters)
                    "quali_gap_to_pole":  q_gaps.get("gap_to_pole",  np.nan),
                    "quali_teammate_gap": q_gaps.get("teammate_gap", np.nan),

                    # did the driver reach Q3? grid position doesn't tell you whether
                    # a P10 was comfortably in Q3 or barely sneaking in from Q2 borderline
                    # helps the model at the tricky P10/P11 boundary cases
                    "q3_appearance": q3_appeared.get(abbr, 0),
                    "quali_gap_pct": q_gaps.get("gap_pct", np.nan),  # fractional gap vs pole, scale-invariant

                    # session weather (pre-race only - no race-day weather, that's leakage)
                    "fp1_track_temp": fp1_track_temp,
                    "fp1_wet":        fp1_wet,
                    "fp2_track_temp": fp2_track_temp,
                    "fp2_wet":        fp2_wet,
                    "quali_track_temp": quali_track_temp,
                    "quali_air_temp":   quali_air_temp,
                    "quali_wet":        q_wet,

                    # rolling form and season stats (all computed from prior races only)
                    **drv_roll,    # rolling_avg_finish_3/5/10, dnf_rate_5, points_5/10, win_rate_5, podium_rate_5
                    **team_roll,   # team_rolling_avg_5
                    **season,      # season_pts_pct, season_avg_finish, season_wins_pct

                    # driver vs team: negative means outperforming the car baseline
                    "driver_vs_team_avg_delta": drv_vs_team,

                    # circuit history (from prior races at this specific track only)
                    "circuit_driver_avg_finish":        circ_drv,
                    "circuit_team_avg_finish":          circ_team,
                    "circuit_experience":               circ_exp,
                    "driver_circuit_positions_gained":  circ_gains,  # avg positions gained at this circuit
                }

                # add race result targets if we want them
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

            # update rolling and circuit state AFTER emitting - this ordering is important
            if race_results:
                rolling.update(race_results, new_season=(is_new_season and idx == 0))
                circuit_hist.update(race_results, circuit)
                is_new_season = False  # only trigger the season reset once per year

            print(f"    → {len(drivers_df)} drivers processed")

    return pd.DataFrame(all_rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  F1 Feature Engineering")
    print(f"  Training:   {TRAIN_YEARS}")
    print(f"  Prediction: {PREDICT_YEAR}")
    print(f"  Cache:      {CACHE_DIR}/")
    print(f"  Output:     {OUTPUT_DIR}/")
    print("=" * 60)

    # build training features (2022-2024)
    print("\n[1/2] Building training features (2022-2024)...")
    train_df = build_features(TRAIN_YEARS, include_race_target=True)

    if train_df.empty:
        print("[!] No training data - check FastF1 cache.")
        return

    train_df   = train_df.dropna(subset=["finish_position"])
    train_path = os.path.join(OUTPUT_DIR, "ml_training_features.csv")
    train_df.to_csv(train_path, index=False)
    print(f"\n  → Saved {train_path}")
    print(f"    {len(train_df)} rows | {train_df['Driver'].nunique()} drivers | "
          f"{train_df['EventName'].nunique()} circuits")

    # build 2025 prediction features
    print("\n[2/2] Building 2025 prediction features...")
    pred_df   = build_features([PREDICT_YEAR], include_race_target=True)
    pred_path = os.path.join(OUTPUT_DIR, "ml_2025_features.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  → Saved {pred_path}  ({len(pred_df)} rows)")

    # print feature summary
    print("\n" + "=" * 60)
    print("  Feature Summary")
    print("=" * 60)
    meta_cols = {"Year", "Round", "EventName", "Circuit", "Driver", "Team",
                 "finish_position", "is_winner", "is_top10"}
    feat_cols = [c for c in train_df.columns if c not in meta_cols]
    print(f"\n  {len(feat_cols)} features:")
    for col in feat_cols:
        pct_na = train_df[col].isna().mean() * 100
        print(f"    {col:<35} ({pct_na:5.1f}% NaN)")

    print(f"\n  Targets:")
    print(f"    Winners  (is_winner=1): {int(train_df['is_winner'].sum())}")
    print(f"    Top-10   (is_top10=1):  {int(train_df['is_top10'].sum())}")
    print(f"    Total rows:             {len(train_df)}")


if __name__ == "__main__":
    main()
