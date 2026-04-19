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


import sys
import os
import logging
import warnings

# Windows consoles default to cp1252 which can't print the box-drawing chars (→, ─, etc.)
# that are used throughout the progress output - reconfigure to utf-8 so they don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import defaultdict

import fastf1
import numpy as np
import pandas as pd
from scipy.stats import linregress

# config
CACHE_DIR    = "discovery_raw_data"
OUTPUT_DIR   = "data_organized"
TRAIN_YEARS  = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
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
    # per-driver best sector time vs the session best, for S1/S2/S3
    # the idea is that sectors tell you WHERE on track a car is fast or slow:
    #   S1 is usually the high-speed aero stuff (think Maggots-Becketts at Silverstone)
    #   S2 is medium-speed, more about mechanical grip and aero balance
    #   S3 is traction zones, slow chicanes, the heavy braking areas
    # so if a car is fast in S1 but slow in S3, you know something specific about its setup
    # the model can potentially use that to figure out which circuits suit which cars
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
    # median long-stint pace per driver vs the field median, measured in FP2
    # basically the best proxy for actual race pace you can get before Sunday
    # teams specifically run fuel-heavy race simulations in FP2 - same compounds,
    # similar fuel load to their race start plan - so it's reasonably representative
    # a driver who looks P6 in quali but whose long run is top-3 is almost always a threat
    # negative delta = faster than the field's median long-run pace (i.e. good)
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


def tyre_degradation_rate_fp2(session, min_stint_laps=5):
    # how quickly does each driver's car chew through its tyres in FP2?
    # the trick here is a linear regression on lap number vs lap time across long stints
    # if the slope is positive, the car is getting slower as the rubber wears off - that's deg
    # a steeper positive slope = worse degradation = they're going to struggle late in the race
    #
    # this is NOT the same thing as fp2_long_run_delta, which just measures average pace
    # this one tells you how FAST the car falls off a cliff, not just where it starts
    # think Ferrari 2023: strong in quali and the first stint, but the tyre cliff came early
    # by lap 15-20 the Mercs and Max were catching them even if they started well ahead
    #
    # we throw out laps more than 12% above the session median (those are SC laps or outlaps)
    # and we only count laps where the driver wasn't pitting in or out - so clean mid-stint laps only
    # then we average the slope across all long stints for each driver
    # small gotcha: the slope technically mixes degradation AND fuel burn (fuel is about -0.05 s/lap
    # at the start of a stint), but since fuel burn affects everyone equally the model learns to
    # cancel it out - the relative slope between drivers is what matters
    # ref: scipy linregress: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
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
        # 1.12x threshold: keeps normal traffic laps but drops SC/VSC-affected ones
        laps = laps[laps["LapTime_s"] < med * 1.12]

        driver_slopes = defaultdict(list)
        for (driver, stint), group in laps.groupby(["Driver", "Stint"]):
            if len(group) < min_stint_laps:
                continue
            group = group.sort_values("LapNumber")
            x = np.arange(len(group), dtype=float)  # lap number within stint (0, 1, 2, ...)
            y = group["LapTime_s"].values
            slope, _, _, _, _ = linregress(x, y)  # seconds per lap - positive means degrading
            driver_slopes[driver].append(slope)

        if not driver_slopes:
            return {}
        # average across stints: a driver who did 2 long runs gets both slopes averaged
        return {d: round(float(np.mean(slopes)), 4) for d, slopes in driver_slopes.items()}
    except Exception as e:
        print(f"    [warn] tyre_degradation_rate_fp2: {e}")
        return {}


def lap_consistency(session):
    # standard deviation of a driver's quicklap times - basically how much they're varying lap to lap
    # lower is better: a driver within 0.1s of their own pace every lap is in a much better rhythm
    # than someone bouncing around by 0.5s, even if their mean lap time is the same
    # this captures setup stability and driver confidence as much as raw speed
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
    # extracts two gap features from qualifying results:
    #   gap_to_pole: how many seconds behind pole were they? raw pace benchmark for the weekend
    #   teammate_gap: their best time minus their fastest teammate's best time
    #
    # the teammate gap thing is the clever one - comparing drivers in the same car cancels out
    # the car's contribution and leaves you with something closer to pure driver performance
    # the idea comes from Bell et al. 2016, and honestly it works pretty well
    # 0 means they were the faster of the two teammates in that car
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
    # returns (track_temp, air_temp, is_wet) from a session's weather log
    # we take the median across the whole session rather than a snapshot to smooth out weird spikes
    # FP2 on Friday afternoon is actually a decent proxy for Sunday race conditions -
    # same time of day, similar track temperatures as the rubber builds up
    # qualifying air temp matters because tyres behave differently when it's cold vs hot
    # is_wet is 1 if there was any rainfall at all during the session
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
    # keeps a running history of recent races for each driver and team
    # CRITICAL: call .update() AFTER you've emitted the feature rows for that race
    # if you call it beforehand, the current race result sneaks into the "pre-race" features
    # and you get beautiful training accuracy that falls completely apart on real data - classic leakage
    # (this bit us early on - the model looked amazing until we ran it forward in time)
    # https://scikit-learn.org/stable/common_pitfalls.html#data-leakage

    def __init__(self, window=5):
        self.window = window
        self._driver      = defaultdict(list)  # {abbr: [(finish_pos, points, is_dnf, is_win, is_podium), ...]}
        self._team        = defaultdict(list)  # {team: [finish_pos, ...]}
        self._season_pts  = defaultdict(list)  # for season-level accumulators
        self._season_finsh = defaultdict(list)
        self._season_wins  = defaultdict(int)  # win count this season - resets at start of each year

    def get_driver_rolling(self, driver):
        # three windows because different windows catch different things:
        #   3-race: reactive - catches hot streaks and sudden slumps, but noisy
        #   5-race: the main workhorse - stable enough to trust, recent enough to matter
        #   10-race: the long view - smooths out outlier races and shows the general trend
        # win rate and podium rate on top of average finish because avg finish misses dominant phases
        # think Verstappen in 2023 - avg finish was around 1.3 but win_rate_5 was basically 1.0 all year
        # that 1.0 is a way cleaner signal to the model that "this guy is just winning everything right now"
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
        # average finishing position for the team across both drivers over recent races
        # two windows because of the upgrade problem: a 5-race window is stable but slow to react,
        # a 3-race window is noisier but catches mid-season surges much faster
        # this distinction was basically designed for catching McLaren in 2024 - by race 10 or so
        # the 3-race avg had already noticed they were on a tear while the 5-race was still catching up
        hist5     = self._team[team][-(self.window * 2):]
        hist3     = self._team[team][-(3 * 2):]   # 3 races × 2 drivers = 6 entries
        pos5      = [p for p in hist5 if pd.notna(p)]
        pos3      = [p for p in hist3 if pd.notna(p)]
        return {
            "team_rolling_avg_5": np.mean(pos5) if pos5 else np.nan,
            "team_rolling_avg_3": np.mean(pos3) if pos3 else np.nan,
        }

    def get_season_stats(self, driver):
        # season totals up to but NOT including this race (no leakage)
        # season_pts_pct is normalised against 25 pts/race so it's comparable across different
        # points eras and different numbers of races completed - 1.0 means they won every race
        # season_wins_pct is wins / races_so_far, which tells you if someone's in a dominant streak
        # (vs. just accumulating points through consistent podiums without actually winning)
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


# ── circuit structural characteristics ───────────────────────────────────────

def get_circuit_characteristics(circuit):
    # hardcoded lookup for three things about each track that rolling stats can't tell you:
    #   overtaking_score: basically "how easy is it to pass here" - 0.05 = Monaco, 0.7 = Monza
    #   sc_rate: rough historical rate of safety car deployments at this venue (Singapore is ~90%)
    #   is_street: 1 if it's a temporary street circuit with walls everywhere, 0 for permanent tracks
    # these are the same for every driver at a race, they describe the track not the driver
    # ref: https://en.wikipedia.org/wiki/List_of_Formula_One_circuits

    c = str(circuit).lower()

    # Street circuits: no runoffs, walls everywhere, much higher SC probability
    is_street = int(any(s in c for s in [
        "monaco", "monte carlo", "baku", "marina bay", "jeddah", "miami", "las vegas"
    ]))

    # Overtaking score: lower means harder to pass, grid position dominates more
    OVERTAKING = {
        "monaco": 0.05, "monte carlo": 0.05,        # legendary difficulty - maybe 1 real move per race
        "budapest": 0.15,                             # Hungaroring: narrow, everyone on same line
        "marina bay": 0.20,                           # Singapore: street walls, very limited
        "zandvoort": 0.25,                            # designed narrow, banked but no DRS advantage
        "imola": 0.25,                                # old-school Italian circuit, narrow
        "mugello": 0.25,                              # beautiful but very limited passing
        "barcelona": 0.30,                            # tire deg matters, DRS doesn't help much
        "castellet": 0.30, "le castellet": 0.30,     # Paul Ricard: flat, limited corners
        "jeddah": 0.35,                               # fast street circuit but DRS helps
        "sochi": 0.35,                                # two slow corners only
        "istanbul": 0.40,                             # famous turn 8 but little overtaking
        "portim": 0.40,                               # Portimao: flowing but hard to follow
        "melbourne": 0.40,                            # Albert Park: medium post-2022 redesign
        "suzuka": 0.40,                               # chicane only; great for racing, hard to pass
        "nurburgring": 0.40, "n\ufffdrburgring": 0.40,
        "silverstone": 0.45,                          # Copse/Becketts: following easy, passing harder
        "yas island": 0.45,                           # Abu Dhabi: much improved after 2021 redesign
        "shanghai": 0.50,                             # hairpin T6 + back straight DRS
        "spa": 0.50,                                  # Kemmel straight: slipstream but only 1 spot
        "austin": 0.50,                               # COTA: T1 + T12 DRS zones
        "miami": 0.50,                                # medium (new in 2022, DRS zones but chaotic)
        "lusail": 0.50,                               # Qatar: multiple DRS but lacks braking spots
        "baku": 0.55,                                 # massive main straight, SC chaos adds passing
        "las vegas": 0.55,                            # long straights but tricky street layout
        "mexico": 0.55,                               # main straight + T1 hairpin
        "sakhir": 0.60, "bahrain": 0.60,             # designed for overtaking, 3 DRS zones
        "paulo": 0.60, "s\ufffdao paulo": 0.60,      # Interlagos: Senna S + back straight
        "spielberg": 0.65,                            # Red Bull Ring: short lap, T3 + T4
        "monza": 0.70,                                # longest DRS zones, slipstream festival
    }

    SC_RATE = {
        "marina bay": 0.90,                           # almost always has a SC in Singapore
        "baku": 0.80,                                 # most chaotic race on the calendar
        "monaco": 0.75, "monte carlo": 0.75,         # crashes and VSC nearly every year
        "jeddah": 0.70,                               # wall everywhere + fast = frequent SC
        "paulo": 0.70, "s\ufffdao paulo": 0.70,      # Interlagos is known for chaotic races
        "las vegas": 0.65,                            # new street-style, unpredictable
        "miami": 0.60,                                # new circuit, chaotic first few years
        "mugello": 0.60,                              # only raced once (2020) - famous pile-up
        "nurburgring": 0.55, "n\ufffdrburgring": 0.55, # changeable weather = high SC risk
        "melbourne": 0.55,                            # reconfigured in 2022, some chaos
        "monza": 0.50,                                # slipstream wars sometimes end in contact
        "suzuka": 0.50,                               # changeable weather sometimes; flag incidents
        "zandvoort": 0.50,                            # narrow + tyre deg = VSC deployments
        "spa": 0.45,                                  # weather; red flags sometimes
        "austin": 0.45,                               # medium, usually clean
        "istanbul": 0.45,
        "spielberg": 0.45,                            # DRS train sometimes causes pile-ups
        "imola": 0.45,                                # chaotic at T1 historically
        "silverstone": 0.40,                          # wing damage + T1 chaos, less frequent now
        "portim": 0.40,
        "mexico": 0.40,
        "shanghai": 0.45,
        "lusail": 0.40,
        "sochi": 0.35,                                # boring = clean
        "budapest": 0.35,                             # typically clean race
        "sakhir": 0.35, "bahrain": 0.35,             # designed for clean racing
        "castellet": 0.30, "le castellet": 0.30,     # very rare SC
        "barcelona": 0.30,
        "yas island": 0.30,                           # clean circuit, rare SC
    }

    overtaking = 0.45  # default: medium (for circuits not in the lookup)
    sc_rate    = 0.45  # default: medium

    for key, val in OVERTAKING.items():
        if key in c:
            overtaking = val
            break

    for key, val in SC_RATE.items():
        if key in c:
            sc_rate = val
            break

    return {
        "circuit_overtaking_score": overtaking,
        "circuit_sc_rate":          sc_rate,
        "circuit_is_street":        is_street,
    }


# ── circuit history tracker ───────────────────────────────────────────────────

class CircuitHistory:
    # remembers each driver's and team's historical average finishing position at each track
    # also tracks how many times they've raced there, because track experience is a real thing
    #
    # turns out first-timers at a circuit consistently underperform their qualifying position
    # the explanation makes sense: unfamiliar braking points, different kerb profiles, setup
    # knowledge that only comes from doing the race before - it all shows up in the finishing data
    # you can literally see rookies or circuit debutants lose a place or two vs their grid spot
    # ref: similar observation in https://github.com/mar-antaya/F1-Race-Predictions
    #
    # the warmup years (2019-2021) populate this before we start emitting feature rows for training
    # so by 2022 we have real historical averages rather than all NaN for every driver
    # this was especially important for Miami and Las Vegas which were brand new in our training window

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
        # how many places does this driver typically gain (or lose) here vs where they start?
        # positive = they tend to move forward during the race at this track
        # negative = they often fall back (maybe bad race pace or traffic issues on this layout)
        # Hamilton at Silverstone is a classic example of a positive outlier here
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
    # the main workhorse - churns through every race in `years` and emits one feature row per driver
    # warmup years run first to seed the rolling stats and circuit history, but don't emit any rows
    # (without warmup, the first race of 2022 would have completely empty history for everyone)
    #
    # the order for each race matters a lot - basically it's:
    #   1. load FP sessions -> pace deltas, sector splits, long runs, consistency
    #   2. load qualifying  -> grid position, quali gaps, q3 appearance, weather
    #   3. read the rolling stats as they stand RIGHT NOW (before this race)
    #   4. emit one row per driver into all_rows
    #   5. THEN update rolling stats and circuit history with the race results
    #
    # step 5 absolutely must come after step 4 - that ordering is what prevents leakage
    # if you flip it even once, the current race result bleeds into the pre-race features
    # and the model learns to "predict" something it already knew - garbage on new data
    # ref: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split

    rolling      = RollingStats(window=ROLLING_N)
    circuit_hist = CircuitHistory()
    all_rows     = []

    min_year     = min(years)
    # FastF1 coverage starts at 2018; always warm up from at least one year before
    # training to populate rolling stats and circuit history properly
    # if training starts at 2018 we try 2017 as warmup - it'll gracefully skip if not cached
    # removing the old min(2019,...) cap which broke when we added 2018 to TRAIN_YEARS
    warmup_start = max(min_year - 1, 2017)
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
            fp2_deg     = tyre_degradation_rate_fp2(fp2, min_stint_laps=5)
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

            # quick pre-pass to grab everyone's season points before emitting rows
            # we need this to figure out championship rank AND who the team leader is
            # doing it in a separate loop first is cleaner than computing it mid-row
            pre_season_pts = {}
            for _, _drow in drivers_df.iterrows():
                _abbr = str(_drow["Abbreviation"])
                _stats = rolling.get_season_stats(_abbr)
                pre_season_pts[_abbr] = _stats.get("season_pts_pct", 0.0) or 0.0

            # figure out who's the team's points leader - that's who gets priority on strategy calls
            # Verstappen over Perez, Hamilton over Russell - real effect, worth flagging to the model
            team_leader_abbr = {}
            for _, _drow in drivers_df.iterrows():
                _abbr = str(_drow["Abbreviation"])
                _team = str(_drow.get("TeamName", ""))
                if _team not in team_leader_abbr or pre_season_pts.get(_abbr, 0) > pre_season_pts.get(team_leader_abbr[_team], 0):
                    team_leader_abbr[_team] = _abbr

            # pre-sort for championship rank lookup
            sorted_pts_desc = sorted(pre_season_pts.values(), reverse=True)

            # which era are we in? 2022 introduced the ground-effect cars - completely different dynamics
            car_regulation_era = int(year >= 2022)

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
                circ_chars = get_circuit_characteristics(circuit)

                # pace progression: did they find pace across the weekend?
                # positive means FP2 delta was smaller than FP1 delta (got closer to the front)
                # teams run different programmes so this tells you whether the setup clicked on day 2
                progression = (fp1_d - fp2_d) if (pd.notna(fp1_d) and pd.notna(fp2_d)) else np.nan

                # derived grid position features
                grid_pos     = grid_pos_dict.get(abbr, np.nan)
                n_drivers    = max(len(drivers_df), 1)
                is_front_row = int(grid_pos <= 2) if pd.notna(grid_pos) else 0
                grid_pos_pct = (grid_pos / n_drivers) if pd.notna(grid_pos) else np.nan  # normalised to [0,1]

                # how is this driver doing relative to their own team's baseline pace?
                # negative means they're finishing ahead of the car's average (beating the car),
                # positive means they're underperforming it - basically a driver skill isolator
                drv_avg5    = drv_roll.get("rolling_avg_finish_5", np.nan)
                team_avg5   = team_roll.get("team_rolling_avg_5", np.nan)
                drv_vs_team = (drv_avg5 - team_avg5) if (pd.notna(drv_avg5) and pd.notna(team_avg5)) else np.nan

                # circuit-specific position gains history - before this race
                circ_gains = circuit_hist.get_driver_circuit_gains(abbr, circuit)

                # ── new features for accuracy push ───────────────────────────

                # where does this driver sit in the championship right now?
                # 0.0 = championship leader, 1.0 = dead last in the standings
                # turns out this is a really strong signal - the points leader is almost always in
                # the fastest car, so it's a decent proxy for car quality without explicitly
                # encoding car identity (which would change every year)
                pts_val = pre_season_pts.get(abbr, 0.0)
                if sorted_pts_desc and max(sorted_pts_desc) > 0:
                    rank_idx  = sorted_pts_desc.index(pts_val) if pts_val in sorted_pts_desc else len(sorted_pts_desc) - 1
                    champ_rank_pct = rank_idx / max(len(sorted_pts_desc) - 1, 1)
                else:
                    champ_rank_pct = np.nan  # season hasn't started yet, no rankings to speak of

                # is this the team's P1 driver right now (highest points scorer)?
                # the P1 driver gets the undercut call, the best strategy, the new parts first
                # this is a real and consistent pattern in the data (turns out it matters a lot)
                is_number_one_driver = int(team_leader_abbr.get(team, "") == abbr)

                # does this driver have a special relationship with this track?
                # compare their average finish here vs the team average here
                # negative = they tend to struggle at this track vs teammates, positive = specialist
                # think Hamilton at Silverstone, Verstappen at Zandvoort - genuinely shows up in the data
                circuit_specialist_delta = (
                    (circ_drv - circ_team)
                    if (pd.notna(circ_drv) and pd.notna(circ_team))
                    else np.nan
                )

                # team momentum: 3-race avg vs 5-race avg tells you if the car is on an upward trend
                # positive = car is getting faster (3-race avg better than 5-race baseline)
                # we basically added this to solve the McLaren 2024 problem - their upgrade surge mid-season
                # was invisible to the 5-race window until it was too late to matter
                team_avg3 = team_roll.get("team_rolling_avg_3", np.nan)
                team_momentum = (
                    (team_avg5 - team_avg3)  # positive = getting faster (lower pos number is better)
                    if (pd.notna(team_avg3) and pd.notna(team_avg5))
                    else np.nan
                )

                # grid position means very different things at different tracks
                # P1 at Monaco is basically a free win - almost no one overtakes there
                # P1 at Monza means very little - people are flying past on the Kemmel straight all race
                # (1 - overtaking_score) amplifies grid position at low-overtaking tracks,
                # so starting at the back of the grid at Monaco is doubly punished
                ot_score = circ_chars["circuit_overtaking_score"]
                grid_x_overtaking_inv = (
                    grid_pos_pct * (1.0 - ot_score)
                    if pd.notna(grid_pos_pct)
                    else np.nan
                )

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
                    "fp2_deg_rate":       fp2_deg.get(abbr, np.nan),
                    # fp2_deg_rate_obs is a binary indicator: 1 if we actually measured tyre deg, 0 if not
                    # this lets the model know WHEN to trust fp2_deg_rate vs when it was imputed/missing
                    # without this indicator the model has no way to know the 0.0 is a sentinel vs a real flat-deg car
                    "fp2_deg_rate_obs":   int(abbr in fp2_deg and pd.notna(fp2_deg.get(abbr))),
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

                    # static circuit structural features - same for every driver at this race
                    # these tell the model about track-level chaos/overtaking regardless of driver identity
                    "circuit_overtaking_score": circ_chars["circuit_overtaking_score"],  # 0=Monaco, 1=Monza
                    "circuit_sc_rate":          circ_chars["circuit_sc_rate"],           # historical SC fraction
                    "circuit_is_street":        circ_chars["circuit_is_street"],         # wall-bounded circuit flag

                    # ── accuracy-push features ────────────────────────────────
                    # car regulation era: 0=turbo-hybrid (2018-2021), 1=ground-effect (2022+)
                    # the two eras have fundamentally different car behaviours and overtaking patterns
                    "car_regulation_era":       car_regulation_era,
                    # grid_pos × (1 - overtaking): captures that front row at Monaco ≈ free win
                    "grid_x_overtaking_inv":    grid_x_overtaking_inv,
                    # championship leader is almost always in the fastest car - very strong pre-race signal
                    "championship_rank_pct":    champ_rank_pct,
                    # team's P1 driver gets priority strategy calls and best car setup updates
                    "is_number_one_driver":     is_number_one_driver,
                    # positive = specialist at this track, negative = tends to struggle here vs teammates
                    "circuit_specialist_delta": circuit_specialist_delta,
                    # positive = car getting faster (3-race avg better than 5-race avg)
                    "team_momentum":            team_momentum,
                    # team_rolling_avg_3 is included via **team_roll above
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

    # first pass: training data (2018-2024 with race targets for supervised learning)
    print(f"\n[1/2] Building training features {TRAIN_YEARS}...")
    train_df = build_features(TRAIN_YEARS, include_race_target=True)

    if train_df.empty:
        print("[!] No training data - check FastF1 cache.")
        return

    train_df   = train_df.dropna(subset=["finish_position"])  # drop rows with no race result (cancelled races, etc.)
    train_path = os.path.join(OUTPUT_DIR, "ml_training_features.csv")
    train_df.to_csv(train_path, index=False)
    print(f"\n  → Saved {train_path}")
    print(f"    {len(train_df)} rows | {train_df['Driver'].nunique()} drivers | "
          f"{train_df['EventName'].nunique()} circuits")

    # second pass: 2025 prediction features (include_race_target=True so actual results show up
    # for races already run, and NaN for races in the future)
    print("\n[2/2] Building 2025 prediction features...")
    pred_df   = build_features([PREDICT_YEAR], include_race_target=True)
    pred_path = os.path.join(OUTPUT_DIR, "ml_2025_features.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  → Saved {pred_path}  ({len(pred_df)} rows)")

    # quick audit of what we built - lists every feature and how much of it is missing
    # high NaN% features are usually things like fp3_long_run_delta (sprint weekends have no FP3)
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
