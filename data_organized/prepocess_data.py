"""Build the four organized CSV files used by the report.

Run from the project root:
    python data_organized/preprocess_data.py

This reads the local FastF1 cache in discovery_raw_data/ and writes smaller,
analysis-ready CSVs to data_organized/. The filename is misspelled because it
was the original project file; preprocess_data.py is the clean wrapper used by
the rubric.
"""

import os
import warnings
import logging

import fastf1
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

# Where the raw cached data lives (must match cache_fastf1_data.py)
CACHE_DIR = "discovery_raw_data"

# Where the cleaned CSVs will be saved
OUTPUT_DIR = "data_organized"

# Seasons we are analyzing
YEARS = list(range(2018, 2026))  # 2018 through 2025

# Suppress noisy FastF1 internal logs so our output is readable
#   Ref: https://docs.fastf1.dev/fastf1.html#fastf1.Cache.enable_cache
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Tell FastF1 where the cached data lives so it reads from disk
#   Ref: https://docs.fastf1.dev/fastf1.html#fastf1.Cache.enable_cache
fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.Cache.offline_mode(enabled=True)


def get_schedule(year):
    """
    Get the race schedule for a given year.

    Returns a DataFrame with one row per event (Grand Prix).
    We exclude pre-season testing events.
    Returns None if the schedule can't be loaded (e.g. future/incomplete year).

    Ref: https://docs.fastf1.dev/events.html
         fastf1.get_event_schedule(year, include_testing=False)
    """
    try:
        return fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"  ⚠ Could not load schedule for {year}: {e}")
        return None


def load_session_safe(year, round_num, session_type):
    """
    Safely load a FastF1 session. Returns None if the session
    doesn't exist or fails to load (e.g. cancelled race, future event).

    Parameters:
        year         : int, the season year (e.g. 2023)
        round_num    : int, the round number in the schedule
        session_type : str, one of 'Q' (Qualifying) or 'R' (Race)
                       Full list: 'FP1','FP2','FP3','Q','S','SS','SQ','R'

    Ref: https://docs.fastf1.dev/events.html#session-identifiers
         https://docs.fastf1.dev/core.html#fastf1.core.Session.load
    """
    try:
        session = fastf1.get_session(year, round_num, session_type)
        # .load() downloads (or reads from cache) the session data.
        # We only need laps + results, not telemetry or weather here.
        #   Ref: https://docs.fastf1.dev/core.html#fastf1.core.Session.load
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        return session
    except BaseException as e:
        print(f"  [!] Skipped {year} R{round_num} {session_type} due to error: {e}")
        return None


def gap_to_winner_seconds(position, time_value):
    """
    Convert FastF1 race results into a gap-to-winner value.

    In session.results, the winner's Time is the total race time. Classified
    non-winners usually store Time as their gap behind the winner. Lapped
    finishers and DNFs often have no usable Time value.
    """
    if pd.notna(position) and float(position) == 1.0:
        return 0.0

    if pd.isna(time_value):
        return np.nan

    gap = pd.to_timedelta(time_value, errors="coerce")
    if pd.isna(gap):
        return np.nan

    return round(gap.total_seconds(), 3)


def save_csv_or_keep_existing(df, path):
    """Avoid replacing a useful CSV with an empty one when cache data is absent."""
    if df.empty and os.path.exists(path):
        print(f"  [!] No rows built; keeping existing {path}\n")
        return
    df.to_csv(path, index=False)
    print(f"  → Saved {path}  ({len(df)} rows)\n")


# ╔════════════════════════════════════════════════════════════════════╗
# ║  CSV 1: qualifying_deltas.csv  (for RQ1 - Driver Skill vs. Car)  ║
# ╚════════════════════════════════════════════════════════════════════╝

def build_qualifying_deltas():
    """
    For each qualifying session (2018–2025), compute the time gap
    between teammates on the same team.

    HOW IT WORKS:
    1. Load the Qualifying session for each race.
    2. From session.results, get each driver's best qualifying time.
       - session.results has columns Q1, Q2, Q3 as pd.Timedelta objects.
       - We take the best (minimum) of Q1, Q2, Q3 as the driver's time.
       Ref: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
    3. Group drivers by TeamName.
    4. For each team with exactly 2 drivers, compute the delta (seconds).

    WHY THIS MATTERS:
    Teammates drive the SAME car. Any consistent qualifying gap must
    come from the driver, not the car. This isolates driver skill.
    """
    print("Building qualifying_deltas.csv ...")
    rows = []

    for year in YEARS:
        schedule = get_schedule(year)
        if schedule is None:
            continue

        for _, event in schedule.iterrows():
            round_num = int(event["RoundNumber"])
            event_name = event["EventName"]

            # Load the Qualifying session (session identifier: 'Q')
            #   Ref: https://docs.fastf1.dev/events.html#session-identifiers
            quali = load_session_safe(year, round_num, "Q")
            if quali is None:
                continue

            # session.results is a SessionResults (subclass of DataFrame)
            # Columns include: Abbreviation, TeamName, Q1, Q2, Q3
            #   Ref: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
            results = quali.results

            # For each driver, find their best qualifying time.
            # Q1, Q2, Q3 are pd.Timedelta objects (or NaT if not set).
            # We convert to total seconds for easy math.
            results = results.copy()
            for q_col in ["Q1", "Q2", "Q3"]:
                results[q_col] = pd.to_timedelta(results[q_col])

            results["BestQ_s"] = (
                results[["Q1", "Q2", "Q3"]]
                .min(axis=1)              # best of Q1/Q2/Q3
                .dt.total_seconds()       # convert Timedelta → float seconds
            )

            # Group by team and compute teammate delta
            for team, group in results.groupby("TeamName"):
                valid = group.dropna(subset=["BestQ_s"])
                if len(valid) != 2:
                    continue  # skip if team didn't have exactly 2 drivers qualify

                # Sort so Driver1 is the faster one (lower time)
                sorted_drivers = valid.sort_values("BestQ_s")
                d1 = sorted_drivers.iloc[0]
                d2 = sorted_drivers.iloc[1]

                rows.append({
                    "Year":          year,
                    "Round":         round_num,
                    "EventName":     event_name,
                    "Team":          team,
                    "Driver1":       d1["Abbreviation"],   # faster driver
                    "Driver2":       d2["Abbreviation"],   # slower driver
                    "Driver1_Time_s": round(d1["BestQ_s"], 3),
                    "Driver2_Time_s": round(d2["BestQ_s"], 3),
                    "Delta_s":       round(d2["BestQ_s"] - d1["BestQ_s"], 3),
                })

            print(f"  ✓ {year} R{round_num:02d} {event_name}")

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "qualifying_deltas.csv")
    save_csv_or_keep_existing(df, path)
    return df


# ╔════════════════════════════════════════════════════════════════════╗
# ║  CSV 2: race_results.csv  (for RQ2 - Cost Cap Impact)            ║
# ╚════════════════════════════════════════════════════════════════════╝

def build_race_results():
    """
    For each race (2018–2025), pull every driver's finishing result
    and compute the gap to the race winner in seconds.

    FastF1 reports the winner's Time as total race duration, but classified
    non-winners usually get a Time value that is already their gap behind the
    winner. Lapped finishers and DNFs are left as NaN.

    WHY THIS MATTERS:
    By tracking the time gap to the winner at each finishing position
    across years, we can see if the cost cap has compressed the field
    (i.e., are P5/P10 finishing closer to P1 in post-2021 years?).
    """
    print("Building race_results.csv ...")
    rows = []

    for year in YEARS:
        schedule = get_schedule(year)
        if schedule is None:
            continue

        for _, event in schedule.iterrows():
            round_num = int(event["RoundNumber"])
            event_name = event["EventName"]

            # Load the Race session (session identifier: 'R')
            race = load_session_safe(year, round_num, "R")
            if race is None:
                continue

            # session.results columns used:
            #   Abbreviation, TeamName, GridPosition, Position, Points,
            #   Status, Time
            #   Ref: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
            results = race.results.copy()

            for _, row in results.iterrows():
                rows.append({
                    "Year":          year,
                    "Round":         round_num,
                    "EventName":     event_name,
                    "Driver":        row["Abbreviation"],
                    "Team":          row["TeamName"],
                    "GridPosition":  row["GridPosition"],
                    "Position":      row["Position"],
                    "Points":        row["Points"],
                    "Status":        row["Status"],
                    "GapToWinner_s": gap_to_winner_seconds(row["Position"], row["Time"]),
                })

            print(f"  ✓ {year} R{round_num:02d} {event_name}")

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "race_results.csv")
    save_csv_or_keep_existing(df, path)
    return df


# ╔════════════════════════════════════════════════════════════════════╗
# ║  CSV 3: lap_time_spreads.csv  (for RQ2.1 - Cost Cap Function)    ║
# ╚════════════════════════════════════════════════════════════════════╝

def build_lap_time_spreads():
    """
    For each race (2019–2025), compute every driver's median lap time
    and how far behind the fastest driver's median they are.

    HOW IT WORKS:
    1. Load the Race session and access session.laps.
    2. session.laps is a Laps DataFrame with one row per lap per driver.
       Key columns: Driver, Team, LapTime (pd.Timedelta), LapNumber.
       Ref: https://docs.fastf1.dev/core.html#fastf1.core.Laps
    3. Filter out pit in/out laps (they have abnormally long times).
       We use the pick_quicklaps() method or just filter NaN LapTimes.
    4. Calculate the median lap time per driver (in seconds).
    5. The delta is: driver_median - fastest_driver_median.

    WHY THIS MATTERS:
    The FIA's cost cap goal was to make slower teams faster relative
    to the front-runners. If the "tail" (slowest drivers' median)
    is getting closer to the "peak" (fastest drivers' median) over
    the years, the cost cap is working as intended.
    """
    print("Building lap_time_spreads.csv ...")
    rows = []

    # Start from 2019 (to match the RQ2.1 ridge plot range)
    for year in YEARS:
        if year < 2019:
            continue

        schedule = get_schedule(year)
        if schedule is None:
            continue

        for _, event in schedule.iterrows():
            round_num = int(event["RoundNumber"])
            event_name = event["EventName"]

            race = load_session_safe(year, round_num, "R")
            if race is None:
                continue

            try:
                # session.laps: one row per lap per driver
                laps = race.laps.copy()
            except Exception as e:
                print(f"  [!] Skipping {year} R{round_num}: data not loaded ({e})")
                continue

            # Convert LapTime to seconds
            laps["LapTime_s"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()

            # Filter out invalid laps:
            #   - NaN lap times (incomplete laps)
            #   - Pit in/out laps (PitInTime or PitOutTime are not NaT)
            #   - Very slow laps (safety car, formation lap, etc.)
            #     We remove laps slower than 150% of the median as outliers
            valid_laps = laps.dropna(subset=["LapTime_s"]).copy()
            valid_laps = valid_laps[
                valid_laps["PitInTime"].isna() & valid_laps["PitOutTime"].isna()
            ]

            if valid_laps.empty:
                continue

            # Remove extreme outliers (safety car laps, etc.)
            overall_median = valid_laps["LapTime_s"].median()
            valid_laps = valid_laps[valid_laps["LapTime_s"] < overall_median * 1.5]

            # Calculate each driver's median lap time
            driver_medians = (
                valid_laps
                .groupby(["Driver", "Team"])["LapTime_s"]
                .median()
                .reset_index()
                .rename(columns={"LapTime_s": "MedianLapTime_s"})
            )

            if driver_medians.empty:
                continue

            # Fastest driver's median is the baseline (0 seconds delta)
            fastest_median = driver_medians["MedianLapTime_s"].min()

            for _, dm in driver_medians.iterrows():
                rows.append({
                    "Year":               year,
                    "Round":              round_num,
                    "EventName":          event_name,
                    "Driver":             dm["Driver"],
                    "Team":               dm["Team"],
                    "MedianLapTime_s":    round(dm["MedianLapTime_s"], 3),
                    "DeltaFromFastest_s": round(dm["MedianLapTime_s"] - fastest_median, 3),
                })

            print(f"  ✓ {year} R{round_num:02d} {event_name}")

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "lap_time_spreads.csv")
    save_csv_or_keep_existing(df, path)
    return df


# ╔════════════════════════════════════════════════════════════════════╗
# ║  CSV 4: driver_season_profiles.csv  (for RQ3 - WDC Prediction)   ║
# ╚════════════════════════════════════════════════════════════════════╝

def build_driver_season_profiles():
    """
    Aggregate season-level stats for each driver, for use as features
    in a WDC prediction model.

    HOW IT WORKS:
    1. Load every Race and Qualifying session for each year.
    2. From qualifying results, compute average qualifying position.
       Ref: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
    3. From race results, compute:
       - Average finishing position
       - Total points
       - Number of race wins  (Position == 1)
       - Number of DNFs       (Status != 'Finished' and not a '+X Lap' status)
       - Number of races started
    4. Group all of this per driver per season.

    WHY THIS MATTERS:
    To predict the WDC (World Drivers' Championship), we need to know
    each driver's overall performance profile for the season: pace,
    consistency, reliability, and points.
    """
    print("Building driver_season_profiles.csv ...")

    # We will collect per-race stats, then aggregate per season
    quali_rows = []   # one row per driver per qualifying
    race_rows = []    # one row per driver per race

    for year in YEARS:
        schedule = get_schedule(year)
        if schedule is None:
            continue

        for _, event in schedule.iterrows():
            round_num = int(event["RoundNumber"])

            # ── Qualifying data ──
            quali = load_session_safe(year, round_num, "Q")
            if quali is not None:
                try:
                    q_results = quali.results
                    for _, row in q_results.iterrows():
                        # Position in qualifying = their final qualifying position
                        if pd.notna(row["Position"]):
                            quali_rows.append({
                                "Year":      year,
                                "Driver":    row["Abbreviation"],
                                "Team":      row["TeamName"],
                                "QualiPos":  row["Position"],
                            })
                except Exception as e:
                    print(f"  [!] Skipping {year} R{round_num} Quali: data not loaded ({e})")

            # ── Race data ──
            race = load_session_safe(year, round_num, "R")
            if race is not None:
                try:
                    r_results = race.results
                    for _, row in r_results.iterrows():
                        # Check if the driver finished the race
                        status = str(row.get("Status", ""))
                        is_dnf = (
                            status != "Finished"
                            and not status.startswith("+")
                        )

                        race_rows.append({
                            "Year":      year,
                            "Driver":    row["Abbreviation"],
                            "Team":      row["TeamName"],
                            "FinishPos": row["Position"],
                            "Points":    row["Points"],
                            "IsWin":     1 if row["Position"] == 1 else 0,
                            "IsDNF":     1 if is_dnf else 0,
                        })
                except Exception as e:
                    print(f"  [!] Skipping {year} R{round_num} Race: data not loaded ({e})")

        print(f"  ✓ {year} complete")

    # ── Aggregate per driver per season ──

    df_quali = pd.DataFrame(quali_rows)
    df_race = pd.DataFrame(race_rows)

    path = os.path.join(OUTPUT_DIR, "driver_season_profiles.csv")
    if df_quali.empty and df_race.empty:
        save_csv_or_keep_existing(pd.DataFrame(), path)
        return pd.DataFrame()

    # Average qualifying position per driver per season
    if df_quali.empty:
        quali_agg = pd.DataFrame(columns=["Year", "Driver", "Team", "AvgQualiPos"])
    else:
        quali_agg = (
            df_quali
            .groupby(["Year", "Driver", "Team"])
            .agg(AvgQualiPos=("QualiPos", "mean"))
            .reset_index()
        )

    # Race stats per driver per season
    if df_race.empty:
        race_agg = pd.DataFrame(columns=[
            "Year", "Driver", "Team", "AvgFinishPos", "TotalPoints",
            "Wins", "DNFs", "RacesStarted",
        ])
    else:
        race_agg = (
            df_race
            .groupby(["Year", "Driver", "Team"])
            .agg(
                AvgFinishPos=("FinishPos", "mean"),
                TotalPoints=("Points", "sum"),
                Wins=("IsWin", "sum"),
                DNFs=("IsDNF", "sum"),
                RacesStarted=("FinishPos", "count"),
            )
            .reset_index()
        )

    # Merge qualifying and race aggregations
    df_profiles = pd.merge(
        race_agg, quali_agg,
        on=["Year", "Driver", "Team"],
        how="outer",  # keep drivers even if they missed qualifying or race
    )

    # Round for readability
    for col in ["AvgQualiPos", "AvgFinishPos", "TotalPoints"]:
        if col in df_profiles.columns:
            df_profiles[col] = df_profiles[col].round(2)

    # Sort by year and total points (descending) for easy reading
    df_profiles = df_profiles.sort_values(
        ["Year", "TotalPoints"], ascending=[True, False]
    )

    save_csv_or_keep_existing(df_profiles, path)
    return df_profiles


# ──────────────────────────────────────────────────────────────────────
# MAIN - run all four preprocessing steps
# ──────────────────────────────────────────────────────────────────────

def main():
    """
    Entry point. Runs all four data builders and saves CSVs.
    Run from the project root:  python data_organized/prepocess_data.py
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  FastF1 Data Preprocessing Pipeline")
    print("  Seasons: 2018–2025")
    print(f"  Cache:   {CACHE_DIR}/")
    print(f"  Output:  {OUTPUT_DIR}/")
    print("=" * 60 + "\n")

    # Step 1: Qualifying deltas (RQ1 - skill vs. car)
    build_qualifying_deltas()

    # Step 2: Race results with gap to winner (RQ2 - cost cap impact)
    build_race_results()

    # Step 3: Lap time spreads (RQ2.1 - cost cap effectiveness)
    build_lap_time_spreads()

    # Step 4: Driver season profiles (RQ3 - WDC prediction features)
    build_driver_season_profiles()

    print("=" * 60)
    print("  All done! CSV files saved to data_organized/")
    print("=" * 60)


if __name__ == "__main__":
    main()
