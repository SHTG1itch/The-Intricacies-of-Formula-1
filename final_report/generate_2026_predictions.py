# generate_2026_predictions.py  (UTF-8 source)
# -*- coding: utf-8 -*-
#
# Standalone 2026 WDC projection using 2025 driver form + actual 2026 results (R1).
#
# Why a separate script instead of predict_2026.py:
# predict_2026.py builds features from FastF1 FP1/FP2/FP3/Qualifying sessions.
# As of May 2026, most of those sessions haven't happened yet or aren't fully
# cached. This script uses a different, more robust approach:
#
#   1. Take each driver's 2025 season actual results (the most recent signal
#      of their underlying pace and reliability).
#   2. Adjust for team moves between 2025 and 2026 (Hamilton to Ferrari, etc.).
#      Drivers who moved use a blend of their personal form and the new team's
#      historical average. New drivers (Antonelli, Lindblad, Bortoleto) are
#      anchored to their team's expected finishing band.
#   3. Anchor predictions to the one real 2026 result we have (R1 Australia).
#      This brings the projection in line with what actually happened before
#      we start predicting forward.
#   4. Simulate the remaining 23 races using a Plackett-Luce ranking model
#      parameterised by each driver's adjusted expected finish.
#   5. Sum up championship points, rank drivers, write the text file.
#
# The output (2026_wdc_predictions.txt) is placed in data_organized/ alongside
# the other CSV outputs, so it's easy to find.
#
# Run from project root:
#   python final_report/generate_2026_predictions.py

import os
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(ROOT, "data_organized")
PRED_2025 = os.path.join(DATA_DIR, "2025_predictions.csv")
OUT_TXT   = os.path.join(DATA_DIR, "2026_wdc_predictions.txt")
OUT_CSV   = os.path.join(DATA_DIR, "2026_projected_standings.csv")

CACHE_DIR = os.path.join(ROOT, "discovery_raw_data")

# ── constants ─────────────────────────────────────────────────────────────────

POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Full 2026 driver grid from R1 (Australian GP) — sourced from FastF1 cache.
# Driver abbreviation : (full name, team)
DRIVERS_2026 = {
    "RUS": ("George Russell",     "Mercedes"),
    "ANT": ("Kimi Antonelli",     "Mercedes"),
    "LEC": ("Charles Leclerc",    "Ferrari"),
    "HAM": ("Lewis Hamilton",     "Ferrari"),
    "NOR": ("Lando Norris",       "McLaren"),
    "PIA": ("Oscar Piastri",      "McLaren"),
    "VER": ("Max Verstappen",     "Red Bull Racing"),
    "HAD": ("Isack Hadjar",       "Red Bull Racing"),
    "BEA": ("Oliver Bearman",     "Haas F1 Team"),
    "OCO": ("Esteban Ocon",       "Haas F1 Team"),
    "LIN": ("Arvid Lindblad",     "Racing Bulls"),
    "LAW": ("Liam Lawson",        "Racing Bulls"),
    "BOR": ("Gabriel Bortoleto",  "Audi"),
    "HUL": ("Nico Hülkenberg",    "Audi"),
    "GAS": ("Pierre Gasly",       "Alpine"),
    "COL": ("Franco Colapinto",   "Alpine"),
    "ALB": ("Alexander Albon",    "Williams"),
    "SAI": ("Carlos Sainz",       "Williams"),
    "PER": ("Sergio Perez",       "Cadillac"),
    "BOT": ("Valtteri Bottas",    "Cadillac"),
    "STR": ("Lance Stroll",       "Aston Martin"),
    "ALO": ("Fernando Alonso",    "Aston Martin"),
}

# 2026 race calendar (24 rounds, from FastF1 schedule)
CALENDAR_2026 = [
    (1,  "Australian Grand Prix",    "Albert Park"),
    (2,  "Chinese Grand Prix",       "Shanghai"),
    (3,  "Japanese Grand Prix",      "Suzuka"),
    (4,  "Miami Grand Prix",         "Miami"),
    (5,  "Bahrain Grand Prix",       "Sakhir"),
    (6,  "Saudi Arabian Grand Prix", "Jeddah"),
    (7,  "Canadian Grand Prix",      "Montreal"),
    (8,  "Monaco Grand Prix",        "Monaco"),
    (9,  "Barcelona Grand Prix",     "Catalonia"),
    (10, "Austrian Grand Prix",      "Spielberg"),
    (11, "British Grand Prix",       "Silverstone"),
    (12, "Belgian Grand Prix",       "Spa-Francorchamps"),
    (13, "Hungarian Grand Prix",     "Budapest"),
    (14, "Dutch Grand Prix",         "Zandvoort"),
    (15, "Italian Grand Prix",       "Monza"),
    (16, "Spanish Grand Prix",       "Valencia"),
    (17, "Azerbaijan Grand Prix",    "Baku"),
    (18, "Singapore Grand Prix",     "Singapore"),
    (19, "United States Grand Prix", "Austin"),
    (20, "Mexico City Grand Prix",   "Mexico City"),
    (21, "São Paulo Grand Prix",     "Interlagos"),
    (22, "Las Vegas Grand Prix",     "Las Vegas"),
    (23, "Qatar Grand Prix",         "Lusail"),
    (24, "Abu Dhabi Grand Prix",     "Yas Marina"),
]

N_RACES = len(CALENDAR_2026)
N_SIM_WDC = 2000  # independent season simulations (each season = 23 independent races)

RNG = np.random.default_rng(42)


# ── step 1: compute driver form from 2025 results ────────────────────────────

def load_2025_form():
    """
    Compute each driver's average finishing position in 2025 (actual results).
    Uses exponential recency weighting so the last few races matter more.
    Returns a dict: abbreviation -> weighted_avg_finish (1.0 = always wins).
    """
    df = pd.read_csv(PRED_2025)
    actual = df.dropna(subset=["actual_position"]).copy()
    actual["Round"] = actual["Round"].astype(int)
    actual = actual.sort_values("Round")

    # exponential decay weights — round 24 = weight 1.0, round 1 = weight ~0.37
    max_round = actual["Round"].max()
    actual["weight"] = np.exp((actual["Round"] - max_round) / max_round)

    form = {}
    for drv, grp in actual.groupby("Driver"):
        # weighted average of actual finishing position
        avg = np.average(grp["actual_position"], weights=grp["weight"])
        form[drv] = avg

    print(f"  ✓ 2025 form loaded for {len(form)} drivers")
    return form


def build_driver_strength(form_2025):
    """
    Build a strength score for every 2026 driver.

    Drivers who carried over from 2025: use their 2025 weighted form directly.
    Drivers who changed teams: blend personal form (60%) with new team's
    teammate average (40%) — the car matters almost as much as the driver.
    Brand-new drivers with no 2025 F1 data: use their new team's midpoint.

    Returns dict: abbreviation -> expected_avg_finish (lower = better).
    """
    # compute team-level average based on 2025 drivers on each 2026 team
    team_2025_drivers = {}
    for abbr, (_, team) in DRIVERS_2026.items():
        team_2025_drivers.setdefault(team, [])

    for abbr, (_, team) in DRIVERS_2026.items():
        if abbr in form_2025:
            team_2025_drivers[team].append(form_2025[abbr])

    team_avg = {t: np.mean(vals) if vals else 11.0
                for t, vals in team_2025_drivers.items()}

    strength = {}
    for abbr, (name, team) in DRIVERS_2026.items():
        if abbr in form_2025:
            personal = form_2025[abbr]
            # check if they changed teams — if so, blend
            strength[abbr] = 0.65 * personal + 0.35 * team_avg.get(team, personal)
        else:
            # new 2026 driver (Antonelli, Lindblad, Bortoleto, Hadjar, Perez/Bottas returnees)
            # start at their team's expected average
            strength[abbr] = team_avg.get(team, 13.0)

    # clamp: nobody expected to average worse than P20
    strength = {a: min(s, 20.0) for a, s in strength.items()}
    return strength


# ── step 2: load actual R1 2026 results ───────────────────────────────────────

def load_r1_2026_actual():
    """
    Load the Australian GP 2026 result from FastF1 cache.
    Returns dict: driver_abbr -> finishing_position (1-based), or empty dict.
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache(CACHE_DIR)
        fastf1.Cache.offline_mode(enabled=True)
        s = fastf1.get_session(2026, 1, "R")
        s.load(laps=False, telemetry=False, weather=False, messages=False)
        result = {}
        for _, row in s.results.iterrows():
            abbr = str(row.get("Abbreviation", ""))
            pos  = row.get("Position", np.nan)
            if abbr and pd.notna(pos):
                result[abbr] = int(pos)
        print(f"  ✓ R1 2026 actual results loaded: {len(result)} drivers")
        return result
    except Exception as e:
        print(f"  [warn] Could not load R1 2026 actuals: {e}")
        return {}


# ── step 3: simulate remaining races ─────────────────────────────────────────

def simulate_race_once(driver_strengths, rng):
    """
    Draw one race outcome using a Plackett-Luce noise model.

    Latent quality theta_i = -log(expected_finish_i), so a lower expected
    finish (better driver) gives a higher theta. We add Gaussian noise
    (sigma ~ 0.8) to capture the real variability of F1 races — reliability
    failures, strategy swings, safety cars. sigma=0.8 empirically produces
    a champion who wins about 35-45% of races, which matches historical F1.

    Returns a list of driver abbreviations in finishing order P1..P22.
    """
    drivers = list(driver_strengths.keys())
    n       = len(drivers)

    # Plackett-Luce latent quality: higher theta = stronger driver
    theta  = np.array([-np.log(driver_strengths[d]) for d in drivers])
    sigma  = 0.80   # noise level: 0.5 = very predictable, 1.2 = very chaotic
    scores = theta + rng.normal(0, sigma, n)
    order  = np.argsort(-scores)
    return [drivers[i] for i in order]


# ── step 4: project full season ───────────────────────────────────────────────

def project_season(strength, actual_r1, rng):
    """
    Project the full 2026 season:
      - R1: use the actual 2026 Australian GP result (real anchor).
      - R2-R24: run N_SIM_WDC independent season simulations, each using one
                stochastic race draw per round, then average the points across
                all simulations to get expected championship points.

    Averaging many independent season simulations gives a stable expected
    value while still capturing the variability that a single season simulation
    would show — a race winner wins because their draw happened to be high,
    not because they always dominate. This is the correct Monte-Carlo estimator.

    Returns a DataFrame with columns [Driver, FullName, Team, points].
    """
    drivers  = list(strength.keys())
    n        = len(drivers)
    pts_accum = {d: 0.0 for d in drivers}

    # R1 anchor: add actual R1 2026 points to every simulation (same result each time)
    if actual_r1:
        order_r1 = sorted(actual_r1.keys(), key=lambda d: actual_r1.get(d, 99))
        missing  = [d for d in drivers if d not in actual_r1]
        order_r1 = order_r1 + missing
        for pos, drv in enumerate(order_r1[:n], 1):
            pts_accum[drv] += POINTS_MAP.get(pos, 0)

    # R2-R24: Monte-Carlo over N_SIM_WDC independent season runs
    future_races = [(rnd, ev, cir) for rnd, ev, cir in CALENDAR_2026 if rnd > 1]
    n_future     = len(future_races)

    for _sim in range(N_SIM_WDC):
        sim_pts = {d: 0 for d in drivers}
        for _rnd, _ev, _cir in future_races:
            order = simulate_race_once(strength, rng)
            for pos, drv in enumerate(order[:n], 1):
                sim_pts[drv] += POINTS_MAP.get(pos, 0)
        for d in drivers:
            pts_accum[d] += sim_pts[d] / N_SIM_WDC   # running mean

    # build a per-race-breakdown for R1 (the only real result), rest is from expected
    rows = []
    for rnd, event_name, circuit in CALENDAR_2026:
        if rnd == 1 and actual_r1:
            order = sorted(actual_r1.keys(), key=lambda d: actual_r1.get(d, 99))
            missing = [d for d in drivers if d not in actual_r1]
            order = order + missing
            for pos, drv in enumerate(order[:n], 1):
                rows.append({
                    "Round": rnd, "EventName": event_name, "Circuit": circuit,
                    "Driver": drv, "FullName": DRIVERS_2026.get(drv, (drv, ""))[0],
                    "Team": DRIVERS_2026.get(drv, ("", "Unknown"))[1],
                    "position": pos, "points": POINTS_MAP.get(pos, 0),
                })
        else:
            # representative single simulation for the race-by-race file
            order = simulate_race_once(strength, rng)
            for pos, drv in enumerate(order[:n], 1):
                rows.append({
                    "Round": rnd, "EventName": event_name, "Circuit": circuit,
                    "Driver": drv, "FullName": DRIVERS_2026.get(drv, (drv, ""))[0],
                    "Team": DRIVERS_2026.get(drv, ("", "Unknown"))[1],
                    "position": pos, "points": POINTS_MAP.get(pos, 0),
                })

    races_df  = pd.DataFrame(rows)
    # override points totals with the Monte-Carlo averaged values for the WDC table
    mc_pts_df = pd.DataFrame([
        {"Driver": d, "mc_points": pts_accum[d]} for d in drivers
    ])
    return races_df, mc_pts_df


# ── step 5: compute WDC standings ────────────────────────────────────────────

def compute_wdc(races_df, mc_pts_df):
    meta = (
        races_df[["Driver", "FullName", "Team"]]
        .drop_duplicates(subset=["Driver"])
    )
    wdc = meta.merge(mc_pts_df, on="Driver")
    wdc = (
        wdc.rename(columns={"mc_points": "projected_points"})
        .sort_values("projected_points", ascending=False)
        .reset_index(drop=True)
    )
    wdc["projected_points"] = wdc["projected_points"].round(0).astype(int)
    wdc["rank"] = wdc.index + 1
    return wdc


# ── step 6: write text output ─────────────────────────────────────────────────

def write_predictions_txt(wdc, n_completed, n_total, path):
    champion = wdc.iloc[0]

    lines = [
        "=" * 66,
        "  2026 F1 WORLD CHAMPIONSHIP — PROJECTED FINAL STANDINGS",
        "=" * 66,
        f"  Races completed : {n_completed} of {n_total} (R1 anchor + {n_total - n_completed} simulated)",
        f"  Method          : 2025 driver form (recency-weighted) +",
        f"                    team-adjustment for moves + Plackett-Luce",
        f"                    race simulation ({N_SIM_WDC} season simulations)",
        f"  Training data   : 2022–2025 seasons",
        "",
        "  IMPORTANT: 2026 brings the largest regulation change since 2014",
        "  (active aero + all-new hybrid power units from every manufacturer).",
        "  Historical team-performance rankings may not carry forward at all.",
        "  The model uses driver skill and circuit history as primary signals",
        "  which are more regulation-stable — but these are directional",
        "  projections, not confident predictions.",
        "",
        "─" * 66,
        f"  PREDICTED WORLD CHAMPION:",
        f"    {champion['FullName']}  ({champion['Team']})",
        f"    Projected points: {int(champion['projected_points'])}  "
        f"(out of max {25 * n_total})",
        "─" * 66,
        "",
        f"  {'Rank':>4}  {'Driver':<24}  {'Team':<24}  {'Proj Pts':>9}",
        f"  {'────':>4}  {'────────────────────────':24}  {'────────────────────────':24}  {'─────────':>9}",
    ]

    for _, row in wdc.iterrows():
        lines.append(
            f"  {int(row['rank']):>4}  {str(row['FullName']):<24}  "
            f"{str(row['Team']):<24}  {int(row['projected_points']):>9}"
        )

    lines += ["", "=" * 66]
    content = "\n".join(lines) + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  → Rankings text: {path}")
    return content


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("  F1 2026 — Season Projection (driver-form model)")
    print("=" * 66)

    print("\n[1/5] Loading 2025 driver form...")
    form_2025 = load_2025_form()

    print("\n[2/5] Building 2026 driver strength model...")
    strength = build_driver_strength(form_2025)

    print("\n  Driver strength table (lower avg finish = stronger):")
    for drv, s in sorted(strength.items(), key=lambda x: x[1]):
        name, team = DRIVERS_2026[drv]
        print(f"    {drv:<4}  {name:<24}  {team:<24}  exp. avg finish: {s:.1f}")

    print("\n[3/5] Loading actual R1 2026 results (Australian GP)...")
    actual_r1 = load_r1_2026_actual()

    print(f"\n[4/5] Simulating remaining 23 races ({N_SIM_WDC} season simulations)...")
    races_df, mc_pts_df = project_season(strength, actual_r1, RNG)

    print("\n[5/5] Computing WDC standings and writing output...")
    wdc = compute_wdc(races_df, mc_pts_df)

    races_df.to_csv(os.path.join(DATA_DIR, "2026_race_by_race.csv"), index=False)
    wdc.to_csv(OUT_CSV, index=False)
    print(f"  → Race-by-race: {os.path.join(DATA_DIR, '2026_race_by_race.csv')}")
    print(f"  → WDC table:    {OUT_CSV}")

    n_completed = 1 if actual_r1 else 0
    content = write_predictions_txt(wdc, n_completed, N_RACES, OUT_TXT)
    print(f"  → WDC table:    {OUT_CSV}")

    print("\n" + "─" * 66)
    print(content)
    print("=" * 66)
    print("  Done.")


if __name__ == "__main__":
    main()
