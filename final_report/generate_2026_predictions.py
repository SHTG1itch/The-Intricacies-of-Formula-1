# generate_2026_predictions.py  (UTF-8 source)
# -*- coding: utf-8 -*-
#
# 2026 WDC projection — updated with actual 2026 race results (R1-R3).
#
# How the strength model works (3-layer blend):
#
#   Layer 1 — 2025 driver form (25% weight)
#     Recency-weighted average finish from all 24 rounds of 2025.
#     Captures each driver's baseline skill and reliability.
#     Team-adjustment for drivers who changed constructors in 2026:
#     their personal form is blended with the new team's 2025 car level.
#
#   Layer 2 — 2026 actual results (65% weight)
#     Average actual finishing position from the 3 completed races.
#     Three results already tell us a great deal: Mercedes (ANT 2 wins,
#     RUS 1 win) is clearly dominant; McLaren (NOR/PIA) has underperformed
#     versus 2025 expectations; Ferrari (LEC/HAM) is consistently 3rd-4th.
#     65% weight because the regulation change makes 2025 form less reliable
#     than real 2026 data, even from just 3 races.
#
#   Layer 3 — championship points pace (10% weight)
#     A driver's current points / races converts to an implied avg finish.
#     Useful as a small tie-breaker for drivers with similar raw averages.
#
# After strength is computed, the remaining 21 races are simulated with
# Plackett-Luce noise (sigma=0.80) to capture F1's inherent unpredictability.
# 2000 independent season simulations are averaged to get stable expected
# championship totals; the actual completed race points are then added on top.
#
# Run from project root:
#   python -X utf8 final_report/generate_2026_predictions.py

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(ROOT, "data_organized")
PRED_2025 = os.path.join(DATA_DIR, "2025_predictions.csv")
OUT_TXT   = os.path.join(DATA_DIR, "2026_wdc_predictions.txt")
OUT_CSV   = os.path.join(DATA_DIR, "2026_projected_standings.csv")
CACHE_DIR = os.path.join(ROOT, "discovery_raw_data")

# ── constants ─────────────────────────────────────────────────────────────────

POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

DRIVERS_2026 = {
    "RUS": ("George Russell",    "Mercedes"),
    "ANT": ("Kimi Antonelli",    "Mercedes"),
    "LEC": ("Charles Leclerc",   "Ferrari"),
    "HAM": ("Lewis Hamilton",    "Ferrari"),
    "NOR": ("Lando Norris",      "McLaren"),
    "PIA": ("Oscar Piastri",     "McLaren"),
    "VER": ("Max Verstappen",    "Red Bull Racing"),
    "HAD": ("Isack Hadjar",      "Red Bull Racing"),
    "BEA": ("Oliver Bearman",    "Haas F1 Team"),
    "OCO": ("Esteban Ocon",      "Haas F1 Team"),
    "LIN": ("Arvid Lindblad",    "Racing Bulls"),
    "LAW": ("Liam Lawson",       "Racing Bulls"),
    "BOR": ("Gabriel Bortoleto", "Audi"),
    "HUL": ("Nico Hülkenberg",   "Audi"),
    "GAS": ("Pierre Gasly",      "Alpine"),
    "COL": ("Franco Colapinto",  "Alpine"),
    "ALB": ("Alexander Albon",   "Williams"),
    "SAI": ("Carlos Sainz",      "Williams"),
    "PER": ("Sergio Perez",      "Cadillac"),
    "BOT": ("Valtteri Bottas",   "Cadillac"),
    "STR": ("Lance Stroll",      "Aston Martin"),
    "ALO": ("Fernando Alonso",   "Aston Martin"),
}

# Full 2026 season calendar (24 rounds)
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
    (12, "Belgian Grand Prix",       "Spa"),
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

N_RACES      = len(CALENDAR_2026)
N_SIM_WDC    = 2000   # independent season Monte-Carlo runs
W_2025       = 0.25   # weight for 2025 form
W_2026       = 0.65   # weight for actual 2026 results
W_PACE       = 0.10   # weight for 2026 championship-pace implied finish
RNG          = np.random.default_rng(42)


# ── step 1: load 2025 form ────────────────────────────────────────────────────

def load_2025_form():
    df     = pd.read_csv(PRED_2025)
    actual = df.dropna(subset=["actual_position"]).copy()
    actual["Round"] = actual["Round"].astype(int)
    actual = actual.sort_values("Round")

    max_round = actual["Round"].max()
    actual["weight"] = np.exp((actual["Round"] - max_round) / max_round)

    form = {}
    for drv, grp in actual.groupby("Driver"):
        form[drv] = float(np.average(grp["actual_position"], weights=grp["weight"]))

    print(f"  [ok] 2025 form: {len(form)} drivers")
    return form


# ── step 2: load all completed 2026 race results ──────────────────────────────

def load_2026_actuals():
    """
    Fetch every completed 2026 race from FastF1 (online mode to catch recent events).
    Returns:
      completed_rounds  : list of round numbers with results
      actual_by_round   : {round: {driver: position}}
      avg_finish_2026   : {driver: average finish across completed rounds}
      pts_2026          : {driver: total championship points so far}
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache(CACHE_DIR)
        fastf1.Cache.offline_mode(enabled=False)
    except ImportError:
        print("  [warn] FastF1 not available — falling back to R1 offline only")
        return _offline_r1_fallback()

    completed_rounds = []
    actual_by_round  = {}
    pts_accum        = {d: 0 for d in DRIVERS_2026}
    finish_accum     = {d: [] for d in DRIVERS_2026}

    for rnd in range(1, N_RACES + 1):
        try:
            s = fastf1.get_session(2026, rnd, "R")
            s.load(laps=False, telemetry=False, weather=False, messages=False)
            if s.results is None or len(s.results) == 0:
                break  # no results = future race, stop scanning
            results = {}
            for _, row in s.results.iterrows():
                abbr = str(row.get("Abbreviation", ""))
                pos  = row.get("Position", np.nan)
                if abbr and pd.notna(pos):
                    pos = int(pos)
                    results[abbr] = pos
                    if abbr in pts_accum:
                        pts_accum[abbr]    += POINTS_MAP.get(pos, 0)
                        finish_accum[abbr].append(pos)
            actual_by_round[rnd] = results
            completed_rounds.append(rnd)
            w = s.results.sort_values("Position").head(3)
            top3 = "  ".join(f"P{int(r['Position'])}: {r['Abbreviation']}" for _, r in w.iterrows())
            print(f"  [ok] R{rnd:02d} {s.event.EventName:<30}  {top3}")
        except Exception:
            break

    avg_finish_2026 = {
        d: float(np.mean(vals)) if vals else 13.0
        for d, vals in finish_accum.items()
    }

    n = len(completed_rounds)
    print(f"  [ok] {n} completed 2026 races loaded  "
          f"({'R' + ', R'.join(str(r) for r in completed_rounds)})")
    return completed_rounds, actual_by_round, avg_finish_2026, pts_accum


def _offline_r1_fallback():
    """Minimal fallback: load just R1 from the local cache."""
    import fastf1
    fastf1.Cache.enable_cache(CACHE_DIR)
    fastf1.Cache.offline_mode(enabled=True)
    s = fastf1.get_session(2026, 1, "R")
    s.load(laps=False, telemetry=False, weather=False, messages=False)
    results = {}
    for _, row in s.results.iterrows():
        abbr = str(row.get("Abbreviation", ""))
        pos  = row.get("Position", np.nan)
        if abbr and pd.notna(pos):
            results[abbr] = int(pos)
    avg_finish = {d: float(results.get(d, 13)) for d in DRIVERS_2026}
    pts        = {d: POINTS_MAP.get(results.get(d, 99), 0) for d in DRIVERS_2026}
    print(f"  [ok] R01 Australian Grand Prix  (offline fallback)")
    return [1], {1: results}, avg_finish, pts


# ── step 3: blend into a single driver strength score ────────────────────────

def build_driver_strength(form_2025, avg_finish_2026, pts_2026, n_completed):
    """
    Three-layer blend:
      25% 2025 recency-weighted form   (historical baseline, adjusted for team moves)
      65% 2026 actual average finish   (real regulation-reset evidence)
      10% championship-pace implied    (points/race converted to implied position)

    The high 2026 weight reflects that the regulation change makes past seasons
    less predictive — 3 actual 2026 races are worth more than a full 2025 season
    at this stage of the championship.
    """
    # team-level average from 2025 (for team-change blending)
    team_2025 = {}
    for abbr, (_, team) in DRIVERS_2026.items():
        team_2025.setdefault(team, [])
    for abbr, (_, team) in DRIVERS_2026.items():
        if abbr in form_2025:
            team_2025[team].append(form_2025[abbr])
    team_avg_2025 = {t: float(np.mean(v)) if v else 11.0 for t, v in team_2025.items()}

    # 2025 layer: blend personal form + new-team car level for team movers
    # (HAM, SAI, BEA, etc. all drove for different teams in 2025)
    team_moves_2025 = {
        # driver: team they drove for in 2025 (None = same team)
        "HAM": "Mercedes",  # was at Mercedes, now Ferrari
        "SAI": "Ferrari",   # was at Ferrari, now Williams
        "BEA": "Ferrari",   # was a Ferrari reserve, loan to Haas
        "ANT": None,        # new driver (no 2025 form)
        "LIN": None,        # new driver
        "BOR": None,        # new driver
        "HAD": None,        # new driver
        "PER": "Red Bull Racing",  # was at RBR, now Cadillac
        "BOT": "Kick Sauber",      # was at Sauber, now Cadillac
        "OCO": "Alpine",    # was at Alpine, now Haas
        "COL": "Alpine",    # was at Alpine (partial 2024/25), now Alpine still
    }

    layer_2025 = {}
    for abbr, (_, team_2026) in DRIVERS_2026.items():
        if abbr in form_2025:
            personal = form_2025[abbr]
            car_2026 = team_avg_2025.get(team_2026, personal)
            # moved team? blend personal skill with new car level
            if team_moves_2025.get(abbr) is not None:
                layer_2025[abbr] = 0.55 * personal + 0.45 * car_2026
            else:
                layer_2025[abbr] = 0.70 * personal + 0.30 * car_2026
        else:
            # brand-new driver: anchor to new team's expected midpoint
            layer_2025[abbr] = team_avg_2025.get(team_2026, 13.0)

    # 2026 layer: actual average finish (lower = better)
    layer_2026 = {d: avg_finish_2026.get(d, 13.0) for d in DRIVERS_2026}

    # pace layer: convert pts/race to implied finishing position
    # a driver scoring 25 pts/race (all wins) maps to P1; 0 pts/race maps to ~P14
    def pts_to_implied_pos(pts, n):
        if n == 0:
            return 13.0
        pts_per_race = pts / n
        # linear interpolation: 25 pt/race -> 1.0, 0 pt/race -> 14.0
        return max(1.0, min(20.0, 14.0 - 13.0 * pts_per_race / 25.0))

    layer_pace = {
        d: pts_to_implied_pos(pts_2026.get(d, 0), n_completed)
        for d in DRIVERS_2026
    }

    # blend all three layers
    strength = {}
    for d in DRIVERS_2026:
        strength[d] = (
            W_2025 * layer_2025[d]
            + W_2026 * layer_2026[d]
            + W_PACE * layer_pace[d]
        )

    strength = {d: float(np.clip(s, 1.0, 20.0)) for d, s in strength.items()}
    return strength, layer_2025, layer_2026, layer_pace


# ── step 4: Plackett-Luce race simulation ────────────────────────────────────

def simulate_race_once(driver_strengths, rng):
    """
    One race draw from the Plackett-Luce model.
    theta_i = -log(expected_finish_i); higher theta = stronger driver.
    sigma=0.80 gives a realistic upset frequency (~35% win rate for leader).
    """
    drivers = list(driver_strengths.keys())
    n       = len(drivers)
    theta   = np.array([-np.log(driver_strengths[d]) for d in drivers])
    scores  = theta + rng.normal(0, 0.80, n)
    order   = np.argsort(-scores)
    return [drivers[i] for i in order]


# ── step 5: project the remaining season ─────────────────────────────────────

def project_season(strength, actual_by_round, completed_rounds, rng):
    """
    Build the full season projection:
      - Completed rounds: use actual results (fixed, add once).
      - Future rounds: Monte-Carlo average over N_SIM_WDC independent simulations.
        Each simulation draws one race outcome per remaining round.

    Returns:
      races_df  : per-race detail DataFrame (for CSV export)
      mc_pts    : {driver: Monte-Carlo expected points from future rounds}
    """
    drivers = list(strength.keys())
    n       = len(drivers)

    # Monte-Carlo average for future races
    future_rounds = [r for r, _, _ in CALENDAR_2026 if r not in completed_rounds]
    mc_pts = {d: 0.0 for d in drivers}
    for _ in range(N_SIM_WDC):
        for _ in future_rounds:
            order = simulate_race_once(strength, rng)
            for pos, drv in enumerate(order[:n], 1):
                mc_pts[drv] += POINTS_MAP.get(pos, 0) / N_SIM_WDC

    # representative single-run race-by-race breakdown for the CSV file
    rows = []
    for rnd, event_name, circuit in CALENDAR_2026:
        if rnd in actual_by_round:
            res = actual_by_round[rnd]
            order = sorted(res.keys(), key=lambda d: res.get(d, 99))
            order += [d for d in drivers if d not in res]
        else:
            order = simulate_race_once(strength, rng)

        for pos, drv in enumerate(order[:n], 1):
            rows.append({
                "Round": rnd, "EventName": event_name, "Circuit": circuit,
                "Driver": drv,
                "FullName": DRIVERS_2026.get(drv, (drv, ""))[0],
                "Team": DRIVERS_2026.get(drv, ("", "Unknown"))[1],
                "position": pos, "points": POINTS_MAP.get(pos, 0),
                "is_actual": rnd in actual_by_round,
            })

    return pd.DataFrame(rows), mc_pts


# ── step 6: WDC standings ──────────────────────────────────────────────────────

def compute_wdc(races_df, mc_pts, pts_2026_actual):
    """
    Final points = actual points from completed races + MC-expected future points.
    """
    meta = races_df[["Driver", "FullName", "Team"]].drop_duplicates(subset=["Driver"])

    rows = []
    for drv in DRIVERS_2026:
        name, team = DRIVERS_2026[drv]
        actual_so_far = pts_2026_actual.get(drv, 0)
        future_exp    = mc_pts.get(drv, 0.0)
        rows.append({
            "Driver":           drv,
            "FullName":         name,
            "Team":             team,
            "actual_pts":       actual_so_far,
            "projected_future": round(future_exp),
            "projected_total":  actual_so_far + round(future_exp),
        })

    wdc = (pd.DataFrame(rows)
           .sort_values("projected_total", ascending=False)
           .reset_index(drop=True))
    wdc["rank"] = wdc.index + 1
    return wdc


# ── step 7: write text file ────────────────────────────────────────────────────

def write_predictions_txt(wdc, completed_rounds, n_total, strength,
                          layer_2025, layer_2026, layer_pace, path):
    champion  = wdc.iloc[0]
    n_done    = len(completed_rounds)
    n_future  = n_total - n_done

    lines = [
        "=" * 70,
        "  2026 F1 WORLD CHAMPIONSHIP — UPDATED PROJECTED FINAL STANDINGS",
        "=" * 70,
        f"  Races completed : {n_done} of {n_total}  "
        f"(R{min(completed_rounds)}-R{max(completed_rounds)} actual + {n_future} simulated)",
        f"  Simulation      : {N_SIM_WDC} Monte-Carlo season runs (Plackett-Luce, sigma=0.80)",
        f"  Strength model  : {int(W_2026*100)}% 2026 actual form  |  "
        f"{int(W_2025*100)}% 2025 historical  |  {int(W_PACE*100)}% pts pace",
        "",
        "  NOTE: 2026 regulations (active aero + new PUs) make historical",
        "  data less reliable. The 3 real races already outweigh all of 2025",
        "  in this model. Treat as directional — not a confident prediction.",
        "",
        "─" * 70,
        "  PREDICTED WORLD CHAMPION:",
        f"    {champion['FullName']}  ({champion['Team']})",
        f"    {champion['actual_pts']} actual pts  +  "
        f"~{champion['projected_future']} projected  =  "
        f"{champion['projected_total']} total",
        "─" * 70,
        "",
        f"  {'Rk':>3}  {'Driver':<22}  {'Team':<22}  "
        f"{'Actual':>7}  {'~Future':>8}  {'Total':>7}",
        f"  {'──':>3}  {'──────────────────────':22}  {'──────────────────────':22}  "
        f"{'───────':>7}  {'────────':>8}  {'───────':>7}",
    ]

    for _, row in wdc.iterrows():
        lines.append(
            f"  {int(row['rank']):>3}  {str(row['FullName']):<22}  "
            f"{str(row['Team']):<22}  "
            f"{int(row['actual_pts']):>7}  "
            f"{int(row['projected_future']):>8}  "
            f"{int(row['projected_total']):>7}"
        )

    lines += [
        "",
        "─" * 70,
        "  Driver strength scores used for simulation (lower = better car+driver):",
        f"  {'Driver':<6}  {'2025 form':>10}  {'2026 actual':>12}  {'Pts pace':>9}  {'Blended':>8}",
    ]
    for d in sorted(strength, key=lambda x: strength[x]):
        name = DRIVERS_2026[d][0]
        lines.append(
            f"  {d:<6}  {layer_2025[d]:>10.2f}  {layer_2026[d]:>12.2f}  "
            f"{layer_pace[d]:>9.2f}  {strength[d]:>8.2f}   {name}"
        )

    lines += ["", "=" * 70]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  [ok] Rankings text: {path}")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  F1 2026 — Updated Season Projection  (3-race data + 2025 form)")
    print("=" * 70)

    print("\n[1/6] Loading 2025 driver form (historical baseline)...")
    form_2025 = load_2025_form()

    print("\n[2/6] Loading all completed 2026 race results...")
    completed_rounds, actual_by_round, avg_finish_2026, pts_2026 = \
        load_2026_actuals()

    print("\n[3/6] Building blended driver strength model...")
    strength, layer_2025, layer_2026, layer_pace = build_driver_strength(
        form_2025, avg_finish_2026, pts_2026, len(completed_rounds)
    )

    print("\n  Driver strength table (lower = stronger):")
    print(f"  {'Abbr':<5}  {'Name':<22}  {'Team':<22}  "
          f"{'Str':>6}  {'2026Avg':>8}  {'Pts':>6}")
    for d, s in sorted(strength.items(), key=lambda x: x[1]):
        name, team = DRIVERS_2026[d]
        print(f"  {d:<5}  {name:<22}  {team:<22}  "
              f"{s:>6.2f}  {layer_2026[d]:>8.2f}  "
              f"{pts_2026.get(d, 0):>6}")

    print(f"\n[4/6] Simulating {N_RACES - len(completed_rounds)} remaining races "
          f"({N_SIM_WDC} season simulations)...")
    races_df, mc_pts = project_season(strength, actual_by_round, completed_rounds, RNG)

    print("\n[5/6] Computing WDC standings...")
    wdc = compute_wdc(races_df, mc_pts, pts_2026)

    print("\n  === PROJECTED 2026 CHAMPIONSHIP ===")
    for _, row in wdc.head(10).iterrows():
        print(f"  P{int(row['rank']):02d}  {row['Driver']:<4}  "
              f"{row['actual_pts']:3d} actual + ~{row['projected_future']:3d} future"
              f" = {row['projected_total']:3d} pts  ({row['Team']})")

    print("\n[6/6] Saving outputs...")
    races_df.to_csv(os.path.join(DATA_DIR, "2026_race_by_race.csv"), index=False)
    wdc.to_csv(OUT_CSV, index=False)
    print(f"  [ok] Race-by-race: {os.path.join(DATA_DIR, '2026_race_by_race.csv')}")
    print(f"  [ok] WDC table:    {OUT_CSV}")

    write_predictions_txt(
        wdc, completed_rounds, N_RACES,
        strength, layer_2025, layer_2026, layer_pace, OUT_TXT
    )

    print("\n" + "=" * 70)
    print("  Done. Key output: data_organized/2026_wdc_predictions.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
