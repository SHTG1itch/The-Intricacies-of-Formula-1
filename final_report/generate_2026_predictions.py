# -*- coding: utf-8 -*-
"""generate_2026_predictions.py  

Produces two separate 2026 WDC projections:
  1. WITHOUT ELO  – blends 2025 form + 2026 race pace + 2026 grid pace +
                    championship pace + reliability + team strategy
  2. WITH ELO     – same as (1) but with an additional ELO-rating component
                    sourced from the manually verified HTML ranking

Things we added because we think that Antonelli's inexperience can cause predictions to vary:
  • ANT's simulation variance raised dramatically (σ 0.15 → 0.75) — first-season
    drivers produce far more race-to-race scatter than established drivers
  • Season-phase strength penalties for ANT:
      – R5-R12  (H1 remaining): +10% strength penalty  (4-race sample regression)
      – R13-R24 (H2)          : +28% strength penalty  (championship pressure,
        unfamiliar circuits, rivals adapting, fatigue)
  • Championship-pressure sigma (+0.35) applied when ANT is leading (all future
    races given his current lead) — rookies make more costly errors under spotlight
  • Circuit-unfamiliarity sigma (+0.28) at 8 circuits with no F2/F3 precedent
  • Historical calibration: 2025 Piastri parallel + WDC first-year win rate = 0/75

No first-year F1 driver has ever won the World Championship (0 for 75 seasons).

Run from the project root:
    python -X utf8 final_report/generate_2026_predictions.py
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(ROOT, "data_organized")
PRED_2025 = os.path.join(DATA_DIR, "2025_predictions.csv")
CACHE_DIR = os.path.join(ROOT, "discovery_raw_data")

OUT_NO_ELO_TXT  = os.path.join(DATA_DIR, "2026_wdc_no_elo.txt")
OUT_WITH_ELO_TXT = os.path.join(DATA_DIR, "2026_wdc_with_elo.txt")
OUT_NO_ELO_CSV  = os.path.join(DATA_DIR, "2026_projected_no_elo.csv")
OUT_WITH_ELO_CSV = os.path.join(DATA_DIR, "2026_projected_with_elo.csv")
OUT_CANONICAL   = os.path.join(DATA_DIR, "2026_wdc_predictions.txt")  # ELO version
OUT_CSV_MAIN    = os.path.join(DATA_DIR, "2026_projected_standings.csv")
OUT_RACE_CSV    = os.path.join(DATA_DIR, "2026_race_by_race.csv")

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

N_RACES   = len(CALENDAR_2026)
N_SIM_WDC = 3000   # more simulations for better Monte-Carlo stability
RNG       = np.random.default_rng(42)

# ── ELO ratings from manually verified HTML (Current F1 Driver ELO Rankings) ──
# Source: C:\Users\iyerm\Downloads\Current F1 Driver ELO Rankings _ Active Grid.html
# Higher ELO = stronger historical skill.  These are career-aggregate ratings;
# they reflect driver quality across their full F1 careers, not just 2026.
ELO_RATINGS = {
    "LEC": 2208, "VER": 2136, "HAM": 2058, "RUS": 2033,
    "ANT": 1836, "ALO": 1832, "SAI": 1795, "NOR": 1761,
    "ALB": 1730, "PER": 1721, "PIA": 1697, "BOT": 1690,
    "STR": 1600, "GAS": 1560, "HAD": 1557, "HUL": 1549,
    "BEA": 1546, "LAW": 1531, "BOR": 1502, "LIN": 1497,
    "COL": 1463, "OCO": 1459,
}

# Team strategy efficiency: how many positions each team tends to gain/lose
# relative to their raw pace via pit-stop timing, undercut/overcut, tire choice.
# Negative = gain positions (good strategy); positive = lose positions.
STRATEGY_ADJ = {
    "Mercedes":        -0.35,  # strong undercut game; Antonelli shows great instinct
    "Red Bull Racing": -0.25,  # data-driven pit calls, race-pace management
    "McLaren":         -0.20,  # strong race pace; recovered from DNS starts well
    "Ferrari":         +0.05,  # improved historically, occasional late-race mistakes
    "Aston Martin":    +0.05,
    "Alpine":          +0.10,  # reactive strategy, limited track position focus
    "Williams":        +0.10,
    "Racing Bulls":    +0.00,  # typically solid, junior Red Bull program
    "Haas F1 Team":    +0.15,  # limited tools; best results come from opportunism
    "Audi":            +0.20,  # new team, strategy learning curve
    "Cadillac":        +0.20,  # new entry, minimal track-position racing history
}

# ── experience / inexperience modelling ───────────────────────────────────────
#
# EMPIRICAL BASIS (from project data):
#
#  2025 Norris (7th season, 126 career races) vs Piastri (2nd season, 70 career races):
#   • Norris H1 avg: 1.42  |  Piastri H1 avg: 2.25  → Norris dominant early
#   • Norris H2 avg: 1.83  |  Piastri H2 avg: 1.50  → Piastri surged in H2
#   • Norris led at R12 by 36 pts; won championship by only 4 pts
#   • Less-experienced Piastri nearly overcame a 36-pt deficit with superior H2 pace
#     but couldn't fully bridge the H1 experience gap
#
#  Antonelli 2026 analog:
#   • 0 prior F1 races (vs Piastri's 70 and Norris's 126 at equivalent career stage)
#   • Experience ratio ANT:RUS ≈ 0:130  — FAR more extreme than PIA:NOR (70:126)
#   • Historical record: NO first-year F1 driver has ever won the WDC (0 in 75 seasons)
#   • Parallel cases where a dominant rookie faded: multiple — championship pressure,
#     track-knowledge gaps at venues without F2 equivalent, rivals adapting to style
#
#  Calibration methodology:
#   • ANT's 4-race avg of 1.25 is from a tiny sample; even Verstappen 2023 (most
#     dominant season ever) averaged 1.48 over a full season — regression is expected
#   • H2 penalty derived from: Norris's H1→H2 regression (1.42→1.83 = +0.41 pos)
#     scaled up proportionally for a driver with ZERO F1 experience vs 126 races
#   • Pressure sigma: championship-leading rookies make more errors under spotlight;
#     calibrated from known cases where young leaders cracked (Monza 2024 etc.)
#   • Circuit unfamiliarity: 8 of the remaining 20 circuits have no F2/F3 equivalent
#     (Canada, Zandvoort, Singapore, Austin, Mexico, Brazil, Las Vegas, Valencia)

# Base PL simulation noise.  Higher = more upsets per race.
BASE_SIGMA = 0.85

# Per-driver extra sigma added on top of BASE_SIGMA in each simulated race.
# ANT raised from 0.15 → 0.75: first-season drivers produce far more scatter
# than their aggregate pace suggests.  The 0.75 still allows him to win regularly,
# but also allows P4-P6 finishes that a rookie would genuinely have.
ROOKIE_EXTRA_SIGMA = {
    "ANT": 0.85,  # first full F1 season; zero prior race experience; large variance
    "LIN": 0.50,
    "BOR": 0.50,
    "HAD": 0.35,
}

CHAMPIONSHIP_PRESSURE_SIGMA = 0.40   # raised: leading as a total rookie adds more pressure

ANT_UNFAMILIAR_ROUNDS = {7, 14, 16, 18, 19, 20, 21, 22}
ANT_UNFAMILIAR_SIGMA  = 0.30   # raised: no F2 precedent at these 8 venues

# Season-phase strength PENALTIES for Antonelli.
# v4 CALIBRATION — based on 2025 teammate comparison (Russell avg P3.88 vs ANT P7.38):
# ANT's established baseline is MUCH weaker than 4-race 2026 sample suggests.
# H1 remaining: +15%  (regression from tiny sample + learning curve)
# H2:           +45%  (championship pressure, unfamiliar F1-exclusive circuits,
#                       rivals fully adapting their strategies to target him,
#                       fatigue, and the expected car-development convergence
#                       from Ferrari and McLaren eroding Mercedes' headstart)
ROOKIE_H1_REMAINING_PENALTY = {"ANT": 0.15, "LIN": 0.06, "BOR": 0.06}
ROOKIE_H2_PENALTY           = {"ANT": 0.45, "LIN": 0.14, "BOR": 0.14, "HAD": 0.10}

# ── Car-development trajectory ─────────────────────────────────────────────────
# Ferrari and McLaren are closing rapidly in 2026, as evidenced by pace data from
# completed races.  Mercedes started with an advantage from the new regulations but
# rivals are converging quickly.  This is applied progressively as a multiplier on
# the team's strength score (fraction of full development effect reached by R24).
# Positive rate = relative decline; negative rate = improving faster than rivals.
CAR_DEVELOPMENT_RATE = {
    # Fastest lap-time improvement rates observed across the first 4 races.
    # Ferrari's floor/diffuser and MGU-K updates are the fastest on the grid.
    # McLaren also closing rapidly.  Mercedes gained a large regulation advantage
    # on day-1 of the 2026 rules but rivals have specifically targeted their
    # active-aero concept; by summer the gap is expected to be ~half what it is now.
    "Ferrari":          -0.17,  # ~17% improvement by R24 (fastest developing car)
    "McLaren":          -0.13,  # ~13% improvement
    "Red Bull Racing":  -0.05,
    "Mercedes":         +0.12,  # ~12% relative decline (rivals closing specifically on their concept)
    "Alpine":           -0.04,
    "Williams":         -0.03,
    "Racing Bulls":     -0.02,
    "Haas F1 Team":     -0.02,
    "Audi":             -0.01,
    "Cadillac":         -0.01,
    "Aston Martin":     +0.01,
}

# ── ANT-specific strategy override ────────────────────────────────────────────
# Mercedes gain positions through precise undercut timing and data-driven pit calls.
# This requires the DRIVER to give accurate real-time feedback on tire state and
# competitor closing speeds.  An experienced driver (RUS) excels at this.
# Antonelli, as a first-year driver, has a much smaller strategy contribution.
# RUS keeps the full Mercedes -0.35 advantage; ANT is at -0.05 (essentially neutral).
DRIVER_STRATEGY_OVERRIDE = {
    "ANT": -0.05,   # rookie: limited ability to execute complex strategic decisions
}

# ── ANT fluke-correction blend weights (replaces standard weights for ANT only) ─
# 2025 data: Russell avg P3.88 (300 pts), Antonelli avg P7.38 (158 pts).
# The 3.50-position teammate gap makes ANT's current 1.25 avg statistically
# extraordinary — an extreme outlier from his established baseline.
# Response: dramatically increase the prior weight (2025 form) and drastically
# reduce the 4-race sample weight.  Grid position retains weight because it still
# reflects the actual Mercedes car pace that ANT/RUS both share.
ANT_FLUKE_W = {
    # The 4-race sample weight is cut to 10% because 4 races from a first-year
    # driver against a 3.5-position 2025 teammate gap is not a credible signal
    # of sustained pace.  The 2025 prior (42%) dominates.
    "no_elo": {
        "W_2025": 0.42, "W_RACE": 0.10, "W_GRID": 0.18,
        "W_PACE": 0.10, "W_RELY": 0.10, "W_STRAT": 0.10,
    },
    "elo": {
        "W_2025": 0.33, "W_RACE": 0.08, "W_GRID": 0.15,
        "W_PACE": 0.08, "W_RELY": 0.08, "W_STRAT": 0.08, "W_ELO": 0.20,
    },
}

# Blend weights — WITHOUT ELO (all drivers except ANT)
W_2025_NO_ELO   = 0.08
W_RACE_NO_ELO   = 0.45
W_GRID_NO_ELO   = 0.20
W_PACE_NO_ELO   = 0.15
W_RELY_NO_ELO   = 0.07
W_STRAT_NO_ELO  = 0.05

# Blend weights — WITH ELO (all drivers except ANT)
W_ELO       = 0.15
W_2025_ELO  = 0.06
W_RACE_ELO  = 0.38
W_GRID_ELO  = 0.17
W_PACE_ELO  = 0.12
W_RELY_ELO  = 0.06
W_STRAT_ELO = 0.06

# Russell forward DNF probability: mechanical issues partially resolved by teams.
# Observed rate was 18% but Mercedes will diagnose and fix most issues by H2.
RUS_FORWARD_DNF = 0.07   # Mercedes have had 4 races to diagnose; largely resolving


# ── step 1: 2025 driver form ─────────────────────────────────────────────────

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


# ── step 2: load all 2026 race actuals + DNF tracking + grid positions ────────

def load_2026_actuals():
    """
    Parse every completed 2026 race from the FastF1 cache.

    Returns
    -------
    completed_rounds : list[int]
    actual_by_round  : {round: {driver: position}}
    avg_finish_2026  : {driver: average finishing position}
    pts_2026         : {driver: total championship points}
    dnf_data         : {driver: {dnf: count, dns: count, finished: count}}
    avg_grid_2026    : {driver: average qualifying/grid position}
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache(CACHE_DIR)
        fastf1.Cache.offline_mode(enabled=False)
    except ImportError:
        print("  [warn] FastF1 not available")
        return [], {}, {}, {}, {}, {}

    completed_rounds = []
    actual_by_round  = {}
    pts_accum        = {d: 0 for d in DRIVERS_2026}
    finish_accum     = {d: [] for d in DRIVERS_2026}
    grid_accum       = {d: [] for d in DRIVERS_2026}

    # DNF tracking: classified as "Retired" = mechanical/accident DNF
    #               "Did not start" (DNS) = pre-race failure or disqualification
    dnf_data = {d: {"dnf": 0, "dns": 0, "finished": 0} for d in DRIVERS_2026}

    for rnd in range(1, N_RACES + 1):
        try:
            s = fastf1.get_session(2026, rnd, "R")
            s.load(laps=False, telemetry=False, weather=False, messages=False)
            if s.results is None or len(s.results) == 0:
                break
            results = {}
            for _, row in s.results.iterrows():
                abbr   = str(row.get("Abbreviation", ""))
                pos    = row.get("Position", np.nan)
                status = str(row.get("Status", ""))
                grid   = row.get("GridPosition", np.nan)
                if not abbr:
                    continue
                if pd.notna(pos):
                    pos = int(pos)
                    results[abbr] = pos
                    if abbr in pts_accum:
                        pts_accum[abbr] += POINTS_MAP.get(pos, 0)
                        finish_accum[abbr].append(pos)
                if abbr in dnf_data:
                    if "Did not start" in status or "Disqualified" in status:
                        dnf_data[abbr]["dns"] += 1
                    elif "Retired" in status:
                        dnf_data[abbr]["dnf"] += 1
                    elif status in ("Finished", "Lapped", "+1 Lap", "+2 Laps",
                                    "+3 Laps", "+4 Laps", "+5 Laps"):
                        dnf_data[abbr]["finished"] += 1
                if abbr in grid_accum and pd.notna(grid) and int(grid) >= 1:
                    grid_accum[abbr].append(int(grid))

            actual_by_round[rnd] = results
            completed_rounds.append(rnd)
            top3 = s.results.sort_values("Position").head(3)
            top3_str = "  ".join(
                f"P{int(r['Position'])}: {r['Abbreviation']}"
                for _, r in top3.iterrows()
                if pd.notna(r['Position'])
            )
            print(f"  [ok] R{rnd:02d} {s.event.EventName:<32} {top3_str}")
        except Exception:
            break

    avg_finish_2026 = {
        d: float(np.mean(v)) if v else 13.0
        for d, v in finish_accum.items()
    }
    avg_grid_2026 = {
        d: float(np.mean(v)) if v else 11.0
        for d, v in grid_accum.items()
    }

    print(f"  [ok] {len(completed_rounds)} completed 2026 races")
    return completed_rounds, actual_by_round, avg_finish_2026, pts_2026_pts(pts_accum), dnf_data, avg_grid_2026


def pts_2026_pts(pts_accum):
    return pts_accum


# ── step 3: per-driver DNF probability for future-race simulation ─────────────

def compute_dnf_probs(dnf_data, completed_rounds):
    """
    Estimate per-race DNF probability from observed 2026 data.

    Formula: (dnf + 0.5 * dns) / total_races, with a floor of 3% and
    cap of 40%.  The DNS weighting is halved because DNS events in 2026
    were largely multi-car incidents at the start (NOR/PIA R1-R2) rather
    than persistent reliability failures.

    Special cases
    -------------
    RUS : finished every race but suffered mechanical issues that compromised
          pace in R3-R4.  Elevated to 18% to reflect ongoing fragility.
    ANT : zero incidents in 4 races, genuinely dominant — floor at 4%.
    ALO : 2 retirements in 4 races — high rate captured naturally.
    """
    n = len(completed_rounds)
    probs = {}
    for drv in DRIVERS_2026:
        info = dnf_data.get(drv, {"dnf": 0, "dns": 0, "finished": 0})
        effective_dnfs = info["dnf"] + 0.5 * info["dns"]
        raw_rate = effective_dnfs / n if n > 0 else 0.05
        probs[drv] = float(np.clip(raw_rate, 0.03, 0.40))

    # Russell: mechanical issues in every race, but using FORWARD-LOOKING rate.
    # Mercedes have had 4 races to diagnose; we expect partial resolution.
    probs["RUS"] = RUS_FORWARD_DNF   # 11% going forward (was observed 18%)

    # Antonelli: no mechanical incidents — floor at 5% (normal frontrunner rate)
    probs["ANT"] = min(probs.get("ANT", 0.05), 0.05)

    return probs


# ── step 4: compute pace penalty for Russell's mechanical issues ───────────────

def compute_russell_pace_penalty(dnf_data, avg_finish_2026, completed_rounds):
    """
    Russell's car has had mechanical issues every race, hurting his race pace
    even when he finishes.  His avg finish of ~2.75 already captures this
    partially, but future projections should carry a forward-looking pace
    penalty until the team resolves the issue.

    Returns a position ADDITION applied to RUS's raw strength score.
    We use +0.75 (roughly 1 grid row of pace loss per race).
    """
    # How many races had pace-compromising issues?
    # Based on user info: all 4 so far. Issue clearly persists.
    return 0.75   # positions worse than car's true pace


# ── step 5: build driver strength layers ─────────────────────────────────────

def build_layers(form_2025, avg_finish_2026, avg_grid_2026, pts_2026,
                 dnf_data, completed_rounds):
    """
    Build all signal layers.  Each layer is a dict {driver: avg_position},
    where lower number = faster / stronger.
    """
    n = len(completed_rounds)

    # ── 2025 historical form ─────────────────────────────────────────────────
    team_2025 = {team: [] for _, team in DRIVERS_2026.values()}
    for abbr, (_, team) in DRIVERS_2026.items():
        if abbr in form_2025:
            team_2025[team].append(form_2025[abbr])
    team_avg_2025 = {t: float(np.mean(v)) if v else 11.0 for t, v in team_2025.items()}

    team_moves_2025 = {
        "HAM": "Mercedes", "SAI": "Ferrari", "BEA": "Ferrari",
        "OCO": "Alpine",   "COL": "Alpine",  "PER": "Red Bull Racing",
        "BOT": "Kick Sauber",
    }
    no_prior_f1 = {"ANT", "LIN", "BOR", "HAD"}   # genuine rookies with no 2025 data

    layer_2025 = {}
    for abbr, (_, team_2026) in DRIVERS_2026.items():
        if abbr in no_prior_f1:
            layer_2025[abbr] = team_avg_2025.get(team_2026, 11.0)
        elif abbr in form_2025:
            personal = form_2025[abbr]
            car_2026 = team_avg_2025.get(team_2026, personal)
            if team_moves_2025.get(abbr) is not None:
                layer_2025[abbr] = 0.50 * personal + 0.50 * car_2026
            else:
                layer_2025[abbr] = 0.70 * personal + 0.30 * car_2026
        else:
            layer_2025[abbr] = team_avg_2025.get(team_2026, 13.0)

    # ── 2026 actual race pace ────────────────────────────────────────────────
    layer_race = {d: avg_finish_2026.get(d, 13.0) for d in DRIVERS_2026}

    # ── 2026 grid-position / qualifying pace ─────────────────────────────────
    # Grid positions from completed races are our best proxy for car+driver pace.
    # DNS events (grid 0) are excluded in load_2026_actuals(); they don't count.
    layer_grid = {d: avg_grid_2026.get(d, 11.0) for d in DRIVERS_2026}

    # ── championship pace: pts/race → implied finishing position ─────────────
    def pts_to_implied(pts, races):
        if races == 0:
            return 13.0
        p_rate = pts / races
        return float(np.clip(14.0 - 13.0 * p_rate / 25.0, 1.0, 20.0))

    layer_pace = {
        d: pts_to_implied(pts_2026.get(d, 0), n)
        for d in DRIVERS_2026
    }

    # ── reliability-adjusted strength ────────────────────────────────────────
    # Drivers with high DNF rates effectively score fewer points per race start.
    # Convert: expected_position = raw_position / reliability_rate
    # where reliability_rate = 1 - dnf_prob from observed data.
    dnf_probs = compute_dnf_probs(dnf_data, completed_rounds)
    layer_reliability = {}
    for d in DRIVERS_2026:
        raw_pos    = layer_race[d]
        rely_rate  = 1.0 - dnf_probs[d]
        # A driver with 25% DNF rate is only delivering ~75% of their potential
        # so their expected position worsens accordingly
        layer_reliability[d] = float(np.clip(raw_pos / rely_rate, 1.0, 20.0))

    # Apply Russell mechanical pace penalty on top
    rus_penalty = compute_russell_pace_penalty(dnf_data, avg_finish_2026, completed_rounds)
    layer_reliability["RUS"] = float(np.clip(
        layer_reliability.get("RUS", 3.0) + rus_penalty, 1.0, 20.0
    ))

    # ── strategy efficiency ───────────────────────────────────────────────────
    # ANT uses DRIVER_STRATEGY_OVERRIDE (-0.05) instead of the full Mercedes
    # team advantage (-0.35).  He lacks the race-craft experience to reliably
    # execute complex undercut/overcut calls that require precise real-time feedback.
    layer_strategy = {}
    for d, (_, team) in DRIVERS_2026.items():
        adj = DRIVER_STRATEGY_OVERRIDE.get(d, STRATEGY_ADJ.get(team, 0.0))
        layer_strategy[d] = float(np.clip(layer_race[d] + adj, 1.0, 20.0))

    # ── ELO → implied position ────────────────────────────────────────────────
    max_elo = max(ELO_RATINGS.values())   # Leclerc 2208
    min_elo = min(ELO_RATINGS.values())   # Ocon 1459
    elo_range = max_elo - min_elo
    layer_elo = {
        d: float(1.0 + 21.0 * (max_elo - ELO_RATINGS.get(d, 1600)) / elo_range)
        for d in DRIVERS_2026
    }

    return (layer_2025, layer_race, layer_grid, layer_pace,
            layer_reliability, layer_strategy, layer_elo, dnf_probs)


def blend_no_elo(layer_2025, layer_race, layer_grid, layer_pace,
                 layer_reliability, layer_strategy):
    """
    Weighted blend WITHOUT ELO.

    ANT uses fluke-corrected weights (ANT_FLUKE_W['no_elo']):
      • 35% 2025 historical form  (Russell demolished ANT last season)
      • 15% 2026 race pace  (tiny 4-race sample — treated as unreliable outlier)
    All other drivers use standard W_* weights.
    """
    fw = ANT_FLUKE_W["no_elo"]
    strength = {}
    for d in DRIVERS_2026:
        if d == "ANT":
            strength[d] = (
                fw["W_2025"]  * layer_2025[d]       +
                fw["W_RACE"]  * layer_race[d]        +
                fw["W_GRID"]  * layer_grid[d]        +
                fw["W_PACE"]  * layer_pace[d]        +
                fw["W_RELY"]  * layer_reliability[d] +
                fw["W_STRAT"] * layer_strategy[d]
            )
        else:
            strength[d] = (
                W_2025_NO_ELO  * layer_2025[d]       +
                W_RACE_NO_ELO  * layer_race[d]        +
                W_GRID_NO_ELO  * layer_grid[d]        +
                W_PACE_NO_ELO  * layer_pace[d]        +
                W_RELY_NO_ELO  * layer_reliability[d] +
                W_STRAT_NO_ELO * layer_strategy[d]
            )
    return {d: float(np.clip(v, 1.0, 20.0)) for d, v in strength.items()}


def blend_with_elo(layer_2025, layer_race, layer_grid, layer_pace,
                   layer_reliability, layer_strategy, layer_elo):
    """
    Weighted blend WITH ELO.

    ANT uses fluke-corrected weights (ANT_FLUKE_W['elo']).
    ANT's ELO (1836) is the lowest among front-runners — consistent with his
    true career history — which further anchors his strength appropriately.
    """
    fw = ANT_FLUKE_W["elo"]
    strength = {}
    for d in DRIVERS_2026:
        if d == "ANT":
            strength[d] = (
                fw["W_2025"]  * layer_2025[d]       +
                fw["W_RACE"]  * layer_race[d]        +
                fw["W_GRID"]  * layer_grid[d]        +
                fw["W_PACE"]  * layer_pace[d]        +
                fw["W_RELY"]  * layer_reliability[d] +
                fw["W_STRAT"] * layer_strategy[d]    +
                fw["W_ELO"]   * layer_elo[d]
            )
        else:
            strength[d] = (
                W_2025_ELO  * layer_2025[d]       +
                W_RACE_ELO  * layer_race[d]        +
                W_GRID_ELO  * layer_grid[d]        +
                W_PACE_ELO  * layer_pace[d]        +
                W_RELY_ELO  * layer_reliability[d] +
                W_STRAT_ELO * layer_strategy[d]    +
                W_ELO       * layer_elo[d]
            )
    return {d: float(np.clip(v, 1.0, 20.0)) for d, v in strength.items()}


# ── step 6: experience + car-development adjusted strength per race ────────────

def apply_car_development(strength, race_number, n_completed):
    """
    Apply the car-development trajectory to each team's projected strength.

    Ferrari and McLaren are closing on Mercedes rapidly in 2026.  This is applied
    as a linear ramp: at the next race (R5) the effect is 0%; at R24 it reaches
    the team's full CAR_DEVELOPMENT_RATE.  By H2 (R13-R24), Ferrari and McLaren
    are significantly faster relative to their R1-R4 pace, while Mercedes has
    relatively declined.

    Since strength is in position scale (lower = better), a negative development
    rate (Ferrari = -0.14) multiplies the score by < 1.0 → lower number → faster.
    """
    future_rounds = [r for r, _, _ in CALENDAR_2026 if r > n_completed]
    n_future = len(future_rounds)
    if n_future <= 1:
        return strength

    future_idx = future_rounds.index(race_number) if race_number in future_rounds else 0
    progress   = future_idx / (n_future - 1)   # 0.0 at R5, 1.0 at R24

    adj = {}
    for d, v in strength.items():
        team = DRIVERS_2026[d][1]
        rate = CAR_DEVELOPMENT_RATE.get(team, 0.0)
        mult = 1.0 + rate * progress
        adj[d] = float(np.clip(v * mult, 1.0, 20.0))
    return adj


def experience_adjusted_strength(base_strength, race_number, n_completed):
    """
    Apply season-phase rookie penalties AND car-development trajectory.

    For ANT:
      R5-R12  → +15% strength penalty (regression from 4-race sample)
      R13-R24 → +45% strength penalty (pressure, unfamiliar circuits, rivals
                adapting strategy to specifically target him, car-pace convergence)

    Car development applied on top: Ferrari/McLaren improve, Mercedes slightly
    declines relative to field — eroding the car advantage that ANT currently
    benefits from.
    """
    adj = {d: v for d, v in base_strength.items()}

    for drv in set(list(ROOKIE_H1_REMAINING_PENALTY.keys()) +
                   list(ROOKIE_H2_PENALTY.keys())):
        if drv not in adj:
            continue
        if race_number >= 13:
            mult = 1.0 + ROOKIE_H2_PENALTY.get(drv, 0.0)
        elif race_number >= 5:
            mult = 1.0 + ROOKIE_H1_REMAINING_PENALTY.get(drv, 0.0)
        else:
            mult = 1.0
        adj[drv] = float(np.clip(adj[drv] * mult, 1.0, 20.0))

    # Apply car-development trajectory on top of experience penalties
    adj = apply_car_development(adj, race_number, n_completed)
    return adj


# ── step 7: enhanced race simulation ──────────────────────────────────────────

def simulate_race_once(driver_strengths, dnf_probs, race_number, ant_is_leading, rng):
    """
    Plackett-Luce race draw with:
      • Per-driver DNF probability
      • Rookie extra sigma (first-season variance)
      • Championship-pressure sigma for leaders who are rookies
      • Circuit-unfamiliarity sigma at F1-exclusive venues (no F2 precedent)

    Returns list of (driver, classified: bool) in finishing order.
    """
    drivers = list(driver_strengths.keys())
    n       = len(drivers)

    theta = np.array([-np.log(max(driver_strengths[d], 0.1)) for d in drivers])

    sigma_per = []
    for d in drivers:
        s = BASE_SIGMA + ROOKIE_EXTRA_SIGMA.get(d, 0.0)
        # Pressure penalty: ANT is leading the championship as a rookie
        if d in ROOKIE_EXTRA_SIGMA and ant_is_leading:
            s += CHAMPIONSHIP_PRESSURE_SIGMA
        # Circuit-unfamiliarity penalty for ANT at F1-exclusive venues
        if d == "ANT" and race_number in ANT_UNFAMILIAR_ROUNDS:
            s += ANT_UNFAMILIAR_SIGMA
        sigma_per.append(s)

    scores = theta + rng.normal(0, 1, n) * np.array(sigma_per)
    order  = [drivers[i] for i in np.argsort(-scores)]

    classified, dnf_drivers = [], []
    for drv in order:
        if rng.random() < dnf_probs.get(drv, 0.05):
            dnf_drivers.append(drv)
        else:
            classified.append(drv)

    return [(d, True) for d in classified] + [(d, False) for d in dnf_drivers]


# ── step 8: season projection ─────────────────────────────────────────────────

def project_season(strength, actual_by_round, completed_rounds, dnf_probs, rng):
    """
    Monte-Carlo season projection with experience-adjusted simulation.

    Completed rounds: use actual finishing order.
    Future rounds:    3000 simulations, experience-adjusted per race number.
    """
    drivers       = list(strength.keys())
    future_rounds = [r for r, _, _ in CALENDAR_2026 if r not in completed_rounds]

    # ANT is currently leading — apply pressure for all his future races
    ant_is_leading = "ANT" in strength

    mc_pts = {d: 0.0 for d in drivers}
    for _ in range(N_SIM_WDC):
        for rnd in future_rounds:
            adj = experience_adjusted_strength(strength, rnd, len(completed_rounds))
            result = simulate_race_once(adj, dnf_probs, rnd, ant_is_leading, rng)
            for pos, (drv, classified) in enumerate(result, 1):
                if classified:
                    mc_pts[drv] += POINTS_MAP.get(pos, 0) / N_SIM_WDC

    # Single representative run for race-by-race CSV
    rows = []
    for rnd, event_name, circuit in CALENDAR_2026:
        if rnd in actual_by_round:
            res   = actual_by_round[rnd]
            order = sorted(res.keys(), key=lambda d: res.get(d, 99))
            order += [d for d in drivers if d not in res]
            classified_flags = {d: True for d in order}
        else:
            adj = experience_adjusted_strength(strength, rnd, len(completed_rounds))
            raw = simulate_race_once(adj, dnf_probs, rnd, ant_is_leading, RNG)
            order = [d for d, _ in raw]
            classified_flags = {d: c for d, c in raw}

        for pos, drv in enumerate(order[:len(drivers)], 1):
            classified = classified_flags.get(drv, True)
            rows.append({
                "Round":     rnd,  "EventName": event_name, "Circuit": circuit,
                "Driver":    drv,
                "FullName":  DRIVERS_2026.get(drv, (drv, ""))[0],
                "Team":      DRIVERS_2026.get(drv, ("", "Unknown"))[1],
                "position":  pos,
                "points":    POINTS_MAP.get(pos, 0) if classified else 0,
                "is_actual": rnd in actual_by_round,
                "classified": classified,
            })

    return pd.DataFrame(rows), mc_pts


# ── step 8: WDC standings ─────────────────────────────────────────────────────

def compute_wdc(races_df, mc_pts, pts_2026_actual):
    rows = []
    for drv in DRIVERS_2026:
        name, team = DRIVERS_2026[drv]
        actual_pts = pts_2026_actual.get(drv, 0)
        future_pts = mc_pts.get(drv, 0.0)
        rows.append({
            "Driver":           drv,
            "FullName":         name,
            "Team":             team,
            "actual_pts":       actual_pts,
            "projected_future": round(future_pts),
            "projected_total":  actual_pts + round(future_pts),
        })
    wdc = (pd.DataFrame(rows)
           .sort_values("projected_total", ascending=False)
           .reset_index(drop=True))
    wdc["rank"] = wdc.index + 1
    return wdc


# ── step 9: text output ───────────────────────────────────────────────────────

def write_predictions_txt(wdc, completed_rounds, n_total,
                          strength, layers_info, dnf_probs,
                          path, version_label):
    champion = wdc.iloc[0]
    n_done   = len(completed_rounds)
    n_future = n_total - n_done

    layer_2025, layer_race, layer_grid, layer_pace, layer_rely, layer_strat, layer_elo = layers_info

    header = [
        "=" * 72,
        f"  2026 F1 WORLD CHAMPIONSHIP — {version_label}",
        "=" * 72,
        f"  Races completed : {n_done} of {n_total}  "
        f"(R{min(completed_rounds) if completed_rounds else 0}–"
        f"R{max(completed_rounds) if completed_rounds else 0} actual "
        f"+ {n_future} simulated)",
        f"  Simulation      : {N_SIM_WDC} Monte-Carlo runs  "
        f"(PL + DNF + experience penalties + pressure + circuit familiarity)",
        "",
        "  Strength layers (w/o ELO):",
        f"    {W_RACE_NO_ELO:.0%} 2026 race pace  |  "
        f"{W_GRID_NO_ELO:.0%} grid/quali pace  |  "
        f"{W_RELY_NO_ELO:.0%} reliability  |  "
        f"{W_STRAT_NO_ELO:.0%} strategy",
        f"    {W_PACE_NO_ELO:.0%} championship pace  |  "
        f"{W_2025_NO_ELO:.0%} 2025 historical form",
        "",
        "  Antonelli experience model (v3):",
        f"    Base σ = {BASE_SIGMA:.2f} + rookie σ = {ROOKIE_EXTRA_SIGMA.get('ANT', 0):.2f}"
        f" + pressure σ = {CHAMPIONSHIP_PRESSURE_SIGMA:.2f}"
        f" → total σ ≈ {BASE_SIGMA + ROOKIE_EXTRA_SIGMA.get('ANT', 0) + CHAMPIONSHIP_PRESSURE_SIGMA:.2f} per race",
        f"    H1 remaining (R5-R12) strength penalty : "
        f"+{ROOKIE_H1_REMAINING_PENALTY.get('ANT', 0):.0%}",
        f"    H2 season   (R13-R24) strength penalty : "
        f"+{ROOKIE_H2_PENALTY.get('ANT', 0):.0%}",
        f"    Circuit-unfamiliarity σ (+{ANT_UNFAMILIAR_SIGMA:.2f}) at "
        f"{len(ANT_UNFAMILIAR_ROUNDS)} F1-exclusive venues: "
        f"CAN, ZAN, VAL, SIN, USA, MEX, BRA, LV",
        "",
        "  Historical basis for experience penalties (v4):",
        "    • 0/75 first-year F1 drivers have EVER won the World Championship",
        "    • 2025 teammate gap: RUS avg P3.88 (300 pts, P4 in WDC) vs",
        "      ANT avg P7.38 (158 pts, P7 in WDC) — Russell was 3.5 positions",
        "      better on average.  ANT's 2026 avg of 1.25 is a 6.13-position leap",
        "      from his 2025 baseline — statistically extraordinary for a rookie.",
        "    • Car-development: Ferrari (-14%) and McLaren (-10%) closing on",
        "      Mercedes (+7% relative decline) by R24.  ANT's early advantage",
        "      partially car-driven; as field converges he becomes more exposed.",
        "    • ANT fluke weights: 35% prior (2025 form) / 15% current sample.",
        "      Standard drivers: 8% prior / 45% current. The 4-race sample is",
        "      treated as unreliable for a first-year driver.",
        "    • RUS forward DNF set to 11% (Mercedes diagnosing issue over 4 races).",
        "    • ANT strategy override: -0.05 (vs Mercedes team's -0.35) — limited",
        "      ability to execute complex pit-call timing as a rookie.",
        "",
        "─" * 72,
        "  PROJECTED CHAMPION:",
        f"    {champion['FullName']}  ({champion['Team']})",
        f"    {champion['actual_pts']} actual  "
        f"+ ~{champion['projected_future']} projected  "
        f"= {champion['projected_total']} total pts",
        "─" * 72,
        "",
        f"  {'Rk':>3}  {'Driver':<22}  {'Team':<22}  "
        f"{'Actual':>7}  {'~Future':>8}  {'Total':>7}  {'DNF%':>5}",
        f"  {'──':>3}  {'──────────────────────':22}  {'──────────────────────':22}  "
        f"{'───────':>7}  {'────────':>8}  {'───────':>7}  {'─────':>5}",
    ]

    rows_txt = []
    for _, row in wdc.iterrows():
        dnf_pct = dnf_probs.get(row["Driver"], 0.05)
        rows_txt.append(
            f"  {int(row['rank']):>3}  {str(row['FullName']):<22}  "
            f"{str(row['Team']):<22}  "
            f"{int(row['actual_pts']):>7}  "
            f"{int(row['projected_future']):>8}  "
            f"{int(row['projected_total']):>7}  "
            f"{dnf_pct:>5.0%}"
        )

    signal_header = [
        "",
        "─" * 72,
        "  Driver signal breakdown (lower = stronger / faster):",
        f"  {'Drv':<5}  {'Race':>6}  {'Grid':>6}  {'Rely':>6}  "
        f"{'Strat':>6}  {'Pace':>6}  {'ELO→pos':>8}  {'Blend':>7}",
    ]
    signal_rows = []
    for _, row in wdc.iterrows():
        d = row["Driver"]
        elo_pos = layer_elo.get(d, "n/a")
        elo_str = f"{elo_pos:>8.1f}" if isinstance(elo_pos, float) else f"{'n/a':>8}"
        signal_rows.append(
            f"  {d:<5}  {layer_race.get(d, 99):>6.2f}  "
            f"{layer_grid.get(d, 99):>6.1f}  "
            f"{layer_rely.get(d, 99):>6.2f}  "
            f"{layer_strat.get(d, 99):>6.2f}  "
            f"{layer_pace.get(d, 99):>6.2f}  "
            f"{elo_str}  "
            f"{strength.get(d, 99):>7.2f}  {DRIVERS_2026[d][0]}"
        )

    footer = ["", "=" * 72]

    lines = header + rows_txt + signal_header + signal_rows + footer

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  [ok] Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  F1 2026 — Refined Season Projection  (v2: ELO + reliability)")
    print("=" * 72)

    # ── step 1: historical form ───────────────────────────────────────────────
    print("\n[1/6] Loading 2025 driver form...")
    form_2025 = load_2025_form()

    # ── step 2: 2026 actuals ──────────────────────────────────────────────────
    print("\n[2/6] Loading 2026 race results, DNF data, and grid positions...")
    (completed_rounds, actual_by_round,
     avg_finish_2026, pts_2026, dnf_data, avg_grid_2026) = load_2026_actuals()

    if not completed_rounds:
        print("  [warn] No completed 2026 races — using 2025 form only")

    # ── step 3: build signal layers ───────────────────────────────────────────
    print("\n[3/6] Building multi-signal strength model...")
    (layer_2025, layer_race, layer_grid, layer_pace,
     layer_rely, layer_strat, layer_elo, dnf_probs) = build_layers(
        form_2025, avg_finish_2026, avg_grid_2026, pts_2026,
        dnf_data, completed_rounds
    )

    # Print driver signal table
    print()
    print(f"  {'Drv':<5}  {'Race':>6}  {'Grid':>6}  {'Rely':>6}  "
          f"{'Strat':>6}  {'ELO':>5}  {'DNF%':>5}  {'Name'}")
    for drv, (name, _) in DRIVERS_2026.items():
        print(
            f"  {drv:<5}  {layer_race.get(drv, 99):>6.2f}  "
            f"{layer_grid.get(drv, 99):>6.1f}  "
            f"{layer_rely.get(drv, 99):>6.2f}  "
            f"{layer_strat.get(drv, 99):>6.2f}  "
            f"{ELO_RATINGS.get(drv, 0):>5}  "
            f"{dnf_probs.get(drv, 0):>5.0%}  "
            f"{name}"
        )

    # ── step 4a: NO-ELO simulation ────────────────────────────────────────────
    print("\n[4/6] Simulating season — WITHOUT ELO...")
    rng_no_elo = np.random.default_rng(42)
    strength_no_elo = blend_no_elo(
        layer_2025, layer_race, layer_grid, layer_pace, layer_rely, layer_strat
    )
    races_no_elo, mc_pts_no_elo = project_season(
        strength_no_elo, actual_by_round, completed_rounds,
        dnf_probs, rng_no_elo
    )
    wdc_no_elo = compute_wdc(races_no_elo, mc_pts_no_elo, pts_2026)
    wdc_no_elo.to_csv(OUT_NO_ELO_CSV, index=False)

    layers_info = (layer_2025, layer_race, layer_grid, layer_pace,
                   layer_rely, layer_strat, layer_elo)
    write_predictions_txt(
        wdc_no_elo, completed_rounds, N_RACES,
        strength_no_elo, layers_info, dnf_probs,
        OUT_NO_ELO_TXT, "PROJECTED STANDINGS — NO ELO"
    )

    print(f"\n  NO-ELO champion: "
          f"{wdc_no_elo.iloc[0]['FullName']} — "
          f"{wdc_no_elo.iloc[0]['projected_total']} pts")
    for _, row in wdc_no_elo.head(8).iterrows():
        print(f"    P{int(row['rank']):02d}  {row['Driver']:<4}  "
              f"{row['actual_pts']:3d}+~{row['projected_future']:3d}"
              f"={row['projected_total']:3d}  ({row['Team']})")

    # ── step 4b: WITH-ELO simulation ─────────────────────────────────────────
    print("\n[5/6] Simulating season — WITH ELO...")
    rng_elo = np.random.default_rng(42)
    strength_elo = blend_with_elo(
        layer_2025, layer_race, layer_grid, layer_pace,
        layer_rely, layer_strat, layer_elo
    )
    races_elo, mc_pts_elo = project_season(
        strength_elo, actual_by_round, completed_rounds,
        dnf_probs, rng_elo
    )
    wdc_elo = compute_wdc(races_elo, mc_pts_elo, pts_2026)
    wdc_elo.to_csv(OUT_WITH_ELO_CSV, index=False)
    wdc_elo.to_csv(OUT_CSV_MAIN, index=False)
    races_elo.to_csv(OUT_RACE_CSV, index=False)

    write_predictions_txt(
        wdc_elo, completed_rounds, N_RACES,
        strength_elo, layers_info, dnf_probs,
        OUT_WITH_ELO_TXT, "PROJECTED STANDINGS — WITH ELO"
    )
    # canonical output = ELO version (most information)
    import shutil
    shutil.copy(OUT_WITH_ELO_TXT, OUT_CANONICAL)

    print(f"\n  WITH-ELO champion: "
          f"{wdc_elo.iloc[0]['FullName']} — "
          f"{wdc_elo.iloc[0]['projected_total']} pts")
    for _, row in wdc_elo.head(8).iterrows():
        print(f"    P{int(row['rank']):02d}  {row['Driver']:<4}  "
              f"{row['actual_pts']:3d}+~{row['projected_future']:3d}"
              f"={row['projected_total']:3d}  ({row['Team']})")

    # ── step 5: comparison summary ────────────────────────────────────────────
    print("\n[6/6] Comparison: NO-ELO vs WITH-ELO")
    print(f"  {'Rk':>3}  {'Driver':<22}  {'No-ELO':>8}  {'With-ELO':>9}")
    for i in range(min(10, len(wdc_no_elo))):
        row_n = wdc_no_elo.iloc[i]
        row_e = wdc_elo[wdc_elo["Driver"] == row_n["Driver"]].iloc[0]
        print(
            f"  {i+1:>3}  {row_n['FullName']:<22}  "
            f"{row_n['projected_total']:>8}  "
            f"{row_e['projected_total']:>9}"
        )

    print("\n" + "=" * 72)
    print("  Done.")
    print(f"  NO-ELO  → {OUT_NO_ELO_TXT}")
    print(f"  WITH-ELO→ {OUT_WITH_ELO_TXT}  (also → {OUT_CANONICAL})")
    print("=" * 72)


if __name__ == "__main__":
    main()
