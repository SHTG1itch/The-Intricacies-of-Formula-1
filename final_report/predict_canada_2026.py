"""Predict the 2026 Canadian Grand Prix using the walk-forward Plackett-Luce model.

This script is intentionally a one-race analogue to the full predict_2026.py pipeline.
Instead of running across the whole season, we freeze at race 5 (Canadian GP), feed the
PL the four completed results (Australia, China, Japan, Miami) and the live qualifying +
sprint session data for Canada, then produce a ranked finishing-order prediction.

Results are written to: data_organized/2026_canadian_gp_prediction.txt

Run from the project root:
    python final_report/predict_canada_2026.py

What's different from predict_2026.py:
  - we don't retrain models; we load the pre-trained artifacts from models_2026/
  - 2026 is treated as a NEW ERA year (active aerodynamics + all-new 1.6L hybrid PUs)
    even though NEW_ERA_YEARS in walk_forward_model.py only lists {2022}. that constant
    controls the LOSO evaluation loop - here we override locally with is_new_era=True.
  - because 2026_features.csv is empty (feature_engineering.py hasn't processed 2026 yet),
    we build the feature vectors manually from:
      a) live-fetched Q and S session results via FastF1 API
      b) historical Canadian GP rows in ml_training_features.csv + ml_2025_features.csv
      c) 2026 R1-R4 results (also live-fetched) for rolling-form features
    the FP / quali-time features (fp1_pace_delta, quali_gap_to_pole, etc.) are left NaN
    and filled by the training-data imputer, which is the correct thing to do when you
    don't have access to exact lap times.
"""

import os
import sys
import warnings
import logging
import datetime

import fastf1
import joblib
import numpy as np
import pandas as pd

from scipy.special import logsumexp
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# make sure the sibling modules (walk_forward_model, mach_learning, rmt_denoising) are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pull in the pieces we need from the shared infrastructure
# FEATURE_COLS is the 54-column feature list that both the ensemble and the imputer expect
# ensemble_predict wraps the four models + BMA weights into a single ranked score
# _rank_normalize turns arbitrary score vectors into [0,1] rank-normalized values
from mach_learning import FEATURE_COLS, POINTS_MAP, ensemble_predict, _rank_normalize

# the PL layer and its supporting helpers from walk_forward_model
from walk_forward_model import (
    PlackettLuceEstimator,
    compute_blend_alpha,
    ensemble_score_to_pl_prior,
    build_rmt_historical_priors,
    _rank_normalize_1d,
    BMA_WEIGHTS,
    RF_TEMPERATURE,
    SIGMA2_BASE,
    DECAY_LAMBDA,
    ALPHA_CAP,
    WINNER_WEIGHT,
    QUAL_PL_WEIGHT,
    TAIL_WEIGHT,
    RMT_SCORE_BLEND,
    N_RECENT_RACES,
    PRIOR_SIGMA_NEW_ERA,
    SIGMA2_BASE_NEW_ERA,
)

from rmt_denoising import rmt_denoise_psd   # Marchenko-Pastur driver-cluster regularisation

# ── paths ─────────────────────────────────────────────────────────────────────

DATA_DIR     = "data_organized"
MODEL_DIR    = os.path.join(DATA_DIR, "models_2026")   # pre-trained models from predict_2026.py
CACHE_DIR    = "discovery_raw_data"                    # FastF1 disk cache

TRAIN_FILE   = os.path.join(DATA_DIR, "ml_training_features.csv")  # 2022-2024 training set
FEAT_2025    = os.path.join(DATA_DIR, "ml_2025_features.csv")       # 2025 features
OUT_FILE     = os.path.join(DATA_DIR, "2026_canadian_gp_prediction.txt")

# ── event constants ───────────────────────────────────────────────────────────

PREDICT_YEAR = 2026
CANADA_ROUND = 5       # FastF1 schedule assigns Round 5 to the Canadian GP
EVENT_NAME   = "Canadian Grand Prix"
CIRCUIT_NAME = "Circuit Gilles Villeneuve"

# 2026 is a new-regulation era: active aerodynamics + completely redesigned hybrid PUs.
# the ensemble was trained on 2022-2025 data where neither feature existed, so the base
# predictions are less calibrated than they would be mid-season in a stable-rules year.
# we signal this to compute_blend_alpha by passing is_new_era=True so it uses the
# higher SIGMA2_BASE_NEW_ERA (4.0 vs 2.0) - meaning PL can override the stale base
# predictions more aggressively once real race data arrives.
IS_NEW_ERA   = True    # override locally; walk_forward_model.NEW_ERA_YEARS only covers LOSO eval

# ── circuit characteristics (Canadian GP, permanent values) ────────────────────

# these are the static circuit lookup values feature_engineering.py would compute
# from historical F1 data. they don't change from year to year.
# overtaking_score: 0.45 - Gilles Villeneuve has good DRS zones but tight sections limit passing
# sc_rate: 0.45 - Montreal consistently produces safety cars (walls, chicanes)
# is_street: 0 - technically a purpose-built semi-permanent island circuit, not a street circuit
CIRCUIT_OVERTAKING = 0.45
CIRCUIT_SC_RATE    = 0.45
CIRCUIT_IS_STREET  = 0

# car_regulation_era: 1 = ground-effect era (2022+). this helps models distinguish
# pre-2022 aerodynamic characteristics from the post-2022 regulation reset.
# 2026's active aero is technically another reset but we keep it at 1 because
# the ground-effect era flag captures the most relevant historical boundary.
CAR_REG_ERA  = 1

# ── circuit history note ──────────────────────────────────────────────────────
#
# No manual per-driver overrides are needed. compute_full_circuit_history()
# (step 3 in the pipeline below) reads actual finish_position values from both
# ml_training_features.csv and ml_2025_features.csv and computes accurate
# Canadian GP statistics for every driver in the field.
#
# Key values produced by that function (based on 2018/2019/2022-2025 data range
# — 2020/2021 were COVID-cancelled, pre-2018 data is not in our CSVs):
#
#   ANT: 1 Canada start (2025: P3) → experience=1, avg_finish=3.0.
#        Relatively inexperienced here vs the field average of 3-4 starts, but
#        his only result was a podium, so the feature data is decent.
#
#   RUS: 5 starts (2019:P16 / 2022:P4 / 2023:P19 / 2024:P3 / 2025:P1)
#        → experience=5, avg_finish=8.6.
#        The 2019 P16 (Williams backmarker) and 2023 P19 (penalty weekend)
#        drag his career average down. His 2025 win is the most recent signal.
#
#   VER: 6 starts (2018:P3 / 2019:P5 / 2022:P1 / 2023:P1 / 2024:P1 / 2025:P2)
#        → experience=6, avg_finish=2.17. Dominant specialist at this track.
#
#   HAM: 6 starts (2018:P5 / 2019:P1 / 2022:P3 / 2023:P3 / 2024:P4 / 2025:P6)
#        → experience=6, avg_finish=3.67. Consistent Canada podium contender.
#
# The CIRCUIT_PL_PRIOR_BOOST dict below handles the residual specialist
# corrections that the low-importance circuit features (3.3% total) can't convey.

# ── PL prior circuit-specialist correction ────────────────────────────────────
#
# Why this is needed: circuit history features have only 3.3% combined importance.
# Even with accurate avg_finish/experience values, the feature space can't
# adequately convey circuit-specialist knowledge against the 35% weight on
# season form. This direct prior correction fills that gap.
#
# Values are in PL log-quality units (range ≈ ±1.5 from last to first in a 22-
# driver field). ±0.20 ≈ ±1 position in the middle of the grid.
#
# RUS +0.20: "most recent winner" advantage. His career avg (8.6) is dragged
#   down by the 2019 Williams and 2023 penalty weekend; his last result here was
#   a pole+win and his qualifying was again P1 for 2026. Modest recency boost.
#
# ANT -0.10: only 1 Canada start (P3 in 2025). The P3 is reflected accurately
#   in his circuit_driver_avg_finish=3.0 feature, but 1 race of experience is
#   still thin compared to the field average of 3-4 starts. Small penalty for
#   the experience gap; NOT a large penalty since he proved competitive.
#
# The PL MAP optimization will partially absorb these corrections, modulated by
# the strength of the race data (n_obs=4, prior_sigma=3.0 for new-era years).
CIRCUIT_PL_PRIOR_BOOST = {
    "RUS": +0.20,   # most recent Canada winner (2025 pole+win); recency edge
    "ANT": -0.10,   # 1 Canada start vs field avg 3-4; inexperienced but showed P3 pace
}

# ── driver ELO ratings (career-aggregate, from F1 ELO tracking website) ──────
#
# Source: "Current F1 Driver ELO Rankings — Active Grid" HTML
# (same source used in generate_2026_predictions.py for the full-season WDC model)
# Higher ELO = stronger driver across their FULL career history, not just 2026.
#
# These are intentionally NOT recomputed from our training CSVs because:
#   1. the website maintains a proper running ELO updated after every race,
#      with all historical F1 data behind it — far more complete than our 2022+ CSVs
#   2. career-aggregate ELO gives genuine multi-season driver-quality context that
#      complements the within-2026 PL estimates (which only see 4 races so far)
#   3. using the same source as generate_2026_predictions.py keeps the two models
#      consistent and avoids contradicting the full-season WDC projection
#
# ANT's rating (1836) is the lowest among front-runners, which is correct:
# his career record before 2026 was limited, and 4 exceptional races in one season
# should NOT inflate a career-aggregate ELO. the website ratings correctly reflect
# that he is still an unproven quantity over a full season and multiple circuits.
#
# Ratings will be centred at 0 for the 22-driver field and divided by 400 (the
# standard ELO logistic scale factor) before blending into the PL prior.
ELO_RATINGS_2026 = {
    "LEC": 2208,    # Leclerc  – highest active-grid rating; long dominant career
    "VER": 2136,    # Verstappen – 4× champion; dominant 2022-2024
    "HAM": 2058,    # Hamilton – 7× champion
    "RUS": 2033,    # Russell  – rising; consistent podiums since 2022
    "ANT": 1836,    # Antonelli – rookie; strong 2026 but limited prior career
    "ALO": 1832,    # Alonso   – veteran; 2× champion; long career behind him
    "SAI": 1795,    # Sainz    – multiple race wins
    "NOR": 1761,    # Norris   – consistent frontrunner
    "ALB": 1730,    # Albon    – improved mid-fielder
    "PER": 1721,    # Perez    – 2× race winner; Red Bull tenure
    "PIA": 1697,    # Piastri  – strong 2024-2025 form
    "BOT": 1690,    # Bottas   – veteran; 10× race winner historically
    "STR": 1600,    # Stroll   – consistent midfield
    "GAS": 1560,    # Gasly    – race winner (Monza 2021)
    "HAD": 1557,    # Hadjar   – rookie rating
    "HUL": 1549,    # Hülkenberg – veteran; multiple seasons without wins
    "BEA": 1546,    # Bearman  – rookie/sophomore
    "LAW": 1531,    # Lawson   – limited career starts
    "BOR": 1502,    # Nico Bortoleto – rookie (Audi)
    "LIN": 1497,    # Gabriel Bortoleto – rookie (Racing Bulls)
    "COL": 1463,    # Colapinto – limited prior career
    "OCO": 1459,    # Ocon     – lowest active-grid rating
}

# ── full driver names (for the output file) ───────────────────────────────────

FULL_NAMES = {
    "VER": "Max Verstappen",       "NOR": "Lando Norris",
    "LEC": "Charles Leclerc",      "PIA": "Oscar Piastri",
    "SAI": "Carlos Sainz",         "RUS": "George Russell",
    "HAM": "Lewis Hamilton",       "ALO": "Fernando Alonso",
    "STR": "Lance Stroll",         "GAS": "Pierre Gasly",
    "OCO": "Esteban Ocon",         "LAW": "Liam Lawson",
    "ALB": "Alexander Albon",      "BOT": "Valtteri Bottas",
    "HUL": "Nico Hülkenberg",      "BEA": "Oliver Bearman",
    "ANT": "Kimi Antonelli",       "HAD": "Isack Hadjar",
    "COL": "Franco Colapinto",     "LIN": "Gabriel Bortoleto",
    "BOR": "Nico Bortoleto",       "PER": "Sergio Perez",
}
# note: BOR = Nico Bortoleto (Audi), LIN = Gabriel Bortoleto (Racing Bulls)
# the abbreviations match what FastF1 returns in 2026

# ── 2026 team mappings (from R4 results, verified against R1-R3) ───────────────

# team memberships are stable across all 4 completed races - no mid-season swaps yet
TEAMS_2026 = {
    "ANT": "Mercedes",        "RUS": "Mercedes",
    "NOR": "McLaren",         "PIA": "McLaren",
    "VER": "Red Bull Racing", "HAD": "Red Bull Racing",
    "LEC": "Ferrari",         "HAM": "Ferrari",
    "SAI": "Williams",        "ALB": "Williams",
    "GAS": "Alpine",          "COL": "Alpine",
    "LIN": "Racing Bulls",    "LAW": "Racing Bulls",
    "ALO": "Aston Martin",    "STR": "Aston Martin",
    "HUL": "Audi",            "BOR": "Audi",
    "BEA": "Haas F1 Team",    "OCO": "Haas F1 Team",
    "PER": "Cadillac",        "BOT": "Cadillac",
}


# ── 2026 completed race results (fetched from FastF1, R1-R4) ─────────────────
# these are the actual race classifications from rounds 1-4 of 2026.
# we hard-code them here so the script can run without hitting the network for these
# particular sessions. the Canadian GP qualifying and sprint we DO fetch live because
# that's the whole point - the Q and S sessions are the new data this script adds.
# if you're running this and one of these results looks wrong, the FastF1 command to
# re-check is: fastf1.get_session(2026, rnd, 'R').results

RACE_RESULTS_2026 = {
    1: {  # Australian GP - RUS dominant, ANT 2nd
        "RUS": 1, "ANT": 2, "LEC": 3, "HAM": 4, "NOR": 5,
        "VER": 6, "BEA": 7, "LIN": 8, "BOR": 9, "GAS": 10,
        "OCO": 11, "ALB": 12, "LAW": 13, "COL": 14, "SAI": 15,
        "PER": 16, "STR": 17, "ALO": 18, "BOT": 19, "HAD": 20,
        "PIA": 21, "HUL": 22,
    },
    2: {  # Chinese GP - ANT wins again, RUS 2nd
        "ANT": 1, "RUS": 2, "HAM": 3, "LEC": 4, "BEA": 5,
        "GAS": 6, "LAW": 7, "HAD": 8, "SAI": 9, "COL": 10,
        "HUL": 11, "LIN": 12, "BOT": 13, "OCO": 14, "PER": 15,
        "VER": 16, "ALO": 17, "STR": 18, "PIA": 19, "NOR": 20,
        "BOR": 21, "ALB": 22,
    },
    3: {  # Japanese GP - ANT wins, PIA 2nd, McLaren rebounding
        "ANT": 1, "PIA": 2, "LEC": 3, "RUS": 4, "NOR": 5,
        "HAM": 6, "GAS": 7, "VER": 8, "LAW": 9, "OCO": 10,
        "HUL": 11, "HAD": 12, "BOR": 13, "LIN": 14, "SAI": 15,
        "COL": 16, "PER": 17, "ALO": 18, "BOT": 19, "ALB": 20,
        "STR": 21, "BEA": 22,
    },
    4: {  # Miami GP - ANT hat-trick, NOR 2nd, PIA 3rd
        "ANT": 1, "NOR": 2, "PIA": 3, "RUS": 4, "VER": 5,
        "HAM": 6, "COL": 7, "LEC": 8, "SAI": 9, "ALB": 10,
        "BEA": 11, "BOR": 12, "OCO": 13, "LIN": 14, "ALO": 15,
        "PER": 16, "STR": 17, "BOT": 18, "HUL": 19, "LAW": 20,
        "GAS": 21, "HAD": 22,
    },
}

# 2026 R1-R4 qualifying grids (used as soft PL observations for past qualifying history)
# these are the official qualifying positions, fetched from FastF1 Q sessions R1-R4
QUAL_GRIDS_2026 = {
    1: {"RUS": 1, "ANT": 2, "HAD": 3, "LEC": 4, "PIA": 5, "NOR": 6,
        "HAM": 7, "VER": 8, "OCO": 9, "BOR": 10, "BEA": 11, "GAS": 12,
        "SAI": 13, "HUL": 14, "LIN": 15, "STR": 16, "COL": 17, "ALB": 18,
        "ALO": 19, "LAW": 20, "BOT": 21, "PER": 22},
    2: {"ANT": 1, "RUS": 2, "HAM": 3, "LEC": 4, "PIA": 5, "NOR": 6,
        "VER": 7, "HAD": 8, "BEA": 9, "GAS": 10, "OCO": 11, "SAI": 12,
        "LIN": 13, "HUL": 14, "COL": 15, "BOR": 16, "ALO": 17, "STR": 18,
        "ALB": 19, "LAW": 20, "BOT": 21, "PER": 22},
    3: {"ANT": 1, "RUS": 2, "PIA": 3, "LEC": 4, "NOR": 5, "HAM": 6,
        "VER": 7, "LAW": 8, "GAS": 9, "OCO": 10, "HUL": 11, "HAD": 12,
        "LIN": 13, "SAI": 14, "BOR": 15, "COL": 16, "BEA": 17, "ALB": 18,
        "STR": 19, "ALO": 20, "BOT": 21, "PER": 22},
    4: {"ANT": 1, "VER": 2, "LEC": 3, "NOR": 4, "RUS": 5, "HAM": 6,
        "PIA": 7, "SAI": 8, "COL": 9, "BEA": 10, "OCO": 11, "HAD": 12,
        "LIN": 13, "GAS": 14, "HUL": 15, "BOT": 16, "LAW": 17, "ALO": 18,
        "ALB": 19, "BOR": 20, "STR": 21, "PER": 22},
}

# Canadian GP R5: qualifying and sprint results (live-fetched from FastF1)
# Q session determines the race starting grid. RUS took pole - first non-Mercedes front row
# since Melbourne. ANT qualified P2 despite being 93 points clear in the championship.
# note: SQ (sprint qualifying) results were not available via the API at time of running.
CANADA_Q_RESULT = {
    "RUS": 1, "ANT": 2, "NOR": 3, "PIA": 4, "HAM": 5, "VER": 6,
    "HAD": 7, "LEC": 8, "LIN": 9, "COL": 10, "HUL": 11, "LAW": 12,
    "BOR": 13, "GAS": 14, "SAI": 15, "BEA": 16, "OCO": 17, "ALB": 18,
    "ALO": 19, "PER": 20, "STR": 21, "BOT": 22,
}

# Sprint result: RUS won ahead of NOR. ANT finished 3rd - still dominant but below qualifying.
# this is an important signal for PL: even with a sprint win the sprint is NOR's territory here.
# sprint positions are fed in as part of the feature vector (sprint_pos column) AND
# as an additional PL observation at QUAL_PL_WEIGHT weight, because the sprint is a real
# competitive ranking just like qualifying is.
CANADA_S_RESULT = {
    "RUS": 1, "NOR": 2, "ANT": 3, "PIA": 4, "LEC": 5, "HAM": 6,
    "VER": 7, "LIN": 8, "COL": 9, "SAI": 10, "LAW": 11, "BOR": 12,
    "OCO": 13, "PER": 14, "HUL": 15, "STR": 16, "BOT": 17, "BEA": 18,
    "ALB": 19, "GAS": 20, "HAD": 21, "ALO": 22,
}


# ── step 1: load pre-trained models from models_2026/ ─────────────────────────

def load_models():
    # load the four models + imputer that predict_2026.py trained and saved.
    # if any required artifact is missing, print a clear instruction and exit.
    # we don't retrain here - that would take 10-15 minutes and the user already has
    # the models from running predict_2026.py. if you don't have them, run that script first.

    required = {
        "xgb":      os.path.join(MODEL_DIR, "xgb_model.joblib"),
        "ranker":   os.path.join(MODEL_DIR, "xgb_ranker.joblib"),
        "rf_win":   os.path.join(MODEL_DIR, "rf_win_model.joblib"),
        "rf_top10": os.path.join(MODEL_DIR, "rf_top10_model.joblib"),
        "imputer":  os.path.join(MODEL_DIR, "feature_imputer.joblib"),
    }

    missing = [k for k, p in required.items() if not os.path.exists(p)]
    if missing:
        print(f"\n[!] Missing model artifacts: {missing}")
        print(f"    Run  python final_report/predict_2026.py  to train and save them.")
        sys.exit(1)

    xgb_model   = joblib.load(required["xgb"])
    xgb_ranker  = joblib.load(required["ranker"])
    rf_win      = joblib.load(required["rf_win"])
    rf_top10    = joblib.load(required["rf_top10"])
    imputer     = joblib.load(required["imputer"])

    et_path     = os.path.join(MODEL_DIR, "et_model.joblib")
    et_model    = joblib.load(et_path) if os.path.exists(et_path) else None

    bma_path    = os.path.join(MODEL_DIR, "bma_weights.joblib")
    bma_weights = joblib.load(bma_path) if os.path.exists(bma_path) else BMA_WEIGHTS

    cal_path    = os.path.join(MODEL_DIR, "cal_temperature.joblib")
    cal_temp    = joblib.load(cal_path) if os.path.exists(cal_path) else RF_TEMPERATURE

    print(f"  ✓ Models loaded from {MODEL_DIR}/")
    return xgb_model, xgb_ranker, rf_win, rf_top10, et_model, imputer, bma_weights, cal_temp


# ── step 2: compute 2026 season rolling stats for all drivers ─────────────────

def compute_rolling_stats_2026():
    # build per-driver rolling form features from the 4 completed 2026 races.
    # the feature engineering pipeline computes a moving window across seasons -
    # we can't replicate that exactly here without running feature_engineering.py,
    # so we approximate by using R1-R4 for the 3-race and 4-race windows and
    # noting that 10-race stats are effectively the same as 5-race stats for round 5.
    # the circuit-specific stats (rolling_avg_finish_3, rolling_points_5, etc.) will
    # be combined with historical circuit data loaded from the training CSVs below.

    drivers = sorted(CANADA_Q_RESULT.keys())   # all 22 drivers in the Canadian GP field

    # collect results in a more convenient shape: {driver: [pos_r1, pos_r2, pos_r3, pos_r4]}
    # a driver who didn't finish a race (DNF) keeps their classified position.
    # in 2026 R1-R4 all 22 drivers appear to have been classified - no obvious DNFs in the data.
    results_by_driver = {d: [] for d in drivers}
    for rnd in sorted(RACE_RESULTS_2026.keys()):
        for d in drivers:
            pos = RACE_RESULTS_2026[rnd].get(d, 20.0)   # default 20 if driver missed a race
            results_by_driver[d].append(float(pos))

    # compute per-driver season statistics
    stats = {}
    for d in drivers:
        pos_arr = np.array(results_by_driver[d])   # [r1, r2, r3, r4]
        n_races = len(pos_arr)

        # rolling averages: over last 3 and last 4-5 races
        avg3  = float(np.mean(pos_arr[-3:]))  if n_races >= 3 else float(np.mean(pos_arr))
        avg5  = float(np.mean(pos_arr))        # we only have 4 races - best approximation
        avg10 = avg5                            # same reasoning - use what we have

        # win, podium, points in last 5 races (= last 4 available)
        wins5    = int(np.sum(pos_arr <= 1))
        pods5    = int(np.sum(pos_arr <= 3))
        pts5     = sum(POINTS_MAP.get(int(p), 0) for p in pos_arr)
        pts10    = pts5    # same - only 4 races available

        win_rate5  = float(wins5 / n_races)
        pod_rate5  = float(pods5 / n_races)
        dnf_rate5  = 0.0   # we have no DNF flag in the raw results dict above

        # season aggregate stats (same thing as the 4-race window right now)
        season_pts_pct   = float(pts5 / (n_races * 25.0))
        season_avg_fin   = avg5
        season_wins_pct  = win_rate5

        stats[d] = {
            "rolling_avg_finish_3":  avg3,
            "rolling_avg_finish_5":  avg5,
            "rolling_avg_finish_10": avg10,
            "rolling_dnf_rate_5":    dnf_rate5,
            "rolling_points_5":      float(pts5),
            "rolling_points_10":     float(pts10),
            "rolling_win_rate_5":    win_rate5,
            "driver_podium_rate_5":  pod_rate5,
            "season_pts_pct":        season_pts_pct,
            "season_avg_finish":     season_avg_fin,
            "season_wins_pct":       season_wins_pct,
        }

    return stats


def compute_team_rolling_stats(rolling_stats):
    # compute team-level rolling averages to fill in team_rolling_avg_5 and team_rolling_avg_3.
    # both drivers' recent finishes are averaged per team. for teams where one driver
    # had a much better season (ANT 93pts vs RUS 67pts in Mercedes), the team average
    # will still be competitive because the top driver pulls it down.

    # group drivers by team
    team_to_drivers = {}
    for d, t in TEAMS_2026.items():
        if d in rolling_stats:
            team_to_drivers.setdefault(t, []).append(d)

    team_avg5 = {}
    team_avg3 = {}
    for team, dvrs in team_to_drivers.items():
        a5 = float(np.mean([rolling_stats[d]["rolling_avg_finish_5"]  for d in dvrs]))
        a3 = float(np.mean([rolling_stats[d]["rolling_avg_finish_3"] for d in dvrs]))
        team_avg5[team] = a5
        team_avg3[team] = a3

    return team_avg5, team_avg3


def compute_championship_rank_pct():
    # compute each driver's points tally after R1-R4 and normalise their rank to [0,1].
    # 0 = championship leader, 1 = last place in standings.
    # this feature tells the model who's driving the fastest car package mid-season.

    pts = {}
    for d in CANADA_Q_RESULT.keys():
        total = 0
        for rnd_res in RACE_RESULTS_2026.values():
            pos = rnd_res.get(d, 0)
            total += POINTS_MAP.get(int(pos), 0)
        pts[d] = total

    # sort by points descending to get championship ranks
    sorted_drivers = sorted(pts.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_drivers)
    rank_pct = {}
    for i, (d, _) in enumerate(sorted_drivers):
        rank_pct[d] = float(i) / max(n - 1, 1)   # 0 = champion, 1 = last

    return rank_pct


# ── driver ELO normaliser ─────────────────────────────────────────────────────

def get_elo_for_pl_prior():
    # normalise the website ELO_RATINGS_2026 for blending into the PL prior.
    #
    # the raw website ratings (1459–2208) need two transformations before they
    # can be blended with prior_mu_arr (which lives in PL log-quality space):
    #
    #   1. zero-mean: subtract the field mean so the blend is additive-neutral
    #      (i.e. doesn't inflate or deflate the overall prior_mu_arr magnitude).
    #
    #   2. divide by 400: the standard ELO logistic scale factor. a 400-point
    #      ELO gap means the stronger driver wins 90% of head-to-head matchups
    #      (10:1 odds). dividing by 400 converts to log-odds space, which is
    #      directly comparable to PL log-quality. the resulting range for this
    #      field is roughly ±1.2 (LEC ≈ +1.23, OCO ≈ −0.64), which matches the
    #      ≈ ±1.5 span of the ensemble-derived prior_mu_arr.
    #
    # the std-rescaling used in the previous computed-ELO approach is NOT needed
    # here because the /400 already puts these on a comparable scale.
    #
    # default rating for any driver not in ELO_RATINGS_2026: 1600 (field median).
    # in the 2026 field all 22 drivers have entries so the default is never hit.

    field   = set(CANADA_Q_RESULT.keys())
    ratings = {d: float(ELO_RATINGS_2026.get(d, 1600)) for d in field}
    mu      = float(np.mean(list(ratings.values())))
    return {d: (v - mu) / 400.0 for d, v in ratings.items()}


# ── step 3: build circuit history features from training data ─────────────────

def compute_full_circuit_history():
    # compute per-driver Canadian GP circuit history from actual race results.
    #
    # the old approach read pre-race feature rows from training CSVs, where
    # circuit_driver_avg_finish is computed from PRIOR races only. that left
    # circuit_driver_avg_finish=NaN for Antonelli (first-timer in 2025) because
    # there were no prior Canada races to average — even though he finished P3
    # in 2025. this function reads actual finish_position values instead, so
    # every driver who has raced at Canada gets a real circuit average.
    #
    # experience = number of Canada appearances in our data range (no +1 needed:
    # all rows in the training CSVs represent pre-2026 races, so the count of
    # rows directly equals the experience the driver brings INTO the 2026 race).
    #
    # circuit_team_avg_finish and circuit_specialist_delta: these depend on
    # knowing which team each driver raced for in each historical season. since
    # team lineups change (Hamilton left Mercedes for Ferrari), we take these
    # from the precomputed values in the training-data feature rows (last row
    # per driver). they remain correct as long as the team label in the training
    # data reflects the team AT THE TIME of the historical race.

    # ── part A: pull precomputed team-avg and specialist-delta ────────────────
    # these two features are computed from team-level history and don't need to
    # be recomputed from scratch; reading the last training-data row is fine.
    base_vals = {}
    base_cols = ["circuit_team_avg_finish", "circuit_specialist_delta"]
    for path in [TRAIN_FILE, FEAT_2025]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        mask = df["EventName"].str.contains("Canadian", na=False)
        avail = [c for c in base_cols if c in df.columns]
        if not avail:
            continue
        df_can = df[mask].sort_values("Year") if "Year" in df.columns else df[mask]
        last   = df_can.groupby("Driver")[avail].last()
        for drv, row in last.iterrows():
            entry = base_vals.setdefault(str(drv), {})
            for c in avail:
                val = row.get(c, np.nan)
                entry[c] = float(val) if pd.notna(val) else np.nan

    # ── part B: read actual finish positions from all Canadian GP rows ────────
    # each row in the training CSVs represents one (driver, year) appearance at
    # Canada. collecting all of them per driver gives the full Canada race history
    # with real finish positions (not pre-race estimated values).
    driver_races = {}   # {abbr: [(finish_position, grid_position_or_nan)]}

    for path in [TRAIN_FILE, FEAT_2025]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        mask = (df["EventName"].str.contains("Canadian", na=False) &
                df["finish_position"].notna())
        for _, row in df[mask].iterrows():
            drv = str(row.get("Driver", "")).strip()
            if not drv:
                continue
            fin    = float(row["finish_position"])
            grd_v  = row.get("grid_position", np.nan)
            grd    = float(grd_v) if pd.notna(grd_v) else np.nan
            driver_races.setdefault(drv, []).append((fin, grd))

    # ── part C: merge actual-race stats with precomputed team features ────────
    all_drvs = set(base_vals.keys()) | set(driver_races.keys())
    result   = {}

    for drv in all_drvs:
        entry = dict(base_vals.get(drv, {}))   # start with team_avg, specialist_delta

        if drv in driver_races:
            races    = driver_races[drv]
            finishes = [f for f, _ in races]
            gains    = [g - f for f, g in races if not np.isnan(g)]

            # use actual race count as experience (all rows are pre-2026 races)
            entry["circuit_experience"]              = float(len(finishes))
            # true average over all Canada visits in our data range
            entry["circuit_driver_avg_finish"]       = float(np.mean(finishes))
            # positions gained from grid to finish (positive = moved up)
            entry["driver_circuit_positions_gained"] = (float(np.mean(gains))
                                                         if gains else np.nan)

            # recompute specialist_delta with the updated (real) driver average;
            # the precomputed team_avg is used as-is (it was computed historically)
            t_avg = float(entry.get("circuit_team_avg_finish", np.nan))
            d_avg = float(np.mean(finishes))
            if not np.isnan(t_avg):
                entry["circuit_specialist_delta"] = d_avg - t_avg

        result[drv] = entry

    return result   # {driver: {col: value, ...}}


# ── step 4: determine is_number_one_driver flags ──────────────────────────────

def compute_number_one_flags():
    # within each team, the driver with more season points is tagged as the "number one".
    # in practice Mercedes is prioritising ANT (93pts vs RUS 67pts), Ferrari is HAM vs LEC
    # after the team switch, McLaren is NOR vs PIA with NOR marginally ahead in points.
    # this is an imperfect proxy but it captures the basic priority-driver reality.

    # first compute season pts per driver after R1-R4
    pts = {}
    for d in CANADA_Q_RESULT.keys():
        total = 0
        for rnd_res in RACE_RESULTS_2026.values():
            pos = rnd_res.get(d, 0)
            total += POINTS_MAP.get(int(pos), 0)
        pts[d] = total

    # group by team and pick the higher-points driver as #1
    team_to_drivers = {}
    for d, t in TEAMS_2026.items():
        team_to_drivers.setdefault(t, []).append(d)

    is_number_one = {}
    for team, dvrs in team_to_drivers.items():
        # driver with most points is "number one" for that team
        top = max(dvrs, key=lambda d: pts.get(d, 0))
        for d in dvrs:
            is_number_one[d] = 1.0 if d == top else 0.0

    return is_number_one


# ── step 5: assemble the 54-column feature matrix ────────────────────────────

def build_feature_matrix(imputer):
    # assemble one feature row per driver for the 2026 Canadian GP.
    # the columns exactly match FEATURE_COLS (54 features) as required by the models.
    #
    # what we can populate directly:
    #   - qualifying/sprint positions (from the live session data above)
    #   - rolling form features (from R1-R4 results)
    #   - circuit history features (from training CSVs)
    #   - static circuit characteristics
    #   - season stats (championship rank, win rate, etc.)
    #
    # what we leave as NaN (imputer fills with training-data medians):
    #   - FP1/FP2/FP3 pace deltas (sprint weekend: FP2 and FP3 don't exist)
    #   - quali_gap_to_pole / quali_teammate_gap / quali_gap_pct (need raw lap times)
    #   - session weather (track_temp, wet flag) - unknown pre-race
    #   - sprint_quali_pos (SQ session data not available from FastF1 API at time of run)
    #
    # the imputer was fit on 2022-2025 training data. at Canadian GP, the median
    # fp2_pace_delta across all years is a reasonable stand-in when actual FP data
    # is unavailable. this is how the model would behave in the early days of a new season.

    print("  [2/7] Building feature matrix for 22 drivers...")

    rolling_stats          = compute_rolling_stats_2026()
    team_avg5, team_avg3   = compute_team_rolling_stats(rolling_stats)
    champ_rank_pct         = compute_championship_rank_pct()
    # compute_full_circuit_history() reads actual finish positions from both training
    # CSVs rather than pre-race feature values. this ensures drivers like ANT who
    # debuted at Canada in 2025 get their real circuit avg (3.0 from his P3) instead
    # of NaN → imputed median. see the function definition above for full details.
    circuit_hist           = compute_full_circuit_history()
    is_num1                = compute_number_one_flags()

    drivers = sorted(CANADA_Q_RESULT.keys())   # 22 drivers alphabetically
    rows    = []

    for d in drivers:
        gp       = float(CANADA_Q_RESULT.get(d, 15))    # race grid position from Q session
        sprint_p = float(CANADA_S_RESULT.get(d, 15))    # sprint race finish position
        team     = TEAMS_2026.get(d, "Unknown")
        rs       = rolling_stats.get(d, {})
        ch       = dict(circuit_hist.get(d, {}))    # copy so we don't mutate the original
        # no manual overrides needed: compute_full_circuit_history() already produced
        # accurate values from actual race results for all 22 drivers in the field.

        # ── grid / sprint features ────────────────────────────────────────────
        is_sprint_weekend = 1.0
        is_front_row      = 1.0 if gp <= 2 else 0.0
        grid_pos_pct      = (gp - 1.0) / 21.0   # normalised: P1 → 0.0, P22 → 1.0

        # grid × (1/overtaking) interaction: the harder it is to pass, the more
        # qualifying position matters. monaco (overtaking=0.05) → this is huge.
        # montreal (0.45) is intermediate - qualifying matters but isn't everything.
        grid_x_ot_inv = gp * (1.0 - CIRCUIT_OVERTAKING)

        # ── circuit history features ──────────────────────────────────────────
        cir_exp        = float(ch.get("circuit_experience",            np.nan))
        cir_drv_avg    = float(ch.get("circuit_driver_avg_finish",     np.nan))
        cir_team_avg   = float(ch.get("circuit_team_avg_finish",       np.nan))
        cir_pos_gained = float(ch.get("driver_circuit_positions_gained", np.nan))
        cir_spec_delta = float(ch.get("circuit_specialist_delta",      np.nan))

        # ── team rolling stats ────────────────────────────────────────────────
        t_avg5  = float(team_avg5.get(team, np.nan))
        t_avg3  = float(team_avg3.get(team, np.nan))
        t_mom   = float(t_avg5 - t_avg3) if not (np.isnan(t_avg5) or np.isnan(t_avg3)) else np.nan
        drv_vs_team = float(rs.get("rolling_avg_finish_5", np.nan) - t_avg5) \
                      if not np.isnan(t_avg5) else np.nan

        row = {
            # sprint / grid
            "IsSprintWeekend":    is_sprint_weekend,
            "grid_position":      gp,
            "sprint_quali_pos":   np.nan,   # SQ not available from API
            "sprint_pos":         sprint_p,
            "is_front_row":       is_front_row,
            "grid_position_pct":  grid_pos_pct,

            # free practice deltas - all NaN (sprint weekend has no FP2/FP3; FP1 lap times not loaded)
            "fp1_pace_delta":     np.nan,
            "fp2_pace_delta":     np.nan,
            "fp3_pace_delta":     np.nan,
            "fp2_s1_delta":       np.nan,
            "fp2_s2_delta":       np.nan,
            "fp2_s3_delta":       np.nan,
            "fp2_long_run_delta": np.nan,
            "fp2_consistency":    np.nan,
            "fp_pace_progression":np.nan,
            "fp3_long_run_delta": np.nan,

            # qualifying time deltas - set exactly for the pole sitter (factual: gap=0);
            # leave NaN for everyone else since we don't have exact lap times.
            # quali_gap_to_pole is the 15th most important feature (0.0158) so setting
            # it to 0.0 for RUS gives a small but real improvement over the imputed median.
            "quali_gap_to_pole":    0.0 if gp == 1 else np.nan,
            "quali_teammate_gap":   np.nan,
            "q3_appearance":        1.0 if gp <= 10 else 0.0,
            "quali_gap_pct":        0.0 if gp == 1 else np.nan,

            # session weather (unknown pre-race; imputer fills with training medians)
            "fp1_track_temp":  np.nan,
            "fp1_wet":         0.0,   # assume dry for Canada in June
            "fp2_track_temp":  np.nan,
            "fp2_wet":         0.0,
            "quali_track_temp":np.nan,
            "quali_air_temp":  np.nan,
            "quali_wet":       0.0,

            # rolling form features (from R1-R4 results)
            "rolling_avg_finish_3":  rs.get("rolling_avg_finish_3",  np.nan),
            "rolling_avg_finish_5":  rs.get("rolling_avg_finish_5",  np.nan),
            "rolling_dnf_rate_5":    rs.get("rolling_dnf_rate_5",    0.0),
            "rolling_points_5":      rs.get("rolling_points_5",      np.nan),
            "rolling_win_rate_5":    rs.get("rolling_win_rate_5",    np.nan),
            "driver_podium_rate_5":  rs.get("driver_podium_rate_5",  np.nan),
            "rolling_avg_finish_10": rs.get("rolling_avg_finish_10", np.nan),
            "rolling_points_10":     rs.get("rolling_points_10",     np.nan),
            "driver_vs_team_avg_delta": drv_vs_team,
            "team_rolling_avg_5":    t_avg5,
            "season_pts_pct":        rs.get("season_pts_pct",        np.nan),
            "season_avg_finish":     rs.get("season_avg_finish",     np.nan),
            "season_wins_pct":       rs.get("season_wins_pct",       np.nan),

            # circuit history features (from historical CSVs)
            "circuit_experience":              cir_exp,
            "circuit_driver_avg_finish":       cir_drv_avg,
            "circuit_team_avg_finish":         cir_team_avg,
            "driver_circuit_positions_gained": cir_pos_gained,

            # static circuit characteristics for Canada
            "circuit_overtaking_score": CIRCUIT_OVERTAKING,
            "circuit_sc_rate":          CIRCUIT_SC_RATE,
            "circuit_is_street":        float(CIRCUIT_IS_STREET),

            # accuracy-push features
            "car_regulation_era":     float(CAR_REG_ERA),
            "grid_x_overtaking_inv":  grid_x_ot_inv,
            "championship_rank_pct":  champ_rank_pct.get(d, 0.5),
            "is_number_one_driver":   is_num1.get(d, 0.0),
            "circuit_specialist_delta": cir_spec_delta,
            "team_momentum":          t_mom,
            "team_rolling_avg_3":     t_avg3,
        }

        # sanity check: we must have exactly FEATURE_COLS columns, in the right order
        rows.append({col: row.get(col, np.nan) for col in FEATURE_COLS})

    df_features = pd.DataFrame(rows, index=drivers)

    # apply the training-data imputer to fill NaN values with appropriate medians
    # this is the same imputer used in predict_2026.py - transform only, never refit
    X_raw = df_features[FEATURE_COLS].astype(float)
    X     = pd.DataFrame(imputer.transform(X_raw), columns=FEATURE_COLS, index=drivers)

    print(f"  ✓ Features ready: {X.shape[0]} drivers × {X.shape[1]} features")
    nan_before = int(X_raw.isnull().sum().sum())
    print(f"    ({nan_before} NaN values filled by imputer)")
    return X, drivers


# ── step 6: ensemble prediction ───────────────────────────────────────────────

def run_ensemble(xgb_model, xgb_ranker, rf_win, rf_top10, et_model,
                 imputer, bma_weights, cal_temp, X, drivers):
    # run the four-model BMA ensemble on the Canadian GP feature matrix.
    # this is the same prediction logic as walk_forward_evaluate's inner block:
    #   XGBReg predicts position (float, higher = worse)
    #   XGBRanker predicts relevance (float, higher = better) - note sign flip
    #   RF win classifier predicts win probability
    #   ExtraTrees predicts position (float, higher = worse)
    # ensemble_predict normalises each to [0,1] rank space and blends by BMA weights.

    print("  [3/7] Running 4-model BMA ensemble prediction...")

    n_driv = len(drivers)

    xgb_preds    = xgb_model.predict(X)
    ranker_preds = xgb_ranker.predict(X)
    et_preds     = et_model.predict(X) if et_model is not None else xgb_preds.copy()
    rf_proba     = rf_win.predict_proba(X)[:, 1]

    # construct the meta dataframe that ensemble_predict expects
    df_meta = pd.DataFrame({
        "Year":      PREDICT_YEAR,
        "Round":     CANADA_ROUND,
        "EventName": EVENT_NAME,
        "Driver":    drivers,
        "Team":      [TEAMS_2026.get(d, "Unknown") for d in drivers],
    })

    ens_df = ensemble_predict(
        xgb_preds, rf_proba, df_meta,
        ranker_raw_scores = ranker_preds,
        et_raw_pos        = et_preds,
        ensemble_weights  = bma_weights,
        X_features        = X,
        apply_bayesian_prior = True,
        cal_temperature   = cal_temp,
        meta_model        = None,
    )

    # reindex ens_df back to our `drivers` list order (ensemble_predict may sort internally)
    ens_score_by_driver = ens_df.set_index("Driver")["ensemble_score"].reindex(drivers)
    base_scores = ens_score_by_driver.values.astype(float)

    base_winner = drivers[int(np.argmin(base_scores))]
    print(f"  ✓ Base ensemble prediction: {base_winner} ({base_scores.min():.4f})")
    return base_scores, df_meta


# ── step 7: RMT driver-cluster regularisation ─────────────────────────────────

def apply_rmt_regularisation(base_scores, X, n_obs_so_far, season_unique_winners):
    # same approach as in walk_forward_evaluate (lines ~956-999 of walk_forward_model.py).
    # with 4 completed races and 4 unique winners (RUS won R1, ANT won R2-R4), the
    # entropy is 2/4 = 0.5. RMT blend is scaled by entropy/0.45, capped at 1.
    # so adaptive_blend = 0.15 × min(0.5/0.45, 1) = 0.15 × 1.0 = 0.15.
    # at that level the tier correction is moderate - it won't override individual
    # qualifying signals but does gently cluster same-car drivers together.

    n_driv     = len(base_scores)
    unique_win = len(season_unique_winners)
    entropy    = unique_win / max(n_obs_so_far, 1) if n_obs_so_far >= 3 else 0.0
    adaptive   = RMT_SCORE_BLEND * min(entropy / 0.45, 1.0)

    if adaptive <= 0.01 or n_driv < 5 or np.isnan(base_scores).any():
        return base_scores, adaptive, 0

    try:
        X_np    = X.values
        X_mean  = X_np.mean(axis=0)
        X_std2  = np.clip(X_np.std(axis=0), 1e-8, None)
        X_s     = (X_np - X_mean) / X_std2
        C_drv   = X_s @ X_s.T / X_np.shape[1]

        C_clean, n_sig, lam_rmt = rmt_denoise_psd(
            C_drv, T_eff=float(X_np.shape[1]), return_diagnostics=True
        )

        if n_sig > 0:
            eig_vals, eig_vecs = np.linalg.eigh(C_clean)
            sig_mask      = eig_vals > lam_rmt
            sig_vecs      = eig_vecs[:, sig_mask]
            scores_proj   = sig_vecs @ (sig_vecs.T @ base_scores)
            sig_frac      = float(n_sig) / max(float(n_driv), 1.0)
            final_blend   = adaptive * (0.5 + 0.5 * sig_frac)
            blended       = (1.0 - final_blend) * base_scores + final_blend * scores_proj
            return blended, adaptive, n_sig

    except Exception:
        pass   # if anything goes wrong, return original scores unchanged

    return base_scores, adaptive, 0


# ── step 8: Plackett-Luce update layer ────────────────────────────────────────

def run_pl_layer(base_scores, X, drivers, df_meta):
    # the heart of the walk-forward method: update the PL model with R1-R4 results
    # and blend its season-ability estimates into the base ensemble scores.
    # this is the same logic as lines 1001-1213 of walk_forward_evaluate but
    # extracted for a single race (Canadian GP, round 5, n_observed=4).

    print("  [4/7] Loading historical priors (Marchenko-Pastur denoising)...")

    # build the MP-denoised historical driver ability priors from the training CSV.
    # these give the PL a better starting point than random initialisation, especially
    # for drivers who don't appear in the 2026 race results yet (though at round 5
    # everyone has raced so the historical signal is somewhat moot here).
    df_train_parts = []
    for path in [TRAIN_FILE, FEAT_2025]:
        if os.path.exists(path):
            df_part = pd.read_csv(path)
            if "finish_position" in df_part.columns:
                df_train_parts.append(df_part.dropna(subset=["finish_position"]))

    if df_train_parts:
        df_train_hist = pd.concat(df_train_parts, ignore_index=True)
        hist_priors = build_rmt_historical_priors(df_train_hist, holdout_year=PREDICT_YEAR)
        n_hist = len(hist_priors)
        if hist_priors:
            best = max(hist_priors, key=hist_priors.get)
            print(f"  ✓ Historical priors: {n_hist} drivers "
                  f"(top: {best} θ={hist_priors[best]:.2f})")
    else:
        hist_priors = {}
        print("  [!] No training data found for historical priors")

    n_driv = len(drivers)

    # ── convert ensemble scores to PL prior log-quality ──────────────────────
    prior_mu_arr = ensemble_score_to_pl_prior(base_scores, n_driv)

    # blend historical priors into current prior for early-season stability.
    # n_obs=4: hist_warmup_weight = max(0, 0.25 × (1 - 4/8)) = 0.125 (12.5%)
    n_obs_for_prior   = 4   # we've seen 4 completed races
    hist_warmup_weight = max(0.0, 0.25 * (1.0 - n_obs_for_prior / 8.0))
    if hist_warmup_weight > 0.01 and hist_priors:
        for i, d in enumerate(drivers):
            if d in hist_priors:
                prior_mu_arr[i] = ((1.0 - hist_warmup_weight) * prior_mu_arr[i]
                                   + hist_warmup_weight * hist_priors[d])

    # ── driver ELO blend (career-aggregate, from F1 ELO tracking website) ───────
    # website ELO gives a career-aggregate quality estimate that is NOT inflated
    # by ANT's exceptional 4-race 2026 sample. LEC (2208) and VER (2136) sit at
    # the top because their full F1 careers are the evidence, not just 2026.
    # ANT (1836) is correctly mid-pack relative to the veterans: his career record
    # before 2026 was limited, and 4 wins in one season shouldn't override that.
    #
    # get_elo_for_pl_prior() normalises by field mean and divides by 400 (the ELO
    # logistic scale factor), putting the ratings in the same order of magnitude
    # as prior_mu_arr. no std-rescaling is needed because the /400 already aligns
    # the ELO spread with PL log-quality space (both span roughly ±1.0-1.5 peak-to-trough).
    #
    # weight: 15% — meaningful enough to pull experienced drivers like LEC/VER/HAM
    # closer to their true career level, not so strong that it overrides the 4-race
    # 2026 PL signal that correctly says ANT has been the fastest car this season.
    print("  [4b/7] Loading driver ELO ratings (career-aggregate, website source)...")
    elo_ratings  = get_elo_for_pl_prior()
    elo_arr      = np.array([elo_ratings.get(d, 0.0) for d in drivers])

    ELO_BLEND    = 0.15   # 15% website ELO, 85% ensemble + historical priors
    prior_mu_arr = (1.0 - ELO_BLEND) * prior_mu_arr + ELO_BLEND * elo_arr

    if elo_ratings:
        top_elo = max(elo_ratings, key=elo_ratings.get)
        bot_elo = min(elo_ratings, key=elo_ratings.get)
        print(f"  ✓ ELO blend: top={top_elo} ({elo_ratings[top_elo]:+.3f}), "
              f"bottom={bot_elo} ({elo_ratings[bot_elo]:+.3f}), weight={ELO_BLEND}")

    prior_means = {d: float(prior_mu_arr[i]) for i, d in enumerate(drivers)}

    # ── circuit-specialist PL prior correction ────────────────────────────────
    # the base ensemble weights circuit history at only 3.3% total importance,
    # which means even very large changes to circuit_driver_avg_finish barely move
    # the base scores. to give circuit-specific knowledge genuine weight in the
    # prediction, we apply a direct correction to prior_means in PL log-quality
    # space. the PL will use this corrected prior when estimating θ* via MAP.
    # see CIRCUIT_PL_PRIOR_BOOST above for the rationale and scale explanation.
    for d, boost in CIRCUIT_PL_PRIOR_BOOST.items():
        if d in prior_means:
            prior_means[d] = prior_means[d] + boost
            print(f"    circuit prior correction: {d} {'+' if boost >= 0 else ''}{boost:.2f} "
                  f"→ prior_mu={prior_means[d]:.3f}")

    # ── fit the season-long PL from R1-R4 ─────────────────────────────────────
    # 2026 is new-era, so we skip qualifying observations - early qualifying in a
    # new-regulation year reflects car setup rather than race pace, and feeding those
    # signals could make PL over-bullish on drivers who happened to qualify well.
    # instead we use past race results only (R1-R4). this is the same logic as
    # `if is_new_era: past_quals = None` in walk_forward_evaluate.
    pl_prior_sigma = PRIOR_SIGMA_NEW_ERA   # 3.0 - looser prior for regulation-change year

    pl_est = PlackettLuceEstimator(
        prior_sigma   = pl_prior_sigma,
        decay_lambda  = DECAY_LAMBDA,
        winner_weight = WINNER_WEIGHT,
        tail_weight   = TAIL_WEIGHT,
    )
    past_races = [RACE_RESULTS_2026[r] for r in sorted(RACE_RESULTS_2026.keys())]
    pl_est.fit(
        past_races, prior_means,
        current_race_num = CANADA_ROUND,   # round 5
        qualifying_results = None,          # skip quali - new era (see comment above)
    )

    n_obs = len(past_races)
    # find the driver who finished P1 in each race
    season_winners_seen = set()
    for r_results in RACE_RESULTS_2026.values():
        for drv, pos in r_results.items():
            if pos == 1:
                season_winners_seen.add(drv)
    # in 2026 R1-R4: RUS won R1, ANT won R2/R3/R4 → {RUS, ANT}
    winner_entropy = len(season_winners_seen) / max(n_obs, 1)   # 2/4 = 0.5

    print(f"  ✓ Season PL fitted: n_obs={n_obs}, unique_winners={len(season_winners_seen)}")
    print(f"    entropy={winner_entropy:.3f}, PL leader: "
          f"{max(pl_est.theta_, key=pl_est.theta_.get) if pl_est.theta_ else 'N/A'}")

    # ── recent-window PL (last N_RECENT_RACES races only) ─────────────────────
    # with only 4 races completed the full-season and recent PL will overlap heavily.
    # but we still run both to detect any mid-season power shift between R1 and R4.
    # decay_lambda=0.0 in the recent PL means R2, R3, R4 all count equally (no decay).
    past_recent = past_races[-N_RECENT_RACES:]  # last 4 (or fewer) races

    pl_recent = PlackettLuceEstimator(
        prior_sigma   = pl_prior_sigma * 1.5,  # weaker prior for recent window
        decay_lambda  = 0.0,                   # no decay - all recent races equal weight
        winner_weight = WINNER_WEIGHT,
        tail_weight   = TAIL_WEIGHT,
    )
    pl_recent.fit(past_recent, prior_means, current_race_num=CANADA_ROUND,
                  qualifying_results=None)

    # ── team-level PL ──────────────────────────────────────────────────────────
    # builds team best-finish-per-race and fits a team PL alongside the driver PL.
    # the blended θ_race includes 25% team signal, which means if Mercedes' car
    # was consistently fast (ANT+RUS both scoring), that boosts both their predictions
    # slightly compared to a pure individual-driver model.

    driver_to_team = {d: TEAMS_2026.get(d, "Unknown") for d in drivers}
    teams_present  = sorted(set(v for v in driver_to_team.values()
                                if v not in ("Unknown", "")))

    team_result_history = {}
    for rnd, r_results in sorted(RACE_RESULTS_2026.items()):
        team_bests: dict[str, float] = {}
        for drv, pos in r_results.items():
            t = TEAMS_2026.get(drv, "")
            if t:
                team_bests[t] = float(min(team_bests.get(t, 20.0), float(pos)))
        if len(team_bests) >= 2:
            team_result_history[rnd] = team_bests

    team_prior = {
        t: float(np.mean([prior_means.get(d, 0.0)
                           for d in drivers
                           if driver_to_team.get(d) == t] or [0.0]))
        for t in teams_present
    }

    pl_team = PlackettLuceEstimator(prior_sigma=pl_prior_sigma, decay_lambda=DECAY_LAMBDA)
    past_team_races = [team_result_history[r] for r in sorted(team_result_history.keys())]
    pl_team.fit(past_team_races, team_prior, current_race_num=CANADA_ROUND)
    team_theta_map = pl_team.theta_

    # ── composite θ: season + recent + team ────────────────────────────────────
    theta_season  = np.array([pl_est.theta_.get(d, prior_means.get(d, 0.0))
                               for d in drivers])
    theta_recent  = np.array([pl_recent.theta_.get(d, prior_means.get(d, 0.0))
                               for d in drivers])

    # transition detection: compare rank order of season vs recent PL
    if n_obs >= 3 and n_driv >= 5:
        rank_s  = pd.Series(theta_season).rank()
        rank_r  = pd.Series(theta_recent).rank()
        corr_sr = float(np.corrcoef(rank_s.values, rank_r.values)[0, 1])
        corr_sr = 0.0 if np.isnan(corr_sr) else corr_sr
        transition_factor = float(max(0.0, 1.0 - max(corr_sr, 0.0)))
    else:
        transition_factor = 0.0

    recent_w         = 0.45 * transition_factor
    theta_base_blend = (1.0 - recent_w) * theta_season + recent_w * theta_recent

    # blend 25% team θ signal into driver θ
    team_weight = 0.25
    if teams_present and n_obs >= 2:
        team_t_arr = np.array([team_theta_map.get(driver_to_team.get(d, ""), 0.0)
                                for d in drivers])
        t_std = float(np.std(team_t_arr))
        b_std = float(np.std(theta_base_blend))
        if t_std > 1e-6 and b_std > 1e-6:
            team_scaled = (team_t_arr - team_t_arr.mean()) / t_std * b_std
            theta_race  = (1.0 - team_weight) * theta_base_blend + team_weight * team_scaled
        else:
            theta_race = theta_base_blend
    else:
        theta_race = theta_base_blend

    # ── grid alignment gate: does PL's predicted champion start near the front? ─
    grid_alignment = 1.0
    pl_champ_idx   = int(np.argmax(theta_race))
    pl_champ       = drivers[pl_champ_idx]
    champ_gp       = float(CANADA_Q_RESULT.get(pl_champ, 15))
    if n_obs >= 3:
        # full trust P1-P4; exponential decay from P5 onward
        grid_alignment = float(min(1.0, np.exp(-(max(champ_gp, 1.0) - 4.0) * 0.35)))

    # ── dominance gate: is the PL leader STATISTICALLY ahead of P2? ───────────
    season_theta_vals = pl_est.theta_
    if len(season_theta_vals) >= 2 and pl_est.posterior_var_:
        s_sorted = sorted(season_theta_vals.items(), key=lambda x: x[1], reverse=True)
        sd1, st1 = s_sorted[0]
        sd2, st2 = s_sorted[1]
        raw_dom_gate = float(np.clip((st1 - st2) / 1.5, 0.0, 1.0))
        var_gap_12   = pl_est.var_gap(sd1, sd2)
        z_dom        = (st1 - st2) / np.sqrt(var_gap_12)
        z_gate       = float(np.clip(z_dom / 2.0, 0.0, 1.0))
        precomp_dom  = min(raw_dom_gate, z_gate)
    else:
        precomp_dom = 1.0

    # ── five-factor alpha ──────────────────────────────────────────────────────
    top_driver      = max(season_theta_vals, key=season_theta_vals.get) if season_theta_vals else None
    sigma2_pl       = (float(pl_est.posterior_var_.get(top_driver, pl_est.prior_sigma ** 2))
                       if top_driver and pl_est.posterior_var_ else
                       pl_est.get_mean_posterior_variance(drivers))

    alpha_k = compute_blend_alpha(
        sigma2_pl, n_obs,
        winner_entropy   = winner_entropy,
        dominance_gate   = precomp_dom,
        grid_alignment   = grid_alignment,
        is_new_era       = IS_NEW_ERA,
        sigma2_base      = SIGMA2_BASE,
        alpha_cap        = ALPHA_CAP,
    )

    # ── convert composite θ to rank-normalised scores (lower = better) ─────────
    pl_scores = _rank_normalize_1d(theta_race, ascending=False)

    return (pl_scores, theta_race, alpha_k, sigma2_pl, n_obs,
            winner_entropy, grid_alignment, precomp_dom,
            transition_factor, pl_champ, champ_gp)


# ── step 9: blend and rank ────────────────────────────────────────────────────

def blend_and_rank(base_scores, pl_scores, alpha_k, drivers):
    # final blend: (1-α) × ensemble + α × PL.
    # lower blended score = predicted winner (score 0.0 = P1, score 1.0 = last).
    blended = (1.0 - alpha_k) * base_scores + alpha_k * pl_scores

    # convert blended scores to predicted finishing positions 1..22
    order   = np.argsort(blended)
    predictions = []
    for pos, idx in enumerate(order):
        predictions.append({
            "predicted_pos":  pos + 1,
            "driver":         drivers[idx],
            "team":           TEAMS_2026.get(drivers[idx], ""),
            "blended_score":  float(blended[idx]),
            "base_score":     float(base_scores[idx]),
            "pl_score":       float(pl_scores[idx]),
        })

    return predictions


# ── step 10: write output file ────────────────────────────────────────────────

def write_output(predictions, alpha_k, sigma2_pl, n_obs, winner_entropy,
                 grid_alignment, precomp_dom, transition_factor,
                 pl_champ, champ_gp, base_scores, pl_scores, theta_race, drivers):
    # write a plain-text prediction report modelled after write_wdc_predictions_txt
    # in predict_2026.py. includes the predicted finishing order, key model diagnostics
    # (α, gates, season stats), and a disclaimer about regulation-change uncertainty.

    os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)

    pred_winner     = predictions[0]["driver"]
    pred_winner_name = FULL_NAMES.get(pred_winner, pred_winner)
    pred_winner_team = TEAMS_2026.get(pred_winner, "")
    base_leader_idx  = int(np.argmin(base_scores))
    base_leader      = drivers[base_leader_idx]
    pl_leader_idx    = int(np.argmin(pl_scores))
    pl_leader        = drivers[pl_leader_idx]

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 64,
        "  2026 CANADIAN GRAND PRIX – RACE PREDICTION",
        "  Circuit Gilles Villeneuve, Montréal  (Round 5 of 2026)",
        "=" * 64,
        f"  Generated    : {now_str}",
        f"  Model        : Walk-Forward Plackett-Luce + 4-model BMA ensemble",
        f"                 (XGBoost Regressor + XGBoost Ranker + Random Forest",
        f"                  + ExtraTrees, BMA-weighted with RMT portfolio opt.)",
        f"  Training data: 2022–2025 seasons (predict_2026.py models)",
        "",
        "  DATA SOURCES FOR THIS PREDICTION:",
        f"    Qualifying (Q) : {len(CANADA_Q_RESULT)} drivers (RUS pole, ANT P2, NOR P3)",
        f"    Sprint Race (S): {len(CANADA_S_RESULT)} drivers (RUS P1, NOR P2, ANT P3)",
        f"    Past results   : {n_obs} completed 2026 races (R1 AUS, R2 CHN, R3 JPN, R4 MIA)",
        "",
        "  NOTE: 2026 introduces active aerodynamics and all-new 1.6L hybrid PUs –",
        "  the largest regulation change since 2014. The ensemble was trained on",
        "  2022–2025 data under different aerodynamic rules; predictions in the",
        "  first half of the season carry higher uncertainty than mid-season. The",
        "  is_new_era=True override (SIGMA2_BASE=4.0) loosens the prior so real",
        "  race results can override stale pre-season calibration faster.",
        "",
        "  CIRCUIT SPECIALIST ADJUSTMENTS APPLIED (FOUR LEVERS):",
        "    [1] DATA-DRIVEN CIRCUIT HISTORY (compute_full_circuit_history) —",
        "        Circuit statistics computed from actual finish_position values in",
        "        ml_training_features.csv + ml_2025_features.csv, NOT from pre-race",
        "        feature rows (which are NaN for first-timers like ANT in 2025).",
        "        Key driver records at Canada (2018/19 + 2022-2025 data range):",
        "          ANT: 1 race (2025:P3)  → experience=1, avg_finish=3.0",
        "          RUS: 5 races            → experience=5, avg_finish=8.6",
        "               (2019:P16/2022:P4/2023:P19/2024:P3/2025:P1)",
        "          VER: 6 races            → experience=6, avg_finish=2.17 ← best",
        "               (2018:P3/2019:P5/2022:P1/2023:P1/2024:P1/2025:P2)",
        "          HAM: 6 races            → experience=6, avg_finish=3.67",
        "    [2] QUALIFYING GAP (factual) —",
        "        RUS quali_gap_to_pole=0.0, quali_gap_pct=0.0 (he IS on pole).",
        "        This feature has 0.0158 avg importance vs imputed median ≈0.5s.",
        "    [3] PL PRIOR CIRCUIT CORRECTION (log-quality) —",
        "        Circuit history features have only 3.3% combined importance vs",
        "        ~35% for season form. PL prior correction adjusts the Bayesian",
        "        starting point in log-quality space to give specialist knowledge",
        "        real weight. ±0.20 ≈ ±1 grid position in the middle of the field.",
        "        RUS: +0.20 (most recent winner; 2025 pole+win; recency edge).",
        "        ANT: -0.10 (1 Canada start vs field avg 3-4; inexperienced here",
        "                    despite showing P3 pace in his debut).",
        "    [4] DRIVER ELO (career-aggregate, F1 ELO tracking website) —",
        "        Same source as generate_2026_predictions.py WDC model. Ratings",
        "        are career-aggregate (LEC=2208, VER=2136, HAM=2058, RUS=2033,",
        "        ANT=1836). ANT's rating is correctly mid-pack vs veterans because",
        "        4 wins in one season should not inflate a career ELO. Blended at",
        "        15% weight into PL prior to add multi-season driver quality without",
        "        overriding the 4-race 2026 PL dominance signal.",
        "        Note: ELO at PL prior stage (not a model feature) — ensemble",
        "        cannot be retrained mid-prediction.",
        "",
        "─" * 64,
        "  MODEL DIAGNOSTICS",
        "─" * 64,
        f"  α (PL blend weight)  : {alpha_k:.4f}  (of max {ALPHA_CAP})",
        f"  PL σ² (leader var.)  : {sigma2_pl:.4f}",
        f"  Races observed (n)   : {n_obs}  (n≥3 required for PL to activate)",
        f"  Winner entropy       : {winner_entropy:.3f}  "
              f"(unique winners {int(round(winner_entropy * n_obs))}/{n_obs}; "
              f"RUS won R1, ANT won R2-R4)",
        f"  Dominance gate       : {precomp_dom:.4f}  (raw Δθ + z-score gate)",
        f"  Grid alignment gate  : {grid_alignment:.4f}  "
              f"(PL champion {pl_champ} starts P{int(champ_gp)})",
        f"  Transition factor    : {transition_factor:.4f}  (0=stable, 1=power shift)",
        "",
        f"  Base ensemble leader : {base_leader} "
              f"({FULL_NAMES.get(base_leader, base_leader)}, "
              f"score {base_scores[drivers.index(base_leader)]:.4f})",
        f"  PL layer leader      : {pl_leader} "
              f"({FULL_NAMES.get(pl_leader, pl_leader)})",
        f"  FINAL PREDICTION     : {pred_winner} "
              f"({pred_winner_name}, {pred_winner_team})",
        "",
        "─" * 64,
        "  PREDICTED FINISHING ORDER",
        "─" * 64,
        f"  {'P':>3}  {'Driver':<24}  {'Team':<22}  {'Blended':>8}  {'Base':>7}  {'PL':>6}",
        f"  {'─'*3}  {'─'*24}  {'─'*22}  {'─'*8}  {'─'*7}  {'─'*6}",
    ]

    for row in predictions:
        pos   = int(row["predicted_pos"])
        drv   = row["driver"]
        name  = FULL_NAMES.get(drv, drv)
        team  = row["team"][:22]
        bscore = float(row["blended_score"])
        raw_b  = float(row["base_score"])
        raw_p  = float(row["pl_score"])
        lines.append(
            f"  {pos:>3}  {name:<24}  {team:<22}  {bscore:8.4f}  {raw_b:7.4f}  {raw_p:6.4f}"
        )

    lines += [
        "",
        "─" * 64,
        "  SPRINT WEEKEND CONTEXT",
        "─" * 64,
        "  The 2026 Canadian GP is a sprint format weekend (no FP2/FP3).",
        "  Sprint result: RUS won ahead of NOR and ANT.",
        "  The sprint race finishing position (sprint_pos feature) is included",
        "  in the 54-column feature matrix and influences the base ensemble.",
        "  However, because is_new_era=True, the sprint is NOT fed as a separate",
        "  Plackett-Luce observation. In a non-new-era year, past sprint results",
        "  would be fed as soft PL observations at weight QUAL_PL_WEIGHT=0.25.",
        "  The new-era override skips these to avoid noisy early-season signals",
        "  contaminating the PL prior before the regulation picture stabilises.",
        "",
        "─" * 64,
        "  2026 SEASON CONTEXT (before Canadian GP)",
        "─" * 64,
        "  Kimi Antonelli (Mercedes) leads the championship with 93 points,",
        "  winning R2 China, R3 Japan, and R4 Miami. George Russell won R1",
        "  Australia from pole, demonstrating Mercedes' strong early-season",
        "  package advantage. McLaren (Norris/Piastri) is showing strong pace",
        "  recovery with back-to-back podiums through R3-R4.",
        "",
        "  The PL model captures this: ANT's θ reflects dominant season",
        "  performance, while RUS's Q session pole reduces the grid-alignment",
        f"  penalty (PL champion {pl_champ} starts P{int(champ_gp)}, alignment={grid_alignment:.3f}).",
        "",
        "=" * 64,
    ]

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  ✓ Prediction written: {OUT_FILE}")
    return OUT_FILE


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 60)
    print("  2026 Canadian GP – Walk-Forward PL Prediction")
    print("=" * 60)

    # enable FastF1 cache - primarily for historical session lookup
    # note: the qualifying and sprint data is already hard-coded above because
    # we fetched it earlier in the analysis pipeline (see session summary).
    # the load_circuit_history() call below only reads CSV files, not FastF1.
    fastf1.Cache.enable_cache(CACHE_DIR)
    fastf1.Cache.offline_mode(enabled=True)  # prefer cache; only needs CSVs

    # ── [1/7] load models ──────────────────────────────────────────────────────
    print("\n[1/7] Loading pre-trained models from models_2026/...")
    (xgb_model, xgb_ranker, rf_win, rf_top10, et_model,
     imputer, bma_weights, cal_temp) = load_models()

    # ── [2/7] build feature matrix ─────────────────────────────────────────────
    print("\n[2/7] Building feature matrix...")
    X, drivers = build_feature_matrix(imputer)

    # ── [3/7] base ensemble prediction ────────────────────────────────────────
    print("\n[3/7] Base ensemble prediction...")
    base_scores, df_meta = run_ensemble(
        xgb_model, xgb_ranker, rf_win, rf_top10, et_model,
        imputer, bma_weights, cal_temp, X, drivers,
    )

    # ── [4/7] RMT driver-cluster regularisation ────────────────────────────────
    print("\n[4/7] RMT Marchenko-Pastur driver-cluster regularisation...")
    # 2026 unique race winners so far: RUS (R1) and ANT (R2, R3, R4)
    season_winners = {"RUS", "ANT"}
    base_scores, rmt_blend, n_rmt_sig = apply_rmt_regularisation(
        base_scores, X, n_obs_so_far=4, season_unique_winners=season_winners,
    )
    print(f"  ✓ RMT: adaptive_blend={rmt_blend:.4f}, signal_dims={n_rmt_sig}")

    # ── [5/7] Plackett-Luce update ─────────────────────────────────────────────
    print("\n[5/7] Walk-forward Plackett-Luce update...")
    (pl_scores, theta_race, alpha_k, sigma2_pl, n_obs,
     winner_entropy, grid_alignment, precomp_dom,
     transition_factor, pl_champ, champ_gp) = run_pl_layer(
        base_scores, X, drivers, df_meta,
    )
    print(f"  ✓ α = {alpha_k:.4f}  (PL blend weight out of max {ALPHA_CAP})")

    # ── [6/7] blend and rank ────────────────────────────────────────────────────
    print("\n[6/7] Blending scores and ranking drivers...")
    predictions = blend_and_rank(base_scores, pl_scores, alpha_k, drivers)

    pred_winner = predictions[0]["driver"]
    print(f"  ✓ Predicted winner: {pred_winner} ({FULL_NAMES.get(pred_winner, pred_winner)})")
    print("  Top 5:")
    for p in predictions[:5]:
        d = p["driver"]
        print(f"    P{p['predicted_pos']}  {FULL_NAMES.get(d, d)} ({TEAMS_2026.get(d,'')})"
              f"  score={p['blended_score']:.4f}")

    # ── [7/7] write output file ────────────────────────────────────────────────
    print("\n[7/7] Writing prediction to file...")
    out_path = write_output(
        predictions, alpha_k, sigma2_pl, n_obs, winner_entropy,
        grid_alignment, precomp_dom, transition_factor,
        pl_champ, champ_gp, base_scores, pl_scores, theta_race, drivers,
    )

    print("\n" + "=" * 60)
    print("  Done.")
    print(f"  Output: {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
