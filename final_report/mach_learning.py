"""Train and evaluate the race-prediction models used in the final report.

The goal is to predict each driver's finishing order before the race starts.
No race-result fields from the target race are allowed in the input features.

Model pieces:
- XGBoost regressor: predicts numeric finishing position.
- XGBoost ranker: focuses on ordering drivers within each race.
- Random forest classifier: estimates win probability.
- ExtraTrees regressor: adds a less-correlated tree model to the ensemble.

The ensemble weights come from leave-one-season-out validation instead of a
hand-picked blend. The report explains the final metrics and why exact race
prediction has a ceiling because of safety cars, incidents, and failures.

References:
- XGBoost paper: https://arxiv.org/abs/1603.02754
- Random Forests paper: https://link.springer.com/article/10.1023/A:1010933404324
- XGBRanker docs: https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html
- Temperature scaling: https://arxiv.org/abs/1706.04599
"""


import sys
import os
import warnings
import numpy as np
import pandas as pd
import joblib

# Windows consoles default to cp1252 which can't print box-drawing chars (─, →, etc.)
# that appear in all the progress banners - reconfigure stdout to utf-8 early so we don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor  # classifiers + ET
from sklearn.metrics import mean_absolute_error       # position MAE
from sklearn.impute import SimpleImputer              # fill NaN with median
from sklearn.model_selection import RandomizedSearchCV  # hyperparameter search
from sklearn.linear_model import LogisticRegression   # stacking meta-model fallback

import xgboost as xgb
from scipy.stats import randint, uniform  # for parameter distributions
from scipy.optimize import minimize_scalar, minimize  # minimize_scalar: MAP temperature calibration; minimize: RMT portfolio optimization

warnings.filterwarnings("ignore")  # suppress fastf1 + xgboost deprecation noise, because it was really annoying to deal with this
                                   # I don't know if that's proper methodology but it works

# pull in the Marchenko-Pastur denoising tools so we can use them for two things:
#   1. rmt_minimum_variance_weights() below: denoise the 4×4 inter-model OOF score
#      covariance, then run Markowitz minimum-variance portfolio optimization on it.
#      this is the "Laloux 1999 applied to ensemble weights" innovation - the key
#      insight is that models with correlated errors are less valuable to the ensemble
#      than models that fail on different races (independent errors add diversity).
#   2. marchenko_pastur_edges() is used as a diagnostic sanity check to confirm how
#      much of the 4×4 covariance structure is genuine signal vs sampling noise.
# rmt_denoising.py lives in the same directory as this file. we add its directory
# to sys.path explicitly so the import works whether we run mach_learning.py
# directly or import it as a module from a parent script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmt_denoising import rmt_denoise_psd, marchenko_pastur_edges

# ──────────────────────────────────
# Our paths and constants
# ──────────────────────────────────

DATA_DIR   = "data_organized"
TRAIN_FILE = os.path.join(DATA_DIR, "ml_training_features.csv")
PRED_FILE  = os.path.join(DATA_DIR, "ml_2025_features.csv")
OUT_DIR    = DATA_DIR
MODEL_DIR  = os.path.join(DATA_DIR, "models")

SEED = 42  # same seed everywhere, makes results reproducible across runs
# Basically I'm trying to make it so that XGBoost, RandomForest and randomizeed search use the same random sequence
# This lets me tune hyperparameters
# 47 features - must exactly match what feature_engineering.py writes out
# grouped by category for readability, not for any ML reason
# if you change these you also have to change feature_engineering.py or models won't load
FEATURE_COLS = [
    # ── sprint / qualifying grid ──────────────────────────────────────────
    "IsSprintWeekend",       # 1 if sprint format; affects race distance and dynamics(sprints have less practice time and have an additional sprint race)
    "grid_position",         # official qualifying position(not sprint quali); single most predictive feature
    "sprint_quali_pos",      # sprint qualifying position (NaN if not a sprint weekend)
    "sprint_pos",            # sprint race position (NaN if not a sprint weekend)
    "is_front_row",          # 1 if starting P1 or P2; front row starters win ~70% of races historically
    "grid_position_pct",     # grid_pos / field_size normalised to [0,1]; handles variable field sizes cleanly

    # ── free practice best-lap deltas (seconds behind session best) ────────
    # these answer: how quick was the car vs the field in each session? and how well did the teams set up the car?
    # Free practice enables teams to fine tune the car for the race which can help us predict outcomes
    # ref: fastf1 pick_quicklaps: https://docs.fastf1.dev/core.html#fastf1.core.Laps.pick_quicklaps
    "fp1_pace_delta",        # FP1 best lap behind session best (seconds)
    "fp2_pace_delta",        # FP2 best lap behind session best (FP2 and FP3 depend on whether it's a sprint weekend or not)
    "fp3_pace_delta",        # FP3 best lap behind session best (Saturday, fresher signal)

    # ── FP2 detailed session features ─────────────────────────────────────
    # sector splits reveal where on track the car is strong or weak
    # S1 = high speed aero corners, S2 = medium speed mechanical grip, S3 = traction zones
    # long-run delta is the best proxy for actual race pace (5+ consecutive laps)
    # We don't do this for FP1 because it's too early for teams to get a good setup
    "fp2_s1_delta",          # sector 1 gap vs session best
    "fp2_s2_delta",          # sector 2 gap vs session best
    "fp2_s3_delta",          # sector 3 gap vs session best
    "fp2_long_run_delta",    # median of 5+ lap stints vs field median - best race pace indicator(depends on if team does a long run or not)
    # fp2_deg_rate and fp2_deg_rate_obs are computed and saved to CSV but excluded from training
    # 2022=18% NaN, 2023=88% NaN, 2024=100% NaN - even with a 0.0/indicator approach, 2023 holdout drops from 59% to 45%
    # the indicator can't overcome the near-complete absence of real data in 2023-2024
    # re-add when at least 3 seasons have <50% NaN coverage in the FP2 long-run data
    "fp2_consistency",       # std dev of FP2 quicklaps; low = smooth driver or well-balanced setup
    "fp_pace_progression",   # fp1_delta minus fp2_delta: positive = driver improved vs field overnight

    # ── FP3 long run (Saturday, closer to race conditions than FP2) ────────
    "fp3_long_run_delta",    # Saturday long run vs field

    # ── qualifying gap features ────────────────────────────────────────────
    # using seconds not positions because 0.050s gap vs 2.0s gap are very different stories
    # teammate gap normalizes out the car: isolates what the driver contributes
    "quali_gap_to_pole",     # driver's best quali lap minus pole lap in seconds
    "quali_teammate_gap",    # driver vs fastest teammate in qualifying (seconds)
    "q3_appearance",         # 1 if driver made it to Q3, 0 if knocked out in Q1/Q2
    "quali_gap_pct",         # gap_to_pole / pole_time: scale-invariant across circuits and regulation changes

    # ── session weather features (pre-race only; race weather = data leakage) ──
    # FP2 Friday afternoon track temp is a reasonable comparison and prediction for Sunday conditions
    # quali air temp affects tyre cooling and thermal degradation on race day
    # we deliberately exclude race weather - it's unknowable before the race
    "fp1_track_temp",        # FP1 track surface temperature (celsius)
    "fp1_wet",               # was FP1 wet? (0 or 1)
    "fp2_track_temp",        # FP2 track surface temp - closer to Sunday conditions
    "fp2_wet",               # was FP2 wet?
    "quali_track_temp",      # qualifying track temp
    "quali_air_temp",        # qualifying air temp; affects rubber degradation prediction
    "quali_wet",             # was qualifying wet?

    # ── rolling form features (driver momentum) ────────────────────────────
    # 3-race window: captures very recent form, noisier but more current
    # 5-race window: smoother baseline, better for mid-season trends(driver performance fluctuates A LOT)
    # 10-race window: longer trend that smooths anomalies; good for reliability patterns over a half-season
    # dnf_rate_5 is the reliability/bad luck signal - teams with mechanical issues score 0
    # win_rate and podium_rate capture dominant-car phases better than avg finish alone
    # (VER 2023: avg finish ~1.3 but win_rate_5 often 1.0 - a much cleaner dominance signal)
    "rolling_avg_finish_3",  # mean finish position over last 3 races
    "rolling_avg_finish_5",  # mean finish position over last 5 races
    "rolling_dnf_rate_5",    # fraction of last 5 races ending in DNF/retirement
    "rolling_points_5",      # total championship points scored in last 5 races
    "rolling_win_rate_5",    # fraction of last 5 races won; strong signal in dominant-car seasons
    "driver_podium_rate_5",  # fraction of last 5 races finishing P1/P2/P3
    "rolling_avg_finish_10", # mean finish over last 10 races; longer horizon than 5-race window
    "rolling_points_10",     # total points over last 10 races; captures sustained scoring pace
    "driver_vs_team_avg_delta", # driver rolling_avg_finish_5 minus team_rolling_avg_5; negative = outperforming car
    "team_rolling_avg_5",    # team's average finish over last 5 (car competitiveness signal)
    "season_pts_pct",        # driver's season pts divided by (races_so_far * 25)
    "season_avg_finish",     # driver's mean finish position so far this season
    "season_wins_pct",       # wins / races_so_far this season; identifies when a driver is in a dominant run

    # ── circuit history features ───────────────────────────────────────────
    # first-time drivers at a circuit consistently underperform vs their qualifying position(obviously)
    # circuit_team_avg captures track-specific setup knowledge accumulated over years
    "circuit_experience",              # how many times has this driver raced at this circuit
    "circuit_driver_avg_finish",       # driver's historical mean finish at this circuit
    "circuit_team_avg_finish",         # team's historical mean finish at this circuit
    "driver_circuit_positions_gained", # avg (grid_pos - finish_pos) at this circuit; positive = gains places in races here

    # ── circuit structural characteristics (static per-circuit lookup) ─────────
    # these are genuinely new information that rolling stats can't derive
    # overtaking_score tells the model: at Monaco pole is nearly guaranteed win; at Monza it's a DRS gamble
    # sc_rate tells it how chaotic this track is historically (affects how much qualifying matters)
    # is_street captures Monaco/Baku/Singapore/Jeddah/Miami/LasVegas - wall-bounded = fundamentally different dynamics
    # ref: https://en.wikipedia.org/wiki/List_of_Formula_One_circuits
    "circuit_overtaking_score",        # 0.05=Monaco (nearly impossible to pass) to 0.70=Monza (DRS festival)
    "circuit_sc_rate",                 # fraction of races at this circuit historically with a safety car
    "circuit_is_street",               # 1 if wall-bounded temporary circuit (Monaco, Baku, Singapore, etc.)

    # ── accuracy-push features (added to beat academic 60-65% ceiling) ──────
    # car regulation era: 0=turbo-hybrid (2018-2021), 1=ground-effect (2022+)
    # helps the model know that car characteristics fundamentally changed - crucial for LOSO folds
    "car_regulation_era",
    # front-row at Monaco = near-certain win; front-row at Monza = easily overtaken
    # explicit interaction because tree models need this handed to them, they won't find it alone
    "grid_x_overtaking_inv",
    # championship rank normalised to [0,1]: 0 = leader, 1 = last - leader is almost always in fastest car
    "championship_rank_pct",
    # team's P1 driver gets priority pit calls, best upgrades first - real strategy signal
    "is_number_one_driver",
    # driver's circuit avg minus team's circuit avg: Hamilton+Silverstone has a real negative value here
    "circuit_specialist_delta",
    # team_avg5 - team_avg3: positive = car getting faster recently (catches McLaren-style surges)
    "team_momentum",
    # team's mean finish over last 3 races (shorter window, more reactive than 5-race)
    "team_rolling_avg_3",
]  # 54 total: original 47 + 7 new features targeting specific weaknesses in 2019/2024 holdout
  # fp2_deg_rate / fp2_deg_rate_obs computed and saved to CSV but excluded until coverage improves across seasons

# F1 points - used for championship projection in predict_2025 and predict_2026
# Maps race standings to points granted
POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


# ──────────────────────────────────
# How we load data 
# ──────────────────────────────────

def load_and_prepare(filepath, fit_imputer=None):
    # loads the feature CSV and returns everything needed for training or prediction
    # fit_imputer=None → we're in training mode, fit a fresh imputer on this data
    # fit_imputer=<already fitted imputer> → prediction mode, just transform using the existing one
    # (we do it this way so the imputer learns the medians from training data only, never from test data)
    df = pd.read_csv(filepath)

    meta_cols = ["Year", "Round", "EventName", "Circuit", "Driver", "Team"]

    df_meta = df[meta_cols].copy()         # keep metadata separate from features
    X_raw   = df[FEATURE_COLS].copy().astype(float)  # typecast to float; NaN stays NaN

    # fill in missing values with the column median
    # we use median not mean because some features have heavy right tails
    # (a driver who missed FP2 entirely gets a really large pace delta, which would skew the mean)
    # ref: https://scikit-learn.org/stable/modules/impute.html#univariate-imputation
    if fit_imputer is None:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)  # fit on this data; keep_empty_features=True preserves all-NaN columns (e.g. sprint cols on non-sprint data) so shape always matches FEATURE_COLS
        X = pd.DataFrame(imputer.fit_transform(X_raw), columns=FEATURE_COLS)
    else:
        imputer = fit_imputer  # reuse training-set fitted imputer
        X = pd.DataFrame(imputer.transform(X_raw), columns=FEATURE_COLS)

    # targets - won't exist in pure prediction CSVs
    y_pos   = df["finish_position"].astype(float) if "finish_position" in df else None
    y_win   = df["is_winner"].astype(float)       if "is_winner"   in df else None
    y_top10 = df["is_top10"].astype(float)        if "is_top10"    in df else None

    return X, y_pos, y_win, y_top10, df_meta, df, imputer


# ──────────────────────────────────────────────────────────────────
# BAYESIAN STATISTICS
# ──────────────────────────────────────────────────────────────────
#
# Quick overview of Bayesian statistics (since this shapes the model and I highly doubt that many people know what this is):
# ─────────────────────────────────────────────────────────────────
# Bayesian statistics is a framework for updating probability estimates
# as you get more evidence. The core equation is Bayes' theorem:
#
#     P(hypothesis | data) ∝ P(data | hypothesis) × P(hypothesis)
#     i.e.,   posterior  ∝  likelihood  ×  prior
#
# "prior" = what you believe before seeing the data
# "likelihood" = how well the data fits the hypothesis
# "posterior" = your updated belief after seeing the data
#
# This is fundamentally different from frequentist statistics. Instead
# of asking "what does the data say?", Bayesian methods let you
# incorporate domain knowledge (priors) and get better estimates when
# data is limited - exactly our situation with 3-4 years of F1 data(since FastF1 unfortunately DOES NOT CARRY DETAILED DATA BEFORE 2018).
# Yes this was annoying, so we had to revert to something that we had discovered earlier
#
# Reference: https://en.wikipedia.org/wiki/Bayesian_statistics
# Introductory tutorial that we used: https://seeing-theory.brown.edu/bayesian-inference/
#
# We use Bayesian reasoning in two ways in this pipeline(as mentioned before):
#
# 1. Normal-Normal conjugate hierarchical model (James-Stein shrinkage) for win probability prior
#    - Treats each driver's circuit-specific avg finish as a noisy observation of their latent ability
#    - Posterior mean shrinks toward the global mid-field mean (mu_0=10) with shrinkage factor B
#      B = n*tau^2 / (n*tau^2 + sigma_obs^2)  - strong shrinkage when history is thin
#    - ref: Gelman et al. BDA3 ch. 5 (https://stat.columbia.edu/~gelman/book/)
#    - ref: James-Stein estimator (https://doi.org/10.1007/BF02888369)
#
# 2. Bayesian temperature scaling (Platt scaling with log-normal MAP prior)
#    - Calibrates RF win probabilities before blending with the circuit-history prior
#    - MAP objective: NLL(T) + log(T)^2 / (2*sigma_T^2), solved via minimize_scalar
#    - T is fit from out-of-fold (OOF) LOSO RF probabilities so it's never trained on test data
#    - ref: Guo et al. 2017 "On Calibration of Modern Neural Networks" https://arxiv.org/abs/1706.04599
#
# 3. Inverse-variance Bayesian Model Averaging (BMA) for ensemble weights
#    - Instead of fixed 40/40/20 weights, compute them from LOSO(leave one season out) data
#    - Combines two signal sources:
#        (a) Dirichlet posterior from correct-winner counts (60% weight)
#        (b) Inverse-MAE weights across LOSO folds (40% weight)
#    - ref: Dirichlet distribution (https://en.wikipedia.org/wiki/Dirichlet_distribution)
#    - ref: Hoeting et al. 1999 BMA paper https://doi.org/10.1214/ss/1009212519
# ──────────────────────────────────────────────────────────────────

def compute_hierarchical_win_prior(X_features):
    # upgraded version of the old Beta-Binomial prior - now using a Normal-Normal
    # hierarchical model which is a proper James-Stein shrinkage estimator
    # the idea is the same as before (hamilton has won at silverstone 8 times so factor that in)
    # but the math is more principled this time around
    #
    # the problem with the old approach was it treated every race history the same
    # 1 race at a new circuit and 15 races at a familiar one got the same treatment
    # with James-Stein shrinkage, the more history a driver has the more we trust it
    # when they've only raced somewhere once we pull their estimate all the way back
    # toward the middle of the field (mu_0 = 10.0, which is basically "we don't know")
    #
    # the math (we looked this up from Gelman's BDA3 book):
    #   B = (n * tau_sq) / (n * tau_sq + sigma_obs_sq)   <- the shrinkage factor
    #   posterior = B * observed_avg + (1-B) * mu_0
    # when n is small B is close to 0 so we mostly use the prior (mid-field)
    # when n is large B approaches 1 so we mostly trust the historical data
    # ref: Gelman et al. BDA3 ch. 5 (https://stat.columbia.edu/~gelman/book/)
    # ref: the James-Stein estimator paper (https://doi.org/10.1007/BF02888369)
    # ref: https://efron.ckirby.su.domains/other/Article1977.pdf (empirical Bayes overview)

    avg_finish = X_features["circuit_driver_avg_finish"].fillna(10.0).values.astype(float)
    experience = X_features["circuit_experience"].fillna(0.0).values.astype(float)

    mu_0         = 10.0   # global prior mean: if we know nothing, assume mid-field
    sigma_obs_sq = 22.0   # how much finishing positions vary race to race at a given circuit (~std 4.7)
    tau_sq       = 10.0   # how spread out drivers actually are around mid-field (~std 3.2 places)

    n = np.maximum(experience, 1.0)   # need at least 1 to avoid division by zero
    B = (n * tau_sq) / (n * tau_sq + sigma_obs_sq)   # shrinkage: large n -> trust the data more

    # the posterior is a weighted average between what we observed and what we'd guess with no data
    posterior_pos = B * avg_finish + (1.0 - B) * mu_0

    # now convert that position estimate (1-20 scale) to something like a win probability
    # P1 average -> ~0.25 win rate (even Verstappen in 2023 didn't win every race)
    # P10 average -> ~0.05, which is about right for a midfield driver
    prior_score = np.clip((20.0 - posterior_pos) / 19.0 * 0.25, 0.003, 0.25)

    return prior_score  # one score per driver row in X_features


def bayesian_adjust_win_proba(rf_proba, X_features, base_blend=0.18):
    # blend the (already temperature-calibrated) RF win probability with the hierarchical prior
    # this is empirical Bayes - instead of computing the full posterior analytically
    # we approximate it as a weighted average between the model and the prior
    # the important thing here is that we call calibrate_with_temperature() first
    # before this function - calibration has to happen before the prior blend or
    # we'd be computing an awkward mixture of an overconfident model with a prior
    # ref: https://en.wikipedia.org/wiki/Empirical_Bayes_method
    # similar weighted-average approach is used in sports analytics (basketball, baseball)

    prior_score = compute_hierarchical_win_prior(X_features)  # James-Stein circuit history prior
    experience  = X_features["circuit_experience"].fillna(0.0).values.astype(float)

    # the blend weight scales with how much circuit history the driver has
    # if a driver has never raced here before: prior barely influences anything (~4.5% weight)
    # if they've raced here 15+ times (like Hamilton at Silverstone): prior gets ~18% weight
    # 18% felt right empirically - enough to matter but the model still dominates
    adaptive_blend = base_blend * np.clip(experience / 12.0, 0.25, 1.0)

    adjusted = (1.0 - adaptive_blend) * np.asarray(rf_proba) + adaptive_blend * prior_score
    return np.clip(adjusted, 1e-6, 1.0 - 1e-6)


def bayesian_calibrate_rf_temperature(oof_proba, oof_labels):
    # Random Forest classifiers are notoriously overconfident - they often spit out
    # probabilities like 0.97 or 0.02 when the truth is more like 0.80 or 0.10
    # this matters for us because we then blend these probabilities with the Bayesian prior
    # if the RF is overconfident the blend ends up being dominated by the RF output
    #
    # temperature scaling is a super simple fix for this
    # we just divide the logit (log-odds) by a temperature T before converting back to a probability
    # p_cal = sigmoid( logit(p_raw) / T )
    # T > 1 makes everything more uncertain (spreads predictions toward 0.5)
    # T < 1 makes everything more extreme (squishes toward 0 or 1)
    # T = 1 is no change (the identity)
    #
    # we fit T using MAP estimation (maximum a posteriori) on out-of-fold LOSO predictions
    # using OOF predictions is important - we don't want to fit T on the same data that
    # trained the RF or we'd just be overfitting the calibration too
    # the prior on T is log-normal centered at 1.0 (no calibration) with sigma = 0.5
    # this prevents T from going crazy if the OOF data is small
    # we optimize over log(T) instead of T directly so the search range is unconstrained
    # ref: Platt scaling (https://en.wikipedia.org/wiki/Platt_scaling)
    # ref: Guo et al. 2017 "On Calibration of Modern Neural Networks"
    #      https://arxiv.org/abs/1706.04599 - great paper that shows even big models are miscalibrated

    p     = np.clip(np.asarray(oof_proba,  dtype=float), 1e-7, 1.0 - 1e-7)
    y     = np.asarray(oof_labels, dtype=float)
    log_p = np.log(p / (1.0 - p))  # logit - basically "how many log-odds points does the model give it"

    sigma_T_sq = 0.25  # log-normal prior on T: std = 0.5 means T is probably between ~0.6 and 1.6

    def neg_map_objective(log_T):
        # we minimize negative log-posterior = NLL + prior penalty
        # optimizing over log(T) means we never accidentally evaluate T <= 0
        T     = np.exp(log_T)
        p_cal = 1.0 / (1.0 + np.exp(-log_p / T))  # apply temperature, convert back to probability
        p_cal = np.clip(p_cal, 1e-7, 1.0 - 1e-7)
        nll   = -np.mean(y * np.log(p_cal) + (1.0 - y) * np.log(1.0 - p_cal))
        prior_penalty = (log_T ** 2) / (2.0 * sigma_T_sq)  # pulls T back toward 1.0 if data is noisy
        return nll + prior_penalty

    result = minimize_scalar(neg_map_objective, bounds=(-2.0, 2.0), method="bounded")
    T_opt  = float(np.exp(result.x))
    print(f"    Bayesian temperature T = {T_opt:.4f}  (1.0 = no calibration)")
    return T_opt


def calibrate_with_temperature(rf_proba, T):
    # apply the fitted temperature T to a set of RF win probabilities
    # this is called at inference time to calibrate raw RF outputs before blending with the prior
    # formula: p_cal = sigmoid(logit(p_raw) / T)
    # if T is basically 1.0 we just skip the math and return unchanged
    if abs(T - 1.0) < 1e-6:
        return np.asarray(rf_proba)
    p        = np.clip(np.asarray(rf_proba, dtype=float), 1e-7, 1.0 - 1e-7)
    log_odds = np.log(p / (1.0 - p))         # back to logit space
    p_cal    = 1.0 / (1.0 + np.exp(-log_odds / T))  # rescale and back to probability
    return np.clip(p_cal, 1e-7, 1.0 - 1e-7)


def bayesian_model_averaging_weights(loso_correct_per_model, prior_strength=2.0,
                                      model_mae=None, rmt_portfolio=None):
    # so instead of just hardcoding "XGBoost gets 40%, Ranker gets 40%, RF gets 20%"
    # we let the actual LOSO validation data decide how much to trust each model
    # that's the whole idea of Bayesian Model Averaging - use the evidence, not your gut
    #
    # the trick is combining two signals:
    #   1. "did you call the winner?" (60% of the final weight)
    #      for each model, count how many race winners it correctly predicted across all LOSO folds
    #      we use a Dirichlet prior (think of it as starting with imaginary "base wins" before
    #      looking at real data) then update with the actual winner count from LOSO
    #      Dirichlet is the conjugate prior here - meaning the math works out really cleanly
    #      ref: https://en.wikipedia.org/wiki/Dirichlet_distribution
    #   2. "was your whole ranking sensible?" (40% of the final weight)
    #      a model that gets winners right but puts Albon in P2 every race is still bad
    #      so we also weight by inverse position MAE: lower error = higher weight
    #      formula: weight_i = (1/MAE_i) / sum(1/MAE_j) for all models j
    #   3. (optional) RMT Markowitz minimum-variance portfolio weights (25% blend if provided)
    #      see rmt_minimum_variance_weights() below for the full explanation
    #      short version: models that fail on the same races are less valuable than models
    #      that fail independently - the portfolio weights explicitly reward this diversity
    #
    # why 60/40? winner prediction is the main thing we're judged on (it's in the abstract)
    # but ignoring ranking quality entirely felt like leaving real signal on the table
    # ref: Hoeting et al. 1999 BMA paper: https://doi.org/10.1214/ss/1009212519

    alpha_prior = np.array([
        prior_strength,        # XGBoost Regressor
        prior_strength,        # XGBoost Ranker
        prior_strength * 0.5,  # Random Forest (weaker: binary classifier, not a ranker)
        prior_strength,        # ExtraTrees Regressor (same family as XGB reg, full prior)
    ])

    observed = np.array([
        loso_correct_per_model.get("xgb",    0),
        loso_correct_per_model.get("ranker",  0),
        loso_correct_per_model.get("rf",      0),
        loso_correct_per_model.get("et",      0),
    ], dtype=float)

    alpha_post        = alpha_prior + observed
    dirichlet_weights = alpha_post / alpha_post.sum()

    # inverse-MAE weights - fall back to equal if MAE not provided
    if model_mae is not None:
        mae_arr = np.array([
            model_mae.get("xgb",    np.nan),
            model_mae.get("ranker", np.nan),
            model_mae.get("rf",     np.nan),
            model_mae.get("et",     np.nan),
        ], dtype=float)
        mae_arr = np.where(mae_arr > 0, mae_arr, np.nanmean(mae_arr[mae_arr > 0]))
        inv_mae = 1.0 / mae_arr
        inv_mae_weights = inv_mae / inv_mae.sum()
    else:
        inv_mae_weights = np.ones(4) / 4.0

    weights = 0.6 * dirichlet_weights + 0.4 * inv_mae_weights

    # if RMT minimum-variance portfolio weights are provided, blend them in
    # this is a third signal that captures inter-model ERROR CORRELATION:
    # the Dirichlet + MAE approach asks "how often did each model win independently?"
    # the portfolio approach asks "which blend minimizes the joint prediction variance?"
    # they're complementary - we blend conservatively (25% RMT, 75% Dirichlet+MAE)
    # to avoid double-counting the OOF data signal that's already in the Dirichlet weights.
    # 25% was chosen after trying 0%, 15%, 25%, 35%: 25% consistently gave the best
    # diversity without overwhelming the winner-count signal from the Dirichlet posterior.
    if rmt_portfolio is not None:
        rmt_arr = np.array([
            rmt_portfolio["w_xgb"],
            rmt_portfolio["w_ranker"],
            rmt_portfolio["w_rf"],
            rmt_portfolio["w_et"],
        ])
        rmt_blend = 0.25   # conservative: portfolio is complementary, not a replacement
        weights   = (1.0 - rmt_blend) * weights + rmt_blend * rmt_arr
        weights   = weights / weights.sum()   # renormalize after blend (shouldn't change by more than ~1e-10)

    print(f"\n  Bayesian Model Averaging Weights (posterior from LOSO):")
    print(f"    XGBoost Regressor : {weights[0]:.3f}  (prior={alpha_prior[0]:.1f}, correct={observed[0]:.0f})")
    print(f"    XGBoost Ranker    : {weights[1]:.3f}  (prior={alpha_prior[1]:.1f}, correct={observed[1]:.0f})")
    print(f"    Random Forest     : {weights[2]:.3f}  (prior={alpha_prior[2]:.1f}, correct={observed[2]:.0f})")
    print(f"    ExtraTrees        : {weights[3]:.3f}  (prior={alpha_prior[3]:.1f}, correct={observed[3]:.0f})")
    if model_mae is not None:
        print(f"    MAE: xgb={model_mae.get('xgb', '?'):.3f}  ranker={model_mae.get('ranker', '?'):.3f}  rf={model_mae.get('rf', '?'):.3f}  et={model_mae.get('et', '?'):.3f}")
    if rmt_portfolio is not None:
        print(f"    RMT portfolio blend (25%): xgb={rmt_portfolio['w_xgb']:.3f}  "
              f"ranker={rmt_portfolio['w_ranker']:.3f}  "
              f"rf={rmt_portfolio['w_rf']:.3f}  et={rmt_portfolio['w_et']:.3f}")

    return {
        "w_xgb":    float(weights[0]),
        "w_ranker": float(weights[1]),
        "w_rf":     float(weights[2]),
        "w_et":     float(weights[3]),
    }


def rmt_minimum_variance_weights(fold_cache, years, verbose=True):
    # Markowitz minimum-variance portfolio weights for the model ensemble, using
    # a Marchenko-Pastur denoised inter-model OOF score covariance.
    #
    # the core idea comes directly from Laloux et al. (1999) who first applied RMT
    # to financial portfolio construction: clean up the empirical covariance matrix
    # by removing noise eigenvalues, then find the portfolio that minimizes the total
    # variance. here we apply the same principle to the 4 ensemble models.
    #
    # why it helps beyond the Dirichlet BMA weights:
    # the Dirichlet approach asks "how often did each model call the winner correctly?"
    # but it doesn't capture the JOINT failure structure. two models that always fail
    # on the SAME races are less valuable together than two models that fail on
    # different races (even if they have the same individual accuracy). the minimum-variance
    # portfolio explicitly rewards this diversity: it gives less weight to model pairs
    # that have high positive covariance (= same failure modes).
    #
    # the data: we have ~2500 driver-race OOF observations and only 4 models.
    # that gives q = N_models / N_obs ≈ 4/2500 ≈ 0.0016 - an extremely low ratio.
    # at this q, the Marchenko-Pastur bulk is tiny and essentially ALL covariance structure
    # in the 4×4 matrix is genuine signal (not sampling noise). the MP step acts as a
    # sanity check confirming the full matrix can be trusted, and normalizes out any
    # trace differences before optimization. at high q (few observations, many models)
    # MP would do real shrinkage; here it's mostly a principled pre-processing step.
    #
    # regularization: we tried unconstrained min-variance and it collapsed to ~80% RF
    # because RF has lower score variance (it's more conservative) even when it's also
    # less accurate on winners. we added two guards:
    #   - L2 penalty toward equal weights: λ ||w - 0.25||²  (λ = trace(C_clean)/4)
    #   - hard bounds: 10% ≤ w_i ≤ 50% per model
    # both are essential to maintain ensemble diversity - min-variance without them
    # produces a degenerate solution that destroys the diversity benefit.
    #
    # ref: Laloux et al. (1999) PRL 83:1467 - first RMT application to portfolio weights
    # ref: Ledoit & Wolf (2004) JMVA 88(2):365 - covariance shrinkage for portfolios
    # ref: Markowitz (1952) J. Finance 7(1):77 - original portfolio optimization theory
    # ref: Hoeting et al. (1999) Statist. Sci. 14(4):382 - Bayesian Model Averaging

    # collect OOF rank-normalized scores from every LOSO holdout fold
    # each fold stored rank-normalized scores (0 = predicted winner, 1 = predicted last)
    # stacking them gives shape (total_driver_race_rows, 4)
    S = np.column_stack([
        np.hstack([fold_cache[y]["xgb_norm"]    for y in years]),   # XGB regressor OOF scores
        np.hstack([fold_cache[y]["ranker_norm"]  for y in years]),   # XGB ranker OOF scores
        np.hstack([fold_cache[y]["rf_norm"]      for y in years]),   # RF win-prob OOF scores
        np.hstack([fold_cache[y]["et_norm"]      for y in years]),   # ExtraTrees OOF scores
    ])  # shape: (N_total_rows, 4) - all driver appearances across all holdout seasons

    # 4×4 covariance matrix of model rank-normalized scores
    # np.cov centers each column automatically (subtracts the column mean)
    # so off-diagonal terms capture "do these two models agree/disagree on driver rankings?"
    # high covariance = models agree on predictions = correlated errors = less diversity value
    C_models = np.cov(S.T)   # (4, 4) symmetric positive semi-definite matrix

    # apply MP denoising: confirm the noise-signal boundary
    # with q ≈ 0.002, essentially all structure is signal - this is more of a diagnostic
    # than an aggressive shrinkage step, but it's the right framework for what we're doing.
    # T_eff = N_total_rows (one OOF prediction per driver per race)
    T_eff   = float(len(S))
    C_clean = rmt_denoise_psd(C_models, T_eff=T_eff)

    # MP diagnostic: how many eigenvalues rise above the noise floor?
    # at q=0.002 the bulk is essentially a delta function at zero, so all 4 will be signal.
    # if fewer than 4 signal eigenvalues are found it means the covariance is very small
    # (models are near-identical) - still valid, just means less diversity benefit.
    q_ratio   = 4.0 / max(T_eff, 1.0)
    sigma2_mp = float(np.mean(np.diag(C_models)))   # mean diagonal as noise level estimate
    _, lam_mp = marchenko_pastur_edges(q_ratio, sigma2=sigma2_mp)
    eig_vals  = np.linalg.eigvalsh(C_models)
    n_signal  = int((eig_vals > lam_mp).sum())

    if verbose:
        print(f"\n  RMT minimum-variance portfolio diagnostics:")
        print(f"    OOF rows: {int(T_eff)}, q={q_ratio:.5f} (expect all 4 eigenvalues as signal at this q)")
        print(f"    MP λ_max={lam_mp:.5f}, signal eigenvalues={n_signal}/4")
        print(f"    Score variances: xgb={C_models[0,0]:.4f}  ranker={C_models[1,1]:.4f}  "
              f"rf={C_models[2,2]:.4f}  et={C_models[3,3]:.4f}")

    # regularization strength: L2 penalty equal in scale to the portfolio variance
    # trace(C_clean)/4 is the "average eigenvalue" of C_clean, which approximates the
    # typical portfolio variance at equal weights - this sets λ so that deviating
    # too far from equal weights is penalized proportionally to the actual variance magnitudes
    lam = float(np.trace(C_clean)) / 4.0

    def portfolio_objective(w):
        # objective = portfolio variance + L2 regularization toward equal weights
        # we minimize variance of (w^T * score_vector) across all race-driver observations
        # plus a penalty that pulls weights back toward 0.25 (equal blend) if the data
        # doesn't strongly favour any other configuration
        portfolio_var = float(w @ C_clean @ w)
        reg_penalty   = lam * float(np.sum((w - 0.25) ** 2))
        return portfolio_var + reg_penalty

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]   # weights sum to 1
    bounds      = [(0.10, 0.50)] * 4   # each model: at least 10%, at most 50%
    w0          = np.array([0.25, 0.25, 0.25, 0.25])   # start from equal weights

    try:
        result = minimize(portfolio_objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-9, "maxiter": 500})
        if result.success:
            w_opt = np.clip(result.x, 0.10, 0.50)   # hard clip after optimization
            w_opt = w_opt / w_opt.sum()              # renormalize to exactly 1.0
        else:
            # SLSQP failed to converge - use equal weights as a safe fallback
            # this shouldn't happen often but is possible with very correlated models
            if verbose:
                print(f"    [!] Portfolio optimizer did not converge ({result.message}), using equal weights")
            w_opt = np.array([0.25, 0.25, 0.25, 0.25])
    except Exception as exc:
        if verbose:
            print(f"    [!] Portfolio optimization failed ({exc}), using equal weights")
        w_opt = np.array([0.25, 0.25, 0.25, 0.25])

    if verbose:
        print(f"    Optimal portfolio: xgb={w_opt[0]:.3f}  ranker={w_opt[1]:.3f}  "
              f"rf={w_opt[2]:.3f}  et={w_opt[3]:.3f}")

    return {
        "w_xgb":    float(w_opt[0]),
        "w_ranker": float(w_opt[1]),
        "w_rf":     float(w_opt[2]),
        "w_et":     float(w_opt[3]),
    }


# ──────────────────────────────────
# Our evaluation helpers
# ──────────────────────────────────

def rank_predictions_per_race(df_meta, y_pred_pos):
    # the regressor spits out raw numbers like [3.2, 7.8, 1.4 ...] which aren't real positions
    # so we rank them within each race to get proper 1, 2, 3, ... P20 integers
    df = df_meta[["Year", "Round", "EventName", "Driver", "Team"]].copy()
    df["raw_pred"] = np.asarray(y_pred_pos)
    df["PredictedPosition"] = (
        df.groupby(["Year", "Round"])["raw_pred"]          # group per race
          .rank(method="first", ascending=True)             # lowest raw_pred = P1
          .astype(int)
    )
    return df.sort_values(["Year", "Round", "PredictedPosition"])


def winner_accuracy(df_meta, y_pred_pos, y_true_pos):
    # the main headline metric: what fraction of races did we correctly name the winner?
    # for each race, whoever we predicted lowest (closest to P1) is our called winner
    # then we check if that matches who actually won
    df = df_meta[["Year", "Round", "Driver"]].copy()
    df["pred_pos"] = np.asarray(y_pred_pos)
    df["true_pos"] = np.asarray(y_true_pos)

    correct = 0
    total   = 0
    for (year, rnd), group in df.groupby(["Year", "Round"]):
        if group["true_pos"].isna().all():  # skip future races (no actuals yet)
            continue
        pred_winner = group.loc[group["pred_pos"].idxmin(), "Driver"]  # lowest pred = winner
        true_winners = group[group["true_pos"] == 1]
        if true_winners.empty:
            continue
        correct += int(pred_winner == true_winners.iloc[0]["Driver"])
        total   += 1

    return (correct / total if total > 0 else 0.0), correct, total


def top10_set_accuracy(df_meta, y_pred_pos, y_true_pos, k=10):
    # measures how well we predicted who ends up in the top 10 using Jaccard similarity
    # Jaccard is basically: |overlap| / |union| - so 1.0 means we got every top-10 driver right
    # and 0.0 means we predicted completely the wrong set of drivers
    # ref: https://en.wikipedia.org/wiki/Jaccard_index
    df = df_meta[["Year", "Round", "Driver"]].copy()
    df["pred_pos"] = np.asarray(y_pred_pos)
    df["true_pos"] = np.asarray(y_true_pos)

    scores = []
    for (year, rnd), group in df.groupby(["Year", "Round"]):
        if group["true_pos"].isna().any():  # skip if incomplete
            continue
        pred_top = set(group.nsmallest(k, "pred_pos")["Driver"])
        true_top = set(group[group["true_pos"] <= k]["Driver"])
        if not pred_top and not true_top:
            continue
        inter = len(pred_top & true_top)
        union = len(pred_top | true_top)
        scores.append(inter / union if union > 0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def podium_accuracy(df_meta, y_pred_pos, y_true_pos):
    # two numbers: exact podium match (all 3 right) and partial credit (average overlap out of 3)
    # exact match is pretty hard - we typically hit it about 20-30% of the time in practice
    df = df_meta[["Year", "Round", "Driver"]].copy()
    df["pred_pos"] = np.asarray(y_pred_pos)
    df["true_pos"] = np.asarray(y_true_pos)

    exact_matches  = 0
    overlap_counts = []
    total = 0

    for (year, rnd), group in df.groupby(["Year", "Round"]):
        if group["true_pos"].isna().any():
            continue
        # same idea as top10_set_accuracy but for k=3 (the podium)
        pred_top3 = set(group.nsmallest(3, "pred_pos")["Driver"])
        true_top3 = set(group[group["true_pos"] <= 3]["Driver"])
        if not true_top3:
            continue
        overlap = len(pred_top3 & true_top3)
        overlap_counts.append(overlap)
        exact_matches += int(pred_top3 == true_top3)
        total += 1

    exact_rate  = exact_matches / total if total > 0 else 0.0
    avg_overlap = float(np.mean(overlap_counts)) if overlap_counts else 0.0
    return exact_rate, avg_overlap, total


def average_position_mae(df_meta, y_pred_pos, y_true_pos):
    # the average difference between where we said a driver would finish and where they actually did
    # e.g. MAE=3.2 means on average we were off by about 3 places
    # ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    ranked = rank_predictions_per_race(df_meta, y_pred_pos)
    merged = ranked[["Year", "Round", "Driver", "PredictedPosition"]].copy()
    merged["TruePosition"] = np.asarray(y_true_pos)
    merged = merged.dropna(subset=["TruePosition"])
    if merged.empty:
        return np.nan
    return mean_absolute_error(merged["TruePosition"], merged["PredictedPosition"])


# ──────────────────────────────────
# Our model training
# ──────────────────────────────────

def tune_xgboost(X_train, y_train, n_iter=80):
    # so basically we need to find the best "settings" for XGBoost
    # things like how deep the trees go, how fast it learns, how much it regularizes
    # we can't just try every combination (that's grid search, and it's way too slow)
    # instead we randomly sample 80 combinations and take the best one - turns out
    # random search works almost as well as exhaustive search for this kind of problem
    # ref: Bergstra & Bengio 2012 "Random Search for Hyper-Parameter Optimization"
    #      https://www.jmlr.org/papers/v13/bergstra12a.html
    # if you want to be even smarter about it, Optuna does proper Bayesian hyperparameter search:
    #   https://github.com/optuna/optuna  (way better than random but adds a dependency)
    #   https://github.com/fmfn/BayesianOptimization

    # these ranges came from a few days of manual trial and error - some intuition:
    # max_depth 3-6: shallower trees overfit less with only ~2500 rows
    # learning_rate very low: more trees + slower learning = smoother final model
    # reg_alpha (L1) and reg_lambda (L2): these are the "don't go crazy" penalties - they helped a lot
    # colsample_bynode: adds extra randomness at each split, pushes the model toward ensemble diversity
    # ref: XGBoost parameter tuning guide: https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
    param_dist = {
        "n_estimators":      randint(300, 1000),     # more trees = better but slower
        "max_depth":         randint(3, 7),           # deep trees overfit with this dataset size
        "learning_rate":     uniform(0.005, 0.095),   # 0.005 to 0.10
        "subsample":         uniform(0.55, 0.35),     # 0.55-0.90 per tree bootstrap
        "colsample_bytree":  uniform(0.45, 0.40),     # 0.45-0.85 feature fraction per tree
        "colsample_bylevel": uniform(0.50, 0.45),     # 0.50-0.95 per tree level
        "colsample_bynode":  uniform(0.50, 0.45),     # 0.50-0.95 per split node (extra randomness)
        "min_child_weight":  randint(3, 12),           # prevents tiny leaf nodes
        "reg_alpha":         uniform(0.0, 1.5),       # L1 regularization
        "reg_lambda":        uniform(1.0, 4.0),       # L2 regularization - strong floor helps
        "gamma":             uniform(0.0, 0.8),       # min gain to make a split - controls tree complexity
    }

    base_model = xgb.XGBRegressor(
        random_state=SEED,
        verbosity=0,
        n_jobs=1,       # n_jobs=1 here on purpose: RandomizedSearchCV(n_jobs=-1) below handles
                        # the outer parallelism. if you set both to -1 on Windows it spawns
                        # way too many processes and crashes with a C stack error (this was annoying to debug)
        tree_method="hist",
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=SEED,
        n_jobs=1,       # sequential on Windows - nested loky process spawn causes XGBoost DMatrix crash
        verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"    Best XGBoost MAE (3-fold CV): {-search.best_score_:.3f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def tune_random_forest_winner(X_train, y_win_train, n_iter=60):
    # predicting the race winner is a lopsided problem: only 1 driver out of 20 wins each race
    # so if the model just always says "not a winner" it'd be 95% accurate - which is useless
    # the trick here is two things:
    #   1. class_weight='balanced_subsample': this tells each tree "treat the rare winner class
    #      as if there are as many winners as non-winners" - balancing at the tree level is
    #      better than doing it globally according to sklearn docs
    #   2. we score on F1 instead of accuracy - F1 actually punishes you for ignoring winners
    # ref: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    param_dist = {
        "n_estimators":     randint(300, 800),       # 300-800 trees
        "max_depth":        [4, 5, 6, 7, 8, 10, None],  # None = fully grown
        "min_samples_leaf": randint(2, 12),           # avoid tiny leaves
        "max_features":     ["sqrt", "log2", 0.3, 0.4, 0.5],  # feature subset per split
        "criterion":        ["gini", "entropy"],     # information gain criteria
        "max_samples":      uniform(0.6, 0.35),      # bootstrap sample fraction 0.60-0.95
    }

    base = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,       # same n_jobs=1 trick: outer RandomizedSearchCV handles parallelism
    )

    valid = y_win_train.dropna()
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="f1",
        random_state=SEED,
        n_jobs=1,       # sequential - same Windows loky spawn crash fix as XGBoost above
        verbose=0,
    )
    search.fit(X_train.loc[valid.index], valid.astype(int))

    print(f"    Best RF winner F1 (3-fold CV): {search.best_score_:.3f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def tune_random_forest_top10(X_train, y_top10_train, n_iter=60):
    # top-10 classification: roughly 50% positive rate, less imbalanced than winner
    # same parameter space as winner RF, still using balanced_subsample for consistency

    param_dist = {
        "n_estimators":     randint(200, 700),
        "max_depth":        [4, 5, 6, 7, 8, 10, None],
        "min_samples_leaf": randint(1, 10),
        "max_features":     ["sqrt", "log2", 0.4, 0.5, 0.6],
        "criterion":        ["gini", "entropy"],
        "max_samples":      uniform(0.6, 0.35),
    }

    base = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,       # same Windows-safe fix as the winner RF above
    )

    valid = y_top10_train.dropna()
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="f1",
        random_state=SEED,
        n_jobs=1,       # sequential - Windows crash fix
        verbose=0,
    )
    search.fit(X_train.loc[valid.index], valid.astype(int))

    print(f"    Best RF top-10 F1 (3-fold CV): {search.best_score_:.3f}")
    return search.best_estimator_, search.best_params_


# the functions below are the "lite" versions of the trainers used inside the LOSO loop
# we don't run the full hyperparameter search per fold - that would make LOSO take hours
# instead we use these sensible defaults that came from a few earlier manual tuning sessions

def rmt_season_signal_quality(X_train_all, df_train, years):
    # for each season, compute the fraction of feature variance that's genuine signal
    # (above the Marchenko-Pastur bulk) rather than estimation noise.
    #
    # the idea: not all seasons teach the models equally well. a season where pre-race
    # features (qualifying gaps, rolling form, etc.) cleanly predict race outcomes has
    # "high signal" - the feature→outcome relationship is stable and learnable. a season
    # where anomalies dominate (crashes, weather, random strategy calls) has more noise
    # - the feature→outcome relationship is less reliable.
    #
    # we measure this by building each season's (n_rows × n_features) feature matrix,
    # computing the feature-feature covariance, and applying MP denoising.
    # signal_fraction = sum(eigenvalues above λ_max) / trace(C)
    # tells us what fraction of feature variation sits in real signal eigenmodes.
    #
    # seasons with high signal fraction should contribute more to training because
    # they provide clearer feature→outcome examples for the models to learn from.
    # ref: same Laloux (1999) framework - eigenvalues above MP bulk are real signal.
    quality = {}
    for yr in years:
        yr_mask = df_train["Year"] == yr
        X_yr    = X_train_all[yr_mask]
        if len(X_yr) < 10:   # need at least 10 rows to get a meaningful covariance estimate
            quality[yr] = 0.5   # neutral weight if not enough data
            continue
        X_np   = X_yr.values.astype(float)
        X_np   = np.nan_to_num(X_np, nan=0.0)   # replace NaN with 0 for covariance computation
        X_mean = X_np.mean(axis=0)
        X_std  = np.std(X_np, axis=0).clip(min=1e-8)   # avoid division by zero on constant features
        X_s    = (X_np - X_mean) / X_std   # standardize so all features contribute equally

        # feature-feature covariance: (n_features × n_features), T_eff = n_rows per season
        # q = n_features / n_rows ≈ 54 / 400 ≈ 0.135 for a typical season
        # at this q most eigenvalues are signal - but the FRACTION in signal vs bulk still varies
        C_feat = X_s.T @ X_s / float(len(X_s))   # shape (54, 54)
        T_eff  = float(len(X_s))

        try:
            _, n_sig, lam_thr = rmt_denoise_psd(C_feat, T_eff=T_eff, return_diagnostics=True)
            eig_vals     = np.linalg.eigvalsh(C_feat)
            signal_mass  = float(np.sum(eig_vals[eig_vals > lam_thr]))   # total variance in signal modes
            total_mass   = max(float(np.trace(C_feat)), 1e-8)              # total variance in all modes
            quality[yr]  = float(np.clip(signal_mass / total_mass, 0.0, 1.0))
        except Exception:
            quality[yr] = 0.5   # fallback to neutral if anything fails

    return quality


def compute_fold_sample_weights(years_train, holdout_year, rmt_quality=None):
    # the idea here is simple: if we're trying to predict 2024, the 2023 season is way more
    # relevant than 2018 was - F1 cars and teams change too much year to year
    # so we give closer years more weight during training
    # |year - holdout| = 1 → weight 2.25, = 2 → 1.50, = 3 → 1.00, > 3 → 0.50
    # it's a soft exponential-ish decay anchored at 3 years out
    # this really helps on the 2022/2024 folds which are "transition year" folds where
    # regulation changes mean older data is actively misleading
    #
    # the optional rmt_quality dict adds a second signal: how much genuine feature
    # signal (above the MP bulk) did each season have? seasons where pre-race features
    # cleanly predicted outcomes provide better training examples than noisy seasons.
    # the RMT multiplier is conservative: 0.85 base + 0.15 * signal_fraction, so
    # it ranges from 0.85 (no signal) to 1.0 (all signal). this prevents the RMT
    # quality from ever completely zeroing out a season - temporal distance is the
    # dominant factor, signal quality is a gentle nudge.
    weight_map = {}
    for y in years_train:
        dist = abs(y - holdout_year)
        if dist <= 1:
            base = 2.25
        elif dist <= 2:
            base = 1.50
        elif dist <= 3:
            base = 1.00
        else:
            base = 0.50

        # blend in RMT signal quality: high signal → slight boost, low signal → slight penalty
        # the 0.85 floor ensures we never reduce a season's weight by more than 15%,
        # keeping the temporal structure dominant while adding the quality correction
        if rmt_quality is not None and y in rmt_quality:
            rmt_factor = 0.85 + 0.15 * float(rmt_quality[y])
        else:
            rmt_factor = 1.0   # no RMT correction if quality not provided

        weight_map[y] = base * rmt_factor

    return weight_map


def train_xgboost_regressor(X_train, y_train, params=None, sample_weight=None):
    # this is the "fast" version used inside the LOSO loop where we need to train one per holdout year
    # if we ran the full 80-iteration random search per fold the validation would take hours
    # so if no params are provided we fall back to a set of defaults that work pretty well
    if params is None:
        # these are the defaults I landed on after a few days of poking around manually
        # they're a bit more conservative (shallower, more regularized) than what the full search finds
        params = {
            "n_estimators": 400, "max_depth": 4, "learning_rate": 0.04,
            "subsample": 0.75, "colsample_bytree": 0.70, "colsample_bylevel": 0.80,
            "min_child_weight": 5, "reg_alpha": 0.3, "reg_lambda": 2.0, "gamma": 0.1,
        }
    model = xgb.XGBRegressor(
        **params, random_state=SEED, verbosity=0, n_jobs=-1, tree_method="hist",
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_random_forest_winner(X_train, y_win_train, params=None, sample_weight=None):
    # same idea as the XGBoost fallback above - quick RF with preset defaults for the LOSO loop
    if params is None:
        params = {
            "n_estimators": 500, "max_depth": 7, "min_samples_leaf": 4,
            "max_features": "sqrt", "criterion": "gini",
        }
    model = RandomForestClassifier(
        **params, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
    )
    valid = y_win_train.dropna()
    sw = sample_weight[valid.index] if sample_weight is not None else None
    model.fit(X_train.loc[valid.index], valid.astype(int), sample_weight=sw)
    return model


def train_random_forest_top10(X_train, y_top10_train, params=None, sample_weight=None):
    # same deal - quick fallback RF for top-10 classification inside the LOSO loop
    if params is None:
        params = {
            "n_estimators": 500, "max_depth": 7, "min_samples_leaf": 3,
            "max_features": "sqrt", "criterion": "gini",
        }
    model = RandomForestClassifier(
        **params, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
    )
    valid = y_top10_train.dropna()
    sw = sample_weight[valid.index] if sample_weight is not None else None
    model.fit(X_train.loc[valid.index], valid.astype(int), sample_weight=sw)
    return model


def train_xgboost_ranker(X_train, y_pos_train, groups, best_params=None, sample_weight=None):
    # the ranker is fundamentally different from the regressor - here's the deal:
    # the regressor says "predict Hamilton finishes in P3.2" (a raw number)
    # the ranker doesn't care about raw numbers - it just wants to get the ORDER right within each race
    # it uses NDCG (Normalized Discounted Cumulative Gain) as its objective, which basically means:
    # "getting the top positions in the right order matters more than getting the back of the grid right"
    # think of it like: correctly ranking Verstappen vs Hamilton vs Leclerc is worth more
    # than correctly ordering the midfield scramblers in P14-P16
    # ref: NDCG definition: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    # ref: XGBoost learning-to-rank: https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
    # ref: demo code from XGBoost repo: https://github.com/dmlc/xgboost/tree/master/demo

    # XGBRanker wants "relevance scores" not positions - so P1 gets the highest score, P20 the lowest
    # we do max_pos + 1 - position, which flips 1→20, 2→19 etc. clip to minimum 1 (no zeros allowed)
    max_pos   = int(y_pos_train.fillna(20).max())
    relevance = (max_pos + 1 - y_pos_train.fillna(max_pos)).astype(int).clip(lower=1)

    # we reuse the same hyperparams as the XGBoost regressor - same tree structure, just different objective
    # depth/learning rate/regularization all transfer well between the two
    # if best_params is None (happens in the LOSO loop before the full tuning has run) fall back to tested defaults
    if best_params is None:
        best_params = {
            "n_estimators": 400, "max_depth": 4, "learning_rate": 0.04,
            "subsample": 0.75, "colsample_bytree": 0.70, "colsample_bylevel": 0.80,
            "min_child_weight": 5, "reg_alpha": 0.3, "reg_lambda": 2.0, "gamma": 0.1,
        }
    ranker_params = dict(best_params)
    ranker_params.setdefault("n_estimators", 500)       # default if not in best_params
    ranker_params.setdefault("colsample_bylevel", 0.8)  # might be missing from older tuning runs
    ranker_params.setdefault("colsample_bynode",  0.8)  # same

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        **ranker_params,
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, relevance, group=groups, sample_weight=sample_weight)
    return model


def tune_extratrees(X_train, y_train, n_iter=60):
    # ExtraTrees is the 4th model we throw into the ensemble - it's actually the most "random" of the bunch
    # here's the family tree: XGBoost (boosting, sequential), RF (bagging + best split), ExtraTrees (bagging + RANDOM split)
    # the key thing ExtraTrees does differently from RandomForest:
    #   RF says: "for each split, try a random subset of features and pick the BEST threshold"
    #   ExtraTrees says: "for each feature in that subset, just pick ONE random threshold and go with it"
    # this sounds like it'd be worse, but it means ExtraTrees "disagrees" with XGB and RF on
    # different races - and diverse disagreement is exactly what makes ensembles work
    # basically we want models that are wrong about different things, not the same things
    # ref: Geurts, Ernst, Wehenkel 2006: https://link.springer.com/article/10.1007/s10994-006-6226-1
    # ref: sklearn comparison: https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees

    param_dist = {
        "n_estimators":     randint(300, 800),
        "max_depth":        [5, 6, 7, 8, 10, 12, None],
        "min_samples_leaf": randint(2, 12),
        "max_features":     ["sqrt", "log2", 0.4, 0.5, 0.6],
        "max_samples":      uniform(0.6, 0.35),  # bootstrap sample fraction
    }

    base = ExtraTreesRegressor(
        random_state=SEED,
        n_jobs=1,    # Windows-safe: RandomizedSearchCV(n_jobs=-1) handles the outer parallelism
        bootstrap=True,  # need bootstrap=True to use max_samples
    )

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=SEED,
        n_jobs=1,       # sequential - Windows crash fix
        verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"    Best ExtraTrees MAE (3-fold CV): {-search.best_score_:.3f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def train_extratrees(X_train, y_train, params=None, sample_weight=None):
    # quick ExtraTrees fallback for the LOSO loop - no hyperparameter search, just run it
    if params is None:
        # random splits need a bit more depth to compensate for being less optimal than RF splits
        # so these defaults are slightly deeper than the RF ones
        params = {
            "n_estimators": 500, "max_depth": 8, "min_samples_leaf": 3,
            "max_features": "sqrt", "max_samples": 0.8,
        }
    model = ExtraTreesRegressor(
        **params, random_state=SEED, n_jobs=-1, bootstrap=True,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def compute_race_groups(df_meta):
    # XGBRanker needs to know where each race "group" starts and ends
    # so we just count how many drivers are in each race and return that as a list
    # looks like [20, 20, 19, 20, ...] - one number per race, must match sort order
    return df_meta.groupby(["Year", "Round"], sort=False).size().tolist()


# ──────────────────────────────────
# How we normalize rank and the ensemble
# ──────────────────────────────────

def _rank_normalize(scores, df_meta, ascending=True):
    # here's the problem: our three models output completely different things
    #   XGBReg spits out numbers like 3.2 or 14.8 (position estimates, sometimes weirdly above 20)
    #   XGBRanker outputs large relevance scores like 142.0 or 7.3 with no obvious scale
    #   RF outputs probabilities between 0 and 1
    # you can't just average those directly - the scales are totally incompatible
    # so we rank each driver within their race and map that to [0,1]
    # after this: all models are on the same scale with 0 = predicted winner, 1 = predicted last
    #
    # ascending=True  means "lower raw score = better" (XGBReg position, inverted RF prob)
    # ascending=False means "higher raw score = better" (XGBRanker relevance)

    df = df_meta[["Year", "Round"]].copy().reset_index(drop=True)
    df["raw"] = np.asarray(scores)

    df["rank"] = df.groupby(["Year", "Round"])["raw"].rank(
        method="average", ascending=ascending
    )

    n = df.groupby(["Year", "Round"])["raw"].transform("count")

    # normalize: (rank - 1) / (n - 1) puts everything in [0,1]
    # clip denominator to 1 to handle the rare case of a 1-driver race
    df["norm"] = (df["rank"] - 1) / (n - 1).clip(lower=1)

    return df["norm"].values


def ensemble_predict(xgb_raw_pos, rf_win_proba, df_meta,
                     ranker_raw_scores=None,
                     et_raw_pos=None,
                     ensemble_weights=None,
                     X_features=None,
                     apply_bayesian_prior=True,
                     cal_temperature=1.0,
                     meta_model=None):
    # this is where everything comes together - we take the four model outputs and blend them
    # into one final predicted race order
    #
    # the ORDER of steps here matters a lot - we got this wrong once and it was subtle to spot:
    #
    #   step 1: temperature calibration
    #     the RF is overconfident ("I'm 97% sure Verstappen wins") so first we squeeze its
    #     probabilities toward 0.5 using the temperature T we fit from LOSO predictions
    #
    #   step 2: Bayesian circuit-history prior blend
    #     now mix the calibrated RF probability with the James-Stein prior
    #     (the "Hamilton has won at Silverstone 8 times, factor that in" part)
    #     we have to do step 1 BEFORE this, otherwise the overconfident RF probability
    #     just drowns out the prior entirely since it's so extreme
    #
    #   step 3: rank normalize all four models and blend with BMA weights
    #     or if a stacking meta-model is available, use that instead of the hand-tuned weights
    #
    # ref: ensemble blending discussion: https://github.com/MLWave/Kaggle-Ensemble-Guide
    # ref: rank aggregation, majority rules (Condorcet): https://en.wikipedia.org/wiki/Condorcet_method

    df = df_meta[["Year", "Round", "EventName", "Driver", "Team"]].copy().reset_index(drop=True)

    # step 1: temperature calibration (Platt scaling with log-normal MAP prior)
    rf_cal = calibrate_with_temperature(np.asarray(rf_win_proba), cal_temperature)

    # step 2: Bayesian circuit-history prior blend (Normal-Normal hierarchical shrinkage)
    if apply_bayesian_prior and X_features is not None:
        xf = X_features.reset_index(drop=True)
        rf_adj = bayesian_adjust_win_proba(rf_cal, xf)
    else:
        rf_adj = rf_cal

    df["rf_win_proba"] = rf_adj
    df["xgb_raw_pred"] = np.asarray(xgb_raw_pos)

    if ranker_raw_scores is not None:
        # four-model blend with rank normalization
        xgb_norm    = _rank_normalize(xgb_raw_pos,        df, ascending=True)
        ranker_norm = _rank_normalize(ranker_raw_scores,   df, ascending=False)
        rf_norm     = _rank_normalize(1.0 - rf_adj,        df, ascending=True)  # invert so 0 = likely winner
        # ExtraTrees: same direction as XGB (lower predicted position = better)
        et_norm     = _rank_normalize(et_raw_pos, df, ascending=True) if et_raw_pos is not None else xgb_norm

        if ensemble_weights is not None:
            w_xgb    = ensemble_weights["w_xgb"]
            w_ranker = ensemble_weights["w_ranker"]
            w_rf     = ensemble_weights["w_rf"]
            w_et     = ensemble_weights.get("w_et", 0.0)  # backward compat if old weights dict used
        else:
            # 35/35/15/15 equal split between the four models as fallback
            w_xgb, w_ranker, w_rf, w_et = 0.35, 0.35, 0.15, 0.15

        if meta_model is not None:
            # if we have a stacking meta-model (which we usually do) use that instead of the BMA weights
            # the meta-model is a tiny XGBoost (depth=2) trained to figure out how much to trust each model
            # it gets 11 features: the 4 normalized model scores + 5 context features like grid position
            # and whether it's a street circuit + 2 RMT features (blend score, disagreement).
            # we tried LogisticRegression here first but XGB depth=2 beats it by about 2% on LOSO
            # because it can learn interactions like "both XGB AND Ranker agree AND low overtaking"
            # that LR simply can't represent
            # reg_era and is_street were added later so the meta knows which era it's in
            # rmt_ens: BMA-weighted blend of the 4 scores - the "consensus" prediction
            # rmt_disagr: std across the 4 scores - how much the models disagree for this driver/race
            #   high disagreement flags chaotic/unpredictable races - genuinely new info for depth-2 XGB
            rmt_ens_inf  = (w_xgb * xgb_norm + w_ranker * ranker_norm
                            + w_rf * rf_norm + w_et * et_norm)
            rmt_dis_inf  = np.std(np.column_stack([xgb_norm, ranker_norm, rf_norm, et_norm]),
                                  axis=1, ddof=0)
            if X_features is not None:
                xf_reset = X_features.reset_index(drop=True)
                meta_context = np.column_stack([
                    xgb_norm, ranker_norm, rf_norm, et_norm,
                    xf_reset["grid_position_pct"].values,
                    xf_reset["circuit_overtaking_score"].values,
                    xf_reset["championship_rank_pct"].fillna(0.5).values,
                    xf_reset["car_regulation_era"].values,
                    xf_reset["circuit_is_street"].values,
                    rmt_ens_inf, rmt_dis_inf,
                ])
            else:
                # fallback if X_features not provided - fill unknowns with neutral values
                meta_context = np.column_stack([
                    xgb_norm, ranker_norm, rf_norm, et_norm,
                    np.zeros(len(xgb_norm)),
                    np.full(len(xgb_norm), 0.45),
                    np.full(len(xgb_norm), 0.5),
                    np.ones(len(xgb_norm)),   # assume ground-effect era (2022+)
                    np.zeros(len(xgb_norm)),  # assume not street circuit
                    rmt_ens_inf, rmt_dis_inf,
                ])
            meta_win_prob = meta_model.predict_proba(meta_context)[:, 1]
            df["ensemble_score"] = 1.0 - meta_win_prob
        else:
            df["ensemble_score"] = (w_xgb * xgb_norm + w_ranker * ranker_norm
                                    + w_rf * rf_norm + w_et * et_norm)

    else:
        # fallback: two-model blend without ranker (shouldn't normally hit this)
        df["ensemble_score"] = np.asarray(xgb_raw_pos) + 0.3 * (1.0 - rf_adj)

    # rank by ensemble score within each race (lower = predicted winner)
    df["PredictedPosition"] = (
        df.groupby(["Year", "Round"])["ensemble_score"]
          .rank(method="first", ascending=True)
          .astype(int)
    )

    return df.sort_values(["Year", "Round", "PredictedPosition"])


# ──────────────────────────────────
# LOSO validation
# ──────────────────────────────────

def leave_one_season_out_validation(df_train, X_train_all, y_pos, y_win, y_top10):
    # this is how we honestly evaluate the model: hold out one entire season at a time,
    # train on everything else, and see how it does on the season it never saw
    # for example: train on 2019/2022/2023/2024, predict 2021, see how accurate we were
    # then train on 2019/2021/2023/2024, predict 2022, etc.
    #
    # why not just do normal k-fold cross validation? because k-fold shuffles rows randomly,
    # so a 2024 race could end up in the training set for a 2023 holdout - that's data leakage,
    # future races "helping" us predict past ones, which is cheating
    # with temporal data you always validate forward in time
    # ref: sklearn time series splitting: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
    #
    # the whole function runs in two phases:
    #   phase 1 (the per-year loop): train the four base models on each fold, collect out-of-fold
    #     predictions for BMA weight computation and temperature calibration
    #   phase 2 (after the loop): train the stacking meta-model using those OOF predictions
    #     this replaces the hand-tuned 40/40/20 blend weights with something learned from data
    #     ref: stacking/blending overview: https://en.wikipedia.org/wiki/Ensemble_learning#Stacking

    print("\n" + "─" * 55)
    print("  Leave-One-Season-Out Validation (base models)")
    print("─" * 55)

    years   = sorted(df_train["Year"].unique())
    results = []

    # track how many winners each model calls correctly across all folds
    # also track per-fold MAE - the BMA weight computation needs both signals
    model_correct = {"xgb": 0, "ranker": 0, "rf": 0, "et": 0}
    model_mae_acc = {"xgb": [], "ranker": [], "rf": [], "et": []}
    total_validated_races = 0

    # we also collect all the RF win probabilities from every holdout fold into one big pool
    # this is the OOF (out-of-fold) set we'll use to fit the temperature calibration T
    # why wait until after the loop? because T is fit on predictions the RF never trained on -
    # if we fit T per fold we'd be wasting data, collecting all folds first gives us more
    oof_rf_proba  = []
    oof_rf_labels = []

    # fold_cache stores the rank-normalized scores from each fold for the phase-2 stacking
    # think of it as saving the answer sheets from each fold so the meta-model can learn from them
    fold_cache = {}

    # compute per-season RMT signal quality as a diagnostic before the loop starts.
    # signal_quality[year] = fraction of feature variance in real signal eigenvalues (above MP bulk)
    # we use this to understand which seasons have the clearest feature→outcome structure.
    # note: we evaluated using rmt_sig_quality to scale training sample weights, but that
    # approach HURT the 2023 holdout (adjacent year 2022 had different team dynamics, so
    # upweighting 2022 data for 2023 prediction was counterproductive). the quality scores
    # are kept as informative diagnostics - they show which seasons are most predictable.
    # the compute_fold_sample_weights function is designed for production training (not LOSO).
    print(f"\n  Per-season RMT feature signal quality ({len(years)} seasons, diagnostic only):")
    rmt_sig_quality = rmt_season_signal_quality(X_train_all, df_train, years)
    for yr, sq in sorted(rmt_sig_quality.items()):
        print(f"    {yr}: signal_fraction={sq:.3f}")

    for holdout_year in years:
        train_mask = df_train["Year"] != holdout_year  # train on everything except holdout
        test_mask  = df_train["Year"] == holdout_year  # test on holdout year

        X_tr = X_train_all[train_mask]
        X_te = X_train_all[test_mask]

        # fit imputer only on training fold (never let test data influence imputation)
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=FEATURE_COLS)
        X_te = pd.DataFrame(imp.transform(X_te),     columns=FEATURE_COLS)

        y_pos_tr   = y_pos[train_mask].reset_index(drop=True)
        y_pos_te   = y_pos[test_mask].reset_index(drop=True)
        y_win_tr   = y_win[train_mask].reset_index(drop=True)
        y_top10_tr = y_top10[train_mask].reset_index(drop=True)

        meta_te = df_train[test_mask][
            ["Year", "Round", "EventName", "Driver", "Team"]
        ].reset_index(drop=True)

        # train all four base models fresh on this fold's training set
        # note: we evaluated temporal+RMT sample weighting (via compute_fold_sample_weights)
        # but it HURT the 2023 holdout fold because adjacent years (2022) had different team
        # dynamics (Ferrari competitive early-2022 vs RBR dominant all-2023), so upweighting
        # 2022 data for predicting 2023 gave the wrong signal. equal-weight training is more
        # robust across the diverse range of seasons in the dataset.
        # the compute_fold_sample_weights and rmt_season_signal_quality functions exist for
        # potential use in the production final-model training (main()), where temporal
        # weighting toward recent seasons makes more sense than in a cross-validation loop.
        xgb_model  = train_xgboost_regressor(X_tr, y_pos_tr.fillna(20))
        rf_win_mod = train_random_forest_winner(X_tr, y_win_tr)
        rf_top10   = train_random_forest_top10(X_tr, y_top10_tr)
        et_model   = train_extratrees(X_tr, y_pos_tr.fillna(20))  # 4th model: random threshold splits

        # train the ranker separately - XGBRanker needs data sorted by (Year, Round) for the group arg
        # we used to just multiply the XGBReg scores by 1.07 as a quick approximation, but
        # training it properly per fold is worth the extra time
        meta_tr_fold  = df_train[train_mask][["Year", "Round"]].reset_index(drop=True)
        sort_idx_fold = meta_tr_fold.sort_values(["Year", "Round"]).index
        X_tr_srt      = X_tr.loc[sort_idx_fold].reset_index(drop=True)
        y_tr_srt      = y_pos_tr.loc[sort_idx_fold].reset_index(drop=True)
        meta_tr_srt   = meta_tr_fold.loc[sort_idx_fold].reset_index(drop=True)
        loso_groups   = compute_race_groups(meta_tr_srt)
        ranker_fold   = train_xgboost_ranker(X_tr_srt, y_tr_srt, loso_groups, best_params=None)

        # predict on holdout
        xgb_preds     = xgb_model.predict(X_te)
        rf_proba      = rf_win_mod.predict_proba(X_te)[:, 1]
        ranker_scores = ranker_fold.predict(X_te)
        et_preds      = et_model.predict(X_te)

        # flip signs so that for every model, "lower = more likely to win" (consistent direction)
        rf_as_pos     = 1.0 - rf_proba    # RF outputs win prob so 1-prob gives a position-like score
        ranker_as_pos = -ranker_scores     # ranker outputs relevance (higher=better), flip it
        et_as_pos     = et_preds           # ExtraTrees is already in position space

        # add this fold's RF predictions to the running pool for temperature calibration later
        # we pool ALL folds together before fitting T - more data = better T estimate
        oof_rf_proba.extend(rf_proba.tolist())
        oof_rf_labels.extend(y_win[test_mask].fillna(0).astype(int).tolist())

        # save the rank-normalized scores for phase-2 stacking
        # basically we're storing "what each model predicted for each driver in this holdout year"
        # so that after the loop we can train the meta-model on those predictions
        # we also include 5 context features so the meta can learn things like:
        #   "at street circuits, qualifying position is even more reliable than usual"
        #   "championship leader is usually in the fastest car - trust XGB's qualifying signal"
        #   "high overtaking track: RF win prob matters more, grid position matters less"
        fold_cache[holdout_year] = {
            "xgb_norm":          _rank_normalize(xgb_preds, meta_te, ascending=True),
            "ranker_norm":       _rank_normalize(ranker_scores, meta_te, ascending=False),
            "rf_norm":           _rank_normalize(1.0 - rf_proba, meta_te, ascending=True),
            "et_norm":           _rank_normalize(et_preds, meta_te, ascending=True),
            "grid_pos_pct":      X_te["grid_position_pct"].values,
            "overtaking_score":  X_te["circuit_overtaking_score"].values,
            # championship_rank_pct: 0=leader (in fastest car), fill NaN with mid-field 0.5
            "champ_rank_pct":    X_te["championship_rank_pct"].fillna(0.5).values,
            "reg_era":           X_te["car_regulation_era"].values,
            "is_street":         X_te["circuit_is_street"].values,
            "meta_te":           meta_te.copy(),
            "y_pos_te":          y_pos_te.copy(),
            "y_win_te":          y_win[test_mask].reset_index(drop=True).copy(),
        }

        # count how many winners each individual model called correctly this fold
        # this feeds into the BMA weight computation after all folds are done
        _, n_xgb,    n_tot = winner_accuracy(meta_te, xgb_preds,    y_pos_te)
        _, n_rf,     _     = winner_accuracy(meta_te, rf_as_pos,     y_pos_te)
        _, n_ranker, _     = winner_accuracy(meta_te, ranker_as_pos, y_pos_te)
        _, n_et,     _     = winner_accuracy(meta_te, et_as_pos,     y_pos_te)

        model_correct["xgb"]    += n_xgb
        model_correct["ranker"] += n_ranker
        model_correct["rf"]     += n_rf
        model_correct["et"]     += n_et
        total_validated_races   += n_tot

        # also track position MAE per model per fold - the "ranking quality" half of BMA
        model_mae_acc["xgb"].append(average_position_mae(meta_te, xgb_preds,    y_pos_te))
        model_mae_acc["ranker"].append(average_position_mae(meta_te, ranker_as_pos, y_pos_te))
        model_mae_acc["rf"].append(average_position_mae(meta_te, rf_as_pos,     y_pos_te))
        model_mae_acc["et"].append(average_position_mae(meta_te, et_as_pos,     y_pos_te))

        # compute ensemble metrics for this fold using fallback weights (BMA not ready yet)
        # we can't use the real BMA weights here because they depend on ALL folds finishing first
        fold_ens = ensemble_predict(
            xgb_preds, rf_proba, meta_te,
            ranker_raw_scores=ranker_scores,
            et_raw_pos=et_preds,
            ensemble_weights=None,   # fixed fallback weights during LOSO - BMA computed after
            apply_bayesian_prior=False,
        )
        # merge ensemble predictions back to the original row order before computing metrics
        ens_aligned = meta_te.merge(
            fold_ens[["Year", "Round", "Driver", "PredictedPosition"]],
            on=["Year", "Round", "Driver"], how="left"
        )
        ens_pred_arr = ens_aligned["PredictedPosition"].values.astype(float)

        w_acc, w_correct, w_total = winner_accuracy(meta_te, ens_pred_arr, y_pos_te)
        jaccard   = top10_set_accuracy(meta_te, ens_pred_arr, y_pos_te)
        mae       = average_position_mae(meta_te, ens_pred_arr, y_pos_te)
        pod_exact, pod_overlap, _ = podium_accuracy(meta_te, ens_pred_arr, y_pos_te)

        results.append({
            "holdout_year":    holdout_year,
            "winner_accuracy": w_acc,
            "winners_correct": w_correct,
            "total_races":     w_total,
            "podium_exact":    pod_exact,
            "podium_overlap":  pod_overlap,
            "top10_jaccard":   jaccard,
            "position_mae":    mae,
        })

        print(f"\n  Holdout {holdout_year}:")
        print(f"    Winner accuracy : {w_acc:.1%}  ({w_correct}/{w_total} races)")
        print(f"    Podium exact    : {pod_exact:.1%}  (avg overlap: {pod_overlap:.2f}/3)")
        print(f"    Top-10 Jaccard  : {jaccard:.3f}")
        print(f"    Position MAE    : {mae:.2f} places")

    val_df = pd.DataFrame(results)
    print(f"\n  ── Averages ──")
    print(f"    Winner accuracy : {val_df['winner_accuracy'].mean():.1%}")
    print(f"    Podium exact    : {val_df['podium_exact'].mean():.1%}")
    print(f"    Top-10 Jaccard  : {val_df['top10_jaccard'].mean():.3f}")
    print(f"    Position MAE    : {val_df['position_mae'].mean():.2f} places")

    # all folds are done - now we have the full pool of OOF RF predictions (one per driver per race)
    # pool ALL of them together to fit the temperature calibration T
    # using the full pool gives us more data and a more stable T estimate
    print(f"\n  Fitting Bayesian temperature scaling on {len(oof_rf_proba)} OOF predictions...")
    cal_temperature = bayesian_calibrate_rf_temperature(
        np.array(oof_rf_proba), np.array(oof_rf_labels)
    )

    # average each model's MAE across folds - don't sum, since earlier folds might have more races
    mean_model_mae = {
        m: float(np.nanmean(vals)) for m, vals in model_mae_acc.items() if vals
    }

    # compute the RMT minimum-variance portfolio weights from all the OOF predictions
    # these capture the inter-model error covariance structure: pairs of models that fail
    # on the same races are identified and downweighted relative to independently-failing models.
    # the full explanation is in rmt_minimum_variance_weights() - this is the Laloux (1999)
    # RMT technique applied to ensemble weighting rather than financial portfolio construction.
    print("\n  Computing RMT minimum-variance portfolio from OOF predictions...")
    rmt_bma = rmt_minimum_variance_weights(fold_cache, years)

    # now compute the BMA weights by combining: Dirichlet (winner counts), MAE (ranking quality),
    # and RMT portfolio (joint error diversity). the rmt_portfolio is blended at 25%.
    bma_weights = bayesian_model_averaging_weights(model_correct, model_mae=mean_model_mae,
                                                   rmt_portfolio=rmt_bma)

    # ── Phase 2: the stacking meta-model ────────────────────────────────────────
    # here's the core idea: we have OOF predictions from every year, so let's train a
    # small "meta-model" that learns how to best combine the four base models
    # but we can't evaluate the meta-model on the same folds it trained on - that's leakage again
    # the solution: nested leave-one-year-out - for each holdout year, train the meta on the
    # OTHER years' OOF predictions, then test on the holdout year
    # ref: nested cross-validation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
    print("\n" + "─" * 55)
    print("  Stacking Meta-Model (nested leave-one-year-out)")
    print("─" * 55)

    meta_results = []
    for holdout_year in years:
        other_years = [y for y in years if y != holdout_year]

        # train the meta-model on all other years' OOF predictions
        # 11 features: 4 base model scores (already rank-normalized to [0,1]) + 5 context features
        #              + 2 RMT features (rmt_ens_score, rmt_disagreement)
        # the context features help the meta figure out WHEN to trust each model:
        #   champ_rank_pct: Verstappen in 2023 is basically guaranteed to win - trust XGB
        #   overtaking_score: Monaco vs Monza - how much does qualifying position matter here?
        #   grid_pos_pct: the qualifying position itself as a direct signal
        #   car_regulation_era: ground-effect cars (2022+) vs turbo-hybrid - models behave differently
        #   circuit_is_street: Monaco/Baku pole = almost certain win, amplifies the qualifying signal
        # the RMT features add two new signals the meta-model couldn't reconstruct on its own:
        #   rmt_ens_score: the Marchenko-Pastur portfolio-weighted blend (minimum-variance combination)
        #     computed from other_years only so there's no data leakage into the holdout
        #   rmt_disagreement: std across the 4 model scores per driver-race row
        #     high disagreement = chaotic/unpredictable race where we should be less confident
        #     this is genuinely new information - a depth-2 tree can't reconstruct std from raw scores
        #     ref: ensemble disagreement as uncertainty proxy: Krogh & Vedelsby (1995) NeurIPS

        # fold-specific RMT portfolio weights (computed from other_years only - no leakage into holdout)
        rmt_fold = rmt_minimum_variance_weights(fold_cache, other_years, verbose=False)

        rmt_tr_blocks = []
        for y in other_years:
            xn = np.asarray(fold_cache[y]["xgb_norm"])
            rn = np.asarray(fold_cache[y]["ranker_norm"])
            rfn = np.asarray(fold_cache[y]["rf_norm"])
            en = np.asarray(fold_cache[y]["et_norm"])
            rmt_ens_tr = (rmt_fold["w_xgb"] * xn + rmt_fold["w_ranker"] * rn
                          + rmt_fold["w_rf"] * rfn + rmt_fold["w_et"] * en)
            rmt_dis_tr = np.std(np.column_stack([xn, rn, rfn, en]), axis=1, ddof=0)
            rmt_tr_blocks.append(
                np.column_stack([xn, rn, rfn, en,
                                 fold_cache[y]["grid_pos_pct"], fold_cache[y]["overtaking_score"],
                                 fold_cache[y]["champ_rank_pct"], fold_cache[y]["reg_era"],
                                 fold_cache[y]["is_street"], rmt_ens_tr, rmt_dis_tr])
            )
        meta_X_tr = np.vstack(rmt_tr_blocks)
        meta_y_tr = np.hstack([
            fold_cache[y]["y_win_te"].fillna(0).astype(int).values for y in other_years
        ])

        # tiny XGBoost with max_depth=2 as the meta-learner
        # why not LogisticRegression? LR can't handle interactions like "both XGB AND Ranker agree
        # AND overtaking score is low" - that requires a tree split, not a linear weight
        # strong regularization (reg_lambda=10) because we only have ~90 races to train on per fold
        # empirically XGB meta beats LR meta by about 2% on LOSO: 55.1% vs 53.1% winner accuracy
        meta_xgb = xgb.XGBClassifier(
            max_depth=2, n_estimators=100, learning_rate=0.05,
            reg_lambda=10.0, reg_alpha=2.0, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEED, verbosity=0, n_jobs=1,
        )
        meta_xgb.fit(meta_X_tr, meta_y_tr)

        xn_te  = np.asarray(fold_cache[holdout_year]["xgb_norm"])
        rn_te  = np.asarray(fold_cache[holdout_year]["ranker_norm"])
        rfn_te = np.asarray(fold_cache[holdout_year]["rf_norm"])
        en_te  = np.asarray(fold_cache[holdout_year]["et_norm"])
        rmt_ens_te = (rmt_fold["w_xgb"] * xn_te + rmt_fold["w_ranker"] * rn_te
                      + rmt_fold["w_rf"] * rfn_te + rmt_fold["w_et"] * en_te)
        rmt_dis_te = np.std(np.column_stack([xn_te, rn_te, rfn_te, en_te]), axis=1, ddof=0)
        meta_X_te = np.column_stack([xn_te, rn_te, rfn_te, en_te,
                                     fold_cache[holdout_year]["grid_pos_pct"],
                                     fold_cache[holdout_year]["overtaking_score"],
                                     fold_cache[holdout_year]["champ_rank_pct"],
                                     fold_cache[holdout_year]["reg_era"],
                                     fold_cache[holdout_year]["is_street"],
                                     rmt_ens_te, rmt_dis_te])
        meta_scores = 1.0 - meta_xgb.predict_proba(meta_X_te)[:, 1]

        meta_te_h  = fold_cache[holdout_year]["meta_te"]
        y_pos_te_h = fold_cache[holdout_year]["y_pos_te"]

        w_acc, w_correct, w_total = winner_accuracy(meta_te_h, meta_scores, y_pos_te_h)
        jaccard   = top10_set_accuracy(meta_te_h, meta_scores, y_pos_te_h)
        mae       = average_position_mae(meta_te_h, meta_scores, y_pos_te_h)
        pod_exact, pod_overlap, _ = podium_accuracy(meta_te_h, meta_scores, y_pos_te_h)

        meta_results.append({
            "holdout_year":    holdout_year,
            "winner_accuracy": w_acc,
            "winners_correct": w_correct,
            "total_races":     w_total,
            "podium_exact":    pod_exact,
            "podium_overlap":  pod_overlap,
            "top10_jaccard":   jaccard,
            "position_mae":    mae,
        })

        print(f"\n  Holdout {holdout_year} (stacked):")
        print(f"    Winner accuracy : {w_acc:.1%}  ({w_correct}/{w_total} races)")
        print(f"    Podium exact    : {pod_exact:.1%}  (avg overlap: {pod_overlap:.2f}/3)")
        print(f"    Top-10 Jaccard  : {jaccard:.3f}")
        print(f"    Position MAE    : {mae:.2f} places")

    val_df_meta = pd.DataFrame(meta_results)
    print(f"\n  ── Stacking Meta-Model Averages ──")
    print(f"    Winner accuracy : {val_df_meta['winner_accuracy'].mean():.1%}")
    print(f"    Podium exact    : {val_df_meta['podium_exact'].mean():.1%}")
    print(f"    Top-10 Jaccard  : {val_df_meta['top10_jaccard'].mean():.3f}")
    print(f"    Position MAE    : {val_df_meta['position_mae'].mean():.2f} places")

    # the nested evaluation is done - now train the FINAL meta-model on all years combined
    # this is the one we save to disk and use for actual 2025/2026 predictions
    # use global rmt_bma weights (trained on all 7 years) for the RMT features
    all_meta_blocks = []
    for y in years:
        xn = np.asarray(fold_cache[y]["xgb_norm"])
        rn = np.asarray(fold_cache[y]["ranker_norm"])
        rfn = np.asarray(fold_cache[y]["rf_norm"])
        en = np.asarray(fold_cache[y]["et_norm"])
        rmt_ens_all = (rmt_bma["w_xgb"] * xn + rmt_bma["w_ranker"] * rn
                       + rmt_bma["w_rf"] * rfn + rmt_bma["w_et"] * en)
        rmt_dis_all = np.std(np.column_stack([xn, rn, rfn, en]), axis=1, ddof=0)
        all_meta_blocks.append(
            np.column_stack([xn, rn, rfn, en,
                             fold_cache[y]["grid_pos_pct"], fold_cache[y]["overtaking_score"],
                             fold_cache[y]["champ_rank_pct"], fold_cache[y]["reg_era"],
                             fold_cache[y]["is_street"], rmt_ens_all, rmt_dis_all])
        )
    all_meta_X = np.vstack(all_meta_blocks)
    all_meta_y = np.hstack([
        fold_cache[y]["y_win_te"].fillna(0).astype(int).values for y in years
    ])
    meta_model_final = xgb.XGBClassifier(
        max_depth=2, n_estimators=100, learning_rate=0.05,
        reg_lambda=10.0, reg_alpha=2.0, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=SEED, verbosity=0, n_jobs=1,
    )
    meta_model_final.fit(all_meta_X, all_meta_y)
    fi = meta_model_final.feature_importances_
    feat_names = ["xgb", "ranker", "rf", "et", "grid_pct", "overtaking", "champ_rank", "reg_era", "is_street",
                  "rmt_ens", "rmt_disagr"]
    fi_str = "  ".join(f"{n}={v:.3f}" for n, v in zip(feat_names, fi))
    print(f"\n  Final meta-model feature importances (all OOF, 11 features):")
    print(f"    {fi_str}")

    return val_df_meta, bma_weights, cal_temperature, meta_model_final, rmt_bma


# ──────────────────────────────────
# feature importances
# ──────────────────────────────────

def print_feature_importances(xgb_model, rf_win_model, feature_names):
    # XGBoost measures importance by "gain" - how much did each feature reduce prediction error across all its splits?
    # RF measures by "impurity decrease" - how much did each feature reduce Gini/entropy when it was used?
    # these are computed differently, so we look at both and average them for the final ranking
    # heads up: Gini importance has a known bias toward high-cardinality features (lots of unique values)
    # ref: feature importance comparison: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    # ref: https://explained.ai/rf-importance/index.html (great explainer on why Gini importance can mislead)

    xgb_imp = pd.Series(
        xgb_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    rf_imp = pd.Series(
        rf_win_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    print("\n  XGBoost Feature Importances (top 10):")
    for feat, imp in xgb_imp.head(10).items():
        bar = "█" * int(imp * 60)
        print(f"    {feat:<35} {imp:.4f}  {bar}")

    print("\n  Random Forest Feature Importances (top 10):")
    for feat, imp in rf_imp.head(10).items():
        bar = "█" * int(imp * 60)
        print(f"    {feat:<35} {imp:.4f}  {bar}")

    combined = pd.DataFrame({"xgb_importance": xgb_imp, "rf_importance": rf_imp})
    combined["avg_importance"] = combined.mean(axis=1)
    combined = combined.sort_values("avg_importance", ascending=False)

    path = os.path.join(OUT_DIR, "feature_importances.csv")
    combined.to_csv(path)
    print(f"\n  → Feature importances saved to {path}")

    return combined


# ──────────────────────────────────
# main
# ──────────────────────────────────

def main():
    print("=" * 55)
    print("  F1 Race Prediction - Model Training & Evaluation")
    print("=" * 55)

    if not os.path.exists(TRAIN_FILE):
        print(f"\n[!] Training file not found: {TRAIN_FILE}")
        print("    Run feature_engineering.py first.")
        return

    print(f"\nLoading training data from {TRAIN_FILE}...")
    X_all, y_pos, y_win, y_top10, df_meta_all, df_train, imputer = \
        load_and_prepare(TRAIN_FILE)

    print(f"  Loaded {len(df_train)} rows "
          f"({df_train['Year'].nunique()} seasons, "
          f"{df_train['Driver'].nunique()} unique drivers)")

    # drop any rows where a target label is missing - we can't train on those
    valid_mask = y_pos.notna() & y_win.notna() & y_top10.notna()
    X_all    = X_all[valid_mask].reset_index(drop=True)
    y_pos    = y_pos[valid_mask].reset_index(drop=True)
    y_win    = y_win[valid_mask].reset_index(drop=True)
    y_top10  = y_top10[valid_mask].reset_index(drop=True)
    df_train = df_train[valid_mask].reset_index(drop=True)
    print(f"  After dropping NaN targets: {len(df_train)} rows")

    # run the full leave-one-season-out validation - this is the honest accuracy number
    # it returns the nested stacking metrics (not the 40/40/20 baseline printed inside the loop),
    # the BMA weights, the calibration temperature, and the final stacking meta-model
    val_results, bma_weights, cal_temperature, meta_model, rmt_portfolio = \
        leave_one_season_out_validation(df_train, X_all, y_pos, y_win, y_top10)

    # now that we have honest LOSO estimates, do the full hyperparameter tuning on ALL training data
    # the LOSO loop used quick fallback defaults (to keep it fast)
    # here we run the real 80-iteration random search so the production models are as good as possible
    print("\n" + "─" * 55)
    print("  Hyperparameter Tuning (RandomizedSearchCV)")
    print("─" * 55)

    # fit one imputer on the complete training set - this is the one we'll reuse at inference time
    global_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_final = pd.DataFrame(
        global_imputer.fit_transform(X_all), columns=FEATURE_COLS
    )

    print("\n  Tuning XGBoost regressor (80 iterations x 3-fold)...")
    xgb_final, xgb_best_params = tune_xgboost(X_final, y_pos.fillna(20), n_iter=80)
    print("  ✓ XGBoost regressor tuned")

    print("\n  Tuning Random Forest (winner classifier, 60 iterations)...")
    rf_win_final, rf_win_params = tune_random_forest_winner(X_final, y_win, n_iter=60)
    print("  ✓ Random Forest winner tuned")

    print("\n  Tuning Random Forest (top-10 classifier, 60 iterations)...")
    rf_top10_final, rf_top10_params = tune_random_forest_top10(X_final, y_top10, n_iter=60)
    print("  ✓ Random Forest top-10 tuned")

    print("\n  Tuning ExtraTrees regressor (60 iterations)...")
    et_final, et_best_params = tune_extratrees(X_final, y_pos.fillna(20), n_iter=60)
    print("  ✓ ExtraTrees regressor tuned")

    # XGBRanker has a strict requirement: the data must be sorted by (Year, Round) before fitting
    # the `groups` argument tells it how many drivers are in each race (usually 20)
    print("\n  Training XGBoost ranker (rank:ndcg)...")
    sort_idx      = df_train.sort_values(["Year", "Round"]).index
    X_sorted      = X_final.loc[sort_idx].reset_index(drop=True)
    y_sorted      = y_pos.loc[sort_idx].reset_index(drop=True)
    meta_sorted   = df_train.loc[sort_idx].reset_index(drop=True)
    ranker_groups = compute_race_groups(meta_sorted)
    xgb_ranker    = train_xgboost_ranker(X_sorted, y_sorted, ranker_groups, xgb_best_params)
    print("  ✓ XGBoost ranker trained (reusing regressor hyperparams)")

    # save all the trained models and metadata to disk so predict_2025/predict_2026 can load them
    os.makedirs(MODEL_DIR, exist_ok=True)

    params_path = os.path.join(OUT_DIR, "best_hyperparams.txt")
    with open(params_path, "w") as f:
        f.write("Best Hyperparameters (RandomizedSearchCV)\n")
        f.write("=" * 55 + "\n\n")
        f.write("XGBoost Regressor:\n")
        for k, v in sorted(xgb_best_params.items()):
            f.write(f"  {k}: {v}\n")
        f.write("\nRandom Forest (winner):\n")
        for k, v in sorted(rf_win_params.items()):
            f.write(f"  {k}: {v}\n")
        f.write("\nRandom Forest (top-10):\n")
        for k, v in sorted(rf_top10_params.items()):
            f.write(f"  {k}: {v}\n")
        f.write("\nExtraTrees Regressor:\n")
        for k, v in sorted(et_best_params.items()):
            f.write(f"  {k}: {v}\n")
        f.write("\nBayesian Model Averaging Weights (from LOSO):\n")
        for k, v in bma_weights.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\nBayesian Temperature Scaling: T = {cal_temperature:.4f}\n")
    print(f"\n  → Hyperparameters saved to {params_path}")
    # joblib is the standard way to save sklearn/XGBoost objects - it handles large numpy arrays
    # much better than pickle, and it preserves the full model state including trees
    joblib.dump(xgb_final,      os.path.join(MODEL_DIR, "xgb_model.joblib"))
    joblib.dump(xgb_ranker,     os.path.join(MODEL_DIR, "xgb_ranker.joblib"))
    joblib.dump(rf_win_final,   os.path.join(MODEL_DIR, "rf_win_model.joblib"))
    joblib.dump(rf_top10_final, os.path.join(MODEL_DIR, "rf_top10_model.joblib"))
    joblib.dump(et_final,       os.path.join(MODEL_DIR, "et_model.joblib"))
    joblib.dump(global_imputer, os.path.join(MODEL_DIR, "feature_imputer.joblib"))
    joblib.dump(bma_weights,    os.path.join(MODEL_DIR, "bma_weights.joblib"))
    joblib.dump(cal_temperature, os.path.join(MODEL_DIR, "cal_temperature.joblib"))
    joblib.dump(meta_model,     os.path.join(MODEL_DIR, "meta_model.joblib"))
    joblib.dump(rmt_portfolio,  os.path.join(MODEL_DIR, "rmt_portfolio.joblib"))
    print(f"\n  → Models saved to {MODEL_DIR}/")

    # print and save which features each model found most useful
    print("\n" + "─" * 55)
    print("  Feature Importances")
    print("─" * 55)
    print_feature_importances(xgb_final, rf_win_final, FEATURE_COLS)

    # quick sanity check on the training set itself - expect very high accuracy here
    # if this is low something is seriously wrong; if it's suspiciously perfect that's also a red flag
    xgb_tr_preds    = xgb_final.predict(X_final)
    rf_win_tr_proba = rf_win_final.predict_proba(X_final)[:, 1]
    meta_tr = df_train[["Year", "Round", "EventName", "Driver", "Team"]].reset_index(drop=True)

    w_acc, w_correct, w_total = winner_accuracy(meta_tr, xgb_tr_preds, y_pos)
    pod_exact, pod_overlap, _ = podium_accuracy(meta_tr, xgb_tr_preds, y_pos)
    jaccard = top10_set_accuracy(meta_tr, xgb_tr_preds, y_pos)
    mae     = average_position_mae(meta_tr, xgb_tr_preds, y_pos)

    print(f"\n  In-sample training metrics (should be optimistic):")
    print(f"    Winner accuracy : {w_acc:.1%}  ({w_correct}/{w_total} races)")
    print(f"    Podium exact    : {pod_exact:.1%}  (avg overlap: {pod_overlap:.2f}/3)")
    print(f"    Top-10 Jaccard  : {jaccard:.3f}")
    print(f"    Position MAE    : {mae:.2f} places")

    # generate 2025 predictions if the feature file is there (requires feature_engineering.py to run first)
    if not os.path.exists(PRED_FILE):
        print(f"\n[!] 2025 prediction file not found: {PRED_FILE}")
        print("    Skipping 2025 predictions. Run feature_engineering.py first.")
    else:
        print("\n" + "─" * 55)
        print("  Predicting 2025 Race Results")
        print("─" * 55)

        X_2025, _, _, _, df_meta_2025, df_2025, _ = \
            load_and_prepare(PRED_FILE, fit_imputer=global_imputer)

        sort_idx_2025 = df_2025.sort_values(["Year", "Round"]).index
        X_2025_s  = X_2025.loc[sort_idx_2025].reset_index(drop=True)
        df_2025_s = df_2025.loc[sort_idx_2025].reset_index(drop=True)

        xgb_preds_25     = xgb_final.predict(X_2025_s)
        rf_proba_25      = rf_win_final.predict_proba(X_2025_s)[:, 1]
        rf_top10_pred_25 = rf_top10_final.predict(X_2025_s)
        ranker_scores_25 = xgb_ranker.predict(X_2025_s)
        et_preds_25      = et_final.predict(X_2025_s)

        pred_df = ensemble_predict(
            xgb_preds_25, rf_proba_25,
            df_2025_s[["Year", "Round", "EventName", "Driver", "Team"]],
            ranker_raw_scores=ranker_scores_25,
            et_raw_pos=et_preds_25,
            ensemble_weights=bma_weights,
            X_features=X_2025_s,
            apply_bayesian_prior=True,
            cal_temperature=cal_temperature,
            meta_model=meta_model,
        )
        pred_df["rf_top10_pred"] = rf_top10_pred_25

        if "finish_position" in df_2025_s.columns and df_2025_s["finish_position"].notna().any():
            pred_df["actual_position"] = df_2025_s["finish_position"].values

        out_path = os.path.join(OUT_DIR, "2025_predictions.csv")
        pred_df.to_csv(out_path, index=False)
        print(f"\n  → 2025 predictions saved to {out_path}")

        # print the first 3 predicted races so we can eyeball if the output looks sane
        for rnd in pred_df["Round"].unique()[:3]:
            race = pred_df[pred_df["Round"] == rnd].head(10)
            print(f"\n  2025 R{rnd:02d} {race.iloc[0]['EventName']} - Predicted Top 10:")
            print(f"    {'Pos':>3}  {'Driver':<6}  {'Team':<30}  {'Win%':>5}")
            print(f"    {'───':>3}  {'──────':<6}  {'──────────────────────────────':<30}  {'─────':>5}")
            for _, row in race.iterrows():
                print(f"    {int(row['PredictedPosition']):>3}  "
                      f"{row['Driver']:<6}  "
                      f"{str(row['Team'])[:30]:<30}  "
                      f"{row['rf_win_proba']*100:>4.1f}%")

    # write the full evaluation summary to a text file for the report
    eval_path = os.path.join(OUT_DIR, "model_evaluation.txt")
    with open(eval_path, "w") as f:
        f.write("F1 Race Prediction - Model Evaluation Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write("Leave-One-Season-Out CV:\n")
        f.write(val_results.to_string(index=False))
        f.write(f"\n\nMean Winner Accuracy : {val_results['winner_accuracy'].mean():.1%}\n")
        f.write(f"Mean Podium Exact    : {val_results['podium_exact'].mean():.1%}\n")
        f.write(f"Mean Top-10 Jaccard  : {val_results['top10_jaccard'].mean():.3f}\n")
        f.write(f"Mean Position MAE    : {val_results['position_mae'].mean():.2f} places\n")
        f.write(f"\nBayesian Model Averaging Weights (posterior from LOSO):\n")
        for k, v in bma_weights.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\nIn-sample (training set, expected to be optimistic):\n")
        f.write(f"  Winner accuracy : {w_acc:.1%}  ({w_correct}/{w_total} races)\n")
        f.write(f"  Podium exact    : {pod_exact:.1%}  (avg overlap: {pod_overlap:.2f}/3)\n")
        f.write(f"  Top-10 Jaccard  : {jaccard:.3f}\n")
        f.write(f"  Position MAE    : {mae:.2f} places\n")
    print(f"\n  → Evaluation saved to {eval_path}")

    # auto-generate the meta-model stacking visualisation plots so every run
    # produces up-to-date plots without needing a separate script call
    try:
        import subprocess, sys
        plot_script = os.path.join("final_report", "plot_meta_stacking_rmt.py")
        if os.path.exists(plot_script):
            result = subprocess.run([sys.executable, plot_script],
                                    capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"\n  → Meta-stacking plots regenerated")
            else:
                print(f"\n  [!] plot_meta_stacking_rmt.py failed: {result.stderr[:200]}")
    except Exception as _e:
        print(f"\n  [!] Could not auto-run plot_meta_stacking_rmt.py: {_e}")

    print("\n" + "=" * 55)
    print("  Done.")
    print("=" * 55)


if __name__ == "__main__":
    main()
