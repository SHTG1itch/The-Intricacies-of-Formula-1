"""
mach_learning.py — F1 Race Prediction: XGBoost + XGBRanker + Random Forest
===========================================================================
Trains and evaluates a three-model ensemble to predict Formula 1 race outcomes:

  1. XGBoost Regressor (XGBReg)  → predicts numeric finishing position
  2. XGBoost Ranker   (XGBRank)  → ranking-aware model (rank:ndcg objective)
  3. Random Forest    (RF)       → predicts winner probability (binary)

ITERATION HISTORY:
  v1 — XGBReg + RF (winner/top-10), 25 features, hardcoded ensemble alpha.
       Proved multi-circuit training outperforms the reference model.
  v2 — Same architecture, +12 features (FP2 sector/long-run/consistency,
       qualifying time gaps, weather flag, season form).  Tuned hyperparams
       with RandomizedSearchCV; added LOSO cross-validation.
  v3 — Added XGBRanker to ensemble; 33 features including FP1/FP2 weather,
       FP3 long-run pace, qualifying air temp, circuit experience, 3-race
       rolling average.  Ensemble now rank-normalises all three model outputs
       before blending (40% XGBReg + 40% XGBRank + 20% RF) instead of the
       old hardcoded alpha=0.3 position-shift blend.

KEY DESIGN DECISIONS:
  Position-based framing (XGBReg): predicting a numeric position then ranking
    drivers per race is more stable than direct classification.  The model
    never needs to know how many drivers are in a race.
  Ranking objective (XGBRank): rank:ndcg is directly aligned with our goal
    of ordering drivers correctly within a race, unlike squared-error which
    penalises absolute position errors equally regardless of race context.
  Rank-normalisation before blending: converts each model's raw output to a
    per-race [0,1] rank score before combining, so the three models are
    comparable in scale regardless of their output distributions.
  LOSO uses XGBReg only: the ranker requires sorted groups, which complicates
    fold splitting.  LOSO metrics thus reflect the regressor — the ranker
    is an ensemble boost on top.

HOW TO RUN (from project root):
    python final_report/feature_engineering.py
    python final_report/mach_learning.py

OUTPUT:
    data_organized/model_evaluation.txt    — accuracy metrics + LOSO summary
    data_organized/feature_importances.csv — XGBReg + RF importance, 33 features
    data_organized/best_hyperparams.txt    — best hyperparameters from tuning
    data_organized/2025_predictions.csv    — predicted race results for 2025
    data_organized/models/                 — joblib-serialised trained models

REFERENCES:
    Random Forest:    Breiman (2001), scikit-learn RandomForestClassifier
    XGBoost Reg:      Chen & Guestrin (2016), https://xgboost.readthedocs.io/
    XGBoost Ranker:   rank:ndcg objective, XGBoost learning-to-rank docs
    RandomizedSearchCV: sklearn.model_selection.RandomizedSearchCV
    FastF1 docs:      https://docs.fastf1.dev/
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

DATA_DIR   = "data_organized"
TRAIN_FILE = os.path.join(DATA_DIR, "ml_training_features.csv")
PRED_FILE  = os.path.join(DATA_DIR, "ml_2025_features.csv")
OUT_DIR    = DATA_DIR
MODEL_DIR  = os.path.join(DATA_DIR, "models")

# All 33 feature columns used by the models (must match feature_engineering.py output).
#
# ITERATION HISTORY:
#   v1 (Q1): 13 features — grid_position, FP pace deltas, sprint positions,
#             rolling form (5-race), circuit history averages.
#   v2 (Q2): +12 = 25 features — FP2 sector splits, FP2 long-run race pace,
#             lap consistency, FP1→FP2 pace progression, qualifying time gaps
#             to pole and teammate, qualifying weather (track_temp, wet flag),
#             season points percentage, season avg finish.
#   v3 (current): +8 = 33 features — FP1/FP2 session weather (track_temp, wet),
#             FP3 Saturday long-run pace, qualifying air temperature, circuit
#             experience count, 3-race short rolling average.
#
# The Q1/Q2 split was a development artefact.  All features now live in a
# single flat list grouped by logical category.

FEATURE_COLS = [
    # ── Grid / qualifying position ─────────────────────────────────────
    "IsSprintWeekend",
    "grid_position",
    "sprint_quali_pos",
    "sprint_pos",

    # ── FP best-lap pace deltas ────────────────────────────────────────
    "fp1_pace_delta",
    "fp2_pace_delta",
    "fp3_pace_delta",

    # ── FP2 session-derived features ───────────────────────────────────
    "fp2_s1_delta",          # sector-1 pace vs session best
    "fp2_s2_delta",          # sector-2 pace vs session best
    "fp2_s3_delta",          # sector-3 pace vs session best
    "fp2_long_run_delta",    # long-run (race-sim) pace vs session median
    "fp2_consistency",       # std dev of FP2 quicklaps (lower = steadier)
    "fp_pace_progression",   # FP1→FP2 improvement vs field

    # ── FP3 long-run (Saturday, fresher signal than FP2) ───────────────
    "fp3_long_run_delta",

    # ── Qualifying gap features ────────────────────────────────────────
    "quali_gap_to_pole",     # seconds behind pole (finer than integer rank)
    "quali_teammate_gap",    # driver vs fastest teammate (isolates driver skill)

    # ── Weather features (pre-race sessions only — no leakage) ─────────
    # v1/v2 had only quali_track_temp + quali_wet.
    # v3 adds FP1/FP2 weather and qualifying air temperature.
    # Race weather is NOT included — it cannot be known before the race.
    "fp1_track_temp",
    "fp1_wet",
    "fp2_track_temp",
    "fp2_wet",
    "quali_track_temp",
    "quali_air_temp",        # proxy for tyre cooling and thermal degradation
    "quali_wet",

    # ── Rolling / season form ──────────────────────────────────────────
    "rolling_avg_finish_3",  # 3-race window (recent momentum signal)
    "rolling_avg_finish_5",  # 5-race window (stable baseline)
    "rolling_dnf_rate_5",
    "rolling_points_5",
    "team_rolling_avg_5",
    "season_pts_pct",
    "season_avg_finish",

    # ── Circuit history ────────────────────────────────────────────────
    "circuit_experience",         # count of prior races at this circuit
    "circuit_driver_avg_finish",
    "circuit_team_avg_finish",
]  # 33 total

# Random seed for reproducibility
SEED = 42

# ─────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────

def load_and_prepare(filepath, fit_imputer=None):
    """
    Load the feature CSV and return (X, y_pos, y_win, y_top10, df_meta, imputer).

    - X          : feature matrix (NaN imputed with column median)
    - y_pos      : target — finish_position (regression)
    - y_win      : target — is_winner (binary classification)
    - y_top10    : target — is_top10 (binary classification)
    - df_meta    : identifier columns (Year, Round, Driver, Team, EventName)
    - imputer    : fitted SimpleImputer (reuse for prediction set)

    Imputation strategy:
      NaN in FP deltas → impute with median (driver did not participate in session)
      NaN in rolling stats → impute with median (new driver or first few races)
      NaN in grid_position → impute with 20 (back of grid assumption)

    Ref: https://scikit-learn.org/stable/modules/impute.html
    """
    df = pd.read_csv(filepath)

    meta_cols = ["Year", "Round", "EventName", "Circuit", "Driver", "Team"]
    target_cols = ["finish_position", "is_winner", "is_top10"]

    df_meta = df[meta_cols].copy()
    X_raw   = df[FEATURE_COLS].copy().astype(float)

    # Impute missing values
    if fit_imputer is None:
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(
            imputer.fit_transform(X_raw),
            columns=FEATURE_COLS
        )
    else:
        imputer = fit_imputer
        X = pd.DataFrame(
            imputer.transform(X_raw),
            columns=FEATURE_COLS
        )

    y_pos   = df["finish_position"].astype(float) if "finish_position" in df else None
    y_win   = df["is_winner"].astype(float)       if "is_winner" in df else None
    y_top10 = df["is_top10"].astype(float)        if "is_top10" in df else None

    return X, y_pos, y_win, y_top10, df_meta, df, imputer


# ─────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────

def rank_predictions_per_race(df_meta, y_pred_pos):
    """
    Given predicted finishing positions, rank the drivers within each race.
    Returns a DataFrame with columns [Year, Round, Driver, PredictedPosition].

    This converts the regressor's raw numeric output into an ordered 1–20 ranking.
    """
    df = df_meta[["Year", "Round", "EventName", "Driver", "Team"]].copy()
    df["raw_pred"] = y_pred_pos
    df["PredictedPosition"] = (
        df.groupby(["Year", "Round"])["raw_pred"]
          .rank(method="first", ascending=True)
          .astype(int)
    )
    return df.sort_values(["Year", "Round", "PredictedPosition"])


def winner_accuracy(df_meta, y_pred_pos, y_true_pos):
    """
    Compute the fraction of races where the model correctly predicted the winner
    (i.e., the driver ranked 1st by the model actually won the race).
    """
    df = df_meta[["Year", "Round", "Driver"]].copy()
    df["pred_pos"] = y_pred_pos
    df["true_pos"] = y_true_pos.values

    correct = 0
    total   = 0
    for (year, rnd), group in df.groupby(["Year", "Round"]):
        if group["true_pos"].isna().all():
            continue
        pred_winner = group.loc[group["pred_pos"].idxmin(), "Driver"]
        true_winner_rows = group[group["true_pos"] == 1]
        if true_winner_rows.empty:
            continue
        true_winner = true_winner_rows.iloc[0]["Driver"]
        correct += int(pred_winner == true_winner)
        total   += 1

    return correct / total if total > 0 else 0.0, correct, total


def top10_set_accuracy(df_meta, y_pred_pos, y_true_pos, k=10):
    """
    For each race, compute the Jaccard similarity between the predicted top-k
    and the actual top-k set of drivers.
    Returns the mean Jaccard score across all races.
    """
    df = df_meta[["Year", "Round", "Driver"]].copy()
    df["pred_pos"] = y_pred_pos
    df["true_pos"] = y_true_pos.values

    jaccard_scores = []
    for (year, rnd), group in df.groupby(["Year", "Round"]):
        if group["true_pos"].isna().any():
            continue
        pred_top = set(
            group.nsmallest(k, "pred_pos")["Driver"].tolist()
        )
        true_top = set(
            group[group["true_pos"] <= k]["Driver"].tolist()
        )
        if not pred_top and not true_top:
            continue
        intersection = len(pred_top & true_top)
        union        = len(pred_top | true_top)
        jaccard_scores.append(intersection / union if union > 0 else 0.0)

    return np.mean(jaccard_scores) if jaccard_scores else 0.0


def podium_accuracy(df_meta, y_pred_pos, y_true_pos):
    """
    Fraction of races where the predicted top-3 set matches the actual podium set.

    Exact match: all three predicted podium drivers are among the actual top-3.
    Also returns partial credit: average overlap count per race.

    Ref: Inspired by Kaggle "Predict the podium" evaluation metric.
    """
    df = df_meta[["Year", "Round", "Driver"]].copy()
    df["pred_pos"] = y_pred_pos
    df["true_pos"] = y_true_pos.values

    exact_matches = 0
    overlap_counts = []
    total = 0

    for (year, rnd), group in df.groupby(["Year", "Round"]):
        if group["true_pos"].isna().any():
            continue
        pred_top3 = set(group.nsmallest(3, "pred_pos")["Driver"].tolist())
        true_top3 = set(group[group["true_pos"] <= 3]["Driver"].tolist())
        if not true_top3:
            continue
        overlap = len(pred_top3 & true_top3)
        overlap_counts.append(overlap)
        exact_matches += int(pred_top3 == true_top3)
        total += 1

    exact_rate = exact_matches / total if total > 0 else 0.0
    avg_overlap = np.mean(overlap_counts) if overlap_counts else 0.0
    return exact_rate, avg_overlap, total


def average_position_mae(df_meta, y_pred_pos, y_true_pos):
    """
    Mean Absolute Error of the predicted rank vs actual finishing position.
    """
    ranked_df = rank_predictions_per_race(df_meta, y_pred_pos)
    merged = ranked_df[["Year", "Round", "Driver", "PredictedPosition"]].copy()
    merged["TruePosition"] = y_true_pos.values
    merged = merged.dropna(subset=["TruePosition"])
    if merged.empty:
        return np.nan
    return mean_absolute_error(merged["TruePosition"], merged["PredictedPosition"])


# ─────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────

def tune_xgboost(X_train, y_train, n_iter=50):
    """
    RandomizedSearchCV over XGBoost hyperparameter space.

    WHY RANDOMIZED SEARCH?
    Grid search over all combinations is O(n^k) — infeasible for XGBoost's
    large parameter space. RandomizedSearchCV samples n_iter points from the
    joint distribution, covering the space efficiently.
    With n_iter=50 and 3-fold CV we run 150 fits — fast enough on a laptop.

    Parameter ranges informed by:
    - XGBoost docs: https://xgboost.readthedocs.io/en/stable/parameter.html
    - Kaggle F1 prediction community notebooks
    - "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin 2016)
    """
    # v3 hyperparameter range changes vs v2:
    #   max_depth ceiling lowered (8→6): shallower trees resist overfitting
    #     with 33 features; LOSO showed depth>6 gave minimal CV gain.
    #   min_child_weight floor raised (1→3): avoids fitting tiny leaf nodes
    #     that memorise individual race outcomes.
    #   reg_alpha widened (0–0.5 → 0–1.0): more L1 regularization headroom.
    #   reg_lambda floor raised (0.5→1.0): ensures meaningful L2 penalty.
    #   gamma widened (0–0.3 → 0–0.5): stronger minimum-split-gain control.
    #   colsample_bylevel added: additional column subsampling per tree level.
    param_dist = {
        "n_estimators":      randint(300, 900),
        "max_depth":         randint(3, 6),          # shallower = less overfit
        "learning_rate":     uniform(0.01, 0.09),    # 0.01–0.10
        "subsample":         uniform(0.6, 0.25),     # 0.6–0.85
        "colsample_bytree":  uniform(0.5, 0.35),     # 0.5–0.85
        "colsample_bylevel": uniform(0.5, 0.5),      # 0.5–1.0 (new in v3)
        "min_child_weight":  randint(3, 10),          # raised floor
        "reg_alpha":         uniform(0.0, 1.0),      # wider L1
        "reg_lambda":        uniform(1.0, 3.0),      # higher L2 floor
        "gamma":             uniform(0.0, 0.5),      # wider split-gain control
    }
    base = xgb.XGBRegressor(
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"    Best XGBoost MAE (CV): {-search.best_score_:.3f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def tune_random_forest_winner(X_train, y_win_train, n_iter=40):
    """
    RandomizedSearchCV for the winner Random Forest classifier.

    class_weight='balanced_subsample' rebalances at each tree, better than
    'balanced' (which rebalances once globally) for deep forests.
    Ref: https://scikit-learn.org/stable/modules/generated/
         sklearn.ensemble.RandomForestClassifier.html
    """
    param_dist = {
        "n_estimators":   randint(200, 700),
        "max_depth":      [4, 5, 6, 7, 8, 10, None],
        "min_samples_leaf": randint(2, 10),
        "max_features":   ["sqrt", "log2", 0.4, 0.6],
        "criterion":      ["gini", "entropy"],
    }
    base = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    valid = y_win_train.dropna()
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="f1",
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train.loc[valid.index], valid.astype(int))
    print(f"    Best RF winner F1 (CV): {search.best_score_:.3f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def tune_random_forest_top10(X_train, y_top10_train, n_iter=40):
    """
    RandomizedSearchCV for the top-10 Random Forest classifier.
    Less imbalanced than winner (~50% positive), so standard F1 scoring works.
    """
    param_dist = {
        "n_estimators":     randint(200, 700),
        "max_depth":        [4, 5, 6, 7, 8, 10, None],
        "min_samples_leaf": randint(1, 8),
        "max_features":     ["sqrt", "log2", 0.4, 0.6],
        "criterion":        ["gini", "entropy"],
    }
    base = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    valid = y_top10_train.dropna()
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="f1",
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train.loc[valid.index], valid.astype(int))
    print(f"    Best RF top-10 F1 (CV): {search.best_score_:.3f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


# ── Fallback trainers (used inside LOSO loop where tuning would be too slow) ──

def train_xgboost_regressor(X_train, y_train, params=None):
    """Train XGBoost with given params (or sensible defaults if None)."""
    if params is None:
        params = {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "gamma": 0.0,
        }
    model = xgb.XGBRegressor(
        **params, random_state=SEED, verbosity=0, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_winner(X_train, y_win_train, params=None):
    """Train RF winner classifier with given params or defaults."""
    if params is None:
        params = {
            "n_estimators": 500, "max_depth": 8, "min_samples_leaf": 5,
            "max_features": "sqrt", "criterion": "gini",
        }
    model = RandomForestClassifier(
        **params, class_weight="balanced_subsample",
        random_state=SEED, n_jobs=-1,
    )
    valid = y_win_train.dropna()
    model.fit(X_train.loc[valid.index], valid.astype(int))
    return model


def train_random_forest_top10(X_train, y_top10_train, params=None):
    """Train RF top-10 classifier with given params or defaults."""
    if params is None:
        params = {
            "n_estimators": 500, "max_depth": 8, "min_samples_leaf": 3,
            "max_features": "sqrt", "criterion": "gini",
        }
    model = RandomForestClassifier(
        **params, class_weight="balanced_subsample",
        random_state=SEED, n_jobs=-1,
    )
    valid = y_top10_train.dropna()
    model.fit(X_train.loc[valid.index], valid.astype(int))
    return model


def train_xgboost_ranker(X_train, y_pos_train, groups, best_params):
    """
    Train XGBRanker with rank:ndcg objective.

    v3 addition: the ranking objective is directly aligned with our goal
    of ordering all 20 drivers correctly within a race, whereas XGBReg
    (squared-error) penalises all position errors equally regardless of
    race context.  rank:ndcg focuses on getting the top of the order right,
    which is where F1 prediction accuracy matters most (winner, podium).

    XGBRanker is NOT used in LOSO validation — it requires groups sorted by
    race, which complicates fold splitting.  It is trained once on the full
    dataset using the best hyperparameters found for the regressor (they
    transfer well; both models share the same tree architecture).

    Parameters
    ----------
    X_train     : DataFrame — feature matrix sorted by (Year, Round)
    y_pos_train : Series    — finish_position (1=P1, 20=P20), same sort order
    groups      : list[int] — drivers per race in (Year, Round) sorted order
    best_params : dict      — hyperparameters from tune_xgboost()

    Returns
    -------
    Fitted XGBRanker model.
    """
    # Convert position to relevance: higher relevance = better finishing position.
    # P1 → max relevance, P20 → 1.  XGBRanker expects non-negative integers.
    max_pos   = int(y_pos_train.fillna(20).max())
    relevance = (max_pos + 1 - y_pos_train.fillna(max_pos)).astype(int).clip(lower=1)

    # Build ranker params from regressor best_params; colsample_bylevel may
    # not be present if it wasn't in the search space — default to 0.8.
    ranker_params = {k: v for k, v in best_params.items()}
    ranker_params.setdefault("n_estimators", 400)

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        **ranker_params,
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, relevance, group=groups)
    return model


def compute_race_groups(df_meta):
    """
    Return per-race driver counts as required by XGBRanker's `group` argument.
    df_meta must already be sorted by (Year, Round) — the caller is responsible
    for sorting before calling this function.

    Returns list[int] where element i = number of rows in race i.
    """
    return df_meta.groupby(["Year", "Round"], sort=False).size().tolist()


# ─────────────────────────────────────────────────────────────────
# ENSEMBLE PREDICTION
# ─────────────────────────────────────────────────────────────────

def _rank_normalize(scores, df_meta, ascending=True):
    """
    Rank scores within each race and normalise to [0, 1].

    ascending=True  → lower score is better (XGBReg position: P1 = lowest pred)
    ascending=False → higher score is better (XGBRanker: P1 = highest relevance)

    Returns a numpy array of normalised ranks in [0, 1] where 0 = best.
    This allows the three models to be blended on a common scale regardless
    of their raw output distributions.

    Edge case: a race with a single driver gets normalised rank 0.0 — correct,
    since they rank P1 by default.
    """
    df = df_meta[["Year", "Round"]].copy().reset_index(drop=True)
    df["raw"] = np.asarray(scores)
    df["rank"] = df.groupby(["Year", "Round"])["raw"].rank(
        method="average", ascending=ascending
    )
    n = df.groupby(["Year", "Round"])["raw"].transform("count")
    df["norm"] = (df["rank"] - 1) / (n - 1).clip(lower=1)
    return df["norm"].values


def ensemble_predict(xgb_raw_pos, rf_win_proba, df_meta, ranker_raw_scores=None):
    """
    Three-model rank-normalised ensemble (v3) or two-model fallback (v1/v2).

    v1/v2 blend (ranker_raw_scores=None):
      final_score = xgb_raw_pos + 0.3 * (1 - rf_win_proba)
      Retained for LOSO validation where the ranker is not trained per fold.

    v3 three-model blend:
      Rank-normalise each model's output to [0, 1] per race (0 = best),
      then combine:
        final = 0.40 * xgb_norm + 0.40 * ranker_norm + 0.20 * rf_norm

      Weights rationale:
        XGBReg (40%): stable position regression, well-tuned hyperparams.
        XGBRank (40%): ranking-aware objective, compensates for regression
                       errors near the top-3 where it matters most.
        RF (20%): calibrated win probability, adds diversity but is less
                  precise about mid-field ordering.

    Parameters
    ----------
    xgb_raw_pos       : array — XGBReg predicted position scores
    rf_win_proba      : array — RF P(win) probabilities
    df_meta           : DataFrame with [Year, Round, EventName, Driver, Team]
    ranker_raw_scores : array or None — XGBRanker relevance scores (higher=better)

    Returns
    -------
    DataFrame with PredictedPosition column (1 = predicted winner per race).
    """
    df = df_meta[["Year", "Round", "EventName", "Driver", "Team"]].copy()
    df = df.reset_index(drop=True)
    df["rf_win_proba"] = rf_win_proba
    df["xgb_raw_pred"] = xgb_raw_pos

    if ranker_raw_scores is not None:
        # v3 rank-normalised blend
        xgb_norm    = _rank_normalize(xgb_raw_pos,       df, ascending=True)
        ranker_norm = _rank_normalize(ranker_raw_scores,  df, ascending=False)
        rf_norm     = _rank_normalize(1 - rf_win_proba,  df, ascending=True)

        df["ensemble_score"] = (0.40 * xgb_norm
                                + 0.40 * ranker_norm
                                + 0.20 * rf_norm)
    else:
        # v1/v2 fallback: original hardcoded alpha blend
        df["ensemble_score"] = xgb_raw_pos + 0.3 * (1 - rf_win_proba)

    df["PredictedPosition"] = (
        df.groupby(["Year", "Round"])["ensemble_score"]
          .rank(method="first", ascending=True)
          .astype(int)
    )
    return df.sort_values(["Year", "Round", "PredictedPosition"])


# ─────────────────────────────────────────────────────────────────
# VALIDATION: Leave-one-season-out
# ─────────────────────────────────────────────────────────────────

def leave_one_season_out_validation(df_train, X_train_all, y_pos, y_win, y_top10):
    """
    Performs leave-one-season-out cross-validation.

    For each year Y in the training set:
      - Train on all years except Y
      - Test on year Y
      - Report winner accuracy, top-10 Jaccard, and position MAE

    This simulates the real use case: training on past seasons, predicting
    the next unseen season.
    """
    print("\n" + "─" * 55)
    print("  Leave-One-Season-Out Validation")
    print("─" * 55)

    years = sorted(df_train["Year"].unique())
    results = []

    for holdout_year in years:
        train_mask = df_train["Year"] != holdout_year
        test_mask  = df_train["Year"] == holdout_year

        X_tr = X_train_all[train_mask]
        X_te = X_train_all[test_mask]

        # Fit imputer on train split only; reset index so X and y align
        imp = SimpleImputer(strategy="median")
        X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=FEATURE_COLS)
        X_te = pd.DataFrame(imp.transform(X_te),     columns=FEATURE_COLS)

        y_pos_tr   = y_pos[train_mask].reset_index(drop=True)
        y_pos_te   = y_pos[test_mask].reset_index(drop=True)
        y_win_tr   = y_win[train_mask].reset_index(drop=True)
        y_top10_tr = y_top10[train_mask].reset_index(drop=True)

        meta_te = df_train[test_mask][
            ["Year", "Round", "EventName", "Driver", "Team"]
        ].reset_index(drop=True)

        # Train models
        xgb_model = train_xgboost_regressor(X_tr, y_pos_tr.fillna(20))
        rf_win     = train_random_forest_winner(X_tr, y_win_tr)
        rf_top10   = train_random_forest_top10(X_tr, y_top10_tr)

        # Predict
        xgb_preds = xgb_model.predict(X_te)
        rf_win_proba = rf_win.predict_proba(X_te)[:, 1]
        rf_top10_pred = rf_top10.predict(X_te)

        # Metrics
        w_acc, w_correct, w_total = winner_accuracy(
            meta_te.reset_index(drop=True),
            xgb_preds,
            y_pos_te.reset_index(drop=True)
        )
        jaccard = top10_set_accuracy(
            meta_te.reset_index(drop=True),
            xgb_preds,
            y_pos_te.reset_index(drop=True)
        )
        mae = average_position_mae(
            meta_te.reset_index(drop=True),
            xgb_preds,
            y_pos_te.reset_index(drop=True)
        )

        pod_exact, pod_overlap, _ = podium_accuracy(
            meta_te.reset_index(drop=True),
            xgb_preds,
            y_pos_te.reset_index(drop=True)
        )

        results.append({
            "holdout_year":   holdout_year,
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

    return val_df


# ─────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────

def print_feature_importances(xgb_model, rf_win_model, feature_names):
    """
    Display and return ranked feature importances from both models.
    XGBoost uses gain-based importance; RF uses impurity-based importance.
    """
    xgb_imp = pd.Series(
        xgb_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    rf_imp = pd.Series(
        rf_win_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    print("\n  XGBoost Feature Importances (top 10):")
    for feat, imp in xgb_imp.head(10).items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<35} {imp:.4f}  {bar}")

    print("\n  Random Forest Feature Importances (top 10):")
    for feat, imp in rf_imp.head(10).items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<35} {imp:.4f}  {bar}")

    combined = pd.DataFrame({
        "xgb_importance": xgb_imp,
        "rf_importance":  rf_imp,
    })
    combined["avg_importance"] = combined.mean(axis=1)
    combined = combined.sort_values("avg_importance", ascending=False)

    path = os.path.join(OUT_DIR, "feature_importances.csv")
    combined.to_csv(path)
    print(f"\n  → Feature importances saved to {path}")

    return combined


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  F1 Race Prediction — Model Training & Evaluation")
    print("=" * 55)

    # ── 1. Load training data ──────────────────────────────────────
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

    # Drop rows with NaN targets
    valid_mask = y_pos.notna() & y_win.notna() & y_top10.notna()
    X_all    = X_all[valid_mask].reset_index(drop=True)
    y_pos    = y_pos[valid_mask].reset_index(drop=True)
    y_win    = y_win[valid_mask].reset_index(drop=True)
    y_top10  = y_top10[valid_mask].reset_index(drop=True)
    df_train = df_train[valid_mask].reset_index(drop=True)

    print(f"  After dropping NaN targets: {len(df_train)} rows")

    # ── 2. Leave-one-season-out cross-validation ───────────────────
    val_results = leave_one_season_out_validation(
        df_train, X_all, y_pos, y_win, y_top10
    )

    # ── 3. Hyperparameter tuning on ALL training data ──────────────
    print("\n" + "─" * 55)
    print("  Hyperparameter Tuning (RandomizedSearchCV, 50/40 iters)")
    print("─" * 55)

    # Fit one global imputer on all training data
    global_imputer = SimpleImputer(strategy="median")
    X_final = pd.DataFrame(
        global_imputer.fit_transform(X_all),
        columns=FEATURE_COLS
    )

    print("\n  Tuning XGBoost regressor...")
    xgb_final, xgb_best_params = tune_xgboost(X_final, y_pos.fillna(20), n_iter=50)
    print("  ✓ XGBoost regressor tuned")

    print("\n  Tuning Random Forest (winner)...")
    rf_win_final, rf_win_best_params = tune_random_forest_winner(X_final, y_win, n_iter=40)
    print("  ✓ Random Forest winner tuned")

    print("\n  Tuning Random Forest (top-10)...")
    rf_top10_final, rf_top10_best_params = tune_random_forest_top10(X_final, y_top10, n_iter=40)
    print("  ✓ Random Forest top-10 tuned")

    # ── Train XGBRanker (v3) ────────────────────────────────────────
    # Sort training data by (Year, Round) — XGBRanker requires contiguous
    # groups of rows belonging to the same race.
    print("\n  Training XGBoost ranker (rank:ndcg)...")
    sort_idx      = df_train.sort_values(["Year", "Round"]).index
    X_sorted      = X_final.loc[sort_idx].reset_index(drop=True)
    y_sorted      = y_pos.loc[sort_idx].reset_index(drop=True)
    meta_sorted   = df_train.loc[sort_idx].reset_index(drop=True)
    ranker_groups = compute_race_groups(meta_sorted)
    xgb_ranker    = train_xgboost_ranker(X_sorted, y_sorted, ranker_groups, xgb_best_params)
    print("  ✓ XGBoost ranker trained (reusing regressor best params)")

    # Save best hyperparameters
    params_path = os.path.join(OUT_DIR, "best_hyperparams.txt")
    with open(params_path, "w") as f:
        f.write("Best Hyperparameters (RandomizedSearchCV)\n")
        f.write("=" * 55 + "\n\n")
        f.write("XGBoost Regressor:\n")
        for k, v in sorted(xgb_best_params.items()):
            f.write(f"  {k}: {v}\n")
        f.write("\nXGBoost Ranker (rank:ndcg):\n")
        f.write("  (reuses XGBoost Regressor best params — no separate tuning)\n")
        f.write("\nRandom Forest (winner):\n")
        for k, v in sorted(rf_win_best_params.items()):
            f.write(f"  {k}: {v}\n")
        f.write("\nRandom Forest (top-10):\n")
        for k, v in sorted(rf_top10_best_params.items()):
            f.write(f"  {k}: {v}\n")
    print(f"\n  → Best hyperparameters saved to {params_path}")

    # ── 4. Persist trained models to disk ──────────────────────────
    # predict_2025.py loads all four models so it doesn't need to retrain.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(xgb_final,       os.path.join(MODEL_DIR, "xgb_model.joblib"))
    joblib.dump(xgb_ranker,      os.path.join(MODEL_DIR, "xgb_ranker.joblib"))
    joblib.dump(rf_win_final,    os.path.join(MODEL_DIR, "rf_win_model.joblib"))
    joblib.dump(rf_top10_final,  os.path.join(MODEL_DIR, "rf_top10_model.joblib"))
    joblib.dump(global_imputer,  os.path.join(MODEL_DIR, "feature_imputer.joblib"))
    print(f"\n  → Models saved to {MODEL_DIR}/")

    # ── 5. Feature importances ─────────────────────────────────────
    print("\n" + "─" * 55)
    print("  Feature Importances")
    print("─" * 55)
    feat_imp = print_feature_importances(xgb_final, rf_win_final, FEATURE_COLS)

    # ── 5. In-sample sanity check (full training set) ──────────────
    xgb_train_preds = xgb_final.predict(X_final)
    rf_win_train_proba = rf_win_final.predict_proba(X_final)[:, 1]

    meta_tr = df_train[["Year", "Round", "EventName", "Driver", "Team"]].reset_index(drop=True)
    w_acc, w_correct, w_total = winner_accuracy(meta_tr, xgb_train_preds, y_pos)
    pod_exact, pod_overlap, _ = podium_accuracy(meta_tr, xgb_train_preds, y_pos)
    jaccard = top10_set_accuracy(meta_tr, xgb_train_preds, y_pos)
    mae     = average_position_mae(meta_tr, xgb_train_preds, y_pos)

    print(f"\n  In-sample training metrics:")
    print(f"    Winner accuracy : {w_acc:.1%}  ({w_correct}/{w_total} races)")
    print(f"    Podium exact    : {pod_exact:.1%}  (avg overlap: {pod_overlap:.2f}/3)")
    print(f"    Top-10 Jaccard  : {jaccard:.3f}")
    print(f"    Position MAE    : {mae:.2f} places")

    # ── 6. Predict 2025 races ──────────────────────────────────────
    if not os.path.exists(PRED_FILE):
        print(f"\n[!] 2025 prediction file not found: {PRED_FILE}")
        print("    Skipping 2025 predictions.")
    else:
        print("\n" + "─" * 55)
        print("  Predicting 2025 Race Results")
        print("─" * 55)

        X_2025, y_pos_2025, y_win_2025, y_top10_2025, \
            df_meta_2025, df_2025, _ = load_and_prepare(PRED_FILE, fit_imputer=imputer)

        # Sort by (Year, Round) — consistent sort for rank normalisation
        sort_idx_2025 = df_2025.sort_values(["Year", "Round"]).index
        X_2025_s      = X_2025.loc[sort_idx_2025].reset_index(drop=True)
        df_2025_s     = df_2025.loc[sort_idx_2025].reset_index(drop=True)

        xgb_2025_preds     = xgb_final.predict(X_2025_s)
        rf_win_2025_proba  = rf_win_final.predict_proba(X_2025_s)[:, 1]
        rf_top10_2025_pred = rf_top10_final.predict(X_2025_s)
        ranker_2025_scores = xgb_ranker.predict(X_2025_s)

        pred_df = ensemble_predict(
            xgb_2025_preds, rf_win_2025_proba,
            df_2025_s[["Year", "Round", "EventName", "Driver", "Team"]],
            ranker_raw_scores=ranker_2025_scores,
        )
        pred_df["rf_top10_pred"] = rf_top10_2025_pred

        # If 2025 race results exist (for completed races), add them
        if "finish_position" in df_2025_s.columns and df_2025_s["finish_position"].notna().any():
            pred_df["actual_position"] = df_2025_s["finish_position"].values

        out_path = os.path.join(OUT_DIR, "2025_predictions.csv")
        pred_df.to_csv(out_path, index=False)
        print(f"\n  → 2025 predictions saved to {out_path}")

        # Pretty-print first few races
        rounds = pred_df["Round"].unique()[:3]
        for rnd in rounds:
            race_pred = pred_df[pred_df["Round"] == rnd].head(10)
            race_name = race_pred.iloc[0]["EventName"]
            print(f"\n  2025 R{rnd:02d} {race_name} — Predicted Top 10:")
            print(f"    {'Pos':>3}  {'Driver':<6}  {'Team':<30}  {'Win%':>5}")
            print(f"    {'───':>3}  {'──────':<6}  {'──────────────────────────────':<30}  {'─────':>5}")
            for _, row in race_pred.iterrows():
                print(f"    {int(row['PredictedPosition']):>3}  "
                      f"{row['Driver']:<6}  "
                      f"{row['Team'][:30]:<30}  "
                      f"{row['rf_win_proba']*100:>4.1f}%")

    # ── 7. Save evaluation summary ─────────────────────────────────
    eval_path = os.path.join(OUT_DIR, "model_evaluation.txt")
    with open(eval_path, "w") as f:
        f.write("F1 Race Prediction — Model Evaluation Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write("Leave-One-Season-Out Cross-Validation:\n")
        f.write(val_results.to_string(index=False))
        f.write(f"\n\nMean Winner Accuracy : {val_results['winner_accuracy'].mean():.1%}\n")
        f.write(f"Mean Podium Exact    : {val_results['podium_exact'].mean():.1%}\n")
        f.write(f"Mean Top-10 Jaccard  : {val_results['top10_jaccard'].mean():.3f}\n")
        f.write(f"Mean Position MAE    : {val_results['position_mae'].mean():.2f} places\n")
        f.write(f"\nIn-sample (training set):\n")
        f.write(f"  Winner accuracy : {w_acc:.1%}  ({w_correct}/{w_total} races)\n")
        f.write(f"  Podium exact    : {pod_exact:.1%}\n")
        f.write(f"  Top-10 Jaccard  : {jaccard:.3f}\n")
        f.write(f"  Position MAE    : {mae:.2f} places\n")
    print(f"\n  → Evaluation summary saved to {eval_path}")

    print("\n" + "=" * 55)
    print("  Done! Models trained and evaluated.")
    print("=" * 55)
    print(f"\nKey outputs:")
    print(f"  {os.path.join(OUT_DIR, 'feature_importances.csv')}")
    print(f"  {os.path.join(OUT_DIR, 'best_hyperparams.txt')}")
    print(f"  {os.path.join(OUT_DIR, 'model_evaluation.txt')}")
    print(f"  {os.path.join(OUT_DIR, '2025_predictions.csv')}")


if __name__ == "__main__":
    main()
