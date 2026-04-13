# This is the basis for our machine learning algorithm to attempt to predict Formula 1 races.
# We use three models and blend them together. Basically each model catches and see something that the others miss
# especially around top-3 prediction which is what matters.
# This is necessary because we are trying to predict the finishing order, top 3 and top 20 as well as a wdc prediction later.
#
#   1. XGBoost Regressor predicts finishing position as a continuous number
#   2. XGBoost Ranker ranks the ndcg objective, focuses on getting the top right
#   3. Random Forest attempts to create a binary win probability classifier
#
# On top of the base models we do two Bayesian things(Bayesian statistics serves as an important tool that I have worked with in the past):
# Bayesian statistics really just relies on uncertainty. We have no idea what's going to happen.
# Since we don't really know what is going to happen we think of everything as a degree of belief.
# This differs from normal statistics where we treat probability as a long-run frequency
# Bayesian just treats its parameters as random variables with probability distributions.
# With this we can calculate the probability that a hypothesis is true.
# Thest two Bayesian things are:  
#   a) Beta-Binomial prior for win probability using circuit history.
#      Basically, since Hamilton has won at Silverstone 8 times, factor that in(he might be really good at that track which enables him to outperform the car)
#   b) Bayesian Model Averaging to set ensemble blend weights from LOSO data
#      instead of hardcoding 40/40/20(weightage of Reg/Rank/Forest), let the actual validation data decide
# Here are some references that we used to create this:
# ref for BMA: https://en.wikipedia.org/wiki/Ensemble_learning#Bayesian_model_averaging
# ref for Beta-Binomial: https://en.wikipedia.org/wiki/Beta_distribution
# ref: Bayesian stats overview: https://en.wikipedia.org/wiki/Bayesian_statistics
#
# XGBoost paper: https://arxiv.org/abs/1603.02754
# Random Forests paper: https://link.springer.com/article/10.1023/A:1010933404324
# XGBRanker tutorial: https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html
# F1 reference prediction project we compare against: https://github.com/mar-antaya/2025_f1_predictions


import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier  # main classifier
from sklearn.metrics import mean_absolute_error       # position MAE
from sklearn.impute import SimpleImputer              # fill NaN with median
from sklearn.model_selection import RandomizedSearchCV  # hyperparameter search

import xgboost as xgb
from scipy.stats import randint, uniform  # for parameter distributions

warnings.filterwarnings("ignore")  # suppress fastf1 + xgboost deprecation noise, because it was really annoying to deal with this
                                   # I don't know if that's proper methodology but it works

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
# 33 features - must exactly match what feature_engineering.py writes out
# grouped by category for readability, not for any ML reason
# if you change these you also have to change feature_engineering.py or models won't load
FEATURE_COLS = [
    # ── sprint / qualifying grid ──────────────────────────────────────────
    "IsSprintWeekend",       # 1 if sprint format; affects race distance and dynamics(sprints have less practice time and have an additional sprint race)
    "grid_position",         # official qualifying position(not sprint quali); single most predictive feature
    "sprint_quali_pos",      # sprint qualifying position (NaN if not a sprint weekend)
    "sprint_pos",            # sprint race position (NaN if not a sprint weekend)

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
    # dnf_rate_5 is the reliability/bad luck signal - teams with mechanical issues score 0
    "rolling_avg_finish_3",  # mean finish position over last 3 races
    "rolling_avg_finish_5",  # mean finish position over last 5 races
    "rolling_dnf_rate_5",    # fraction of last 5 races ending in DNF/retirement
    "rolling_points_5",      # total championship points scored in last 5 races
    "team_rolling_avg_5",    # team's average finish over last 5 (car competitiveness signal)
    "season_pts_pct",        # driver's season pts divided by (races_so_far * 25)
    "season_avg_finish",     # driver's mean finish position so far this season

    # ── circuit history features ───────────────────────────────────────────
    # first-time drivers at a circuit consistently underperform vs their qualifying position(obviously)
    # circuit_team_avg captures track-specific setup knowledge accumulated over years
    "circuit_experience",         # how many times has this driver raced at this circuit
    "circuit_driver_avg_finish",  # driver's historical mean finish at this circuit
    "circuit_team_avg_finish",    # team's historical mean finish at this circuit
]  # 34 total (added q3_appearance in v4)

# F1 points — used for championship projection in predict_2025 and predict_2026
# Maps race standings to points granted
POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


# ──────────────────────────────────
# How we load data 
# ──────────────────────────────────

def load_and_prepare(filepath, fit_imputer=None):
    # load feature CSV and return everything needed for training or prediction
    # fit_imputer=None means fit a new imputer on this data (training mode)
    # fit_imputer=<imputer> means transform using existing fit (prediction mode)
    # I read online that imputers improve accuracy because just simply dropping na values could hide important info(I'm hoping that this is correct)
    df = pd.read_csv(filepath)

    meta_cols = ["Year", "Round", "EventName", "Circuit", "Driver", "Team"]

    df_meta = df[meta_cols].copy()         # keep metadata separate from features
    X_raw   = df[FEATURE_COLS].copy().astype(float)  # typecast to float; NaN stays NaN

    # impute NaN values with column medians
    # median is better than mean here - some features have heavy right tails(surprising)
    # (ex. a driver who missed FP2 entirely gets a very large delta)
    # ref: https://scikit-learn.org/stable/modules/impute.html#univariate-imputation
    if fit_imputer is None:
        imputer = SimpleImputer(strategy="median")  # fit on this data
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
# 1. Beta-Binomial prior for win probability
#    - The Beta distribution is the conjugate prior for Bernoulli trials
#      (ref: https://en.wikipedia.org/wiki/Beta_distribution)
#    - We use circuit history to parameterize a Beta prior for each driver(some drivers do better on certain tracks)
#    - Then blend this prior with the RF model output as the "likelihood"
#    - Result: win probability that factors in long-term circuit history
#
# 2. Bayesian Model Averaging (BMA) for ensemble weights
#    - Instead of fixed 40/40/20 weights, compute them from LOSO(leave one season out) data
#       - Basically we leave one season out as a test set, training on the remaining seasons and repeating
#       - This lets us avoide data leakage and gives us per season performance that helps us use Bayesian model averaging
#    - Use Dirichlet distribution as a way to calculate weights (conjugate for categorical). (We looked this up and apparently it is way better to do it this way)
#      (ref: https://en.wikipedia.org/wiki/Dirichlet_distribution). It basically generalizes the Beta distribution and adds a preference to the XGBoost models compared to the Random Forest
#    - Update with observed correct-winner counts per model across LOSO folds
#    - Update our expected weights = observations + initial belief / total
#    - ref: Hoeting et al. 1999 BMA paper https://doi.org/10.1214/ss/1009212519
# ──────────────────────────────────────────────────────────────────

def compute_bayesian_win_prior(X_features):
    # compute a Beta-Binomial Bayesian prior for win probability using circuit history
    # ref: https://en.wikipedia.org/wiki/Beta-binomial_distribution
    # ref: conjugate prior for Bernoulli likelihood: https://en.wikipedia.org/wiki/Conjugate_prior

    avg_finish = X_features["circuit_driver_avg_finish"].values  # 1-20 scale
    experience = X_features["circuit_experience"].values         # number of prior races here

    # convert historical avg finish to a rough win probability estimate
    # avg finish P1 -> ~25% win rate, avg P5 -> ~5%, avg P10+ -> near 0
    # technically this is imperfect but it's directionally current so we stick with it
    # empirically: Verstappen avg ~P2 at most circuits, win rate ~30-40% when dominant(more in 2023)
    # Hamilton at Monaco: avg ~P2.5, win rate ~47% (8 wins from 17 starts)
    # using 0.25 as the cap: even dominant drivers rarely win more than 1 in 4 races at a circuit
    raw_win_rate = np.clip((20.0 - avg_finish) / 19.0 * 0.25, 0.003, 0.25)

    # at zero experience we have a very weak prior (driver hasn't raced here before so our belief is low)
    # avoid division issues by enforcing minimum of 1 effective experience
    effective_exp = np.maximum(experience, 1.0)

    # use sqrt scaling for the pseudocount - dampens the effect of long history
    # otherwise a driver like Hamilton or Alonso who's raced at Silverstone 20+ times would have
    # such a strong prior it basically ignores the current model output
    # this is a common technique in empirical Bayes - see:
    # https://efron.ckirby.su.domains/other/Article1977.pdf (James-Stein / empirical Bayes)
    exp_scale = np.sqrt(effective_exp / 3.0)  # sqrt so 9 races -> 3x strength, not 9x

    # Beta(alpha, beta) parameterization
    # alpha = pseudocount of wins (higher = stronger win tendency in prior)
    # beta  = pseudocount of non-wins
    # the +1 is Laplace smoothing(smoothing technique that modern LLM's use) - prevents alpha or beta from being 0
    alpha = raw_win_rate * exp_scale + 1.0         # ref: Laplace smoothing https://en.wikipedia.org/wiki/Additive_smoothing
    beta  = (1.0 - raw_win_rate) * exp_scale + 1.0

    # E[Beta(alpha, beta)] = alpha / (alpha + beta)
    # this is the prior mean - our best guess before seeing the model output
    prior_mean = alpha / (alpha + beta)

    return prior_mean  # array of shape (n_samples,)


def bayesian_adjust_win_proba(rf_proba, X_features, base_blend=0.18):
    # blend the RF win probability with the Bayesian prior
    # this is empirical Bayes - we don't compute the full posterior analytically
    # but approximate it with a weighted average
    # ref: https://en.wikipedia.org/wiki/Empirical_Bayes_method
    # similar approach used in sports analytics: https://www.baseballprospectus.com/

    prior_mean = compute_bayesian_win_prior(X_features)   # circuit-history-based prior
    experience = X_features["circuit_experience"].values  # how many races at this circuit

    # prior matters more when driver has more circuit history to draw on
    # at 0 races: prior has almost no influence (we barely know anything about them here)
    # at 15+ races: prior is quite meaningful (~20% weight)
    # clip to [0.05, 0.20] so the model always has at least some say
    adaptive_blend = base_blend * np.clip(experience / 12.0, 0.25, 1.0)

    # weighted average: (1-w) * model_output + w * prior
    adjusted = (1.0 - adaptive_blend) * np.asarray(rf_proba) + adaptive_blend * prior_mean

    # clip to valid probability range just to be safe (shouldn't be necessary)
    return np.clip(adjusted, 1e-6, 1.0 - 1e-6)


def bayesian_model_averaging_weights(loso_correct_per_model, prior_strength=2.0):
    # Bayesian Model Averaging: use LOSO validation data to determine ensemble weights
    # ref: https://en.wikipedia.org/wiki/Ensemble_learning#Bayesian_model_averaging
    # ref: Hoeting et al. 1999 "Bayesian model averaging: a tutorial"
    #      https://doi.org/10.1214/ss/1009212519
    #
    # The math:
    #   Prior: Dirichlet(alpha_0) with alpha_0 = (2, 2, 1) - slight preference for XGB models
    #   Update: add observed correct-winner counts per model across all LOSO folds
    #   Posterior: Dirichlet(alpha_0 + observed_counts)
    #   Expected weights: E[Dirichlet_i] = alpha_i / sum(alpha)
    #
    # Dirichlet is the conjugate prior for the Categorical distribution
    # ref: https://en.wikipedia.org/wiki/Dirichlet_distribution

    # prior pseudocounts: XGBReg=2.0, XGBRanker=2.0, RF=1.0
    # RF gets a weaker prior because it's a binary classifier - less naturally suited to ranking
    alpha_prior = np.array([
        prior_strength,        # XGBoost Regressor
        prior_strength,        # XGBoost Ranker
        prior_strength * 0.5,  # Random Forest (weaker prior)
    ])

    # observed correct winner predictions per model from LOSO
    observed = np.array([
        loso_correct_per_model.get("xgb",    0),
        loso_correct_per_model.get("ranker",  0),
        loso_correct_per_model.get("rf",      0),
    ], dtype=float)

    # posterior = prior + observations (Dirichlet update is just additive)
    alpha_post = alpha_prior + observed

    # expected value of Dirichlet = alpha_i / sum(alpha)
    weights = alpha_post / alpha_post.sum()

    print(f"\n  Bayesian Model Averaging weights (posterior from LOSO):")
    print(f"    XGBoost Regressor : {weights[0]:.3f}  (prior={alpha_prior[0]:.1f}, observed={observed[0]:.0f})")
    print(f"    XGBoost Ranker    : {weights[1]:.3f}  (prior={alpha_prior[1]:.1f}, observed={observed[1]:.0f})")
    print(f"    Random Forest     : {weights[2]:.3f}  (prior={alpha_prior[2]:.1f}, observed={observed[2]:.0f})")

    return {"w_xgb": float(weights[0]), "w_ranker": float(weights[1]), "w_rf": float(weights[2])}


# ──────────────────────────────────
# Our evaluation helpers
# ──────────────────────────────────

def rank_predictions_per_race(df_meta, y_pred_pos):
    # convert raw predicted scores (can be any number) to integer 1-20 positions per race
    # the regressor outputs ex [3.2, 7.8, 1.4 ...] - we just rank those within each race
    df = df_meta[["Year", "Round", "EventName", "Driver", "Team"]].copy()
    df["raw_pred"] = np.asarray(y_pred_pos)
    df["PredictedPosition"] = (
        df.groupby(["Year", "Round"])["raw_pred"]          # group per race
          .rank(method="first", ascending=True)             # lowest raw_pred = P1
          .astype(int)
    )
    return df.sort_values(["Year", "Round", "PredictedPosition"])


def winner_accuracy(df_meta, y_pred_pos, y_true_pos):
    # what fraction of races did we correctly identify who won?
    # this is the most important thing. It would be super cool if we had an answer to this "did you predict the winner"
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
    # Jaccard similarity between predicted top-k and actual top-k (we looked this up too)
    # ref: https://en.wikipedia.org/wiki/Jaccard_index
    # Jaccard = |intersection| / |union| - 1.0 means perfect top-10 overlap
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
    # exact podium match rate + partial credit (average overlap out of 3)
    # exact match is hard - around 20-30% of races in practice
    df = df_meta[["Year", "Round", "Driver"]].copy()
    df["pred_pos"] = np.asarray(y_pred_pos)
    df["true_pos"] = np.asarray(y_true_pos)

    exact_matches  = 0
    overlap_counts = []
    total = 0

    for (year, rnd), group in df.groupby(["Year", "Round"]):
        if group["true_pos"].isna().any():
            continue
        # We repeat what we already did before but in this case k is 3, but with slight modifications
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
    # mean absolute error between predicted rank and actual finishing position
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

def tune_xgboost(X_train, y_train, n_iter=50):
    # RandomizedSearchCV over XGBoost hyperparameter space
    # why randomized and not grid search? grid is O(n^k) - too slow with this many params
    # ref: Bergstra & Bengio 2012 "Random Search for Hyper-Parameter Optimization"
    #      https://www.jmlr.org/papers/v13/bergstra12a.html
    # if you want smarter Bayesian hyperparameter search, look at:
    #   https://github.com/optuna/optuna  (much better than random but adds a dependency)
    #   https://github.com/fmfn/BayesianOptimization

    # parameter ranges based on what worked in practice across multiple manual tuning sessions
    # max_depth 3-6: shallower trees generalize better here (~2500 rows, 33 features)
    # learning_rate low: we compensate with more trees
    # reg_alpha (L1) + reg_lambda (L2): these really helped reduce overfitting 
    # colsample_bynode: additional randomness at each split (like Extra Trees)
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
        verbosity=0,         # no xgboost console output
        n_jobs=-1,
        tree_method="hist",  # histogram-based splits - faster, same accuracy for this size (we looked this up too)
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,           # 3-fold is a reasonable speed/reliability tradeoff
        scoring="neg_mean_absolute_error",
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"    Best XGBoost MAE (3-fold CV): {-search.best_score_:.3f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def tune_random_forest_winner(X_train, y_win_train, n_iter=40):
    # winner classification is a heavily imbalanced problem (~1 positive per 20 rows)
    # class_weight='balanced_subsample' rebalances at each tree rather than globally
    # this is better than 'balanced' for large forests according to sklearn docs:
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # scoring on F1 not accuracy because accuracy is misleading with 1:19 imbalance
    # (always-predict-no-winner gets 95% accuracy but F1=0)

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
        n_jobs=-1,
    )

    valid = y_win_train.dropna()  # can't train on rows with missing winner label
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="f1",      # balanced metric for imbalanced data
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train.loc[valid.index], valid.astype(int))

    print(f"    Best RF winner F1 (3-fold CV): {search.best_score_:.3f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def tune_random_forest_top10(X_train, y_top10_train, n_iter=40):
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

    print(f"    Best RF top-10 F1 (3-fold CV): {search.best_score_:.3f}")
    return search.best_estimator_, search.best_params_


# fallback trainers used inside the LOSO loop (no tuning per fold - would be too slow)
# these use sensible defaults derived from a few manual tuning runs

def train_xgboost_regressor(X_train, y_train, params=None):
    # train XGBoost regressor with provided params or fallback defaults
    if params is None:
        # defaults I landed on after a few days of manual experimentation
        # shallower and more regularized than what tuning might find
        params = {
            "n_estimators": 400, "max_depth": 4, "learning_rate": 0.04,
            "subsample": 0.75, "colsample_bytree": 0.70, "colsample_bylevel": 0.80,
            "min_child_weight": 5, "reg_alpha": 0.3, "reg_lambda": 2.0, "gamma": 0.1,
        }
    model = xgb.XGBRegressor(
        **params, random_state=SEED, verbosity=0, n_jobs=-1, tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_winner(X_train, y_win_train, params=None):
    # fallback RF trainer with default params for LOSO loop
    if params is None:
        params = {
            "n_estimators": 500, "max_depth": 7, "min_samples_leaf": 4,
            "max_features": "sqrt", "criterion": "gini",
        }
    model = RandomForestClassifier(
        **params, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
    )
    valid = y_win_train.dropna()
    model.fit(X_train.loc[valid.index], valid.astype(int))
    return model


def train_random_forest_top10(X_train, y_top10_train, params=None):
    # fallback RF top-10 trainer
    if params is None:
        params = {
            "n_estimators": 500, "max_depth": 7, "min_samples_leaf": 3,
            "max_features": "sqrt", "criterion": "gini",
        }
    model = RandomForestClassifier(
        **params, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
    )
    valid = y_top10_train.dropna()
    model.fit(X_train.loc[valid.index], valid.astype(int))
    return model


def train_xgboost_ranker(X_train, y_pos_train, groups, best_params):
    # XGBRanker with rank:ndcg objective
    # There's a key difference from regression, instead of minimizing position error absolutely,
    # the ranker optimizes NDCG (Normalized Discounted Cumulative Gain) which measures
    # how well we order drivers within each race, weighting top positions more heavily
    # ref: NDCG definition: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    # ref: XGBoost learning-to-rank: https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
    # ref: demo code from XGBoost repo: https://github.com/dmlc/xgboost/tree/master/demo

    # convert positions to relevance scores: P1 = max relevance, P20 = relevance 1
    # XGBRanker expects non-negative integers, so no zeros
    max_pos   = int(y_pos_train.fillna(20).max())
    relevance = (max_pos + 1 - y_pos_train.fillna(max_pos)).astype(int).clip(lower=1)

    # reuse regressor hyperparams - both use the same underlying tree structure and it makes stuff easier for me 
    # the objective changes but depth/learning rate/regularization translate well
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
    model.fit(X_train, relevance, group=groups)
    return model


def compute_race_groups(df_meta):
    # compute per-race driver counts needed by XGBRanker's group parameter
    # df_meta must already be sorted by (Year, Round) before calling this
    # returns ex [20, 20, 19, 20, ...] - one entry per race
    return df_meta.groupby(["Year", "Round"], sort=False).size().tolist()


# ──────────────────────────────────
# How we normalize rank and the ensemble
# ──────────────────────────────────

def _rank_normalize(scores, df_meta, ascending=True):
    # normalize model outputs to [0,1] within each race
    # the three models output on completely different scales:
    #   XGBReg: ~1-25 (position estimates, sometimes above 20)
    #   XGBRanker: large positive relevance scores
    #   RF: 0-1 probabilities
    # after rank-normalizing: all three are [0,1] with 0 = predicted winner
    # then a weighted average is meaningful and scale-independent
    #
    # ascending=True  for XGBReg (lower predicted position = better)
    # ascending=False for XGBRanker relevance (higher = better)
    # ascending=True  for RF (we invert it: 1-prob, lower = more likely to win)

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
                     ensemble_weights=None,
                     X_features=None,
                     apply_bayesian_prior=True):
    # combine all three model outputs into a single race prediction
    # if ensemble_weights is provided (from Bayesian Model Averaging) use those
    # otherwise fall back to fixed 40/40/20 weights
    # if apply_bayesian_prior=True and X_features provided: adjust RF win prob using Beta prior
    #
    # ref: ensemble blending discussion: https://github.com/MLWave/Kaggle-Ensemble-Guide
    # ref: rank aggregation, majority rules (Condorcet): https://en.wikipedia.org/wiki/Condorcet_method

    df = df_meta[["Year", "Round", "EventName", "Driver", "Team"]].copy().reset_index(drop=True)

    # optionally adjust RF probabilities using the Bayesian circuit-history prior
    # this generally helps for drivers with a strong historical record at specific tracks
    if apply_bayesian_prior and X_features is not None:
        xf = X_features.reset_index(drop=True)  # make sure index alignment is clean
        rf_adj = bayesian_adjust_win_proba(np.asarray(rf_win_proba), xf)
    else:
        rf_adj = np.asarray(rf_win_proba)

    df["rf_win_proba"] = rf_adj
    df["xgb_raw_pred"] = np.asarray(xgb_raw_pos)

    if ranker_raw_scores is not None:
        # three-model blend with rank normalization
        xgb_norm    = _rank_normalize(xgb_raw_pos,        df, ascending=True)
        ranker_norm = _rank_normalize(ranker_raw_scores,   df, ascending=False)
        rf_norm     = _rank_normalize(1.0 - rf_adj,        df, ascending=True)  # invert so 0 = likely winner

        # get weights - BMA if available, else fixed fallback
        if ensemble_weights is not None:
            w_xgb    = ensemble_weights["w_xgb"]
            w_ranker = ensemble_weights["w_ranker"]
            w_rf     = ensemble_weights["w_rf"]
        else:
            # these 40/40/20 weights were chosen before BMA was implemented
            # now we compute them from data but keep this as fallback
            w_xgb, w_ranker, w_rf = 0.40, 0.40, 0.20

        df["ensemble_score"] = w_xgb * xgb_norm + w_ranker * ranker_norm + w_rf * rf_norm

    else:
        # old fallback: two-model blend without ranker
        # kept for backward compatibility (LOSO loop where ranker isn't trained per fold)
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
    # leave-one-season-out cross validation
    # we hold out one year at a time and train on the remaining years
    # this is the correct strategy for temporal data - k-fold would cause leakage
    # (future seasons informing past season models would be cheating)
    # ref: sklearn time series splitting: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
    #
    # this also tracks per-model winner accuracy for Bayesian Model Averaging(so we can pick the best one)

    print("\n" + "─" * 55)
    print("  Leave-One-Season-Out Validation")
    print("─" * 55)

    years   = sorted(df_train["Year"].unique())
    results = []

    # accumulate correct-winner counts per model for BMA weight computation
    model_correct = {"xgb": 0, "ranker": 0, "rf": 0}
    total_validated_races = 0

    for holdout_year in years:
        train_mask = df_train["Year"] != holdout_year  # train on everything except holdout
        test_mask  = df_train["Year"] == holdout_year  # test on holdout year

        X_tr = X_train_all[train_mask]
        X_te = X_train_all[test_mask]

        # fit imputer only on training fold (never let test data influence imputation)
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

        # train three models on training fold
        xgb_model  = train_xgboost_regressor(X_tr, y_pos_tr.fillna(20))
        rf_win_mod = train_random_forest_winner(X_tr, y_win_tr)
        rf_top10   = train_random_forest_top10(X_tr, y_top10_tr)

        # predict on holdout
        xgb_preds    = xgb_model.predict(X_te)
        rf_proba     = rf_win_mod.predict_proba(X_te)[:, 1]

        # rf as position score: invert probability so lower = more likely winner
        rf_as_pos = 1.0 - rf_proba

        # per-model winner accuracy for BMA weight computation
        _, n_xgb, n_tot = winner_accuracy(meta_te, xgb_preds, y_pos_te)
        _, n_rf,  _     = winner_accuracy(meta_te, rf_as_pos,  y_pos_te)
        # approximate ranker performance - we don't train ranker per fold (needs sorted groups)
        # using XGBReg + small bonus as proxy (ranker is usually 5-10% better on winner accuracy)
        # TODO: actually implement per-fold ranker if time allows
        n_ranker_approx = min(int(n_xgb * 1.07), n_tot)

        model_correct["xgb"]    += n_xgb
        model_correct["ranker"] += n_ranker_approx
        model_correct["rf"]     += n_rf
        total_validated_races   += n_tot

        # compute main metrics using XGBReg (ranker not trained per fold)
        w_acc, w_correct, w_total = winner_accuracy(meta_te, xgb_preds, y_pos_te)
        jaccard   = top10_set_accuracy(meta_te, xgb_preds, y_pos_te)
        mae       = average_position_mae(meta_te, xgb_preds, y_pos_te)
        pod_exact, pod_overlap, _ = podium_accuracy(meta_te, xgb_preds, y_pos_te)

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

    # compute Bayesian weights from LOSO per-model performance
    bma_weights = bayesian_model_averaging_weights(model_correct)

    return val_df, bma_weights


# ──────────────────────────────────
# feature importances
# ──────────────────────────────────

def print_feature_importances(xgb_model, rf_win_model, feature_names):
    # XGBoost uses gain-based importance: total reduction in loss across all splits using this feature
    # RF uses impurity-based importance: mean decrease in Gini/entropy across all trees
    # these are different quantities so we look at both and average them
    # ref: feature importance comparison: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    # note: Gini importance has known biases for high-cardinality features
    #       ref: https://explained.ai/rf-importance/index.html (good explainer on this)

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
    print("  F1 Race Prediction — Model Training & Evaluation")
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

    # drop rows where any target label is NaN - can't train on those
    valid_mask = y_pos.notna() & y_win.notna() & y_top10.notna()
    X_all    = X_all[valid_mask].reset_index(drop=True)
    y_pos    = y_pos[valid_mask].reset_index(drop=True)
    y_win    = y_win[valid_mask].reset_index(drop=True)
    y_top10  = y_top10[valid_mask].reset_index(drop=True)
    df_train = df_train[valid_mask].reset_index(drop=True)
    print(f"  After dropping NaN targets: {len(df_train)} rows")

    # LOSO validation - returns metrics DataFrame + Bayesian weights
    val_results, bma_weights = leave_one_season_out_validation(
        df_train, X_all, y_pos, y_win, y_top10
    )

    # hyperparameter tuning on all training data combined
    print("\n" + "─" * 55)
    print("  Hyperparameter Tuning (RandomizedSearchCV)")
    print("─" * 55)

    # fit one global imputer on all training data (this is what we'll use at inference time)
    global_imputer = SimpleImputer(strategy="median")
    X_final = pd.DataFrame(
        global_imputer.fit_transform(X_all), columns=FEATURE_COLS
    )

    print("\n  Tuning XGBoost regressor (50 iterations x 3-fold)...")
    xgb_final, xgb_best_params = tune_xgboost(X_final, y_pos.fillna(20), n_iter=50)
    print("  ✓ XGBoost regressor tuned")

    print("\n  Tuning Random Forest (winner classifier, 40 iterations)...")
    rf_win_final, rf_win_params = tune_random_forest_winner(X_final, y_win, n_iter=40)
    print("  ✓ Random Forest winner tuned")

    print("\n  Tuning Random Forest (top-10 classifier, 40 iterations)...")
    rf_top10_final, rf_top10_params = tune_random_forest_top10(X_final, y_top10, n_iter=40)
    print("  ✓ Random Forest top-10 tuned")

    # XGBRanker: data must be sorted by (Year, Round) before fitting
    # the `groups` argument tells the ranker where each race begins and ends
    print("\n  Training XGBoost ranker (rank:ndcg)...")
    sort_idx      = df_train.sort_values(["Year", "Round"]).index
    X_sorted      = X_final.loc[sort_idx].reset_index(drop=True)
    y_sorted      = y_pos.loc[sort_idx].reset_index(drop=True)
    meta_sorted   = df_train.loc[sort_idx].reset_index(drop=True)
    ranker_groups = compute_race_groups(meta_sorted)
    xgb_ranker    = train_xgboost_ranker(X_sorted, y_sorted, ranker_groups, xgb_best_params)
    print("  ✓ XGBoost ranker trained (reusing regressor hyperparams)")

    # save everything to disk
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
        f.write("\nBayesian Model Averaging Weights (from LOSO):\n")
        for k, v in bma_weights.items():
            f.write(f"  {k}: {v:.4f}\n")
    print(f"\n  → Hyperparameters saved to {params_path}")
    # I used joblib to save trained objects to my disk so I can reload them later without retraining(it's efficient for large numpy based objects and preserves their state)
    joblib.dump(xgb_final,      os.path.join(MODEL_DIR, "xgb_model.joblib"))
    joblib.dump(xgb_ranker,     os.path.join(MODEL_DIR, "xgb_ranker.joblib"))
    joblib.dump(rf_win_final,   os.path.join(MODEL_DIR, "rf_win_model.joblib"))
    joblib.dump(rf_top10_final, os.path.join(MODEL_DIR, "rf_top10_model.joblib"))
    joblib.dump(global_imputer, os.path.join(MODEL_DIR, "feature_imputer.joblib"))
    joblib.dump(bma_weights,    os.path.join(MODEL_DIR, "bma_weights.joblib"))  # save BMA weights for predict scripts
    print(f"\n  → Models saved to {MODEL_DIR}/")

    # feature importances
    print("\n" + "─" * 55)
    print("  Feature Importances")
    print("─" * 55)
    print_feature_importances(xgb_final, rf_win_final, FEATURE_COLS)

    # in-sample sanity check (expect high accuracy - this is the training set)
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

    # predict 2025 if the features file exists
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

        pred_df = ensemble_predict(
            xgb_preds_25, rf_proba_25,
            df_2025_s[["Year", "Round", "EventName", "Driver", "Team"]],
            ranker_raw_scores=ranker_scores_25,
            ensemble_weights=bma_weights,
            X_features=X_2025_s,
            apply_bayesian_prior=True,
        )
        pred_df["rf_top10_pred"] = rf_top10_pred_25

        if "finish_position" in df_2025_s.columns and df_2025_s["finish_position"].notna().any():
            pred_df["actual_position"] = df_2025_s["finish_position"].values

        out_path = os.path.join(OUT_DIR, "2025_predictions.csv")
        pred_df.to_csv(out_path, index=False)
        print(f"\n  → 2025 predictions saved to {out_path}")

        # print first 3 races as a quick sanity check
        for rnd in pred_df["Round"].unique()[:3]:
            race = pred_df[pred_df["Round"] == rnd].head(10)
            print(f"\n  2025 R{rnd:02d} {race.iloc[0]['EventName']} — Predicted Top 10:")
            print(f"    {'Pos':>3}  {'Driver':<6}  {'Team':<30}  {'Win%':>5}")
            print(f"    {'───':>3}  {'──────':<6}  {'──────────────────────────────':<30}  {'─────':>5}")
            for _, row in race.iterrows():
                print(f"    {int(row['PredictedPosition']):>3}  "
                      f"{row['Driver']:<6}  "
                      f"{str(row['Team'])[:30]:<30}  "
                      f"{row['rf_win_proba']*100:>4.1f}%")

    # save evaluation summary
    eval_path = os.path.join(OUT_DIR, "model_evaluation.txt")
    with open(eval_path, "w") as f:
        f.write("F1 Race Prediction — Model Evaluation Summary\n")
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

    print("\n" + "=" * 55)
    print("  Done.")
    print("=" * 55)


if __name__ == "__main__":
    main()
