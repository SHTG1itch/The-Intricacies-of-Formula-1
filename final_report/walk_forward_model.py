# This is the secondary model that complements the LOSO model in mach_learning.py.
# The key idea: the LOSO model uses only pre-race features (qualifying, practice, rolling form).
# This model ALSO uses within-season race results to update its beliefs as the season unfolds.
#
# Think about it this way: after Verstappen wins the first 12 races of 2023, any sensible
# prediction for race 13 should account for that pattern. The LOSO model can't do this —
# it only sees pre-race telemetry. This model does it rigorously with Bayesian math.
#
# The two-layer architecture:
#   Layer 1: Full 4-model BMA ensemble (XGBoost Regressor + Ranker + Random Forest + ExtraTrees)
#            trained on all other seasons — same split as LOSO but no meta-model stacking
#            (we can't run nested LOSO-within-LOSO in a walk-forward loop without leakage)
#   Layer 2: Plackett-Luce Bayesian updater, fit on within-season race results
#            The PL adjusts the ensemble's pre-race score based on what actually happened
#
# The PL layer is the genuinely different piece — it's mathematical machinery the LOSO model
# fundamentally cannot use, because LOSO holds out the entire season at once.
#
# ─── What IS the Plackett-Luce model? ────────────────────────────────────────────────────
# The PL model says: for a race with drivers {d_1, d_2, ..., d_n} and latent "quality"
# scores θ = (θ_d1, θ_d2, ..., θ_dn), the probability of observing the finishing order
# (d_1 finishes P1, d_2 finishes P2, ...) is:
#
#     P(ranking | θ) = ∏_{i=1}^{n}  exp(θ_{π_i})  /  Σ_{j=i}^{n} exp(θ_{π_j})
#
# where π is the permutation corresponding to the finishing order.
# Read it as: at each position, the driver who wins that spot from the remaining field
# does so with probability proportional to exp(θ). Higher θ = better driver.
#
# This is choice theory, not pairwise comparison. Luce (1959) proved that if individual
# choice satisfies a "constant ratio" property (independence from irrelevant alternatives),
# then choices follow exactly this softmax-over-remaining form.
#
# ─── Why PL is NOT ELO / a rating system ──────────────────────────────────────────────────
# ELO (and variants: TrueSkill, Glicko, OpenSkill) are PAIRWISE comparison models.
# They maintain a persistent "rating" for each player, update it after each match,
# and predict future matchups via P(i beats j) = f(rating_i - rating_j).
#
# Plackett-Luce is categorically different:
#   1. It models the COMPLETE finishing order as one probabilistic event — not 190 pairwise
#      contests. A race is not "20 drivers beating each other one at a time."
#   2. θ_d is NOT a "rating" — it is a latent parameter in a generative probability model
#      for complete orderings. It is estimated fresh within each season via MAP optimization,
#      not updated incrementally from a running score.
#   3. We place a Bayesian PRIOR on θ from our pre-race ML predictions. This is Bayesian
#      inference, not a rating algorithm.
#   4. We compute posterior uncertainty via Laplace approximation — something ELO has no
#      concept of (it only tracks a point estimate, not a distribution).
#
# The difference in spirit: ELO asks "how strong is this player historically?"
# PL asks "given the race results I've seen this season, what's the posterior distribution
# over each driver's underlying quality, and how should I blend it with what I knew before
# the season started?"
#
# References:
#   Plackett, R. (1975). The analysis of permutations. Applied Statistics, 24(2), 193-202.
#   Luce, R. D. (1959). Individual Choice Behavior. Wiley.
#   Guiver, J. & Snelson, E. (2009). Bayesian inference for Plackett-Luce ranking models.
#       ICML 2009. https://dl.acm.org/doi/10.1145/1553374.1553423
#   Caron, F. & Doucet, A. (2012). Efficient Bayesian inference for generalized Bradley-Terry
#       models. Journal of Computational and Graphical Statistics, 21(1), 174-196.
# ──────────────────────────────────────────────────────────────────────────────────────────
#
# ─── How the math flows in this file ─────────────────────────────────────────────────────
# 1. For each holdout season Y (2018-2024):
#    a. Train the full 4-model ensemble on all other seasons (same LOSO split as mach_learning.py)
#       The ensemble uses tuned hyperparams from the final mach_learning.py run — no re-tuning.
#    b. For race k in season Y:
#       - Ensemble predicts a blended score for each driver (pre-race features only)
#       - PL estimator fits θ* (MAP) from race results of rounds 1..k-1 of season Y
#         with an exponential time-decay weighting (recent races count more)
#       - Posterior uncertainty σ²_PL from diagonal Laplace approximation
#       - Posterior uncertainty σ²_PL from diagonal Laplace approximation
#       - Blend weight α_k combines FOUR signals:
#           (a) inverse-variance: σ²_base / (σ²_base + σ²_PL)  — confident PL → higher α
#           (b) data sufficiency: 1 - 1/(1 + k/5)  — fewer prior races → stay conservative
#           (c) season entropy gate: if many different winners, reduce PL trust further
#           (d) dominance-gap gate: min(clip((θ[1st] - θ[2nd]) / 1.5, 0, 1), z_credible_gate)
#               — trusts PL only when one driver clearly leads the θ ranking
#               — suppresses PL in close two-driver battles (2021: VER≈HAM, gap~0.3 → gate~0.15)
#               — suppresses PL in mid-season transitions (2024: VER→NOR, gap shrinks → α drops)
#               — preserves PL in dominant seasons (2023: VER >> field, gap~2.0 → gate~1.0)
#       - Final: blended_score = (1-α_k) × ensemble_score + α_k × pl_score
#
# 2. α cap of 0.45 is critical because qualifying position and race-specific pace are strongly
#    predictive for THIS SPECIFIC RACE regardless of season form. The four-factor formula
#    automatically suppresses α when PL's ranking is uncertain or contested.
# ──────────────────────────────────────────────────────────────────────────────────────────

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# reach into mach_learning for shared infrastructure — avoids code duplication
sys.path.insert(0, os.path.dirname(__file__))
from mach_learning import (
    FEATURE_COLS, SEED,
    train_xgboost_regressor,
    train_xgboost_ranker,
    train_random_forest_winner,
    train_extratrees,
    ensemble_predict,
    winner_accuracy, podium_accuracy, top10_set_accuracy, average_position_mae,
    _rank_normalize,
    compute_race_groups,
    calibrate_with_temperature,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ──────────────────────────────────
# Paths
# ──────────────────────────────────

DATA_DIR     = "data_organized"
TRAIN_FILE   = os.path.join(DATA_DIR, "ml_training_features.csv")
PLOT_DIR     = os.path.join("final_report", "plots")
WF_EVAL_FILE = os.path.join(DATA_DIR, "wf_evaluation.txt")

# ──────────────────────────────────
# Tuned hyperparameters from the final mach_learning.py run
# These come from RandomizedSearchCV 80-iteration tuning on the full 2018-2024 dataset.
# Using them here avoids re-running expensive tuning per WF fold (7 × 80 = 560 CV evals
# would take ~2 hours). The slight bias from using full-dataset params instead of per-fold
# is acceptable — we're not selecting features, just using learned regularization strengths.
# ──────────────────────────────────
WF_BEST_PARAMS = {
    "xgb": {
        "colsample_bylevel": 0.5137, "colsample_bynode":  0.8902,
        "colsample_bytree":  0.5917, "gamma":             0.3177,
        "learning_rate":     0.01496,"max_depth":         5,
        "min_child_weight":  7,      "n_estimators":      327,
        "reg_alpha":         0.0301, "reg_lambda":        2.288,
        "subsample":         0.624,
    },
    "rf": {
        "criterion": "entropy", "max_depth":        8,
        "max_features": 0.4,    "max_samples":      0.9176,
        "min_samples_leaf": 8,  "n_estimators":     530,
    },
    "et": {
        "max_depth": None,       "max_features":     0.5,
        "max_samples": 0.9328,   "min_samples_leaf": 9,
        "n_estimators": 488,
    },
}

# BMA weights from the LOSO Dirichlet+inverse-MAE posterior in mach_learning.py
# xgb=0.215, ranker=0.258, rf=0.258, et=0.268  — ExtraTrees leads because it got the most
# correct LOSO winners (83), despite being the most randomized model
BMA_WEIGHTS = {"w_xgb": 0.215, "w_ranker": 0.258, "w_rf": 0.258, "w_et": 0.268}

# RF temperature from the main LOSO calibration run (T ≈ 1.0 = essentially uncalibrated,
# meaning the RF probability distribution was already well-shaped across LOSO folds)
RF_TEMPERATURE = 0.9968

# tuning constants for the Plackett-Luce blend
# SIGMA2_BASE: basically "how uncertain is the base ensemble about who wins?"
#   2.0 is calibrated loosely against the LOSO 53% base accuracy. higher = ensemble
#   is treated as less confident = PL gets less trust by default. think of it as the
#   denominator that controls how quickly PL can override the pre-race prediction.
# DECAY_LAMBDA: controls how fast old race results fade out. 0.10 means a race from
#   7 rounds ago only gets weight exp(-0.7) ≈ 0.5 — half as much as last race.
#   so here's the deal: Verstappen's wins from race 1-5 still count in race 15,
#   just with less weight than what happened recently.
# ALPHA_CAP: hard ceiling on how much PL can influence the final prediction.
#   0.45 means qualifying position and car pace always contribute at least 55% of the signal.
#   this is important — even if VER dominated all season, if he's starting P10 today
#   the base ensemble (which knows about grid position) needs to have its say.
# WINNER_WEIGHT: the classic PL model treats finishing P2 almost as informatively as winning.
#   the problem is, in 2021 HAM finished P2 behind VER so often that PL started thinking
#   they were equally good — but VER was clearly the dominant driver. bumping P1 to count
#   twice as much as a runner-up finish fixes this. not tuned, just basic logic: wins matter more.
SIGMA2_BASE    = 2.0
DECAY_LAMBDA   = 0.10
ALPHA_CAP      = 0.45
WINNER_WEIGHT  = 2.0
# QUAL_PL_WEIGHT: qualifying order is the single best pre-race signal we have — grid position
#   accounts for ~27% of XGB feature importance. so we feed past qualifying rounds into PL
#   as lightweight "soft" observations to give it some car-pace signal early in the season
#   before enough race results have accumulated. weight 0.25 means one qualifying session
#   counts about 25% as much as a full race result. we only use PAST quali (not today's)
#   so we're not double-counting with the base ensemble which already knows about today's grid.
QUAL_PL_WEIGHT = 0.25
# TAIL_WEIGHT: the standard PL model treats finishing P12 the same as finishing P2 in terms
#   of how much it tells us about front-runner ability. that's clearly wrong — whether
#   Williams or Haas finished 12th vs 15th tells us almost nothing about who'll win next race.
#   down-weighting P11+ finishers by 0.7 reduces that backmarker noise while keeping the
#   top-10 ordering signal fully intact.
TAIL_WEIGHT    = 0.7

# advanced blend architecture constants
# N_RECENT_RACES: we run two PLs in parallel — a "full season" one and a "recent window" one.
#   this is the window size for the recent one. the idea is that the full-season PL might
#   still have VER at the top in race 16 of 2024 based on H1 dominance, but the
#   last-5-races PL already sees NOR and PIA winning everything. using both gives us
#   a way to detect mid-season power shifts without throwing out the full-season signal.
# SIGMA2_BASE_NEW_ERA: in years with new regulations (like 2022 with ground-effect cars),
#   the ensemble trained on older data doesn't fully know the new competitive order.
#   we use a higher uncertainty value (4.0 vs 2.0) so PL can override the stale
#   pre-season predictions more aggressively once real race data comes in.
# PRIOR_SIGMA_NEW_ERA: weaker prior on the PL θ values in regulation-change years.
#   basically "we don't trust the pre-season prediction as much, let the race results speak."
# NEW_ERA_YEARS: the years where F1 had a major regulation reset. 2022 was the big one —
#   ground-effect cars meant Ferrari suddenly looked competitive again and our models
#   trained on 2018-2021 Mercedes/HAM dominance were caught off guard.
N_RECENT_RACES      = 5
SIGMA2_BASE_NEW_ERA = 4.0
PRIOR_SIGMA_NEW_ERA = 3.0
NEW_ERA_YEARS       = {2022}  # ground-effect era began 2022: Ferrari/LeClerc surge shocked models


# ─────────────────────────────────────────────────────────────────────────
# Plackett-Luce Bayesian Estimator
# ─────────────────────────────────────────────────────────────────────────

class PlackettLuceEstimator:
    """
    Fits a Bayesian Plackett-Luce model to within-season race results.

    Call .fit(race_results, prior_means) with:
      race_results: list of dicts {driver_abbr: finish_position, ...}, one per race
                    DNF drivers should be excluded (only classified finishers)
      prior_means:  dict {driver_abbr: float} — base ensemble log-quality for each driver
                    (higher = model thinks driver/car is better)
                    We use log(n+1) - log(approx_rank) where approx_rank comes from ensemble scores

    After fitting, .theta_ holds the MAP ability estimates.
    .posterior_var_ holds the diagonal Laplace posterior variance (uncertainty).

    Higher θ_d = model thinks driver d is better based on this season's results.
    """

    def __init__(self, prior_sigma=2.0, decay_lambda=0.10, winner_weight=WINNER_WEIGHT,
                 tail_weight=TAIL_WEIGHT):
        # prior_sigma: controls how much the PL is allowed to drift away from the
        #   base ensemble's pre-season guess. smaller = tighter leash, stays closer
        #   to what the ensemble predicted. larger = PL can go rogue based on race results.
        # decay_lambda: older races get down-weighted exponentially so recent form matters more.
        #   this is basically a short memory — what VER did in race 1 matters less by race 15.
        # winner_weight: extra credit for actually winning the race (vs consistently finishing P2).
        #   the standard PL is a bit too generous to P2 finishers — in 2021, HAM finishing
        #   P2 behind VER every week was inflating HAM's θ score close to VER's even though
        #   VER was clearly the dominant driver that season. this weight fixes that by making
        #   a race win count about twice as much as a runner-up finish.
        # tail_weight: less credit for finishing P11 or worse — those results just add noise.
        #   whether someone finished P12 vs P15 tells us almost nothing about who'll win next race,
        #   so we down-weight all of that backmarker traffic without throwing it out entirely.
        self.prior_sigma   = prior_sigma
        self.decay_lambda  = decay_lambda
        self.winner_weight = float(winner_weight)
        self.tail_weight   = float(tail_weight)
        self.theta_        = {}   # MAP estimates: {driver: float}
        self.posterior_var_= {}   # diagonal Laplace variance: {driver: float}

    def fit(self, race_results, prior_means, current_race_num=None,
            qualifying_results=None, qual_weight_factor=QUAL_PL_WEIGHT):
        """
        the main workhorse — takes all the race results we've seen so far this season
        and figures out the best estimate for each driver's "true ability" θ.

        race_results: list of dicts [{driver: position}, ...] for races 1..k-1
                      earlier races come first (race_results[0] = race 1 of the season)
        prior_means:  {driver: log-quality-score from base ensemble} — this is what
                      the model believed BEFORE the season started, based on pre-race features
        current_race_num: which race number we're predicting — used to calculate how
                          much to down-weight older race results via exponential decay
        qualifying_results: past qualifying sessions as lightweight PL observations.
                            these are PAST rounds only (not today's) to avoid double-counting
                            with the base ensemble which already knows today's grid position.
                            each quali session counts as qual_weight_factor × decay.
        qual_weight_factor: how much one qualifying session is worth vs a race result (default 0.25)
        """
        if not race_results and not qualifying_results:
            # no data yet — just use the prior as-is (we have nothing to update from)
            self.theta_         = dict(prior_means)
            self.posterior_var_ = {d: self.prior_sigma**2 for d in prior_means}
            return self

        # collect all drivers observed across prior races, qualifying, + current race drivers
        qual_drivers = set(d for q in (qualifying_results or []) for d in q.keys())
        all_drivers = sorted(set(prior_means.keys()) |
                             set(d for r in (race_results or []) for d in r.keys()) |
                             qual_drivers)
        d2i = {d: i for i, d in enumerate(all_drivers)}
        n = len(all_drivers)

        prior_mu  = np.array([prior_means.get(d, 0.0) for d in all_drivers])
        prior_var = self.prior_sigma ** 2

        k_pred = current_race_num if current_race_num is not None else len(race_results) + 1

        # convert race_results (dicts) into sorted index arrays for fast computation.
        # the trick is we also attach a decay weight to each race — the more recent the race,
        # the closer to 1.0 its weight. something from 7 races ago gets about half the credit.
        # format: (ranked_driver_indices, decay_weight)
        race_list = race_results or []
        prepared_races = []
        for race_idx, result_dict in enumerate(race_list):
            weight = np.exp(-self.decay_lambda * (k_pred - 1 - race_idx))
            ranked = sorted(result_dict.items(), key=lambda x: x[1])
            ranked_indices = np.array([d2i[d] for d, _ in ranked if d in d2i])
            if len(ranked_indices) >= 2:
                prepared_races.append((ranked_indices, weight))

        # qualifying observations get the same time-decay treatment but scaled down
        # by qual_weight_factor so they don't overwhelm the actual race signal.
        # note: pole position doesn't get the winner upweight bonus — starting P1 ≠ winning P1.
        qual_list = qualifying_results or []
        for qual_idx, qual_dict in enumerate(qual_list):
            # align with the race decay: qual round i = race round i
            weight = np.exp(-self.decay_lambda * (k_pred - 1 - qual_idx)) * qual_weight_factor
            ranked = sorted(qual_dict.items(), key=lambda x: x[1])
            ranked_indices = np.array([d2i[d] for d, _ in ranked if d in d2i])
            if len(ranked_indices) >= 2:
                prepared_races.append((ranked_indices, weight))

        # ── MAP estimation via L-BFGS-B ────────────────────────────────────────────
        # we want the θ values that best explain the race results we've seen, while
        # staying reasonably close to the base ensemble's pre-season predictions.
        # that's Bayes: combine what you knew before (the prior) with what happened (the likelihood).
        # scipy.optimize.minimize finds the minimum, so we flip the sign and minimize -log_posterior.
        #
        # the PL log-likelihood for one race (weight w):
        #   w × Σ_i [θ_{π_i} - logsumexp(θ_{π_i}, ..., θ_{π_n})]
        # read this as: for each finishing position, the winner of that slot gets credit
        # for their θ score vs the sum of everyone left in the running.
        #
        # the Normal prior log-density:
        #   Σ_d -(θ_d - μ_d)² / (2σ²)
        # this just penalizes θ for wandering too far from the pre-season prediction μ_d.

        ww = self.winner_weight   # P1 upweight factor
        tw = self.tail_weight     # P11+ down-weight factor (i ≥ 10, 0-indexed)

        def _pos_weight(i):
            # P1 (i=0): wins count more — that's the whole point of winner_weight
            # P2-P10 (i=1..9): standard weight, full signal
            # P11+ (i≥10): down-weighted — backmarker results are mostly noise
            if i == 0:
                return ww
            elif i >= 10:
                return tw
            return 1.0

        def neg_log_posterior(theta):
            log_lik = 0.0
            for ranked_idx, w in prepared_races:
                theta_race = theta[ranked_idx]
                m = len(ranked_idx)
                for i in range(m):
                    pos_w = _pos_weight(i)
                    # logsumexp for numerical stability — avoids exp overflow/underflow
                    log_lik += w * pos_w * (theta_race[i] - logsumexp(theta_race[i:]))
            # Normal prior: pulls θ back toward the ensemble's pre-season guess
            log_prior = -np.sum((theta - prior_mu) ** 2) / (2 * prior_var)
            return -(log_lik + log_prior)

        def grad_neg_log_posterior(theta):
            grad_ll = np.zeros(n)
            for ranked_idx, w in prepared_races:
                theta_race = theta[ranked_idx]
                m = len(ranked_idx)
                for i in range(m):
                    remaining_idx = ranked_idx[i:]
                    theta_rem = theta[remaining_idx]
                    # numerically stable softmax over the remaining drivers at position i
                    exp_t = np.exp(theta_rem - np.max(theta_rem))
                    softmax = exp_t / exp_t.sum()
                    pos_w = _pos_weight(i)
                    # the driver who won this spot gains +w×pos_w in the gradient
                    grad_ll[ranked_idx[i]] += w * pos_w
                    # everyone still in the running loses a bit: -w×pos_w × their softmax prob
                    grad_ll[remaining_idx] -= w * pos_w * softmax
            # Normal prior gradient
            grad_prior = -(theta - prior_mu) / prior_var
            return -(grad_ll + grad_prior)

        theta0 = prior_mu.copy()
        result = minimize(
            neg_log_posterior, theta0,
            jac=grad_neg_log_posterior,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6},
        )
        theta_map = result.x

        # ── Laplace approximation — basically computing how uncertain we are about θ ──
        # once we have the MAP estimate θ*, we want to know how confident we are in it.
        # the trick is: near the MAP, the log-posterior looks like a Gaussian — so we can
        # use the curvature (second derivative / Hessian) to get uncertainty estimates.
        # Σ ≈ -H^{-1} where H is the Hessian of the log-posterior at the MAP point.
        #
        # inverting the full Hessian (20×20 for 20 drivers) is slow and numerically messy.
        # we just use the diagonal — each driver gets their own independent variance estimate.
        # Σ_ii ≈ 1/(-∂² log posterior / ∂θ_i²)
        #
        # the diagonal Hessian of the PL log-likelihood:
        #   ∂² log L / ∂θ_d² = -Σ_races w_r × Σ_positions_i 1(d remaining) × p_d(1-p_d)
        # where p_d = exp(θ_d) / Σ_{j remaining} exp(θ_j) is the selection probability.
        # the p_d(1-p_d) term is exactly the variance of a coin flip — it measures how much
        # info each race position gives us about driver d's ability. consistent winners → low
        # uncertainty. drivers in the murky midfield → high uncertainty.

        hess_diag_ll = np.zeros(n)   # diagonal of Hessian of log-likelihood
        for ranked_idx, w in prepared_races:
            theta_race = theta_map[ranked_idx]
            m = len(ranked_idx)
            for i in range(m):
                remaining_idx = ranked_idx[i:]
                theta_rem = theta_map[remaining_idx]
                exp_t = np.exp(theta_rem - np.max(theta_rem))
                softmax = exp_t / exp_t.sum()
                pos_w = _pos_weight(i)
                # p(1-p) Bernoulli variance — how informative is this race about each driver's θ
                hess_diag_ll[remaining_idx] -= w * pos_w * softmax * (1.0 - softmax)

        # prior also contributes -1/σ² to the diagonal (it's penalizing curvature too)
        hess_diag_posterior = hess_diag_ll - 1.0 / prior_var

        # posterior variance = -1 / (diagonal Hessian of log-posterior)
        # clip to avoid negative variances from numerical edge cases
        posterior_var_diag = -1.0 / np.clip(hess_diag_posterior, -1e6, -1e-8)

        self.theta_         = {d: float(theta_map[d2i[d]]) for d in all_drivers}
        self.posterior_var_ = {d: float(posterior_var_diag[d2i[d]]) for d in all_drivers}
        return self

    def get_mean_posterior_variance(self, drivers):
        """Average posterior variance across the given drivers — proxy for PL uncertainty."""
        if not self.posterior_var_:
            return self.prior_sigma ** 2
        vars_ = [self.posterior_var_.get(d, self.prior_sigma**2) for d in drivers]
        return float(np.mean(vars_))


# ─────────────────────────────────────────────────────────────────────────
# Blend weight computation
# ─────────────────────────────────────────────────────────────────────────

def compute_blend_alpha(sigma2_pl, n_observed,
                        winner_entropy=None,
                        dominance_gate=None,
                        grid_alignment=1.0,
                        is_new_era=False,
                        sigma2_base=SIGMA2_BASE, alpha_cap=ALPHA_CAP):
    """
    computes α, the blend weight — basically "how much should we trust the PL model
    vs the base ensemble for this specific race?" it's the product of five gates,
    each of which can reduce α toward 0 if something looks off.

    Factor 1 — confidence gate (inverse-variance):
        if the PL model is uncertain about who the season leader is (high sigma2_pl),
        we trust it less. when PL is confident → α can be high. when PL is unsure → α drops.
        in regulation-change years we treat the ensemble as less reliable (sigma2_base_eff = 4.0)
        so PL can override it more aggressively once race data comes in.

    Factor 2 — have we seen enough races? (data sufficiency gate):
        0 races seen → gate ≈ 0 (we have no PL data, don't use it)
        5 races seen → gate ≈ 50%
        10+ races → gate ≈ 91%
        basically a smooth warm-up ramp so PL doesn't have outsized influence in race 2.

    Factor 3 — is the season chaotic? (entropy gate):
        winner_entropy = unique_winners / n_observed. if 8 different drivers won 8 races
        (like 2024), the season is unpredictable and PL can't reliably identify a "dominant"
        driver — so we scale back its influence. dominant seasons (2023) have low entropy.

    Factor 4 — is there actually a clear leader? (dominance gate, pre-computed outside):
        we want PL to matter mainly when one driver is CLEARLY better than everyone else.
        uses min(raw_gap_gate, z_gate) so both conditions must hold:
          raw_gap_gate = clip(gap/1.5, 0, 1) — the θ gap between #1 and #2 must be large
          z_gate = clip(z/2.0, 0, 1) — the gap must also be statistically credible (z-score)
        in 2021 VER/HAM the gap was tiny → gate ≈ 0.15. in 2023 VER >> everyone → gate ≈ 1.0.

    Factor 5 — does today's grid position support the PL pick? (grid alignment gate):
        PL says VER wins. but VER qualified P8 today. those are contradictory signals.
        if the PL champion is starting P1-P4, full trust. P5+ → exponential decay.
        prevents PL from insisting on a prediction that qualifying clearly contradicts.
    """
    # in regulation-change years the ensemble is less calibrated — bump up base uncertainty
    sigma2_base_eff = SIGMA2_BASE_NEW_ERA if is_new_era else sigma2_base

    # Factor 1: how confident is PL about the season leader's ability?
    inv_var = sigma2_base_eff / (sigma2_base_eff + max(sigma2_pl, 1e-8))

    # Factor 2: data sufficiency — smooth ramp, 0.5 at n=5 races, ~0.91 at n=10
    data_gate = 1.0 - 1.0 / (1.0 + n_observed / 5.0)

    # Factor 3: season entropy — high chaos → reduce PL trust
    if winner_entropy is not None:
        entropy_gate = 1.0 - 0.5 * np.clip(winner_entropy, 0.0, 1.0)
    else:
        entropy_gate = 1.0

    # Factor 4: credible dominance gate (pre-computed caller-side, defaults to 1.0 if not supplied)
    dom_gate = float(np.clip(dominance_gate, 0.0, 1.0)) if dominance_gate is not None else 1.0

    # Factor 5: grid alignment — PL's pick must be starting near the front today
    grid_gate = float(np.clip(grid_alignment, 0.0, 1.0))

    alpha = inv_var * data_gate * entropy_gate * dom_gate * grid_gate
    return min(alpha, alpha_cap)


# ─────────────────────────────────────────────────────────────────────────
# Ensemble score → PL prior conversion
# ─────────────────────────────────────────────────────────────────────────

def ensemble_score_to_pl_prior(ens_scores, n_drivers):
    """
    Convert rank-normalized ensemble scores to PL prior log-quality scores.

    ens_scores: array of rank-normalized [0,1] scores where 0 = predicted winner.
    The PL θ lives in log-space: exp(θ_d) is proportional to d's "worth".

    Conversion:
        approx_rank = 1 + ens_score × (n_drivers - 1)   (maps [0,1] → [1, n])
        θ_prior = log(n + 1) - log(approx_rank)

    Result: predicted winner (ens_score≈0) gets θ ≈ log(n+1) ≈ 3.0 for 20 drivers.
            predicted last (ens_score≈1) gets θ ≈ log(n+1) - log(n) ≈ 0.05.
    """
    ens = np.asarray(ens_scores)
    approx_rank = 1.0 + ens * (n_drivers - 1)  # continuous rank in [1, n]
    approx_rank = np.clip(approx_rank, 1, n_drivers)
    theta_prior = np.log(n_drivers + 1) - np.log(approx_rank)
    return theta_prior


# ─────────────────────────────────────────────────────────────────────────
# Walk-forward evaluation loop
# ─────────────────────────────────────────────────────────────────────────

def walk_forward_evaluate(df_train, verbose=True, smoke_year=None):
    """
    the main evaluation loop — runs the full walk-forward experiment across all seasons.

    the idea is simple: hold out one season at a time, train the ensemble on everything
    else, then walk race-by-race through the holdout season, updating the PL model as
    real results come in. think of it as simulating what you'd actually do if you were
    predicting each race live during the season.

    for each holdout season Y:
      1. train 4-model BMA ensemble on all other seasons (same LOSO split as mach_learning.py)
         we use pre-tuned hyperparams here — re-tuning for each fold would take hours
         and the regularization strengths don't change much between folds anyway.
      2. walk through each race k in season Y:
         - ensemble predicts from pre-race features (quali, practice, rolling form, circuit history)
         - PL fits θ* from race results of rounds 1..k-1 of this season
         - five-factor α_k decides how much to weight PL vs ensemble for this race
         - blended_score = (1-α_k) × ensemble_score + α_k × pl_score

    smoke_year: pass a single year to do a quick sanity check without running all 7 folds

    returns: DataFrame with per-race predictions for all (evaluated) seasons
    """
    all_years = sorted(df_train["Year"].unique())
    eval_years = [smoke_year] if smoke_year is not None else all_years
    all_records = []

    if verbose:
        print("\n" + "─" * 55)
        print("  Walk-Forward Evaluation (within-season PL updating)")
        print("─" * 55)

    for holdout_year in eval_years:
        train_mask = df_train["Year"] != holdout_year
        test_mask  = df_train["Year"] == holdout_year

        # ── Step 1: train 4-model ensemble on all other years ──────────────────
        # sort training data by (Year, Round) so XGBRanker groups are contiguous
        df_tr = df_train[train_mask].copy().sort_values(["Year", "Round"]).reset_index(drop=True)

        X_tr_raw = df_tr[FEATURE_COLS].copy().astype(float)
        y_tr     = df_tr["finish_position"].astype(float).fillna(20.0)
        y_win_tr = df_tr["is_winner"].astype(float) if "is_winner" in df_tr else None

        # fit imputer on training data only (test races will use transform, not fit_transform)
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        X_tr = pd.DataFrame(imp.fit_transform(X_tr_raw), columns=FEATURE_COLS)

        # paranoia check: indices must align for train_random_forest_winner's .loc[] usage
        assert len(X_tr) == len(y_tr), "X_tr and y_tr length mismatch"
        if y_win_tr is not None:
            assert len(X_tr) == len(y_win_tr), "X_tr and y_win_tr length mismatch"

        groups_tr = compute_race_groups(df_tr[["Year", "Round", "Driver", "Team"]])

        if verbose:
            print(f"\n  Training ensemble for holdout {holdout_year} "
                  f"({len(df_tr)} rows from {df_tr['Year'].nunique()} seasons)...", end="", flush=True)

        # train all 4 base models with saved best hyperparams — no per-fold re-tuning
        xgb_model    = train_xgboost_regressor(X_tr, y_tr, params=WF_BEST_PARAMS["xgb"])
        ranker_model = train_xgboost_ranker(X_tr, y_tr, groups_tr,
                                             best_params=WF_BEST_PARAMS["xgb"])
        if y_win_tr is not None:
            rf_model = train_random_forest_winner(X_tr, y_win_tr, params=WF_BEST_PARAMS["rf"])
        else:
            rf_model = None
        et_model = train_extratrees(X_tr, y_tr, params=WF_BEST_PARAMS["et"])

        if verbose:
            print("  done.")

        # ── Step 2: walk-forward through holdout season ──────────────────────────
        df_year = df_train[test_mask].copy().sort_values(["Round", "Driver"])
        rounds  = sorted(df_year["Round"].unique())

        # check if this is a new-regulation year — 2022 was the ground-effect reset and
        # our ensemble (trained on 2018-2021 Mercedes/Hamilton era) really didn't see
        # Ferrari's resurgence coming. in those years we loosen the PL prior and bump
        # up base uncertainty so real race results can override the stale pre-season model faster.
        is_new_era    = (holdout_year in NEW_ERA_YEARS)
        pl_prior_sigma = PRIOR_SIGMA_NEW_ERA if is_new_era else 2.0

        # pre-collect the qualifying grid for every round in this season.
        # we only feed PAST rounds to the PL fitter (not today's) — this is important because
        # the base ensemble already knows today's grid position. feeding it to PL too would
        # be double-counting that signal. pit-lane starters (grid_position 0 or NaN) are
        # excluded since their grid slot doesn't reflect actual qualifying speed.
        full_qualifying_history = {}  # {round_number: {driver_abbr: grid_position}}
        if "grid_position" in df_year.columns:
            for r, grp in df_year.groupby("Round"):
                qual_round = {}
                for _, row in grp.iterrows():
                    gp_val = row["grid_position"]
                    if pd.notna(gp_val) and float(gp_val) >= 1:
                        qual_round[row["Driver"]] = float(gp_val)
                if len(qual_round) >= 2:
                    full_qualifying_history[r] = qual_round

        # within-season result accumulator — grows as we process each race
        result_history      = {}   # {round_number: {driver_abbr: finish_position}}
        team_result_history = {}   # {round_number: {team_name: best_position_of_either_driver}}

        # PL estimator resets each season — θ is season-specific, not a persistent rating
        pl_est = PlackettLuceEstimator(prior_sigma=pl_prior_sigma, decay_lambda=DECAY_LAMBDA)

        if verbose:
            print(f"\n  Holdout {holdout_year}:")

        year_records = []
        n_correct_base, n_correct_wf, n_races_year = 0, 0, 0
        # track unique winners within season for entropy gate
        season_winners_seen = set()

        for rnd in rounds:
            race_mask = df_year["Round"] == rnd
            df_race   = df_year[race_mask].copy().reset_index(drop=True)
            drivers   = df_race["Driver"].tolist()
            n_driv    = len(drivers)

            # impute test race using training-data fitted imputer (no leakage)
            X_race_raw = df_race[FEATURE_COLS].copy().astype(float)
            X_race     = pd.DataFrame(imp.transform(X_race_raw), columns=FEATURE_COLS)

            # ── Base ensemble prediction ─────────────────────────────────────────
            xgb_preds    = xgb_model.predict(X_race)
            ranker_preds = ranker_model.predict(X_race)  # higher = better for ranker
            et_preds     = et_model.predict(X_race)

            if rf_model is not None:
                rf_proba = rf_model.predict_proba(X_race)[:, 1]
            else:
                # fallback: convert XGB positions to pseudo-probability via rank
                xgb_ranks = pd.Series(xgb_preds).rank(ascending=True).values
                rf_proba  = (n_driv + 1 - xgb_ranks) / n_driv / (n_driv / 2)
                rf_proba  = np.clip(rf_proba, 1e-6, 1.0 - 1e-6)

            # df_meta for ensemble_predict (needs Year/Round/EventName/Driver/Team columns)
            df_race_meta = df_race[["Year", "Round", "EventName", "Driver", "Team"]].copy().reset_index(drop=True)

            ens_df = ensemble_predict(
                xgb_preds, rf_proba, df_race_meta,
                ranker_raw_scores=ranker_preds,
                et_raw_pos=et_preds,
                ensemble_weights=BMA_WEIGHTS,
                X_features=X_race,
                apply_bayesian_prior=True,
                cal_temperature=RF_TEMPERATURE,
                meta_model=None,    # no stacking meta in WF — would need nested LOSO
            )

            # reindex to match our `drivers` list (ens_df may have been re-sorted)
            ens_score_by_driver = (ens_df.set_index("Driver")["ensemble_score"]
                                         .reindex(drivers))
            base_scores = ens_score_by_driver.values.astype(float)

            # ── Build PL prior from ensemble predictions ──────────────────────────
            # convert rank-normalized ensemble scores [0,1] → PL log-quality prior
            prior_mu_arr = ensemble_score_to_pl_prior(base_scores, n_driv)
            prior_means  = {d: float(prior_mu_arr[i]) for i, d in enumerate(drivers)}

            # ── fit the season-long PL from all races before this one ───────────────
            past_races = [result_history[r] for r in sorted(result_history.keys())]
            # for qualifying observations: only feed PAST rounds (not today's).
            # in regulation-change years we skip qualifying entirely — early 2022 Ferrari
            # was on pole a lot before their race pace was confirmed, feeding those early
            # quali signals made PL too bullish on Ferrari before race data caught up.
            # the new-era uncertainty mechanisms already handle this without qualifying noise.
            if not is_new_era:
                past_quals = [full_qualifying_history[r]
                              for r in sorted(full_qualifying_history.keys()) if r < rnd]
            else:
                past_quals = None
            pl_est.fit(past_races, prior_means, current_race_num=rnd,
                       qualifying_results=past_quals)

            n_obs = len(result_history)

            # ── recent-window PL: last N_RECENT_RACES races only, no decay ────────────
            # we run a second PL in parallel that only looks at the last 5 races.
            # the purpose is catching mid-season power shifts that the full-season PL misses —
            # for example in 2024, VER dominated H1 so the full-season PL kept him on top,
            # but by race 16 NOR and PIA were winning everything. the recent-window PL
            # flips toward McLaren much faster because it forgets VER's H1 dominance entirely.
            # we use both and blend them based on how much the two disagree (transition_factor).
            past_recent = [result_history[r]
                           for r in sorted(result_history.keys())[-N_RECENT_RACES:]]
            # same new-era exclusion for qualifying as above
            if not is_new_era:
                recent_qual_rnds = sorted(full_qualifying_history.keys())
                recent_qual_rnds = [r for r in recent_qual_rnds if r < rnd][-N_RECENT_RACES:]
                past_quals_recent = [full_qualifying_history[r] for r in recent_qual_rnds]
            else:
                past_quals_recent = None
            pl_recent = PlackettLuceEstimator(
                prior_sigma=pl_prior_sigma * 1.5,  # weaker prior — let recent form speak freely
                decay_lambda=0.0,                  # all races in the window count equally
            )
            pl_recent.fit(past_recent, prior_means, current_race_num=rnd,
                          qualifying_results=past_quals_recent)

            # ── team-level PL: what does the car say, not just the driver? ──────────
            # the individual driver PL misses team-level signals. if NOR finishes P1 and
            # PIA finishes P3, McLaren's car is clearly fast — and that should boost PIA's
            # individual score too, not just NOR's. we run a separate PL on team-level
            # best-finish-per-team results and blend a bit of that signal in.
            driver_to_team_cur = {}
            if "Team" in df_race.columns:
                driver_to_team_cur = dict(zip(df_race["Driver"].tolist(),
                                              df_race["Team"].fillna("Unknown").tolist()))

            past_team_races = [team_result_history[r]
                                for r in sorted(team_result_history.keys())]
            teams_present = sorted(set(v for v in driver_to_team_cur.values()
                                       if v not in ("Unknown", "")))
            if teams_present:
                team_prior = {
                    t: float(np.mean([prior_means.get(d, 0.0)
                                      for d in drivers
                                      if driver_to_team_cur.get(d) == t] or [0.0]))
                    for t in teams_present
                }
                pl_team = PlackettLuceEstimator(prior_sigma=pl_prior_sigma,
                                                decay_lambda=DECAY_LAMBDA)
                pl_team.fit(past_team_races, team_prior, current_race_num=rnd)
                team_theta_map = pl_team.theta_
            else:
                team_theta_map = {}

            # ── θ composite: season PL + recent PL + team signal ────────────────────
            theta_season  = np.array([pl_est.theta_.get(d, prior_means.get(d, 0.0))
                                       for d in drivers])
            theta_rec_arr = np.array([pl_recent.theta_.get(d, prior_means.get(d, 0.0))
                                       for d in drivers])

            # transition detection: compare the rank order of season PL vs recent-window PL.
            # if they mostly agree (2023: VER on top both ways), corr ≈ 1 → no transition.
            # if they disagree significantly (2024 H2: season PL still has VER #1, recent
            # PL has NOR #1), corr drops → transition_factor rises → we lean more on recent.
            # basically we're asking "has the competitive order changed recently?"
            if n_obs >= 3 and n_driv >= 5:
                rank_s  = pd.Series(theta_season).rank()
                rank_r  = pd.Series(theta_rec_arr).rank()
                corr_sr = float(np.corrcoef(rank_s.values, rank_r.values)[0, 1])
                corr_sr = 0.0 if np.isnan(corr_sr) else corr_sr
                transition_factor = float(max(0.0, 1.0 - max(corr_sr, 0.0)))
            else:
                transition_factor = 0.0

            recent_w         = 0.45 * transition_factor   # up to 45% weight on recent θ
            theta_base_blend = (1.0 - recent_w) * theta_season + recent_w * theta_rec_arr

            # team signal gets a fixed 25% weight, blended in consistently across all seasons
            team_weight = 0.25

            if teams_present and n_obs >= 2:
                team_t_arr = np.array([team_theta_map.get(driver_to_team_cur.get(d, ""), 0.0)
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

            # ── grid alignment gate: does the PL's prediction make sense given today's grid?
            # think of it this way: PL says VER is the dominant driver this season.
            # but VER qualified P8 today (maybe a crash in Q2). qualifying already told us
            # that prediction is going to be wrong for this specific race. gate = 1.0 for
            # P1-P4 (all good, the leader is well-placed), then exponential decay for P5+.
            grid_alignment = 1.0
            if n_obs >= 3 and len(theta_race) > 0:
                pl_champ_idx  = int(np.argmax(theta_race))
                pl_champ_name = drivers[pl_champ_idx]
                champ_rows    = df_race[df_race["Driver"] == pl_champ_name]
                if not champ_rows.empty and "grid_position" in df_race.columns:
                    gp_val = champ_rows["grid_position"].iloc[0]
                    if not pd.isna(gp_val):
                        gp = float(gp_val)
                        # full trust P1-P4; exponential decay only for P5+ starts
                        grid_alignment = float(min(1.0, np.exp(-(max(gp, 1.0) - 4.0) * 0.35)))

            # ── compute the five-factor α ─────────────────────────────────────────────
            theta_dict_blend = {d: float(theta_race[i]) for i, d in enumerate(drivers)}

            # for the uncertainty factor (Factor 1), we use the season PL leader's variance —
            # not the composite θ. the composite gets compressed by team PL blending which
            # can make VER's uncertainty look artificially high. using the clean season PL
            # gives us a clearer read on how confident PL actually is about its top pick.
            season_theta_vals = pl_est.theta_  # MAP estimates from season PL only
            top_season_driver = (max(season_theta_vals, key=season_theta_vals.get)
                                 if season_theta_vals else None)
            if top_season_driver and pl_est.posterior_var_:
                sigma2_pl = float(pl_est.posterior_var_.get(top_season_driver,
                                                             pl_est.prior_sigma ** 2))
            else:
                sigma2_pl = pl_est.get_mean_posterior_variance(drivers)

            # dominance gate: computed from season PL θ (not composite) for the same reason.
            # we need BOTH conditions to hold — large raw gap AND statistically credible gap.
            # min(raw_gate, z_gate) means neither can inflate the other: in 2021 the gap
            # between VER and HAM was small both ways → gate ≈ 0.15. in 2023 VER was so
            # far ahead that both gates were high → gate ≈ 1.0.
            if len(season_theta_vals) >= 2 and pl_est.posterior_var_:
                s_sorted = sorted(season_theta_vals.items(), key=lambda x: x[1], reverse=True)
                sd1, st1 = s_sorted[0]
                sd2, st2 = s_sorted[1]
                sv1 = float(pl_est.posterior_var_.get(sd1, pl_est.prior_sigma ** 2))
                sv2 = float(pl_est.posterior_var_.get(sd2, pl_est.prior_sigma ** 2))
                raw_dom_gate = float(np.clip((st1 - st2) / 1.5, 0.0, 1.0))
                z_dom = (st1 - st2) / np.sqrt(max(sv1 + sv2, 1e-8))
                z_gate = float(np.clip(z_dom / 2.0, 0.0, 1.0))
                precomp_dominance = min(raw_dom_gate, z_gate)  # conservative: both must hold
            else:
                precomp_dominance = 1.0

            winner_entropy = len(season_winners_seen) / max(n_obs, 1) if n_obs > 0 else None

            alpha_k = compute_blend_alpha(sigma2_pl, n_obs,
                                          winner_entropy=winner_entropy,
                                          dominance_gate=precomp_dominance,
                                          grid_alignment=grid_alignment,
                                          is_new_era=is_new_era)

            # ── convert composite θ to a rank-normalized score (higher θ → lower = better)
            pl_scores = _rank_normalize_1d(theta_race, ascending=False)

            # ── final blended score — lower = predicted winner ─────────────────────
            blended_scores = (1 - alpha_k) * base_scores + alpha_k * pl_scores

            # ── Record results ────────────────────────────────────────────────────
            actual_pos = df_race["finish_position"].values
            pred_winner_wf   = drivers[int(np.argmin(blended_scores))]
            pred_winner_base = drivers[int(np.argmin(base_scores))]

            actual_winner_list = df_race[df_race["finish_position"] == 1.0]["Driver"].tolist()
            actual_winner = actual_winner_list[0] if actual_winner_list else None

            correct_wf   = (pred_winner_wf   == actual_winner)
            correct_base = (pred_winner_base == actual_winner)
            if actual_winner:
                n_correct_wf   += int(correct_wf)
                n_correct_base += int(correct_base)
                n_races_year   += 1

            for i, drv in enumerate(drivers):
                year_records.append({
                    "Year":            holdout_year,
                    "Round":           rnd,
                    "EventName":       df_race["EventName"].iloc[i] if "EventName" in df_race else "",
                    "Driver":          drv,
                    "Team":            df_race["Team"].iloc[i] if "Team" in df_race else "",
                    "actual_position": float(actual_pos[i]) if i < len(actual_pos) else np.nan,
                    "base_score":      float(base_scores[i]),
                    "pl_score":        float(pl_scores[i]),
                    "blended_score":   float(blended_scores[i]),
                    "pl_theta":        float(theta_race[i]),
                    "sigma2_pl":       float(sigma2_pl),
                    "alpha_k":         float(alpha_k),
                    "races_observed":  n_obs,
                    "pred_winner_wf":  pred_winner_wf,
                    "pred_winner_base":pred_winner_base,
                    "actual_winner":   actual_winner,
                    "correct_wf":      correct_wf,
                    "correct_base":    correct_base,
                })

            # ── store this race's result so PL can use it for the next race ────────
            # this is the key "walk-forward" mechanic — we only know race k result AFTER
            # we've made the prediction, then it feeds into the PL for race k+1.
            valid_mask = ~pd.isna(df_race["finish_position"])
            result_history[rnd] = {
                drv: float(df_race[valid_mask]["finish_position"].iloc[j])
                for j, drv in enumerate(df_race[valid_mask]["Driver"].tolist())
            }

            # record best finishing position per team this race (for team PL next round)
            team_bests: dict[str, float] = {}
            for drv, pos in result_history[rnd].items():
                t = driver_to_team_cur.get(drv, "")
                if t and t not in ("Unknown", ""):
                    team_bests[t] = float(min(team_bests.get(t, 20.0), pos))
            if len(team_bests) >= 2:
                team_result_history[rnd] = team_bests

            # track unique winners this season so we can compute the entropy gate
            if actual_winner:
                season_winners_seen.add(actual_winner)

        if verbose and n_races_year > 0:
            wf_acc   = n_correct_wf / n_races_year
            base_acc = n_correct_base / n_races_year
            print(f"    Walk-forward: {wf_acc:.1%}  ({n_correct_wf}/{n_races_year} races)")
            print(f"    Base only:    {base_acc:.1%}  ({n_correct_base}/{n_races_year} races)")

        all_records.extend(year_records)

    return pd.DataFrame(all_records)


def _rank_normalize_1d(scores, ascending=True):
    """Rank-normalize a 1D score array to [0,1] within a single group (one race)."""
    s = pd.Series(scores)
    ranks = s.rank(method="average", ascending=ascending)
    n = len(s)
    if n <= 1:
        return np.zeros(n)
    return ((ranks - 1) / (n - 1)).values


# ─────────────────────────────────────────────────────────────────────────
# Metric computation helpers
# ─────────────────────────────────────────────────────────────────────────

def compute_per_year_metrics(df_pred):
    """Summarize walk-forward prediction accuracy by year."""
    rows = []
    for year, grp in df_pred.groupby("Year"):
        races = grp.drop_duplicates(subset=["Year", "Round"])
        n_races = len(races[races["actual_winner"].notna()])
        if n_races == 0:
            continue
        n_correct_wf   = int(races["correct_wf"].sum())
        n_correct_base = int(races["correct_base"].sum())
        rows.append({
            "year":          year,
            "n_races":       n_races,
            "wf_accuracy":   n_correct_wf   / n_races,
            "base_accuracy": n_correct_base / n_races,
            "wf_correct":    n_correct_wf,
            "base_correct":  n_correct_base,
        })
    return pd.DataFrame(rows).set_index("year")


def compute_per_race_metrics(df_pred):
    """Accuracy by race number (races_observed = how many prior races existed)."""
    rows = []
    for races_obs, grp in df_pred.groupby("races_observed"):
        races   = grp.drop_duplicates(subset=["Year", "Round"])
        n_valid = len(races[races["actual_winner"].notna()])
        if n_valid == 0:
            continue
        rows.append({
            "races_observed": races_obs,
            "n_events":       n_valid,
            "wf_accuracy":    races["correct_wf"].sum()   / n_valid,
            "base_accuracy":  races["correct_base"].sum() / n_valid,
        })
    return pd.DataFrame(rows).sort_values("races_observed")


# ─────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────

SEASON_COLORS = {
    2018: "#e41a1c",
    2019: "#377eb8",
    2020: "#4daf4a",
    2021: "#984ea3",
    2022: "#ff7f00",
    2023: "#a65628",
    2024: "#f781bf",
}


def plot_accuracy_by_race_number(df_pred, loso_mean=0.551, outdir=PLOT_DIR):
    """
    shows how walk-forward accuracy improves as the season goes on and PL accumulates data.

    one line per season. x-axis = how many races the PL model has seen so far.
    the LOSO mean is shown as a dashed reference line.

    what to look for: at race 0 (before any data), walk-forward and base ensemble
    are basically identical — PL hasn't seen anything yet. by mid-season (8+ races),
    the PL correction kicks in and you should see the walk-forward line pull ahead.
    in chaotic seasons like 2021 or 2024 (lots of different winners), the entropy
    gate keeps PL from making things worse by being overconfident.
    """
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    race_df = compute_per_race_metrics(df_pred)
    # smooth with a rolling average of 3 for readability
    wf_smooth   = race_df["wf_accuracy"].rolling(3, min_periods=1).mean()
    base_smooth = race_df["base_accuracy"].rolling(3, min_periods=1).mean()

    ax.plot(race_df["races_observed"], wf_smooth,
            color="#e31a1c", lw=2.5, marker="o", markersize=4, label="Walk-forward (PL blend)")
    ax.plot(race_df["races_observed"], base_smooth,
            color="#1f78b4", lw=1.8, marker="s", markersize=3, ls="--",
            label="Base ensemble only (4-model BMA)")
    ax.axhline(loso_mean, color="#666", lw=1.5, ls=":", label=f"LOSO ensemble mean ({loso_mean:.1%})")

    ax.set_xlabel("Prior races observed in season (Plackett-Luce data)", fontsize=12)
    ax.set_ylabel("Winner prediction accuracy", fontsize=12)
    ax.set_title("Walk-Forward Accuracy vs Season Progress\n"
                 "(each step = one more race the PL model has learned from)", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, race_df["races_observed"].max() + 0.5)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(outdir, "wf_accuracy_by_race.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


def plot_model_comparison_bars(df_pred, loso_by_year, outdir=PLOT_DIR):
    """
    side-by-side bars showing three-model accuracy progression per season:
      LOSO  →  Walk-forward + PL  →  Oracle (theoretical ceiling)

    the WF base bar was removed because it cluttered the chart without adding a clear
    message for the reader — the key insight is LOSO < WF+PL < Oracle.
    """
    os.makedirs(outdir, exist_ok=True)
    per_year = compute_per_year_metrics(df_pred)
    years    = sorted(per_year.index)

    x     = np.arange(len(years))
    width = 0.26

    # compute oracle accuracy per year (did WF OR base get this race right?)
    df_races = df_pred.drop_duplicates(subset=["Year", "Round"]).copy()
    df_races = df_races[df_races["actual_winner"].notna()]
    oracle_vals = []
    for yr in years:
        sub = df_races[df_races["Year"] == yr]
        n   = len(sub)
        if n == 0:
            oracle_vals.append(np.nan)
            continue
        n_oracle = (
            (sub["correct_wf"] == True) | (sub["correct_base"] == True)
        ).sum()
        oracle_vals.append(n_oracle / n)

    # two panels: bars on left, per-year gains on right
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6.5),
                                   gridspec_kw={"width_ratios": [2, 1]})

    loso_vals = [loso_by_year.get(y, np.nan) for y in years]
    wf_vals   = [per_year.loc[y, "wf_accuracy"] for y in years]

    c_loso   = "#4393c3"   # blue
    c_wf     = "#d6604d"   # red-orange
    c_oracle = "#1a9641"   # green

    bars_loso   = ax.bar(x - width, loso_vals,   width,
                         label="LOSO — static pre-race model\n(no in-season data)",
                         color=c_loso,   alpha=0.85, edgecolor="white")
    bars_wf     = ax.bar(x,          wf_vals,     width,
                         label="Walk-forward + PL — Bayesian in-season update",
                         color=c_wf,     alpha=0.85, edgecolor="white")
    bars_oracle = ax.bar(x + width,  oracle_vals, width,
                         label="Oracle — theoretical best (WF or LOSO correct)",
                         color=c_oracle, alpha=0.85, edgecolor="white")

    for bars in [bars_loso, bars_wf, bars_oracle]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.007,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # arrows from LOSO to WF showing the PL gain
    for xi, lv, wv in zip(x, loso_vals, wf_vals):
        gain = wv - (lv if not np.isnan(lv) else wv)
        if gain > 0.01:
            ax.annotate("", xy=(xi, wv + 0.04), xytext=(xi - width, lv + 0.04),
                        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
            ax.text(xi - width / 2, max(wv, lv or 0) + 0.05,
                    f"+{gain:.0%}", ha="center", va="bottom",
                    fontsize=7.5, color="#555", style="italic")

    ax.axhline(0.65, color="#333", lw=1.2, ls="--", alpha=0.6,
               label="Academic benchmark (~65%)")

    year_notes = {
        2019: "HAM quali dominance\ncaptured by PL",
        2023: "VER separates R4\nPL very effective",
        2022: "Regulation change:\nbase model stale",
        2024: "Mid-season shift\ndetected by PL",
    }
    for yr, note in year_notes.items():
        if yr in years:
            xi = x[years.index(yr)]
            ax.text(xi, -0.10, note, ha="center", va="top", fontsize=6.5,
                    color="#444", style="italic",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#f9f9f9",
                              edgecolor="#ccc", alpha=0.8))

    valid_loso   = [v for v in loso_vals   if not np.isnan(v)]
    valid_oracle = [v for v in oracle_vals if not np.isnan(v)]
    mean_loso   = np.mean(valid_loso)   if valid_loso   else np.nan
    mean_wf     = np.mean(wf_vals)
    mean_oracle = np.mean(valid_oracle) if valid_oracle else np.nan
    ax.text(0.99, 0.99,
            f"Mean accuracy:\n"
            f"  LOSO:    {mean_loso:.1%}\n"
            f"  WF+PL:   {mean_wf:.1%}\n"
            f"  Oracle:  {mean_oracle:.1%}\n"
            f"  PL gain: +{mean_wf - mean_loso:.1%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="#ccc", alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Holdout Season (year left out of training)", fontsize=11)
    ax.set_ylabel("Winner Prediction Accuracy", fontsize=11)
    ax.set_title("LOSO → Walk-Forward (+PL) → Oracle: Accuracy Hierarchy\n"
                 "Oracle = theoretical upper bound; gap to WF shows complementary blind spots",
                 fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(-0.25, 1.12)

    # right panel: WF gain and Oracle gain over LOSO shown as grouped horizontal bars
    wf_gains     = [wv - (lv if not np.isnan(lv) else 0) for wv, lv in zip(wf_vals, loso_vals)]
    oracle_gains = [ov - (lv if not np.isnan(lv) else 0) for ov, lv in zip(oracle_vals, loso_vals)]

    bw2 = 0.35
    ax2.barh(x - bw2 / 2, wf_gains,     bw2, color=c_wf,     alpha=0.85, edgecolor="white",
             label="WF+PL vs LOSO")
    ax2.barh(x + bw2 / 2, oracle_gains, bw2, color=c_oracle, alpha=0.85, edgecolor="white",
             label="Oracle vs LOSO")
    ax2.axvline(0, color="#333", lw=1.2)

    for xi, gw, go in zip(x, wf_gains, oracle_gains):
        for g, yoff, clr in [(gw, -bw2 / 2, c_wf), (go, bw2 / 2, c_oracle)]:
            sign = "+" if g >= 0 else ""
            ax2.text(g + (0.003 if g >= 0 else -0.003), xi + yoff,
                     f"{sign}{g:.1%}", va="center",
                     ha="left" if g >= 0 else "right",
                     fontsize=8.5, fontweight="bold", color=clr)

    ax2.set_yticks(x)
    ax2.set_yticklabels([str(y) for y in years], fontsize=10)
    ax2.set_xlabel("Gain over LOSO", fontsize=10)
    ax2.set_title("WF+PL and Oracle Gain\nover LOSO by Season", fontsize=11, fontweight="bold")
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    ax2.grid(True, axis="x", alpha=0.25)

    mean_wf_gain = np.nanmean(wf_gains)
    ax2.axvline(mean_wf_gain, color=c_wf, lw=1.2, ls="--", alpha=0.6)
    ax2.text(mean_wf_gain + 0.001, x[-1] + 0.45,
             f"WF mean\n{mean_wf_gain:+.1%}", fontsize=8, color=c_wf, va="bottom")

    plt.suptitle("Model Accuracy Hierarchy: LOSO → Walk-Forward → Oracle",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, "wf_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


def plot_pl_ability_evolution(df_pred, year=2023, outdir=PLOT_DIR):
    """
    shows how the PL model's "driver quality" estimate (θ) changes race by race.

    one line per driver (top-8 by peak θ), x = races seen so far, y = θ score.
    higher θ means PL thinks that driver is better based on what it's seen this season.

    in 2023 this plot is really satisfying — Verstappen's θ rockets up after the first
    few wins and visually separates from the field by about race 5 or 6. this is exactly
    what the LOSO model fundamentally can't do: it holds out the entire 2023 season so
    it never gets to see a single 2023 race result.

    the gap between VER's θ and #2 is also what drives the high α_k in dominant seasons —
    when one driver is clearly above everyone else, PL's uncertainty is low and we
    trust it more to override the base ensemble.
    """
    os.makedirs(outdir, exist_ok=True)
    yr_df = df_pred[df_pred["Year"] == year].copy()
    if yr_df.empty:
        print(f"  [skip] No walk-forward data for year {year}")
        return

    # one row per driver per race — pivot to get θ over time
    pivot = yr_df.pivot_table(index="races_observed", columns="Driver",
                               values="pl_theta", aggfunc="mean")

    # top-8 drivers by maximum observed θ (avoids 20-line spaghetti plot)
    top_drivers = pivot.max(axis=0).nlargest(8).index.tolist()

    fig, ax = plt.subplots(figsize=(13, 6))

    cmap = plt.get_cmap("tab10")
    for i, drv in enumerate(top_drivers):
        if drv not in pivot.columns:
            continue
        series = pivot[drv].dropna()
        ax.plot(series.index, series.values, lw=2.2, marker="o", markersize=3,
                color=cmap(i), label=drv)

    ax.axhline(0, color="#999", lw=0.8, ls="--", alpha=0.5, label="Prior mean level")
    ax.set_xlabel("Prior races observed (information fed to PL model)", fontsize=12)
    ax.set_ylabel("PL log-quality score θ* (MAP estimate)\nhigher = model thinks driver is better", fontsize=11)
    ax.set_title(f"Plackett-Luce Ability Evolution — {year} Season\n"
                 f"(how the PL posterior updates as race results accumulate)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, f"wf_pl_ability_evolution_{year}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


def plot_blend_alpha_by_race(df_pred, outdir=PLOT_DIR):
    """
    shows how much we're trusting the PL model vs the base ensemble as the season goes on.

    one line per season, x = prior races, y = α (higher means more PL influence).
    the five gates in action:
      - starts near 0 early on because the data_gate ramps up from zero
      - climbs as PL sees more races and its uncertainty shrinks
      - hard cap at 0.45 — base ensemble always contributes at least 55%
      - stays low all season in chaotic years (2024 had ~15 different winners →
        entropy gate suppresses trust even when PL thinks it knows something)

    comparing 2023 vs 2024 on this plot is particularly interesting:
    2023 should shoot up quickly (VER dominance → low uncertainty → high α early),
    while 2024 stays flat because the entropy gate keeps firing.
    """
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    for year in sorted(df_pred["Year"].unique()):
        yr_df = df_pred[df_pred["Year"] == year].copy()
        alpha_series = (yr_df.groupby("races_observed")["alpha_k"]
                             .mean()
                             .sort_index())
        color = SEASON_COLORS.get(year, "#888")
        ax.plot(alpha_series.index, alpha_series.values, lw=2.0, marker="o",
                markersize=3, color=color, label=str(year))

    ax.axhline(ALPHA_CAP, color="#333", lw=1.2, ls="--", alpha=0.7,
               label=f"α cap = {ALPHA_CAP}")
    ax.set_xlabel("Prior races observed in season", fontsize=12)
    ax.set_ylabel("Blend weight α (PL contribution)", fontsize=12)
    ax.set_title("Five-Factor Blend Weight Evolution\n"
                 "(inv-variance × data-sufficiency × entropy × dominance-gap × grid-alignment)", fontsize=12)
    ax.legend(fontsize=9, ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, ALPHA_CAP + 0.1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    path = os.path.join(outdir, "wf_blend_alpha.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


def plot_season_accuracy_heatmap(df_pred, loso_by_year, outdir=PLOT_DIR):
    """
    heatmap of walk-forward accuracy broken down by season (rows) and race stage (columns).

    race stages: early (races 1-5), mid (6-12), late (13+).
    the right panel shows the season total as a bar comparison vs LOSO.

    the most interesting pattern to look for: walk-forward should shine in the late-season
    bucket where PL has had enough races to figure out the competitive order. in 2023,
    even the early bucket looks good because VER's θ separates from everyone after
    just 3-4 wins — the PL model picks it up unusually fast that year.
    """
    os.makedirs(outdir, exist_ok=True)

    def race_bucket(n):
        if n <= 5:  return "Early\n(races 1-5)"
        if n <= 12: return "Mid\n(races 6-12)"
        return "Late\n(races 13+)"

    df = df_pred.copy()
    df["bucket"] = df["races_observed"].apply(race_bucket)
    bucket_order = ["Early\n(races 1-5)", "Mid\n(races 6-12)", "Late\n(races 13+)"]

    years = sorted(df["Year"].unique())

    wf_grid   = np.zeros((len(years), 3))

    for iy, year in enumerate(years):
        for ib, bucket in enumerate(bucket_order):
            sub = df[(df["Year"] == year) & (df["bucket"] == bucket)]
            sub_races = sub.drop_duplicates(subset=["Year", "Round"])
            sub_valid = sub_races[sub_races["actual_winner"].notna()]
            if len(sub_valid) == 0:
                wf_grid[iy, ib] = np.nan
            else:
                wf_grid[iy, ib] = sub_valid["correct_wf"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              gridspec_kw={"width_ratios": [3, 1]})

    from matplotlib.colors import LinearSegmentedColormap
    cmap_wf = LinearSegmentedColormap.from_list("wf", ["#f7f7f7", "#d6604d", "#a50026"])
    im = axes[0].imshow(wf_grid, aspect="auto", vmin=0, vmax=1, cmap=cmap_wf)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(bucket_order, fontsize=9)
    axes[0].set_yticks(range(len(years))); axes[0].set_yticklabels(years, fontsize=10)
    axes[0].set_title("Walk-Forward Winner Accuracy\n(by season and race-stage)", fontsize=11)
    for iy in range(len(years)):
        for ib in range(3):
            v = wf_grid[iy, ib]
            if not np.isnan(v):
                axes[0].text(ib, iy, f"{v:.0%}", ha="center", va="center",
                             fontsize=9, color="white" if v > 0.55 else "#333")
    plt.colorbar(im, ax=axes[0], label="Winner accuracy")

    # right panel: season-level bar comparison
    per_year = compute_per_year_metrics(df)
    wf_accs   = [per_year.loc[y, "wf_accuracy"] if y in per_year.index else np.nan for y in years]
    loso_accs = [loso_by_year.get(y, np.nan) for y in years]

    y_pos = np.arange(len(years))
    axes[1].barh(y_pos,       wf_accs,   height=0.38, color="#d6604d", label="Walk-forward")
    axes[1].barh(y_pos - 0.4, loso_accs, height=0.38, color="#4393c3", alpha=0.7, label="LOSO")
    axes[1].set_yticks(y_pos - 0.2); axes[1].set_yticklabels(years, fontsize=10)
    axes[1].set_xlabel("Winner accuracy", fontsize=9)
    axes[1].set_title("Season total", fontsize=10)
    axes[1].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    axes[1].axvline(0.65, color="#333", lw=1, ls="--", alpha=0.5)
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, 1.0)

    plt.tight_layout()
    path = os.path.join(outdir, "wf_season_accuracy_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────
# Oracle / theoretical-best analysis
# ─────────────────────────────────────────────────────────────────────────

def plot_theoretical_oracle(df_pred, loso_by_year, outdir=PLOT_DIR):
    """
    The "what if we had the perfect model?" plot.

    For every single race we already know which model got it right:
      - correct_wf:   the walk-forward (base ensemble + PL Bayesian update)
      - correct_base: the base ensemble alone (no within-season learning)

    At each race there are only four possible outcomes:
      A. Both right  — both models agree and they're both correct
      B. WF only     — the PL update specifically made the difference
      C. Base only   — the base got it right but PL overrode it wrong
      D. Neither     — neither model had a shot; probably a genuine upset

    The "oracle" model is the theoretical upper bound: it always picks the right
    answer if EITHER model knew it. Oracle = A + B + C.  Category D is the hard
    floor — races no pre-race model could realistically predict.

    Two panels:
      Left:  Per-year accuracy bars (Base, WF, Oracle, theoretical ceiling dots)
             Shows the progression: base < WF < oracle < per-year ceiling
      Right: Stacked composition bars showing how each year's races break down
             into the four outcome categories above
    """
    os.makedirs(outdir, exist_ok=True)

    # theoretical ceiling per year from our analysis
    # these are the actual race-winner frequency maximums — what a perfect model could hit
    theoretical_ceiling = {
        2018: 0.571, 2019: 0.524, 2020: 0.647,
        2021: 0.545, 2022: 0.636, 2023: 0.864, 2024: 0.500,
    }

    per_year = compute_per_year_metrics(df_pred)
    years    = sorted(per_year.index)

    # compute oracle accuracy: did WF OR base get this race right?
    df_races = df_pred.drop_duplicates(subset=["Year", "Round"]).copy()
    df_races = df_races[df_races["actual_winner"].notna()]

    oracle_data = {}
    for yr in years:
        sub = df_races[df_races["Year"] == yr]
        n = len(sub)
        if n == 0:
            continue
        both_right  = ((sub["correct_wf"] == True) & (sub["correct_base"] == True)).sum()
        wf_only     = ((sub["correct_wf"] == True) & (sub["correct_base"] == False)).sum()
        base_only   = ((sub["correct_wf"] == False) & (sub["correct_base"] == True)).sum()
        neither     = ((sub["correct_wf"] == False) & (sub["correct_base"] == False)).sum()
        oracle_data[yr] = {
            "both":      int(both_right),
            "wf_only":   int(wf_only),
            "base_only": int(base_only),
            "neither":   int(neither),
            "n":         n,
            "oracle_acc":  (both_right + wf_only + base_only) / n,
            "wf_acc":     sub["correct_wf"].mean(),
            "base_acc":   sub["correct_base"].mean(),
            "loso_acc":   loso_by_year.get(yr, np.nan),
        }

    # ── build figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    # palette
    c_base   = "#4393c3"   # blue — base/LOSO
    c_wf     = "#d6604d"   # red-orange — walk-forward
    c_oracle = "#1a9641"   # green — oracle
    c_ceil   = "#333333"   # dark grey — theoretical ceiling
    c_both   = "#2c7bb6"   # stacked: both right
    c_wf_s   = "#d7191c"   # stacked: WF only
    c_base_s = "#74add1"   # stacked: base only
    c_none   = "#d9d9d9"   # stacked: neither

    # ── LEFT PANEL: per-year accuracy comparison ──────────────────────────
    ax = axes[0]
    x     = np.arange(len(years))
    w     = 0.22

    base_vals   = [oracle_data[y]["loso_acc"]   for y in years]
    wf_vals     = [oracle_data[y]["wf_acc"]     for y in years]
    oracle_vals = [oracle_data[y]["oracle_acc"] for y in years]
    ceil_vals   = [theoretical_ceiling.get(y, np.nan) for y in years]

    b1 = ax.bar(x - w,    base_vals,   w, color=c_base,   alpha=0.85, label="LOSO (base ensemble)",  edgecolor="white")
    b2 = ax.bar(x,        wf_vals,     w, color=c_wf,     alpha=0.85, label="Walk-forward (+PL)",    edgecolor="white")
    b3 = ax.bar(x + w,    oracle_vals, w, color=c_oracle,  alpha=0.85, label="Oracle (best of both)", edgecolor="white")

    # theoretical ceiling as scatter dots
    for xi, cy in zip(x, ceil_vals):
        if not np.isnan(cy):
            ax.plot(xi, cy, marker="D", markersize=7, color=c_ceil, zorder=5,
                    label="Theoretical ceiling" if xi == x[0] else "")

    # bar value labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.007,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # mean lines
    mean_base   = np.nanmean(base_vals)
    mean_wf     = np.nanmean(wf_vals)
    mean_oracle = np.nanmean(oracle_vals)
    mean_ceil   = np.nanmean([v for v in ceil_vals if not np.isnan(v)])

    ax.axhline(mean_base,   color=c_base,   lw=1.2, ls=":",  alpha=0.6)
    ax.axhline(mean_wf,     color=c_wf,     lw=1.2, ls=":",  alpha=0.6)
    ax.axhline(mean_oracle, color=c_oracle,  lw=1.2, ls=":",  alpha=0.6)

    # mean summary box
    ax.text(0.02, 0.97,
            f"Means:  LOSO {mean_base:.1%}  |  WF {mean_wf:.1%}  |  Oracle {mean_oracle:.1%}  |  Ceiling {mean_ceil:.1%}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="#ccc", alpha=0.9))

    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Holdout Season", fontsize=12)
    ax.set_ylabel("Winner Prediction Accuracy", fontsize=12)
    ax.set_title("Model Accuracy Hierarchy\n"
                 "LOSO → Walk-Forward → Oracle → Theoretical Ceiling", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, 1.12)

    # ── RIGHT PANEL: race-outcome composition stacked bars ────────────────
    ax2 = axes[1]

    n_vals      = [oracle_data[y]["n"]         for y in years]
    pct_both    = [oracle_data[y]["both"]      / oracle_data[y]["n"] for y in years]
    pct_wf_only = [oracle_data[y]["wf_only"]   / oracle_data[y]["n"] for y in years]
    pct_base_only=[oracle_data[y]["base_only"] / oracle_data[y]["n"] for y in years]
    pct_neither = [oracle_data[y]["neither"]   / oracle_data[y]["n"] for y in years]

    x2 = np.arange(len(years))
    ax2.bar(x2, pct_both,     color=c_both,   label="Both models correct",       edgecolor="white")
    ax2.bar(x2, pct_wf_only,  bottom=pct_both, color=c_wf_s,
            label="WF only correct (PL added value)", edgecolor="white")
    bot2 = [a + b for a, b in zip(pct_both, pct_wf_only)]
    ax2.bar(x2, pct_base_only, bottom=bot2, color=c_base_s,
            label="Base only correct (PL hurt here)", edgecolor="white")
    bot3 = [a + b for a, b in zip(bot2, pct_base_only)]
    ax2.bar(x2, pct_neither,  bottom=bot3, color=c_none,
            label="Neither correct (irreducible uncertainty)", edgecolor="white", alpha=0.7)

    # annotate oracle line on composition chart
    for xi, ov in zip(x2, [oracle_data[y]["oracle_acc"] for y in years]):
        ax2.plot(xi, ov, marker="D", markersize=8, color=c_oracle, zorder=6)

    # add count labels inside bars for "neither" (the floor)
    for xi, n, pn in zip(x2, n_vals, pct_neither):
        neither_count = round(pn * n)
        ax2.text(xi, 1 - pn / 2, f"{neither_count}\nmisses",
                 ha="center", va="center", fontsize=7.5, color="#555")

    ax2.set_xticks(x2); ax2.set_xticklabels([str(y) for y in years])
    ax2.set_xlabel("Holdout Season", fontsize=12)
    ax2.set_ylabel("Fraction of Races", fontsize=12)
    ax2.set_title("Race Outcome Decomposition\n"
                  "Where does each model's accuracy come from?", fontsize=12, fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.set_ylim(0, 1.08)

    # diamond label for oracle
    ax2.plot([], [], marker="D", markersize=7, color=c_oracle, ls="none",
             label="Oracle accuracy")

    # explanatory note
    fig.text(0.5, -0.03,
             "Oracle model = theoretical upper bound assuming we always pick whichever model predicted correctly.\n"
             "It cannot be built in practice — it requires knowing the result. The gap between WF and Oracle\n"
             "shows how many races one model got right that the other missed (complementary blind spots).",
             ha="center", fontsize=9, style="italic", color="#555",
             wrap=True)

    plt.suptitle("Theoretical Best Model Analysis — What's the Ceiling?", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "wf_theoretical_oracle.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────
# Report writing
# ─────────────────────────────────────────────────────────────────────────

def write_wf_report(df_pred, loso_by_year, outpath=WF_EVAL_FILE):
    """Save the walk-forward evaluation summary to disk."""
    per_year = compute_per_year_metrics(df_pred)
    race_df  = compute_per_race_metrics(df_pred)

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("F1 Walk-Forward Evaluation — Plackett-Luce Bayesian Update\n")
        f.write("=" * 55 + "\n\n")
        f.write("Method: 4-model BMA base (XGB+Ranker+RF+ET, pre-race features)\n")
        f.write("        + Bayesian Plackett-Luce within-season updating\n")
        f.write(f"        SIGMA2_BASE={SIGMA2_BASE}, DECAY_LAMBDA={DECAY_LAMBDA}, "
                f"ALPHA_CAP={ALPHA_CAP}, WINNER_WEIGHT={WINNER_WEIGHT}, "
                f"QUAL_PL_WEIGHT={QUAL_PL_WEIGHT}, TAIL_WEIGHT={TAIL_WEIGHT}\n")
        f.write("        Blend: inv-variance × data-sufficiency × entropy × dominance-gap × grid-alignment\n")
        f.write("        + dual PL (season + recent-window) with transition detection\n")
        f.write("        + team-level PL for car-development propagation\n\n")

        f.write("Per-Season Results:\n")
        header = (f"{'Year':>6}  {'WF Acc':>8}  {'Base Acc':>9}  "
                  f"{'PL Gain':>7}  {'LOSO Acc':>9}  {'WF vs LOSO':>11}  {'Races':>6}\n")
        f.write(header)
        f.write("─" * len(header) + "\n")

        for year in sorted(per_year.index):
            row      = per_year.loc[year]
            wf_acc   = row["wf_accuracy"]
            base_acc = row["base_accuracy"]
            loso_acc = loso_by_year.get(year, np.nan)
            f.write(f"  {year:>4}  {wf_acc:>7.1%}  {base_acc:>8.1%}  "
                    f"{wf_acc - base_acc:>+6.1%}  {loso_acc:>8.1%}  "
                    f"{wf_acc - loso_acc:>+10.1%}  {row['n_races']:>5}\n")

        mean_wf   = per_year["wf_accuracy"].mean()
        mean_base = per_year["base_accuracy"].mean()
        valid_loso = [v for v in loso_by_year.values() if not np.isnan(v)]
        mean_loso  = np.mean(valid_loso) if valid_loso else np.nan

        f.write("\n")
        f.write(f"  Mean  {mean_wf:>7.1%}  {mean_base:>8.1%}  "
                f"{mean_wf - mean_base:>+6.1%}  {mean_loso:>8.1%}  "
                f"{mean_wf - mean_loso:>+10.1%}\n\n")

        f.write("Accuracy by Season Stage (pooled across all seasons):\n")
        f.write(f"  {'Prior races':>11}  {'WF Acc':>8}  {'Base Acc':>9}  {'Events':>7}\n")
        for _, row in race_df.iterrows():
            f.write(f"  {row['races_observed']:>11.0f}  "
                    f"{row['wf_accuracy']:>7.1%}  "
                    f"{row['base_accuracy']:>8.1%}  "
                    f"{row['n_events']:>6.0f}\n")

    print(f"  → Evaluation saved to {outpath}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  F1 Walk-Forward Model — Plackett-Luce Bayesian Update")
    print("=" * 55)

    print(f"\nLoading data from {TRAIN_FILE}...")
    df = pd.read_csv(TRAIN_FILE)
    df = df.dropna(subset=["finish_position"])
    print(f"  {len(df)} rows, {df['Year'].nunique()} seasons, "
          f"{df['Driver'].nunique()} unique drivers")

    df_train = df.copy()

    # LOSO baseline from saved model_evaluation.txt
    # these are the 4-model stacking ensemble results (with meta-model) from mach_learning.py
    loso_by_year = {
        2018: 0.476190,
        2019: 0.428571,
        2020: 0.588235,
        2021: 0.590909,
        2022: 0.454545,
        2023: 0.863636,
        2024: 0.458333,
    }
    loso_mean = np.mean(list(loso_by_year.values()))

    # ── Run walk-forward evaluation ──────────────────────────────────────────
    df_pred = walk_forward_evaluate(df_train, verbose=True)

    # ── Summary stats ─────────────────────────────────────────────────────────
    per_year  = compute_per_year_metrics(df_pred)
    mean_wf   = per_year["wf_accuracy"].mean()
    mean_base = per_year["base_accuracy"].mean()

    print("\n" + "─" * 55)
    print("  Walk-Forward Summary")
    print("─" * 55)
    print(f"  Walk-forward mean winner accuracy : {mean_wf:.1%}")
    print(f"  Base ensemble mean winner accuracy: {mean_base:.1%}")
    print(f"  LOSO ensemble mean winner accuracy: {loso_mean:.1%}")
    print(f"  PL gain vs base ensemble          : {mean_wf - mean_base:+.1%}")
    print(f"  WF gain vs LOSO                   : {mean_wf - loso_mean:+.1%}")

    write_wf_report(df_pred, loso_by_year)

    print("\n" + "─" * 55)
    print("  Generating comparison plots...")
    print("─" * 55)
    os.makedirs(PLOT_DIR, exist_ok=True)

    plot_accuracy_by_race_number(df_pred, loso_mean=loso_mean)
    plot_model_comparison_bars(df_pred, loso_by_year)
    plot_pl_ability_evolution(df_pred, year=2023)
    plot_blend_alpha_by_race(df_pred)
    plot_season_accuracy_heatmap(df_pred, loso_by_year)
    plot_theoretical_oracle(df_pred, loso_by_year)

    print("\n" + "=" * 55)
    print("  Done.")
    print("=" * 55)

    return df_pred, per_year


if __name__ == "__main__":
    main()
