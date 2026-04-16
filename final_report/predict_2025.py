# Load the trained models, run predictions on every 2025 race, and then check
# how well we did against the actual results. The idea is to simulate what you'd
# actually do in the real world - on Thursday before a race you have FP/quali data,
# you make your predictions, and then on Sunday night you find out how wrong you were.
#
# We had a lot of trouble with this file early on. The first version was loading
# race results BEFORE making predictions to "help" the feature builder, which was
# obviously cheating - the model had seen the answers before the exam. Took us
# an embarrassingly long time to notice the validation accuracy was suspiciously good.
# Fixed by making a strict rule: race results only get loaded AFTER predictions are made,
# and only for the purpose of checking accuracy.
#
# Another issue: we originally re-fit the imputer on 2025 data here, which sounds
# reasonable but is actually wrong. If the median of a feature shifts between 2024 and
# 2025 (e.g. a new team joins and the whole field distribution shifts), re-fitting
# means the training and prediction features are on different scales and the model
# gets confused. The fix was to save the imputer from training and reuse it here.
# That's the fit_imputer parameter in load_and_prepare in mach_learning.py.
#
# Run order (from project root):
#   python cache_fastf1_data.py
#   python final_report/feature_engineering.py
#   python final_report/mach_learning.py
#   python final_report/predict_2025.py
#
# Outputs go to data_organized/ and final_report/plots/:
#   2025_predictions.csv      - one row per driver per race, with predicted + actual pos
#   2025_race_metrics.csv     - winner accuracy, podium overlap, Jaccard, MAE per race
#   season_accuracy_2025.png  - how accuracy varies race by race over the season
#   feature_importances.png   - which features the models care about most
#   predicted_vs_actual.png   - scatter of predicted rank vs actual finishing pos
#   top10_heatmap.png         - how often each driver was placed in each position
#
# FastF1 docs: https://docs.fastf1.dev/
# joblib model serialisation: https://joblib.readthedocs.io/
# matplotlib reference: https://matplotlib.org/stable/api/index.html
# sklearn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

import os
import sys
import logging
import warnings

import fastf1
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend - needed for running without a display
                       # learned this the hard way when the script crashed on a headless server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so we can import from final_report/

# import from mach_learning - everything lives there so we don't duplicate code across files
# having it centralised also means if we tweak ensemble_predict it affects both predict scripts
from mach_learning import (
    FEATURE_COLS,              # the 44-column list - must match what feature_engineering.py wrote
    ensemble_predict,          # the three-model blend: xgb regressor + ranker + RF
    winner_accuracy,           # did we call P1 right?
    podium_accuracy,           # how much of the top-3 did we get right?
    top10_set_accuracy,        # Jaccard similarity of our top-10 vs the real top-10
    average_position_mae,      # mean absolute error on predicted rank vs actual position
    rank_predictions_per_race, # turns raw model scores into integer positions 1-N per race
    bayesian_adjust_win_proba, # Beta-Binomial Bayesian win prior blending
)

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)   # fastf1 logs constantly otherwise
logging.getLogger("urllib3").setLevel(logging.ERROR)  # so does urllib3

# ── paths ────────────────────────────────────────────────────────────────────

DATA_DIR  = "data_organized"
MODEL_DIR = os.path.join(DATA_DIR, "models")       # where mach_learning.py saves models
PLOT_DIR  = os.path.join("final_report", "plots")  # where we save output plots
CACHE_DIR = "discovery_raw_data"                   # FastF1 session cache

FEAT_FILE = os.path.join(DATA_DIR, "ml_2025_features.csv")  # built by feature_engineering.py
PRED_FILE = os.path.join(DATA_DIR, "2025_predictions.csv")  # output: our predictions
METR_FILE = os.path.join(DATA_DIR, "2025_race_metrics.csv") # output: per-race accuracy numbers

PREDICT_YEAR = 2025

# F1 timing-screen-inspired colour scheme for plots
# the dark background just looks a lot better for race data - tried white background once,
# didn't feel right for motorsport visualisations
F1_DARK  = "#15151E"
F1_RED   = "#E8002D"
F1_BLUE  = "#3A86FF"
F1_WHITE = "#FFFFFF"
F1_GREY  = "#38383F"
F1_AMBER = "#FFBE0B"
F1_GREEN = "#00C853"

fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.Cache.offline_mode(enabled=True)  # no network calls - only use what's already cached


# ── model loading ─────────────────────────────────────────────────────────────

def load_models():
    # load all the model files that mach_learning.py saved
    # the error message here is important - forgetting to run mach_learning.py first
    # is the most common reason this script fails and the default FileNotFoundError
    # message doesn't tell you what to do about it
    required = {
        "xgb":      os.path.join(MODEL_DIR, "xgb_model.joblib"),
        "ranker":   os.path.join(MODEL_DIR, "xgb_ranker.joblib"),
        "rf_win":   os.path.join(MODEL_DIR, "rf_win_model.joblib"),
        "rf_top10": os.path.join(MODEL_DIR, "rf_top10_model.joblib"),
        "imputer":  os.path.join(MODEL_DIR, "feature_imputer.joblib"),
    }

    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n[!] Model file not found: {path}\n"
                f"    Run  python final_report/mach_learning.py  first to train and save models."
            )

    xgb_model  = joblib.load(required["xgb"])
    xgb_ranker = joblib.load(required["ranker"])
    rf_win     = joblib.load(required["rf_win"])
    rf_top10   = joblib.load(required["rf_top10"])
    imputer    = joblib.load(required["imputer"])
    print(f"  ✓ Models loaded from {MODEL_DIR}/")

    # try to load the Bayesian Model Averaging weights that LOSO computed during training
    # if the file doesn't exist (older model dir, or someone deleted it) we fall back to
    # the fixed 40/40/20 weighting inside ensemble_predict
    # ref: https://en.wikipedia.org/wiki/Ensemble_learning#Bayesian_model_averaging
    bma_path    = os.path.join(MODEL_DIR, "bma_weights.joblib")
    bma_weights = None
    if os.path.exists(bma_path):
        bma_weights = joblib.load(bma_path)
        print(f"  ✓ Bayesian ensemble weights loaded: "
              f"xgb={bma_weights['w_xgb']:.3f}, "
              f"ranker={bma_weights['w_ranker']:.3f}, "
              f"rf={bma_weights['w_rf']:.3f}")
    else:
        # this just means LOSO didn't run or the file got lost - not a crash, just a degradation
        print("  [note] bma_weights.joblib not found - falling back to fixed 40/40/20 weights")
        print("         Retrain with the latest mach_learning.py to get the Bayesian weights")

    return xgb_model, xgb_ranker, rf_win, rf_top10, imputer, bma_weights


# ── feature loading ───────────────────────────────────────────────────────────

def load_2025_features(imputer):
    # load the 2025 pre-race features that feature_engineering.py wrote out
    # and apply the imputer that was fitted on 2022-2024 training data
    #
    # important: we're calling imputer.transform(), NOT fit_transform()
    # re-fitting here would let 2025 medians influence imputation, which breaks
    # the consistency assumption between training and prediction feature scales
    # this was a subtle bug we had for a while - predictions looked fine but
    # certain features (especially FP weather temps) were being imputed differently
    # ref: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
    if not os.path.exists(FEAT_FILE):
        raise FileNotFoundError(
            f"\n[!] 2025 feature file not found: {FEAT_FILE}\n"
            f"    Run  python final_report/feature_engineering.py  first."
        )

    df = pd.read_csv(FEAT_FILE)

    # reindex to FEATURE_COLS in case the CSV has extra or missing columns
    # missing columns get filled with NaN, which the imputer then handles
    # this lets us handle CSVs built with slightly different feature sets gracefully
    X_raw = df.reindex(columns=FEATURE_COLS).astype(float)
    X = pd.DataFrame(
        imputer.transform(X_raw),
        columns=FEATURE_COLS,
        index=df.index,
    )

    print(f"  ✓ 2025 features: {len(df)} rows, "
          f"{df['EventName'].nunique()} race weekends, "
          f"{df['Driver'].nunique()} drivers")
    return X, df


# ── actual results loading ────────────────────────────────────────────────────

def load_actual_2025_results():
    # load the real finishing positions from 2025 races that are in the cache
    # these only get used AFTER predictions are already made, strictly for validation
    # we never pass these to the models - that would completely defeat the purpose
    #
    # for races that aren't cached yet (future races), the try/except just skips them
    # so this function always returns whatever completed races we have
    # ref: https://docs.fastf1.dev/core.html#fastf1.core.SessionResults
    rows = []
    try:
        schedule = fastf1.get_event_schedule(PREDICT_YEAR, include_testing=False)
    except Exception as e:
        print(f"  [!] Cannot load 2025 schedule: {e}")
        return pd.DataFrame()

    for _, event in schedule.iterrows():
        rnd  = int(event["RoundNumber"])
        name = str(event["EventName"])
        try:
            session = fastf1.get_session(PREDICT_YEAR, rnd, "R")
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            for _, row in session.results.iterrows():
                abbr = str(row.get("Abbreviation", ""))
                pos  = row.get("Position", np.nan)
                if abbr and pd.notna(pos):
                    rows.append({
                        "Round":           rnd,
                        "EventName":       name,
                        "Driver":          abbr,
                        "actual_position": float(pos),
                    })
        except Exception:
            pass  # race not in cache yet - fine, we just won't validate that one

    df = pd.DataFrame(rows)
    n_completed = df["Round"].nunique() if not df.empty else 0
    print(f"  ✓ Actual 2025 results: {n_completed} completed races found in cache")
    return df


# ── per-race accuracy metrics ─────────────────────────────────────────────────

def compute_per_race_metrics(pred_df):
    # calculate accuracy on a race-by-race basis, not just season average
    # season-level averages hide a lot - you can have 80% winner accuracy but
    # if those correct picks all came from dominant races (Verstappen at Monza)
    # the model isn't actually good, it just got lucky on easy races
    # looking per-race lets you see where it struggles: wet races, new circuits,
    # safety car restarts, etc.
    if "actual_position" not in pred_df.columns:
        return pd.DataFrame()

    completed = pred_df.dropna(subset=["actual_position"])
    if completed.empty:
        return pd.DataFrame()

    rows = []
    for (year, rnd), g in completed.groupby(["Year", "Round"]):
        g    = g.reset_index(drop=True)
        meta = g[["Year", "Round", "EventName", "Driver", "Team"]]
        pred = g["PredictedPosition"].astype(float)
        true = g["actual_position"].astype(float)

        w_acc,  _, _      = winner_accuracy(meta, pred, true)
        pod_ex, pod_ol, _ = podium_accuracy(meta, pred, true)
        jaccard           = top10_set_accuracy(meta, pred, true)
        mae               = average_position_mae(meta, pred, true)

        rows.append({
            "Year":           year,
            "Round":          rnd,
            "EventName":      g.iloc[0]["EventName"],
            "winner_correct": w_acc,
            "podium_overlap": pod_ol,
            "top10_jaccard":  jaccard,
            "position_mae":   mae,
        })

    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_season_accuracy(metrics_df, out_dir):
    # two-panel chart: top panel is winner correct (green bar) or wrong (red bar),
    # bottom panel is Jaccard score (how much of the top-10 did we get right)
    # ref: https://en.wikipedia.org/wiki/Jaccard_index
    #
    # we went back and forth on what to plot - first tried a table, then a line chart,
    # settled on bar + line because it's easier to scan at a glance
    if metrics_df.empty:
        print("  [skip] no completed race data yet for accuracy plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor(F1_DARK)
    for ax in (ax1, ax2):
        ax.set_facecolor(F1_DARK)

    rounds = metrics_df["Round"].values
    labels = [str(n).replace(" Grand Prix", "").replace(" GP", "")
              for n in metrics_df["EventName"].values]

    # top panel: green = we got the winner, red = we didn't
    win_correct = metrics_df["winner_correct"].values.astype(float)
    ax1.bar(rounds, win_correct, color=[F1_GREEN if v else F1_RED for v in win_correct],
            edgecolor=F1_DARK, linewidth=0.5, width=0.6)
    ax1.axhline(win_correct.mean(), color=F1_AMBER, linewidth=1.5, linestyle="--",
                label=f"avg {win_correct.mean():.1%}")
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_ylabel("Winner Correct", color=F1_WHITE, fontsize=10)
    ax1.tick_params(colors=F1_WHITE)
    ax1.legend(fontsize=9, facecolor=F1_DARK, edgecolor=F1_GREY, labelcolor=F1_WHITE)
    ax1.set_title("2025 Season — Prediction Accuracy", color=F1_WHITE, fontsize=13, fontweight="bold")

    # bottom panel: Jaccard score 0-1 - how much of the top-10 did we correctly identify
    # a score of 0.5 means 5 of the 10 drivers we predicted matched the actual top-10
    jaccard = metrics_df["top10_jaccard"].values
    ax2.plot(rounds, jaccard, color=F1_BLUE, linewidth=2, marker="o", markersize=5)
    ax2.fill_between(rounds, jaccard, alpha=0.2, color=F1_BLUE)
    ax2.axhline(jaccard.mean(), color=F1_AMBER, linewidth=1.5, linestyle="--",
                label=f"avg {jaccard.mean():.3f}")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Top-10 Jaccard", color=F1_WHITE, fontsize=10)
    ax2.set_xlabel("Round", color=F1_WHITE, fontsize=10)
    ax2.tick_params(colors=F1_WHITE)
    ax2.legend(fontsize=9, facecolor=F1_DARK, edgecolor=F1_GREY, labelcolor=F1_WHITE)

    # angled race name labels on x-axis - rotation=45 prevents overlap for long GP names
    ax2.set_xticks(rounds)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color=F1_WHITE)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor(F1_GREY)

    plt.tight_layout()
    path = os.path.join(out_dir, "season_accuracy_2025.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")


def plot_feature_importances(model_dir, out_dir):
    # load the feature_importances.csv that mach_learning.py writes out after training
    # then plot the top 20 by average importance across XGBoost and RF
    # color-coded by which model thinks it's more important - tells you something about
    # what signal each model is actually picking up on
    # ref: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBModel.feature_importances_
    imp_path = os.path.join(DATA_DIR, "feature_importances.csv")
    if not os.path.exists(imp_path):
        print(f"  [skip] feature importances file not found at {imp_path}")
        print(f"         This gets created by mach_learning.py - check that it ran fully")
        return

    imp_df = pd.read_csv(imp_path, index_col=0).sort_values("avg_importance", ascending=True)
    imp_df = imp_df.tail(20)  # just the top 20, the rest are noise basically

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(F1_DARK)
    ax.set_facecolor(F1_DARK)

    # colour each bar based on whether XGB or RF thinks this feature matters more
    # when they disagree that's interesting - means they're exploiting different aspects
    # ref: https://scikit-learn.org/stable/modules/ensemble.html#feature-importance-evaluation
    colors = []
    for feat in imp_df.index:
        xgb_v = imp_df.loc[feat, "xgb_importance"]
        rf_v  = imp_df.loc[feat, "rf_importance"]
        if xgb_v > rf_v * 1.3:
            colors.append(F1_BLUE)   # XGBoost thinks it's more important
        elif rf_v > xgb_v * 1.3:
            colors.append(F1_RED)    # RF thinks it's more important
        else:
            colors.append(F1_AMBER)  # roughly agree

    ax.barh(range(len(imp_df)), imp_df["avg_importance"], color=colors,
            edgecolor=F1_DARK, linewidth=0.3)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df.index, fontsize=9, color=F1_WHITE)
    ax.set_xlabel("Average Feature Importance (XGBoost + RF)", color=F1_WHITE, fontsize=10)
    ax.tick_params(colors=F1_WHITE)
    ax.set_title("Feature Importances — Top 20", color=F1_WHITE, fontsize=12, fontweight="bold")

    patches = [
        mpatches.Patch(color=F1_BLUE,  label="XGBoost-dominant"),
        mpatches.Patch(color=F1_RED,   label="RF-dominant"),
        mpatches.Patch(color=F1_AMBER, label="Both agree"),
    ]
    ax.legend(handles=patches, fontsize=8, facecolor=F1_DARK,
              edgecolor=F1_GREY, labelcolor=F1_WHITE, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(F1_GREY)

    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importances.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")


def plot_predicted_vs_actual(pred_df, out_dir):
    # scatter plot: x = actual position, y = predicted position
    # perfect predictions sit exactly on the y=x diagonal
    # points far from the diagonal are the races where we got it badly wrong
    # color encodes magnitude of error: green = close, red = way off
    #
    # this is probably the most honest plot in the whole set - it shows
    # exactly where the model is good and where it falls apart
    # we noticed midfield drivers (P8-P15) have way more scatter than front runners
    # which makes sense - midfield is genuinely harder to predict
    if "actual_position" not in pred_df.columns:
        print("  [skip] no actual positions available for scatter plot yet")
        return

    completed = pred_df.dropna(subset=["actual_position"])
    if completed.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(F1_DARK)
    ax.set_facecolor(F1_DARK)

    pred_p = completed["PredictedPosition"].values
    true_p = completed["actual_position"].values

    # colour by absolute error magnitude - RdYlGn_r means red=bad, green=good
    error = np.abs(pred_p - true_p)
    sc = ax.scatter(true_p, pred_p, c=error, cmap="RdYlGn_r",
                    vmin=0, vmax=10, alpha=0.6, s=15, edgecolors="none")

    ax.plot([1, 20], [1, 20], color=F1_WHITE, linewidth=1.5,
            linestyle="--", alpha=0.6, label="Perfect prediction")

    # annotate overall MAE in the corner - this is the headline number
    from sklearn.metrics import mean_absolute_error as mae_fn
    overall_mae = mae_fn(true_p, pred_p)
    ax.text(0.05, 0.95, f"MAE = {overall_mae:.2f} positions",
            transform=ax.transAxes, color=F1_WHITE, fontsize=11,
            va="top", ha="left")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Position error (|pred - actual|)", color=F1_WHITE, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=F1_WHITE)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=F1_WHITE)

    ax.set_xlabel("Actual Position", color=F1_WHITE, fontsize=11)
    ax.set_ylabel("Predicted Position", color=F1_WHITE, fontsize=11)
    ax.set_title("Predicted vs Actual Position — 2025 Races",
                 color=F1_WHITE, fontsize=12, fontweight="bold")
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.5, 20.5)
    ax.tick_params(colors=F1_WHITE)
    ax.legend(fontsize=9, facecolor=F1_DARK, edgecolor=F1_GREY, labelcolor=F1_WHITE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(F1_GREY)

    plt.tight_layout()
    path = os.path.join(out_dir, "predicted_vs_actual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")


def plot_top10_heatmap(pred_df, out_dir):
    # shows how often each driver was placed in each predicted position 1-10
    # rows are the top 15 drivers by top-10 frequency, columns are positions
    # brighter cell = we predicted that driver there more often
    #
    # this is genuinely useful for spotting systematic biases in the model
    # if a certain driver always has a bright P3 cell it means we keep predicting
    # them 3rd even when they're not actually finishing there
    if "actual_position" not in pred_df.columns:
        print("  [skip] no actuals available for heatmap")
        return

    completed = pred_df.dropna(subset=["actual_position"])
    if completed.empty:
        return

    top_drivers = (
        completed[completed["PredictedPosition"] <= 10]
        .groupby("Driver")
        .size()
        .nlargest(15)
        .index.tolist()
    )

    # build the count matrix: [driver, position] = how many races predicted there
    matrix = np.zeros((len(top_drivers), 10))
    for i, drv in enumerate(top_drivers):
        drv_rows = completed[completed["Driver"] == drv]
        for pos in range(1, 11):
            matrix[i, pos - 1] = (drv_rows["PredictedPosition"] == pos).sum()

    fig, ax = plt.subplots(figsize=(12, max(6, len(top_drivers) * 0.5)))
    fig.patch.set_facecolor(F1_DARK)
    ax.set_facecolor(F1_DARK)

    # custom colormap: dark background -> red -> amber (feels very F1)
    cmap = LinearSegmentedColormap.from_list("f1_heat", [F1_DARK, F1_RED, F1_AMBER])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(10))
    ax.set_xticklabels([f"P{p}" for p in range(1, 11)], color=F1_WHITE, fontsize=10)
    ax.set_yticks(range(len(top_drivers)))
    ax.set_yticklabels(top_drivers, color=F1_WHITE, fontsize=10)
    ax.set_title("Predicted Position Heatmap — 2025 (top 15 drivers)",
                 color=F1_WHITE, fontsize=12, fontweight="bold")

    # annotate cells with the actual count where > 0 (0 cells stay blank, cleaner)
    for i in range(len(top_drivers)):
        for j in range(10):
            val = int(matrix[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=F1_WHITE, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Race count", color=F1_WHITE, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=F1_WHITE)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=F1_WHITE)

    plt.tight_layout()
    path = os.path.join(out_dir, "top10_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  F1 2025 — Predictions & Validation")
    print("=" * 60)

    # step 1: get models
    # if this fails the error message tells you exactly what to run first
    print("\n[1/5] Loading trained models...")
    xgb_model, xgb_ranker, rf_win, rf_top10, imputer, bma_weights = load_models()

    # step 2: get 2025 pre-race features
    # these were built by feature_engineering.py from FastF1 sessions
    print("\n[2/5] Loading 2025 pre-race features...")
    X_2025, df_2025 = load_2025_features(imputer)

    # sort by (Year, Round) - this matters for XGBRanker which needs consistent
    # group ordering, and for rank normalisation in ensemble_predict to work correctly
    # had a really subtle bug early on where predictions were slightly wrong because
    # the sort order was non-deterministic across runs depending on CSV read order
    sort_order = df_2025.sort_values(["Year", "Round"]).index
    X_2025_s   = X_2025.loc[sort_order].reset_index(drop=True)
    df_2025_s  = df_2025.loc[sort_order].reset_index(drop=True)

    # step 3: run the three models
    print("\n[3/5] Running ensemble predictions...")

    xgb_preds     = xgb_model.predict(X_2025_s)           # raw position estimates (continuous)
    ranker_scores = xgb_ranker.predict(X_2025_s)           # relevance scores - higher is better finishing
    rf_win_proba  = rf_win.predict_proba(X_2025_s)[:, 1]   # win probability 0 to 1
    rf_top10_pred = rf_top10.predict(X_2025_s)             # top-10 binary call (0 or 1)

    # ensemble_predict does several things in sequence:
    #   1. optionally adjusts rf_win_proba with the Bayesian Beta-Binomial circuit prior
    #      (drivers with a strong circuit history get a slight boost to their win probability)
    #      ref: https://en.wikipedia.org/wiki/Beta-binomial_distribution
    #   2. rank-normalises all three model outputs to [0,1] per race so they're comparable
    #   3. blends them using BMA weights if available, otherwise falls back to 40/40/20
    #   4. converts blended score to integer predicted position per race
    pred_df = ensemble_predict(
        xgb_preds,
        rf_win_proba,
        df_2025_s[["Year", "Round", "EventName", "Driver", "Team"]],
        ranker_raw_scores=ranker_scores,
        ensemble_weights=bma_weights,     # Bayesian weights from LOSO, or None for fixed
        X_features=X_2025_s,             # needed so Bayesian adjustment can read circuit history
        apply_bayesian_prior=True,
    )
    pred_df["rf_top10_pred"] = rf_top10_pred

    n_races = pred_df["Round"].nunique()
    print(f"  ✓ Predictions done: {n_races} races, {pred_df['Driver'].nunique()} drivers")

    # step 4: load actuals and measure how well we did
    # this is the "moment of truth" step - we see actual results only after predicting
    print("\n[4/5] Loading actual 2025 race results for validation...")
    actual_df = load_actual_2025_results()

    if not actual_df.empty:
        pred_df = pred_df.merge(
            actual_df[["Round", "Driver", "actual_position"]],
            on=["Round", "Driver"],
            how="left",
        )
        n_completed = pred_df.dropna(subset=["actual_position"])["Round"].nunique()
        print(f"  ✓ Merged with actuals: {n_completed} completed races")
    else:
        print("  (No actual results in cache - all future races or cache not populated)")

    pred_df.to_csv(PRED_FILE, index=False)
    print(f"  → Predictions saved: {PRED_FILE}")

    # compute per-race accuracy metrics
    metrics_df = compute_per_race_metrics(pred_df)
    if not metrics_df.empty:
        metrics_df.to_csv(METR_FILE, index=False)
        print(f"  → Metrics saved: {METR_FILE}")

        n_comp  = len(metrics_df)
        w_avg   = metrics_df["winner_correct"].mean()
        jac_avg = metrics_df["top10_jaccard"].mean()
        mae_avg = metrics_df["position_mae"].mean()
        pod_avg = metrics_df["podium_overlap"].mean()

        print(f"\n  Validation summary ({n_comp} completed races):")
        print(f"    Winner accuracy : {w_avg:.1%}")
        print(f"    Podium overlap  : {pod_avg:.2f}/3 drivers avg")
        print(f"    Top-10 Jaccard  : {jac_avg:.3f}")
        print(f"    Position MAE    : {mae_avg:.2f} places")

        # highlight the races where we nailed it and where we struggled the most
        # useful for the write-up and for understanding model limitations
        if len(metrics_df) >= 3:
            best_race  = metrics_df.loc[metrics_df["top10_jaccard"].idxmax()]
            worst_race = metrics_df.loc[metrics_df["top10_jaccard"].idxmin()]
            print(f"\n    Best predicted:  {best_race['EventName']} "
                  f"(Jaccard={best_race['top10_jaccard']:.3f})")
            print(f"    Worst predicted: {worst_race['EventName']} "
                  f"(Jaccard={worst_race['top10_jaccard']:.3f})")

    # step 5: plots
    print("\n[5/5] Generating plots...")
    plot_season_accuracy(metrics_df if not metrics_df.empty else pd.DataFrame(), PLOT_DIR)
    plot_feature_importances(MODEL_DIR, PLOT_DIR)
    plot_predicted_vs_actual(pred_df, PLOT_DIR)
    plot_top10_heatmap(pred_df, PLOT_DIR)

    # quick sanity check printout for the first few races
    # good to eyeball this before trusting the CSV output
    print("\n" + "─" * 60)
    print("  Sample Predictions — First 3 Races")
    print("─" * 60)
    for rnd in sorted(pred_df["Round"].unique())[:3]:
        race       = pred_df[pred_df["Round"] == rnd].sort_values("PredictedPosition")
        event_name = race.iloc[0]["EventName"]
        print(f"\n  Round {rnd:02d} — {event_name}")
        print(f"  {'Pos':>4}  {'Driver':<6}  {'Team':<30}  {'Win%':>5}  {'Actual':>6}")
        print(f"  {'────':>4}  {'──────':<6}  {'──────────────────────────────':<30}  {'─────':>5}  {'──────':>6}")
        for _, row in race.head(10).iterrows():
            actual_str = str(int(row["actual_position"])) if pd.notna(row.get("actual_position")) else "?"
            print(f"  {int(row['PredictedPosition']):>4}  "
                  f"{row['Driver']:<6}  "
                  f"{str(row['Team'])[:30]:<30}  "
                  f"{row['rf_win_proba']*100:>4.1f}%  "
                  f"{actual_str:>6}")

    print("\n" + "=" * 60)
    print("  Prediction complete.")
    print(f"  Outputs: {DATA_DIR}/")
    print(f"  Plots:   {PLOT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
