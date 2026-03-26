"""
predict_2025.py — F1 2025 Season Race Prediction & Validation
=============================================================
Standalone script that loads the models trained by mach_learning.py,
applies them to 2025 pre-race session features, and validates predictions
against actual 2025 race results.

PHILOSOPHY:
  We treat every 2025 race as if it has NOT happened yet when making
  predictions. Only FP sessions, qualifying, and sprint sessions are used
  as input features. Race results are loaded ONLY afterwards for validation.

  This mirrors the real-world use case: before a race, you have FP pace,
  qualifying times, and (on sprint weekends) a sprint result.  After the
  race you find out how well the model did.

PREREQUISITES (run these first):
  1. python cache_fastf1_data.py          (cache FastF1 data including 2025)
  2. python final_report/feature_engineering.py   (builds ml_2025_features.csv)
  3. python final_report/mach_learning.py         (trains & saves models)

OUTPUTS (all written to data_organized/):
  2025_predictions.csv         per-race top-10 predictions with actual pos
  2025_race_metrics.csv        per-race accuracy metrics (winner, podium, Jaccard)

  Plots (written to final_report/plots/):
  season_accuracy_2025.png     winner accuracy & top-10 Jaccard per round
  feature_importances.png      horizontal bar chart of feature importances
  predicted_vs_actual.png      scatter of predicted rank vs actual position
  top10_heatmap.png            heatmap of who the model placed in each position

HOW TO RUN (from project root):
  python final_report/predict_2025.py

REFERENCES:
  FastF1 docs   : https://docs.fastf1.dev/
  joblib        : https://joblib.readthedocs.io/
  matplotlib    : https://matplotlib.org/
"""

import os
import sys
import logging
import warnings

import fastf1
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Allow imports from the final_report package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mach_learning import (
    FEATURE_COLS,
    ensemble_predict,
    winner_accuracy,
    podium_accuracy,
    top10_set_accuracy,
    average_position_mae,
    rank_predictions_per_race,
)

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

DATA_DIR  = "data_organized"
MODEL_DIR = os.path.join(DATA_DIR, "models")
PLOT_DIR  = os.path.join("final_report", "plots")
CACHE_DIR = "discovery_raw_data"

FEAT_FILE = os.path.join(DATA_DIR, "ml_2025_features.csv")
PRED_FILE = os.path.join(DATA_DIR, "2025_predictions.csv")
METR_FILE = os.path.join(DATA_DIR, "2025_race_metrics.csv")

PREDICT_YEAR = 2025

fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.Cache.offline_mode(enabled=True)


# ─────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────

def load_models():
    """
    Load the three serialised models and fitted imputer from MODEL_DIR.
    Raises a clear error if mach_learning.py hasn't been run yet.
    """
    required = {
        "xgb":      os.path.join(MODEL_DIR, "xgb_model.joblib"),
        "rf_win":   os.path.join(MODEL_DIR, "rf_win_model.joblib"),
        "rf_top10": os.path.join(MODEL_DIR, "rf_top10_model.joblib"),
        "imputer":  os.path.join(MODEL_DIR, "feature_imputer.joblib"),
    }
    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n[!] Model file not found: {path}\n"
                f"    Run  python final_report/mach_learning.py  first."
            )
    xgb_model  = joblib.load(required["xgb"])
    rf_win     = joblib.load(required["rf_win"])
    rf_top10   = joblib.load(required["rf_top10"])
    imputer    = joblib.load(required["imputer"])
    print(f"  ✓ Models loaded from {MODEL_DIR}/")
    return xgb_model, rf_win, rf_top10, imputer


# ─────────────────────────────────────────────────────────────────
# FEATURE LOADING
# ─────────────────────────────────────────────────────────────────

def load_2025_features(imputer):
    """
    Load the 2025 pre-race feature CSV and apply the training imputer.
    If the CSV doesn't exist, instruct the user to run feature_engineering.py.

    Returns (X_2025, df_2025) where X_2025 is the imputed feature matrix.
    """
    if not os.path.exists(FEAT_FILE):
        raise FileNotFoundError(
            f"\n[!] 2025 feature file not found: {FEAT_FILE}\n"
            f"    Run  python final_report/feature_engineering.py  first."
        )
    df = pd.read_csv(FEAT_FILE)

    # Use only the feature columns that the models were trained on.
    # Any Q2 columns missing from an older CSV get filled with NaN then imputed.
    X_raw = df.reindex(columns=FEATURE_COLS).astype(float)
    X     = pd.DataFrame(
        imputer.transform(X_raw),
        columns=FEATURE_COLS,
        index=df.index,
    )
    print(f"  ✓ 2025 features loaded: {len(df)} rows, "
          f"{df['EventName'].nunique()} race weekends, "
          f"{df['Driver'].nunique()} drivers")
    return X, df


# ─────────────────────────────────────────────────────────────────
# ACTUAL 2025 RESULTS (for validation)
# ─────────────────────────────────────────────────────────────────

def load_actual_2025_results():
    """
    Load actual finishing positions for all 2025 races from FastF1 cache.
    Only used for validation — not given to the model before prediction.

    Returns DataFrame with columns [Round, Driver, actual_position, EventName]
    for all completed races.  Returns empty DataFrame if no race data available.
    """
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
            session.load(laps=False, telemetry=False,
                         weather=False, messages=False)
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
            pass   # race not yet in cache (future race)

    df = pd.DataFrame(rows)
    n_races = df["Round"].nunique() if not df.empty else 0
    print(f"  ✓ Actual 2025 results: {n_races} completed races loaded")
    return df


# ─────────────────────────────────────────────────────────────────
# PER-RACE METRICS
# ─────────────────────────────────────────────────────────────────

def compute_per_race_metrics(pred_df):
    """
    For each race in pred_df (which must have both PredictedPosition and
    actual_position columns), compute winner accuracy, podium overlap,
    top-10 Jaccard, and position MAE.

    Returns a DataFrame with one row per race.
    """
    if "actual_position" not in pred_df.columns:
        print("  [!] No actual results available — skipping metrics.")
        return pd.DataFrame()

    completed = pred_df.dropna(subset=["actual_position"])
    if completed.empty:
        return pd.DataFrame()

    rows = []
    for (year, rnd), group in completed.groupby(["Year", "Round"]):
        g = group.reset_index(drop=True)
        meta = g[["Year", "Round", "EventName", "Driver", "Team"]]

        pred_pos = g["PredictedPosition"].astype(float)
        true_pos = g["actual_position"].astype(float)

        # Winner: driver ranked 1st by model vs actual P1
        pred_win = g.loc[g["PredictedPosition"].idxmin(), "Driver"]
        true_win_rows = g[g["actual_position"] == 1]
        true_win = true_win_rows.iloc[0]["Driver"] if not true_win_rows.empty else None
        winner_correct = int(pred_win == true_win)

        # Podium overlap (top-3)
        pred_p3 = set(g.nsmallest(3, "PredictedPosition")["Driver"])
        true_p3 = set(g[g["actual_position"] <= 3]["Driver"])
        pod_overlap = len(pred_p3 & true_p3)

        # Top-10 Jaccard
        pred_t10 = set(g.nsmallest(10, "PredictedPosition")["Driver"])
        true_t10 = set(g[g["actual_position"] <= 10]["Driver"])
        union = len(pred_t10 | true_t10)
        jaccard = len(pred_t10 & true_t10) / union if union > 0 else 0.0

        # Position MAE
        ranked_pred = g["PredictedPosition"].astype(float)
        mae = float(np.mean(np.abs(ranked_pred.values - true_pos.values)))

        rows.append({
            "Year":           year,
            "Round":          rnd,
            "EventName":      g.iloc[0]["EventName"],
            "winner_correct": winner_correct,
            "pred_winner":    pred_win,
            "true_winner":    true_win,
            "podium_overlap": pod_overlap,
            "top10_jaccard":  round(jaccard, 3),
            "position_mae":   round(mae, 2),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────

F1_RED   = "#E8002D"
F1_WHITE = "#FFFFFF"
F1_DARK  = "#15151E"
F1_GREY  = "#38383F"

plt.rcParams.update({
    "figure.facecolor": F1_DARK,
    "axes.facecolor":   F1_DARK,
    "axes.edgecolor":   F1_GREY,
    "axes.labelcolor":  F1_WHITE,
    "xtick.color":      F1_WHITE,
    "ytick.color":      F1_WHITE,
    "text.color":       F1_WHITE,
    "grid.color":       F1_GREY,
    "grid.alpha":       0.4,
    "font.family":      "monospace",
})


def plot_season_accuracy(metrics_df, out_dir):
    """
    Bar chart showing per-round winner accuracy (0/1) and top-10 Jaccard score.
    Green bar = correct winner prediction; red bar = wrong.

    This gives the clearest at-a-glance view of which races the model nailed
    and which it got wrong — more informative than a single season average.
    """
    if metrics_df.empty:
        print("  [skip] No metrics to plot (season_accuracy)")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("2025 F1 Season — Model Prediction Accuracy", fontsize=14,
                 color=F1_WHITE, fontweight="bold", y=0.98)

    rounds     = metrics_df["Round"].tolist()
    labels     = [f"R{r:02d}" for r in rounds]
    win_colors = [("#00D26A" if w else F1_RED)
                  for w in metrics_df["winner_correct"]]

    # ── Top panel: winner correct/incorrect ───────────────────────
    ax1 = axes[0]
    bars = ax1.bar(labels, metrics_df["winner_correct"], color=win_colors,
                   edgecolor=F1_DARK, linewidth=0.5)
    ax1.set_ylim(0, 1.3)
    ax1.set_ylabel("Winner\nCorrect", fontsize=9)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["✗", "✓"], fontsize=11)
    ax1.grid(axis="y")

    # Annotate with predicted winner name
    for bar, (_, row) in zip(bars, metrics_df.iterrows()):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.04,
                 row["pred_winner"],
                 ha="center", va="bottom", fontsize=7,
                 color=(F1_WHITE if row["winner_correct"] else "#FF6B6B"))

    correct_n = int(metrics_df["winner_correct"].sum())
    total_n   = len(metrics_df)
    ax1.set_title(f"Winner Prediction  ({correct_n}/{total_n} correct, "
                  f"{correct_n/total_n:.0%})",
                  fontsize=10, color=F1_WHITE, pad=6)

    green_patch = mpatches.Patch(color="#00D26A", label="Correct")
    red_patch   = mpatches.Patch(color=F1_RED,   label="Wrong")
    ax1.legend(handles=[green_patch, red_patch], fontsize=8,
               facecolor=F1_DARK, edgecolor=F1_GREY, loc="upper right")

    # ── Bottom panel: top-10 Jaccard and podium overlap ───────────
    ax2 = axes[1]
    x    = np.arange(len(rounds))
    w    = 0.4
    jac  = metrics_df["top10_jaccard"].tolist()
    pod  = (metrics_df["podium_overlap"] / 3).tolist()  # normalise to 0-1

    ax2.bar(x - w/2, jac, width=w, label="Top-10 Jaccard",
            color="#3A86FF", edgecolor=F1_DARK, linewidth=0.5)
    ax2.bar(x + w/2, pod, width=w, label="Podium overlap / 3",
            color="#FFBE0B", edgecolor=F1_DARK, linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Score (0–1)", fontsize=9)
    ax2.set_title(
        f"Top-10 Jaccard (avg {np.mean(jac):.2f})  ·  "
        f"Podium overlap (avg {np.mean(metrics_df['podium_overlap']):.1f}/3)",
        fontsize=10, color=F1_WHITE, pad=6,
    )
    ax2.grid(axis="y")
    ax2.legend(fontsize=8, facecolor=F1_DARK, edgecolor=F1_GREY, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "season_accuracy_2025.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")


def plot_feature_importances(feat_imp_path, out_dir):
    """
    Horizontal bar chart of XGBoost and RF feature importances side by side.
    Uses the feature_importances.csv saved by mach_learning.py.

    This directly answers RQ1 (does driver skill matter?) by showing whether
    session-pace features outrank grid position in importance.
    """
    if not os.path.exists(feat_imp_path):
        print("  [skip] feature_importances.csv not found")
        return

    df = pd.read_csv(feat_imp_path, index_col=0)

    # Friendly display names
    rename = {
        "grid_position":             "Grid position",
        "quali_gap_to_pole":         "Quali gap to pole",
        "quali_teammate_gap":        "Quali teammate gap",
        "rolling_avg_finish_5":      "Rolling avg finish (5)",
        "rolling_points_5":          "Rolling points (5)",
        "rolling_dnf_rate_5":        "Rolling DNF rate (5)",
        "team_rolling_avg_5":        "Team rolling avg (5)",
        "circuit_driver_avg_finish": "Circuit driver history",
        "circuit_team_avg_finish":   "Circuit team history",
        "fp2_pace_delta":            "FP2 best-lap delta",
        "fp1_pace_delta":            "FP1 best-lap delta",
        "fp3_pace_delta":            "FP3 best-lap delta",
        "fp2_long_run_delta":        "FP2 long-run pace",
        "fp2_s1_delta":              "FP2 sector-1 delta",
        "fp2_s2_delta":              "FP2 sector-2 delta",
        "fp2_s3_delta":              "FP2 sector-3 delta",
        "fp2_consistency":           "FP2 lap consistency",
        "fp_pace_progression":       "FP1→FP2 progression",
        "season_pts_pct":            "Season points %",
        "season_avg_finish":         "Season avg finish",
        "quali_track_temp":          "Qualifying track temp",
        "quali_wet":                 "Qualifying wet flag",
        "sprint_quali_pos":          "Sprint quali position",
        "sprint_pos":                "Sprint race position",
        "IsSprintWeekend":           "Sprint weekend flag",
    }
    df.index = [rename.get(i, i) for i in df.index]
    df = df.sort_values("avg_importance", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    fig.suptitle("Feature Importances — XGBoost & Random Forest",
                 fontsize=13, color=F1_WHITE, fontweight="bold")

    for ax, col, color, title in [
        (axes[0], "xgb_importance", "#3A86FF", "XGBoost (gain)"),
        (axes[1], "rf_importance",  "#FFBE0B", "Random Forest (impurity)"),
    ]:
        vals = df[col].fillna(0)
        bars = ax.barh(df.index, vals, color=color, edgecolor=F1_DARK,
                       linewidth=0.4)
        ax.set_xlabel("Importance", fontsize=9)
        ax.set_title(title, fontsize=10, color=F1_WHITE, pad=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x")
        # Annotate top feature
        top_idx = vals.idxmax()
        ax.axhline(y=df.index.tolist().index(top_idx),
                   color=F1_WHITE, linewidth=0.4, alpha=0.4, linestyle="--")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "feature_importances.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")


def plot_predicted_vs_actual(pred_df, out_dir):
    """
    Scatter plot: x = actual finishing position, y = predicted position.
    One point per (race, driver).  Points on the diagonal = perfect prediction.
    Colour encodes error magnitude (|pred − actual|).

    The reference model (mar-antaya) has no equivalent visualisation.
    This plot makes model accuracy interpretable to a non-technical reader.
    """
    df = pred_df.dropna(subset=["actual_position", "PredictedPosition"])
    if df.empty:
        print("  [skip] No actual results for predicted_vs_actual plot")
        return

    actual = df["actual_position"].astype(float)
    pred   = df["PredictedPosition"].astype(float)
    error  = np.abs(pred - actual)

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle("Predicted vs Actual Finishing Position — 2025",
                 fontsize=13, color=F1_WHITE, fontweight="bold")

    cmap  = LinearSegmentedColormap.from_list(
        "err", ["#00D26A", "#FFBE0B", F1_RED]
    )
    sc = ax.scatter(actual, pred, c=error, cmap=cmap, vmin=0, vmax=10,
                    s=28, alpha=0.75, edgecolors="none")

    # Perfect-prediction diagonal
    ax.plot([1, 20], [1, 20], color=F1_WHITE, linewidth=1,
            linestyle="--", alpha=0.5, label="Perfect prediction")
    # ±3 position bands
    ax.fill_between([1, 20], [1-3, 20-3], [1+3, 20+3],
                    color=F1_WHITE, alpha=0.05, label="±3 positions")

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("|Predicted − Actual|  (positions)", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=F1_WHITE)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=F1_WHITE)

    ax.set_xlabel("Actual Finishing Position", fontsize=10)
    ax.set_ylabel("Predicted Finishing Position", fontsize=10)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.5, 20.5)
    ax.set_xticks(range(1, 21))
    ax.set_yticks(range(1, 21))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, facecolor=F1_DARK, edgecolor=F1_GREY)

    within3 = (error <= 3).mean()
    mae_val  = error.mean()
    ax.set_title(
        f"MAE = {mae_val:.2f} positions  ·  "
        f"{within3:.0%} predicted within ±3 places",
        fontsize=10, color=F1_WHITE, pad=8,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "predicted_vs_actual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")


def plot_top10_heatmap(pred_df, out_dir, max_races=12):
    """
    Heatmap: rows = predicted finishing position (1–10),
             columns = races (up to max_races most recent).
    Cell = driver abbreviation placed there by the model.
    Green cell = that driver actually finished in the predicted position.

    Immediately shows which positions the model is confident about (usually
    P1–P3, where the field is most predictable) vs. where it struggles.
    """
    df = pred_df.copy()
    # Take the most recent races that have actual results
    if "actual_position" in df.columns:
        completed_rounds = df.dropna(subset=["actual_position"])["Round"].unique()
        rounds = sorted(completed_rounds)[-max_races:]
    else:
        rounds = sorted(df["Round"].unique())[-max_races:]

    df = df[df["Round"].isin(rounds)]
    if df.empty:
        print("  [skip] No data for top10_heatmap")
        return

    race_labels = (df.groupby("Round")["EventName"]
                     .first()
                     .reindex(rounds)
                     .apply(lambda n: n.replace(" Grand Prix", "").replace(" GP", "")))

    fig, ax = plt.subplots(figsize=(max(10, len(rounds) * 1.1), 8))
    fig.suptitle(f"Predicted Top-10 Grid — 2025 Season (last {len(rounds)} races)",
                 fontsize=12, color=F1_WHITE, fontweight="bold")

    cell_colors = np.full((10, len(rounds), 3), 0.18)   # dark grey default

    for col_idx, rnd in enumerate(rounds):
        race = df[df["Round"] == rnd].sort_values("PredictedPosition")
        top10 = race[race["PredictedPosition"] <= 10].head(10)

        for _, row in top10.iterrows():
            predicted_pos = int(row["PredictedPosition"]) - 1  # 0-indexed row
            if predicted_pos < 0 or predicted_pos >= 10:
                continue
            driver = row["Driver"]
            actual = row.get("actual_position", np.nan)

            # Colour: green if position correct, amber if within ±2, red otherwise
            if pd.notna(actual):
                err = abs(predicted_pos + 1 - int(actual))
                if err == 0:
                    colour = np.array([0, 0.82, 0.42])     # green
                elif err <= 2:
                    colour = np.array([1.0, 0.75, 0.04])   # amber
                else:
                    colour = np.array([0.91, 0.0, 0.18])   # red
            else:
                colour = np.array([0.22, 0.22, 0.27])      # grey (no result yet)

            cell_colors[predicted_pos, col_idx] = colour

            ax.text(col_idx + 0.5, predicted_pos + 0.5, driver,
                    ha="center", va="center", fontsize=8,
                    color=F1_WHITE, fontweight="bold")

    ax.imshow(cell_colors, aspect="auto", origin="upper",
              extent=[0, len(rounds), 10, 0])

    ax.set_xticks(np.arange(len(rounds)) + 0.5)
    ax.set_xticklabels([race_labels[r] for r in rounds],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(10) + 0.5)
    ax.set_yticklabels([f"P{i+1}" for i in range(10)], fontsize=9)
    ax.set_xlabel("Race", fontsize=9)
    ax.set_ylabel("Predicted Position", fontsize=9)

    # Legend
    patches = [
        mpatches.Patch(color=(0, 0.82, 0.42), label="Exact (±0)"),
        mpatches.Patch(color=(1.0, 0.75, 0.04), label="Close (±1-2)"),
        mpatches.Patch(color=(0.91, 0.0, 0.18), label="Wrong (>±2)"),
        mpatches.Patch(color=(0.22, 0.22, 0.27), label="No result yet"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              facecolor=F1_DARK, edgecolor=F1_GREY)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "top10_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")


# ─────────────────────────────────────────────────────────────────
# PRETTY-PRINT TABLE
# ─────────────────────────────────────────────────────────────────

def print_race_table(pred_df, round_num):
    """Print a formatted top-10 prediction table for one race."""
    race = pred_df[pred_df["Round"] == round_num].copy()
    race = race.sort_values("PredictedPosition")
    name = race.iloc[0]["EventName"] if not race.empty else f"Round {round_num}"

    print(f"\n  2025 R{round_num:02d}  {name}")
    print(f"  {'Pos':>3}  {'Driver':<6}  {'Team':<30}  {'Win%':>5}  {'Actual':>6}")
    print(f"  {'───':>3}  {'──────':<6}  {'──────────────────────────────':<30}  {'─────':>5}  {'──────':>6}")

    for _, row in race.head(10).iterrows():
        actual_str = (f"  P{int(row['actual_position'])}"
                      if "actual_position" in row and pd.notna(row["actual_position"])
                      else "      ")
        win_pct = f"{row['rf_win_proba']*100:>4.1f}%" if "rf_win_proba" in row else "   N/A"
        print(f"  {int(row['PredictedPosition']):>3}  "
              f"{row['Driver']:<6}  "
              f"{str(row['Team'])[:30]:<30}  "
              f"{win_pct}  "
              f"{actual_str}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 60)
    print("  F1 2025 Season — Race Predictions & Validation")
    print("=" * 60)

    # ── 1. Load models ─────────────────────────────────────────────
    print("\n[1/6] Loading trained models...")
    xgb_model, rf_win, rf_top10, imputer = load_models()

    # ── 2. Load 2025 pre-race features ────────────────────────────
    print("\n[2/6] Loading 2025 pre-race features...")
    X_2025, df_2025 = load_2025_features(imputer)

    # ── 3. Generate predictions ───────────────────────────────────
    print("\n[3/6] Generating predictions...")
    xgb_preds     = xgb_model.predict(X_2025)
    rf_win_proba  = rf_win.predict_proba(X_2025)[:, 1]
    rf_top10_pred = rf_top10.predict(X_2025)

    pred_df = ensemble_predict(
        xgb_preds, rf_win_proba,
        df_2025[["Year", "Round", "EventName", "Driver", "Team"]],
    )
    pred_df["rf_win_proba"]   = rf_win_proba
    pred_df["rf_top10_pred"]  = rf_top10_pred
    pred_df["xgb_raw_pred"]   = xgb_preds
    print(f"  ✓ Predictions generated for {pred_df['Round'].nunique()} races")

    # ── 4. Load actual results & validate ─────────────────────────
    print("\n[4/6] Loading actual 2025 race results...")
    actual_df = load_actual_2025_results()

    if not actual_df.empty:
        pred_df = pred_df.merge(
            actual_df[["Round", "Driver", "actual_position"]],
            on=["Round", "Driver"],
            how="left",
        )
        completed_rounds = pred_df.dropna(subset=["actual_position"])["Round"].nunique()
        print(f"  ✓ Matched results for {completed_rounds} completed races")
    else:
        pred_df["actual_position"] = np.nan

    # Save full prediction CSV
    pred_df.to_csv(PRED_FILE, index=False)
    print(f"  ✓ Predictions saved to {PRED_FILE}")

    # ── 5. Compute per-race metrics ────────────────────────────────
    print("\n[5/6] Computing per-race metrics...")
    metrics_df = compute_per_race_metrics(pred_df)

    if not metrics_df.empty:
        metrics_df.to_csv(METR_FILE, index=False)
        print(f"  ✓ Metrics saved to {METR_FILE}")

        total   = len(metrics_df)
        correct = int(metrics_df["winner_correct"].sum())
        avg_jac = metrics_df["top10_jaccard"].mean()
        avg_pod = metrics_df["podium_overlap"].mean()
        avg_mae = metrics_df["position_mae"].mean()

        print(f"\n  ── 2025 Season Summary ({'partial' if total < 24 else 'full'} season) ──")
        print(f"    Races evaluated   : {total}")
        print(f"    Winner accuracy   : {correct}/{total} = {correct/total:.1%}")
        print(f"    Podium overlap    : {avg_pod:.2f}/3 per race")
        print(f"    Top-10 Jaccard    : {avg_jac:.3f}")
        print(f"    Position MAE      : {avg_mae:.2f} places")

        # Per-race table
        print(f"\n  Per-race breakdown:")
        print(f"  {'Rnd':>3}  {'Circuit':<28}  {'Win':>3}  {'Pod':>3}  {'Jac':>5}  {'MAE':>5}  "
              f"{'Pred Winner':<6}  {'True Winner':<6}")
        print(f"  {'───':>3}  {'────────────────────────────':28}  {'───':>3}  {'───':>3}  "
              f"{'─────':>5}  {'─────':>5}  {'──────────':6}  {'──────────':6}")
        for _, row in metrics_df.iterrows():
            tick = "✓" if row["winner_correct"] else "✗"
            print(f"  {int(row['Round']):>3}  {str(row['EventName'])[:28]:<28}  "
                  f"{tick:>3}  {int(row['podium_overlap']):>3}  "
                  f"{row['top10_jaccard']:>5.3f}  {row['position_mae']:>5.2f}  "
                  f"{str(row['pred_winner']):<6}  {str(row['true_winner']):<6}")

    # ── 6. Generate plots ──────────────────────────────────────────
    print(f"\n[6/6] Generating plots → {PLOT_DIR}/")

    plot_season_accuracy(metrics_df, PLOT_DIR)

    feat_imp_path = os.path.join("data_organized", "feature_importances.csv")
    plot_feature_importances(feat_imp_path, PLOT_DIR)

    plot_predicted_vs_actual(pred_df, PLOT_DIR)
    plot_top10_heatmap(pred_df, PLOT_DIR)

    # ── Print top-10 tables for first 3 (and last 3) races ─────────
    print("\n" + "─" * 60)
    rounds = sorted(pred_df["Round"].unique())
    for rnd in rounds[:3]:
        print_race_table(pred_df, rnd)
    if len(rounds) > 3:
        print(f"\n  ... ({len(rounds) - 6} more races) ...")
        for rnd in rounds[-3:]:
            print_race_table(pred_df, rnd)

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)
    print(f"\n  Key outputs:")
    print(f"    {PRED_FILE}")
    print(f"    {METR_FILE}")
    print(f"    {PLOT_DIR}/season_accuracy_2025.png")
    print(f"    {PLOT_DIR}/feature_importances.png")
    print(f"    {PLOT_DIR}/predicted_vs_actual.png")
    print(f"    {PLOT_DIR}/top10_heatmap.png")


if __name__ == "__main__":
    main()
