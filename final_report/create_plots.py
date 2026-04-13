"""
create_plots.py — Quarter 4: WDC Analysis Plots for Final Report
================================================================
Generates two research-question plots from the 2025 prediction output:

  (a) WDC Feature Profile Radar Chart       → plots/wdc_radar_2025.png
  (b) Championship Rank Confusion Matrix    → plots/championship_confusion_matrix_2025.png

These answer RQ3: "Can the model predict the WDC from in-season data?"

PREREQUISITES (run these first):
  1. python cache_fastf1_data.py
  2. python final_report/feature_engineering.py
  3. python final_report/mach_learning.py
  4. python final_report/predict_2025.py

INPUTS (from data_organized/):
  ml_2025_features.csv       per-driver pre-race features for all 2025 races
  2025_predictions.csv       ensemble predictions + actual positions (from predict_2025.py)

OUTPUTS (written to final_report/plots/):
  wdc_radar_2025.png
      Polar spider chart comparing the predicted WDC vs actual WDC across four
      normalised performance dimensions: Car Pace, Reliability, Driver Form,
      Circuit Experience.  Mirrors the sketch in sketches/rq3a_wdc_radar.png.

  championship_confusion_matrix_2025.png
      10×10 heatmap: rows = actual running championship rank (P1–P10),
      columns = predicted running championship rank (P1–P10), cell value =
      number of race rounds where the driver's running predicted rank matched j.
      Y-axis is labelled with the final top-10 driver abbreviations.
      Mirrors the sketch in sketches/rq3b_standings_confusion_matrix.png.

HOW TO RUN (from project root):
  python final_report/create_plots.py

REFERENCES:
  Spearman correlation: scipy.stats.spearmanr
  F1 points system:     https://www.formula1.com/en/championship/standings
  Radar chart:          matplotlib polar axes
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

DATA_DIR  = "data_organized"
PLOT_DIR  = os.path.join("final_report", "plots")

FEAT_FILE = os.path.join(DATA_DIR, "ml_2025_features.csv")
PRED_FILE = os.path.join(DATA_DIR, "2025_predictions.csv")

# Visual theme: matches predict_2025.py so all report plots are consistent
F1_DARK  = "#15151E"
F1_RED   = "#E8002D"
F1_BLUE  = "#3A86FF"
F1_WHITE = "#FFFFFF"
F1_GREY  = "#38383F"
F1_AMBER = "#FFBE0B"

# Official F1 championship points per finishing position
POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
              6:  8, 7:  6, 8:  4, 9:  2, 10:  1}


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────

def load_data():
    """
    Load the feature and prediction CSVs produced by the upstream pipeline.
    Raises a descriptive FileNotFoundError if either file is missing.
    """
    missing = [f for f in [FEAT_FILE, PRED_FILE] if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            "\n[!] Missing required files:\n"
            + "\n".join(f"    {f}" for f in missing)
            + "\n    Run feature_engineering.py → mach_learning.py → predict_2025.py first."
        )
    feat_df = pd.read_csv(FEAT_FILE)
    pred_df = pd.read_csv(PRED_FILE)
    return feat_df, pred_df


# ─────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────

def pos_to_points(pos):
    """Convert a finishing position (float or int) to F1 championship points."""
    try:
        return POINTS_MAP.get(int(pos), 0)
    except (ValueError, TypeError):
        return 0


def _points_to_ranks(pts_dict):
    """
    Convert {driver: total_points} to {driver: rank} where rank 1 = most points.
    Ties are broken alphabetically by driver code for deterministic results.
    """
    ordered = sorted(pts_dict, key=lambda d: (-pts_dict[d], d))
    return {drv: rank + 1 for rank, drv in enumerate(ordered)}


def compute_championship_standings(pred_df):
    """
    Compute final 2025 predicted and actual championship standings.

    Predicted standings: sum PredictedPosition→points for every completed race.
    Actual standings:    sum actual_position→points for every completed race
                         (only if 'actual_position' column is present and populated).

    Returns
    -------
    pred_stands   : DataFrame [Driver, predicted_points, predicted_rank]
    actual_stands : DataFrame [Driver, actual_points, actual_rank]  (may be empty)
    """
    # Predicted points
    pred_pts = (
        pred_df.groupby("Driver")["PredictedPosition"]
        .apply(lambda s: sum(pos_to_points(p) for p in s))
        .reset_index()
        .rename(columns={"PredictedPosition": "predicted_points"})
        .sort_values("predicted_points", ascending=False)
        .reset_index(drop=True)
    )
    pred_pts["predicted_rank"] = pred_pts.index + 1

    # Actual points (optional — only when actual results are available)
    actual_stands = pd.DataFrame()
    if "actual_position" in pred_df.columns:
        actual_rows = pred_df.dropna(subset=["actual_position"])
        if not actual_rows.empty:
            act_pts = (
                actual_rows.groupby("Driver")["actual_position"]
                .apply(lambda s: sum(pos_to_points(p) for p in s))
                .reset_index()
                .rename(columns={"actual_position": "actual_points"})
                .sort_values("actual_points", ascending=False)
                .reset_index(drop=True)
            )
            act_pts["actual_rank"] = act_pts.index + 1
            actual_stands = act_pts

    return pred_pts, actual_stands


# ─────────────────────────────────────────────────────────────────
# PLOT 1 — WDC FEATURE PROFILE RADAR CHART  (RQ3a)
# ─────────────────────────────────────────────────────────────────

def plot_wdc_radar(feat_df, pred_df, out_dir):
    """
    WDC Feature Profile Radar Chart (RQ3a).

    Four performance axes (all normalized to [0, 1], where 1 = field best):
      Car Pace         — inverse of average qualifying gap-to-pole (s)
      Reliability      — inverse of rolling DNF rate over last 5 races
      Driver Form      — inverse of rolling average finishing position (5-race)
      Circuit Exp.     — normalized circuit-specific experience count

    Two drivers are overlaid as filled polygon shapes:
      Red  — predicted 2025 WDC (driver with most predicted championship points)
      Blue — actual 2025 WDC, or runner-up if the same driver predicted and won

    This directly answers RQ3 by illustrating what dimensions the model
    "sees" in its top-ranked driver compared to the actual champion.

    Matches sketch: sketches/rq3a_wdc_radar.png
    """
    pred_stands, actual_stands = compute_championship_standings(pred_df)
    if pred_stands.empty:
        print("  [skip] No predictions for WDC radar chart")
        return

    # Compute per-driver season-average feature values
    drv_feats = feat_df.groupby("Driver").agg(
        quali_gap   =("quali_gap_to_pole",    "mean"),
        dnf_rate    =("rolling_dnf_rate_5",   "mean"),
        avg_finish  =("rolling_avg_finish_5",  "mean"),
        circuit_exp =("circuit_experience",    "mean"),
    ).reset_index()

    # Normalize each axis: 1 = best in field
    def field_normalize(series, invert):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        normed = (series - mn) / (mx - mn)
        return 1.0 - normed if invert else normed

    drv_feats["car_pace"]    = field_normalize(drv_feats["quali_gap"],   invert=True)
    drv_feats["reliability"] = field_normalize(drv_feats["dnf_rate"],    invert=True)
    drv_feats["form"]        = field_normalize(drv_feats["avg_finish"],  invert=True)
    drv_feats["experience"]  = field_normalize(drv_feats["circuit_exp"], invert=False)

    axes_labels = [
        "Car Pace\n(Qualifying)",
        "Reliability\n(Inverse DNFs)",
        "Driver Form\n(Rolling Finish)",
        "Circuit\nExperience",
    ]
    axes_cols = ["car_pace", "reliability", "form", "experience"]
    N = len(axes_cols)

    def driver_vals(abbr):
        row = drv_feats[drv_feats["Driver"] == abbr]
        return [float(row[c].iloc[0]) if not row.empty else 0.5 for c in axes_cols]

    # Identify the two drivers to compare
    pred_wdc    = pred_stands.iloc[0]["Driver"]
    actual_wdc  = actual_stands.iloc[0]["Driver"] if not actual_stands.empty else None

    if actual_wdc and actual_wdc != pred_wdc:
        compare_drv   = actual_wdc
        compare_label = f"Actual WDC: {compare_drv}"
    elif actual_wdc and len(actual_stands) >= 2:
        compare_drv   = actual_stands.iloc[1]["Driver"]
        compare_label = f"Actual Runner-up: {compare_drv}"
    elif len(pred_stands) >= 2:
        compare_drv   = pred_stands.iloc[1]["Driver"]
        compare_label = f"Predicted P2: {compare_drv}"
    else:
        print("  [skip] Not enough drivers to compare in WDC radar")
        return

    vals_pred    = driver_vals(pred_wdc)
    vals_compare = driver_vals(compare_drv)

    # Close polygons
    angles = [n / N * 2 * np.pi for n in range(N)] + [0.0]
    vals_pred    = vals_pred    + [vals_pred[0]]
    vals_compare = vals_compare + [vals_compare[0]]

    # ── Draw ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    fig.patch.set_facecolor(F1_DARK)
    ax.set_facecolor(F1_DARK)

    ax.plot(angles, vals_pred,    color=F1_RED,  linewidth=2.5, zorder=3)
    ax.fill(angles, vals_pred,    color=F1_RED,  alpha=0.25,    zorder=2)

    ax.plot(angles, vals_compare, color=F1_BLUE, linewidth=2.5, zorder=3)
    ax.fill(angles, vals_compare, color=F1_BLUE, alpha=0.25,    zorder=2)

    # Vertex annotations for predicted WDC
    for angle, val in zip(angles[:-1], vals_pred[:-1]):
        ax.annotate(
            f"{val:.2f}",
            xy=(angle, val),
            xytext=(angle, min(val + 0.10, 1.05)),
            fontsize=9, color=F1_RED, ha="center", va="center",
        )

    # Axis formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, color=F1_WHITE, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       color=F1_GREY, fontsize=7, zorder=0)
    ax.grid(color=F1_GREY, alpha=0.35, linewidth=0.8)
    ax.spines["polar"].set_color(F1_GREY)

    # Legend
    legend_handles = [
        mpatches.Patch(color=F1_RED,  label=f"Predicted WDC: {pred_wdc}"),
        mpatches.Patch(color=F1_BLUE, label=compare_label),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=10,
        facecolor=F1_DARK,
        edgecolor=F1_GREY,
        labelcolor=F1_WHITE,
    )

    fig.suptitle(
        "2025 WDC Feature Profile Comparison",
        fontsize=14, color=F1_WHITE, fontweight="bold", y=1.01,
    )
    ax.set_title(
        "Normalised performance (1.0 = field best)",
        fontsize=9, color=F1_GREY, pad=22,
    )

    path = os.path.join(out_dir, "wdc_radar_2025.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")


# ─────────────────────────────────────────────────────────────────
# PLOT 2 — CHAMPIONSHIP RANK CONFUSION MATRIX  (RQ3b)
# ─────────────────────────────────────────────────────────────────

def plot_championship_confusion_matrix(pred_df, out_dir):
    """
    Championship Running Rank Confusion Matrix (RQ3b).

    After each race round r:
      1. Compute each driver's cumulative actual points → actual_running_rank
      2. Compute each driver's cumulative predicted points → predicted_running_rank

    For every driver whose final-season actual rank is in the top 10, accumulate:
      matrix[actual_running_rank - 1, predicted_running_rank - 1] += 1

    A perfect model would produce a diagonal matrix.
    Off-diagonal entries reveal races where the model assigned a driver
    the wrong running championship rank — typically caused by strategy
    calls, safety cars, or incidents that cannot be predicted from
    pre-race session data.

    Y-axis labels = the final top-10 drivers (their abbreviations).
    Matches sketch: sketches/rq3b_standings_confusion_matrix.png
    """
    if "actual_position" not in pred_df.columns:
        print("  [skip] No actual results — championship confusion matrix requires actual_position")
        return

    df = pred_df.dropna(subset=["actual_position"]).copy()
    if df.empty:
        print("  [skip] actual_position column exists but has no values")
        return

    rounds = sorted(df["Round"].unique())
    n_rounds = len(rounds)

    # Final actual championship top-10 (used for Y-axis labels and row assignment)
    final_actual_pts = {
        drv: sum(pos_to_points(p) for p in grp["actual_position"])
        for drv, grp in df.groupby("Driver")
    }
    top10_drivers = sorted(final_actual_pts, key=final_actual_pts.get, reverse=True)[:10]

    # Accumulate confusion matrix
    matrix         = np.zeros((10, 10), dtype=int)
    cum_actual     = {}   # driver → cumulative actual points
    cum_predicted  = {}   # driver → cumulative predicted points

    for rnd in rounds:
        race = df[df["Round"] == rnd]

        for _, row in race.iterrows():
            drv = row["Driver"]
            cum_actual[drv]    = cum_actual.get(drv, 0)    + pos_to_points(row["actual_position"])
            cum_predicted[drv] = cum_predicted.get(drv, 0) + pos_to_points(row["PredictedPosition"])

        all_drivers     = set(cum_actual) | set(cum_predicted)
        actual_ranks    = _points_to_ranks({d: cum_actual.get(d, 0)    for d in all_drivers})
        predicted_ranks = _points_to_ranks({d: cum_predicted.get(d, 0) for d in all_drivers})

        # Only track drivers in the final top-10 actual championship
        for i, drv in enumerate(top10_drivers):
            actual_r    = actual_ranks.get(drv, 99)
            predicted_r = predicted_ranks.get(drv, 99)
            if actual_r <= 10 and predicted_r <= 10:
                matrix[actual_r - 1, predicted_r - 1] += 1

    # Accuracy = fraction of diagonal entries vs total accumulated observations
    total_obs = matrix.sum()
    diagonal  = int(np.trace(matrix))
    accuracy  = diagonal / total_obs if total_obs > 0 else 0.0

    # ── Draw ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(F1_DARK)
    ax.set_facecolor(F1_DARK)

    # Blues colormap — consistent with the sketch
    cmap = plt.cm.Blues
    im   = ax.imshow(matrix, cmap=cmap, aspect="auto",
                     vmin=0, vmax=matrix.max() or 1)

    # Cell annotations
    thresh = matrix.max() * 0.55
    for i in range(10):
        for j in range(10):
            val = matrix[i, j]
            if val > 0:
                cell_color = "black" if val > thresh else "white"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=9, color=cell_color, fontweight="bold")

    # Axes
    ax.set_yticks(range(10))
    ax.set_yticklabels(top10_drivers, fontsize=10, color=F1_WHITE)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"P{i+1}" for i in range(10)], fontsize=10, color=F1_WHITE)
    ax.tick_params(colors=F1_WHITE, length=0)

    ax.set_xlabel("Predicted Championship Rank", fontsize=11, color=F1_WHITE, labelpad=10)
    ax.set_ylabel("Actual Championship Rank",    fontsize=11, color=F1_WHITE, labelpad=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f"Number of races (out of {n_rounds})", fontsize=9, color=F1_WHITE)
    cbar.ax.yaxis.set_tick_params(color=F1_WHITE, labelcolor=F1_WHITE)

    # Title & annotation
    ax.set_title(
        f"Championship Running Rank — Predicted vs Actual\n"
        f"Model Accuracy: {accuracy:.1%}  (diagonal / total)",
        fontsize=12, color=F1_WHITE, pad=14, fontweight="bold",
    )
    ax.text(
        0.99, 0.01,
        "Off-diagonal: driver skill / strategy override car advantage",
        transform=ax.transAxes, fontsize=8, color=F1_GREY,
        ha="right", va="bottom",
    )

    plt.tight_layout()
    path = os.path.join(out_dir, "championship_confusion_matrix_2025.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=F1_DARK)
    plt.close(fig)
    print(f"  ✓ Plot saved: {path}")
    print(f"     Accuracy:  {accuracy:.1%}  "
          f"({diagonal} diagonal / {total_obs} total observations, {n_rounds} races)")


# ─────────────────────────────────────────────────────────────────
# RQ3 ANALYSIS PRINTOUT
# ─────────────────────────────────────────────────────────────────

def analyze_wdc_prediction(pred_df):
    """
    Print Spearman rank correlation of predicted vs actual final championship
    standings, then summarise the model's WDC prediction accuracy.

    Answers RQ3: "Can the model predict the World Drivers' Champion from
    in-season pre-race session data?"
    """
    pred_stands, actual_stands = compute_championship_standings(pred_df)

    if actual_stands.empty:
        print("  [skip] No actual results available for WDC analysis.")
        return

    merged = pred_stands.merge(actual_stands, on="Driver", how="inner")
    if len(merged) < 3:
        print("  [skip] Too few drivers with both predicted and actual standings.")
        return

    rho, pval = stats.spearmanr(merged["predicted_rank"], merged["actual_rank"])

    print("\n" + "─" * 58)
    print("  RQ3: Can the model predict the WDC from in-season data?")
    print("─" * 58)
    print(f"\n  Spearman ρ (predicted vs actual rank): {rho:+.3f}")
    print(f"  p-value:                               {pval:.4f}  "
          f"{'✓ significant (p < 0.05)' if pval < 0.05 else '✗ not significant'}")

    pred_wdc   = pred_stands.iloc[0]["Driver"]
    actual_wdc = actual_stands.iloc[0]["Driver"]
    wdc_match  = pred_wdc == actual_wdc

    print(f"\n  Predicted WDC: {pred_wdc}")
    print(f"  Actual WDC:    {actual_wdc}  "
          f"[{'✓ CORRECT' if wdc_match else '✗ INCORRECT'}]")

    print("\n  Predicted top-10  →  Actual rank:")
    for _, row in pred_stands.head(10).iterrows():
        actual_row = actual_stands[actual_stands["Driver"] == row["Driver"]]
        act_r      = int(actual_row.iloc[0]["actual_rank"]) if not actual_row.empty else "—"
        marker     = "●" if row["Driver"] in [pred_wdc, actual_wdc] else " "
        print(f"  {marker} Pred P{int(row['predicted_rank']):>2}  "
              f"{row['Driver']:<6}  "
              f"Pred pts: {int(row['predicted_points']):>4}  "
              f"Actual rank: {act_r}")

    # Interpretive summary
    print()
    if rho >= 0.70 and pval < 0.05:
        print("  Interpretation:")
        print("    Strong positive correlation: the model orders drivers in a sequence")
        print("    closely matching the actual championship. Pre-race session data (FP pace,")
        print("    qualifying gap, rolling form) is sufficient to approximate WDC outcome.")
        print("    Car performance dominates — consistent quali pace predicts points haul.")
    elif rho >= 0.40:
        print("  Interpretation:")
        print("    Moderate correlation: the model distinguishes top teams from midfield")
        print("    but cannot reliably separate team-mates with similar machinery.")
        print("    Gaps between predicted and actual rank are driven by safety cars,")
        print("    strategy calls, and mechanical failures — unpredictable from FP data.")
    else:
        print("  Interpretation:")
        print("    Weak correlation this season: championship outcomes were heavily")
        print("    influenced by in-race incidents and safety car periods that are")
        print("    not observable from pre-race session data.")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 60)
    print("  F1 2025 — Quarter 4 Final Report Plots")
    print("=" * 60)

    print("\n[1/4] Loading features and predictions...")
    feat_df, pred_df = load_data()
    n_races   = pred_df["Round"].nunique()
    n_drivers = pred_df["Driver"].nunique()
    print(f"  ✓ Features:    {len(feat_df)} rows")
    print(f"  ✓ Predictions: {n_races} races,  {n_drivers} drivers")

    has_actuals = (
        "actual_position" in pred_df.columns
        and pred_df["actual_position"].notna().any()
    )
    print(f"  ✓ Actual results available: {has_actuals}")

    print("\n[2/4] Generating WDC Feature Profile Radar Chart...")
    plot_wdc_radar(feat_df, pred_df, PLOT_DIR)

    print("\n[3/4] Generating Championship Rank Confusion Matrix...")
    plot_championship_confusion_matrix(pred_df, PLOT_DIR)

    print("\n[4/4] RQ3 Analysis:")
    analyze_wdc_prediction(pred_df)

    print("\n" + "=" * 60)
    print("  Quarter 4 plots complete.")
    print(f"  Output: {PLOT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
