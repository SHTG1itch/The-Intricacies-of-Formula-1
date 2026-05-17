"""Walk-forward Plackett-Luce accuracy on 2025 season.

Trains the 4-model ensemble on 2018-2024 data (no 2025 leakage) then does
a race-by-race Plackett-Luce walk through the 2025 season. Outputs a comparison
plot against the Bayesian ensemble model from predict_2025.py.

Run from the project root:
    python final_report/wf_predict_2025.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walk_forward_model import walk_forward_evaluate, compute_per_year_metrics

DATA_DIR = "data_organized"
PLOT_DIR = os.path.join("final_report", "plots")

# same palette as the other WF plots so this one looks like it belongs
C_WF     = "#d6604d"   # red-orange: walk-forward PL model
C_BASE   = "#4393c3"   # blue: Bayesian static ensemble
C_REF    = "#888888"   # grey: dashed average lines
BG       = "#FFFFFF"
DARK     = "#1a1a2e"


def load_combined_data():
    train_path = os.path.join(DATA_DIR, "ml_training_features.csv")
    feat_2025  = os.path.join(DATA_DIR, "ml_2025_features.csv")

    df_train = pd.read_csv(train_path)
    df_2025  = pd.read_csv(feat_2025)

    # only include 2025 rows where we have the actual result
    df_2025 = df_2025.dropna(subset=["finish_position"]).copy()
    df_2025["Year"] = df_2025["Year"].astype(int)

    combined = pd.concat([df_train, df_2025], ignore_index=True)
    years = sorted(combined["Year"].unique())
    print(f"  Combined dataset: {len(combined)} rows, years {years}")
    return combined


def run_wf_2025(df_all):
    # smoke_year=2025 tells walk_forward_evaluate to only evaluate the 2025 holdout
    # training is done on all other years (2018-2024) - no 2025 leakage
    print("\n  Running walk-forward evaluation for 2025...")
    print("  (Training ensemble on 2018-2024, then walking through 2025 race by race)")
    df_pred = walk_forward_evaluate(df_all, smoke_year=2025, verbose=True)
    return df_pred


def load_bayesian_metrics():
    metrics_path = os.path.join(DATA_DIR, "2025_race_metrics.csv")
    df = pd.read_csv(metrics_path)
    df["Round"] = df["Round"].astype(int)
    return df.sort_values("Round").reset_index(drop=True)


def make_plot(df_wf_pred, df_bay_metrics):
    # per-race winner correctness for WF model
    wf_races = (df_wf_pred
                .drop_duplicates(subset=["Year", "Round"])
                .query("actual_winner.notna()", engine="python")
                .sort_values("Round")
                .reset_index(drop=True))

    wf_rounds   = wf_races["Round"].values
    wf_correct  = wf_races["correct_wf"].astype(int).values

    # Bayesian model results aligned to same rounds
    bay_rounds  = df_bay_metrics["Round"].values
    bay_correct = df_bay_metrics["winner_correct"].astype(int).values

    # running cumulative accuracy
    wf_cumulative  = np.cumsum(wf_correct)  / (np.arange(len(wf_correct)) + 1)
    bay_cumulative = np.cumsum(bay_correct) / (np.arange(len(bay_correct)) + 1)

    wf_mean  = wf_correct.mean()
    bay_mean = bay_correct.mean()

    fig, (ax_bar, ax_cum) = plt.subplots(2, 1, figsize=(14, 9),
                                          gridspec_kw={"height_ratios": [1.4, 1]})
    fig.patch.set_facecolor(BG)

    # ── TOP PANEL: per-race correctness bars ─────────────────────────────────
    n_races = max(len(wf_rounds), len(bay_rounds))
    all_rounds = sorted(set(wf_rounds) | set(bay_rounds))

    width = 0.38
    x     = np.array(all_rounds) - 1    # 0-indexed for spacing

    wf_dict  = dict(zip(wf_rounds,  wf_correct))
    bay_dict = dict(zip(bay_rounds, bay_correct))

    wf_vals  = [wf_dict.get(r,  np.nan) for r in all_rounds]
    bay_vals = [bay_dict.get(r, np.nan) for r in all_rounds]

    bars_wf  = ax_bar.bar(x - width / 2, wf_vals,  width, color=C_WF,   alpha=0.85,
                          label="Walk-Forward PL (WF)", edgecolor="white", zorder=3)
    bars_bay = ax_bar.bar(x + width / 2, bay_vals, width, color=C_BASE, alpha=0.85,
                          label="Bayesian Ensemble (static pre-race)", edgecolor="white", zorder=3)

    # colour-code bars by outcome: solid = correct, hatched = wrong
    for bar, val in zip(bars_wf,  wf_vals):
        if not np.isnan(val) and val == 0:
            bar.set_hatch("///")
            bar.set_edgecolor(C_WF)
            bar.set_alpha(0.35)
    for bar, val in zip(bars_bay, bay_vals):
        if not np.isnan(val) and val == 0:
            bar.set_hatch("///")
            bar.set_edgecolor(C_BASE)
            bar.set_alpha(0.35)

    ax_bar.set_xlim(-0.7, len(all_rounds) - 0.3)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"R{r}" for r in all_rounds], rotation=45, ha="right", fontsize=8)
    ax_bar.set_yticks([0, 1])
    ax_bar.set_yticklabels(["Wrong", "Correct"], fontsize=9)
    ax_bar.set_ylabel("Winner\nPrediction", fontsize=10)
    ax_bar.set_title(
        f"2025 Race Winner Accuracy  ·  Walk-Forward PL vs. Bayesian Ensemble\n"
        f"WF PL: {wf_mean:.1%} correct ({int(wf_correct.sum())}/{len(wf_correct)} races)  |  "
        f"Bayesian: {bay_mean:.1%} correct ({int(bay_correct.sum())}/{len(bay_correct)} races)",
        fontsize=12, fontweight="bold", color=DARK, pad=8,
    )
    ax_bar.legend(fontsize=9, loc="upper right", framealpha=0.92)
    ax_bar.grid(axis="y", alpha=0.25, zorder=0)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_facecolor(BG)

    # ── BOTTOM PANEL: cumulative running accuracy ─────────────────────────────
    ax_cum.plot(wf_rounds,  wf_cumulative,
                color=C_WF,   lw=2.5, marker="o", markersize=4, label="WF PL cumulative")
    ax_cum.plot(bay_rounds, bay_cumulative,
                color=C_BASE, lw=2.5, marker="s", markersize=4, ls="--",
                label="Bayesian cumulative")

    ax_cum.axhline(wf_mean,  color=C_WF,   lw=1.2, ls=":", alpha=0.7,
                   label=f"WF final avg {wf_mean:.1%}")
    ax_cum.axhline(bay_mean, color=C_BASE, lw=1.2, ls=":", alpha=0.7,
                   label=f"Bayesian final avg {bay_mean:.1%}")

    # annotate final accuracy at the right edge
    ax_cum.annotate(f"{wf_mean:.1%}", xy=(wf_rounds[-1], wf_cumulative[-1]),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=9, color=C_WF, fontweight="bold", va="center")
    ax_cum.annotate(f"{bay_mean:.1%}", xy=(bay_rounds[-1], bay_cumulative[-1]),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=9, color=C_BASE, fontweight="bold", va="center")

    ax_cum.set_xlim(0.5, len(all_rounds) + 2.5)
    ax_cum.set_xticks(all_rounds)
    ax_cum.set_xticklabels([f"R{r}" for r in all_rounds], rotation=45, ha="right", fontsize=8)
    ax_cum.set_ylabel("Cumulative\nAccuracy", fontsize=10)
    ax_cum.set_xlabel("Race Round (2025 season)", fontsize=10)
    ax_cum.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_cum.set_ylim(-0.05, 1.05)
    ax_cum.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax_cum.grid(alpha=0.25, zorder=0)
    ax_cum.spines["top"].set_visible(False)
    ax_cum.spines["right"].set_visible(False)
    ax_cum.set_facecolor(BG)

    plt.tight_layout(rect=[0, 0, 0.97, 1])
    out = os.path.join(PLOT_DIR, "wf_2025_accuracy_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n  → Plot saved: {out}")
    return out


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 58)
    print("  Walk-Forward PL  -  2025 Accuracy vs Bayesian Ensemble")
    print("=" * 58)

    print("\n[1/4] Loading training + 2025 data...")
    df_all = load_combined_data()

    print("\n[2/4] Running WF evaluation for 2025 (trains on 2018-2024)...")
    df_wf = run_wf_2025(df_all)

    # per-year metrics
    per_year = compute_per_year_metrics(df_wf)
    if 2025 in per_year.index:
        row = per_year.loc[2025]
        print(f"\n  WF 2025 results:")
        print(f"    Winner accuracy : {row['wf_accuracy']:.1%}  ({row['wf_correct']:.0f}/{row['n_races']:.0f} races)")
        print(f"    Base-only accy  : {row['base_accuracy']:.1%}")
        print(f"    PL gain         : {row['wf_accuracy'] - row['base_accuracy']:+.1%}")

    print("\n[3/4] Loading Bayesian ensemble 2025 metrics...")
    df_bay = load_bayesian_metrics()
    bay_mean = df_bay["winner_correct"].mean()
    print(f"  Bayesian 2025 winner accuracy: {bay_mean:.1%} ({int(df_bay['winner_correct'].sum())}/{len(df_bay)} races)")

    print("\n[4/4] Generating comparison plot...")
    make_plot(df_wf, df_bay)

    print("\n" + "=" * 58)
    print("  Done.")
    print("=" * 58)


if __name__ == "__main__":
    main()
