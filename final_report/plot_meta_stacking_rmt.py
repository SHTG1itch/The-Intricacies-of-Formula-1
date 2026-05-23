"""
Visualise the impact of adding RMT features to the stacking meta-model.

Generates two figures:
  meta_model_feature_importances.png  — before/after feature importance comparison
                                        (9-feature baseline vs 11-feature RMT version)
  meta_model_loso_metrics.png         — LOSO metric shift from adding rmt_ens + rmt_disagr

Run from project root:
    python final_report/plot_meta_stacking_rmt.py
"""

import os
import warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

PLOT_DIR = os.path.join("final_report", "plots")
MODEL_DIR = os.path.join("data_organized", "models")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────────
BG     = "#FFFFFF"
DARK   = "#1a1a2e"
C_GRID = "#e0e0e0"
C_OLD  = "#888888"    # grey = old 9-feature meta-model
C_NEW  = "#4393c3"    # blue = new 11-feature meta-model
C_RMT  = "#2ca25f"    # green = the two new RMT features specifically
C_ACC  = "#d6604d"    # red-orange = improvement annotation

# ── feature importances ────────────────────────────────────────────────────────
# Old 9-feature meta-model importances (from mach_learning_run.log before RMT stacking)
OLD_FEAT_NAMES = ["xgb", "ranker", "rf", "et", "grid_pct",
                  "overtaking", "champ_rank", "reg_era", "is_street"]
OLD_FEAT_IMP   = [0.020, 0.146, 0.482, 0.215, 0.050,
                  0.006, 0.067, 0.007, 0.008]

# New 11-feature meta-model importances (loaded live from meta_model.joblib)
NEW_FEAT_NAMES = ["xgb", "ranker", "rf", "et", "grid_pct",
                  "overtaking", "champ_rank", "reg_era", "is_street",
                  "rmt_ens", "rmt_disagr"]

meta_path = os.path.join(MODEL_DIR, "meta_model.joblib")
if os.path.exists(meta_path):
    meta_model = joblib.load(meta_path)
    NEW_FEAT_IMP = list(meta_model.feature_importances_)
    print(f"  Loaded meta_model.joblib  (n_features={meta_model.n_features_in_})")
else:
    # fallback to the values we captured from the run log
    NEW_FEAT_IMP = [0.0046, 0.1683, 0.2622, 0.1294, 0.0736,
                    0.0047, 0.0999, 0.0075, 0.0088, 0.2342, 0.0066]
    print("  [!] meta_model.joblib not found - using hardcoded run-log values")

# ── LOSO metrics before / after ────────────────────────────────────────────────
# "before" = 9-feature meta-model (from mach_learning_run.log, previous session)
# "after"  = 11-feature meta-model (from model_evaluation.txt, this session)
METRIC_LABELS = ["Winner\nAccuracy", "Podium\nExact", "Top-10\nJaccard", "Position\nMAE"]
BEFORE_VALS   = [0.551, 0.228, 0.640, 3.52]
AFTER_VALS    = [0.551, 0.255, 0.600, 3.52]
# MAE is "lower is better" so we display it separately at a different scale


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Feature importance before vs after
# ════════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig.subplots_adjust(wspace=0.08)

for ax in axes:
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=DARK, labelsize=10)

# ── Left panel: old 9-feature model ──────────────────────────────────────────
ax_old = axes[0]
old_sorted = sorted(zip(OLD_FEAT_IMP, OLD_FEAT_NAMES), reverse=True)
old_imp, old_names = zip(*old_sorted)
y_old = np.arange(len(old_names))

ax_old.barh(y_old, old_imp, color=C_OLD, alpha=0.85,
            edgecolor="white", linewidth=0.4, zorder=3)
for y, v in zip(y_old, old_imp):
    ax_old.text(v + 0.005, y, f"{v:.3f}", va="center",
                fontsize=9, color=DARK)

ax_old.set_yticks(y_old)
ax_old.set_yticklabels(old_names, fontsize=11, color=DARK)
ax_old.set_xlim(0, 0.58)
ax_old.set_xlabel("Feature Importance (XGBoost gain)", fontsize=10, color=DARK)
ax_old.set_title("Before — 9-Feature Meta-Model\n(no RMT features)",
                 fontsize=12, fontweight="bold", color=DARK, pad=10)
ax_old.invert_yaxis()

# label total
ax_old.text(0.97, 0.02, "9 features", transform=ax_old.transAxes,
            ha="right", va="bottom", fontsize=10, color=C_OLD,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C_GRID, alpha=0.5))

# ── Right panel: new 11-feature model ────────────────────────────────────────
ax_new = axes[1]
new_sorted = sorted(zip(NEW_FEAT_IMP, NEW_FEAT_NAMES), reverse=True)
new_imp, new_names = zip(*new_sorted)
y_new = np.arange(len(new_names))

# colour RMT features differently
bar_colors = [C_RMT if n in ("rmt_ens", "rmt_disagr") else C_NEW
              for n in new_names]
ax_new.barh(y_new, new_imp, color=bar_colors, alpha=0.88,
            edgecolor="white", linewidth=0.4, zorder=3)
for y, v, n in zip(y_new, new_imp, new_names):
    col = C_RMT if n in ("rmt_ens", "rmt_disagr") else DARK
    ax_new.text(v + 0.005, y, f"{v:.3f}", va="center",
                fontsize=9, color=col, fontweight="bold" if n in ("rmt_ens", "rmt_disagr") else "normal")

ax_new.set_yticks(y_new)
ax_new.set_yticklabels(new_names, fontsize=11, color=DARK)
ax_new.set_xlim(0, 0.58)
ax_new.set_xlabel("Feature Importance (XGBoost gain)", fontsize=10, color=DARK)
ax_new.set_title("After — 11-Feature Meta-Model\n(+ RMT blend + model disagreement)",
                 fontsize=12, fontweight="bold", color=DARK, pad=10)
ax_new.invert_yaxis()

# annotate the two new features
rmt_idx = list(new_names).index("rmt_ens")
ax_new.annotate("NEW — RMT blend score\n2nd most important feature!",
                xy=(new_imp[rmt_idx], rmt_idx),
                xytext=(0.32, rmt_idx + 0.5),
                fontsize=8.5, color=C_RMT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_RMT, lw=1.5))

# label total
ax_new.text(0.97, 0.02, "11 features", transform=ax_new.transAxes,
            ha="right", va="bottom", fontsize=10, color=C_NEW,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C_GRID, alpha=0.5))

# legend
legend_patches = [
    mpatches.Patch(color=C_OLD,  label="Original 9 features"),
    mpatches.Patch(color=C_NEW,  label="Retained 9 features"),
    mpatches.Patch(color=C_RMT,  label="New RMT features (rmt_ens, rmt_disagr)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=9.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    "Stacking Meta-Model Feature Importances: Before vs After RMT Integration\n"
    "rmt_ens (Marchenko-Pastur portfolio blend) becomes the 2nd most important predictor",
    fontsize=13, fontweight="bold", color=DARK, y=1.02
)

out1 = os.path.join(PLOT_DIR, "meta_model_feature_importances.png")
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved: {out1}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2: LOSO metrics before vs after — what actually changed?
# ════════════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG)
fig2.subplots_adjust(wspace=0.38)

# we show the 3 meaningful metrics (skip MAE since it didn't change)
metric_labels_3 = ["Winner Accuracy", "Podium Exact", "Top-10 Jaccard"]
before_3 = [0.551, 0.228, 0.640]
after_3  = [0.551, 0.255, 0.600]
ylims_3  = [(0.45, 0.62), (0.15, 0.32), (0.52, 0.72)]
# whether higher is better (True) or lower is better (False)
higher_better = [True, True, True]

for ax, lbl, bef, aft, ylim, hb in zip(axes2, metric_labels_3, before_3, after_3, ylims_3, higher_better):
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.grid(axis="y", color=C_GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=DARK, labelsize=10)

    x = np.array([0, 1])
    w = 0.5
    bars = ax.bar(x, [bef, aft], width=w,
                  color=[C_OLD, C_NEW], alpha=0.88,
                  edgecolor="white", linewidth=0.5, zorder=3)

    # value labels on each bar
    for bar, val in zip(bars, [bef, aft]):
        ax.text(bar.get_x() + bar.get_width()/2, val + (ylim[1]-ylim[0])*0.015,
                f"{val:.1%}" if lbl != "Top-10 Jaccard" else f"{val:.3f}",
                ha="center", fontsize=11, fontweight="bold", color=DARK)

    # delta annotation
    delta = aft - bef
    if abs(delta) > 1e-4:
        sign = "+" if delta > 0 else ""
        col  = C_RMT if (delta > 0 and hb) else C_ACC
        y_arrow = max(bef, aft) + (ylim[1]-ylim[0])*0.07
        ax.annotate("", xy=(1, aft), xytext=(0, bef),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.0))
        ax.text(0.5, y_arrow,
                f"{sign}{delta:.1%}" if lbl != "Top-10 Jaccard" else f"{sign}{delta:.3f}",
                ha="center", fontsize=12, fontweight="bold", color=col)
    else:
        ax.text(0.5, max(bef, aft) + (ylim[1]-ylim[0])*0.07,
                "unchanged", ha="center", fontsize=10, color="#999999", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(["Before\n(9 features)", "After\n(11 features)"], fontsize=10, color=DARK)
    ax.set_ylabel(lbl, fontsize=11, color=DARK)
    ax.set_ylim(ylim)

    if lbl in ("Winner Accuracy", "Podium Exact"):
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

fig2.suptitle(
    "LOSO Holdout Metrics — 9-Feature vs 11-Feature Stacking Meta-Model (2018–2024)\n"
    "RMT features improve podium prediction; winner accuracy is at the dataset ceiling",
    fontsize=12, fontweight="bold", color=DARK, y=1.04
)

out2 = os.path.join(PLOT_DIR, "meta_model_loso_metrics.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print(f"Saved: {out2}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Per-year importance shift — how much did each feature change?
# ════════════════════════════════════════════════════════════════════════════════

# map old importances onto the same 9 shared features for delta comparison
old_dict = dict(zip(OLD_FEAT_NAMES, OLD_FEAT_IMP))
new_dict = dict(zip(NEW_FEAT_NAMES, NEW_FEAT_IMP))

shared_feats = OLD_FEAT_NAMES   # the 9 features present in both
deltas = {f: new_dict[f] - old_dict[f] for f in shared_feats}
# add the two new features (their "before" was 0)
deltas["rmt_ens"]   = new_dict["rmt_ens"]
deltas["rmt_disagr"] = new_dict["rmt_disagr"]

# sort by delta magnitude
sorted_deltas = sorted(deltas.items(), key=lambda x: x[1])
feat_d, val_d = zip(*sorted_deltas)
y_d = np.arange(len(feat_d))

fig3, ax3 = plt.subplots(figsize=(10, 6), facecolor=BG)
ax3.set_facecolor(BG)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_color(C_GRID)
ax3.spines["bottom"].set_color(C_GRID)
ax3.grid(axis="x", color=C_GRID, lw=0.6, zorder=0)
ax3.tick_params(colors=DARK, labelsize=10)

bar_colors_d = []
for f, v in zip(feat_d, val_d):
    if f in ("rmt_ens", "rmt_disagr"):
        bar_colors_d.append(C_RMT)   # new RMT features: green
    elif v > 0:
        bar_colors_d.append(C_NEW)   # gained importance: blue
    else:
        bar_colors_d.append(C_ACC)   # lost importance: red-orange (redistributed to RMT)

ax3.barh(y_d, val_d, color=bar_colors_d, alpha=0.88,
         edgecolor="white", linewidth=0.4, zorder=3)

for y, v, f in zip(y_d, val_d, feat_d):
    sign = "+" if v >= 0 else ""
    col = DARK
    ax3.text(v + (0.005 if v >= 0 else -0.005), y,
             f"{sign}{v:.3f}",
             va="center", ha="left" if v >= 0 else "right",
             fontsize=9, color=col)

ax3.axvline(0, color=DARK, lw=0.9, alpha=0.5)
ax3.set_yticks(y_d)
ax3.set_yticklabels(feat_d, fontsize=11, color=DARK)
ax3.set_xlabel("Change in Feature Importance (after − before)", fontsize=10, color=DARK)
ax3.set_title(
    "Feature Importance Shift: Where Did the Signal Go?\n"
    "RF's dominance is redistributed to rmt_ens — the RMT blend absorbs the ensemble consensus signal",
    fontsize=11, fontweight="bold", color=DARK, pad=10
)

legend_patches2 = [
    mpatches.Patch(color=C_RMT,  label="New RMT features (added from zero)"),
    mpatches.Patch(color=C_NEW,  label="Gained importance"),
    mpatches.Patch(color=C_ACC,  label="Lost importance (redistributed to RMT)"),
]
ax3.legend(handles=legend_patches2, fontsize=9, framealpha=0.9, loc="lower right")

out3 = os.path.join(PLOT_DIR, "meta_model_importance_shift.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig3)
print(f"Saved: {out3}")

print("\nAll meta-stacking plots saved to final_report/plots/")
print("  meta_model_feature_importances.png  — before/after side-by-side")
print("  meta_model_loso_metrics.png          — metric changes")
print("  meta_model_importance_shift.png      — where the signal redistributed")
