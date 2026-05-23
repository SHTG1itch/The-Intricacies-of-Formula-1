"""
Generate comparison plots showing accuracy improvements from:
  1. Marchenko-Pastur RMT driver-cluster denoising (entropy-gated)
  2. Early-season alpha gate fix (require n>=3 race results before PL activates)

Run from project root:
    python final_report/plot_rmt_accuracy_comparison.py
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

PLOT_DIR = os.path.join("final_report", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Hardcoded results from the three-scenario evaluation ────────────────────
# Scenario 1: Baseline (no RMT, original alpha gate)
# Scenario 2: + Marchenko-Pastur RMT driver-cluster denoising
# Scenario 3: + data_gate fix (require n>=3 observations before PL activates)

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

S1_WF   = [0.4762, 0.5238, 0.6471, 0.6364, 0.4545, 0.8636, 0.5417]
S1_BASE = [0.4762, 0.4286, 0.5882, 0.6364, 0.4091, 0.8636, 0.5000]

S2_WF   = [0.4762, 0.5238, 0.6471, 0.6364, 0.4545, 0.8636, 0.5417]
S2_BASE = [0.4762, 0.4762, 0.5882, 0.6364, 0.4091, 0.8636, 0.5000]

S3_WF   = [0.4762, 0.5238, 0.7059, 0.6364, 0.4545, 0.8636, 0.5417]
S3_BASE = [0.4762, 0.4762, 0.5882, 0.6364, 0.4091, 0.8636, 0.5000]

OVR_S1_WF,   OVR_S1_BASE  = 0.5919, 0.5574
OVR_S2_WF,   OVR_S2_BASE  = 0.5919, 0.5642
OVR_S3_WF,   OVR_S3_BASE  = 0.6003, 0.5642

EARLY_BASE = 0.5714   # base ensemble accuracy in races 1-5 (same across all scenarios)
EARLY_S1   = 0.5429   # WF accuracy races 1-5: original alpha gate (WF < base)
EARLY_S2   = 0.5429   # WF accuracy races 1-5: with RMT, still no gate fix
EARLY_S3   = 0.5714   # WF accuracy races 1-5: with gate fix (WF = base now)
LATE_S1    = 0.6053
LATE_S2    = 0.6053
LATE_S3    = 0.6053

# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#FFFFFF"
DARK      = "#1a1a2e"
C_BASE    = "#888888"   # grey for baseline
C_RMT     = "#4393c3"   # blue for +RMT
C_FIX     = "#2ca25f"   # green for +fix
C_ACCENT  = "#d6604d"   # red-orange for highlight deltas
C_GRID    = "#e0e0e0"

LABELS = ["Baseline\n(no RMT)", "+ MP/RMT\nDenoising", "+ Alpha Gate\nFix (n≥3)"]
COLORS = [C_BASE, C_RMT, C_FIX]


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Four-panel comparison overview
# ════════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 13), facecolor=BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

ax_wf   = fig.add_subplot(gs[0, 0])   # per-year WF accuracy
ax_base = fig.add_subplot(gs[0, 1])   # per-year base accuracy (RMT effect)
ax_early= fig.add_subplot(gs[1, 0])   # early vs late season
ax_sum  = fig.add_subplot(gs[1, 1])   # overall summary bars

for ax in [ax_wf, ax_base, ax_early, ax_sum]:
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=C_GRID, linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=DARK, labelsize=9)

x = np.arange(len(YEARS))
w = 0.26

# ── Panel A: per-year walk-forward winner accuracy ───────────────────────────
for i, (vals, lbl, col) in enumerate(zip([S1_WF, S2_WF, S3_WF], LABELS, COLORS)):
    ax_wf.bar(x + (i-1)*w, vals, width=w, label=lbl, color=col,
              alpha=0.88, edgecolor="white", linewidth=0.4, zorder=3)

# annotate 2020 improvement (biggest single-year win from data_gate fix)
yr_idx = YEARS.index(2020)
ax_wf.annotate("+5.9pp", xy=(x[yr_idx]+w, S3_WF[yr_idx]+0.01),
               fontsize=8, color=C_FIX, fontweight="bold", ha="center")

ax_wf.set_xticks(x)
ax_wf.set_xticklabels(YEARS, fontsize=9)
ax_wf.set_ylabel("Winner Accuracy", fontsize=10, color=DARK)
ax_wf.set_title("A  Walk-Forward Race Winner Accuracy per Year",
                fontsize=11, fontweight="bold", color=DARK, pad=8)
ax_wf.set_ylim(0.3, 1.0)
ax_wf.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax_wf.legend(fontsize=8, framealpha=0.9, loc="upper right")

# overall averages as dashed horizontal lines
ax_wf.axhline(OVR_S1_WF, color=C_BASE, lw=1.1, ls="--", alpha=0.6)
ax_wf.axhline(OVR_S2_WF, color=C_RMT,  lw=1.1, ls="--", alpha=0.6)
ax_wf.axhline(OVR_S3_WF, color=C_FIX,  lw=1.3, ls="--", alpha=0.9)
ax_wf.text(len(YEARS)-0.35, OVR_S3_WF+0.008, f"avg {OVR_S3_WF:.1%}",
           color=C_FIX, fontsize=8, fontweight="bold")

# ── Panel B: per-year BASE accuracy (shows RMT's direct contribution) ────────
for i, (vals, lbl, col) in enumerate(zip([S1_BASE, S2_BASE, S3_BASE], LABELS, COLORS)):
    ax_base.bar(x + (i-1)*w, vals, width=w, label=lbl, color=col,
                alpha=0.88, edgecolor="white", linewidth=0.4, zorder=3)

# annotate 2019 base improvement from RMT
yr_idx2 = YEARS.index(2019)
ax_base.annotate("+4.7pp", xy=(x[yr_idx2]+0, S2_BASE[yr_idx2]+0.01),
                 fontsize=8, color=C_RMT, fontweight="bold", ha="center")

ax_base.set_xticks(x)
ax_base.set_xticklabels(YEARS, fontsize=9)
ax_base.set_ylabel("Winner Accuracy (Base Ensemble)", fontsize=10, color=DARK)
ax_base.set_title("B  Base Ensemble Accuracy — Effect of RMT Denoising",
                  fontsize=11, fontweight="bold", color=DARK, pad=8)
ax_base.set_ylim(0.3, 1.0)
ax_base.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax_base.axhline(OVR_S1_BASE, color=C_BASE, lw=1.1, ls="--", alpha=0.6)
ax_base.axhline(OVR_S2_BASE, color=C_RMT,  lw=1.3, ls="--", alpha=0.9)
ax_base.text(len(YEARS)-0.35, OVR_S2_BASE+0.008, f"avg {OVR_S2_BASE:.1%}",
             color=C_RMT, fontsize=8, fontweight="bold")
ax_base.legend(fontsize=8, framealpha=0.9, loc="upper right")

# ── Panel C: early vs late season breakdown ──────────────────────────────────
cat_labels = ["Races 1–5\n(early season)", "Races 6+\n(mid/late season)"]
early_vals = [EARLY_S1, EARLY_S2, EARLY_S3]
late_vals  = [LATE_S1, LATE_S2, LATE_S3]

x2 = np.array([0.0, 1.0])
for i, (ev, lv, lbl, col) in enumerate(zip(early_vals, late_vals, LABELS, COLORS)):
    ax_early.bar(x2 + (i-1)*w, [ev, lv], width=w, label=lbl, color=col,
                 alpha=0.88, edgecolor="white", linewidth=0.4, zorder=3)

# annotate the critical baseline penalty
ax_early.annotate("PL hurts\nearly races",
                  xy=(x2[0]+(0-1)*w, EARLY_S1+0.005),
                  xytext=(x2[0]+(0-1)*w - 0.2, EARLY_S1 + 0.045),
                  fontsize=7.5, color=C_ACCENT, ha="center",
                  arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2))
ax_early.annotate("Fixed!",
                  xy=(x2[0]+(2-1)*w, EARLY_S3+0.005),
                  fontsize=8, color=C_FIX, fontweight="bold", ha="center")

# draw the base-ensemble level as a reference
# The reference line runs only across the early-season group (xmax=0.40 axes frac).
# Label is placed in the GAP between the two groups (data x≈0.52, just right of the
# early bars) so it does not collide with either the "PL hurts" arrow on the left or
# the "Fixed!" label above the third early bar.
ax_early.axhline(EARLY_BASE, xmin=0.0, xmax=0.40, color="black",
                 ls=":", lw=1.2, alpha=0.5)
ax_early.text(0.52, EARLY_BASE + 0.003, "base level",
              fontsize=7.5, color=DARK, ha="left", alpha=0.7)

ax_early.set_xticks(x2)
ax_early.set_xticklabels(cat_labels, fontsize=10)
ax_early.set_ylabel("WF Winner Accuracy", fontsize=10, color=DARK)
ax_early.set_title("C  Early vs Mid/Late Season WF Accuracy",
                   fontsize=11, fontweight="bold", color=DARK, pad=8)
ax_early.set_ylim(0.45, 0.70)
ax_early.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax_early.legend(fontsize=8, framealpha=0.9, loc="lower right")

# ── Panel D: headline overall summary ────────────────────────────────────────
scenarios = ["Baseline\n(no RMT)", "+ MP/RMT\nDenoising", "+ Alpha Gate\nFix (n≥3)"]
wf_vals   = [OVR_S1_WF, OVR_S2_WF, OVR_S3_WF]
base_vals = [OVR_S1_BASE, OVR_S2_BASE, OVR_S3_BASE]

x3 = np.arange(3)
bw = 0.38
ax_sum.bar(x3 - bw/2, wf_vals,   width=bw, color=COLORS, alpha=0.9,
           edgecolor="white", linewidth=0.5, label="WF (with PL)", zorder=3)
ax_sum.bar(x3 + bw/2, base_vals, width=bw, color=COLORS, alpha=0.45,
           edgecolor="white", linewidth=0.5, label="Base ensemble only", zorder=3,
           hatch="//")

# annotate improvement arrows
for i, (wf, base) in enumerate(zip(wf_vals, base_vals)):
    ax_sum.text(i - bw/2, wf + 0.004, f"{wf:.1%}", ha="center",
                fontsize=8.5, fontweight="bold", color=COLORS[i])
    ax_sum.text(i + bw/2, base + 0.004, f"{base:.1%}", ha="center",
                fontsize=7.5, color=COLORS[i], alpha=0.8)

# draw gain arrows between scenarios
y_arrow = 0.545
for i, j, gain, c in [(0, 2, OVR_S3_WF - OVR_S1_WF, C_FIX)]:
    ax_sum.annotate("", xy=(j - bw/2, wf_vals[j] + 0.012),
                    xytext=(0 - bw/2, wf_vals[0] + 0.012),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.8))
    ax_sum.text(1.0 - bw/2, wf_vals[0] + 0.022,
                f"+{gain:.2%} total\nWF gain", ha="center",
                fontsize=9, color=c, fontweight="bold")

ax_sum.set_xticks(x3)
ax_sum.set_xticklabels(scenarios, fontsize=9)
ax_sum.set_ylabel("Overall Winner Accuracy (2018–2024)", fontsize=10, color=DARK)
ax_sum.set_title("D  Overall Accuracy — All Improvements Combined",
                 fontsize=11, fontweight="bold", color=DARK, pad=8)
ax_sum.set_ylim(0.52, 0.64)
ax_sum.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax_sum.legend(fontsize=8, framealpha=0.9, loc="upper left")

fig.suptitle(
    "Impact of Marchenko-Pastur RMT Denoising on F1 Walk-Forward Prediction Accuracy\n"
    "Holdout evaluation: 2018–2024 seasons  ·  149 races total",
    fontsize=13, fontweight="bold", color=DARK, y=0.995
)

out1 = os.path.join(PLOT_DIR, "rmt_accuracy_comparison.png")
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved: {out1}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Stacked gain waterfall — shows the two-step improvement story
# ════════════════════════════════════════════════════════════════════════════════

fig2, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig2.subplots_adjust(wspace=0.35)

for ax in axes:
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.grid(axis="y", color=C_GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=DARK, labelsize=9)

# ── Left: waterfall of WF accuracy improvements ──────────────────────────────
ax_wf2 = axes[0]
steps = ["Baseline\n(no RMT,\noriginal gate)", "+ MP/RMT\nDriver-Cluster\nDenoising",
         "+ Alpha Gate\nFix\n(n ≥ 3 races)"]
accs  = [OVR_S1_WF, OVR_S2_WF, OVR_S3_WF]
cols  = [C_BASE, C_RMT, C_FIX]

bar_w = 0.52
for i, (step, acc, col) in enumerate(zip(steps, accs, cols)):
    ax_wf2.bar(i, acc, width=bar_w, color=col, alpha=0.88,
               edgecolor="white", linewidth=0.5, zorder=3)
    ax_wf2.text(i, acc + 0.003, f"{acc:.2%}", ha="center",
                fontsize=10, fontweight="bold", color=col)

# draw delta annotations
for i in range(len(accs)-1):
    delta = accs[i+1] - accs[i]
    y_mid = (accs[i] + accs[i+1]) / 2
    if abs(delta) > 1e-4:
        ax_wf2.annotate(
            f"{delta:+.2%}",
            xy=(i+1, accs[i+1]),
            xytext=(i+0.5, y_mid + 0.006),
            ha="center", fontsize=9, fontweight="bold",
            color=C_FIX if delta > 0 else C_ACCENT,
            arrowprops=dict(arrowstyle="-", color=C_GRID, lw=0.8)
        )
    else:
        # Place the "no WF change" note BELOW the bar value labels (near bar mid-height)
        # so it does not collide with the "Total gain" headline that sits above the bars.
        # A white-background box keeps it readable against the bar fill colour.
        ax_wf2.text(i+0.5, accs[i] - 0.006, "no WF\nchange\n(base↑)", ha="center",
                    fontsize=7.5, color=C_RMT, style="italic", va="top",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.85))

ax_wf2.set_xticks(range(len(steps)))
ax_wf2.set_xticklabels(steps, fontsize=9)
ax_wf2.set_ylabel("Walk-Forward Winner Accuracy", fontsize=10, color=DARK)
ax_wf2.set_title("Walk-Forward Accuracy Progression\n(2018–2024 holdout average)",
                 fontsize=11, fontweight="bold", color=DARK)
# Extend ylim upward so "Total gain" headline has clear space above the bars
ax_wf2.set_ylim(0.565, 0.621)
ax_wf2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))

# bracket for total gain – text placed higher to stay clear of bar value labels
total = OVR_S3_WF - OVR_S1_WF
ax_wf2.annotate("",
                xy=(2 + bar_w/2 + 0.08, OVR_S3_WF),
                xytext=(0 - bar_w/2 - 0.08, OVR_S1_WF),
                arrowprops=dict(arrowstyle="<->", color=C_FIX, lw=2.0))
ax_wf2.text(1.0, OVR_S3_WF + 0.011,
            f"Total gain: +{total:.2%}", ha="center",
            fontsize=10, fontweight="bold", color=C_FIX)

# ── Right: per-year improvement heatmap-style ─────────────────────────────────
ax_ht = axes[1]

rmt_delta  = np.array(S2_BASE) - np.array(S1_BASE)    # RMT base improvement per year
gate_delta = np.array(S3_WF)   - np.array(S2_WF)     # data_gate WF improvement per year

bar_h = 0.3
y = np.arange(len(YEARS), 0, -1, dtype=float)

# RMT base improvements (blue)
bars_rmt = ax_ht.barh(y + bar_h/2, rmt_delta * 100, height=bar_h,
                       color=C_RMT, alpha=0.85, edgecolor="white", linewidth=0.3,
                       label="Base accuracy gain (RMT denoising)")
# gate-fix WF improvements (green)
bars_gate = ax_ht.barh(y - bar_h/2, gate_delta * 100, height=bar_h,
                        color=C_FIX, alpha=0.85, edgecolor="white", linewidth=0.3,
                        label="WF accuracy gain (alpha gate fix)")

# labels on non-zero bars
for bar, val in zip(bars_rmt, rmt_delta):
    if abs(val) > 0.001:
        ax_ht.text(val*100 + 0.2, bar.get_y()+bar.get_height()/2,
                   f"+{val*100:.1f}pp", va="center", fontsize=8, color=C_RMT, fontweight="bold")
for bar, val in zip(bars_gate, gate_delta):
    if abs(val) > 0.001:
        ax_ht.text(val*100 + 0.2, bar.get_y()+bar.get_height()/2,
                   f"+{val*100:.1f}pp", va="center", fontsize=8, color=C_FIX, fontweight="bold")

ax_ht.axvline(0, color=DARK, lw=0.8, alpha=0.4)
ax_ht.set_yticks(y)
ax_ht.set_yticklabels([str(yr) for yr in YEARS], fontsize=9)
ax_ht.set_xlabel("Improvement in Winner Accuracy (percentage points)", fontsize=9, color=DARK)
ax_ht.set_title("Per-Year Accuracy Gains by Component\n(pp = percentage points)",
                fontsize=11, fontweight="bold", color=DARK)
ax_ht.legend(fontsize=8.5, framealpha=0.9, loc="lower right")
ax_ht.set_xlim(-1, 8)

fig2.suptitle(
    "Decomposing the Accuracy Improvements: RMT Denoising vs Alpha Gate Fix",
    fontsize=12, fontweight="bold", color=DARK, y=1.01
)

out2 = os.path.join(PLOT_DIR, "rmt_accuracy_waterfall.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print(f"Saved: {out2}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Early vs late season deep-dive
# ════════════════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(figsize=(10, 5.5), facecolor=BG)
ax3.set_facecolor(BG)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_color(C_GRID)
ax3.spines["bottom"].set_color(C_GRID)
ax3.grid(axis="y", color=C_GRID, lw=0.6, zorder=0)
ax3.tick_params(colors=DARK, labelsize=9)

x_pos = np.array([0, 1.5, 3.5, 5])
bars_y = [EARLY_S1, EARLY_S3, LATE_S1, LATE_S3]
cols_y = [C_ACCENT, C_FIX, C_BASE, C_FIX]
labels_y = ["Early (races 1–5)\nBaseline", "Early (races 1–5)\n+ Gate Fix",
            "Late (races 6+)\nBaseline", "Late (races 6+)\n+ All Improvements"]
bw3 = 0.7

for xp, yv, col, lbl in zip(x_pos, bars_y, cols_y, labels_y):
    ax3.bar(xp, yv, width=bw3, color=col, alpha=0.85,
            edgecolor="white", linewidth=0.5, zorder=3)
    ax3.text(xp, yv + 0.004, f"{yv:.1%}", ha="center",
             fontsize=11, fontweight="bold", color=col)

# base ensemble reference line for early races
ax3.axhline(EARLY_BASE, xmin=0.0, xmax=0.42, color=DARK, ls=":",
            lw=1.5, alpha=0.5, zorder=2)
ax3.text(-0.5, EARLY_BASE + 0.004, "Base ensemble\nbenchmark (57.1%)",
         fontsize=8, color=DARK, alpha=0.7)

# penalty annotation
ax3.annotate("",
             xy=(x_pos[0], EARLY_S1 - 0.003),
             xytext=(x_pos[0], EARLY_BASE - 0.003),
             arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=1.8))
ax3.text(x_pos[0] + 0.55, (EARLY_S1 + EARLY_BASE)/2,
         f"−{(EARLY_BASE-EARLY_S1)*100:.1f}pp\npenalty\n(PL fires\ntoo early)",
         fontsize=8, color=C_ACCENT, ha="left", va="center")

# fix annotation
ax3.annotate("",
             xy=(x_pos[1], EARLY_S3 - 0.003),
             xytext=(x_pos[0], EARLY_S1 - 0.003),
             arrowprops=dict(arrowstyle="->", color=C_FIX, lw=2.0))
ax3.text((x_pos[0]+x_pos[1])/2, EARLY_S1 - 0.016,
         f"+{(EARLY_S3-EARLY_S1)*100:.1f}pp\npenalty eliminated",
         fontsize=8.5, color=C_FIX, ha="center", fontweight="bold")

ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels_y, fontsize=9)
ax3.set_ylabel("Walk-Forward Winner Accuracy", fontsize=11, color=DARK)
ax3.set_title(
    "Early vs Late Season: How the Alpha Gate Fix Removes the Early-Race Penalty\n"
    "Requiring ≥3 observed results before PL activates eliminates the 2.8pp early-season deficit",
    fontsize=11, fontweight="bold", color=DARK, pad=8
)
ax3.set_ylim(0.50, 0.67)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

out3 = os.path.join(PLOT_DIR, "rmt_early_season_fix.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig3)
print(f"Saved: {out3}")

print("\nAll comparison plots saved to final_report/plots/")
print("  rmt_accuracy_comparison.png  — 4-panel overview")
print("  rmt_accuracy_waterfall.png   — gain decomposition waterfall + per-year breakdown")
print("  rmt_early_season_fix.png     — early vs late season deep-dive")
