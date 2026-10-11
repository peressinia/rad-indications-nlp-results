#!/usr/bin/env python3
"""
Script: generate_figures-v1-5.py
# Version: 1.4 [06 Jul 26] updated recall_at_90 annotation values to v4-0 results
# Version: 1.5 [07 Jul 26] make_label_freq now loads prev_full from
#                           label_prevalence_report.csv (produced by
#                           get-reasons/label_prevalence_report-v1.py) instead 
#                           of a hardcoded array, so Figure 4 is always
#                           consistent with the actual dataset.

Generates four publication-quality figures for the manuscript:

  fig:perlabel    — Per-label F1 horizontal grouped bar chart, all 4 models,
                    13 CheXpert labels sorted by best BERT F1 (Figure 2)

  fig:pr_curves   — Binary precision-recall curves for all 4 models with
                    annotated recall at 90%, 93%, 95% precision targets (Figure 3)

  fig:calibration — Two-panel reliability diagrams: binary task (Panel A) and
                    multi-label task mean across 13 labels (Panel B),
                    with Brier scores in legend (Figure 4)

  fig:label_freq  — CheXpert label prevalence grouped bar chart comparing
                    full dataset vs held-out test set (Figure 1)

Output files (saved to FIGURES_DIR):
  fig11-perlabel-f1.pdf      (+ .jpg)
  fig12-pr-curves.pdf        (+ .jpg)
  fig13-calibration.pdf      (+ .jpg)
  fig14-label-freq.pdf       (+ .jpg)

Usage:
  python scripts/generate_figures-v1-5.py

Configuration (set at top of script):
  DATA_DIR      — results_model_comparison/  (bootstrap CI CSVs)
  THRESHOLD_DIR — results_threshold/         (PR curve CSVs)
  PREDS_DIR     — data_numpy/                (.npy prediction arrays)
  PREVALENCE_CSV — results_label_report/label_prevalence_report-v4-0.csv
  FIGURES_DIR   — figures/                   (output destination)

Style targets:
  - Radiology: Artificial Intelligence figure guidelines
  - Double-column width: 7.0 inches (180 mm)
  - Single-column width: 3.5 inches (89 mm)
  - Font: sans-serif (Helvetica/Arial equivalent), 8-9 pt axis labels
  - No chart borders (spines removed top/right)
  - 300 dpi minimum for submission
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
DATA_DIR       = "./results_model_comparison"        # binary_bootstrap_ci.csv, multilabel_bootstrap_ci.csv
THRESHOLD_DIR  = "./results_threshold" # binary_pr_curves.csv, binary_operating_points.csv
PREDS_DIR      = "./data_numpy"  # .npy prediction files
FIGURES_DIR    = "./figures"           # output figures
PREVALENCE_CSV = "./results_label_report/label_prevalence_report-v4-0.csv"  # produced by label_prevalence_report-v1-1.py
DPI         = 300

FULL_DATASET   = 219995     # Values pulled manually from label_prevalence_report-v1-1.out
TEST_DATASET   = 22644      # line numbers 7 and 9

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────
# Colorblind-safe, print-friendly
COLORS = {
    "baseBERT"      : "#1A6EA8",   # dark blue
    "radBERT"       : "#2E9E6B",   # dark green
    "bioClinBERT"   : "#C05A1F",   # dark orange
    "BernoulliNB"   : "#7B4F9E",   # purple
    "MultinomialNB" : "#7B4F9E",   # purple (same — NB baseline)
}

# Line styles for PR curves
LINESTYLES = {
    "baseBERT"      : "-",
    "radBERT"       : "--",
    "bioClinBERT"   : "-.",
    "MultinomialNB" : ":",
    "BernoulliNB"   : ":",
}

# Display names for legend
DISPLAY_NAMES = {
    "baseBERT"      : "baseBERT",
    "radBERT"       : "radBERT",
    "bioClinBERT"   : "BioClinicalBERT",
    "BernoulliNB"   : "BernoulliNB",
    "MultinomialNB" : "MultinomialNB",
}

# Global matplotlib style
plt.rcParams.update({
    "font.family"       : "sans-serif",
    "font.sans-serif"   : ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size"         : 8,
    "axes.titlesize"    : 9,
    "axes.labelsize"    : 8,
    "xtick.labelsize"   : 7,
    "ytick.labelsize"   : 7,
    "legend.fontsize"   : 7,
    "legend.frameon"    : False,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "figure.dpi"        : DPI,
    "savefig.dpi"       : DPI,
    "savefig.bbox"      : "tight",
    "savefig.pad_inches": 0.05,
})


# ══════════════════════════════════════════════════════════════════
#  FIGURE 1: Per-Label F1 Horizontal Grouped Bar Chart
#  fig:perlabel  →  fig11-perlabel-f1.pdf / .jpg
# ══════════════════════════════════════════════════════════════════
def make_perlabel_f1():
    """
    Horizontal grouped bar chart showing per-label F1 for all 4 models
    across 13 CheXpert labels, sorted by best BERT F1 (ascending so
    best label is at top in horizontal orientation).
    """
    print("Generating fig11-perlabel-f1 ...")

    ml_ci = pd.read_csv(
        os.path.join(DATA_DIR, "multilabel_bootstrap_ci.csv")
    )

    per = ml_ci[
        (ml_ci["metric"] == "f1") & (ml_ci["label"] != "AGGREGATE")
    ].copy()

    # Build pivot: labels × models
    val_piv = per.pivot_table(
        index="label", columns="model", values="value"
    )
    lo_piv = per.pivot_table(
        index="label", columns="model", values="ci_lower"
    )
    hi_piv = per.pivot_table(
        index="label", columns="model", values="ci_upper"
    )

    # Sort by best BERT F1 ascending (best label at top)
    val_piv["_best_bert"] = val_piv[
        ["baseBERT", "radBERT", "bioClinBERT"]
    ].max(axis=1)
    val_piv = val_piv.sort_values("_best_bert", ascending=True)
    lo_piv  = lo_piv.loc[val_piv.index]
    hi_piv  = hi_piv.loc[val_piv.index]
    val_piv = val_piv.drop(columns="_best_bert")

    model_order = ["BernoulliNB", "bioClinBERT", "baseBERT", "radBERT"]
    labels      = val_piv.index.tolist()
    n_labels    = len(labels)
    n_models    = len(model_order)

    # Bar geometry
    bar_h    = 0.18      # height of each bar
    group_gap = 0.05     # gap between groups
    group_h  = n_models * bar_h + group_gap
    y_positions = np.arange(n_labels) * group_h

    # Figure: double-column width, tall enough for 13 labels
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    for mi, model in enumerate(model_order):
        y_offset = (mi - (n_models - 1) / 2) * bar_h
        y_vals   = y_positions + y_offset
        x_vals   = val_piv[model].values

        # Error bar extents (asymmetric)
        x_lo = x_vals - lo_piv[model].values
        x_hi = hi_piv[model].values - x_vals

        ax.barh(
            y_vals, x_vals,
            height=bar_h * 0.88,
            color=COLORS[model],
            alpha=0.85,
            label=DISPLAY_NAMES[model],
            zorder=2,
        )
        ax.errorbar(
            x_vals, y_vals,
            xerr=[x_lo, x_hi],
            fmt="none",
            ecolor="black",
            elinewidth=0.6,
            capsize=1.5,
            capthick=0.6,
            zorder=3,
        )

    # Threshold line at F1 = 0.35 (used in text as "reliable prediction")
    ax.axvline(0.35, color="gray", lw=0.8, ls="--", alpha=0.7, zorder=1)
    ax.text(0.355, y_positions[-1] + group_h * 0.4,
            "F1 = 0.35", color="gray", fontsize=6.5, va="bottom")

    # Y-axis: label names at group centers
    # Shorten long label name for display
    display_labels = [
        l.replace("Enlarged Cardiomediastinum", "Enlarged CM")
        for l in labels
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(display_labels, fontsize=7.5)

    ax.set_xlabel("F1 Score (with 95% CI)", fontsize=8)
    ax.set_xlim(0, 0.72)
    ax.set_ylim(y_positions[0] - group_h, y_positions[-1] + group_h)
    ax.xaxis.grid(True, color="lightgray", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend — ordered to match visual top-to-bottom
    handles = [
        mpatches.Patch(color=COLORS[m], label=DISPLAY_NAMES[m])
        for m in model_order[::-1]
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=7,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.3,
    )

    fig.tight_layout()

    for ext in ["pdf", "jpg"]:
        path = os.path.join(FIGURES_DIR, f"fig11-perlabel-f1.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {path}")

    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  FIGURE 2: Binary Precision-Recall Curves
#  fig:pr_curves  →  fig12-pr-curves.pdf / .jpg
# ══════════════════════════════════════════════════════════════════
def make_pr_curves():
    """
    Precision-recall curves for all 4 models on the binary task.
    Horizontal reference lines at 90%, 93%, 95% precision.
    Recall annotations at the 90% precision crossing point.
    """
    print("Generating fig12-pr-curves ...")

    pr_df = pd.read_csv(
        os.path.join(THRESHOLD_DIR, "binary_pr_curves.csv")
    )

    # Operating point recall values at 90% precision (from threshold analysis)
    recall_at_90 = {
        "baseBERT"      : 0.523,
        "radBERT"       : 0.550,
        "bioClinBERT"   : 0.532,
        "MultinomialNB" : 0.421,
    }

    model_order = ["radBERT", "bioClinBERT", "baseBERT", "MultinomialNB"]

    # Figure: double-column width, square-ish
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for model in model_order:
        sub = pr_df[pr_df["model"] == model].copy()
        # Sort by recall descending for clean curve plotting
        sub = sub.sort_values("recall", ascending=False)

        ax.plot(
            sub["recall"], sub["precision"],
            color=COLORS[model],
            ls=LINESTYLES[model],
            lw=1.6,
            label=DISPLAY_NAMES[model],
            zorder=3,
        )

    # Horizontal reference lines at target precision levels
    for prec, label_str in [(0.90, "90%"), (0.93, "93%"), (0.95, "95%")]:
        ax.axhline(
            prec, color="gray", lw=0.7, ls=":", alpha=0.8, zorder=1
        )
        ax.text(
            1.002, prec, label_str,
            va="center", ha="left",
            fontsize=6.5, color="gray"
        )

    # Annotate recall at 90% precision for each model
    # Offsets tuned to avoid overlap:
    #   radBERT (green, 0.550)      — above, right
    #   bioClinBERT (orange, 0.532) — below, right, half-char lower
    #   baseBERT (blue, 0.523)      — above, right, half-char higher
    #   MultinomialNB (purple, 0.421) — below curve, right
    annotation_offsets = {
        "radBERT"       : (+0.015, +0.014),
        "bioClinBERT"   : (-0.015, -0.020),
        "baseBERT"      : (-0.055, +0.022),
        "MultinomialNB" : (+0.015, -0.022),
    }
    for model, rec in recall_at_90.items():
        dx, dy = annotation_offsets[model]
        ax.plot(rec, 0.90, "o",
                color=COLORS[model], ms=4, zorder=4)
        ax.annotate(
            f"{rec:.2f}",
            xy=(rec, 0.90),
            xytext=(rec + dx, 0.90 + dy),
            fontsize=6,
            color=COLORS[model],
            arrowprops=dict(
                arrowstyle="-",
                color=COLORS[model],
                lw=0.6,
            ),
        )

    ax.set_xlabel("Recall", fontsize=8)
    ax.set_ylabel("Precision (PPV)", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.65, 1.01)
    ax.xaxis.grid(True, color="lightgray", lw=0.5, zorder=0)
    ax.yaxis.grid(True, color="lightgray", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.1f}")
    )

    # Legend — with note about reference lines folded in as extra handle
    handles = [
        Line2D([0], [0],
               color=COLORS[m], ls=LINESTYLES[m], lw=1.6,
               label=DISPLAY_NAMES[m])
        for m in model_order
    ]
    # Add a legend entry explaining the dotted reference lines
    handles.append(
        Line2D([0], [0], color="gray", ls=":", lw=0.8,
               label="Target precision (90%, 93%, 95%)")
    )
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=7,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.4,
    )

    fig.tight_layout()

    for ext in ["pdf", "jpg"]:
        path = os.path.join(FIGURES_DIR, f"fig12-pr-curves.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {path}")

    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  FIGURE 3: Calibration — Reliability Diagrams + Brier Scores
#  fig:calibration  →  fig13-calibration.pdf / .jpg
#
#  Two-panel figure:
#    Left:  Binary task — reliability diagrams for all 4 models
#    Right: Multi-label task — mean reliability diagram across
#           13 labels for all 4 models
#
#  Also prints Brier scores to console for inclusion in paper.
# ══════════════════════════════════════════════════════════════════
def make_calibration():
    """
    Reliability diagrams (calibration plots) + Brier scores for
    binary and multi-label tasks.

    Loads .npy files directly from PREDS_DIR.
    """
    print("Generating fig13-calibration ...")

    # ── File manifest ──────────────────────────────────────────────
    BINARY_FILES = {
        "baseBERT": {
            "y_true": "CNF-v4-0-baseBERT_bt32-ep3-lr0-bl1-tt0_pyV-1-2_y_true_binary.npy",
            "y_prob": "CNF-v4-0-baseBERT_bt32-ep3-lr0-bl1-tt0_pyV-1-2_binary_probs.npy",
        },
        "radBERT": {
            "y_true": "CNF-v4-0-radBERT_bt32-ep2-lr0-bl1-tt0_pyV-1-2_y_true_binary.npy",
            "y_prob": "CNF-v4-0-radBERT_bt32-ep2-lr0-bl1-tt0_pyV-1-2_binary_probs.npy",
        },
        "bioClinBERT": {
            "y_true": "CNF-v4-0-bioClinBERT_bt32-ep1-lr0-bl3-tt0_pyV-1-2_y_true_binary.npy",
            "y_prob": "CNF-v4-0-bioClinBERT_bt32-ep1-lr0-bl3-tt0_pyV-1-2_binary_probs.npy",
        },
        "MultinomialNB": {
            "y_true": "MultinomialNB-final-1500-NCF-v4-0-bal1_pyV-1-1_y_true_binary.npy",
            "y_prob": "MultinomialNB-final-1500-NCF-v4-0-bal1_pyV-1-1_nb_binary_probs.npy",
        },
    }

    MULTILABEL_FILES = {
        "baseBERT": {
            "y_true": "CNF-v4-0-baseBERT_bt32-ep6-lr0-bl0-tt0_pyV-1-1_y_true_multilabel.npy",
            "y_prob": "CNF-v4-0-baseBERT_bt32-ep6-lr0-bl0-tt0_pyV-1-1_multilabel_probs.npy",
        },
        "radBERT": {
            "y_true": "CNF-v4-0-radBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_y_true_multilabel.npy",
            "y_prob": "CNF-v4-0-radBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_multilabel_probs.npy",
        },
        "bioClinBERT": {
            "y_true": "CNF-v4-0-bioClinBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_y_true_multilabel.npy",
            "y_prob": "CNF-v4-0-bioClinBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_multilabel_probs.npy",
        },
        "BernoulliNB": {
            "y_true": "BernoulliNB-TFIDF-xSplits-1500-NCF-v4-0-bal0_y_true_multilabel.npy",
            "y_prob": "BernoulliNB-TFIDF-xSplits-1500-NCF-v4-0-bal0_nb_multilabel_probs.npy",
        },
    }

    # ── Helper: load and normalise ─────────────────────────────────
    def load_binary(files):
        data = {}
        for model, f in files.items():
            y_true = np.load(os.path.join(PREDS_DIR, f["y_true"]),
                             allow_pickle=True).astype(float)
            y_prob = np.load(os.path.join(PREDS_DIR, f["y_prob"]),
                             allow_pickle=True).astype(float)
            if y_prob.ndim == 2:          # NB saved as (n, 2)
                y_prob = y_prob[:, 1]
            data[model] = {"y_true": y_true, "y_prob": y_prob}
        return data

    def load_multilabel(files):
        data = {}
        for model, f in files.items():
            y_true = np.load(os.path.join(PREDS_DIR, f["y_true"]),
                             allow_pickle=True).astype(float)
            y_prob = np.load(os.path.join(PREDS_DIR, f["y_prob"]),
                             allow_pickle=True).astype(float)
            if y_prob.shape[0] == 13:     # BernoulliNB saved transposed
                y_prob = y_prob.T
            data[model] = {"y_true": y_true, "y_prob": y_prob}
        return data

    # ── Helper: reliability curve (calibration curve) ─────────────
    def reliability_curve(y_true, y_prob, n_bins=10):
        """
        Compute fraction of positives and mean predicted probability
        per bin. Returns (mean_pred, frac_pos, bin_counts).
        """
        bins      = np.linspace(0.0, 1.0, n_bins + 1)
        mean_pred = np.zeros(n_bins)
        frac_pos  = np.zeros(n_bins)
        counts    = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if i == n_bins - 1:           # include right edge in last bin
                mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
            if mask.sum() > 0:
                mean_pred[i] = y_prob[mask].mean()
                frac_pos[i]  = y_true[mask].mean()
                counts[i]    = mask.sum()
            else:
                mean_pred[i] = np.nan
                frac_pos[i]  = np.nan
                counts[i]    = 0

        return mean_pred, frac_pos, counts

    # ── Helper: Brier score ────────────────────────────────────────
    def brier_score(y_true, y_prob):
        return float(np.mean((y_prob - y_true) ** 2))

    # ── Helper: ECE ───────────────────────────────────────────────
    def ece(y_true, y_prob, n_bins=10):
        bins     = np.linspace(0.0, 1.0, n_bins + 1)
        n        = len(y_true)
        ece_val  = 0.0
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if i == n_bins - 1:
                mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
            if mask.sum() > 0:
                acc  = y_true[mask].mean()
                conf = y_prob[mask].mean()
                ece_val += (mask.sum() / n) * abs(acc - conf)
        return ece_val

    # ── Load data ─────────────────────────────────────────────────
    bin_data = load_binary(BINARY_FILES)
    ml_data  = load_multilabel(MULTILABEL_FILES)

    # ── Compute and print Brier scores ────────────────────────────
    print("\n  === Brier Scores (Binary) ===")
    bin_brier = {}
    for model, d in bin_data.items():
        bs = brier_score(d["y_true"], d["y_prob"])
        ec = ece(d["y_true"], d["y_prob"])
        bin_brier[model] = bs
        print(f"    {model:<16} Brier={bs:.4f}  ECE={ec:.4f}")

    print("\n  === Brier Scores (Multi-label, mean across 13 labels) ===")
    ml_brier = {}
    for model, d in ml_data.items():
        # Mean Brier score across all 13 labels
        bs_per_label = [
            brier_score(d["y_true"][:, i], d["y_prob"][:, i])
            for i in range(13)
        ]
        bs_mean = float(np.mean(bs_per_label))
        ec_per_label = [
            ece(d["y_true"][:, i], d["y_prob"][:, i])
            for i in range(13)
        ]
        ec_mean = float(np.mean(ec_per_label))
        ml_brier[model] = bs_mean
        print(f"    {model:<16} Brier={bs_mean:.4f}  ECE={ec_mean:.4f}")

    # ── Plot ──────────────────────────────────────────────────────
    model_order_bin = ["radBERT", "bioClinBERT", "baseBERT", "MultinomialNB"]
    model_order_ml  = ["radBERT", "bioClinBERT", "baseBERT", "BernoulliNB"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.6))

    # ── Panel A: Binary reliability diagram ───────────────────────
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5,
             label="Perfect calibration", zorder=1)

    for model in model_order_bin:
        d = bin_data[model]
        mp, fp, cnt = reliability_curve(
            d["y_true"], d["y_prob"], n_bins=10
        )
        # Mask empty bins
        valid = ~np.isnan(mp)
        bs    = bin_brier[model]
        ax1.plot(
            mp[valid], fp[valid],
            color=COLORS[model],
            ls=LINESTYLES[model],
            lw=1.6, marker="o", ms=3,
            label=f"{DISPLAY_NAMES[model]} (BS={bs:.3f})",
            zorder=3,
        )

    ax1.set_xlabel("Mean predicted probability", fontsize=8)
    ax1.set_ylabel("Fraction of positives", fontsize=8)
    ax1.set_title("A  Binary task", fontsize=9, fontweight="bold",
                  loc="left")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.legend(fontsize=6, frameon=False, loc="upper left",
               handlelength=1.5)
    ax1.grid(True, color="lightgray", lw=0.5, zorder=0)
    ax1.set_aspect("equal")

    # ── Panel B: Multi-label mean reliability diagram ─────────────
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5,
             label="Perfect calibration", zorder=1)

    for model in model_order_ml:
        d = ml_data[model]
        # Average reliability curve across all 13 labels
        all_mp = np.zeros((13, 10))
        all_fp = np.zeros((13, 10))
        all_cnt= np.zeros((13, 10))
        for i in range(13):
            mp_i, fp_i, cnt_i = reliability_curve(
                d["y_true"][:, i], d["y_prob"][:, i], n_bins=10
            )
            all_mp[i]  = np.where(np.isnan(mp_i),  0, mp_i)
            all_fp[i]  = np.where(np.isnan(fp_i),  0, fp_i)
            all_cnt[i] = cnt_i

        # Weight average by bin counts across labels
        total_cnt = all_cnt.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_mp = np.where(
                total_cnt > 0,
                (all_mp * all_cnt).sum(axis=0) / total_cnt,
                np.nan
            )
            mean_fp = np.where(
                total_cnt > 0,
                (all_fp * all_cnt).sum(axis=0) / total_cnt,
                np.nan
            )

        valid = ~np.isnan(mean_mp)
        bs    = ml_brier[model]
        ax2.plot(
            mean_mp[valid], mean_fp[valid],
            color=COLORS[model],
            ls=LINESTYLES[model],
            lw=1.6, marker="o", ms=3,
            label=f"{DISPLAY_NAMES[model]} (BS={bs:.3f})",
            zorder=3,
        )

    ax2.set_xlabel("Mean predicted probability", fontsize=8)
    ax2.set_ylabel("Fraction of positives", fontsize=8)
    ax2.set_title("B  Multi-label task (mean across 13 labels)",
                  fontsize=9, fontweight="bold", loc="left")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.legend(fontsize=6, frameon=False, loc="upper left",
               handlelength=1.5)
    ax2.grid(True, color="lightgray", lw=0.5, zorder=0)
    ax2.set_aspect("equal")

    fig.tight_layout(w_pad=3.0)

    for ext in ["pdf", "jpg"]:
        path = os.path.join(FIGURES_DIR, f"fig13-calibration.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"\n  Saved: {path}")

    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  FIGURE 4: Label Frequency / Prevalence Chart
#  fig:label_freq  →  fig14-label-freq.pdf / .jpg
#
#  Grouped horizontal bar chart showing prevalence of each of the
#  13 CheXpert labels in both the full dataset and the test set,
#  demonstrating that the patient-level split preserved the global
#  label distribution. Motivates the calibration-based explanation
#  for NB's F1 advantage on imbalanced labels.
# ══════════════════════════════════════════════════════════════════
def make_label_freq():
    """
    Grouped horizontal bar chart comparing CheXpert label prevalence
    in the full dataset vs the held-out test set.
    Loads all prevalence values from label_prevalence_report.csv
    (produced by label_prevalence_report-v1-1.py) — no hardcoded arrays.
    """
    print("Generating fig14-label-freq ...")

    LABEL_DISPLAY_MAP = {"Enlarged Cardiomediastinum": "Enlarged CM"}

    # ── Load from label_prevalence_report.csv ─────────────────────
    if not os.path.exists(PREVALENCE_CSV):
        raise FileNotFoundError(
            f"Not found: {PREVALENCE_CSV}\n Run label_prevalence_report-v1-1.py first to generate it."
        )
    prev_df = pd.read_csv(PREVALENCE_CSV)


    # CSV columns: label, full_count, full_pct, train_count, train_pct,
    #              test_count, test_pct
    label_names = prev_df["label"].tolist()
    prev_full   = prev_df["full_pct"].values.astype(float)
    prev_test   = prev_df["test_pct"].values.astype(float)

    # Rather than save in CSV and break form - hard code them
    n_full_records = FULL_DATASET
    n_test_records = TEST_DATASET

    labels_display = [LABEL_DISPLAY_MAP.get(l, l) for l in label_names]

    # ── Sort by full dataset prevalence ascending (rarest at bottom) ─
    order         = np.argsort(prev_full)
    labels_sorted = [labels_display[i] for i in order]
    prev_full_s   = prev_full[order]
    prev_test_s   = prev_test[order]

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    n_labels = len(labels_sorted)
    y        = np.arange(n_labels)
    h        = 0.35

    bars_full = ax.barh(y + h/2, prev_full_s, h,
                        color="#1A6EA8", alpha=0.80,
                        label=f"Full dataset ($n = {n_full_records:,}$)")
    bars_test = ax.barh(y - h/2, prev_test_s, h,
                        color="#2E9E6B", alpha=0.80,
                        label=f"Test set ($n = {n_test_records:,}$)")

    # Value labels on test bars only
    for bar, val in zip(bars_test, prev_test_s):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_sorted, fontsize=8)
    ax.set_xlabel("Prevalence (%)", fontsize=8)
    ax.set_xlim(0, 42)
    ax.grid(True, axis="x", color="lightgray", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()

    for ext in ["pdf", "jpg"]:
        path = os.path.join(FIGURES_DIR, f"fig14-label-freq.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {path}")

    plt.close(fig)

    # Print values for caption verification
    print(f"Test set label prevalences:")
    for l, p_f, p_t in zip(labels_sorted, prev_full_s, prev_test_s):
        print(f"    {l:<28} full={p_f:.1f}%  test={p_t:.1f}%")
# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("Figure generation script")
    print(f"  Data dir      : {DATA_DIR}")
    print(f"  Threshold dir : {THRESHOLD_DIR}")
    print(f"  Preds dir     : {PREDS_DIR}")
    print(f"  Figures dir   : {FIGURES_DIR}")
    print(f"  DPI        : {DPI}")
    print()

    make_perlabel_f1()
    print()
    make_pr_curves()
    print()
    make_calibration()
    print()
    make_label_freq()

    print()
    print("Done. Files written:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if any(f.startswith(x) for x in ["fig11","fig12","fig13","fig14"]):
            size = os.path.getsize(os.path.join(FIGURES_DIR, f))
            print(f"  {f:<35} {size:>10,} bytes")


if __name__ == "__main__":
    main()