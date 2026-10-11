#!/usr/bin/env python3
"""
Script: threshold_analysis.py
Version: 1.4

Purpose:
    Precision-recall threshold analysis for the RESOURCE ALLOCATION use case.

    The application goal is NOT to skip imaging for low-risk patients.
    It IS to identify indication patterns so strongly predictive of a
    significant finding that the X-ray can be skipped and the patient
    sent directly to CT/MRI.

    In this framing:
      - False Positive  = model says "escalate", reference report shows
                          no finding → a false-positive escalation: an
                          advanced study unsupported by the radiographic
                          evidence that would otherwise have been obtained
      - False Negative  = model says "don't escalate", patient needed it
                          → patient follows current X-ray-first pathway
                          → NO HARM — this is just the status quo

    Therefore PRECISION is the primary operating metric.
    RECALL tells us how much waste we can eliminate at a given precision.

Outputs:
  ./results_threshold/
    binary_pr_curves.csv          — full precision/recall/threshold per model
    binary_operating_points.csv   — recall at target precision levels
    binary_xray_savings.csv       — annual X-rays avoidable and false-positive
                                    escalations, scaled to ANNUAL_VOLUME
    multilabel_pr_curves.csv      — per-label PR curves for each model
    multilabel_operating_points.csv — per-label recall at target precisions

  ./diagnostic_figures/
    threshold_analysis.png        — summary diagnostic plot

Usage:
    python threshold_analysis.py

Configuration:
    Set PREDS_DIR to your .npy files folder.
    Set ANNUAL_VOLUME to your institution's annual order volume for scaling.
    Set TARGET_PRECISIONS to the precision levels you want to report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score
)

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
PREDS_DIR    = "./data_numpy"
RESULTS_DIR  = "./results_threshold"
DIAG_DIR     = "./diagnostic_figures"

# Precision levels at which to report recall — tune for your use case.
# 0.90 = 1 false escalation per 10; 0.95 = 1 per 20; 0.99 = 1 per 100
TARGET_PRECISIONS = [0.85, 0.90, 0.93, 0.95, 0.97, 0.99]

# Scale results to annual institutional volume.
# Replace with actual volume for your target institution/publication.
ANNUAL_VOLUME = 100_000   # orders per year

# CheXpert label names — must match array column order
LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax"
]

# ── File manifest ──────────────────────────────────────────────────
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

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  LOAD DATA Routines
# ══════════════════════════════════════════════════════════════════
def load_npy(filename):
    path = os.path.join(PREDS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    arr = np.load(path, allow_pickle=True)
    return arr


def load_binary(files):
    """Load and normalise binary y_true and y_prob."""
    data = {}
    for model, f in files.items():
        y_true = load_npy(f["y_true"]).astype(int)
        y_prob  = load_npy(f["y_prob"]).astype(float)
        # NB probs saved as (n,2) — take positive class column
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        data[model] = {"y_true": y_true, "y_prob": y_prob}
        print(f"  Loaded binary  {model}: n={len(y_true):,}  "
              f"prevalence={y_true.mean():.3f}")
    return data


def load_multilabel(files):
    """Load and normalise multilabel y_true and y_prob."""
    data = {}
    for model, f in files.items():
        y_true = load_npy(f["y_true"]).astype(int)
        y_prob  = load_npy(f["y_prob"]).astype(float)
        # Transpose if saved as (n_labels, n_samples)
        if y_prob.shape == (len(LABEL_NAMES), y_true.shape[0]):
            y_prob = y_prob.T
        data[model] = {"y_true": y_true, "y_prob": y_prob}
        print(f"  Loaded multilabel {model}: shape={y_true.shape}")
    return data


# ══════════════════════════════════════════════════════════════════
#  BINARY PR ANALYSIS
# ══════════════════════════════════════════════════════════════════
def binary_pr_analysis(binary_data):
    """
    For each model compute full PR curve and recall at target precision levels.
    Also compute estimated X-rays avoidable at each operating point.
    """
    print("\n" + "═"*60)
    print("  BINARY — Precision-Recall Threshold Analysis")
    print("═"*60)

    n_test       = len(list(binary_data.values())[0]["y_true"])
    prevalence   = list(binary_data.values())[0]["y_true"].mean()
    n_true_pos   = int(list(binary_data.values())[0]["y_true"].sum())
    scale_factor = ANNUAL_VOLUME / n_test

    print(f"\n  Test set n={n_test:,}  |  prevalence={prevalence:.3f}  "
          f"|  true positives={n_true_pos:,}")
    print(f"  Annual volume={ANNUAL_VOLUME:,}  |  "
          f"scale factor={scale_factor:.2f}x")

    curve_rows   = []
    op_rows      = []
    savings_rows = []

    for model, d in binary_data.items():
        y_true, y_prob = d["y_true"], d["y_prob"]
        auprc = average_precision_score(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)

        # sklearn returns precision/recall in decreasing-threshold order
        # precision[i], recall[i] correspond to threshold[i]
        # len(threshold) = len(precision) - 1  (last point is precision=1, recall=0)
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)

        # Store full curve
        for p, r, t in zip(prec_arr[:-1], rec_arr[:-1], thresh_arr):
            curve_rows.append({
                "model": model, "threshold": round(float(t), 4),
                "precision": round(float(p), 4), "recall": round(float(r), 4),
                "tp_rate": round(float(r), 4),
            })

        print(f"\n  {model}  (AUPRC={auprc:.4f}  AUROC={auroc:.4f})")
        print(f"  {'Target prec':>12} {'Threshold':>10} {'Recall':>8} "
              f"{'TP captured':>13} {'FP cost':>10} "
              f"{'Ann. xrays saved':>18}")
        print(f"  {'-'*75}")

        for target_p in TARGET_PRECISIONS:
            # Find indices where precision >= target.
            # prec_arr[:-1] excludes sklearn's sentinel endpoint
            # (precision=1, recall=0), which has no corresponding
            # threshold; including it yields a degenerate "operating
            # point" with recall 0 and a clamped threshold.
            idx = np.where(prec_arr[:-1] >= target_p)[0]
            if len(idx) == 0:
                print(f"  {target_p:>12.0%}  {'not achievable':>10}")
                op_rows.append({
                    "model": model, "target_precision": target_p,
                    "achievable": False,
                    "threshold": np.nan, "actual_precision": np.nan,
                    "recall": np.nan, "tp_captured_test": np.nan,
                    "fp_cost_test": np.nan,
                    "annual_xrays_saved": np.nan,
                    "annual_fp_escalations": np.nan,
                    "auprc"               : round(auprc, 4),
                    "auroc"               : round(auroc, 4),
                })
                continue

            # Among achievable thresholds, pick the one with highest recall
            # (lowest threshold that still meets precision target)
            best_idx   = idx[np.argmax(rec_arr[idx])]
            actual_p   = float(prec_arr[best_idx])
            actual_r   = float(rec_arr[best_idx])
            threshold  = float(thresh_arr[best_idx])

            # At this threshold, count predicted positives and true
            # positives directly from the arrays. Counting TP rather than
            # reconstructing it as round(recall * n_true_pos) keeps any
            # rounding error out of the FP (escalation) column, which is
            # derived by subtraction.
            flagged       = y_prob >= threshold
            predicted_pos = int(np.sum(flagged))
            tp_test       = int(np.sum(y_true[flagged]))
            fp_test       = predicted_pos - tp_test

            # Scale to annual volume
            ann_saved  = int(round(tp_test * scale_factor))
            ann_fp     = int(round(fp_test * scale_factor))

            print(f"  {target_p:>12.0%} {threshold:>10.4f} {actual_r:>8.3f} "
                  f"{tp_test:>10,} ({actual_r:.1%}) "
                  f"{fp_test:>7,} FP "
                  f"{ann_saved:>12,} saved")

            op_rows.append({
                "model"               : model,
                "target_precision"    : target_p,
                "achievable"          : True,
                "threshold"           : round(threshold, 4),
                "actual_precision"    : round(actual_p, 4),
                "recall"              : round(actual_r, 4),
                "tp_captured_test"    : tp_test,
                "fp_cost_test"        : fp_test,
                "annual_xrays_saved"  : ann_saved,
                "annual_fp_escalations": ann_fp,
                "auprc"               : round(auprc, 4),
                "auroc"               : round(auroc, 4),
            })

            savings_rows.append({
                "model"                   : model,
                "target_precision"        : target_p,
                "annual_xrays_avoided"    : ann_saved,
                "annual_fp_escalations"   : ann_fp,
                "recall_pct"              : round(actual_r * 100, 1),
            })

    curve_df   = pd.DataFrame(curve_rows)
    op_df      = pd.DataFrame(op_rows)
    savings_df = pd.DataFrame(savings_rows)

    curve_df.to_csv(f"{RESULTS_DIR}/binary_pr_curves.csv", index=False)
    op_df.to_csv(f"{RESULTS_DIR}/binary_operating_points.csv", index=False)
    savings_df.to_csv(f"{RESULTS_DIR}/binary_xray_savings.csv", index=False)

    print(f"\n  Saved: binary_pr_curves.csv")
    print(f"  Saved: binary_operating_points.csv")
    print(f"  Saved: binary_xray_savings.csv")

    return curve_df, op_df, savings_df


# ══════════════════════════════════════════════════════════════════
#  MULTI-LABEL PR ANALYSIS
# ══════════════════════════════════════════════════════════════════
def multilabel_pr_analysis(multilabel_data):
    """
    Per-label PR curves and operating points for each model.
    Most useful for identifying which specific finding types
    can drive triage decisions at high precision.
    """
    print("\n" + "═"*60)
    print("  MULTI-LABEL — Per-Label Threshold Analysis")
    print("═"*60)

    n_test       = list(multilabel_data.values())[0]["y_true"].shape[0]
    scale_factor = ANNUAL_VOLUME / n_test

    ml_curve_rows = []
    ml_op_rows    = []

    for model, d in multilabel_data.items():
        y_true_ml = d["y_true"]
        y_prob_ml = d["y_prob"]
        print(f"\n  {model}")

        for i, label in enumerate(LABEL_NAMES):
            y_true_l = y_true_ml[:, i]
            y_prob_l = y_prob_ml[:, i]
            prevalence_l = y_true_l.mean()

            if prevalence_l < 0.005:
                # Skip near-empty labels
                continue

            try:
                auprc_l = average_precision_score(y_true_l, y_prob_l)
            except Exception:
                auprc_l = np.nan

            prec_arr, rec_arr, thresh_arr = precision_recall_curve(
                y_true_l, y_prob_l
            )

            for p, r, t in zip(prec_arr[:-1], rec_arr[:-1], thresh_arr):
                ml_curve_rows.append({
                    "model": model, "label": label,
                    "threshold": round(float(t), 4),
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                })

            for target_p in TARGET_PRECISIONS:
                idx = np.where(prec_arr >= target_p)[0]
                if len(idx) == 0:
                    ml_op_rows.append({
                        "model": model, "label": label,
                        "prevalence": round(prevalence_l, 4),
                        "auprc": round(auprc_l, 4) if not np.isnan(auprc_l) else np.nan,
                        "target_precision": target_p,
                        "achievable": False,
                        "threshold": np.nan, "actual_precision": np.nan,
                        "recall": np.nan, "annual_cases_captured": np.nan,
                    })
                    continue

                best_idx  = idx[np.argmax(rec_arr[idx])]
                actual_p  = float(prec_arr[best_idx])
                actual_r  = float(rec_arr[best_idx])
                threshold = float(thresh_arr[min(best_idx, len(thresh_arr)-1)])
                n_pos_l   = int(y_true_l.sum())
                tp_l      = int(round(actual_r * n_pos_l))
                ann_cases = int(round(tp_l * scale_factor))

                ml_op_rows.append({
                    "model"               : model,
                    "label"               : label,
                    "prevalence"          : round(prevalence_l, 4),
                    "auprc"               : round(auprc_l, 4),
                    "target_precision"    : target_p,
                    "achievable"          : True,
                    "threshold"           : round(threshold, 4),
                    "actual_precision"    : round(actual_p, 4),
                    "recall"              : round(actual_r, 4),
                    "annual_cases_captured": ann_cases,
                })

    ml_curve_df = pd.DataFrame(ml_curve_rows)
    ml_op_df    = pd.DataFrame(ml_op_rows)

    ml_curve_df.to_csv(f"{RESULTS_DIR}/multilabel_pr_curves.csv", index=False)
    ml_op_df.to_csv(f"{RESULTS_DIR}/multilabel_operating_points.csv", index=False)

    # ── Summary: best recall per label at 90% and 95% precision ──
    print(f"\n  {'Label':<30} {'Prev':>5}  "
          f"{'BERT recall@90%':>16}  {'BERT recall@95%':>16}  "
          f"{'NB recall@90%':>14}  {'NB recall@95%':>14}")
    print(f"  {'-'*100}")

    bert_models = ["baseBERT", "radBERT", "bioClinBERT"]
    nb_model    = "BernoulliNB"

    for label in LABEL_NAMES:
        sub = ml_op_df[ml_op_df["label"] == label]
        if sub.empty:
            continue

        prev = sub["prevalence"].iloc[0]

        def get_recall(mdl, tp):
            row = sub[(sub["model"] == mdl) & (sub["target_precision"] == tp)]
            if row.empty or not row.iloc[0]["achievable"]:
                return "n/a"
            return f"{row.iloc[0]['recall']:.3f}"

        # Best BERT at each precision
        bert_r90 = max(
            [sub[(sub["model"]==m)&(sub["target_precision"]==0.90)]["recall"].values[0]
             if (len(sub[(sub["model"]==m)&(sub["target_precision"]==0.90)])>0
                 and sub[(sub["model"]==m)&(sub["target_precision"]==0.90)]["achievable"].values[0])
             else 0.0
             for m in bert_models]
        )
        bert_r95 = max(
            [sub[(sub["model"]==m)&(sub["target_precision"]==0.95)]["recall"].values[0]
             if (len(sub[(sub["model"]==m)&(sub["target_precision"]==0.95)])>0
                 and sub[(sub["model"]==m)&(sub["target_precision"]==0.95)]["achievable"].values[0])
             else 0.0
             for m in bert_models]
        )
        nb_r90 = get_recall(nb_model, 0.90)
        nb_r95 = get_recall(nb_model, 0.95)

        bert_r90_s = f"{bert_r90:.3f}" if bert_r90 > 0 else "n/a"
        bert_r95_s = f"{bert_r95:.3f}" if bert_r95 > 0 else "n/a"

        print(f"  {label:<30} {prev:>5.3f}  "
              f"{bert_r90_s:>16}  {bert_r95_s:>16}  "
              f"{nb_r90:>14}  {nb_r95:>14}")

    print(f"\n  Saved: multilabel_pr_curves.csv")
    print(f"  Saved: multilabel_operating_points.csv")

    return ml_curve_df, ml_op_df


# ══════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════
def make_plots(binary_data, curve_df, op_df,
               ml_op_df, results_dir, diag_dir):
    """Generate summary plots."""

    colors = {
        "baseBERT"    : "#185FA5",
        "radBERT"     : "#0F6E56",
        "bioClinBERT" : "#993C1D",
        "MultinomialNB": "#888780",
        "BernoulliNB" : "#5F5E5A",
    }
    linestyles = {
        "baseBERT"    : "-",
        "radBERT"     : "--",
        "bioClinBERT" : "-.",
        "MultinomialNB": ":",
        "BernoulliNB" : ":",
    }

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel 1: Binary PR curves ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    for model, d in binary_data.items():
        sub = curve_df[curve_df["model"] == model].sort_values("recall", ascending=False)
        ax1.plot(sub["recall"], sub["precision"],
                 color=colors[model], ls=linestyles[model],
                 lw=2, label=model)

    for tp in [0.90, 0.95, 0.99]:
        ax1.axhline(tp, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax1.text(1.01, tp, f"{tp:.0%}", va="center", fontsize=8, color="gray")

    ax1.set_xlabel("Recall (fraction of true findings captured)", fontsize=11)
    ax1.set_ylabel("Precision (PPV)", fontsize=11)
    ax1.set_title("Binary task — precision-recall curves\n"
                  "Dotted lines = target precision operating points", fontsize=11)
    ax1.legend(fontsize=9, loc="lower left")
    ax1.set_xlim([0, 1]); ax1.set_ylim([0.5, 1.01])
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Annual X-rays saved at each precision target ──────
    ax2 = fig.add_subplot(gs[0, 2])
    bert_models = ["baseBERT", "radBERT", "bioClinBERT"]
    for model in bert_models:
        sub = op_df[(op_df["model"]==model) & op_df["achievable"]].copy()
        if sub.empty: continue
        ax2.plot(sub["target_precision"]*100,
                 sub["annual_xrays_saved"] / 1000,
                 color=colors[model], ls=linestyles[model],
                 lw=2, marker="o", ms=5, label=model)

    sub_nb = op_df[(op_df["model"]=="MultinomialNB") & op_df["achievable"]]
    if not sub_nb.empty:
        ax2.plot(sub_nb["target_precision"]*100,
                 sub_nb["annual_xrays_saved"] / 1000,
                 color=colors["MultinomialNB"], ls=":", lw=2,
                 marker="s", ms=5, label="MultinomialNB")

    ax2.set_xlabel("Target precision (%)", fontsize=11)
    ax2.set_ylabel(f"X-rays avoidable per year\n(thousands, at {ANNUAL_VOLUME:,} orders/yr)",
                   fontsize=10)
    ax2.set_title("Estimated annual X-rays avoided\nper institution", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Per-label recall at 90% precision (best BERT vs NB) ──
    ax3 = fig.add_subplot(gs[1, :])

    labels_to_plot = []
    bert_recalls   = []
    nb_recalls     = []

    for label in LABEL_NAMES:
        sub = ml_op_df[
            (ml_op_df["label"]==label) &
            (ml_op_df["target_precision"]==0.90)
        ]
        if sub.empty: continue

        best_bert = 0.0
        for m in ["baseBERT","radBERT","bioClinBERT"]:
            row = sub[sub["model"]==m]
            if not row.empty and row.iloc[0]["achievable"]:
                best_bert = max(best_bert, float(row.iloc[0]["recall"]))

        nb_row = sub[sub["model"]=="BernoulliNB"]
        nb_rec = (float(nb_row.iloc[0]["recall"])
                  if (not nb_row.empty and nb_row.iloc[0]["achievable"])
                  else 0.0)

        labels_to_plot.append(label)
        bert_recalls.append(best_bert)
        nb_recalls.append(nb_rec)

    # Sort by best BERT recall descending
    order = sorted(range(len(labels_to_plot)),
                   key=lambda i: bert_recalls[i], reverse=True)
    labels_s    = [labels_to_plot[i] for i in order]
    bert_recalls_s = [bert_recalls[i] for i in order]
    nb_recalls_s   = [nb_recalls[i]   for i in order]

    x   = np.arange(len(labels_s))
    w   = 0.35
    ax3.bar(x - w/2, bert_recalls_s, w, label="Best BERT",
            color="#185FA5", alpha=0.85)
    ax3.bar(x + w/2, nb_recalls_s, w, label="BernoulliNB",
            color="#888780", alpha=0.85)
    ax3.axhline(0.0, color="black", lw=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_s, rotation=35, ha="right", fontsize=9)
    ax3.set_ylabel("Recall at 90% precision", fontsize=11)
    ax3.set_title(
        "Per-label recall at 90% precision operating point\n"
        "Height = fraction of true cases captured before the X-ray",
        fontsize=11
    )
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Threshold analysis for X-ray triage application\n"
        f"(Annual volume = {ANNUAL_VOLUME:,} orders)",
        fontsize=13, fontweight="bold", y=1.01
    )

    plot_path = os.path.join(diag_dir, "threshold_analysis.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {plot_path}")


# ══════════════════════════════════════════════════════════════════
#  POLICY SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
def policy_summary(op_df, ml_op_df):
    """
    Print a plain-language policy summary table — the kind that
    would appear in a health policy or radiology informatics paper.
    """
    print("\n" + "═"*70)
    print("  POLICY SUMMARY — Binary Task")
    print("═"*70)

    BERT_MODELS = ["baseBERT", "radBERT", "bioClinBERT"]
    auroc_by_model = (op_df[op_df["model"].isin(BERT_MODELS)]
                      .groupby("model")["auroc"].first())
    bert_best  = auroc_by_model.idxmax()
    best_auroc = auroc_by_model.max()

    sub = op_df[op_df["model"] == bert_best].copy()

    print(f"\n  Model: {bert_best}  (highest binary AUROC = {best_auroc:.3f})")
    print(f"\n  {'Precision':>10} {'Recall':>8} {'X-rays saved/yr':>17} "
          f"{'FP escalations/yr':>28}")

    print(f"  {'-'*68}")

    for _, row in sub[sub["achievable"]].iterrows():
        print(f"  {row['target_precision']:>10.0%} "
              f"{row['recall']:>8.3f} "
              f"{int(row['annual_xrays_saved']):>15,}   "
              f"{int(row['annual_fp_escalations']):>20,}")

    print("\n" + "═"*70)
    print("  POLICY SUMMARY — Multi-Label Task")
    print("  Labels where 90% precision is achievable with recall > 0.10")
    print("  (i.e., the model can reliably identify this finding type)")
    print("═"*70)

    bert_models = ["baseBERT","radBERT","bioClinBERT"]
    sub_ml = ml_op_df[
        (ml_op_df["target_precision"] == 0.90) &
        (ml_op_df["achievable"] == True) &
        (ml_op_df["recall"] > 0.10)
    ].copy()

    # Best BERT per label
    bert_sub = sub_ml[sub_ml["model"].isin(bert_models)]
    if not bert_sub.empty:
        best_bert_ml = (bert_sub.sort_values("recall", ascending=False)
                        .drop_duplicates("label"))
        print(f"\n  Best BERT model per label at ≥90% precision, recall > 10%:")
        print(f"  {'Label':<30} {'Model':<14} {'Recall':>7} "
              f"{'Ann. cases':>12} {'AUPRC':>7}")
        print(f"  {'-'*75}")
        for _, row in best_bert_ml.sort_values("recall", ascending=False).iterrows():
            print(f"  {row['label']:<30} {row['model']:<14} "
                  f"{row['recall']:>7.3f} "
                  f"{int(row['annual_cases_captured']):>10,}   "
                  f"{row['auprc']:>7.4f}")

    nb_sub = sub_ml[sub_ml["model"] == "BernoulliNB"]
    if not nb_sub.empty:
        print(f"\n  BernoulliNB at ≥90% precision, recall > 10%:")
        print(f"  {'Label':<30} {'Recall':>7} {'Ann. cases':>12} {'AUPRC':>7}")
        print(f"  {'-'*60}")
        for _, row in nb_sub.sort_values("recall", ascending=False).iterrows():
            print(f"  {row['label']:<30} {row['recall']:>7.3f} "
                  f"{int(row['annual_cases_captured']):>10,}   "
                  f"{row['auprc']:>7.4f}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("Threshold Analysis — X-ray Triage Application")
    print(f"PREDS_DIR   : {PREDS_DIR}")
    print(f"RESULTS_DIR : {RESULTS_DIR}")
    print(f"Annual vol  : {ANNUAL_VOLUME:,}")
    print(f"Target prec : {[f'{p:.0%}' for p in TARGET_PRECISIONS]}")

    print("\nLoading binary predictions...")
    binary_data = load_binary(BINARY_FILES)

    print("\nLoading multilabel predictions...")
    multilabel_data = load_multilabel(MULTILABEL_FILES)

    curve_df, op_df, savings_df = binary_pr_analysis(binary_data)

    ml_curve_df, ml_op_df = multilabel_pr_analysis(multilabel_data)

    print("\nGenerating plots...")
    make_plots(binary_data, curve_df, op_df,
               ml_op_df, RESULTS_DIR, DIAG_DIR)

    policy_summary(op_df, ml_op_df)

    print(f"\n\nAll outputs saved to: {RESULTS_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        size = os.path.getsize(os.path.join(RESULTS_DIR, f))
        print(f"  {f:<45} {size:>8,} bytes")


if __name__ == "__main__":
    main()