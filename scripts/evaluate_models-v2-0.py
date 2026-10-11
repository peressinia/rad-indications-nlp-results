#!/usr/bin/env python3
"""
Final evaluation and statistical comparison script (v2.0).

CLEAN ASCII VERSION – SAFE COPY/PASTE
------------------------------------
This script:
1. Separates binary vs multilabel ground truth
2. Verifies y_true consistency within each task
3. Computes metrics
4. Runs statistical significance tests
   - Binary: McNemar test
   - Multilabel: paired permutation test (micro-F1)
5. Generates ROC and Precision–Recall curves for binary models

Outputs:
- results_final/binary_metrics.csv
- results_final/multilabel_metrics.csv
- results_final/binary_pairwise_stats.csv
- results_final/multilabel_pairwise_stats.csv
- diagnostic_figures/roc_binary.png
- diagnostic_figures/pr_binary.png
"""

import os
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    hamming_loss,
)
from statsmodels.stats.contingency_tables import mcnemar

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ARTIFACT_DIR = "./data_numpy/"
RESULTS_DIR = "./results_final/"
FIG_DIR = "./diagnostic_figures"

N_PERMUTATION = 5000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Ground truth sanity checks
# ---------------------------------------------------------------------

binary_y_true_files = [f for f in os.listdir(ARTIFACT_DIR) if f.endswith("_y_true_binary.npy")]
multilabel_y_true_files = [f for f in os.listdir(ARTIFACT_DIR) if f.endswith("_y_true_multilabel.npy")]

# Binary y_true
y_true_binary = None
if binary_y_true_files:
    ref = np.load(os.path.join(ARTIFACT_DIR, binary_y_true_files[0]))
    for f in binary_y_true_files:
        if not np.array_equal(ref, np.load(os.path.join(ARTIFACT_DIR, f))):
            raise AssertionError("Binary y_true mismatch: " + f)
    y_true_binary = ref

# Multilabel y_true
y_true_multi = None
if multilabel_y_true_files:
    ref = np.load(os.path.join(ARTIFACT_DIR, multilabel_y_true_files[0]))
    for f in multilabel_y_true_files:
        if not np.array_equal(ref, np.load(os.path.join(ARTIFACT_DIR, f))):
            raise AssertionError("Multilabel y_true mismatch: " + f)
    y_true_multi = ref

# ---------------------------------------------------------------------
# 2. Load predictions
# ---------------------------------------------------------------------

binary_preds = {}
binary_probs = {}
multi_preds = {}

for fname in os.listdir(ARTIFACT_DIR):
    path = os.path.join(ARTIFACT_DIR, fname)
    if not fname.endswith('.npy'):
        continue

    if fname.endswith('_binary_preds.npy'):
        binary_preds[fname.replace('_binary_preds.npy', '')] = np.load(path)

    elif fname.endswith('_binary_probs.npy'):
        binary_probs[fname.replace('_binary_probs.npy', '')] = np.load(path)

    elif fname.endswith('_multilabel_preds.npy'):
        multi_preds[fname.replace('_multilabel_preds.npy', '')] = np.load(path)

# ---------------------------------------------------------------------
# 3. Binary metrics + ROC/PR plots
# ---------------------------------------------------------------------

binary_rows = []

if y_true_binary is not None:
    # ROC
    plt.figure(figsize=(7, 7))
    for model, y_pred in binary_preds.items():
        y_prob = binary_probs.get(model)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]

        fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
        roc_auc = roc_auc_score(y_true_binary, y_prob)
        plt.plot(fpr, tpr, label=f"{model} (AUROC={roc_auc:.3f})")

        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred).ravel()
        binary_rows.append({
            "Model": model,
            "Accuracy": accuracy_score(y_true_binary, y_pred),
            "Precision": precision_score(y_true_binary, y_pred),
            "Recall": recall_score(y_true_binary, y_pred),
            "Specificity": tn / (tn + fp),
            "F1": f1_score(y_true_binary, y_pred),
            "AUROC": roc_auc,
            "AUPRC": average_precision_score(y_true_binary, y_prob),
        })

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves – Binary Classification')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'roc_binary.png'), dpi=300)
    plt.close()

    # Precision–Recall
    plt.figure(figsize=(7, 7))
    for model, y_pred in binary_preds.items():
        y_prob = binary_probs.get(model)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]

        prec, rec, _ = precision_recall_curve(y_true_binary, y_prob)
        pr_auc = average_precision_score(y_true_binary, y_prob)
        plt.plot(rec, prec, label=f"{model} (AUPRC={pr_auc:.3f})")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curves – Binary Classification')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'pr_binary.png'), dpi=300)
    plt.close()

    pd.DataFrame(binary_rows).to_csv(os.path.join(RESULTS_DIR, 'binary_metrics.csv'), index=False)

# ---------------------------------------------------------------------
# 4. Binary statistical tests (McNemar)
# ---------------------------------------------------------------------

binary_stats = []
for m1, m2 in combinations(binary_preds.keys(), 2):
    y1 = binary_preds[m1]
    y2 = binary_preds[m2]

    table = [[0, 0], [0, 0]]
    for yt, p1, p2 in zip(y_true_binary, y1, y2):
        table[int(p1 == yt)][int(p2 == yt)] += 1

    res = mcnemar(table, exact=False, correction=True)
    binary_stats.append({
        "Model_A": m1,
        "Model_B": m2,
        "McNemar_chi2": res.statistic,
        "p_value": res.pvalue,
    })

pd.DataFrame(binary_stats).to_csv(os.path.join(RESULTS_DIR, 'binary_pairwise_stats.csv'), index=False)

# ---------------------------------------------------------------------
# 5. Multilabel metrics + permutation test
# ---------------------------------------------------------------------

multi_rows = []
for model, y_pred in multi_preds.items():
    multi_rows.append({
        "Model": model,
        "Micro-F1": f1_score(y_true_multi, y_pred, average='micro'),
        "Macro-F1": f1_score(y_true_multi, y_pred, average='macro'),
        "Weighted-F1": f1_score(y_true_multi, y_pred, average='weighted'),
        "Hamming-Loss": hamming_loss(y_true_multi, y_pred),
    })

pd.DataFrame(multi_rows).to_csv(os.path.join(RESULTS_DIR, 'multilabel_metrics.csv'), index=False)

# Permutation test for micro-F1
def permutation_test(y_true, a, b, n_perm=N_PERMUTATION):
    obs = f1_score(y_true, a, average='micro') - f1_score(y_true, b, average='micro')
    diffs = []
    for _ in range(n_perm):
        mask = np.random.rand(len(y_true)) < 0.5
        a_p = a.copy()
        b_p = b.copy()
        a_p[mask], b_p[mask] = b[mask], a[mask]
        diffs.append(f1_score(y_true, a_p, average='micro') - f1_score(y_true, b_p, average='micro'))
    diffs = np.array(diffs)
    p_val = np.mean(np.abs(diffs) >= np.abs(obs))
    return obs, p_val

multi_stats = []
for m1, m2 in combinations(multi_preds.keys(), 2):
    obs, p = permutation_test(y_true_multi, multi_preds[m1], multi_preds[m2])
    multi_stats.append({
        "Model_A": m1,
        "Model_B": m2,
        "Delta_Micro-F1": obs,
        "p_value": p,
    })

pd.DataFrame(multi_stats).to_csv(os.path.join(RESULTS_DIR, 'multilabel_pairwise_stats.csv'), index=False)

print("Evaluation, statistics, and plots completed successfully")