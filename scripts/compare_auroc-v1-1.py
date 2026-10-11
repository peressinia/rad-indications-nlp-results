#!/usr/bin/env python3
"""
Script: compare_auroc-v1-1.py

Version: 1.0 - Updated all .npy file manifests from v3-2x to v4-0 final run files.
    
Version: 1.1 added binary AUPRC (paired bootstrap, same
             resamples as AUROC) and multi-label micro-/macro-
             AUROC and AUPRC (paired permutation tests). The
             binary AUROC results are unchanged from v1.0.

Pairwise threshold-free comparisons for all 4 models in both tasks.


Binary task, for each model pair:
  - AUROC: DeLong test for two correlated AUROCs (same test set), using
    the fast algorithm of Sun & Xu (2014); two-sided p-value, then
    Benjamini-Hochberg correction across the 6 pairs.
  - AUROC and AUPRC: paired bootstrap (n = 2,000, same resample indices
    for both models) of the difference: percentile 95% CI and two-sided
    p-value. AUPRC p-values are BH-corrected across the 6 pairs.

Multi-label task, for each model pair:
  - Micro-/macro-AUROC and micro-/macro-AUPRC: two-sided paired
    permutation test (n = 5,000), swapping the two models' probability
    rows for a random half of the orders on each permutation, as in
    compare_models_final-v1-3.py. p = (count + 1) / (n + 1).
    BH correction across all 24 rows (6 pairs x 4 metrics). The 6 pairs
    run in parallel processes, each with its own fixed seed.

Inputs:  ./data_numpy/ binary and multi-label prediction arrays (same
         files as compare_models_final-v1-3.py)
Outputs: ./results_model_comparison/binary_auroc_comparison.csv
         ./results_model_comparison/binary_auprc_comparison.csv
         ./results_model_comparison/multilabel_auc_permutation_results.csv

Run from the repository root:
    python scripts/compare_auroc-v1-1.py | tee results_model_comparison/compare_auroc-v1-1.out
"""

import os
import time
import numpy as np
import pandas as pd
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from statsmodels.stats.multitest import multipletests

PREDS_DIR     = "./data_numpy"
RESULTS_DIR   = "./results_model_comparison"
N_BOOTSTRAP   = 2000
N_PERMUTATION = 5000
CI_LEVEL      = 95
RANDOM_STATE  = 42

np.random.seed(RANDOM_STATE)
os.makedirs(RESULTS_DIR, exist_ok=True)

N_LABELS = 13

BINARY_FILES = {
    "baseBERT": {
        "y_true" : "CNF-v4-0-baseBERT_bt32-ep3-lr0-bl1-tt0_pyV-1-2_y_true_binary.npy",
        "y_prob" : "CNF-v4-0-baseBERT_bt32-ep3-lr0-bl1-tt0_pyV-1-2_binary_probs.npy",
    },
    "radBERT": {
        "y_true" : "CNF-v4-0-radBERT_bt32-ep2-lr0-bl1-tt0_pyV-1-2_y_true_binary.npy",
        "y_prob" : "CNF-v4-0-radBERT_bt32-ep2-lr0-bl1-tt0_pyV-1-2_binary_probs.npy",
    },
    "bioClinBERT": {
        "y_true" : "CNF-v4-0-bioClinBERT_bt32-ep1-lr0-bl3-tt0_pyV-1-2_y_true_binary.npy",
        "y_prob" : "CNF-v4-0-bioClinBERT_bt32-ep1-lr0-bl3-tt0_pyV-1-2_binary_probs.npy",
    },
    "MultinomialNB": {
        "y_true" : "MultinomialNB-final-1500-NCF-v4-0-bal1_pyV-1-1_y_true_binary.npy",
        "y_prob" : "MultinomialNB-final-1500-NCF-v4-0-bal1_pyV-1-1_nb_binary_probs.npy",
    },
}

MULTILABEL_FILES = {
    "baseBERT": {
        "y_true" : "CNF-v4-0-baseBERT_bt32-ep6-lr0-bl0-tt0_pyV-1-1_y_true_multilabel.npy",
        "y_prob" : "CNF-v4-0-baseBERT_bt32-ep6-lr0-bl0-tt0_pyV-1-1_multilabel_probs.npy",
    },
    "radBERT": {
        "y_true" : "CNF-v4-0-radBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_y_true_multilabel.npy",
        "y_prob" : "CNF-v4-0-radBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_multilabel_probs.npy",
    },
    "bioClinBERT": {
        "y_true" : "CNF-v4-0-bioClinBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_y_true_multilabel.npy",
        "y_prob" : "CNF-v4-0-bioClinBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_multilabel_probs.npy",
    },
    "BernoulliNB": {
        "y_true" : "BernoulliNB-TFIDF-xSplits-1500-NCF-v4-0-bal0_y_true_multilabel.npy",
        "y_prob" : "BernoulliNB-TFIDF-xSplits-1500-NCF-v4-0-bal0_nb_multilabel_probs.npy",
    },
}

ML_METRICS = {
    "micro_roc_auc" : lambda yt, yp: roc_auc_score(yt, yp, average="micro"),
    "macro_roc_auc" : lambda yt, yp: roc_auc_score(yt, yp, average="macro"),
    "micro_auprc"   : lambda yt, yp: average_precision_score(yt, yp, average="micro"),
    "macro_auprc"   : lambda yt, yp: average_precision_score(yt, yp, average="macro"),
}


def section(title):
    print(f"\n{'═'*60}\n  {title}\n{'═'*60}", flush=True)


def load_npy(filename):
    path = os.path.join(PREDS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    arr = np.load(path, allow_pickle=True)
    print(f"    Loaded  {filename}  shape={arr.shape}  dtype={arr.dtype}")
    return arr


def load_binary():
    data = {}
    for model, files in BINARY_FILES.items():
        print(f"\n  [binary] Loading {model}:")
        y_true = load_npy(files["y_true"]).astype(int)
        y_prob = load_npy(files["y_prob"])
        if y_prob.ndim == 2:
            print(f"    -> y_prob shape (n,2): taking column 1 (positive class)")
            y_prob = y_prob[:, 1]
        data[model] = {"y_true": y_true, "y_prob": y_prob.astype(float)}
    return data


def load_multilabel():
    data = {}
    for model, files in MULTILABEL_FILES.items():
        print(f"\n  [multilabel] Loading {model}:")
        y_true = load_npy(files["y_true"]).astype(int)
        y_prob = load_npy(files["y_prob"])
        if y_prob.shape == (N_LABELS, y_true.shape[0]):
            print(f"    -> y_prob shape {y_prob.shape} is TRANSPOSED — fixing with .T")
            y_prob = y_prob.T
        if y_prob.shape != y_true.shape:
            raise ValueError(f"{model}: y_prob shape {y_prob.shape} != y_true shape {y_true.shape}")
        data[model] = {"y_true": y_true, "y_prob": y_prob.astype(float)}
    return data


def check_ytrue(data, task):
    models = list(data.keys())
    y_true = data[models[0]]["y_true"]
    for model in models[1:]:
        if not np.array_equal(data[model]["y_true"], y_true):
            raise ValueError(f"[{task}] y_true for {model} differs from {models[0]}")
    print(f"\n  ✓ [{task}] All y_true arrays are IDENTICAL — paired tests are valid.")
    print(f"  shape = {y_true.shape}  positives = {int(y_true.sum()):,}")
    return y_true


# ---------------------------------------------------------------------------
# DeLong test (Sun & Xu 2014, "Fast implementation of DeLong's algorithm")
# ---------------------------------------------------------------------------
def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_test(y_true, prob_a, prob_b):
    """Two-sided DeLong test for two correlated AUROCs. Returns
    (auc_a, auc_b, z, p, se_of_difference)."""
    pos = y_true == 1
    m, n = int(pos.sum()), int((~pos).sum())
    preds = np.vstack([prob_a, prob_b])
    aucs, v01, v10 = [], [], []
    for p in preds:
        x, y = p[pos], p[~pos]
        tx, ty, tz = _midrank(x), _midrank(y), _midrank(np.concatenate([x, y]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[:m] - tx) / n)          # structural components, positives
        v10.append(1.0 - (tz[m:] - ty) / m)    # structural components, negatives
    s01, s10 = np.cov(np.array(v01)), np.cov(np.array(v10))
    cov = s01 / m + s10 / n
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(var_diff))
    z = (aucs[0] - aucs[1]) / se
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p), se


# ---------------------------------------------------------------------------
# Paired bootstrap of the AUROC and AUPRC differences (binary)
# ---------------------------------------------------------------------------
def bootstrap_binary(y_true, data, n_boot=N_BOOTSTRAP):
    """AUROC and AUPRC of every model on the same n_boot resamples
    (paired). Draws the same indices as v1.0, so AUROC results match."""
    n = len(y_true)
    boot = {model: {"auroc": [], "auprc": []} for model in data}
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        yt = y_true[idx]
        for model, d in data.items():
            boot[model]["auroc"].append(roc_auc_score(yt, d["y_prob"][idx]))
            boot[model]["auprc"].append(average_precision_score(yt, d["y_prob"][idx]))
    return {model: {k: np.array(v) for k, v in b.items()} for model, b in boot.items()}


def bootstrap_diff(boot_a, boot_b, ci=CI_LEVEL):
    diff = boot_a - boot_b
    lo = np.percentile(diff, (100 - ci) / 2)
    hi = np.percentile(diff, 100 - (100 - ci) / 2)
    # two-sided: share of resamples on the far side of zero, doubled
    p = 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))
    return float(lo), float(hi), float(min(p, 1.0))


# ---------------------------------------------------------------------------
# Paired permutation test of the threshold-free metrics (multi-label)
# ---------------------------------------------------------------------------
def permutation_pair(args):
    """All 4 ML_METRICS for one model pair. Runs in a worker process."""
    name_a, name_b, y_true, prob_a, prob_b, seed, n_perm = args
    rng = np.random.default_rng(seed)
    observed = {k: fn(y_true, prob_a) - fn(y_true, prob_b) for k, fn in ML_METRICS.items()}
    counts = {k: 0 for k in ML_METRICS}
    n = len(y_true)
    for _ in range(n_perm):
        mask = rng.random(n) < 0.5
        perm_a = prob_a.copy(); perm_b = prob_b.copy()
        perm_a[mask], perm_b[mask] = prob_b[mask], prob_a[mask]
        for k, fn in ML_METRICS.items():
            if abs(fn(y_true, perm_a) - fn(y_true, perm_b)) >= abs(observed[k]):
                counts[k] += 1
    rows = []
    for k in ML_METRICS:
        rows.append({
            "comparison": f"{name_a} vs {name_b}", "metric": k,
            "value_A": round(ML_METRICS[k](y_true, prob_a), 4),
            "value_B": round(ML_METRICS[k](y_true, prob_b), 4),
            "delta_A_minus_B": round(float(observed[k]), 4),
            "perm_count": counts[k],
            "p_value": (counts[k] + 1) / (n_perm + 1),
        })
    return rows


def run_binary():
    section("LOADING — Binary Task")
    data = load_binary()
    y_true = check_ytrue(data, "binary")
    models = list(data.keys())

    section(f"BINARY BOOTSTRAP — {N_BOOTSTRAP:,} paired resamples")
    boot = bootstrap_binary(y_true, data)
    print("  Done.")

    section("BINARY — PAIRWISE AUROC COMPARISON")
    rows = []
    for a, b in combinations(models, 2):
        auc_a, auc_b, z, p, se = delong_test(y_true, data[a]["y_prob"], data[b]["y_prob"])
        lo, hi, p_boot = bootstrap_diff(boot[a]["auroc"], boot[b]["auroc"])
        print(f"    {a} vs {b}:  AUROC {auc_a:.4f} vs {auc_b:.4f}  "
              f"diff={auc_a - auc_b:+.4f}  DeLong z={z:.3f} p={p:.4g}  "
              f"boot 95% CI [{lo:+.4f}, {hi:+.4f}]  boot p={p_boot:.4f}")
        rows.append({
            "comparison": f"{a} vs {b}",
            "auroc_A": round(auc_a, 4), "auroc_B": round(auc_b, 4),
            "delta_A_minus_B": round(auc_a - auc_b, 4),
            "delong_se": round(se, 5), "delong_z": round(z, 4),
            "delong_p": p,
            "boot_ci_lower": round(lo, 4), "boot_ci_upper": round(hi, 4),
            "boot_p": round(p_boot, 4),
        })
    df = pd.DataFrame(rows)
    reject, p_bh, _, _ = multipletests(df["delong_p"].values, method="fdr_bh")
    df["delong_p_BH"]    = p_bh
    df["significant_BH"] = reject
    df["delong_p"]    = df["delong_p"].map(lambda v: float(f"{v:.4g}"))
    df["delong_p_BH"] = df["delong_p_BH"].map(lambda v: float(f"{v:.4g}"))
    df.to_csv(f"{RESULTS_DIR}/binary_auroc_comparison.csv", index=False)

    section("SUMMARY — Binary AUROC, DeLong (BH-corrected)")
    print(df[["comparison", "delta_A_minus_B", "delong_p", "delong_p_BH",
              "significant_BH", "boot_ci_lower", "boot_ci_upper"]].to_string(index=False))
    print(f"\n  Saved: {RESULTS_DIR}/binary_auroc_comparison.csv")

    section("BINARY — PAIRWISE AUPRC COMPARISON (paired bootstrap)")
    rows = []
    for a, b in combinations(models, 2):
        ap_a = average_precision_score(y_true, data[a]["y_prob"])
        ap_b = average_precision_score(y_true, data[b]["y_prob"])
        lo, hi, p_boot = bootstrap_diff(boot[a]["auprc"], boot[b]["auprc"])
        print(f"    {a} vs {b}:  AUPRC {ap_a:.4f} vs {ap_b:.4f}  diff={ap_a - ap_b:+.4f}  "
              f"boot 95% CI [{lo:+.4f}, {hi:+.4f}]  boot p={p_boot:.4f}")
        rows.append({
            "comparison": f"{a} vs {b}",
            "auprc_A": round(ap_a, 4), "auprc_B": round(ap_b, 4),
            "delta_A_minus_B": round(ap_a - ap_b, 4),
            "boot_ci_lower": round(lo, 4), "boot_ci_upper": round(hi, 4),
            "boot_p": round(p_boot, 4),
        })
    df = pd.DataFrame(rows)
    reject, p_bh, _, _ = multipletests(df["boot_p"].values, method="fdr_bh")
    df["boot_p_BH"]      = np.round(p_bh, 4)
    df["significant_BH"] = reject
    df.to_csv(f"{RESULTS_DIR}/binary_auprc_comparison.csv", index=False)

    section("SUMMARY — Binary AUPRC, paired bootstrap (BH-corrected)")
    print(df[["comparison", "delta_A_minus_B", "boot_p", "boot_p_BH",
              "significant_BH", "boot_ci_lower", "boot_ci_upper"]].to_string(index=False))
    print(f"\n  Note: bootstrap p = 0 means no resample crossed zero (p < 1/{N_BOOTSTRAP:,}).")
    print(f"  Saved: {RESULTS_DIR}/binary_auprc_comparison.csv")


def run_multilabel():
    section("LOADING — Multi-label Task")
    data = load_multilabel()
    y_true = check_ytrue(data, "multilabel")
    models = list(data.keys())
    pairs = list(combinations(models, 2))

    section(f"MULTI-LABEL PERMUTATION TESTS — {N_PERMUTATION:,} permutations x {len(pairs)} pairs")
    print(f"  Running pairs in parallel ({min(len(pairs), os.cpu_count() or 1)} processes)...", flush=True)
    jobs = [(a, b, y_true, data[a]["y_prob"], data[b]["y_prob"], RANDOM_STATE + i, N_PERMUTATION)
            for i, (a, b) in enumerate(pairs)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=min(len(pairs), os.cpu_count() or 1)) as pool:
        for pair_rows in pool.map(permutation_pair, jobs):
            print(f"    done: {pair_rows[0]['comparison']}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            rows += pair_rows

    df = pd.DataFrame(rows)
    reject, p_bh, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
    df["p_value"]        = df["p_value"].round(4)
    df["p_value_BH"]     = np.round(p_bh, 4)
    df["significant_BH"] = reject
    df.to_csv(f"{RESULTS_DIR}/multilabel_auc_permutation_results.csv", index=False)

    section("SUMMARY — Multi-label AUROC/AUPRC, permutation (BH-corrected)")
    print(df[["comparison", "metric", "value_A", "value_B", "delta_A_minus_B",
              "p_value", "p_value_BH", "significant_BH"]].to_string(index=False))
    print(f"\n  Note: smallest attainable p = 1/{N_PERMUTATION + 1:,} = {1 / (N_PERMUTATION + 1):.4f}.")
    print(f"  Saved: {RESULTS_DIR}/multilabel_auc_permutation_results.csv\n")


def main():
    run_binary()
    run_multilabel()


if __name__ == "__main__":
    main()
