#!/usr/bin/env python3
"""
Script: compare_models_final-v1-3.py
Version: 1.3

Changes from v1.2:
  - Updated all .npy file manifests from v3-2x to v4-0 final run files.
    Binary:    baseBERT ep3/bl1, radBERT ep2/bl1, bioClinBERT ep1/bl3, MultinomialNB bal1
    Multilabel: all BERT ep5-6/bl0, BernoulliNB bal0

Pairwise accuracy test (at threshold 0.5) using McNemar's tests.

Inputs:  ./data_numpy/ binary prediction arrays 
Outputs: ./results_model_comparison/binary_mcnemar_results.csv
         ./results_model_comparison/binary_bootstrap_ci.csv
         ./results_model_comparison/multilabel_bootstrap_ci.csv
         ./results_model_comparison/multilabel_permutation_results.csv

Run from the repository root:
    python scripts/compare_models_final-v1-3.py | tee results_model_comparison/compare_models_final-v1-3.out


"""

import os
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, hamming_loss
)

PREDS_DIR       = "./data_numpy"
RESULTS_DIR     = "./results_model_comparison"
RESULTS_DIR_CSV = "./data_csv"
N_BOOTSTRAP   = 2000
N_PERMUTATION = 5000
CI_LEVEL      = 95
RANDOM_STATE  = 42

LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax"
]

np.random.seed(RANDOM_STATE)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/csv", exist_ok=True)

BINARY_FILES = {
    "baseBERT": {
        "y_true" : "CNF-v4-0-baseBERT_bt32-ep3-lr0-bl1-tt0_pyV-1-2_y_true_binary.npy",
        "y_pred" : "CNF-v4-0-baseBERT_bt32-ep3-lr0-bl1-tt0_pyV-1-2_binary_preds.npy",
        "y_prob" : "CNF-v4-0-baseBERT_bt32-ep3-lr0-bl1-tt0_pyV-1-2_binary_probs.npy",
    },
    "radBERT": {
        "y_true" : "CNF-v4-0-radBERT_bt32-ep2-lr0-bl1-tt0_pyV-1-2_y_true_binary.npy",
        "y_pred" : "CNF-v4-0-radBERT_bt32-ep2-lr0-bl1-tt0_pyV-1-2_binary_preds.npy",
        "y_prob" : "CNF-v4-0-radBERT_bt32-ep2-lr0-bl1-tt0_pyV-1-2_binary_probs.npy",
    },
    "bioClinBERT": {
        "y_true" : "CNF-v4-0-bioClinBERT_bt32-ep1-lr0-bl3-tt0_pyV-1-2_y_true_binary.npy",
        "y_pred" : "CNF-v4-0-bioClinBERT_bt32-ep1-lr0-bl3-tt0_pyV-1-2_binary_preds.npy",
        "y_prob" : "CNF-v4-0-bioClinBERT_bt32-ep1-lr0-bl3-tt0_pyV-1-2_binary_probs.npy",
    },
    "MultinomialNB": {
        "y_true" : "MultinomialNB-final-1500-NCF-v4-0-bal1_pyV-1-1_y_true_binary.npy",
        "y_pred" : "MultinomialNB-final-1500-NCF-v4-0-bal1_pyV-1-1_nb_binary_preds.npy",
        "y_prob" : "MultinomialNB-final-1500-NCF-v4-0-bal1_pyV-1-1_nb_binary_probs.npy",
    },
}

MULTILABEL_FILES = {
    "baseBERT": {
        "y_true" : "CNF-v4-0-baseBERT_bt32-ep6-lr0-bl0-tt0_pyV-1-1_y_true_multilabel.npy",
        "y_pred" : "CNF-v4-0-baseBERT_bt32-ep6-lr0-bl0-tt0_pyV-1-1_multilabel_preds.npy",
        "y_prob" : "CNF-v4-0-baseBERT_bt32-ep6-lr0-bl0-tt0_pyV-1-1_multilabel_probs.npy",
    },
    "radBERT": {
        "y_true" : "CNF-v4-0-radBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_y_true_multilabel.npy",
        "y_pred" : "CNF-v4-0-radBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_multilabel_preds.npy",
        "y_prob" : "CNF-v4-0-radBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_multilabel_probs.npy",
    },
    "bioClinBERT": {
        "y_true" : "CNF-v4-0-bioClinBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_y_true_multilabel.npy",
        "y_pred" : "CNF-v4-0-bioClinBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_multilabel_preds.npy",
        "y_prob" : "CNF-v4-0-bioClinBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_multilabel_probs.npy",
    },
    "BernoulliNB": {
        "y_true" : "BernoulliNB-TFIDF-xSplits-1500-NCF-v4-0-bal0_y_true_multilabel.npy",
        "y_pred" : "BernoulliNB-TFIDF-xSplits-1500-NCF-v4-0-bal0_nb_multilabel_preds.npy",
        "y_prob" : "BernoulliNB-TFIDF-xSplits-1500-NCF-v4-0-bal0_nb_multilabel_probs.npy",
    },
}


def load_npy(filename):
    path = os.path.join(PREDS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    arr = np.load(path, allow_pickle=True)
    print(f"    Loaded  {filename}  shape={arr.shape}  dtype={arr.dtype}")
    return arr


def load_model_files(file_dict, task):
    data = {}
    for model, files in file_dict.items():
        print(f"\n  [{task}] Loading {model}:")
        y_true = load_npy(files["y_true"])
        y_pred = load_npy(files["y_pred"])
        y_prob = load_npy(files["y_prob"])

        if task == "binary" and y_prob.ndim == 2:
            print(f"    -> y_prob shape (n,2): taking column 1 (positive class)")
            y_prob = y_prob[:, 1]

        if task == "multilabel" and hasattr(y_pred, "toarray"):
            y_pred = y_pred.toarray()

        if task == "multilabel" and isinstance(y_prob, list):
            print(f"    -> y_prob is list of arrays: stacking col 1 of each")
            y_prob = np.column_stack([a[:, 1] for a in y_prob])

        if task == "multilabel" and y_prob.ndim == 2:
            n_labels = len(LABEL_NAMES)
            if y_prob.shape == (n_labels, y_true.shape[0]):
                print(f"    -> y_prob shape {y_prob.shape} is TRANSPOSED — fixing with .T")
                y_prob = y_prob.T
            print(f"    -> y_prob final shape: {y_prob.shape}")

        data[model] = {
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "y_prob": y_prob.astype(float),
        }
    return data


def sanity_check_ytrue(data, task):
    print(f"\n{'─'*60}")
    print(f"  SANITY CHECK — y_true consistency [{task}]")
    print(f"{'─'*60}")
    models    = list(data.keys())
    ref_model = models[0]
    ref_ytrue = data[ref_model]["y_true"]
    all_match = True
    print(f"  Reference model: {ref_model}  shape={ref_ytrue.shape}")
    for model in models[1:]:
        ytrue = data[model]["y_true"]
        if ytrue.shape != ref_ytrue.shape:
            print(f"  ✗ {model}: SHAPE MISMATCH  {ytrue.shape} vs {ref_ytrue.shape}")
            all_match = False
            continue
        n_diff = int(np.sum(ytrue != ref_ytrue))
        pct    = 100.0 * n_diff / ytrue.size
        symbol = "✓" if n_diff == 0 else "✗"
        print(f"  {symbol} {model}: {n_diff} differences ({pct:.2f}% of {ytrue.size:,} elements)")
        if n_diff != 0:
            all_match = False
    if all_match:
        print(f"\n  ✓ All y_true arrays are IDENTICAL — row-by-row tests are valid.")
    else:
        print(f"\n  ✗ WARNING: y_true arrays differ. Bootstrap CIs remain valid.")
    return all_match


def export_csvs(binary_data, multilabel_data):
    os.makedirs(RESULTS_DIR_CSV, exist_ok=True)
    for model, d in binary_data.items():
        stub = f"{RESULTS_DIR_CSV}/binary_{model}"
        pd.DataFrame({"y_true": d["y_true"], "y_pred": d["y_pred"],
                      "y_prob": d["y_prob"]}).to_csv(f"{stub}_predictions.csv", index=False)
        print(f"  Saved: binary_{model}_predictions.csv")
    for model, d in multilabel_data.items():
        stub = f"{RESULTS_DIR_CSV}/multilabel_{model}"
        pd.DataFrame(d["y_true"], columns=LABEL_NAMES).to_csv(f"{stub}_y_true.csv", index=False)
        pd.DataFrame(d["y_pred"], columns=LABEL_NAMES).to_csv(f"{stub}_y_pred.csv", index=False)
        pd.DataFrame(d["y_prob"], columns=LABEL_NAMES).to_csv(f"{stub}_y_prob.csv", index=False)
        print(f"  Saved: multilabel_{model}_y_true/pred/prob.csv")


def run_mcnemar(y_true, pred_a, pred_b, name_a, name_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum( correct_a & ~correct_b))
    c = int(np.sum(~correct_a &  correct_b))
    exact  = (b + c) < 25
    table  = [[0, b], [c, 0]]
    result = mcnemar(table, exact=exact, correction=not exact)
    print(f"    {name_a} vs {name_b}:  b={b}  c={c}  stat={result.statistic:.4f}  "
          f"p={result.pvalue:.4f}  {'*' if result.pvalue < 0.05 else 'ns'}")
    return {"comparison": f"{name_a} vs {name_b}", "b_Acorr_Bwrong": b,
            "c_Awrong_Bcorr": c, "statistic": round(result.statistic, 4),
            "p_value": round(result.pvalue, 4),
            "method": "exact" if exact else "chi-sq+continuity",
            "significant_p05": result.pvalue < 0.05}


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    n, scores = len(y_true), []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    if not scores:
        return np.nan, np.nan, np.nan
    point = metric_fn(y_true, y_pred)
    lo    = np.percentile(scores, (100 - ci) / 2)
    hi    = np.percentile(scores, 100 - (100 - ci) / 2)
    return round(float(point), 4), round(float(lo), 4), round(float(hi), 4)


def _specificity(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def bootstrap_binary(y_true, y_pred, y_prob, model_name):
    metrics = {
        "accuracy"           : (y_pred, lambda yt, yp: accuracy_score(yt, yp)),
        "precision"          : (y_pred, lambda yt, yp: precision_score(yt, yp, zero_division=0)),
        "recall_sensitivity" : (y_pred, lambda yt, yp: recall_score(yt, yp, zero_division=0)),
        "specificity"        : (y_pred, lambda yt, yp: _specificity(yt, yp)),
        "f1"                 : (y_pred, lambda yt, yp: f1_score(yt, yp, zero_division=0)),
        "roc_auc"            : (y_prob, lambda yt, yp: roc_auc_score(yt, yp)),
        "auprc"              : (y_prob, lambda yt, yp: average_precision_score(yt, yp)),
    }
    rows = []
    for metric_name, (y_input, fn) in metrics.items():
        pt, lo, hi = bootstrap_ci(y_true, y_input, fn)
        rows.append({"model": model_name, "metric": metric_name, "value": pt,
                     "ci_lower": lo, "ci_upper": hi, "ci_95": f"[{lo:.4f}, {hi:.4f}]"})
    return rows


def bootstrap_multilabel(y_true, y_pred, y_prob, model_name):
    agg = {
        "hamming_loss"   : (y_pred, lambda yt, yp: hamming_loss(yt, yp)),
        "subset_accuracy": (y_pred, lambda yt, yp: accuracy_score(yt, yp)),
        "micro_f1"       : (y_pred, lambda yt, yp: f1_score(yt, yp, average="micro",    zero_division=0)),
        "macro_f1"       : (y_pred, lambda yt, yp: f1_score(yt, yp, average="macro",    zero_division=0)),
        "weighted_f1"    : (y_pred, lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0)),
        "micro_roc_auc"  : (y_prob, lambda yt, yp: roc_auc_score(yt, yp, average="micro")),
        "macro_roc_auc"  : (y_prob, lambda yt, yp: roc_auc_score(yt, yp, average="macro")),
        "micro_auprc"    : (y_prob, lambda yt, yp: average_precision_score(yt, yp, average="micro")),
        "macro_auprc"    : (y_prob, lambda yt, yp: average_precision_score(yt, yp, average="macro")),
    }
    rows = []
    for metric_name, (y_input, fn) in agg.items():
        pt, lo, hi = bootstrap_ci(y_true, y_input, fn)
        rows.append({"model": model_name, "label": "AGGREGATE", "metric": metric_name,
                     "value": pt, "ci_lower": lo, "ci_upper": hi,
                     "ci_95": f"[{lo:.4f}, {hi:.4f}]"})
    for i, lname in enumerate(LABEL_NAMES):
        fn = lambda yt, yp, col=i: f1_score(yt[:, col], yp[:, col], zero_division=0)
        pt, lo, hi = bootstrap_ci(y_true, y_pred, fn)
        rows.append({"model": model_name, "label": lname, "metric": "f1",
                     "value": pt, "ci_lower": lo, "ci_upper": hi,
                     "ci_95": f"[{lo:.4f}, {hi:.4f}]"})
    return rows


def permutation_test(y_true, pred_a, pred_b, metric_fn, n_perm=N_PERMUTATION):
    observed = metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)
    n, count = len(y_true), 0
    for _ in range(n_perm):
        mask   = np.random.rand(n) < 0.5
        perm_a = pred_a.copy(); perm_b = pred_b.copy()
        perm_a[mask], perm_b[mask] = pred_b[mask].copy(), pred_a[mask].copy()
        if abs(metric_fn(y_true, perm_a) - metric_fn(y_true, perm_b)) >= abs(observed):
            count += 1
    return round(float(observed), 4), round(count / n_perm, 4)


def run_permutation_multilabel(y_true, pred_a, pred_b, name_a, name_b):
    metrics = {
        "micro_f1"    : lambda yt, yp: f1_score(yt, yp, average="micro",    zero_division=0),
        "macro_f1"    : lambda yt, yp: f1_score(yt, yp, average="macro",    zero_division=0),
        "weighted_f1" : lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0),
        "hamming_loss": lambda yt, yp: -hamming_loss(yt, yp),
    }
    rows = []
    for metric_name, fn in metrics.items():
        delta, p = permutation_test(y_true, pred_a, pred_b, fn)
        print(f"    {'✓' if p < 0.05 else '✗'} {name_a} vs {name_b}  [{metric_name}]:  "
              f"Δ={delta:+.4f}  p={p:.4f}")
        rows.append({"comparison": f"{name_a} vs {name_b}", "metric": metric_name,
                     "delta_A_minus_B": delta, "p_value": p, "significant_p05": p < 0.05})
    return rows


def section(title):
    print(f"\n{'═'*60}\n  {title}\n{'═'*60}")


def main():
    section("LOADING — Binary Task")
    binary_data = load_model_files(BINARY_FILES, task="binary")

    section("LOADING — Multi-Label Task")
    multilabel_data = load_model_files(MULTILABEL_FILES, task="multilabel")

    section("SANITY CHECK — y_true Consistency")
    binary_aligned     = sanity_check_ytrue(binary_data,     task="binary")
    multilabel_aligned = sanity_check_ytrue(multilabel_data, task="multilabel")

    section("EXPORTING CSVs")
    export_csvs(binary_data, multilabel_data)

    section("BINARY — McNemar's Tests")
    mcnemar_rows = []
    model_names  = list(binary_data.keys())
    if not binary_aligned:
        print("  WARNING: y_true arrays differ — McNemar results may be invalid\n")
    for m1, m2 in combinations(model_names, 2):
        y_true = binary_data[model_names[0]]["y_true"]
        mcnemar_rows.append(run_mcnemar(y_true,
            binary_data[m1]["y_pred"], binary_data[m2]["y_pred"], m1, m2))
    mcnemar_df = pd.DataFrame(mcnemar_rows)
    reject_bh, p_bh, _, _ = multipletests(mcnemar_df["p_value"].values, method="fdr_bh")
    mcnemar_df["p_value_BH"]     = np.round(p_bh, 4)
    mcnemar_df["significant_BH"] = reject_bh
    mcnemar_df.to_csv(f"{RESULTS_DIR}/binary_mcnemar_results.csv", index=False)
    print(f"\n  Saved: binary_mcnemar_results.csv")

    section("BINARY — Bootstrap Confidence Intervals")
    binary_ci_rows = []
    for model, d in binary_data.items():
        print(f"\n  {model}:")
        binary_ci_rows += bootstrap_binary(d["y_true"], d["y_pred"], d["y_prob"], model)
    binary_ci_df = pd.DataFrame(binary_ci_rows)
    binary_ci_df.to_csv(f"{RESULTS_DIR}/binary_bootstrap_ci.csv", index=False)
    print(f"\n  Saved: binary_bootstrap_ci.csv")

    section("MULTI-LABEL — Bootstrap Confidence Intervals")
    ml_ci_rows = []
    for model, d in multilabel_data.items():
        print(f"\n  {model}:")
        ml_ci_rows += bootstrap_multilabel(d["y_true"], d["y_pred"], d["y_prob"], model)
    ml_ci_df = pd.DataFrame(ml_ci_rows)
    ml_ci_df.to_csv(f"{RESULTS_DIR}/multilabel_bootstrap_ci.csv", index=False)
    print(f"\n  Saved: multilabel_bootstrap_ci.csv")

    section("MULTI-LABEL — Paired Permutation Tests")
    perm_rows  = []
    ml_names   = list(multilabel_data.keys())
    ref_y_true = multilabel_data[ml_names[0]]["y_true"]
    if not multilabel_aligned:
        print("  WARNING: y_true arrays differ — permutation results may be invalid.\n")
    for m1, m2 in combinations(ml_names, 2):
        perm_rows += run_permutation_multilabel(
            ref_y_true, multilabel_data[m1]["y_pred"], multilabel_data[m2]["y_pred"], m1, m2)
    perm_df = pd.DataFrame(perm_rows)
    reject_bh, p_bh, _, _ = multipletests(perm_df["p_value"].values, method="fdr_bh")
    perm_df["p_value_BH"]     = np.round(p_bh, 4)
    perm_df["significant_BH"] = reject_bh
    perm_df.to_csv(f"{RESULTS_DIR}/multilabel_permutation_results.csv", index=False)
    print(f"\n  Saved: multilabel_permutation_results.csv")

    section("SUMMARY — Binary Bootstrap CIs")
    key_metrics = ["roc_auc", "auprc", "f1", "accuracy", "precision",
                   "recall_sensitivity", "specificity"]
    summary_bin = binary_ci_df[binary_ci_df["metric"].isin(key_metrics)].copy()
    metric_order = {m: i for i, m in enumerate(key_metrics)}
    summary_bin["_order"] = summary_bin["metric"].map(metric_order)
    summary_bin = summary_bin.sort_values(["model", "_order"]).drop(columns="_order")
    print(summary_bin[["model", "metric", "value", "ci_95"]].to_string(index=False))

    section("SUMMARY — Multi-Label Aggregate Bootstrap CIs")
    agg_df = ml_ci_df[ml_ci_df["label"] == "AGGREGATE"].copy()
    ml_metric_order = {m: i for i, m in enumerate([
        "micro_f1", "macro_f1", "weighted_f1", "hamming_loss", "subset_accuracy",
        "micro_roc_auc", "macro_roc_auc", "micro_auprc", "macro_auprc"])}
    agg_df["_order"] = agg_df["metric"].map(ml_metric_order)
    agg_df = agg_df.sort_values(["model", "_order"]).drop(columns="_order")
    print(agg_df[["model", "metric", "value", "ci_95"]].to_string(index=False))

    section("SUMMARY — McNemar + Permutation Tests")
    print(mcnemar_df[["comparison", "statistic", "p_value", "p_value_BH",
                       "significant_BH"]].to_string(index=False))
    print()
    print(perm_df[["comparison", "metric", "delta_A_minus_B",
                   "p_value", "p_value_BH", "significant_BH"]].to_string(index=False))
    print(f"\n\nAll results saved to: {RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()
