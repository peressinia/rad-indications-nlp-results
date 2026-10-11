#!/usr/bin/env python3
"""
Script: label_prevalence_report-v1-1.py
Version: 1.0 [07 Jul 26]
Version: 1.1 [10 Oct 26] test-set labels now read from the committed
                         data_csv/ y_true file instead of a study-ID list;
                         training-set counts derived as full minus test.
                         Outputs now written to results_label_report/
                         with the -v{REASON_FILE_VERSION} suffix.

Standalone script to report CheXpert label prevalence for the full
dataset, the training set, and the persistent test set, and produce a
grouped bar chart and CSV output. No command line arguments needed.
Run from the repository root.

Extracted and adapted from model-BERT-multi-eval-v1-3.py.

Inputs:
    MIMIC_DIR/mimic-reasons-CNF-v4-0.csv     (not distributed; MIMIC-derived)
    MIMIC_DIR/mimic-cxr-2.0.0-chexpert.csv   (not distributed; PhysioNet)
    ./data_csv/multilabel_baseBERT_y_true.csv (test-set ground truth)

Outputs:
    ./results_label_report/label_prevalence_report-v4-0.csv
    ./results_label_report/label_prevalence_report-v4-0.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration — edit here for different runs
# ---------------------------------------------------------------------------
REASON_FILE_VERSION  = "4-0"
REASON_FILTER        = "CNF"

MIMIC_DIR         = "./get-reasons"   # your local MIMIC-derived files
REASON_FILE_NAME  = f"{MIMIC_DIR}/mimic-reasons-{REASON_FILTER}-v{REASON_FILE_VERSION}.csv"
LABEL_FILE_NAME_CHEX = f"{MIMIC_DIR}/mimic-cxr-2.0.0-chexpert.csv"
# y_true is identical across models (verified by compare_models_final-v1-3.py)
TEST_Y_TRUE_FILE  = "./data_csv/multilabel_baseBERT_y_true.csv"
OUT_DIR           = "./results_label_report"
OUT_CSV           = f"{OUT_DIR}/label_prevalence_report-v{REASON_FILE_VERSION}.csv"
OUT_PNG           = f"{OUT_DIR}/label_prevalence_report-v{REASON_FILE_VERSION}.png"

NUM_OF_LABELS = 13   # after dropping Support Devices

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load reason data
# ---------------------------------------------------------------------------
print(f"Loading reason file: {REASON_FILE_NAME}")
theData = pd.read_csv(REASON_FILE_NAME)
theData["reason_for_exam"] = theData["reason_for_exam"].fillna("").astype(str)
print(f"Total reasons loaded: {len(theData):,}")

# ---------------------------------------------------------------------------
# Load and clean CheXpert labels — same logic as multi-eval
# ---------------------------------------------------------------------------
print(f"Loading CheXpert labels: {LABEL_FILE_NAME_CHEX}")
labelDataChex = pd.read_csv(LABEL_FILE_NAME_CHEX)
labelDataChex = labelDataChex.replace(-1, 0)   # uncertain -> 0
labelDataChex = labelDataChex.fillna(0)
labelDataChex[list(labelDataChex)] = labelDataChex[list(labelDataChex)].astype(int)
labelDataChex = labelDataChex.drop(["subject_id", "Support Devices"], axis=1)

# Merge labels onto reason data
theData = pd.merge(theData, labelDataChex, on="study_id", how="left")
labels_df = theData.iloc[:, -NUM_OF_LABELS:]
labelNamesList = labels_df.columns.tolist()

# ---------------------------------------------------------------------------
# Test set from the committed ground-truth labels; train = full - test
# (the test set is a subset of the full dataset, so the subtraction is exact)
# ---------------------------------------------------------------------------
print(f"Loading test-set labels: {TEST_Y_TRUE_FILE}")
test_labels = pd.read_csv(TEST_Y_TRUE_FILE)
if sorted(test_labels.columns) != sorted(labelNamesList):
    raise ValueError(f"Label columns in {TEST_Y_TRUE_FILE} do not match CheXpert labels")
test_labels = test_labels[labelNamesList]

full_labels = labels_df

n_full  = len(full_labels)
n_test  = len(test_labels)
n_train = n_full - n_test

print(f"\nDataset sizes:")
print(f"  Full dataset:  {n_full:,}")
print(f"  Training set:  {n_train:,}")
print(f"  Test set:      {n_test:,}")

# ---------------------------------------------------------------------------
# Compute prevalence
# ---------------------------------------------------------------------------
def prevalence_report(counts, n, name):
    counts = counts.astype(int)
    pct    = (counts / n * 100).round(1)
    return pd.DataFrame({
        "label": labelNamesList,
        f"{name}_count": counts.values,
        f"{name}_pct":   pct.values,
    })

full_counts  = full_labels.sum()
test_counts  = test_labels.sum()
train_counts = full_counts - test_counts
if (train_counts < 0).any():
    raise ValueError("Test-set counts exceed full-dataset counts; check input files")

full_rpt  = prevalence_report(full_counts,  n_full,  "full")
train_rpt = prevalence_report(train_counts, n_train, "train")
test_rpt  = prevalence_report(test_counts,  n_test,  "test")

report = full_rpt.merge(train_rpt, on="label").merge(test_rpt, on="label")

# Add multi-label co-occurrence info
report["full_pct_of_records"] = (full_labels.sum(axis=1) > 0).sum() / n_full * 100

# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("LABEL PREVALENCE REPORT")
print("="*70)
print(f"\n{'Label':<28} {'Full n':>8} {'Full%':>7} {'Train n':>9} {'Train%':>8} {'Test n':>8} {'Test%':>7}")
print("-"*70)
for _, row in report.iterrows():
    print(f"  {row['label']:<26} {int(row['full_count']):>8,} {row['full_pct']:>7.1f}% "
          f"{int(row['train_count']):>9,} {row['train_pct']:>8.1f}% "
          f"{int(row['test_count']):>8,} {row['test_pct']:>7.1f}%")

print(f"\n  Total records:  Full={n_full:,}  Train={n_train:,}  Test={n_test:,}")
print(f"  Avg labels/record (full): {full_labels.sum(axis=1).mean():.2f}")
print(f"  Multi-label co-occurrence (full): "
      f"{(full_labels.sum(axis=1) > 1).sum():,} records "
      f"({(full_labels.sum(axis=1) > 1).sum()/n_full*100:.1f}%) have >1 label")

# Check full vs test distribution agreement
print(f"\n  Full vs Test prevalence difference (max): "
      f"{(report['full_pct'] - report['test_pct']).abs().max():.1f}pp")
print(f"  Full vs Test prevalence difference (mean): "
      f"{(report['full_pct'] - report['test_pct']).abs().mean():.2f}pp")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
report.to_csv(OUT_CSV, index=False)
print(f"\n  CSV saved: {OUT_CSV}")

# ---------------------------------------------------------------------------
# Plot — grouped bar chart matching generate_figures.py style
# ---------------------------------------------------------------------------
# Sort by full dataset prevalence ascending (rarest at bottom, most common at top)
report_sorted = report.sort_values("full_pct", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7.0, 5.2))

labels_sorted = report_sorted["label"].tolist()
labels_display = [l.replace("Enlarged Cardiomediastinum", "Enlarged CM")
                  for l in labels_sorted]

y   = np.arange(len(labels_sorted))
h   = 0.28

bars_full  = ax.barh(y + h/2, report_sorted["full_pct"],  h,
                     color="#1A6EA8", alpha=0.82,
                     label=f"Full dataset ($n={n_full:,}$)")
bars_train = ax.barh(y,        report_sorted["train_pct"], h,
                     color="#2E9E6B", alpha=0.82,
                     label=f"Training set ($n={n_train:,}$)")
bars_test  = ax.barh(y - h/2, report_sorted["test_pct"],  h,
                     color="#C05A1F", alpha=0.82,
                     label=f"Test set ($n={n_test:,}$)")

# Value labels on test bars
for bar, val in zip(bars_test, report_sorted["test_pct"]):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=6)

ax.set_yticks(y)
ax.set_yticklabels(labels_display, fontsize=8)
ax.set_xlabel("Prevalence (%)", fontsize=8)
ax.set_title(f"CheXpert Label Prevalence — v{REASON_FILE_VERSION}/{REASON_FILTER}\n"
             f"Full / Training / Test sets", fontsize=9)
ax.set_xlim(0, 42)
ax.grid(True, axis="x", color="lightgray", lw=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=7, frameon=False, loc="lower right")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Plot saved: {OUT_PNG}")

print("\nDone.")
