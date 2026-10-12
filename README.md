# Radiology Indication Field NLP — Supporting Data and Code

Supporting data and analysis code for:

> Peressini, A.F. (2026). *Predicting Findings from Radiology Indication Text:
> BERT vs Naïve Bayes Classification.*
> Journal of Imaging Informatics in Medicine (JIIM).
> https://doi.org/10.1007/s10278-026-02404-1 [in press; volume and pages to be added]



---

## Overview

This repository contains the model prediction arrays, evaluation
results, statistical test outputs, figure generation scripts, and
analysis code used to produce all tables and figures in the manuscript.

The study evaluates whether free-text radiology order Indication text
can predict the presence and type of radiological findings using
Naïve Bayes (TF-IDF) and three BERT-based models (baseBERT,
radBERT, BioClinicalBERT) on 219,995 chest radiograph orders from
the MIMIC-CXR-JPG dataset.



### Relationship to Prior Work

This study extends a preliminary feasibility study establishing that
Indication text can predict radiological findings:

> Peressini, A.F. (2024). The radiology report indication field:
> Exploring predictive possibilities with natural language processing.
> Master's thesis, Marquette University, Milwaukee, Wisconsin.

Early versions of the Naïve Bayes and BERT modeling scripts from that
thesis are available at https://github.com/peressinia/radReasons.
The BERT scripts were substantially rewritten for this study using
the kTrain framework; the Naïve Bayes scripts share structural
ancestry with the thesis versions. The journal paper adds multi-label
classification, rigorous statistical testing, calibration analysis,
and threshold-based triage evaluation not present in the thesis.

---

## Data Availability

> ⚠️ **The raw MIMIC-CXR-JPG dataset is not included in this
> repository.** It is publicly available from PhysioNet under a
> data use agreement:
>
> Johnson, A., et al. (2024). MIMIC-CXR-JPG — Chest Radiographs
> with Structured Labels (version 2.1.0). PhysioNet.
> https://doi.org/10.13026/jsn5-t979

This repository contains only derived prediction arrays and
evaluation outputs from the held-out test set (n = 22,644).
No individual patient data or raw text is included.


---

## Repository Structure

```
rad-indications-nlp-results/
│
├── README.md
├── LICENSE
├── CITATION.cff                         # Citation metadata
├── requirements.txt                     # Python dependencies
├── .gitignore
│
├── data_csv/                            # Model predictions (CSV format; written by compare_models_final-v1-3.py)
│   ├── binary_baseBERT_predictions.csv
│   ├── binary_radBERT_predictions.csv
│   ├── binary_bioClinBERT_predictions.csv
│   ├── binary_MultinomialNB_predictions.csv
│   ├── multilabel_baseBERT_y_true.csv
│   ├── multilabel_baseBERT_y_pred.csv
│   ├── multilabel_baseBERT_y_prob.csv
│   ├── multilabel_radBERT_y_true.csv
│   ├── multilabel_radBERT_y_pred.csv
│   ├── multilabel_radBERT_y_prob.csv
│   ├── multilabel_bioClinBERT_y_true.csv
│   ├── multilabel_bioClinBERT_y_pred.csv
│   ├── multilabel_bioClinBERT_y_prob.csv
│   ├── multilabel_BernoulliNB_y_true.csv
│   ├── multilabel_BernoulliNB_y_pred.csv
│   └── multilabel_BernoulliNB_y_prob.csv
│
├── data_numpy/                          # Raw prediction arrays (.npy), 3 per model:
│   │                                    #   <prefix>_y_true_<task>.npy
│   │                                    #   <prefix>[_nb]_<task>_preds.npy
│   │                                    #   <prefix>[_nb]_<task>_probs.npy
│   │                                    # (_nb only on the NB preds/probs files)
│   ├── CNF-v4-0-baseBERT_bt32-ep3-lr0-bl1-tt0_pyV-1-2_*  (binary)
│   ├── CNF-v4-0-radBERT_bt32-ep2-lr0-bl1-tt0_pyV-1-2_*  (binary)
│   ├── CNF-v4-0-bioClinBERT_bt32-ep1-lr0-bl3-tt0_pyV-1-2_*  (binary)
│   ├── MultinomialNB-final-1500-NCF-v4-0-bal1_pyV-1-1_*  (binary)
│   ├── CNF-v4-0-baseBERT_bt32-ep6-lr0-bl0-tt0_pyV-1-1_*  (multilabel)
│   ├── CNF-v4-0-radBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_*  (multilabel)
│   ├── CNF-v4-0-bioClinBERT_bt32-ep5-lr0-bl0-tt0_pyV-1-1_*  (multilabel)
│   └── BernoulliNB-TFIDF-xSplits-1500-NCF-v4-0-bal0_*  (multilabel)
│
├── results_model_comparison/            # Statistical test results
│   ├── binary_bootstrap_ci.csv
│   ├── binary_mcnemar_results.csv
│   ├── binary_auroc_comparison.csv      # DeLong + paired-bootstrap AUROC tests
│   ├── binary_auprc_comparison.csv      # Paired-bootstrap AUPRC tests
│   ├── multilabel_bootstrap_ci.csv
│   ├── multilabel_permutation_results.csv
│   ├── multilabel_auc_permutation_results.csv  # AUROC/AUPRC permutation tests
│   ├── compare_models_final-v1-3.out    # console run log
│   └── compare_auroc-v1-1.out           # console run log
│
├── results_threshold/                   # Threshold analysis outputs
│   ├── binary_pr_curves.csv
│   ├── binary_operating_points.csv
│   ├── binary_xray_savings.csv
│   ├── multilabel_pr_curves.csv
│   ├── multilabel_operating_points.csv
│   └── threshold_analysis-v1-4.out       # console run log
│
├── results_final/                       # Final test set evaluation outputs
│   ├── binary_metrics.csv
│   ├── binary_pairwise_stats.csv
│   ├── multilabel_metrics.csv
│   ├── multilabel_pairwise_stats.csv
│   └── evaluate_models-v2-0.out          # console run log
│
├── results_finetune/                    # Hyperparameter tuning run tallies
│   ├── model-BERT-main-Stat-Tally-Binary-FineTune.csv  (and .xlsx)
│   ├── model-BERT-main-Stat-Tally-MultiL-FineTune.csv  (and .xlsx)
│   ├── NB-Stat-Tally-binary-FineTune.csv               (and .xlsx)
│   └── NB-Stat-Tally-multi-FineTune.csv                (and .xlsx)
│
├── results_label_report/                # Label prevalence report
│   ├── label_prevalence_report-v4-0.csv
│   ├── label_prevalence_report-v4-0.png
│   └── label_prevalence_report-v1-1.out # console run log
│
├── figures/                             # Manuscript figures only
│   ├── fig11-perlabel-f1.pdf / .jpg     # Figure 2: Per-label F1
│   ├── fig12-pr-curves.pdf / .jpg       # Figure 3: PR curves
│   ├── fig13-calibration.pdf / .jpg     # Figure 4: Reliability diagrams
│   ├── fig14-label-freq.pdf / .jpg      # Figure 1: Label prevalence
│   └── generate_figures-v1-5.out        # console run log
│
├── diagnostic_figures/                  # Diagnostic plots (not in manuscript)
│   ├── roc_binary.png                   # ROC curves, binary task
│   ├── pr_binary.png                    # PR curves, binary task
│   └── threshold_analysis.png           # PR curve + operating points summary
│
└── scripts/
    ├── evaluate_models-v2-0.py          # Model evaluation pipeline
    ├── compare_models_final-v1-3.py     # Bootstrap CIs + statistical tests
    ├── compare_auroc-v1-1.py            # Pairwise AUROC/AUPRC tests, both tasks
    ├── threshold_analysis-v1-4.py       # PR curves + triage simulation
    ├── generate_figures-v1-5.py         # All 4 manuscript figures
    └── label_prevalence_report-v1-1.py  # Full/train/test label prevalence report
```


> **Note:** Files with `.out` extension (e.g., `evaluate_models-v2-0.out`,
> `compare_models_final-v1-3.out`) are console run logs retained for
> provenance. They are not required to reproduce results.

---

## Decoding the `data_numpy/` Filenames


Filenames in `data_numpy/` encode the selected configuration directly,
e.g. `CNF-v4-0-bioClinBERT_bt32-ep1-lr0-bl3-tt0_pyV-1-2_binary_preds.npy`:

- `CNF-v4-0` — preprocessing variant and dataset version (cased,
  non-standardized; see Methods)
- `bt32` — batch size 32
- `ep1` — 1 training epoch (matches Table 1: BioClinicalBERT, binary
  task, 1 epoch)
- `lr0` — learning-rate configuration index (not the literal rate;
  actual rates per model are in Table 1)
- `bl3` — balancing-strategy index. The mapping is task-specific; the
  `dBalancing` column of the `results_finetune/` BERT tallies gives the
  technique for every index tried during validation:

  | Index | Binary task          | Multilabel task |
  |-------|----------------------|-----------------|
  | `bl0` | `none`               | `none`          |
  | `bl1` | `classWts`           | `classWts`      |
  | `bl2` | `uSample`            | `OverSampAve`   |
  | `bl3` | `oSample`            | `NLPAugIns`     |
  | `bl4` | `nlpAug-insert`      | `NLPAugSub`     |
  | `bl5` | `nlpAug-substitute`  | —               |

  `OverSampAve` balances the dataset by oversampling based on the
  average number of labels per sentence.

  The final models use `bl1` (class-weighted loss: binary baseBERT and
  radBERT), `bl3` (oversampling: binary BioClinicalBERT), and `bl0`
  (none: all multilabel models), matching Table 1.
- `bal1` — Naïve Bayes balancing index (NB filenames only), from
  `['none', 'SMOTE', 'OSS']`: `bal0` = none, `bal1` = SMOTE, `bal2` =
  OSS (one-sided selection). The final models are binary MultinomialNB
  `bal1` (SMOTE) and multilabel BernoulliNB `bal0` (none). The
  multilabel NB tally tried only `none` and `SMOTE`.
- `tt0` — training-type index: `0` = 1cycle learning-rate policy,
  `1` = auto (kTrain's autofit default). Every file in this repo is
  `tt0` — i.e., every selected final model used the 1cycle policy,
  none used auto — consistent with the cyclical learning-rate finder
  described in Methods (Smith, 2018).
- `pyV-1-2` — the *producing script's own* version number, not a
  shared pipeline version. 

---

## CSV File Descriptions

### Binary predictions (`data_csv/binary_[model]_predictions.csv`)
One file per model. One row per test sample (n = 22,644). Columns:
- `y_true` — ground truth label (1 = finding present, 0 = no finding)
- `y_pred` — binary predicted label at default 0.5 threshold
- `y_prob` — predicted probability of finding (positive class)


### Multilabel predictions (`data_csv/multilabel_[model]_y_*.csv`)
Three files per model, one row per test sample (n = 22,644). Columns
are the 13 CheXpert label names:
- `_y_true.csv` — ground truth labels (binary 0/1)
- `_y_pred.csv` — predicted labels at default 0.5 threshold (binary 0/1)
- `_y_prob.csv` — predicted probabilities (float)

Note: `multilabel_[model]_y_true.csv` files are identical across all
models — verified prior to statistical comparison.

> **Note on structure:** Binary and multilabel predictions use
> intentionally different file layouts. Binary predictions combine
> y_true, y_pred, and y_prob in a single file per model because
> each array is one-dimensional and fits naturally in three columns.
> Multilabel predictions use three separate files per model because
> each file has 13 label columns; combining all three arrays into
> one 39-column file would be unwieldy. In both cases all information
> needed to reproduce the paper's results is present.

### Statistical results (`results_model_comparison/`)
- `binary_bootstrap_ci.csv` — Bootstrap CIs (n = 2,000 resamples,
  percentile method) for all binary metrics. Columns: model, metric,
  value, ci_lower, ci_upper.
- `binary_mcnemar_results.csv` — McNemar's test for all pairwise
  binary comparisons with Benjamini-Hochberg corrected p-values. This
  compares accuracy at the 0.5 threshold, not AUROC.
- `binary_auroc_comparison.csv` — Pairwise binary AUROC comparisons:
  DeLong's test for correlated ROC curves (Benjamini-Hochberg
  corrected) plus a paired bootstrap 95% CI (n = 2,000) of each AUROC
  difference. Generated by `compare_auroc-v1-1.py`.
- `binary_auprc_comparison.csv` — Pairwise binary AUPRC comparisons:
  paired bootstrap 95% CI (n = 2,000, the same resamples as the AUROC
  bootstrap) and two-sided p-value of each AUPRC difference
  (Benjamini-Hochberg corrected). Generated by `compare_auroc-v1-1.py`.
- `multilabel_bootstrap_ci.csv` — Bootstrap CIs for all multilabel
  metrics including per-label F1 for all 13 CheXpert categories.
- `multilabel_permutation_results.csv` — Paired permutation tests
  (n = 5,000) with Benjamini-Hochberg correction, on the 0.5-threshold
  metrics (Micro-F1, Macro-F1, weighted-F1, Hamming loss).
- `multilabel_auc_permutation_results.csv` — Paired permutation tests
  (n = 5,000) on the threshold-free metrics (micro-/macro-AUROC,
  micro-/macro-AUPRC), Benjamini-Hochberg corrected across all 24
  rows. Generated by `compare_auroc-v1-1.py`.

### Threshold results (`results_threshold/`)
- `binary_pr_curves.csv` — Full precision-recall curves, all 4 models.
- `binary_operating_points.csv` — Recall at target precision levels
  (85%, 90%, 93%, 95%, 97%, 99%) with triage simulation results
  scaled to 100,000 annual orders.
- `binary_xray_savings.csv` — Annual avoidable radiograph estimates.
- `multilabel_pr_curves.csv` — Per-label PR curves, all 4 models.
- `multilabel_operating_points.csv` — Per-label operating points.

### Figures (`figures/` and `diagnostic_figures/`)
**`figures/`** contains only the four manuscript figures (cited in the paper),
generated by `generate_figures-v1-5.py`. Available as both PDF (for LaTeX) and JPG:
`fig11-perlabel-f1`, `fig12-pr-curves`, `fig13-calibration`, `fig14-label-freq`.

**`diagnostic_figures/`** contains quick-look plots generated during analysis,
not cited in the manuscript but included for transparency and reproducibility:
- `roc_binary.png` — ROC curves for all binary models (from `evaluate_models-v2-0.py`)
- `pr_binary.png` — PR curves for all binary models (from `evaluate_models-v2-0.py`)
- `threshold_analysis.png` — PR curves + operating point summary (from `threshold_analysis-v1-4.py`)

### Final evaluation outputs (`results_final/`)
All four files are generated by `evaluate_models-v2-0.py` from the
`data_numpy/` prediction arrays.
- `binary_metrics.csv` — Accuracy, precision, recall, specificity,
  F1, AUROC, AUPRC for all binary models on the held-out test set.
- `binary_pairwise_stats.csv` — McNemar's test (continuity-corrected
  chi-square) for all pairwise binary model comparisons.
- `multilabel_metrics.csv` — Micro-F1, Macro-F1, Weighted-F1,
  Hamming loss for all multilabel models.
- `multilabel_pairwise_stats.csv` — Paired permutation test
  (n = 5,000) results for all pairwise multilabel comparisons.

### Hyperparameter tuning tallies (`results_finetune/`)
Validation set performance across all hyperparameter configurations
evaluated during model selection — the metrics used to choose the
final model configurations reported in Table 1 of the manuscript.
Available as both CSV and Excel (.xlsx).

**Selection rule (binary task).** Validation AUROC was the primary
criterion, with differences smaller than 0.005 treated as ties; ties
were broken by validation MCC (`val_mcc`, reported unnormalized in the
tallies — normalizing as (MCC + 1)/2 does not change the ranking). This
is why each selected binary configuration has the highest `val_mcc` for
its model, while an alternative (e.g. no balancing) sometimes has a
marginally higher `val_auroc`: baseBERT 0.7997 vs 0.8007, radBERT
0.8057 vs 0.8061, BioClinicalBERT 0.8038 vs 0.8042, MultinomialNB
0.7728 vs 0.7732. For the multilabel task the selected configuration
has the highest validation Micro-F1 outright.

### Label prevalence report (`results_label_report/`)
`label_prevalence_report-v4-0.csv` and the accompanying `.png` report
CheXpert label prevalence broken out by full dataset (n = 219,995),
training set (n = 197,351), and test set (n = 22,644), plus the
full-vs-test prevalence gap. Generated by
`scripts/label_prevalence_report-v1-1.py`; `label_prevalence_report-v1-1.out`
is the console run log.

The CSV is the data source for manuscript Figure 1 (`fig14-label-freq`):
`generate_figures-v1-5.py` reads the full-dataset and test-set
prevalence columns from it. The report itself (the training-set
columns and the `.png`) is not cited in the manuscript text and is
included for transparency.

Note: full-vs-test prevalence difference is minimal: max 1.1 percentage
points, mean 0.34 percentage points across all 13 labels, which supports
split representativeness.

---

## Script Descriptions

> **Provenance convention:** each script hardcodes its own
> `SCRIPT_VERSION` and `SCRIPT_NAME` constants (e.g.
> `SCRIPT_VERSION = '1-2'`, `SCRIPT_NAME = 'model-BERT-binary-eval'`)
> and stamps them into its own output filenames and console logs.
> Scripts are versioned independently of one another, so it's
> expected for different scripts to be at different version numbers
> at the same time (see the `pyV` note under Decoding the
> `data_numpy/` Filenames, above).

### `evaluate_models-v2-0.py`
Model evaluation pipeline. Loads prediction arrays from `data_numpy/`,
verifies y_true consistency across models, computes all metrics, runs
McNemar's and permutation tests, and generates two quick-look diagnostic
plots (`diagnostic_figures/roc_binary.png`, `diagnostic_figures/pr_binary.png`).
Outputs CSVs to `results_final/`. All results in the manuscript tables are
derived from this script.

```bash
python scripts/evaluate_models-v2-0.py
```

### `compare_models_final-v1-3.py`
Computes bootstrap confidence intervals (n = 2,000) and full
statistical comparison suite (McNemar's test, paired permutation
tests, Benjamini-Hochberg correction) from the prediction arrays.
Outputs to `results_model_comparison/`.

```bash
python scripts/compare_models_final-v1-3.py
```

### `compare_auroc-v1-1.py`
Tests whether the models differ on the threshold-free metrics, pair by
pair, in both tasks. `compare_models_final-v1-3.py` tests only
0.5-threshold metrics.
- **Binary AUROC:** DeLong's test for two correlated AUROCs on the
  same test set (fast algorithm of Sun & Xu, 2014), with
  Benjamini-Hochberg correction across the six pairs, plus a paired
  bootstrap (n = 2,000, seed 42) of each AUROC difference.
- **Binary AUPRC:** paired bootstrap of each AUPRC difference, on the
  same resamples.
- **Multi-label micro-/macro-AUROC and AUPRC:** paired permutation
  tests (n = 5,000), swapping whole orders between the two models, as
  in `compare_models_final-v1-3.py`.

Reads the arrays in `data_numpy/`; outputs `binary_auroc_comparison.csv`,
`binary_auprc_comparison.csv` and `multilabel_auc_permutation_results.csv`
in `results_model_comparison/`. The permutation tests run the six pairs
in parallel and take about 40 minutes.

```bash
python scripts/compare_auroc-v1-1.py | tee results_model_comparison/compare_auroc-v1-1.out
```

References:
- DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under
  two or more correlated receiver operating characteristic curves: a
  nonparametric approach. *Biometrics*. 1988;44(3):837–45.
- Sun X, Xu W. Fast implementation of DeLong's algorithm for comparing
  the areas under correlated receiver operating characteristic curves.
  *IEEE Signal Process Lett*. 2014;21(11):1389–93.

### `threshold_analysis-v1-4.py`
Generates precision-recall curves and triage simulation results
across target precision levels of 85%–99%. Outputs to
`results_threshold/`.

```bash
python scripts/threshold_analysis-v1-4.py
```

### `generate_figures-v1-5.py`
Generates all four manuscript figures (Figures 1–4). Reads from
`results_model_comparison/`, `results_threshold/`, `data_numpy/`, and
`results_label_report/` (Figure 1 only).
Outputs PDF and JPG versions to `figures/`.

```bash
python scripts/generate_figures-v1-5.py
```

### `label_prevalence_report-v1-1.py`
Reports CheXpert label prevalence for the full dataset, training set,
and test set side by side (grouped bar chart + CSV), including the
full-vs-test prevalence gap. Full-dataset counts come from the MIMIC
files; test-set counts come from the committed ground-truth labels
(`data_csv/multilabel_baseBERT_y_true.csv`); training-set counts are
full minus test. Outputs
`results_label_report/label_prevalence_report-v4-0.csv` and the
accompanying `.png` — the version suffix is built automatically from
`REASON_FILE_VERSION`, so a fresh run reproduces the committed
filenames. Console output is saved as
`results_label_report/label_prevalence_report-v1-1.out`, same
provenance convention as the other `.out` logs in this repo.

v1.0 of this script selected the test set from a list of study IDs.
v1.1 derives it from the committed `y_true` labels instead; the
regenerated CSV is byte-identical to the v1.0 output.

> **Requires MIMIC-CXR-JPG.** Unlike the four scripts above, this one
> also reads MIMIC-derived files that are not redistributed here —
> `mimic-reasons-CNF-v4-0.csv` and `mimic-cxr-2.0.0-chexpert.csv` (the
> raw CheXpert label file). Set `MIMIC_DIR` to the folder holding them
> in your own PhysioNet-credentialed copy of MIMIC-CXR-JPG. The
> test-set columns can be checked without MIMIC access by summing the
> columns of any `data_csv/multilabel_*_y_true.csv` file.

```bash
python scripts/label_prevalence_report-v1-1.py
```

---

## Requirements
 
See `requirements.txt` for the exact package versions used (results were produced with Python 3.9.6; Python >= 3.9 required). Install with:
```bash
pip install -r requirements.txt
```
 
Model training used the kTrain framework and is not needed to reproduce
the reported statistics and figures from the saved predictions.
 
---

## Reproducing the Paper Results

All reported statistics and figures can be reproduced from the saved
prediction arrays without retraining any models:

```bash
# 1. Run statistical comparisons (bootstrap CIs + significance tests)
python scripts/compare_models_final-v1-3.py
python scripts/compare_auroc-v1-1.py

# 2. Run threshold analysis (PR curves + triage simulation)
python scripts/threshold_analysis-v1-4.py

# 3. Generate all figures
python scripts/generate_figures-v1-5.py
```

To reproduce the evaluation metrics and pairwise stats from scratch:
```bash
python scripts/evaluate_models-v2-0.py
```

Configure directory paths at the top of each script before running.
All outputs will match the values reported in the manuscript within
floating-point rounding.

---

## Citation

If you use this data or code, please cite:

```bibtex
@article{peressini2026indication,
  author  = {Peressini, Anthony F.},
  title   = {Predicting Findings from Radiology {Indication} Text: {BERT} vs
             Na{\"i}ve Bayes Classification},
  journal = {Journal of Imaging Informatics in Medicine},
  year    = {2026},
  doi     = {10.1007/s10278-026-02404-1},
  note    = {in press}
}
```


---

## License

Code is released under the MIT License.
Data files are derived from MIMIC-CXR-JPG and are subject to the
[PhysioNet Credentialed Health Data License](https://physionet.org/content/mimic-cxr-jpg/view-license/2.1.0/).

---

## Contact
 
Anthony F. Peressini — Marquette University
 
---
 
## Archive
 
A permanent, citable snapshot of this repository is archived on Zenodo:
[DOI: to be added after the first release].
 
---
 
## Changelog
 
- **v1.0.0** — First archived release accompanying the accepted JIIM
  manuscript.
