# Heart Failure Clinical Records: Feature Engineering, Statistics & a Baseline Model

## Overview
**Part 2** of my Data Science & Biomedical Analysis learning roadmap. After **Part 1** (NumPy array operations, see the EEG analyzer project), this project uses **Pandas**, **SciPy** and **scikit-learn** to clean a clinical dataset, engineer risk features, test an association statistically, and fit a simple baseline model. It also compares the original dataset with an expanded version found online.

This is a learning project on a public dataset. Results are exploratory and are **not clinical findings**.

## Data
| File | Rows | Notes |
|---|---|---|
| `heart_failure_clinical_records_dataset.csv` | 299 | **Original dataset (default).** 299 heart failure patients, Faisalabad Institute of Cardiology and Allied Hospital, Pakistan, 2015. No duplicates, no missing values. |
| `heart_failure_clinical_records.csv` | 5,000 | Expanded version downloaded from Kaggle *(add page link and uploader)*. Only **1,320 unique rows** (3,680 exact copies). 296 of the 299 original patients appear in it; how the rest was generated is not documented. Used only for comparison. |

**Source and citation (original data):** Chicco D., Jurman G. "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone." *BMC Medical Informatics and Decision Making* 20, 16 (2020). https://doi.org/10.1186/s12911-020-1023-5. Hosted by the UCI Machine Learning Repository (id 519, CC BY 4.0): https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records

Columns: age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex (1 = male, 0 = female), smoking, `time` (follow-up days), `DEATH_EVENT`.

## What the code does
- **Cleaning:** type conversion, missing-value check, removal of exact duplicates.
- **Feature engineering:** EF category (HFrEF < 40%, HFmrEF 40-49%, Preserved >= 50%), sex-specific creatinine thresholds for renal impairment, severe LV dysfunction, markedly elevated CPK, sodium and platelet disturbances, and a **cardiorenal risk flag** (EF < 40% and creatinine > 1.5 mg/dL). The flag is data-derived and is **not** a diagnosis.
- **Statistics:** chi-square test of the risk flag against death, with an odds ratio.
- **Model:** logistic regression (standardised features; `time` excluded because follow-up length depends on the outcome), evaluated by ROC/AUC on a held-out test set and by 5-fold cross-validation.

## Results
| | Original (299) | Expanded (1,320 unique) |
|---|---|---|
| Overall mortality | 32.1% | 30.1% |
| Mortality, risk flag High | 66.7% (51 patients) | 70.1% (211 rows) |
| Mortality, risk flag Low | 25.0% (248 patients) | 22.5% (1,109 rows) |
| Chi-square p-value / odds ratio | 1.7e-08 / 6.0 | 4.2e-43 / 8.1 |
| Logistic regression test AUC | 0.74 | 0.81 |
| 5-fold CV AUC | 0.77 (+/- 0.05) | 0.80 (+/- 0.02) |
| Strongest coefficients | creatinine (+), ejection fraction (-), age (+) | ejection fraction (-), creatinine (+), age (+), sodium (-) |

**What I take from this**
- The cardiorenal flag is strongly associated with mortality in both versions, but it is an association, not proof of cause.
- The expanded file gives tighter-looking numbers and a different coefficient pattern (for example serum sodium matters there but almost not in the original), which suggests the expansion changed the relationships. Conclusions should rely on the **original** data.
- With only 299 patients and a test set of 75, the AUC varies a lot (0.74 on the test split, 0.77 in cross-validation).

Figures are saved in `figures/` for both files (`heart_failure_clinical_records_dataset_*` for the original).

## Limitations
- Mortality proportions are not adjusted for age or other factors and ignore follow-up time.
- Thresholds are simple cut-offs for learning (CPK, for example, also rises with muscle injury).
- Small sample, no external validation, no hyper-parameter tuning.

## How to run
```bash
git clone https://github.com/mohamedomustafam-ux/heart-failure-clinical-analysis.git
cd heart-failure-clinical-analysis
pip install -r requirements.txt

python Cardiorenal-Heart-Failure-Risk-Stratification.py --export    # feature engineering + report (original data)
python heart_failure_modeling.py                                     # statistics, model, figures (original data)
python heart_failure_modeling.py --data heart_failure_clinical_records.csv   # same, expanded file
jupyter notebook heart_failure_analysis.ipynb                        # notebook version
```

## Roadmap
1. Compare logistic regression with a tree-based model and report calibration.
2. Add confidence intervals for the AUC (bootstrap).

## Tech stack
Python 3, Pandas, NumPy, SciPy, scikit-learn, Matplotlib, Jupyter
