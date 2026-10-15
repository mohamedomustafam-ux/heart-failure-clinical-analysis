# Heart Failure Clinical Records: Feature Engineering, Statistics & a Baseline Model

## Overview
**Part 2** of my Data Science & Biomedical Analysis learning roadmap. After **Part 1** (NumPy array operations, see the EEG analyzer project), this project uses **Pandas**, **SciPy** and **scikit-learn** to clean, engineer clinical features, test an association statistically, and fit a simple baseline model on a heart failure dataset.

This is a learning project on a public dataset. Results are exploratory and are **not clinical findings**.

## Dataset
- File: `heart_failure_clinical_records.csv` (13 columns: age, ejection fraction, serum creatinine, serum sodium, platelets, CPK, anaemia, diabetes, high blood pressure, smoking, `sex` (1 = male, 0 = female), follow-up `time` in days, and the outcome `DEATH_EVENT`).
- **Data quality note:** the file has 5,000 rows but only **1,320 unique rows**; the rest are exact copies. Keeping duplicates would inflate counts and make p-values look extremely small, and copies could leak between training and test sets. All results below use the **unique rows only**. Near-duplicates may still remain, so treat the numbers as optimistic.
- Source: *add the original source and licence here, and state whether this file is the original data or an expanded/augmented version (the widely used original has 299 patients).*

## What the code does
- **Cleaning:** type conversion, missing-value check, removal of exact duplicates.
- **Feature engineering:** EF category (HFrEF < 40%, HFmrEF 40-49%, Preserved >= 50%), sex-specific creatinine thresholds for renal impairment, severe LV dysfunction, markedly elevated CPK, sodium and platelet disturbances, and a **cardiorenal risk flag** (EF < 40% and creatinine > 1.5 mg/dL). The flag is data-derived and is **not** a diagnosis.
- **Statistics:** chi-square test of the risk flag against death, with an odds ratio.
- **Model:** logistic regression (standardised features, `time` excluded because follow-up length depends on the outcome) evaluated by ROC/AUC on a held-out test set and by 5-fold cross-validation.

## Results (1,320 unique rows)
| Cardiorenal risk flag | Rows | Mortality |
|---|---|---|
| High | 211 | 70.1% |
| Low | 1,109 | 22.5% |

- Overall mortality: 30.1%.
- Chi-square test: chi2 = 189.4, p = 4.2e-43, odds ratio = 8.1. The association is strong, but it is an association, not proof of cause.
- Mortality is higher with renal impairment in every EF category (for example 67.5% vs 24.5% in HFrEF).
- Logistic regression: test AUC = 0.81, 5-fold CV AUC = 0.80 (+/- 0.02). Strongest standardised coefficients: ejection fraction (negative), serum creatinine, age (positive), serum sodium (negative).

Figures (saved by `heart_failure_modeling.py` in `figures/`): `ef_by_outcome.png`, `mortality_by_flag.png`, `roc_curve.png`.

## Limitations
- Raw mortality proportions are not adjusted for age or other factors and ignore follow-up time.
- Thresholds are simple cut-offs for learning (CPK, for example, also rises with muscle injury).
- Duplicated and possibly augmented data: see the data quality note above.
- No external validation and no hyper-parameter tuning.

## How to run
```bash
git clone https://github.com/mohamedomustafam-ux/heart-failure-clinical-analysis.git
cd heart-failure-clinical-analysis
pip install -r requirements.txt

python Cardiorenal-Heart-Failure-Risk-Stratification.py --export   # feature engineering + report
python heart_failure_modeling.py                                    # statistics, model, figures
jupyter notebook heart_failure_analysis.ipynb                       # notebook version
```

## Roadmap
1. Compare with a tree-based model and report calibration.
2. Repeat the analysis on the original 299-patient dataset for comparison.

## Tech stack
Python 3, Pandas, NumPy, SciPy, scikit-learn, Matplotlib, Jupyter
