# Heart Failure Clinical Records: Feature Engineering & Risk Stratification

## Overview
This repository represents **Part 2** of my Data Science & Biomedical Analysis learning roadmap. 
Following **Part 1** (focused purely on **NumPy** array operations), this project advances into **Pandas** and **NumPy** to perform clinical data wrangling, feature engineering, and risk aggregation on heart failure clinical records.

---

## Key Technical Highlights
- **Data Standardization & Type Precision:** Converted binary clinical indicators to `bool` and continuous metrics to explicit float/int representations.
- **Clinical Feature Engineering:**
  - **EF Classification:** Categorized Ejection Fraction using `np.select` into **HFrEF** (<40%), **HFmrEF** (40-49%), and **Preserved** (≥50%).
  - **Renal Impairment:** Applied gender-aware creatinine thresholds ($>1.2$ mg/dL for females, $>1.4$ mg/dL for males).
  - **Complex Risk Markers:** Built indicator flags for **Cardiorenal Syndrome Risk**, **Severe LV Dysfunction**, and **Electrolyte Disturbances**.
- **Interactive Notebook Design:** Clean modular cells in Jupyter Notebook (`.ipynb`) featuring a reproducible sample slice (`random_state=42`) and multi-dimensional cross-tabulations (`pd.pivot_table`).

---

## Core Findings & Aggregations
- **Cardiorenal Risk Impact:** Evaluated mortality trends among high-risk patients with combined low EF and elevated Serum Creatinine.
- **Renal vs. Cardiac Function:** Built a mortality matrix cross-tabulating `Renal_Impairment` status against `EF_Category`.

---

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/heart-failure-clinical-analysis.git](https://github.com/mohamedomustafam-ux/heart-failure-clinical-analysis.git)
   cd heart-failure-clinical-analysis