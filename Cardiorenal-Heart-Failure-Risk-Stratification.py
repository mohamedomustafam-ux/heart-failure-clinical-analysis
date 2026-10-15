"""Heart failure clinical records: feature engineering and risk flags.

Usage:
    python Cardiorenal-Heart-Failure-Risk-Stratification.py            # report only
    python Cardiorenal-Heart-Failure-Risk-Stratification.py --export   # also save processed CSV
    python Cardiorenal-Heart-Failure-Risk-Stratification.py --data heart_failure_clinical_records.csv   # extended file
"""

import argparse
import os

import numpy as np
import pandas as pd

# ---- Thresholds (kept in one place so they are easy to review) ----
EF_REDUCED = 40            # EF < 40%        -> HFrEF
EF_MILDLY_REDUCED = 50     # 40% <= EF < 50% -> HFmrEF
EF_SEVERE = 30             # EF < 30%        -> severe LV dysfunction
CPK_ELEVATED = 1000        # U/L, simple cut-off used here for "markedly elevated CPK"
CREATININE_FEMALE = 1.2    # mg/dL
CREATININE_MALE = 1.4      # mg/dL
CREATININE_SEVERE = 2.5    # mg/dL
CREATININE_CARDIORENAL = 1.5  # mg/dL, used together with EF < 40%

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = 'heart_failure_clinical_records_dataset.csv'  # original 299-patient UCI file
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_heart_failure_records.csv')


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_before = len(df)
    # The CSV contains many exact duplicate rows (5,000 rows, only 1,320 unique). Keeping them
    # would inflate every count, so the analysis uses unique rows only.
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Rows in file: {n_before} | exact duplicates removed: {n_before - len(df)} | unique rows kept: {len(df)}\n")
    print("Info:")
    df.info()  # prints directly (it returns None, so do not wrap it in an f-string)
    print(f"\nMissing values per column:\n{df.isna().sum().to_string()}\n")
    print(f"Describe data:\n{df.describe()}\n")
    return df


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            'anaemia': bool,
            'DEATH_EVENT': bool,
            'diabetes': bool,
            'high_blood_pressure': bool,
            'smoking': bool,
            'ejection_fraction': np.int64,
            'time': np.int64,
            'sex': np.int64,  # dataset coding: 1 = male, 0 = female
            'age': np.float64,
            'creatinine_phosphokinase': np.float64,
            'serum_creatinine': np.float64,
            'serum_sodium': np.float64,
            'platelets': np.float64,
        }
    )


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Heart failure classification by ejection fraction
    ef_conditions = [
        df['ejection_fraction'] < EF_REDUCED,
        (df['ejection_fraction'] >= EF_REDUCED) & (df['ejection_fraction'] < EF_MILDLY_REDUCED),
    ]
    df['EF_Category'] = np.select(ef_conditions, ['HFrEF', 'HFmrEF'], default='Preserved')

    df['Severe_Left_Ventricular_Dysfunction'] = np.where(
        df['ejection_fraction'] < EF_SEVERE, 'Severe_LV_Dysfunction', 'Normal'
    )

    # CPK also rises with skeletal-muscle injury, so this flag is NOT a diagnosis
    # of myocardial injury; it only marks a markedly elevated CPK value.
    df['Markedly_Elevated_CPK'] = np.where(
        df['creatinine_phosphokinase'] > CPK_ELEVATED, 'Elevated_CPK', 'Normal'
    )

    # Sex-specific creatinine thresholds (sex: 0 = female, 1 = male)
    df['Renal_Impairment'] = np.where(
        ((df['serum_creatinine'] > CREATININE_FEMALE) & (df['sex'] == 0))
        | ((df['serum_creatinine'] > CREATININE_MALE) & (df['sex'] == 1)),
        'Renal_Impairment',
        'Normal',
    )

    df['Severe_Renal_Failure'] = np.where(
        df['serum_creatinine'] > CREATININE_SEVERE, 'Severe_Failure', 'Normal'
    )

    # Sodium and platelet disturbances
    df['Hyponatremia'] = np.where(df['serum_sodium'] < 135, 'Hyponatremia', 'Normal')
    df['Hypernatremia'] = np.where(df['serum_sodium'] > 145, 'Hypernatremia', 'Normal')
    df['Thrombocytopenia'] = np.where(df['platelets'] < 150000, 'Thrombocytopenia', 'Normal')
    df['Thrombocytosis'] = np.where(df['platelets'] > 450000, 'Thrombocytosis', 'Normal')

    # Combined risk flags (data-derived flags, not clinical diagnoses)
    df['Cardiorenal_Risk_Flag'] = np.where(
        (df['ejection_fraction'] < EF_REDUCED) & (df['serum_creatinine'] > CREATININE_CARDIORENAL),
        'High',
        'Low',
    )
    df['Vascular_Metabolic_Risk_Triad'] = np.where(
        df['diabetes'] & df['high_blood_pressure'] & df['smoking'], 'High', 'Low'
    )
    return df


def report(df: pd.DataFrame) -> None:
    print(f"Overall mortality rate:\n{df['DEATH_EVENT'].value_counts(normalize=True).to_string()}\n")

    print("--- Cardiorenal risk flag: mortality summary ---")
    summary = df.groupby('Cardiorenal_Risk_Flag')['DEATH_EVENT'].agg(['count', 'mean'])
    summary['Mortality_Percentage'] = summary['mean'] * 100
    print(f"{summary}\n")

    print("--- Pivot table: renal impairment vs EF category (mortality %) ---")
    pivot = pd.pivot_table(
        df, values='DEATH_EVENT', index='Renal_Impairment', columns='EF_Category', aggfunc='mean'
    )
    print(f"{pivot * 100}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', action='store_true', help='save the processed dataset as CSV')
    parser.add_argument('--data', default=DEFAULT_DATA, help='CSV file next to this script')
    args = parser.parse_args()

    df = add_clinical_features(convert_types(load_data(os.path.join(BASE_DIR, args.data))))
    report(df)

    if args.export:
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"Saved processed dataset to: {PROCESSED_PATH}")


if __name__ == '__main__':
    main()
