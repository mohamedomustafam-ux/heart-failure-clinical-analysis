import os

import numpy as np
import pandas as pd

# 1. Open data file and Create DataFrame
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, 'heart_failure_clinical_records.csv')

df_heart_fcr = pd.read_csv(FILE_PATH)

# 2. Check data
print(f"Info:\n{df_heart_fcr.info()}\n")
print(f"Check missing data:\n{pd.isna(df_heart_fcr).sum().to_string()}\n")
print(f"Describe data:\n{df_heart_fcr.describe()}\n")

# 3. Data type conversion
df_heart_fcr = df_heart_fcr.astype(
    {
        'anaemia': bool,
        'DEATH_EVENT': bool,
        'diabetes': bool,
        'high_blood_pressure': bool,
        'smoking': bool,
        'ejection_fraction': np.int64,
        'time': np.int64,
        'sex': np.int64,
        'age': np.float64,
        'creatinine_phosphokinase': np.float64,
        'serum_creatinine': np.float64,
        'serum_sodium': np.float64,
        'platelets': np.float64
    }
)

# Baseline mortality rate
print(f"Overall Mortality Rate:\n{df_heart_fcr['DEATH_EVENT'].value_counts(normalize=True).to_string()}\n")

# 4. Clinical Diagnoses & Categorization
# Unified Heart Failure Classification (EF)
ef_conditions = [
    df_heart_fcr['ejection_fraction'] < 40,
    (df_heart_fcr['ejection_fraction'] >= 40) & (df_heart_fcr['ejection_fraction'] < 50)
]
ef_choices = ['HFrEF', 'HFmrEF']
df_heart_fcr['EF_Category'] = np.select(ef_conditions, ef_choices, default='Preserved')

# Severe Left Ventricular Dysfunction
df_heart_fcr['Severe_Left_Ventricular_Dysfunction'] = np.where(
    df_heart_fcr['ejection_fraction'] < 30,
    'Severe_LV_Dysfunction',
    'Normal'
)

# Severe Myocardial Injury (Replaced '/' with '_')
df_heart_fcr['Severe_Myocardial_Injury'] = np.where(
    df_heart_fcr['creatinine_phosphokinase'] > 1000,
    'Severe_Injury',
    'Normal'
)

# Renal Impairment (Explicit sex comparison)
df_heart_fcr['Renal_Impairment'] = np.where(
    ((df_heart_fcr['serum_creatinine'] > 1.2) & (df_heart_fcr['sex'] == 0)) |
    ((df_heart_fcr['serum_creatinine'] > 1.4) & (df_heart_fcr['sex'] == 1)),
    'Renal_Impairment',
    'Normal'
)

# Severe Renal Failure
df_heart_fcr['Severe_Renal_Failure'] = np.where(
    df_heart_fcr['serum_creatinine'] > 2.5,
    'Severe_Failure',
    'Normal'
)

# Sodium Disturbances
df_heart_fcr['Hyponatremia'] = np.where(df_heart_fcr['serum_sodium'] < 135, 'Hyponatremia', 'Normal')
df_heart_fcr['Hypernatremia'] = np.where(df_heart_fcr['serum_sodium'] > 145, 'Hypernatremia', 'Normal')

# Platelet Disturbances
df_heart_fcr['Thrombocytopenia'] = np.where(df_heart_fcr['platelets'] < 150000, 'Thrombocytopenia', 'Normal')
df_heart_fcr['Thrombocytosis'] = np.where(df_heart_fcr['platelets'] > 450000, 'Thrombocytosis', 'Normal')

# Complex Syndromes & Risk Triads
df_heart_fcr['Cardiorenal_Syndrome_Risk'] = np.where(
    (df_heart_fcr['ejection_fraction'] < 40) & (df_heart_fcr['serum_creatinine'] > 1.5),
    'High',
    'Low'
)

df_heart_fcr['Vascular_Metabolic_Risk_Triad'] = np.where(
    (df_heart_fcr['diabetes']) & (df_heart_fcr['high_blood_pressure']) & (df_heart_fcr['smoking']),
    'High',
    'Low'
)

# Aggregations & Reports
print("--- Cardiorenal Syndrome Mortality Summary ---")
cardiorenal = df_heart_fcr.groupby('Cardiorenal_Syndrome_Risk')['DEATH_EVENT'].agg(['count', 'mean'])
cardiorenal['Mortality_Percentage'] = cardiorenal['mean'] * 100
print(f"{cardiorenal}\n")

print("--- Pivot Table: Renal Impairment vs EF Category (Mortality %) ---")
pivot_risk = pd.pivot_table(
    df_heart_fcr,
    values='DEATH_EVENT',
    index='Renal_Impairment',
    columns='EF_Category',
    aggfunc='mean'
)
print(f"{pivot_risk * 100}\n")

# Export Processed Data
key = input("--- Type Y to export or S to Show ---\n :")
if key.lower() == 'y':
    PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_heart_failure_records.csv')
    df_heart_fcr.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved processed dataset to: {PROCESSED_PATH}")
else:
    print("--- Have a nice life ---")