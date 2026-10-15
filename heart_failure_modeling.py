"""Statistical test and a baseline predictive model for the heart failure dataset.

Run:  python heart_failure_modeling.py [--data FILE.csv]
Outputs: printed results + figures saved in ./figures/<dataset name>_*.png

Data note: the ORIGINAL UCI file has 299 unique patients. The extended Kaggle file has 5,000
rows but only 1,320 unique rows (many exact copies). Duplicates would leak between training and
test sets and make every p-value look extremely small, so exact duplicates are always removed.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RANDOM_STATE = 42

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--data",
    default="heart_failure_clinical_records_dataset.csv",
    help="CSV file next to this script (default: the original 299-patient UCI file)",
)
args = parser.parse_args()
TAG = os.path.splitext(os.path.basename(args.data))[0]

# 1. Load and de-duplicate -------------------------------------------------------------
raw = pd.read_csv(os.path.join(BASE_DIR, args.data))
df = raw.drop_duplicates().reset_index(drop=True)
print(f"Dataset: {args.data}")
print(f"Rows in file: {len(raw)} | exact duplicates: {len(raw) - len(df)} | unique rows used: {len(df)}")
print(f"Mortality (unique rows): {df['DEATH_EVENT'].mean():.1%}\n")

# 2. Cardiorenal risk flag (same definition as the main script) --------------------------
df["cardiorenal_flag"] = (df["ejection_fraction"] < 40) & (df["serum_creatinine"] > 1.5)
group = df.groupby("cardiorenal_flag")["DEATH_EVENT"].agg(["count", "mean"])
group["mean"] = (group["mean"] * 100).round(1)
group.columns = ["rows", "mortality_%"]
print("Mortality by cardiorenal risk flag (unique rows):")
print(group.rename(index={False: "Low", True: "High"}).to_string(), "\n")

# 3. Chi-square test of association ------------------------------------------------------
table = pd.crosstab(df["cardiorenal_flag"], df["DEATH_EVENT"])
chi2, p_value, _, _ = chi2_contingency(table)
a, b = table.loc[True, 1], table.loc[True, 0]
c, d = table.loc[False, 1], table.loc[False, 0]
odds_ratio = (a / b) / (c / d)
print(f"Chi-square test (flag vs death): chi2 = {chi2:.1f}, p = {p_value:.2e}, odds ratio = {odds_ratio:.1f}\n")

# 4. Logistic regression baseline ------------------------------------------------------
# 'time' (follow-up days) is excluded: patients who died have shorter follow-up, so using it
# as a predictor leaks the outcome.
features = [c for c in raw.columns if c not in ("DEATH_EVENT", "time")]
X, y = df[features], df["DEATH_EVENT"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_tr, y_tr)
test_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
cv_auc = cross_val_score(
    model, X, y, scoring="roc_auc",
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
)
print(f"Logistic regression, test-set AUC = {test_auc:.3f}")
print(f"5-fold CV AUC = {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})\n")

coefs = pd.Series(model[-1].coef_[0], index=features).sort_values(key=abs, ascending=False)
print("Standardised coefficients (larger magnitude = stronger association):")
print(coefs.round(2).to_string(), "\n")

# 5. Figures -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
for outcome, label in [(0, "Survived"), (1, "Died")]:
    ax.hist(df.loc[df["DEATH_EVENT"] == outcome, "ejection_fraction"], bins=20, alpha=0.6, label=label)
ax.axvline(40, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Ejection fraction (%)"); ax.set_ylabel("Number of rows"); ax.legend()
ax.set_title("Ejection fraction by outcome (EF < 40% = HFrEF)")
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, f"{TAG}_ef_by_outcome.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.5, 4))
rates = df.groupby("cardiorenal_flag")["DEATH_EVENT"].mean() * 100
ax.bar(["Low", "High"], rates.values, color=["#4C72B0", "#C44E52"])
ax.set_ylabel("Mortality (%)"); ax.set_title("Mortality by cardiorenal risk flag")
for i, v in enumerate(rates.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center")
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, f"{TAG}_mortality_by_flag.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 5))
RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax)
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_title(f"ROC curve (test AUC = {test_auc:.2f})")
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, f"{TAG}_roc_curve.png"), dpi=150); plt.close(fig)
print(f"Figures saved in: {FIG_DIR}")
