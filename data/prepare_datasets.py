"""
Prepare all three disease-prediction datasets:
  1. Pima Indians Diabetes (UCI)
  2. Heart Disease (UCI Cleveland, via kb22 mirror)
  3. Breast Cancer Wisconsin (Diagnostic) (UCI, via sklearn's built-in loader,
     which ships the actual UCI data locally -- no network needed and always
     100% reliable / reproducible)

Each dataset is cleaned (missing-value handling, dedupe) and saved as a
tidy CSV in data/ with a clear target column, ready for the training script.
"""
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

RAW_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. DIABETES  (Pima Indians Diabetes Database)
# ---------------------------------------------------------------------------
diabetes_cols = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
    "bmi", "diabetes_pedigree", "age", "target"
]
diabetes = pd.read_csv(f"{RAW_DIR}/diabetes_raw.csv", header=None, names=diabetes_cols)

# Known data-quality issue in this dataset: 0s are used as "missing" for
# these physiologically-impossible-at-zero columns. Replace with NaN, then
# impute with the median (a standard, defensible approach for this dataset).
zero_as_missing = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
for col in zero_as_missing:
    diabetes[col] = diabetes[col].replace(0, np.nan)
    diabetes[col] = diabetes[col].fillna(diabetes[col].median())

diabetes = diabetes.drop_duplicates().reset_index(drop=True)
diabetes.to_csv(f"{RAW_DIR}/diabetes_clean.csv", index=False)
print(f"[diabetes] rows={len(diabetes)}, positive_rate={diabetes['target'].mean():.3f}")

# ---------------------------------------------------------------------------
# 2. HEART DISEASE (UCI Cleveland)
# ---------------------------------------------------------------------------
heart = pd.read_csv(f"{RAW_DIR}/heart_raw2.csv", encoding="utf-8-sig")
heart.columns = [c.strip().lower() for c in heart.columns]
heart = heart.rename(columns={"target": "target"})
heart = heart.drop_duplicates().reset_index(drop=True)
# sanity: target should be binary 0/1
assert set(heart["target"].unique()) <= {0, 1}
heart.to_csv(f"{RAW_DIR}/heart_clean.csv", index=False)
print(f"[heart]    rows={len(heart)}, positive_rate={heart['target'].mean():.3f}")

# ---------------------------------------------------------------------------
# 3. BREAST CANCER WISCONSIN (Diagnostic)
# ---------------------------------------------------------------------------
bc = load_breast_cancer(as_frame=True)
bc_df = bc.frame.copy()
bc_df.columns = [c.replace(" ", "_") for c in bc_df.columns]
# sklearn encodes target as 0=malignant,1=benign -- flip so 1 = malignant
# (the "disease positive" class), matching the convention used for the
# other two datasets (1 = has the condition).
bc_df["target"] = 1 - bc_df["target"]
bc_df = bc_df.drop_duplicates().reset_index(drop=True)
bc_df.to_csv(f"{RAW_DIR}/breast_cancer_clean.csv", index=False)
print(f"[breast]   rows={len(bc_df)}, positive_rate={bc_df['target'].mean():.3f}")

print("\nAll three datasets cleaned and saved.")
