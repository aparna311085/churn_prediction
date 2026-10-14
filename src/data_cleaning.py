"""
data_cleaning.py
----------------
Loads the raw Telco Customer Churn CSV, performs cleaning and data quality
checks, then saves the cleaned dataset for downstream use.

Real columns (from actual file inspection):
  customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
  PhoneService, MultipleLines, InternetService, OnlineSecurity,
  OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
  StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
  MonthlyCharges, TotalCharges, Churn
"""

import os
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH    = os.path.join(ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
CLEAN_PATH  = os.path.join(ROOT, "data", "cleaned_churn.csv")


def load_raw() -> pd.DataFrame:
    """Load the raw CSV file."""
    df = pd.read_csv(RAW_PATH)
    print(f"[INFO] Raw dataset shape: {df.shape}")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a full data quality report."""
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"\nShape              : {df.shape[0]} rows  x  {df.shape[1]} columns")
    print(f"Duplicate rows     : {df.duplicated().sum()}")

    print("\n--- Column dtypes ---")
    print(df.dtypes.to_string())

    print("\n--- Missing / blank values ---")
    missing = df.isnull().sum()
    # TotalCharges stored as object with blank strings
    blank_tc = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    print(missing.to_string())
    print(f"\n[NOTE] 'TotalCharges' blank-string cells (hidden NaN): {blank_tc}")

    print("\n--- Target distribution (Churn) ---")
    print(df["Churn"].value_counts(dropna=False))
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean() * 100:.1f} %")
    print("=" * 60 + "\n")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps:
    1. Drop customerID (non-predictive identifier)
    2. Fix TotalCharges: coerce blank strings -> NaN -> impute with median
    3. Encode target Churn -> 0/1 integer
    4. Drop duplicate rows (none found, but good hygiene)
    """
    df = df.copy()

    # 1. Drop non-predictive identifier
    df.drop(columns=["customerID"], inplace=True)
    print("[CLEAN] Dropped 'customerID' column.")

    # 2. Fix TotalCharges (stored as object with blank strings)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_nan = df["TotalCharges"].isnull().sum()
    if n_nan > 0:
        median_tc = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
        print(f"[CLEAN] Imputed {n_nan} blank TotalCharges values with median ({median_tc:.2f}).")

    # 3. Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    print("[CLEAN] Encoded 'Churn': Yes=1, No=0.")

    # 4. Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"[CLEAN] Removed {before - after} duplicate row(s).")

    print(f"\n[CLEAN] Final clean shape: {df.shape}")
    return df


def save_clean(df: pd.DataFrame) -> None:
    df.to_csv(CLEAN_PATH, index=False)
    print(f"[SAVE] Cleaned dataset saved to: {CLEAN_PATH}")


def run_cleaning_pipeline() -> pd.DataFrame:
    """Full pipeline: load -> inspect -> clean -> save."""
    df_raw   = load_raw()
    inspect_data(df_raw)
    df_clean = clean(df_raw)
    save_clean(df_clean)
    return df_clean


if __name__ == "__main__":
    run_cleaning_pipeline()
