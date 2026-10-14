"""
preprocessing.py
----------------
Builds a reproducible scikit-learn preprocessing pipeline that:
  - Imputes missing values (TotalCharges already handled in cleaning, but kept as safety)
  - Encodes categorical features with OneHotEncoder
  - Scales numeric features with StandardScaler
  - Splits data into train (70%), validation (15%), and test (15%) sets
  - Saves the fitted pipeline (joblib) for inference in the Streamlit dashboard

No data leakage: the pipeline is fit ONLY on the training set.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "cleaned_churn.csv")
MODEL_DIR = os.path.join(ROOT, "outputs", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

PIPELINE_PATH = os.path.join(MODEL_DIR, "preprocessing_pipeline.joblib")
SPLITS_DIR    = os.path.join(ROOT, "data")

# ── Feature definitions (based on actual dataset columns) ────────────────────
# customerID was dropped in cleaning. Churn is the target.
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

TARGET = "Churn"


def build_pipeline() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Scales numeric columns with StandardScaler
    - One-hot encodes categorical columns (drop='first' to avoid dummy trap)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract feature names after transformation for interpretability."""
    cat_encoder = preprocessor.named_transformers_["cat"]["onehot"]
    cat_names   = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERIC_FEATURES + cat_names


def run_preprocessing() -> dict:
    """
    Load cleaned data, split into train / val / test, fit the pipeline on
    train only, transform all splits, and save everything.
    Returns a dict with X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"[PREP] Loaded: {df.shape}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].values

    # ── 2. Split: 70 / 15 / 15 ───────────────────────────────────────────────
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Second split: 50% val, 50% test from the 30% temp -> 15% each
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"[PREP] Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"[PREP] Train churn rate : {y_train.mean():.3f}")
    print(f"[PREP] Val   churn rate : {y_val.mean():.3f}")
    print(f"[PREP] Test  churn rate : {y_test.mean():.3f}")

    # ── 3. Build & fit pipeline on TRAIN only ─────────────────────────────────
    preprocessor = build_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)   # <-- fit here ONLY
    X_val_proc   = preprocessor.transform(X_val)
    X_test_proc  = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)
    print(f"[PREP] Features after encoding: {len(feature_names)}")

    # ── 4. Save pipeline ───────────────────────────────────────────────────────
    joblib.dump(preprocessor, PIPELINE_PATH)
    print(f"[PREP] Pipeline saved: {PIPELINE_PATH}")

    # ── 5. Save splits as numpy arrays ────────────────────────────────────────
    np.save(os.path.join(SPLITS_DIR, "X_train.npy"), X_train_proc)
    np.save(os.path.join(SPLITS_DIR, "X_val.npy"),   X_val_proc)
    np.save(os.path.join(SPLITS_DIR, "X_test.npy"),  X_test_proc)
    np.save(os.path.join(SPLITS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SPLITS_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(SPLITS_DIR, "y_test.npy"),  y_test)

    # Save metadata
    meta = {
        "n_features"       : len(feature_names),
        "feature_names"    : feature_names,
        "train_size"       : int(len(y_train)),
        "val_size"         : int(len(y_val)),
        "test_size"        : int(len(y_test)),
        "train_churn_rate" : float(y_train.mean()),
        "val_churn_rate"   : float(y_val.mean()),
        "test_churn_rate"  : float(y_test.mean()),
    }
    with open(os.path.join(MODEL_DIR, "preprocessing_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[PREP] Metadata saved.")

    return {
        "X_train": X_train_proc, "X_val": X_val_proc, "X_test": X_test_proc,
        "y_train": y_train,      "y_val": y_val,      "y_test": y_test,
        "preprocessor": preprocessor, "feature_names": feature_names
    }


if __name__ == "__main__":
    run_preprocessing()
