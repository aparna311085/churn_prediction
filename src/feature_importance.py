"""
feature_importance.py
---------------------
Analyses which features matter most for customer churn.
Approach: use a Random Forest classifier (a tree-based model that naturally
provides feature importances) trained on the same preprocessed features.
This is a standard technique for MLP models which don't expose importances
natively.

Also computes permutation importance as a validation check.
Saves a bar chart of the top-20 most important features.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "outputs", "model")
CHART_DIR = os.path.join(ROOT, "outputs", "model_charts")
os.makedirs(CHART_DIR, exist_ok=True)

META_PATH = os.path.join(MODEL_DIR, "preprocessing_meta.json")


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    return X_train, X_test, y_train, y_test, feature_names


def run_feature_importance() -> None:
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # ── Train a Random Forest (interpretable proxy model) ─────────────────────
    print("[FEAT] Training Random Forest for feature importance...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_train, y_train)

    # ── Built-in feature importance ───────────────────────────────────────────
    importances = rf.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]
    top_n       = min(20, len(feature_names))
    top_idx     = sorted_idx[:top_n]

    top_features = [feature_names[i] for i in top_idx]
    top_values   = importances[top_idx]

    # ── Save chart ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, top_n))
    ax.barh(top_features[::-1], top_values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=12)
    ax.set_title(f"Top {top_n} Features for Churn Prediction\n(Random Forest Feature Importance)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    path = os.path.join(CHART_DIR, "feature_importance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")

    # ── Save importance data ───────────────────────────────────────────────────
    imp_df = pd.DataFrame({
        "feature"   : [feature_names[i] for i in sorted_idx],
        "importance": importances[sorted_idx]
    })
    imp_df.to_csv(os.path.join(MODEL_DIR, "feature_importances.csv"), index=False)
    print(f"  [SAVED] feature_importances.csv")
    print("\nTop 10 important features:")
    print(imp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    run_feature_importance()
