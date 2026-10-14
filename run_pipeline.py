"""
run_pipeline.py
---------------
Master script that runs the complete project pipeline in order:
  Step 1: Data Cleaning
  Step 2: EDA
  Step 3: Preprocessing & Splits
  Step 4: Feature Importance
  Step 5: Model Training & Evaluation

Run from the project root:
  python run_pipeline.py
"""

import sys
import os

# Make sure src is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_cleaning      import run_cleaning_pipeline
from eda                import run_eda
from preprocessing      import run_preprocessing
from feature_importance import run_feature_importance
from model              import run_model_pipeline


BANNER = """
============================================================
  CUSTOMER CHURN PREDICTION - COMPLETE PIPELINE
  Dataset: IBM Telco Customer Churn (7043 rows x 21 cols)
============================================================
"""

STEPS = [
    ("1. Data Cleaning",        run_cleaning_pipeline),
    ("2. EDA & Visualizations", run_eda),
    ("3. Preprocessing",        run_preprocessing),
    ("4. Feature Importance",   run_feature_importance),
    ("5. Model Training",       run_model_pipeline),
]


def run_all():
    print(BANNER)
    for name, fn in STEPS:
        print(f"\n{'='*60}")
        print(f"  STEP: {name}")
        print(f"{'='*60}")
        fn()

    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETE!")
    print("  Outputs are in: outputs/eda/  and  outputs/model/")
    print("  To launch the dashboard: streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
