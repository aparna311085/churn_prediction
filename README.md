# 📡 Customer Churn Prediction Using EDA and Deep Learning

> **College Project** | IBM Telco Customer Churn Dataset | TensorFlow/Keras MLP | Streamlit Dashboard

---

## 📋 Project Overview

This project builds an end-to-end machine learning pipeline to predict customer churn for a telecommunications company. It includes data cleaning, exploratory data analysis (EDA), a deep learning model (Multi-Layer Perceptron using TensorFlow/Keras), and an interactive Streamlit dashboard.

**Dataset:** IBM Watson Telco Customer Churn  
**Rows:** 7,043 customers | **Columns:** 21 | **Target:** `Churn` (Yes/No)

---

## 📁 Project Structure

```
churn_prediction/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  ← Raw dataset
│   ├── cleaned_churn.csv                      ← After cleaning
│   ├── X_train.npy / X_val.npy / X_test.npy  ← Preprocessed splits
│   └── y_train.npy / y_val.npy / y_test.npy
│
├── src/
│   ├── data_cleaning.py       ← Step 1: Clean & quality check
│   ├── eda.py                 ← Step 2: 9 visualizations
│   ├── preprocessing.py       ← Step 3: Encode + scale + split
│   ├── feature_importance.py  ← Step 4: Feature analysis
│   └── model.py               ← Step 5: Train & evaluate MLP
│
├── dashboard/
│   └── app.py                 ← Streamlit interactive dashboard
│
├── outputs/
│   ├── eda/                   ← All EDA chart PNGs
│   └── model/
│       ├── churn_mlp_model.keras
│       ├── preprocessing_pipeline.joblib
│       ├── evaluation_results.json
│       └── model_charts/      ← Training curves, ROC, confusion matrix
│
├── run_pipeline.py            ← Master script (runs everything)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Step 1 – Install dependencies
```powershell
pip install -r requirements.txt
```

### Step 2 – Run the complete pipeline
```powershell
python run_pipeline.py
```
This single command runs all 5 stages:
1. Data cleaning & quality checks
2. EDA (9 charts saved to `outputs/eda/`)
3. Preprocessing & train/val/test splits
4. Feature importance analysis
5. Model training, tuning & evaluation

### Step 3 – Launch the dashboard
```powershell
streamlit run dashboard/app.py
```
Open your browser at: **http://localhost:8501**

---

## 📊 Dataset Columns

| Column | Type | Description |
|---|---|---|
| `customerID` | string | Unique customer identifier (dropped before modelling) |
| `gender` | categorical | Male / Female |
| `SeniorCitizen` | binary | 1 = Senior citizen |
| `Partner` | categorical | Has a partner |
| `Dependents` | categorical | Has dependents |
| `tenure` | numeric | Months with the company |
| `PhoneService` | categorical | Has phone service |
| `MultipleLines` | categorical | Has multiple lines |
| `InternetService` | categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | categorical | Add-on service |
| `OnlineBackup` | categorical | Add-on service |
| `DeviceProtection` | categorical | Add-on service |
| `TechSupport` | categorical | Add-on service |
| `StreamingTV` | categorical | Add-on service |
| `StreamingMovies` | categorical | Add-on service |
| `Contract` | categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | categorical | Uses paperless billing |
| `PaymentMethod` | categorical | Payment type |
| `MonthlyCharges` | numeric | Monthly bill amount |
| `TotalCharges` | numeric | Total amount charged (had 11 blank values — fixed) |
| `Churn` | **target** | Yes / No → encoded as 1 / 0 |

---

## 🧠 Model Architecture

```
Input (N features)
    ↓
Dense(128, ReLU) + BatchNorm + Dropout(0.4) + L2 regularization
    ↓
Dense(64, ReLU) + BatchNorm + Dropout(0.3) + L2 regularization
    ↓
Dense(32, ReLU) + Dropout(0.2)
    ↓
Dense(1, Sigmoid) → Churn Probability
```

**Training config:**
- Optimizer: Adam (lr=1e-3)
- Loss: Binary Cross-Entropy
- Callbacks: EarlyStopping (patience=10) + ReduceLROnPlateau + ModelCheckpoint
- Class weights: Computed automatically to handle 73/27 imbalance
- Max epochs: 100 (early stopping determines actual number)

---

## 📈 Key Findings (from EDA)

1. **Month-to-month contracts** have the highest churn rate (~42%)
2. **Fiber optic** customers churn more than DSL customers
3. **Short-tenure customers** (< 12 months) are the highest risk
4. **Electronic check** payment method correlates with higher churn
5. **Senior citizens** churn at a higher rate than non-senior customers
6. **Monthly charges > $65** shows significantly higher churn

---

## 🔧 Data Cleaning Notes

| Issue | Fix Applied |
|---|---|
| `TotalCharges` stored as `object` (string) | Converted with `pd.to_numeric(errors='coerce')` |
| 11 blank-string cells in `TotalCharges` | Imputed with column median |
| `customerID` non-predictive | Dropped before modelling |
| `Churn` as Yes/No string | Mapped to 1/0 integer |

---

## 📚 Requirements

```
tensorflow>=2.13.0
scikit-learn>=1.6.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.13.0
plotly>=5.14.0
streamlit>=1.45.0
joblib>=1.3.0
```

---

## 👩‍💻 How to Fix Common Errors

| Error | Fix |
|---|---|
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow` |
| `No module named streamlit` | `pip install streamlit` |
| `FileNotFoundError: cleaned_churn.csv` | Run `python run_pipeline.py` first |
| `Model not found` | Run `python run_pipeline.py` first |
| Dashboard charts missing | Run pipeline — charts save to `outputs/` |
