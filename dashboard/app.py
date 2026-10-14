"""
dashboard/app.py
----------------
Interactive Streamlit dashboard for the Customer Churn Prediction project.

Sections:
  1. Dataset Overview   – raw stats, distributions
  2. EDA Charts         – all saved visualizations
  3. Model Performance  – metrics, confusion matrix, ROC curve
  4. Predict Churn      – real-time single-customer prediction form
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ── Make src importable ────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.main-header p {
    color: #a0aec0;
    font-size: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.3rem;
}
.section-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e2e8f0;
    padding: 0.5rem 0;
    border-bottom: 2px solid #3b82f6;
    margin-bottom: 1.5rem;
}
.prediction-card-churn {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(239,68,68,0.3);
}
.prediction-card-no-churn {
    background: linear-gradient(135deg, #14532d, #166534);
    border: 1px solid #22c55e;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(34,197,94,0.3);
}
.prediction-label {
    font-size: 2rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.5rem;
}
.prediction-prob {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
}
.stSelectbox, .stSlider, .stNumberInput { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join(ROOT, "data", "cleaned_churn.csv")
RAW_PATH     = os.path.join(ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
EDA_DIR      = os.path.join(ROOT, "outputs", "eda")
MODEL_DIR    = os.path.join(ROOT, "outputs", "model")
CHART_DIR    = os.path.join(ROOT, "outputs", "model_charts")
RESULTS_PATH = os.path.join(MODEL_DIR, "evaluation_results.json")
PIPELINE_PATH= os.path.join(MODEL_DIR, "preprocessing_pipeline.joblib")
MODEL_PATH   = os.path.join(MODEL_DIR, "churn_mlp_model.keras")


# ── Loaders ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df_raw = pd.read_csv(RAW_PATH)
    return df, df_raw

@st.cache_resource
def load_pipeline():
    if os.path.exists(PIPELINE_PATH):
        return joblib.load(PIPELINE_PATH)
    return None

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        import tensorflow as tf
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        return tf.keras.models.load_model(MODEL_PATH)
    return None

@st.cache_data
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


def load_img(path):
    if os.path.exists(path):
        return Image.open(path)
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 Navigation")
    section = st.radio(
        "",
        ["🏠 Home & Dataset Overview",
         "📊 Exploratory Data Analysis",
         "🤖 Model Performance",
         "🔮 Predict Customer Churn"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Project Info**")
    st.caption("Dataset: IBM Telco Churn")
    st.caption("Model: MLP (Keras/TF)")
    st.caption("7,043 customers · 20 features")
    st.markdown("---")

    PDF_PATH = os.path.join(ROOT, "Customer_Churn_Prediction_Project_Report.pdf")
    if os.path.exists(PDF_PATH):
        with open(PDF_PATH, "rb") as pdf_file:
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_file,
                file_name="Customer_Churn_Prediction_Project_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    st.caption("© 2026 Customer Churn Prediction")


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📡 Customer Churn Prediction</h1>
    <p>Exploratory Data Analysis &amp; Deep Learning · IBM Telco Dataset · 7,043 customers</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SECTION 1: HOME & DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════
if section == "🏠 Home & Dataset Overview":
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

    df, df_raw = load_data()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    n_churn    = int(df["Churn"].sum())
    n_no_churn = int((df["Churn"] == 0).sum())
    churn_rate = df["Churn"].mean() * 100

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">7,043</div>
            <div class="metric-label">Total Customers</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">20</div>
            <div class="metric-label">Features</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{n_churn:,}</div>
            <div class="metric-label">Churned Customers</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-label">Churn Rate</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Data preview
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("#### 👁️ Raw Data Preview")
        st.dataframe(df_raw.head(20), use_container_width=True, height=350)

    with col_b:
        st.markdown("#### 📑 Column Types")
        dtype_df = pd.DataFrame({
            "Column": df_raw.dtypes.index,
            "Type": df_raw.dtypes.values.astype(str)
        })
        st.dataframe(dtype_df, use_container_width=True, height=350)

    # Interactive churn distribution
    st.markdown("#### 📊 Churn Distribution (Interactive)")
    counts = df["Churn"].value_counts().rename({0: "No Churn", 1: "Churned"})
    fig_pie = px.pie(
        names=counts.index, values=counts.values,
        color_discrete_sequence=["#4A90D9", "#E74C3C"],
        hole=0.45
    )
    fig_pie.update_traces(textposition="outside", textinfo="percent+label")
    fig_pie.update_layout(
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=380
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Summary stats
    st.markdown("#### 📈 Descriptive Statistics (Numeric Columns)")
    numeric_cols = df.select_dtypes(include="number")
    st.dataframe(numeric_cols.describe().round(2), use_container_width=True)

    # Missing / data quality
    st.markdown("#### ✅ Data Quality Summary")
    q_col1, q_col2, q_col3 = st.columns(3)
    q_col1.metric("Missing Values (after cleaning)", "0")
    q_col2.metric("Duplicate Rows", "0")
    q_col3.metric("TotalCharges Blank Strings Fixed", "11")


# ═══════════════════════════════════════════════════════════
# SECTION 2: EDA CHARTS
# ═══════════════════════════════════════════════════════════
elif section == "📊 Exploratory Data Analysis":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    df, _ = load_data()

    CHARTS = [
        ("01_churn_distribution.png",   "1. Churn Class Distribution"),
        ("02_churn_by_contract.png",     "2. Churn Rate by Contract Type"),
        ("03_churn_by_internet.png",     "3. Churn by Internet Service"),
        ("04_tenure_distribution.png",   "4. Tenure Distribution by Churn"),
        ("05_monthly_charges_violin.png","5. Monthly Charges vs Churn"),
        ("06_correlation_heatmap.png",   "6. Correlation Heatmap"),
        ("07_senior_citizen_churn.png",  "7. Senior Citizen vs Churn"),
        ("08_payment_method_churn.png",  "8. Churn Rate by Payment Method"),
        ("09_service_features_churn.png","9. Service Subscription vs Churn"),
    ]

    for i in range(0, len(CHARTS), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(CHARTS):
                fname, title = CHARTS[i + j]
                img = load_img(os.path.join(EDA_DIR, fname))
                with col:
                    st.markdown(f"**{title}**")
                    if img:
                        st.image(img, use_container_width=True)
                    else:
                        st.warning(f"Chart not found: {fname}. Please run the pipeline first.")

    # Interactive EDA
    st.markdown("---")
    st.markdown("#### 🔬 Interactive Explorer")
    df_plot = df.copy()
    df_plot["Churn"] = df_plot["Churn"].map({0: "No Churn", 1: "Churned"})

    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("X Axis", ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"], key="xax")
    with c2:
        color_by = st.selectbox("Color By", ["Churn", "Contract", "InternetService"], key="colax")

    fig_scatter = px.histogram(
        df_plot, x=x_col, color=color_by, barmode="overlay",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.75, nbins=40
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=400
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# SECTION 3: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════
elif section == "🤖 Model Performance":
    st.markdown('<div class="section-header">🤖 Model Performance</div>', unsafe_allow_html=True)

    results = load_results()
    if results is None:
        st.error("❌ Model not trained yet. Please run: `python run_pipeline.py`")
        st.stop()

    # Metric cards
    st.markdown("#### 📊 Evaluation Metrics (Test Set)")
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("Accuracy",  results.get("accuracy",  0), m1),
        ("Precision", results.get("precision", 0), m2),
        ("Recall",    results.get("recall",    0), m3),
        ("F1 Score",  results.get("f1_score",  0), m4),
        ("ROC-AUC",   results.get("roc_auc",   0), m5),
    ]
    for label, val, col in metrics:
        with col:
            color = "#22c55e" if val >= 0.80 else "#f59e0b" if val >= 0.70 else "#ef4444"
            col.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{color}">{val:.4f}</div>
                <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("#### 📉 Training Curves")
        img = load_img(os.path.join(CHART_DIR, "training_curves.png"))
        if img: st.image(img, use_container_width=True)
        else: st.warning("Run pipeline to generate charts.")

    with chart_col2:
        st.markdown("#### 🎯 Confusion Matrix")
        img = load_img(os.path.join(CHART_DIR, "confusion_matrix.png"))
        if img: st.image(img, use_container_width=True)
        else: st.warning("Run pipeline to generate charts.")

    st.markdown("#### 📈 ROC Curve")
    img = load_img(os.path.join(CHART_DIR, "roc_curve.png"))
    if img: st.image(img, use_container_width=True)
    else: st.warning("Run pipeline to generate charts.")

    st.markdown("#### 🔍 Feature Importance")
    img = load_img(os.path.join(CHART_DIR, "feature_importance.png"))
    if img: st.image(img, use_container_width=True)
    else: st.warning("Run pipeline to generate charts.")


# ═══════════════════════════════════════════════════════════
# SECTION 4: PREDICT CHURN
# ═══════════════════════════════════════════════════════════
elif section == "🔮 Predict Customer Churn":
    st.markdown('<div class="section-header">🔮 Real-Time Customer Churn Prediction</div>',
                unsafe_allow_html=True)
    st.markdown("Fill in the customer details below and click **Predict** to see the churn probability.")

    pipeline = load_pipeline()
    model    = load_model()

    if pipeline is None or model is None:
        st.error("❌ Model or pipeline not found. Please run: `python run_pipeline.py`")
        st.stop()

    with st.form("prediction_form"):
        st.markdown("#### 👤 Customer Demographics")
        d1, d2, d3 = st.columns(3)
        gender       = d1.selectbox("Gender",           ["Male", "Female"])
        senior       = d2.selectbox("Senior Citizen",   ["No", "Yes"])
        partner      = d3.selectbox("Has Partner",      ["Yes", "No"])

        d4, d5 = st.columns(2)
        dependents   = d4.selectbox("Has Dependents",   ["No", "Yes"])
        tenure       = d5.slider("Tenure (months)", 0, 72, 12)

        st.markdown("#### 📞 Phone & Internet Services")
        s1, s2, s3 = st.columns(3)
        phone_service   = s1.selectbox("Phone Service",    ["Yes", "No"])
        multiple_lines  = s2.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
        internet_svc    = s3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        s4, s5, s6 = st.columns(3)
        online_security = s4.selectbox("Online Security",  ["No", "Yes", "No internet service"])
        online_backup   = s5.selectbox("Online Backup",    ["No", "Yes", "No internet service"])
        device_protect  = s6.selectbox("Device Protection",["No", "Yes", "No internet service"])

        s7, s8, s9 = st.columns(3)
        tech_support    = s7.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
        streaming_tv    = s8.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
        streaming_movies= s9.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown("#### 💳 Account & Billing")
        a1, a2, a3 = st.columns(3)
        contract     = a1.selectbox("Contract",         ["Month-to-month", "One year", "Two year"])
        paperless    = a2.selectbox("Paperless Billing", ["Yes", "No"])
        payment      = a3.selectbox("Payment Method",   [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

        a4, a5 = st.columns(2)
        monthly_charges = a4.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
        total_charges   = a5.number_input("Total Charges ($)",   0.0, 10000.0, float(tenure * monthly_charges), step=1.0)

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True, type="primary")

    if submitted:
        # Build input DataFrame matching the exact column order expected by the pipeline
        input_data = pd.DataFrame([{
            "gender"          : gender,
            "SeniorCitizen"   : 1 if senior == "Yes" else 0,
            "Partner"         : partner,
            "Dependents"      : dependents,
            "tenure"          : tenure,
            "PhoneService"    : phone_service,
            "MultipleLines"   : multiple_lines,
            "InternetService" : internet_svc,
            "OnlineSecurity"  : online_security,
            "OnlineBackup"    : online_backup,
            "DeviceProtection": device_protect,
            "TechSupport"     : tech_support,
            "StreamingTV"     : streaming_tv,
            "StreamingMovies" : streaming_movies,
            "Contract"        : contract,
            "PaperlessBilling": paperless,
            "PaymentMethod"   : payment,
            "MonthlyCharges"  : monthly_charges,
            "TotalCharges"    : total_charges,
        }])

        # Preprocess
        X_proc = pipeline.transform(input_data)

        # Predict
        prob   = float(model.predict(X_proc, verbose=0)[0][0])
        label  = "Will Churn" if prob >= 0.5 else "Will NOT Churn"
        is_churn = prob >= 0.5

        st.markdown("---")
        st.markdown("#### 📋 Prediction Result")

        card_class = "prediction-card-churn" if is_churn else "prediction-card-no-churn"
        icon       = "🚨" if is_churn else "✅"
        st.markdown(f"""
        <div class="{card_class}">
            <div class="prediction-label">{icon} {label}</div>
            <div class="prediction-prob">Churn Probability: <strong>{prob*100:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Churn Probability (%)", "font": {"color": "#e2e8f0"}},
            number={"suffix": "%", "font": {"color": "#e2e8f0"}},
            gauge={
                "axis"  : {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar"   : {"color": "#ef4444" if is_churn else "#22c55e"},
                "bgcolor": "#1e293b",
                "steps" : [
                    {"range": [0, 40], "color": "#14532d"},
                    {"range": [40, 60], "color": "#713f12"},
                    {"range": [60, 100], "color": "#7f1d1d"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 50}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=300
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if is_churn:
            st.warning("⚠️ **Recommendation**: This customer is at high risk of churning. "
                       "Consider offering a discount, plan upgrade, or proactive support call.")
        else:
            st.success("✅ **Recommendation**: This customer appears stable. "
                       "Continue monitoring monthly usage patterns.")
