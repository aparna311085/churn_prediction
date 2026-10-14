# Customer Churn Prediction Using Exploratory Data Analysis and Deep Learning

---

# 1. Title Page

| Academic Field | Information Details |
|---|---|
| **Project Title** | Customer Churn Prediction Using Exploratory Data Analysis and Deep Learning |
| **Student Name** | Project Candidate / Engineering Team |
| **Roll Number** | ENG-CS-2026-CHURN |
| **Department** | Department of Computer Science & Engineering |
| **Academic Year** | Final Year B.Tech / M.Tech (2025–2026) |
| **Course / Subject** | Applied Machine Learning & Deep Learning |
| **Date of Submission** | October 2026 |
| **Faculty Advisor / Guide** | Project Review Committee |

---

# 2. Abstract (100–150 words)

Customer churn presents a critical financial hurdle in the telecommunications industry, where acquisition costs exceed retention expenses by five to seven times. This project implements an end-to-end machine learning system to forecast customer attrition using the benchmark IBM Telco dataset of 7,043 customer accounts across 21 demographic and subscription attributes. Rigorous data auditing detected and resolved hidden whitespace missing entries in TotalCharges, followed by nine exploratory visualizations revealing tenure and contract longevity as major churn factors. A leakage-free ColumnTransformer pipeline prepared 30 encoded features for a regularized Multi-Layer Perceptron (MLP) deep neural network. Utilizing class-weighted binary cross-entropy and early stopping, the trained model achieved an overall accuracy of 75.81%, an ROC-AUC of 0.8520, and a high recall of 80.65%, successfully identifying eight out of every ten churning subscribers. The model is deployed via an interactive Streamlit dashboard enabling real-time risk scoring and mitigation analysis.

---

# 3. Introduction & Objective

### 3.1 Introduction
In modern telecommunications, market saturation limits organic customer acquisition. Providers face significant margin degradation when customers migrate to competing carriers. Early detection of customer disengagement allows companies to deploy proactive retention campaigns (e.g., targeted fee waivers, contract renegotiation, premium feature bundles). Deep neural networks, specifically Multi-Layer Perceptrons (MLPs), provide high representational capacity to model complex, multi-dimensional interactions between customer demographic characteristics, account tenure, subscribed digital services, and payment preferences.

### 3.2 Objectives
1. **Data Quality Assurance:** Audit the raw IBM dataset, detect silent whitespace anomalies, and eliminate non-predictive identifiers.
2. **Exploratory Data Analysis:** Formulate statistical visualizations identifying leading behavioral indicators of churn.
3. **Leakage-Free Preprocessing:** Engineer a reproducible `ColumnTransformer` pipeline fitted strictly on training data.
4. **Deep Learning Model Design:** Develop an MLP with Batch Normalization, Dropout, and $L_2$ weight regularization.
5. **Class Imbalance Handling:** Implement cost-sensitive class balancing to compensate for the 73.5% vs. 26.5% majority class dominance.
6. **Model Evaluation:** Measure performance using Accuracy, Recall, Precision, F1-Score, ROC-AUC, and Confusion Matrix.
7. **Production Deployment:** Build an interactive Streamlit web dashboard allowing instant single-customer inference.

---

# 4. Dataset Description

### 4.1 Overview & Metadata
* **Source:** IBM Watson Analytics Community (Telco Customer Churn benchmark).
* **Initial Dimensions:** 7,043 instances, 21 attributes.
* **Cleaned Dimensions:** 7,021 instances, 20 predictive features (after dropping `customerID` and removing 22 duplicates).
* **Target Column:** `Churn` (Binary: `Yes` = 1, `No` = 0).
* **Class Balance:** Retained: 5,174 (73.5%) · Churned: 1,869 (26.5%).

### 4.2 Attributes & Statistical Summary

| Feature Name | Type | Value Range / Categories | Mean / Distribution | Preprocessing Action |
|---|---|---|---|---|
| `tenure` | Numeric (int64) | 0 to 72 months | Mean: 32.37 months | `StandardScaler` (Z-Score) |
| `MonthlyCharges` | Numeric (float64) | $18.25 to $118.75 | Mean: $64.76 | `StandardScaler` (Z-Score) |
| `TotalCharges` | Numeric (float64) | $18.80 to $8,684.80 | Mean: $2,283.30 | Coerced 11 blanks to median ($1,397.47) |
| `SeniorCitizen` | Binary (int64) | 0 (No), 1 (Yes) | 16.2% Senior | Standardized / Scaled |
| `Contract` | Categorical | Month-to-month, 1-yr, 2-yr | 55.0% Month-to-month | `OneHotEncoder(drop='first')` |
| `InternetService` | Categorical | DSL, Fiber optic, No | 44.0% Fiber optic | `OneHotEncoder(drop='first')` |
| `PaymentMethod` | Categorical | 4 Electronic / Bank tiers | 33.6% Electronic check | `OneHotEncoder(drop='first')` |
| `TechSupport` | Categorical | Yes, No, No internet | 49.3% No Tech Support | `OneHotEncoder(drop='first')` |
| `OnlineSecurity`| Categorical | Yes, No, No internet | 49.7% No Security | `OneHotEncoder(drop='first')` |

---

# 5. EDA and Preprocessing

### 5.1 Preprocessing Methods Used (No Data Leakage)
1. **Missing Value Resolution:** Inspection revealed 11 blank spaces (`" "`) in `TotalCharges` belonging to new subscribers with `tenure == 0`. These were coerced to NaN and imputed using the median ($1,397.47).
2. **Identifier Removal:** `customerID` was removed as it carries no generalization signal.
3. **Target Encoding:** `Churn` was converted from `"Yes"`/`"No"` to binary `1`/`0`.
4. **Data Leakage Isolation:** The dataset was split into **70% Training (4,914)**, **15% Validation (1,053)**, and **15% Test (1,054)** using target-stratified sampling.
5. **ColumnTransformer Pipeline:**
   * **Numeric pipeline:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
   * **Categorical pipeline:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(drop='first')`.
   * The pipeline was fitted **strictly on the training partition** and then transformed across validation and test sets.

### 5.2 Key Analytical Insights
* **Contract Vulnerability:** Month-to-month contracts exhibit severe attrition (**42.7%**) compared to long-term commitments (**2.8%**).
* **Early Lifecycle Risk:** Customers in their first 6 months have the highest attrition density.
* **Service Friction:** Fiber optic customers churn at higher rates (**41.9%**) than DSL customers (**19.0%**), primarily correlated with high monthly billing ($70–$100+).

---

# 6. Data Visualization (Minimum 5 Visuals with Explanation)

All high-resolution figures are preserved in [`outputs/eda/`](file:///d:/churn_prediction/outputs/eda):

### Visual 1: Churn Class Distribution (`01_churn_distribution.png`)
* **Visualization:** Bar chart and pie chart displaying class counts.
* **Explanation:** Demonstrates class imbalance (73.5% retained vs. 26.5% churned). This finding dictated the implementation of cost-sensitive class weights during training to prevent majority-class bias.

### Visual 2: Churn Rate by Contract Type (`02_churn_by_contract.png`)
* **Visualization:** Horizontal bar plot of churn percentage per contract category.
* **Explanation:** Highlights that contract flexibility is the single largest predictor of churn. Month-to-month customers churn at 42.7%, whereas two-year contracts incur only 2.8% churn.

### Visual 3: Tenure Distribution by Churn Status (`04_tenure_distribution.png`)
* **Visualization:** Dual-overlay histogram displaying customer counts across tenure months.
* **Explanation:** Reveals a steep churn peak between months 1 and 6. The probability of churn decays exponentially after month 24, indicating that retention programs must focus heavily on customer onboarding.

### Visual 4: Monthly Charges Distribution (`05_monthly_charges_violin.png`)
* **Visualization:** Kernel density violin plot comparing bill charges across churn labels.
* **Explanation:** Churning customers have a significantly higher median monthly expenditure ($79.85) than retained customers ($64.40), confirming that price sensitivity drives customer defection.

### Visual 5: Churn Rate by Internet Service (`03_churn_by_internet.png`)
* **Visualization:** Grouped bar chart comparing DSL, Fiber optic, and No Internet customers.
* **Explanation:** Fiber optic subscribers experience an attrition rate of 41.9%, compared to 19.0% for DSL. Customers without internet service exhibit the lowest churn (7.4%).

### Visual 6: Payment Method vs Churn (`08_payment_method_churn.png`)
* **Visualization:** Comparative bar chart across payment modalities.
* **Explanation:** Electronic check users exhibit an elevated 45.3% churn rate, while customers using automatic payment deductions (credit card or bank transfer) churn at under 17%.

---

# 7. Deep Learning Model

### 7.1 Neural Network Architecture
* **Framework:** TensorFlow 2.x / Keras
* **Input Dimension:** 30 encoded features

```
Input (30 features)
   ↓
Dense (128 units, ReLU activation, L2 Regularization 1e-4)
   ↓
Batch Normalization (stabilizes mini-batch distributions)
   ↓
Dropout (Rate = 0.40)
   ↓
Dense (64 units, ReLU activation, L2 Regularization 1e-4)
   ↓
Batch Normalization
   ↓
Dropout (Rate = 0.30)
   ↓
Dense (32 units, ReLU activation)
   ↓
Dropout (Rate = 0.20)
   ↓
Dense (1 unit, Sigmoid activation) → P(Churn ∈ [0, 1])
```

### 7.2 Training Parameters & Hyperparameters

| Hyperparameter | Value | Implementation Purpose |
|---|---|---|
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adaptive moment estimation |
| **Learning Rate** | $0.001$ ($10^{-3}$) | Base rate with `ReduceLROnPlateau` decay |
| **Loss Function** | Binary Cross-Entropy | Logarithmic loss for Bernoulli distribution |
| **Batch Size** | 64 | Gradient stability on tabular sample |
| **Epochs** | 100 max | Controlled by Early Stopping |
| **Class Weights** | `{0: 0.68, 1: 1.89}` | Penalizes false negatives 2.78× higher |
| **Early Stopping** | `patience=10`, `restore_best_weights=True`| Halted at Epoch 24, restored Epoch 14 |
| **ReduceLROnPlateau** | `factor=0.5, patience=5` | Halves LR when validation loss plateaus |

---

# 8. Result Visualization & Interpretation

### 8.1 Measured Test Set Evaluation (1,054 Unseen Samples)

| Evaluation Metric | Measured Value | Minimum Target | Academic Benchmark Status |
|---|---|---|---|
| **Overall Accuracy** | **75.81%** | $\ge 75.0\%$ | **Passed** |
| **Recall (Sensitivity)** | **80.65%** | $\ge 70.0\%$ | **Exceeded (80.65%)** |
| **Precision** | **52.82%** | $\ge 50.0\%$ | **Passed** |
| **F1-Score** | **0.6383** | $\ge 0.600$ | **Passed** |
| **ROC-AUC Score** | **0.8520** | $\ge 0.800$ | **Exceeded (0.8520)** |

### 8.2 Classification Report
```
              precision    recall  f1-score   support

    No Churn       0.91      0.74      0.82       775
       Churn       0.53      0.81      0.64       279

    accuracy                           0.76      1054
   macro avg       0.72      0.77      0.73      1054
weighted avg       0.81      0.76      0.77      1054
```

### 8.3 Performance Curves & Interpretation
* **Confusion Matrix (`outputs/model_charts/confusion_matrix.png`):** The model correctly captures **225 out of 279 true churners** on the test set, keeping missed churners (False Negatives) down to 54.
* **ROC Curve (`outputs/model_charts/roc_curve.png`):** Demonstrates an area under the curve of **0.8520**, reflecting excellent discriminative capacity.
* **Training & Validation Loss Curves (`outputs/model_charts/training_curves.png`):** Batch Normalization and Dropout prevented divergent overfitting, with optimal validation loss achieved at Epoch 14.
* **Feature Importance (`outputs/model_charts/feature_importance.png`):** Identifies `tenure`, `TotalCharges`, `MonthlyCharges`, `InternetService_Fiber optic`, and `Contract_Two year` as the top 5 predictive features.

---

# 9. Conclusion and Future Scope

### 9.1 Conclusion
This project developed, evaluated, and deployed an end-to-end deep learning customer retention system. The key takeaways include:
1. Identifying high-risk churners is far more valuable than maximizing raw accuracy; achieving **80.65% recall** ensures the telecom provider catches 8 out of every 10 cancelling customers.
2. Long-term contracts and automated billing are the strongest stabilizers of customer retention.
3. The regularized Keras MLP architecture successfully generalizes on tabular data without overfitting.

### 9.2 Future Scope
1. **Sequential Deep Learning (LSTM / GRU):** Incorporating sequential time-series logs of monthly billing fluctuations and support ticket timestamps.
2. **Explainable AI (SHAP / LIME):** Integrating local feature attribution into the dashboard to explain why a specific individual was flagged.
3. **Threshold Optimization:** Calibrating decision thresholds based on actual Customer Acquisition Cost (CAC) vs. Customer Lifetime Value (CLV).
4. **Cloud Containerization:** Packaging the pipeline as a Dockerized microservice for automated inference.

---

# 10. References

1. **Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012).** New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research*, 218(1), 211–229. https://doi.org/10.1016/j.ejor.2011.09.031
2. **Wangperawong, A., Brun, C., Laudy, O., & Pavasuthipaisit, R. (2016).** Churn analysis using deep convolutional neural networks and autoencoders. *arXiv preprint arXiv:1604.05377*. https://arxiv.org/abs/1604.05377
3. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).** SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953
4. **Lundberg, S. M., & Lee, S. I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS 30)*, 4765–4774.
5. **IBM Corporation (2017).** *Telco Customer Churn Analytical Benchmark*. IBM Watson Analytics Community Repository.

---

# 11. Appendix (Code Section)

### Appendix A: Data Cleaning & Preprocessing Pipeline (`src/preprocessing.py`)
```python
def build_pipeline() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])

# Fitted strictly on Train set (No Data Leakage)
preprocessor = build_pipeline()
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc   = preprocessor.transform(X_val)
X_test_proc  = preprocessor.transform(X_test)
```

### Appendix B: Keras Deep Learning MLP Model (`src/model.py`)
```python
def build_model(input_dim: int, learning_rate: float = 1e-3) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")
    
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(1, activation="sigmoid", name="churn_probability")(x)
    
    model = keras.Model(inputs, outputs, name="Churn_MLP")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")]
    )
    return model
```

### Appendix C: Master Pipeline Execution (`run_pipeline.py`)
```python
STEPS = [
    ("1. Data Cleaning",        run_cleaning_pipeline),
    ("2. EDA & Visualizations", run_eda),
    ("3. Preprocessing",        run_preprocessing),
    ("4. Feature Importance",   run_feature_importance),
    ("5. Model Training",       run_model_pipeline),
]

def run_all():
    for name, fn in STEPS:
        print(f"\nExecuting: {name}")
        fn()
    print("\nSUCCESS: All pipeline stages completed successfully!")
```
