#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score
)
import plotly.graph_objects as go
import plotly.express as px

# Function to load saved data/models with caching
@st.cache_resource
def load_data(file_path='model_outputs.pkl'):
    data = joblib.load(file_path)
    return data

# Load the data/models
data = load_data()

# Unpack loaded data
y_test = data['y_test']
y_pred_ensemble = data['y_pred_ensemble']
y_proba_ensemble = data['y_proba_ensemble']
feature_names = data['feature_names']
best_rf = data['best_rf']
best_xgb = data['best_xgb']
scaler = data['scaler']

# Streamlit page configuration
st.set_page_config(page_title="Telco Customer Churn Prediction Dashboard - by Gurion", layout="wide")
st.title("Telco Customer Churn Prediction Dashboard - by Gurion")

st.markdown("""
### Project Overview
This dashboard presents evaluation results and insights from ensemble churn prediction models built on the Telco Customer Churn dataset.

Explore performance metrics, curves, feature importance, and try predicting churn for a hypothetical customer interactively.

---
""")

# --- Model Evaluation Metrics ---
st.header("Ensemble Model Evaluation Metrics")
report = classification_report(y_test, y_pred_ensemble, output_dict=True)

# 1. Extract the overall summary metrics (accuracy, macro avg, weighted avg)
accuracy = report.get('accuracy', 'N/A')
macro_avg = report.get('macro avg', {})
weighted_avg = report.get('weighted avg', {})

st.subheader("Overall Summary Metrics")
st.markdown(f"""
* **Accuracy:** {accuracy:.4f}
* **Macro Average F1-Score:** {macro_avg.get('f1-score', 'N/A'):.4f}
* **Weighted Average F1-Score:** {weighted_avg.get('f1-score', 'N/A'):.4f}
""")
st.markdown("---")


# 2. Extract and display the class-wise metrics (0 and 1) in a table
# Note: Since accuracy is a single number and not a row in the dict,
# we remove the overall keys before converting to a DataFrame for the table.
metrics_df = pd.DataFrame(report).transpose()
# Drop the rows that represent the overall summary metrics
metrics_df = metrics_df.drop(labels=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

# Use st.dataframe for the class-wise metrics table
st.subheader("Class-wise Metrics (Churn: 1, No Churn: 0)")
st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

# --- Confusion Matrix ---
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred_ensemble)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# --- ROC Curve ---
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba_ensemble)
roc_auc = roc_auc_score(y_test, y_proba_ensemble)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='darkorange', width=3)))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='navy', width=2, dash='dash'), showlegend=False))
fig_roc.update_layout(title=f'ROC Curve (AUC = {roc_auc:.3f})', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', width=700, height=500)
st.plotly_chart(fig_roc, use_container_width=True)

# --- Precision-Recall Curve ---
st.subheader("Precision-Recall Curve")
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba_ensemble)
avg_precision = average_precision_score(y_test, y_proba_ensemble)
fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', line=dict(color='purple', width=3)))
fig_pr.update_layout(title=f'Precision-Recall Curve (Average Precision = {avg_precision:.3f})', xaxis_title='Recall', yaxis_title='Precision', width=700, height=500)
st.plotly_chart(fig_pr, use_container_width=True)

# --- Feature Importance Random Forest ---
st.header("Top 15 Feature Importances - Random Forest")
feature_imp_rf = pd.Series(best_rf.feature_importances_, index=feature_names)
top15_rf = feature_imp_rf.sort_values(ascending=False).head(15).reset_index()
top15_rf.columns = ['Feature', 'Importance']
fig_rf = px.bar(top15_rf, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='viridis', title='Random Forest Feature Importances (Top 15)')
fig_rf.update_layout(yaxis={'categoryorder':'total ascending'}, width=700, height=500)
st.plotly_chart(fig_rf, use_container_width=True)

# --- Feature Importance XGBoost ---
st.header("Top 15 Feature Importances - XGBoost")
importance_dict = best_xgb.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': [feature_names[int(k[1:])] if k.startswith('f') else k for k in importance_dict.keys()],
    'Importance': list(importance_dict.values())
})
top15_xgb = importance_df.sort_values(by='Importance', ascending=False).head(15)
fig_xgb = px.bar(top15_xgb, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='viridis', title='XGBoost Feature Importances (Top 15)')
fig_xgb.update_layout(yaxis={'categoryorder':'total ascending'}, width=700, height=500)
st.plotly_chart(fig_xgb, use_container_width=True)

# --- Interactive Single Customer Prediction ---
st.header("Predict Churn for a Single Customer")

st.markdown("""
Enter customer details below.  
`TotalCharges` will be **automatically calculated** as `MonthlyCharges × tenure`.  
Engineered features (tenure groups, AvgMonthlyCharge, TotalServices) are computed automatically.
""")

threshold = st.slider("Set classification threshold", 0.0, 1.0, 0.5, 0.01)

# -------------------------
# Feature groups
# -------------------------

binary_features = [
    "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "PaperlessBilling"
]

multiple_lines_options = ["No phone service", "No", "Yes"]
internet_service_options = ["DSL", "Fiber optic", "No"]
contract_options = ["Month-to-month", "One year", "Two year"]
payment_method_options = [
    "Electronic check",
    "Mailed check",
    "Credit card (automatic)",
    "Bank transfer (automatic)"
]

numeric_features = ["tenure", "MonthlyCharges"]

with st.form("customer_form"):

    # Gender
    gender = st.selectbox("Gender", ["Female", "Male"])

    # Binary 0/1 toggles
    inputs = {}
    for feat in binary_features:
        choice = st.selectbox(f"{feat}", ["No", "Yes"])
        # Convert to 0/1 for model input
        inputs[feat] = 1 if choice == "Yes" else 0

    # Numeric inputs
    tenure = st.number_input("Tenure (months)", min_value=0, value=1)
    monthly = st.number_input("MonthlyCharges", min_value=0.0, value=50.0)

    # Auto compute TotalCharges
    total_charges = tenure * monthly
    st.write(f"**Computed TotalCharges:** {total_charges:.2f}")

    # MultipleLines
    multiple_lines = st.selectbox("Multiple Lines", multiple_lines_options)

    # InternetService
    internet_service = st.selectbox("Internet Service", internet_service_options)

    # If customer has NO internet service → disable all internet-dependent services
    internet_no = (internet_service == "No")

    # Per-service Yes/No toggles (appear only if internet exists)
    service_yes_no = {}
    internet_based_services = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    if not internet_no:
        st.markdown("### Internet-based Services")
        for service in internet_based_services:
            service_yes_no[service] = st.selectbox(
                f"{service} (Yes/No)", ["No", "Yes"]
            )
    else:
        # Force all to "No" when no internet
        for service in internet_based_services:
            service_yes_no[service] = "No"

    # Contract
    contract_type = st.selectbox("Contract Type", contract_options)

    # Payment Method
    payment_method = st.selectbox("Payment Method", payment_method_options)

    submitted = st.form_submit_button("Predict Churn")

    if submitted:

        # ----------------------------------------
        # Build input row (START)
        # ----------------------------------------
        input_row = {}

        # Gender (binary)
        input_row["gender"] = 1 if gender == "Male" else 0

        # Binary simple features
        for feat in binary_features:
            input_row[feat] = inputs[feat]

        # Numeric
        input_row["tenure"] = tenure
        input_row["MonthlyCharges"] = monthly
        input_row["TotalCharges"] = total_charges

        # One-hot: MultipleLines
        input_row["MultipleLines_No phone service"] = int(multiple_lines == "No phone service")
        input_row["MultipleLines_Yes"] = int(multiple_lines == "Yes")

        # One-hot: InternetService
        input_row["InternetService_Fiber optic"] = int(internet_service == "Fiber optic")
        input_row["InternetService_No"] = int(internet_service == "No")

        # Internet-dependent services
        for s in internet_based_services:
            input_row[f"{s}_No internet service"] = int(internet_no)
            input_row[f"{s}_Yes"] = int((not internet_no) and (service_yes_no[s] == "Yes"))

        # Contract OHE
        input_row["Contract_One year"] = int(contract_type == "One year")
        input_row["Contract_Two year"] = int(contract_type == "Two year")

        # Payment Method OHE
        input_row["PaymentMethod_Electronic check"] = int(payment_method == "Electronic check")
        input_row["PaymentMethod_Mailed check"] = int(payment_method == "Mailed check")
        input_row["PaymentMethod_Credit card (automatic)"] = int(payment_method == "Credit card (automatic)")

        # --------------------------
        # Engineered Features
        # --------------------------

        # Tenure groups
        input_row["tenure_group_12-24m"] = int(12 <= tenure < 24)
        input_row["tenure_group_24-48m"] = int(24 <= tenure < 48)
        input_row["tenure_group_48-60m"] = int(48 <= tenure < 60)
        input_row["tenure_group_>60m"] = int(tenure >= 60)

        # AvgMonthlyCharge
        input_row["AvgMonthlyCharge"] = (total_charges / tenure) if tenure > 0 else 0

        # TotalServices count
        service_yes_cols = [
            "MultipleLines_Yes",
            "OnlineSecurity_Yes",
            "OnlineBackup_Yes",
            "DeviceProtection_Yes",
            "TechSupport_Yes",
            "StreamingTV_Yes",
            "StreamingMovies_Yes"
        ]
        input_row["TotalServices"] = sum(input_row.get(col, 0) for col in service_yes_cols)

        # ----------------------------------------
        # Convert to ordered DataFrame
        # ----------------------------------------
        input_df = pd.DataFrame([input_row])[feature_names]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        proba_rf = best_rf.predict_proba(input_scaled)[0, 1]
        proba_xgb = best_xgb.predict_proba(input_scaled)[0, 1]
        proba_ensemble = (proba_rf + proba_xgb) / 2

        st.write(f"Predicted churn probability: **{proba_ensemble:.2%}**")

        if proba_ensemble >= threshold:
            st.warning("This customer is likely to churn.")
        else:
            st.success("This customer is unlikely to churn.")
