"""
app.py  –  Heart Disease Detection (Streamlit UI)
Loads model.pkl, label_encoders.pkl, scaler.pkl produced by train.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Detector",
    page_icon="❤️",
    layout="centered"
)

# ── Load artefacts (cached so they load only once) ───────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl",          "rb") as f: model          = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f: label_encoders = pickle.load(f)
    with open("scaler.pkl",         "rb") as f: scaler         = pickle.load(f)
    return model, label_encoders, scaler

model, label_encoders, scaler = load_artifacts()

# ── Feature definitions ──────────────────────────────────────────────────────
CATEGORICAL_OPTIONS = {
    "Gender":               ["Male", "Female"],
    "Exercise Habits":      ["Low", "Medium", "High"],
    "Smoking":              ["No", "Yes"],
    "Family Heart Disease": ["No", "Yes"],
    "Diabetes":             ["No", "Yes"],
    "High Blood Pressure":  ["No", "Yes"],
    "Low HDL Cholesterol":  ["No", "Yes"],
    "High LDL Cholesterol": ["No", "Yes"],
    "Alcohol Consumption":  ["Low", "Medium", "High"],
    "Stress Level":         ["Low", "Medium", "High"],
    "Sugar Consumption":    ["Low", "Medium", "High"],
}

NUMERIC_FEATURES = {
    "Age":                 (18,  90,  45,   1,  "years"),
    "Blood Pressure":      (60, 200,  120,  1,  "mmHg"),
    "Cholesterol Level":   (100, 400, 200,  1,  "mg/dL"),
    "BMI":                 (10.0, 60.0, 25.0, 0.1, ""),
    "Sleep Hours":         (3.0, 12.0, 7.0,  0.1, "hrs/night"),
    "Triglyceride Level":  (50, 500, 150,   1,  "mg/dL"),
    "Fasting Blood Sugar": (50, 300, 100,   1,  "mg/dL"),
    "CRP Level":           (0.0, 20.0, 1.0, 0.1, "mg/L"),
    "Homocysteine Level":  (0.0, 50.0, 10.0, 0.1, "µmol/L"),
}

FEATURE_ORDER = [
    "Age", "Gender", "Blood Pressure", "Cholesterol Level",
    "Exercise Habits", "Smoking", "Family Heart Disease", "Diabetes",
    "BMI", "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol",
    "Alcohol Consumption", "Stress Level", "Sleep Hours", "Sugar Consumption",
    "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level",
]

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("❤️ Heart Disease Risk Detector")
st.markdown(
    """
    This tool uses a **Logistic Regression** model trained on the Heart Disease dataset.
    SMOTE was applied to handle class imbalance (80 % healthy / 20 % disease).

    > ⚠️ **Disclaimer:** This is a research / educational demo, *not* a medical diagnostic tool.
    > Please consult a qualified healthcare professional for medical advice.
    """,
    unsafe_allow_html=False,
)
st.divider()

# ── Input form ───────────────────────────────────────────────────────────────
st.subheader("📋 Enter Patient Details")

col1, col2 = st.columns(2)

inputs = {}

with col1:
    inputs["Age"]                = st.number_input("Age (years)",           min_value=18,  max_value=90,  value=45,  step=1)
    inputs["Gender"]             = st.selectbox("Gender",                   CATEGORICAL_OPTIONS["Gender"])
    inputs["Blood Pressure"]     = st.number_input("Blood Pressure (mmHg)", min_value=60,  max_value=200, value=120, step=1)
    inputs["Cholesterol Level"]  = st.number_input("Cholesterol (mg/dL)",   min_value=100, max_value=400, value=200, step=1)
    inputs["BMI"]                = st.number_input("BMI",                   min_value=10.0,max_value=60.0,value=25.0,step=0.1,format="%.1f")
    inputs["Sleep Hours"]        = st.number_input("Sleep Hours / Night",   min_value=3.0, max_value=12.0,value=7.0, step=0.1,format="%.1f")
    inputs["Triglyceride Level"] = st.number_input("Triglycerides (mg/dL)", min_value=50,  max_value=500, value=150, step=1)
    inputs["Fasting Blood Sugar"]= st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, max_value=300, value=100, step=1)
    inputs["CRP Level"]          = st.number_input("CRP Level (mg/L)",      min_value=0.0, max_value=20.0,value=1.0, step=0.1,format="%.1f")
    inputs["Homocysteine Level"] = st.number_input("Homocysteine (µmol/L)", min_value=0.0, max_value=50.0,value=10.0,step=0.1,format="%.1f")

with col2:
    inputs["Exercise Habits"]      = st.selectbox("Exercise Habits",       CATEGORICAL_OPTIONS["Exercise Habits"])
    inputs["Smoking"]              = st.selectbox("Smoking",               CATEGORICAL_OPTIONS["Smoking"])
    inputs["Family Heart Disease"] = st.selectbox("Family Heart Disease",  CATEGORICAL_OPTIONS["Family Heart Disease"])
    inputs["Diabetes"]             = st.selectbox("Diabetes",              CATEGORICAL_OPTIONS["Diabetes"])
    inputs["High Blood Pressure"]  = st.selectbox("High Blood Pressure",   CATEGORICAL_OPTIONS["High Blood Pressure"])
    inputs["Low HDL Cholesterol"]  = st.selectbox("Low HDL Cholesterol",   CATEGORICAL_OPTIONS["Low HDL Cholesterol"])
    inputs["High LDL Cholesterol"] = st.selectbox("High LDL Cholesterol",  CATEGORICAL_OPTIONS["High LDL Cholesterol"])
    inputs["Alcohol Consumption"]  = st.selectbox("Alcohol Consumption",   CATEGORICAL_OPTIONS["Alcohol Consumption"])
    inputs["Stress Level"]         = st.selectbox("Stress Level",          CATEGORICAL_OPTIONS["Stress Level"])
    inputs["Sugar Consumption"]    = st.selectbox("Sugar Consumption",     CATEGORICAL_OPTIONS["Sugar Consumption"])

st.divider()

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True, type="primary"):

    # Build row in correct column order
    row = {}
    for feat in FEATURE_ORDER:
        val = inputs[feat]
        if feat in label_encoders:
            try:
                val = label_encoders[feat].transform([str(val)])[0]
            except ValueError:
                # Unseen label – use 0 as fallback
                val = 0
        row[feat] = val

    X_input = pd.DataFrame([row])[FEATURE_ORDER]
    X_scaled = scaler.transform(X_input)

    pred     = model.predict(X_scaled)[0]
    proba    = model.predict_proba(X_scaled)[0]

    st.subheader("📊 Prediction Result")

    col_r, col_p = st.columns(2)
    with col_r:
        if pred == 1:
            st.error("**Result: Heart Disease Detected**")
        else:
            st.success("**Result: No Heart Disease Detected**")

    with col_p:
        st.metric("Probability of Heart Disease", f"{proba[1]*100:.1f} %")
        st.metric("Probability of No Disease",    f"{proba[0]*100:.1f} %")

    # Probability bar
    st.progress(float(proba[1]), text=f"Risk score: {proba[1]*100:.1f} %")

    st.caption(
        "ℹ️ The model was trained on an imbalanced dataset (80/20) with oversampling (SMOTE). "
        "Results reflect the model's learned patterns and should be interpreted cautiously."
    )

# ── Model Info ───────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this Model"):
    st.markdown("""
    | Detail | Value |
    |--------|-------|
    | Algorithm | Logistic Regression |
    | Imbalance handling | SMOTE (oversampling minority class) |
    | Dataset size | 10,000 rows |
    | Class split (original) | 80 % No Disease / 20 % Heart Disease |
    | Train/Test split | 80 % / 20 % |
    | Features | 20 |
    | Target column | Heart Disease Status |

    **Why the performance looks modest:** The dataset is imbalanced and some features have
    low discriminative power for logistic regression. The model results are presented
    as-is without cherry-picking — honest evaluation is part of good data science.
    """)
