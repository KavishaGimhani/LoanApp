import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import gdown
import os
import zipfile

# Google Drive file ID (replace this with your real ID)
DRIVE_FILE_ID = "1GhTQ_LdH2aEFHefvnTB3l7HDDvloZq3n"

# File path to save
ZIP_PATH = "model_artifacts.zip"
EXTRACT_PATH = "model_artifacts"
# Download and extract if not present
if not os.path.exists(EXTRACT_PATH):
    print("Downloading model artifacts from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", ZIP_PATH, quiet=False)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print("Model artifacts extracted successfully!")

# ================== Load artifacts ==================
ARTIFACT_DIR = EXTRACT_PATH  # use the extracted folder from gdown

model = joblib.load(os.path.join(ARTIFACT_DIR, "loan_rf_model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoders.pkl"))
features = joblib.load(os.path.join(ARTIFACT_DIR, "features.pkl"))


# ================== Page Config ==================
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💳",
    layout="wide"
)

# ================== Custom CSS ==================
st.markdown("""
    <style>
        body { background-color: #f4f6f8; font-family: "Segoe UI", sans-serif; }
        .stButton>button {
            background: linear-gradient(90deg, #007bff, #0056b3);
            color: white; border-radius: 10px; padding: 10px 25px;
            font-size: 16px; font-weight: 600; transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #0056b3, #004080);
            transform: scale(1.05);
        }
        .highlight {
            background-color: #eaf4ff; padding: 12px; border-radius: 10px;
            font-weight: 500; color: #004080; border-left: 5px solid #007bff;
        }
        .prediction-card {
            background-color: #ffffff; padding: 25px; border-radius: 15px;
            box-shadow: 0px 8px 20px rgba(0,0,0,0.1); text-align: center;
            margin-bottom: 20px; color:black;
        }
        .prediction-card h1 { font-size: 36px; margin: 10px 0; }
        .prediction-card p { font-size: 20px; margin: 5px 0; }
    </style>
""", unsafe_allow_html=True)

# ================== Title ==================
st.title("💳 Loan Default Prediction App")
st.markdown("<p class='highlight'> Fill in borrower details below and get a final loan decision instantly.</p>", unsafe_allow_html=True)

# ================== Layout: Inputs Left, Predictions Right ==================
input_col, result_col = st.columns([1, 1])

# ================== Input Section ==================
with input_col:
    st.header("📝 Borrower Information")
    user_data = {}
    col1, col2 = st.columns(2)
    for i, col in enumerate(features):
        if col in label_encoders:  # categorical
            options = label_encoders[col].classes_
            value = options[0]  # default to first option
            if i % 2 == 0:
                user_data[col] = col1.selectbox(f" {col}", options, index=options.tolist().index(value))
            else:
                user_data[col] = col2.selectbox(f" {col}", options, index=options.tolist().index(value))
        else:  # numeric
            default_val = 0.0
            if col.lower() == "creditscore":
                user_data[col] = col1.slider(" Credit Score", 300, 850, 650)
            else:
                if i % 2 == 0:
                    user_data[col] = col1.number_input(f" {col}", min_value=0.0, value=float(default_val))
                else:
                    user_data[col] = col2.number_input(f" {col}", min_value=0.0, value=float(default_val))

    # Preprocess input
    input_df = pd.DataFrame([user_data])
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])
    input_scaled = scaler.transform(input_df)

# ================== Prediction Section ==================
with result_col:
    st.header("🔮 Prediction & Loan Decision")
    if st.button("🚀 Predict Default Risk"):
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]

        # Final Loan Decision
        if prediction == 1:  # High risk
            decision_text = "⚠️ Loan Not Recommended"
            card_color = "#dc3545"
        else:  # Low risk
            decision_text = "✅ Loan Can Be Granted"
            card_color = "#28a745"

        # Prediction Card
        st.markdown(f"""
            <div class='prediction-card' style='border-left: 6px solid {card_color};'>
                <h1>{decision_text}</h1>
                <p>Default Probability: {proba:.2f}</p>
            </div>
        """, unsafe_allow_html=True)

        # Probability Breakdown Chart
        st.subheader("📊 Risk Probability Breakdown")
        fig, ax = plt.subplots()
        ax.bar(["Low Risk", "High Risk"], model.predict_proba(input_scaled)[0],
               color=["#28a745", "#dc3545"], alpha=0.85)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
