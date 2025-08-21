import streamlit as st
import pickle
import numpy as np

# =========================
# Load Model & Scaler
# =========================
with open("Logistic_Regression_Model.pkl", "rb") as f:
    Logistic_Regression_Model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =========================
# UI Header
# =========================
st.set_page_config(page_title="Customer Segmentation App", layout="centered")
st.markdown(
    """
    <div style="background-color:#000;padding:12px;border-radius:10px">
        <h1 style="color:#fff;text-align:center;">Customer Segmentation Prediction</h1>
        <h4 style="color:#fff;text-align:center;">Streamlit Demo</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Input Form
# =========================
st.subheader("Input Customer Data")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ("Male", "Female"))
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    profession = st.selectbox("Profession", ("Artist", "Doctor", "Engineer", "Entrepreneur", "Other"))

with col2:
    work_experience = st.number_input("Work Experience (Years)", min_value=0, max_value=40, value=5)
    spending_score = st.selectbox("Spending Score", ("Low", "Average", "High"))
    family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3)

# =========================
# Preprocessing Input
# =========================
def preprocess_input(gender, age, profession, work_experience, spending_score, family_size):
    # Encode gender
    gen = 0 if gender == "Male" else 1

    # Encode profession (dummy sederhana)
    prof_dict = {"Artist": 0, "Doctor": 1, "Engineer": 2, "Entrepreneur": 3, "Other": 4}
    prof = prof_dict[profession]

    # Encode spending score
    spend_dict = {"Low": 0, "Average": 1, "High": 2}
    spend = spend_dict[spending_score]

    # Gabungkan jadi array
    features = np.array([[gen, age, prof, work_experience, spend, family_size]])

    # Scaling sesuai training
    features_scaled = scaler.transform(features)
    return features_scaled

# =========================
# Prediction Function
# =========================
def predict_customer_segmentation(gender, age, profession, work_experience, spending_score, family_size):
    features_scaled = preprocess_input(gender, age, profession, work_experience, spending_score, family_size)

    # Predict class
    prediction = Logistic_Regression_Model.predict(features_scaled)[0]

    # Predict probability
    prediction_proba = Logistic_Regression_Model.predict_proba(features_scaled)[0]

    return prediction, prediction_proba

# =========================
# Run Prediction
# =========================
if st.button("Predict"):
    pred, proba = predict_customer_segmentation(gender, age, profession, work_experience, spending_score, family_size)

    st.success(f"Predicted Customer Segment: **{pred}**")

    st.subheader("Prediction Probability")
    for i, p in enumerate(proba):
        st.write(f"Segment {i}: {p:.2f}")
