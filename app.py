import streamlit as st
import pandas as pd
import pickle

# =======================
# Load model & scaler
# =======================
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# =======================
# Column order sesuai training
# (isi dengan urutan kolom X_train Anda)
# =======================
feature_columns = [
    "Age", "Work_Experience", "Family_Size",
    "Profession_Artist", "Profession_Doctor", "Profession_Engineer",
    "Profession_Homemaker", "Profession_Lawyer", "Profession_Marketing",
    "Spending_Score_High", "Spending_Score_Average", "Spending_Score_Low"
]

# =======================
# Streamlit UI
# =======================
st.title("Customer Segmentation Prediction App")

# Input manual user
age = st.number_input("Age", min_value=18, max_value=100, value=30)
work_exp = st.number_input("Work Experience", min_value=0, max_value=40, value=5)
family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3)

# Pilihan profession
profession = st.selectbox("Profession", [
    "Artist", "Doctor", "Engineer", "Homemaker", "Lawyer", "Marketing"
])

# Spending Score
spending = st.selectbox("Spending Score", ["High", "Average", "Low"])

# =======================
# Convert input ke DataFrame
# =======================
# One-hot encoding manual
profession_ohe = {f"Profession_{p}": 0 for p in [
    "Artist", "Doctor", "Engineer", "Homemaker", "Lawyer", "Marketing"
]}
profession_ohe[f"Profession_{profession}"] = 1

spending_ohe = {
    "Spending_Score_High": 0,
    "Spending_Score_Average": 0,
    "Spending_Score_Low": 0
}
spending_ohe[f"Spending_Score_{spending}"] = 1

# Gabung semua input
input_dict = {
    "Age": age,
    "Work_Experience": work_exp,
    "Family_Size": family_size
}
input_dict.update(profession_ohe)
input_dict.update(spending_ohe)

input_data = pd.DataFrame([input_dict])
input_data = input_data[feature_columns]  # pastikan urutan sama

# =======================
# Prediksi
# =======================
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.success(f"Predicted Segmentation: {prediction[0]}")
    st.write("Prediction Probabilities:", prediction_proba)

