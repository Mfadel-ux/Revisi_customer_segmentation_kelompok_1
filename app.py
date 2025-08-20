# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =========================
# Load model & feature columns
# =========================
with open("Logisti_model.pkl", "rb") as file:
    model = pickle.load(file)


st.title("Customer Segmentation Prediction")

# === Input Form ===
st.header("Masukkan Data Customer")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
spending = st.number_input("Spending Score", min_value=0, max_value=100, value=50)
profession = st.selectbox("Profession", ["Artist", "Doctor", "Engineer", "Entertainment", "Healthcare", "Lawyer", "Marketing"])
married = st.selectbox("Ever Married", ["Yes", "No"])

# === Convert to DataFrame ===
input_data = pd.DataFrame({
    "Age": [age],
    "Spending_Score": [spending],
    "Profession": [profession],
    "Ever_Married": [married]
})

# === Predict ===
if st.button("Predict Segmentation"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Segmentation: {prediction[0]}")




