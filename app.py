import streamlit as st
import pickle
import numpy as np

# ==============================
# Load Model & Scaler
# ==============================
with open("Logisti_model.pkl", "rb") as f:
    Logistic_Regression_Model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ==============================
# Preprocess Input Function
# ==============================
def preprocess_input(gender, age, profession, work_experience, spending_score, family_size):
    # Mapping categorical ke numeric (HARUS sama dengan waktu training!)
    gender_dict = {"Male": 1, "Female": 0}
    profession_dict = {
        "Artist": 0,
        "Doctor": 1,
        "Engineer": 2,
        "Entrepreneur": 3,
        "Lawyer": 4,
        "Other": 5
    }
    spending_dict = {"Low": 0, "Average": 1, "High": 2}

    gender_num = gender_dict.get(gender, 0)
    profession_num = profession_dict.get(profession, 5)
    spending_num = spending_dict.get(spending_score, 1)

    # Urutkan fitur sesuai training
    features = [[gender_num, age, profession_num, work_experience, spending_num, family_size]]

    # Scaling
    features_scaled = scaler.transform(features)
    return features_scaled

# ==============================
# Prediction Function
# ==============================
def predict_customer_segmentation(gender, age, profession, work_experience, spending_score, family_size):
    features_scaled = preprocess_input(gender, age, profession, work_experience, spending_score, family_size)
    prediction = Logistic_Regression_Model.predict(features_scaled)
    prediction_proba = Logistic_Regression_Model.predict_proba(features_scaled)
    return prediction[0], prediction_proba

# ==============================
# Streamlit App
# ==============================
def main():
    st.title("Customer Segmentation Prediction")
    st.write("Streamlit Demo")

    st.header("Input Customer Data")
    g
