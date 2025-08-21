import streamlit as st
import pickle
import pandas as pd

# Load model
with open("Logistic_Regression_Model.pkl", "rb") as file:
    Logistic_Regression_Model = pickle.load(file)

# =========================
# MAIN APP
# =========================
def main():
    st.title("Customer Segmentation Prediction App")
    st.write("Aplikasi ini digunakan untuk memprediksi segmen customer berdasarkan input data.")

    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(
            """
            ### Customer Segmentation App  
            Aplikasi ini dibuat untuk mengklasifikasikan pelanggan berdasarkan data demografi dan perilaku.  
            
            #### Data Source  
            Dataset berasal dari project Customer Segmentation.  
            """
        )
    elif choice == "Machine Learning App":
        run_ml_app()

# =========================
# FORM INPUT
# =========================
def run_ml_app():
    st.subheader("Masukkan Data Customer")

    left, right = st.columns((2,2))

    gender = left.selectbox("Gender", ("Male", "Female"))
    age = right.number_input("Age", min_value=18, max_value=100, value=30)

    profession = left.selectbox("Profession", ("Artist", "Doctor", "Engineer", "Healthcare", "Lawyer", "Entertainment"))
    work_experience = right.number_input("Work Experience (years)", min_value=0, max_value=40, value=5)

    spending_score = left.selectbox("Spending Score", ("Low", "Average", "High"))
    family_size = right.number_input("Family Size", min_value=1, max_value=10, value=3)

    button = st.button("Predict Segmentation")

    if button:
        pred, proba = predict_customer_segmentation(
            gender, age, profession, work_experience, spending_score, family_size
        )

        st.success(f"Predicted Customer Segmentation: {pred}")
        st.write("Prediction Probability per Class:")
        st.dataframe(pd.DataFrame(proba, columns=Logistic_Regression_Model.classes_))

# =========================
# PREDICT FUNCTION
# =========================
def predict_customer_segmentation(gender, age, profession, work_experience, spending_score, family_size):
    # Manual Encoding (HARUS sesuai saat training model!)
    gen = 0 if gender == "Male" else 1

    # Profession encoding contoh (sesuaikan dengan training)
    prof_map = {
        "Artist": 0,
        "Doctor": 1,
        "Engineer": 2,
        "Healthcare": 3,
        "Lawyer": 4,
        "Entertainment": 5
    }
    prof = prof_map.get(profession, 0)

    # Spending Score encoding contoh
    spend_map = {"Low": 0, "Average": 1, "High": 2}
    spend = spend_map.get(spending_score, 1)

    # Feature array
    features = [[gen, age, prof, work_experience, spend, family_size]]

    # Predict
    prediction = Logistic_Regression_Model.predict(features)
    prediction_proba = Logistic_Regression_Model.predict_proba(features)

    return prediction[0], prediction_proba


if __name__ == "__main__":
    main()
