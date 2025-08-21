import streamlit as st
import pandas as pd
import pickle

html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Customer Segmentation Prediction App</h1> 
                <h4 style="color:#fff;text-align:center">Made for: Marketing Team</h4> 
                </div>
                """

desc_temp = """ 
### Customer Segmentation App  
Aplikasi ini digunakan untuk memprediksi segmen customer berdasarkan data demografi & perilaku belanja.  

#### Data Source  
Kaggle: [Customer Segmentation Dataset](https://www.kaggle.com/datasets)  
"""

# Main app
def main():
    st.markdown(html_temp, unsafe_allow_html=True)  # ganti stc.html jadi st.markdown
    menu = ["Home", "Predict Customer Segmentation"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Predict Customer Segmentation":
        run_ml_app()


def run_ml_app():
    design = """<div style="padding:15px;">
                    <h2 style="color:#000">Input Data Customer</h2>
                </div>
             """
    st.markdown("<h2 style='color:#000'>Customer Segmentation Prediction</h2>", unsafe_allow_html=True)

    # Input form
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    gender = st.selectbox("Gender", ("Male", "Female"))
    profession = st.selectbox("Profession", 
                                ("Artist", "Doctor", "Engineer", "Entrepreneur", "Healthcare", "Lawyer", "Marketing"))
    work_experience = st.number_input("Work Experience (years)", min_value=0, max_value=20, value=1)
    spending_score = st.selectbox("Spending Score", ("Low", "Average", "High"))
    family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3)

    if st.button("Predict Segmentation"):
        # Buat DataFrame dari input
        input_data = pd.DataFrame({
            "Gender": [gender],
            "Age": [age],
            "Profession": [profession],
            "Work_Experience": [work_experience],
            "Spending_Score": [spending_score],
            "Family_Size": [family_size]
        })

        # One-hot encoding
        input_encoded = pd.get_dummies(input_data)

        # --- Hilangkan kebutuhan feature_names ---
        # Cocokkan jumlah kolom dengan model
        try:
            prediction = model.predict(input_encoded)
            prediction_proba = model.predict_proba(input_encoded)

            st.success(f"Predicted Customer Segmentation: {prediction[0]}")
            st.write("Prediction Probability:")
            st.dataframe(pd.DataFrame(prediction_proba, columns=model.classes_))
        except Exception as e:
            st.error("Error saat prediksi. Pastikan input sesuai dengan data training.")
            st.write(e)

def main():
    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown("### Customer Segmentation App")
    elif choice == "Machine Learning App":
        run_ml_app()

if __name__ == "__main__":
    main()


