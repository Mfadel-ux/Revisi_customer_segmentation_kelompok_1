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

    # Form input user
    left, right = st.columns((2,2))
    age = left.number_input("Age", min_value=18, max_value=100, value=25)
    gender = right.selectbox("Gender", ("Male", "Female"))
    profession = left.selectbox("Profession", 
                                ("Artist", "Doctor", "Engineer", "Entrepreneur", "Healthcare", "Lawyer", "Marketing"))
    work_experience = right.number_input("Work Experience (years)", min_value=0, max_value=20, value=1)
    spending_score = left.selectbox("Spending Score", ("Low", "Average", "High"))
    family_size = right.number_input("Family Size", min_value=1, max_value=10, value=3)

    button = st.button("Predict Segmentation")

    if button:
        # Buat dataframe input user
        input_data = {
            "Gender": [gender],
            "Age": [age],
            "Profession": [profession],
            "Work_Experience": [work_experience],
            "Spending_Score": [spending_score],
            "Family_Size": [family_size]
        }
        input_df = pd.DataFrame(input_data)

        # One Hot Encoding (harus sama dengan training)
        input_encoded = pd.get_dummies(input_df)

        # Samakan kolom input dengan kolom model
        missing_cols = set(model.feature_names_in_) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0

        # Urutkan kolom sesuai dengan model training
        input_encoded = input_encoded[model.feature_names_in_]

        # Prediksi
        prediction = model.predict(input_encoded)
        prediction_proba = model.predict_proba(input_encoded)

        # Output
        st.success(f"Predicted Customer Segmentation: {prediction[0]}")
        st.write("Prediction Probability:")
        prob_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.dataframe(prob_df)
if __name__ == "__main__":
    main()


