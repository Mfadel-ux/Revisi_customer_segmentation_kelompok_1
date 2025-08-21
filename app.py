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
    stc.html(html_temp)
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
    st.markdown(design, unsafe_allow_html=True)

    # Form input sesuai fitur
    col1, col2 = st.columns(2)
    age = col1.number_input("Age", min_value=18, max_value=100, step=1)
    work_exp = col2.number_input("Work Experience (years)", min_value=0, max_value=40, step=1)

    profession = col1.selectbox("Profession", 
        ["Healthcare", "Engineer", "Lawyer", "Artist", "Doctor", "Entertainment", "Marketing", "Homemaker", "Executive", "Other"])

    spending_score = col2.selectbox("Spending Score", ["Low", "Average", "High"])

    family_size = col1.number_input("Family Size", min_value=1, max_value=10, step=1)

    button = st.button("Predict Segmentation")

    if button:
        # Buat DataFrame dari input user
        input_data = pd.DataFrame({
            "Age": [age],
            "Work_Experience": [work_exp],
            "Profession": [profession],
            "Spending_Score": [spending_score],
            "Family_Size": [family_size]
        })

        # Encoding categorical (sesuai dengan training model Anda)
        input_encoded = pd.get_dummies(input_data, drop_first=True)

        # Align ke feature model
        missing_cols = set(model.feature_names_in_) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0
        input_encoded = input_encoded[model.feature_names_in_]

        # Scaling
        input_scaled = scaler.transform(input_encoded)

        # Prediksi
        prediction = model.predict(input_scaled)[0]

        mapping = {0: "A", 1: "B", 2: "C", 3: "D"}  # Sesuaikan dengan label encoding Anda
        result = mapping[prediction]

        st.success(f"Predicted Customer Segmentation: {result}")

if __name__ == "__main__":
    main()
