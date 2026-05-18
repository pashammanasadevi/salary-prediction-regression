import streamlit as st
import pandas as pd
import pickle

# Load Model
with open("salary_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load Encoders
with open("gender_encoder.pkl", "rb") as file:
    gender_encoder = pickle.load(file)

with open("education_encoder.pkl", "rb") as file:
    education_encoder = pickle.load(file)

with open("job_encoder.pkl", "rb") as file:
    job_encoder = pickle.load(file)

# Streamlit Page Config
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

# Title
st.title("💼 Salary Prediction Regression")
st.write("Predict employee salary using Machine Learning")

# Inputs
age = st.number_input(
    "Enter Age",
    min_value=18,
    max_value=65,
    value=25
)

gender = st.selectbox(
    "Select Gender",
    gender_encoder.classes_
)

education = st.selectbox(
    "Select Education Level",
    education_encoder.classes_
)

job_title = st.selectbox(
    "Select Job Title",
    job_encoder.classes_
)

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=40.0,
    value=1.0,
    step=0.5
)

# Prediction Button
if st.button("Predict Salary"):

    # Encode Inputs
    gender_encoded = gender_encoder.transform([gender])[0]
    education_encoded = education_encoder.transform([education])[0]
    job_encoded = job_encoder.transform([job_title])[0]

    # Create DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_encoded],
        "Education Level": [education_encoded],
        "Job Title": [job_encoded],
        "Years of Experience": [experience]
    })

    # Predict
    prediction = model.predict(input_data)

    # Output
    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")

# Footer
st.markdown("---")
st.caption("Machine Learning Salary Prediction Project")