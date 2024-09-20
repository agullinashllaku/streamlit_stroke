import streamlit as st
import joblib 
import pandas as pd

pipe = joblib.load('best_pipe_rfc.pkl')

def show_predict_page():
    st.title("Stroke Prediction")
    age = st.number_input("Enter Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Select Gender", options=["Male", "Female"])
    hypertension = st.selectbox("Do you have hypertension?", options=["Yes", "No"])
    heart_disease = st.selectbox("Do you have heart disease?", options=["Yes", "No"])
    ever_married = st.selectbox("Have you ever been married?", options=["Yes", "No"])
    work_type = st.selectbox("Work Type", options=["Private", "Self-employed", "Governemnt Job", "Stay-at-home parent", "Never worked"])
    residence_type = st.selectbox("Residence Type", options=["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    smoking_status = st.selectbox("Smoking Status", options=["Never smoked", "Formerly smoked", "Smokes"]).lower()

    # Map user input to numerical values
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    if work_type == "Governemnt Job":
        work_type = "Govt_job"
    elif work_type == "Stay-at-home parent":
        work_type = "Children"
    elif work_type == "Never worked":
        work_type = "Never_worked"
    else:
        work_type

    # Create a DataFrame with the mapped user input
    user_input = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # Make prediction and get probabilities using the pipeline
    if st.button('Predict Stroke Probability'):
        probability = pipe.predict_proba(user_input)[0][1]  # Probability of class 1 (stroke)
        st.empty()
        st.write(f"# Probability of Stroke: {probability*100:.2f}")
        st.empty()
        
show_predict_page()