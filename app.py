import streamlit as st
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction System")

st.subheader("Enter Patient Details")

age = st.number_input("Age", 1, 120, 30)

resting_bp_systolic = st.number_input("Resting BP Systolic", 50, 250, 120)

resting_bp_diastolic = st.number_input("Resting BP Diastolic", 30, 150, 80)

cholesterol_total = st.number_input("Total Cholesterol", 50, 500, 180)

hdl = st.number_input("HDL", 10, 150, 50)

ldl = st.number_input("LDL", 10, 300, 100)

triglycerides = st.number_input("Triglycerides", 10, 500, 150)

fasting_blood_sugar = st.number_input("Fasting Blood Sugar", 50, 300, 100)

hba1c = st.number_input("HbA1c", 3.0, 15.0, 5.5)

bmi = st.number_input("BMI", 10.0, 60.0, 24.0)

resting_heart_rate = st.number_input("Resting Heart Rate", 40, 180, 72)

max_heart_rate_achieved = st.number_input("Max Heart Rate Achieved", 60, 250, 150)

exercise_induced_angina = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

st_depression = st.number_input("ST Depression", 0.0, 10.0, 0.0)

family_history = st.selectbox(
    "Family History",
    [0, 1]
)

alcohol_units_per_week = st.number_input("Alcohol Units Per Week", 0.0, 50.0, 2.0)

exercise_minutes_per_week = st.number_input("Exercise Minutes Per Week", 0, 1000, 150)

sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)

stress_score = st.number_input("Stress Score", 0.0, 10.0, 5.0)

wearable_owner = st.selectbox(
    "Wearable Owner",
    [0, 1]
)

daily_steps = st.number_input("Daily Steps", 0, 50000, 7000)

diet_quality_score = st.number_input("Diet Quality Score", 0.0, 100.0, 50.0)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ]
)

smoker_status = st.selectbox(
    "Smoker Status",
    [
        "Never",
        "Former",
        "Current"
    ]
)

cholesterol_hdl_ratio = cholesterol_total / hdl if hdl != 0 else 0

if st.button("Predict"):

    input_data = {
        "age": age,
        "resting_bp_systolic": resting_bp_systolic,
        "resting_bp_diastolic": resting_bp_diastolic,
        "cholesterol_total": cholesterol_total,
        "hdl": hdl,
        "ldl": ldl,
        "triglycerides": triglycerides,
        "fasting_blood_sugar": fasting_blood_sugar,
        "hba1c": hba1c,
        "bmi": bmi,
        "resting_heart_rate": resting_heart_rate,
        "max_heart_rate_achieved": max_heart_rate_achieved,
        "exercise_induced_angina": exercise_induced_angina,
        "st_depression": st_depression,
        "family_history": family_history,
        "alcohol_units_per_week": alcohol_units_per_week,
        "exercise_minutes_per_week": exercise_minutes_per_week,
        "sleep_hours": sleep_hours,
        "stress_score": stress_score,
        "wearable_owner": wearable_owner,
        "daily_steps": daily_steps,
        "diet_quality_score": diet_quality_score,
        "cholesterol_hdl_ratio": cholesterol_hdl_ratio,

        "sex_Female": 1 if sex == "Female" else 0,
        "sex_Male": 1 if sex == "Male" else 0,

        "chest_pain_type_Asymptomatic": 1 if chest_pain_type == "Asymptomatic" else 0,
        "chest_pain_type_Atypical Angina": 1 if chest_pain_type == "Atypical Angina" else 0,
        "chest_pain_type_Non-Anginal Pain": 1 if chest_pain_type == "Non-Anginal Pain" else 0,
        "chest_pain_type_Typical Angina": 1 if chest_pain_type == "Typical Angina" else 0,

        "smoker_status_Current": 1 if smoker_status == "Current" else 0,
        "smoker_status_Former": 1 if smoker_status == "Former" else 0,
        "smoker_status_Never": 1 if smoker_status == "Never" else 0,
    }

    import pandas as pd

    input_df = pd.DataFrame([input_data])

    scaled_data = scaler.transform(input_df)

    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")