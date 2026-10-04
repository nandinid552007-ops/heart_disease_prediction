import joblib
import pandas as pd
import streamlit as st

model = joblib.load("heart_model.joblib")
cols = joblib.load("feature_columns.joblib")

st.title("Heart Disease Risk Estimator")
st.caption("Learning project, not a medical tool.")

age = st.number_input("Age", 20, 90, 55)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest pain type", ["asymptomatic", "typical angina", "atypical angina", "non-anginal"])
trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
chol = st.number_input("Cholesterol (mg/dl)", 80, 600, 236)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.selectbox("Resting ECG", ["lv hypertrophy", "normal", "st-t abnormality"])
thalch = st.number_input("Max heart rate achieved", 60, 220, 140)
exang = st.selectbox("Exercise-induced angina", ["No", "Yes"])
oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
slope = st.selectbox("ST slope", ["downsloping", "flat", "upsloping", "missing"])
thal = st.selectbox("Thal", ["fixed defect", "normal", "reversable defect", "missing"])

threshold = st.slider("Decision threshold", 0.2, 0.7, 0.4, 0.05)

if st.button("Predict"):
    row = dict.fromkeys(cols, 0)
    row.update({
        "age": age, "trestbps": trestbps, "chol": chol,
        "thalch": thalch, "oldpeak": oldpeak,
    })
    for name, value in [
        ("sex_Male", sex == "Male"),
        ("fbs_True", fbs == "Yes"),
        ("exang_True", exang == "Yes"),
        (f"cp_{cp}", True),
        (f"restecg_{restecg}", True),
        (f"slope_{slope}", True),
        (f"thal_{thal}", True),
    ]:
        if name in row:
            row[name] = int(value)

    X_new = pd.DataFrame([row])[cols]
    p = model.predict_proba(X_new)[0, 1]

    st.metric("Estimated probability of heart disease", f"{p:.0%}")
    if p >= threshold:
        st.error("Flagged as higher risk")
    else:
        st.success("Not flagged")