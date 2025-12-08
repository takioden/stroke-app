import streamlit as st
import pandas as pd
import pickle

# Load model, columns, scaler
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Stroke Prediction App")

st.title("🧠 Brain Stroke Prediction App")
st.write("Masukkan data berikut untuk memprediksi risiko stroke.")

# --- FORM INPUT ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married?", ["No", "Yes"])

with col2:
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)

    work_type = st.selectbox("Work Type",
        ["Private", "Self-employed", "Children", "Government job"]
    )

    residence = st.selectbox("Residence Type", ["Urban", "Rural"])

    smoking = st.selectbox("Smoking Status",
        ["Formerly Smoked", "Never Smoked", "Smokes", "Unknown"]
    )

# Build dataframe sesuai kolom model
data = {
    "gender": [1 if gender == "Male" else 0],
    "age": [age],
    "hypertension": [1 if hypertension == "Yes" else 0],
    "heart_disease": [1 if heart_disease == "Yes" else 0],
    "ever_married": [1 if ever_married == "Yes" else 0],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],

    # WORK TYPE
    "work_type_Govt_job": [1 if work_type == "Government job" else 0],
    "work_type_Private": [1 if work_type == "Private" else 0],
    "work_type_Self-employed": [1 if work_type == "Self-employed" else 0],
    "work_type_children": [1 if work_type == "Children" else 0],

    # RESIDENCE
    "Residence_type_Rural": [1 if residence == "Rural" else 0],
    "Residence_type_Urban": [1 if residence == "Urban" else 0],

    # SMOKING
    "smoking_status_Unknown": [1 if smoking == "Unknown" else 0],
    "smoking_status_formerly smoked": [1 if smoking == "Formerly Smoked" else 0],
    "smoking_status_never smoked": [1 if smoking == "Never Smoked" else 0],
    "smoking_status_smokes": [1 if smoking == "Smokes" else 0],
}

df = pd.DataFrame(data)

# Reindex kolom agar sesuai model
df = df.reindex(columns=columns, fill_value=0)

# Apply scaling
num_cols = ["age", "avg_glucose_level", "bmi"]
df[num_cols] = scaler.transform(df[num_cols])

# --- Predict ---
if st.button("Predict Risiko Stroke"):
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ Risiko Stroke: YES")
    else:
        st.success("✓ Risiko Stroke: NO")

    st.write(f"Probabilitas Stroke: `{prob:.4f}`")
