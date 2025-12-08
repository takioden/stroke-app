import streamlit as st
import pandas as pd
import pickle

# Load model, columns, scaler
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
#           CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
    body {
        background-color: #F2F6FC;
    }

    .main {
        background-color: #F2F6FC;
    }

    .title-text {
        font-size: 38px;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        margin-bottom: -10px;
    }

    .subtitle-text {
        font-size: 18px;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 25px;
    }

    /* Card Style */
    .card {
        padding: 25px;
        border-radius: 12px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }

    /* Predict Button */
    .stButton>button {
        width: 100%;
        background-color: #2E86C1;
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
#              TITLE
# -------------------------------
st.markdown("<p class='title-text'>🧠 Stroke Prediction App</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Isi data berikut untuk memprediksi risiko stroke.</p>",
            unsafe_allow_html=True)

# -------------------------------
#     INPUT FORM IN CARD BOX
# -------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

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
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "children"])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking = st.selectbox(
        "Smoking Status",
        ["Formerly Smoked", "Never Smoked", "Smokes", "Unknown"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
#     DATAFRAME FIX & SCALING
# -------------------------------
data = {
    "gender": [1 if gender == "Male" else 0],
    "age": [age],
    "hypertension": [1 if hypertension == "Yes" else 0],
    "heart_disease": [1 if heart_disease == "Yes" else 0],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "ever_married": [1 if ever_married == "Yes" else 0],
    "work_type_Private": [1 if work_type == "Private" else 0],
    "work_type_Self-employed": [1 if work_type == "Self-employed" else 0],
    "work_type_children": [1 if work_type == "children" else 0],
    "Residence_type_Urban": [1 if residence == "Urban" else 0],
    "smoking_status_formerly smoked": [1 if smoking == "Formerly Smoked" else 0],
    "smoking_status_never smoked": [1 if smoking == "Never Smoked" else 0],
    "smoking_status_smokes": [1 if smoking == "Smokes" else 0],
}

df = pd.DataFrame(data)
df = df.reindex(columns=columns, fill_value=0)

# Scaling hanya 3 kolom numerik
num_cols = ["age", "avg_glucose_level", "bmi"]
df[num_cols] = scaler.transform(df[num_cols])

# -------------------------------
#           PREDICT
# -------------------------------
if st.button("Predict Risiko Stroke"):
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    # CARD RESULT
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if prediction == 1:
        st.error("⚠️ **Hasil Prediksi: Risiko Stroke Tinggi**")
    else:
        st.success("✅ **Hasil Prediksi: Risiko Stroke Rendah**")

    st.write(f"**Probabilitas Stroke:** `{prob:.4f}`")

    st.markdown("</div>", unsafe_allow_html=True)
