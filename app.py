from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🩺 Diabetes Outcome Prediction")
st.write(
    "An educational machine-learning application that uses medical measurements "
    "to predict an outcome class from the Pima Indians Diabetes dataset."
)

st.warning(
    "This tool is for learning and demonstration only. It is not a medical "
    "diagnosis, screening tool, or substitute for professional healthcare advice."
)

# --------------------------------------------------
# Feature explanations
# --------------------------------------------------
with st.expander("ℹ️ What do these input features mean?"):
    st.markdown("""
    - **Pregnancies:** Number of times the person has been pregnant.
    - **Glucose:** Blood glucose measurement recorded in the dataset.
    - **Blood Pressure:** Diastolic blood pressure measurement (mm Hg).
    - **Skin Thickness:** Triceps skin-fold thickness (mm).
    - **Insulin:** Insulin measurement recorded in the dataset.
    - **BMI:** Body Mass Index, a measure based on height and weight.
    - **Diabetes Pedigree Function:** A dataset feature representing diabetes-related family-history information.
    - **Age:** Age in years.

    The measurements should be entered in the same units and format as the dataset.
    """)

# --------------------------------------------------
# Input form
# --------------------------------------------------
st.subheader("Enter the medical measurements")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            help="Number of pregnancies recorded."
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=300.0,
            value=120.0,
            help="Blood glucose measurement."
        )

        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )

    with col2:
        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=900.0,
            value=80.0,
            help="Insulin measurement as recorded in the dataset."
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            help="Body Mass Index."
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            help="A dataset-specific feature related to diabetes family history."
        )

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=30
        )

    submitted = st.form_submit_button(
        "Predict Outcome",
        use_container_width=True
    )

# --------------------------------------------------
# Prediction result
# --------------------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    prediction = int(model.predict(input_data)[0])

    # Display a plain-language result
    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "The model predicted the positive outcome class "
            "in the dataset (Class 1)."
        )
        st.write(
            "In simple words: the model's output matches the dataset's "
            "diabetes-positive label for this input."
        )
    else:
        st.success(
            "The model predicted the negative outcome class "
            "in the dataset (Class 0)."
        )
        st.write(
            "In simple words: the model's output matches the dataset's "
            "diabetes-negative label for this input."
        )

    # Show model probability if supported
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        class_labels = list(model.classes_)
        positive_index = class_labels.index(1)
        positive_probability = float(probabilities[positive_index])

        st.metric(
            "Model-estimated probability for Class 1",
            f"{positive_probability * 100:.1f}%"
        )

        st.caption(
            "This is the model's estimated probability for the dataset's "
            "Class 1 label—not a person's medical risk or diagnostic certainty."
        )

    st.info(
        "The prediction is based on patterns learned from a specific dataset. "
        "It may be wrong and should not be used to make healthcare decisions."
    )