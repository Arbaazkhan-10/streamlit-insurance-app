# app.py

import streamlit as st
import pickle
import numpy as np

# Load the model
with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Insurance Cost Predictor")
st.title("💸 Medical Insurance Cost Prediction App")

st.markdown("### Enter the details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=100, step=1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Predict button
if st.button("Predict Charges"):
    input_data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }

    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame([input_data])
    
    # Predict
    prediction = model.predict(df)[0]
    st.success(f"💰 Predicted Insurance Charges: $ {prediction:,.2f}")
