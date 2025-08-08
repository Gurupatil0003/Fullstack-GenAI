b.py

```python
from fastapi import FastAPI
from pydantic import BaseModel
from a import generate_health_insight

app = FastAPI()

class PatientData(BaseModel):
    name: str
    age: int
    weight: float
    history: str
    symptoms: str

@app.post("/predict")
def predict(data: PatientData):
    try:
        gemini_insight = generate_health_insight(data.dict())
        hf_analysis = analyze_with_huggingface(data.history + " " + data.symptoms)
        return {
            "gemini_insight": gemini_insight,
        }


```

streamlit.py
```python

import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="AI Health Analytics", layout="wide")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name")
        age = st.number_input("Age", 0, 120)
    with col2:
        weight = st.number_input("Weight (kg)", 0.0, 300.0)

    history = st.text_area("Medical History")
    symptoms = st.text_area("Current Symptoms")
    submit = st.form_submit_button("Analyze Health")

if submit:
    with st.spinner("Analyzing health data using AI..."):
        payload = {
            "name": name,
            "age": age,
            "weight": weight,
            "history": history,
            "symptoms": symptoms
        }
        try:
            res = requests.post("http://localhost:8000/predict", json=payload)
            res.raise_for_status()
            output = res.json()

            st.write(output["gemini_insight"])

        except Exception as e:
            st.error(f" Error: {str(e)}")

```

a.py
```python


import google.generativeai as genai
from google.generativeai import GenerativeModel

genai.configure(api_key="")
model = GenerativeModel("gemini-2.0-flash")


def generate_health_insight(patient_data: dict):
    prompt = f"""
You are an experienced health expert AI.

Analyze the following patient data:

- Name: {patient_data['name']}
- Age: {patient_data['age']}
- Weight: {patient_data['weight']} kg
- Medical History: {patient_data['history']}
- Current Symptoms: {patient_data['symptoms']}

Provide:
1. A detailed analysis of health patterns
2. Potential disease risks
3. Personalized recommendations for preventive care and lifestyle adjustments
"""
    response = model.generate_content(prompt)
    return response.text

```
