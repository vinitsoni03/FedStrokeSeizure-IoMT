from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI(title="Seizure Prediction API")

# Load trained model
try:
    model = joblib.load("models/stroke_model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


# Health check endpoint
@app.get("/health")
def health():
    return {"status": "API running"}


# Input schema for Stroke Model (10 features)
class PatientData(BaseModel):
    gender: float
    age: float
    hypertension: float
    heart_disease: float
    ever_married: float
    work_type: float
    Residence_type: float
    avg_glucose_level: float
    bmi: float
    smoking_status: float


# Stroke Prediction endpoint
@app.post("/predict_stroke")
def predict_stroke(payload: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Extract 10 features in the exact order the model expects
    features = np.array([[
        payload.gender,
        payload.age,
        payload.hypertension,
        payload.heart_disease,
        payload.ever_married,
        payload.work_type,
        payload.Residence_type,
        payload.avg_glucose_level,
        payload.bmi,
        payload.smoking_status
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "model": "XGBoost Stroke Model",
        "prediction": int(prediction),
        "probability": float(probability)
    }
