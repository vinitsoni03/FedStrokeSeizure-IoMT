from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

app = FastAPI(title="FedStrokeSeizure-IoMT API", description="API serving multiple models for Stroke and Seizure prediction comparison.")

# Model Globals
stroke_model = None
seizure_rf_model = None
seizure_cnn_model = None

# Load models on startup
@app.on_event("startup")
def load_models():
    global stroke_model, seizure_rf_model, seizure_cnn_model

    stroke_path = "models/stroke_model.pkl"
    rf_path = "models/random_forest.pkl"
    cnn_path = "models/cnn_model.h5"

    if os.path.exists(stroke_path):
        stroke_model = joblib.load(stroke_path)
        print("✅ Stroke Model Loaded")

    if os.path.exists(rf_path):
        seizure_rf_model = joblib.load(rf_path)
        print("✅ Seizure Random Forest Loaded")

    if os.path.exists(cnn_path):
        seizure_cnn_model = load_model(cnn_path)
        print("✅ Seizure CNN Model Loaded")


@app.get("/health")
def health():
    return {"status": "API running properly with multi-model support"}


# ==========================================
# 1. STROKE PREDICTION (Tabular Data)
# ==========================================
class StrokePatientData(BaseModel):
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


@app.post("/predict_stroke")
def predict_stroke(payload: StrokePatientData):
    if stroke_model is None:
        raise HTTPException(status_code=503, detail="Stroke model not loaded")

    features = np.array([[
        payload.gender, payload.age, payload.hypertension, payload.heart_disease,
        payload.ever_married, payload.work_type, payload.Residence_type,
        payload.avg_glucose_level, payload.bmi, payload.smoking_status
    ]])

    prediction = stroke_model.predict(features)[0]
    probability = stroke_model.predict_proba(features)[0][1]

    return {
        "model": "XGBoost Stroke Risk Model",
        "prediction": int(prediction),
        "probability": float(probability)
    }


# ==========================================
# 2. SEIZURE PREDICTION (EEG Signals)
# ==========================================
class EEGSignal(BaseModel):
    data: list[float]
    channels: int = 23
    timesteps: int = 256


def preprocess_eeg(payload: EEGSignal):
    raw_data = np.array(payload.data)
    expected_size = payload.channels * payload.timesteps
    if raw_data.size != expected_size:
        raise HTTPException(status_code=400, detail=f"Expected {expected_size} values, got {raw_data.size}")

    signals = raw_data.reshape(1, payload.channels, payload.timesteps)
    mean = np.mean(signals, axis=2, keepdims=True)
    std = np.std(signals, axis=2, keepdims=True)
    std[std == 0] = 1e-8
    return (signals - mean) / std


@app.post("/predict_seizure_rf")
def predict_seizure_rf(payload: EEGSignal):
    """Predict Seizures utilizing the standalone Random Forest Model (Comparitive Model 1)"""
    if seizure_rf_model is None:
        raise HTTPException(status_code=503, detail="Seizure RF model not loaded")

    signals_norm = preprocess_eeg(payload)
    
    # Feature extraction (channels * 4 extracted stats)
    mean_feat = np.mean(signals_norm, axis=2)
    var_feat = np.var(signals_norm, axis=2)
    min_feat = np.min(signals_norm, axis=2)
    max_feat = np.max(signals_norm, axis=2)
    features = np.concatenate([mean_feat, var_feat, min_feat, max_feat], axis=1)

    prediction = seizure_rf_model.predict(features)[0]
    probability = seizure_rf_model.predict_proba(features)[0][1]

    return {
        "model": "Random Forest Seizure Model",
        "prediction": int(prediction),
        "probability": float(probability)
    }


@app.post("/predict_seizure_cnn")
def predict_seizure_cnn(payload: EEGSignal):
    """Predict Seizures utilizing the 1D-CNN Deep Learning Model (Comparitive Model 2)"""
    if seizure_cnn_model is None:
        raise HTTPException(status_code=503, detail="Seizure CNN model not loaded")

    signals_norm = preprocess_eeg(payload)
    cnn_input = np.transpose(signals_norm, (0, 2, 1))

    probability = float(seizure_cnn_model.predict(cnn_input, verbose=0).ravel()[0])
    prediction = int(probability > 0.5)

    return {
        "model": "CNN Seizure Deep Learning Model",
        "prediction": prediction,
        "probability": probability
    }
