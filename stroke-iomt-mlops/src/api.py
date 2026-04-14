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


# Input schema
class EEGSignal(BaseModel):
    data: list[float]
    channels: int = 23
    timesteps: int = 256


# Preprocessing function
def preprocess_payload(payload: EEGSignal):
    raw_data = np.array(payload.data)

    expected_size = payload.channels * payload.timesteps
    if raw_data.size != expected_size:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_size} values, got {raw_data.size}"
        )

    # Reshape to (samples, channels, timesteps)
    signals = raw_data.reshape(1, payload.channels, payload.timesteps)

    # Normalize (Z-score)
    mean = np.mean(signals, axis=2, keepdims=True)
    std = np.std(signals, axis=2, keepdims=True)
    std[std == 0] = 1e-8

    signals_norm = (signals - mean) / std
    return signals_norm


# Feature extraction (IMPORTANT for your model)
def extract_features(signals_norm):
    mean_feat = np.mean(signals_norm, axis=2)
    var_feat = np.var(signals_norm, axis=2)
    min_feat = np.min(signals_norm, axis=2)
    max_feat = np.max(signals_norm, axis=2)

    return np.concatenate([mean_feat, var_feat, min_feat, max_feat], axis=1)


# Random Forest endpoint
@app.post("/predict_rf")
def predict_rf(payload: EEGSignal):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    signals_norm = preprocess_payload(payload)
    features = extract_features(signals_norm)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "model": "Random Forest",
        "prediction": int(prediction),
        "probability": float(probability)
    }


# CNN endpoint (fallback using same model)
@app.post("/predict_cnn")
def predict_cnn(payload: EEGSignal):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    signals_norm = preprocess_payload(payload)
    features = extract_features(signals_norm)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "model": "CNN (fallback using RF)",
        "prediction": int(prediction),
        "probability": float(probability)
    }
