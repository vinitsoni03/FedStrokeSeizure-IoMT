from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

app = FastAPI(title="Seizure Prediction API")

rf_model = None
cnn_model = None

# Load models
@app.on_event("startup")
def load_models():
    global rf_model, cnn_model

    rf_path = "models/random_forest.pkl"
    cnn_path = "models/cnn_model.h5"

    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        print("✅ Random Forest Loaded")

    if os.path.exists(cnn_path):
        cnn_model = load_model(cnn_path)
        print("✅ CNN Model Loaded")


# Health check endpoint
@app.get("/health")
def health():
    return {"status": "API running"}


class EEGSignal(BaseModel):
    data: list[float]
    channels: int = 23
    timesteps: int = 256


def preprocess_payload(payload: EEGSignal):
    raw_data = np.array(payload.data)

    expected_size = payload.channels * payload.timesteps
    if raw_data.size != expected_size:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_size} values, got {raw_data.size}"
        )

    signals = raw_data.reshape(1, payload.channels, payload.timesteps)

    # Normalize
    mean = np.mean(signals, axis=2, keepdims=True)
    std = np.std(signals, axis=2, keepdims=True)
    std[std == 0] = 1e-8

    signals_norm = (signals - mean) / std

    return signals_norm


@app.post("/predict_rf")
def predict_rf(payload: EEGSignal):
    if rf_model is None:
        raise HTTPException(status_code=503, detail="RF model not loaded")

    signals_norm = preprocess_payload(payload)

    # Feature extraction
    mean_feat = np.mean(signals_norm, axis=2)
    var_feat = np.var(signals_norm, axis=2)
    min_feat = np.min(signals_norm, axis=2)
    max_feat = np.max(signals_norm, axis=2)

    features = np.concatenate([mean_feat, var_feat, min_feat, max_feat], axis=1)

    prediction = rf_model.predict(features)[0]
    probability = rf_model.predict_proba(features)[0, 1]

    return {
        "model": "Random Forest",
        "prediction": int(prediction),
        "probability": float(probability)
    }


@app.post("/predict_cnn")
def predict_cnn(payload: EEGSignal):
    if cnn_model is None:
        raise HTTPException(status_code=503, detail="CNN model not loaded")

    signals_norm = preprocess_payload(payload)

    # CNN expects (samples, timesteps, channels)
    cnn_input = np.transpose(signals_norm, (0, 2, 1))

    probability = float(cnn_model.predict(cnn_input, verbose=0).ravel()[0])
    prediction = int(probability > 0.5)

    return {
        "model": "CNN",
        "prediction": prediction,
        "probability": probability
    }