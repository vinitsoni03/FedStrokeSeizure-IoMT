"""
Model Evaluation Module

This module handles model evaluation and performance metrics.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_processed_data():
    """Load processed data from CSV files."""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    
    logger.info(f"Data loaded: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    return X_train, X_test, y_train, y_test


def load_model(model_path: str):
    """Load trained model."""
    if not Path(model_path).exists():
        logger.warning(f"Model not found at {model_path}")
        return None
    
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model performance."""
    logger.info("Evaluating model...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'auc_score': float(roc_auc_score(y_test, y_pred_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"AUC Score: {metrics['auc_score']:.4f}")
    
    return metrics


def save_metrics(metrics: dict, output_path: str) -> None:
    """Save evaluation metrics to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {output_path}")


def main():
    """Main evaluation pipeline."""
    logger.info("Starting model evaluation pipeline...")
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Load model
    model = load_model("models/stroke_model.pkl")
    
    if model is None:
        logger.warning("Skipping evaluation - model not found")
        return
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save metrics
    save_metrics(metrics, "models/evaluation_metrics.json")
    
    logger.info("Evaluation pipeline completed successfully")


if __name__ == "__main__":
    main()
