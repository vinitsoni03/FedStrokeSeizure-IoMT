import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from data_preprocessing import load_data, normalize_signals, extract_features_for_rf, prepare_data_for_cnn

def plot_confusion_matrix(y_true, y_pred, title, filename, plots_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def plot_roc_curve(y_true, y_prob, title, filename, plots_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

def print_model_block(model_name, metrics_train, metrics_val):
    print(f"\n----- {model_name.upper()} -----")
    print(f"Accuracy:  {metrics_val['Accuracy']:.4f}")
    print(f"Precision: {metrics_val['Precision']:.4f}")
    print(f"Recall:    {metrics_val['Recall']:.4f}")
    print(f"F1-score:  {metrics_val['F1-score']:.4f}")
    print(f"ROC-AUC:   {metrics_val['ROC-AUC']:.4f}")
    print("\n[Analysis] Training vs Validation Performance:")
    print(f"  Train Accuracy: {metrics_train['Accuracy']:.4f} | Val Accuracy: {metrics_val['Accuracy']:.4f}")
    print(f"  Train Loss/Gap check -> Difference: {abs(metrics_train['Accuracy'] - metrics_val['Accuracy']):.4f}")
    
    if (metrics_train['Accuracy'] - metrics_val['Accuracy']) > 0.15:
        print("  WARNING: High Overfitting Detected!")
    elif (metrics_val['Accuracy'] < 0.60):
        print("  WARNING: Underfitting Detected!")
    else:
        print("  Model generalizes well.")

def evaluate():
    metrics_export = {}

    print("Loading Validation Data...")
    X_val_raw, y_val = load_data("val")
    X_val_norm = normalize_signals(X_val_raw)
    
    print("Loading Training Data (for validation check)...")
    X_train_raw, y_train = load_data("train")
    X_train_norm = normalize_signals(X_train_raw)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Evaluate RF
    rf_path = os.path.join(project_root, "models", "random_forest.pkl")
    plots_dir = os.path.join(project_root, "outputs", "plots")
    if os.path.exists(rf_path):
        print("\nExtracting Features for Random Forest...")
        rf = joblib.load(rf_path)
        X_val_rf = extract_features_for_rf(X_val_norm)
        X_train_rf = extract_features_for_rf(X_train_norm)
        
        y_val_pred_rf = rf.predict(X_val_rf)
        y_val_prob_rf = rf.predict_proba(X_val_rf)[:, 1]
        
        y_train_pred_rf = rf.predict(X_train_rf)
        y_train_prob_rf = rf.predict_proba(X_train_rf)[:, 1]
        
        rf_metrics_val = compute_metrics(y_val, y_val_pred_rf, y_val_prob_rf)
        rf_metrics_train = compute_metrics(y_train, y_train_pred_rf, y_train_prob_rf)
        
        print_model_block("Random Forest", rf_metrics_train, rf_metrics_val)
        
        plot_confusion_matrix(y_val, y_val_pred_rf, "Random Forest Confusion Matrix", "rf_confusion_matrix.png", plots_dir)
        plot_roc_curve(y_val, y_val_prob_rf, "Random Forest ROC Curve", "rf_roc_curve.png", plots_dir)
        
        metrics_export["Random Forest"] = {
            "validation": rf_metrics_val,
            "train": rf_metrics_train
        }
    else:
        rf_metrics_val = None

    # Evaluate CNN
    cnn_path = os.path.join(project_root, "models", "cnn_model.h5")
    if os.path.exists(cnn_path):
        print("\nPreparing Tensors for CNN...")
        cnn = load_model(cnn_path)
        X_val_cnn = prepare_data_for_cnn(X_val_norm)
        X_train_cnn = prepare_data_for_cnn(X_train_norm)
        
        y_val_prob_cnn = cnn.predict(X_val_cnn).ravel()
        y_val_pred_cnn = (y_val_prob_cnn > 0.5).astype(int)
        
        # Taking a subset of train data for calculation speed
        # We only really need validation metric but checking train to prove non-overfitting
        y_train_prob_cnn = cnn.predict(X_train_cnn).ravel()
        y_train_pred_cnn = (y_train_prob_cnn > 0.5).astype(int)

        cnn_metrics_val = compute_metrics(y_val, y_val_pred_cnn, y_val_prob_cnn)
        cnn_metrics_train = compute_metrics(y_train, y_train_pred_cnn, y_train_prob_cnn)
        
        print_model_block("CNN", cnn_metrics_train, cnn_metrics_val)
        
        plot_confusion_matrix(y_val, y_val_pred_cnn, "CNN Confusion Matrix", "cnn_confusion_matrix.png", plots_dir)
        plot_roc_curve(y_val, y_val_prob_cnn, "CNN ROC Curve", "cnn_roc_curve.png", plots_dir)

        metrics_export["CNN"] = {
            "validation": cnn_metrics_val,
            "train": cnn_metrics_train
        }
    else:
        cnn_metrics_val = None

    if rf_metrics_val and cnn_metrics_val:
        # Save metrics to JSON
        outputs_dir = os.path.join(project_root, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        with open(os.path.join(outputs_dir, "metrics.json"), "w") as f:
            json.dump(metrics_export, f, indent=4)
        print("\n[INFO] Saved metrics.json to outputs/ directory.")

        print("\n" + "="*50)
        print(f"{'COMPARISON TABLE - VALIDATION SET':^50}")
        print("="*50)
        print(f"| {'Metric':<14} | {'Random Forest':<15} | {'CNN':<10} |")
        print("-" * 47)
        for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
            print(f"| {metric:<14} | {rf_metrics_val[metric]:<15.4f} | {cnn_metrics_val[metric]:<10.4f} |")
        print("="*50)

if __name__ == "__main__":
    evaluate()
