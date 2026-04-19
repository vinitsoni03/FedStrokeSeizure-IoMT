"""
evaluate_models.py
==================
Comprehensive evaluation of all trained models on the TRUE HOLDOUT TEST SET.

Models evaluated:
  1. Random Forest (tuned)
  2. XGBoost (tuned)
  3. CNN (improved)

Outputs generated:
  - outputs/metrics.json          -- all metrics in structured JSON
  - outputs/comparison_table.csv  -- clean CSV for report/viva
  - outputs/plots/                -- confusion matrices, ROC curves, bar charts

VIVA: Why evaluate on a held-out TEST set (not validation)_
  The validation set was used to guide hyperparameter tuning and EarlyStopping.
  The model has -- indirectly -- 'seen' the validation set via those decisions.
  Only the true test set (never seen at any training stage) gives an unbiased
  estimate of real-world performance. This is the standard in academic ML papers.

Usage:
    python src/evaluate_models.py
"""

import os
import sys
import json
import csv
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report
)

# -- Path setup ---------------------------------------------------------------
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, script_dir)

from data_preprocessing import (
    load_data, load_test_data, normalize_signals,
    extract_features_for_rf, prepare_data_for_cnn,
)

# -- Directory constants -------------------------------------------------------
MODELS_DIR  = os.path.join(project_root, 'models')
OUTPUTS_DIR = os.path.join(project_root, 'outputs')
PLOTS_DIR   = os.path.join(project_root, 'outputs', 'plots')
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# -- Plotting style ------------------------------------------------------------
plt.rcParams.update({
    'font.family':   'DejaVu Sans',
    'font.size':     11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.dpi':    150,
})

# Model display names and colour palette
MODEL_COLORS = {
    'Random Forest': '#2ECC71',   # Green
    'XGBoost':       '#E67E22',   # Orange
    'CNN':           '#3498DB',   # Blue
}


# ============================================================
# SECTION 1 -- Data Loading
# ============================================================

def load_evaluation_data():
    """
    Load test set using the SAME normalisation stats from training.

    VIVA: Applying identical preprocessing to test data is mandatory.
    Using different stats would distort the signal and produce invalid metrics.
    """
    print("\n[DATA] Loading test set for final evaluation...")

    # Try to load saved normalisation stats from training
    stats_path = os.path.join(OUTPUTS_DIR, 'norm_stats.npz')
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        norm_stats = (stats['mean'], stats['std'])
        print(f"[DATA] Loaded normalisation stats from {stats_path}")
    else:
        print("[DATA] norm_stats.npz not found -- computing from training set (may differ from training).")
        X_train_raw, _ = load_data("train")
        _, norm_stats = normalize_signals(X_train_raw)

    X_test_raw, y_test = load_test_data()
    X_test_norm, _     = normalize_signals(X_test_raw, fitted_stats=norm_stats)

    # Also load val set for completeness
    X_val_raw, y_val = load_data("val")
    X_val_norm, _    = normalize_signals(X_val_raw, fitted_stats=norm_stats)

    print(f"[DATA] Test  set: {X_test_norm.shape}, seizure%={100*y_test.mean():.2f}%")
    print(f"[DATA] Val   set: {X_val_norm.shape},  seizure%={100*y_val.mean():.2f}%")

    return {
        'X_test_norm': X_test_norm,
        'y_test':      y_test,
        'X_val_norm':  X_val_norm,
        'y_val':       y_val,
    }


# ============================================================
# SECTION 2 -- Metric Computation
# ============================================================

def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute the complete set of clinical evaluation metrics.

    VIVA -- Metric Significance:
      Accuracy  : Overall correctness; misleading when classes are imbalanced.
      Precision : Of all predicted seizures, how many were real_ (False Alarm rate)
      Recall    : Of all real seizures, how many did we catch_ (MOST CRITICAL)
                  A missed seizure can result in injury or death.
      F1-score  : Harmonic mean of Precision and Recall. Balanced view.
      ROC-AUC   : Area under ROC curve. Threshold-independent discriminative ability.
                  AUC > 0.9 is considered excellent for medical applications.

    Returns:
        dict with rounded float values for all 5 metrics
    """
    return {
        'Accuracy':  round(float(accuracy_score(y_true, y_pred)),              4),
        'Precision': round(float(precision_score(y_true, y_pred,
                                                  zero_division=0)),           4),
        'Recall':    round(float(recall_score(y_true, y_pred,
                                               zero_division=0)),              4),
        'F1-score':  round(float(f1_score(y_true, y_pred, zero_division=0)),   4),
        'ROC-AUC':   round(float(roc_auc_score(y_true, y_prob)),               4),
    }


# ============================================================
# SECTION 3 -- Plotting Functions
# ============================================================

def plot_confusion_matrix(y_true, y_pred, model_name, plots_dir):
    """
    Plot a styled confusion matrix with TN/FP/FN/TP annotations.

    VIVA: The confusion matrix shows exactly where the model fails.
    For seizure prediction, we specifically want to minimise FN (bottom-left)
    because that represents missed seizures -- the most dangerous error.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = [['TN', 'FP'], ['FN', 'TP']]

    # Create annotated label matrix  e.g. "TN\n1234"
    annot = np.empty_like(cm).astype(str)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{labels[i][j]}\n{cm[i, j]}"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=annot, fmt='', cmap='Blues',
        ax=ax, linewidths=0.5, linecolor='white',
        xticklabels=['No Seizure', 'Seizure'],
        yticklabels=['No Seizure', 'Seizure'],
        annot_kws={'size': 13, 'weight': 'bold'},
    )
    ax.set_title(f'{model_name} -- Confusion Matrix\n(Test Set)', pad=12)
    ax.set_ylabel('Actual Label',    fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    # Highlight FN cell in red to emphasise the costliest error
    ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False,
                                edgecolor='#E74C3C', lw=3))
    ax.text(0.5, 1.5, '<- Critical\nMissed Seizure',
            ha='center', va='center', fontsize=8, color='#E74C3C', style='italic')

    plt.tight_layout()
    fname = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(os.path.join(plots_dir, fname), dpi=150)
    plt.close()
    print(f"[PLOT] Saved {fname}")
    return fname


def plot_combined_roc_curve(roc_data, plots_dir):
    """
    Plot all model ROC curves on a single chart for direct comparison.

    roc_data: dict of {model_name: (fpr, tpr, auc_score)}

    VIVA: The ROC curve shows the trade-off between True Positive Rate (recall)
    and False Positive Rate at every possible threshold. A model with AUC=0.5
    is no better than random guessing; AUC=1.0 is perfect. For clinical systems,
    we generally accept AUC > 0.85 as clinically acceptable.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for model_name, (fpr, tpr, auc_score) in roc_data.items():
        colour = MODEL_COLORS.get(model_name, 'gray')
        ax.plot(fpr, tpr, lw=2.5, color=colour,
                label=f'{model_name}  (AUC = {auc_score:.3f})')

    # Random-guess baseline
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Guess (AUC = 0.50)')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    ax.set_title('ROC Curve Comparison -- All Models (Test Set)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # Shade AUC region for best model
    best_model = max(roc_data, key=lambda m: roc_data[m][2])
    fpr_b, tpr_b, _ = roc_data[best_model]
    ax.fill_between(fpr_b, tpr_b, alpha=0.07,
                    color=MODEL_COLORS.get(best_model, 'blue'))

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curve_all_models.png'), dpi=150)
    plt.close()
    print("[PLOT] Saved roc_curve_all_models.png")


def plot_precision_recall_bar(all_metrics, plots_dir):
    """
    Grouped bar chart: Precision vs Recall for each model.

    VIVA: This chart visually proves why CNN and XGBoost are superior to RF.
    The baseline RF had recall ~ 24% -- completely unacceptable for medical use.
    After tuning, all models should show substantially improved recall values.
    """
    model_names = list(all_metrics.keys())
    precisions  = [all_metrics[m]['Precision'] for m in model_names]
    recalls     = [all_metrics[m]['Recall']    for m in model_names]
    f1s         = [all_metrics[m]['F1-score']  for m in model_names]

    x     = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_p  = ax.bar(x - width,     precisions, width, label='Precision',
                     color='#3498DB', alpha=0.85, edgecolor='white')
    bars_r  = ax.bar(x,             recalls,    width, label='Recall _ Critical',
                     color='#E74C3C', alpha=0.85, edgecolor='white')
    bars_f1 = ax.bar(x + width,     f1s,        width, label='F1-score',
                     color='#2ECC71', alpha=0.85, edgecolor='white')

    # Add value labels on bars
    for bars in [bars_p, bars_r, bars_f1]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall & F1-score -- All Models (Test Set)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)

    # Recall importance annotation
    ax.annotate('!! In medical AI, Recall is the priority metric.\n'
                '  Missing a seizure is far more dangerous than a false alarm.',
                xy=(0.5, 0.02), xycoords='axes fraction',
                ha='center', fontsize=8.5, style='italic',
                color='#7F8C8D',
                bbox=dict(boxstyle='round,pad=0.3', fc='#F8F9FA', ec='#BDC3C7'))

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'precision_recall_comparison.png'), dpi=150)
    plt.close()
    print("[PLOT] Saved precision_recall_comparison.png")


def plot_metrics_radar(all_metrics, plots_dir):
    """
    Radar / spider chart comparing all 5 metrics across all models.
    Gives an excellent visual overview for presentations.
    """
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model_name, metrics in all_metrics.items():
        values = [metrics[c] for c in categories]
        values += values[:1]
        colour = MODEL_COLORS.get(model_name, 'gray')
        ax.plot(angles, values, lw=2, linestyle='solid', color=colour, label=model_name)
        ax.fill(angles, values, alpha=0.08, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, alpha=0.6)
    ax.set_title('Model Performance Radar Chart (Test Set)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'radar_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[PLOT] Saved radar_chart.png")


# ============================================================
# SECTION 4 -- Model Evaluation Functions
# ============================================================

def evaluate_random_forest(data):
    """Evaluate the tuned Random Forest on the test set."""
    rf_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    if not os.path.exists(rf_path):
        print("[RF] Model not found. Train first with train_models.py")
        return None, None, None

    print("\n[EVAL] Random Forest...")
    rf = joblib.load(rf_path)

    # Use pre-extracted features if available
    test_feat_path = os.path.join(OUTPUTS_DIR, 'rf_X_test.npy')
    if os.path.exists(test_feat_path):
        X_test_feat = np.load(test_feat_path)
    else:
        X_test_feat = extract_features_for_rf(data['X_test_norm'])

    y_pred = rf.predict(X_test_feat)
    y_prob = rf.predict_proba(X_test_feat)[:, 1]

    metrics = compute_metrics(data['y_test'], y_pred, y_prob)
    fpr, tpr, _ = roc_curve(data['y_test'], y_prob)

    print(f"  Recall={metrics['Recall']:.4f}  F1={metrics['F1-score']:.4f}  AUC={metrics['ROC-AUC']:.4f}")
    print(classification_report(data['y_test'], y_pred,
                                 target_names=['No Seizure', 'Seizure']))
    return metrics, y_pred, (fpr, tpr, metrics['ROC-AUC'])


def evaluate_xgboost(data):
    """Evaluate the tuned XGBoost on the test set."""
    xgb_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    if not os.path.exists(xgb_path):
        print("[XGB] Model not found. Train first with train_models.py")
        return None, None, None

    print("\n[EVAL] XGBoost...")
    xgb_model = joblib.load(xgb_path)

    test_feat_path = os.path.join(OUTPUTS_DIR, 'rf_X_test.npy')
    if os.path.exists(test_feat_path):
        X_test_feat = np.load(test_feat_path)
    else:
        X_test_feat = extract_features_for_rf(data['X_test_norm'])

    y_pred = xgb_model.predict(X_test_feat)
    y_prob = xgb_model.predict_proba(X_test_feat)[:, 1]

    metrics = compute_metrics(data['y_test'], y_pred, y_prob)
    fpr, tpr, _ = roc_curve(data['y_test'], y_prob)

    print(f"  Recall={metrics['Recall']:.4f}  F1={metrics['F1-score']:.4f}  AUC={metrics['ROC-AUC']:.4f}")
    print(classification_report(data['y_test'], y_pred,
                                 target_names=['No Seizure', 'Seizure']))
    return metrics, y_pred, (fpr, tpr, metrics['ROC-AUC'])


def evaluate_cnn(data):
    """Evaluate the improved CNN on the test set."""
    cnn_path = os.path.join(MODELS_DIR, 'cnn_model.h5')
    if not os.path.exists(cnn_path):
        print("[CNN] Model not found. Train first with train_models.py")
        return None, None, None

    print("\n[EVAL] CNN...")
    from tensorflow.keras.models import load_model as keras_load_model
    cnn = keras_load_model(cnn_path)

    X_test_cnn = prepare_data_for_cnn(data['X_test_norm'])
    y_prob = cnn.predict(X_test_cnn, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    metrics = compute_metrics(data['y_test'], y_pred, y_prob)
    fpr, tpr, _ = roc_curve(data['y_test'], y_prob)

    print(f"  Recall={metrics['Recall']:.4f}  F1={metrics['F1-score']:.4f}  AUC={metrics['ROC-AUC']:.4f}")
    print(classification_report(data['y_test'], y_pred,
                                 target_names=['No Seizure', 'Seizure']))
    return metrics, y_pred, (fpr, tpr, metrics['ROC-AUC'])


# ============================================================
# SECTION 5 -- Reporting
# ============================================================

def print_comparison_table(all_metrics):
    """Pretty-print the comparison table to console."""
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    models       = list(all_metrics.keys())

    # Header
    col_w = 14
    header = f"{'Metric':<{col_w}}" + "".join(f"{m:^{col_w}}" for m in models)
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print(f"{'COMPARISON TABLE -- TEST SET':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print(sep)

    for metric in metrics_list:
        row = f"{metric:<{col_w}}"
        for model in models:
            val = all_metrics[model][metric]
            marker = ' *' if metric == 'Recall' else ''
            row += f"{val:^{col_w}.4f}"
        if metric == 'Recall':
            row += '  <- CRITICAL (Medical Priority)'
        print(row)

    print("=" * len(header))

    # Determine best model by recall
    best_model = max(all_metrics, key=lambda m: all_metrics[m]['Recall'])
    print(f"\n  * Best Recall: {best_model} ({all_metrics[best_model]['Recall']:.4f})")
    print(f"  -> Recommended for clinical deployment: {best_model}")


def save_metrics_json(all_metrics):
    """Save all metrics to outputs/metrics.json."""
    path = os.path.join(OUTPUTS_DIR, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n[SAVE] metrics.json -> {path}")


def save_comparison_csv(all_metrics):
    """Save comparison table to outputs/comparison_table.csv."""
    path = os.path.join(OUTPUTS_DIR, 'comparison_table.csv')
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model'] + metrics_list)
        for model_name, metrics in all_metrics.items():
            writer.writerow(
                [model_name] + [f"{metrics[m]:.4f}" for m in metrics_list]
            )

    print(f"[SAVE] comparison_table.csv -> {path}")


# ============================================================
# SECTION 6 -- Main Entry Point
# ============================================================

def evaluate():
    """Full evaluation pipeline: load data -> evaluate all models -> plot -> save."""
    print("\n" + "="*60)
    print("  SEIZURE PREDICTION -- MODEL EVALUATION PIPELINE")
    print("="*60)

    # -- Load data ------------------------------------------------
    data = load_evaluation_data()

    # -- Evaluate all models --------------------------------------
    all_metrics = {}
    roc_data    = {}
    all_y_pred  = {}

    rf_metrics, rf_preds, rf_roc = evaluate_random_forest(data)
    if rf_metrics:
        all_metrics['Random Forest'] = rf_metrics
        roc_data['Random Forest']    = rf_roc
        all_y_pred['Random Forest']  = rf_preds

    xgb_metrics, xgb_preds, xgb_roc = evaluate_xgboost(data)
    if xgb_metrics:
        all_metrics['XGBoost'] = xgb_metrics
        roc_data['XGBoost']    = xgb_roc
        all_y_pred['XGBoost']  = xgb_preds

    cnn_metrics, cnn_preds, cnn_roc = evaluate_cnn(data)
    if cnn_metrics:
        all_metrics['CNN'] = cnn_metrics
        roc_data['CNN']    = cnn_roc
        all_y_pred['CNN']  = cnn_preds

    if not all_metrics:
        print("\n[ERROR] No models found. Run train_models.py first.")
        return

    # -- Print console table --------------------------------------
    print_comparison_table(all_metrics)

    # -- Generate all plots ---------------------------------------
    print("\n[PLOTS] Generating visualisations...")

    # Confusion matrices
    for model_name, y_pred in all_y_pred.items():
        plot_confusion_matrix(data['y_test'], y_pred, model_name, PLOTS_DIR)

    # Combined ROC curve
    if len(roc_data) > 0:
        plot_combined_roc_curve(roc_data, PLOTS_DIR)

    # Precision / Recall bar chart
    plot_precision_recall_bar(all_metrics, PLOTS_DIR)

    # Radar chart
    plot_metrics_radar(all_metrics, PLOTS_DIR)

    # -- Save outputs ---------------------------------------------
    save_metrics_json(all_metrics)
    save_comparison_csv(all_metrics)

    # -- Final recommendation -------------------------------------
    print("\n" + "="*60)
    print("  CLINICAL RECOMMENDATION")
    print("="*60)
    best_recall_model = max(all_metrics, key=lambda m: all_metrics[m]['Recall'])
    best_auc_model    = max(all_metrics, key=lambda m: all_metrics[m]['ROC-AUC'])
    print("  Highest Recall : " + best_recall_model + " (" + str(all_metrics[best_recall_model]['Recall']) + ")")
    print("  Highest AUC    : " + best_auc_model    + " (" + str(all_metrics[best_auc_model]['ROC-AUC']) + ")")
    print("\n  -> Recommended model for deployment: " + best_recall_model)
    print("    (Maximises seizure detection, minimises missed events)")
    print("="*60)

    print(f"\n[DONE] All outputs saved to {OUTPUTS_DIR}/")
    print("[DONE] All plots  saved to  {PLOTS_DIR}/")


if __name__ == "__main__":
    evaluate()
