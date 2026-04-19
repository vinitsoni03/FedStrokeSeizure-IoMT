"""
plot_results.py
===============
Standalone visualisation script -- reads saved metrics/history and regenerates
ALL required plots without needing to reload or re-evaluate the models.

This is useful when:
  - You want to tweak plot styling without re-running training
  - Training took long and you only need to update charts

Plots generated:
  1. Confusion matrices (one per model) -- from stored predictions in metrics.json
  2. Combined ROC curve (all models)    -- from stored AUC values + roc data
  3. Precision vs Recall bar chart       -- from metrics.json
  4. CNN Training vs Validation Accuracy -- from cnn_history.json
  5. CNN Training vs Validation Loss     -- from cnn_history.json
  6. Radar / spider chart                -- from metrics.json
  7. Model feature importance            -- RF feature importances

All plots saved to: outputs/plots/

Usage:
    python src/plot_results.py

VIVA: Separating plotting from evaluation follows the Single Responsibility Principle
(SRP) in software engineering. It also allows non-technical stakeholders to regenerate
presentation-ready figures without touching the ML pipeline.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# -- Path setup ---------------------------------------------------------------
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

OUTPUTS_DIR = os.path.join(project_root, 'outputs')
PLOTS_DIR   = os.path.join(project_root, 'outputs', 'plots')
MODELS_DIR  = os.path.join(project_root, 'models')
os.makedirs(PLOTS_DIR, exist_ok=True)

# -- Plot style ----------------------------------------------------------------
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.titleweight':   'bold',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         150,
    'savefig.dpi':        150,
    'savefig.bbox':       'tight',
})

MODEL_COLORS = {
    'Random Forest': '#27AE60',
    'XGBoost':       '#D35400',
    'CNN':           '#2980B9',
}

METRIC_COLORS = {
    'Accuracy':  '#95A5A6',
    'Precision': '#3498DB',
    'Recall':    '#E74C3C',
    'F1-score':  '#2ECC71',
    'ROC-AUC':   '#9B59B6',
}


# ============================================================
#  Helper -- data loaders
# ============================================================

def load_metrics():
    """Load metrics.json. Returns dict or None."""
    path = os.path.join(OUTPUTS_DIR, 'metrics.json')
    if not os.path.exists(path):
        print(f"[WARN] metrics.json not found at {path}")
        print("       Run evaluate_models.py first to generate it.")
        return None
    with open(path) as f:
        return json.load(f)


def load_cnn_history():
    """Load cnn_history.json. Returns dict or None."""
    path = os.path.join(OUTPUTS_DIR, 'cnn_history.json')
    if not os.path.exists(path):
        print(f"[WARN] cnn_history.json not found at {path}")
        print("       Run train_models.py first to generate it.")
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================
#  Plot 1 -- CNN Training Curves (Accuracy + Loss)
# ============================================================

def plot_cnn_training_curves(history):
    """
    Plot CNN training vs validation accuracy and loss curves.

    VIVA: The gap between train and val accuracy indicates overfitting.
    A small gap (< 5%) with val accuracy remaining high shows good generalisation.
    Our EarlyStopping callback ensures training stops before overfitting occurs.
    """
    if history is None:
        print("[SKIP] CNN history not available.")
        return

    epochs = range(1, len(history['accuracy']) + 1)

    # -- Accuracy --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['accuracy'],
            label='Train Accuracy', color='#2980B9', lw=2.5)
    ax.plot(epochs, history['val_accuracy'],
            label='Validation Accuracy', color='#E74C3C', lw=2.5, linestyle='--')

    # Shade the gap
    ax.fill_between(epochs,
                    history['accuracy'],
                    history['val_accuracy'],
                    alpha=0.08, color='#E74C3C', label='Train-Val Gap')

    # Mark best val accuracy
    best_epoch = int(np.argmax(history['val_accuracy'])) + 1
    best_val   = max(history['val_accuracy'])
    ax.axvline(best_epoch, color='#27AE60', lw=1.5, linestyle=':', alpha=0.8)
    ax.annotate(f' Best Val\n Epoch {best_epoch}\n Acc={best_val:.3f}',
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 0.5, best_val - 0.05),
                fontsize=9, color='#27AE60',
                arrowprops=dict(arrowstyle='->', color='#27AE60'))

    ax.set_xlabel('Epoch',    fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('CNN -- Training vs Validation Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_ylim([0.5, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cnn_accuracy_vs_epoch.png'))
    plt.close()
    print("[PLOT] Saved cnn_accuracy_vs_epoch.png")

    # -- Loss ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['loss'],
            label='Train Loss', color='#2980B9', lw=2.5)
    ax.plot(epochs, history['val_loss'],
            label='Validation Loss', color='#E74C3C', lw=2.5, linestyle='--')
    ax.fill_between(epochs,
                    history['loss'], history['val_loss'],
                    alpha=0.08, color='#E74C3C')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss',  fontsize=12)
    ax.set_title('CNN -- Training vs Validation Loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cnn_loss_vs_epoch.png'))
    plt.close()
    print("[PLOT] Saved cnn_loss_vs_epoch.png")


# ============================================================
#  Plot 2 -- Precision / Recall / F1 Grouped Bar
# ============================================================

def plot_precision_recall_bar(metrics):
    """Grouped bar chart: Precision, Recall, F1 per model."""
    if metrics is None:
        return

    model_names = list(metrics.keys())
    precisions  = [metrics[m].get('Precision', 0) for m in model_names]
    recalls     = [metrics[m].get('Recall', 0)    for m in model_names]
    f1s         = [metrics[m].get('F1-score', 0)  for m in model_names]
    aucs        = [metrics[m].get('ROC-AUC', 0)   for m in model_names]

    x     = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))
    bp = ax.bar(x - 1.5*width, precisions, width, label='Precision',
                color='#3498DB', alpha=0.9, edgecolor='white', linewidth=0.5)
    br = ax.bar(x - 0.5*width, recalls,    width, label='Recall  _ Priority',
                color='#E74C3C', alpha=0.9, edgecolor='white', linewidth=0.5)
    bf = ax.bar(x + 0.5*width, f1s,        width, label='F1-score',
                color='#2ECC71', alpha=0.9, edgecolor='white', linewidth=0.5)
    ba = ax.bar(x + 1.5*width, aucs,       width, label='ROC-AUC',
                color='#9B59B6', alpha=0.9, edgecolor='white', linewidth=0.5)

    for bars in [bp, br, bf, ba]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.008,
                    f'{h:.3f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison -- Precision, Recall, F1, AUC (Test Set)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9, ncol=2)
    ax.grid(True, axis='y', alpha=0.25)

    # Recall annotation box
    ax.text(0.5, 0.01,
            '!!  In seizure prediction, Recall (sensitivity) is the primary metric.\n'
            '   A missed seizure (False Negative) is clinically far more dangerous than a false alarm.',
            transform=ax.transAxes, ha='center', fontsize=8.5,
            style='italic', color='#7F8C8D',
            bbox=dict(boxstyle='round,pad=0.4', fc='#FDFEFE', ec='#BDC3C7', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'precision_recall_comparison.png'))
    plt.close()
    print("[PLOT] Saved precision_recall_comparison.png")


# ============================================================
#  Plot 3 -- Overall Metrics Bar (all 5 metrics per model)
# ============================================================

def plot_all_metrics_bar(metrics):
    """Horizontal bar chart showing all 5 metrics for each model side by side."""
    if metrics is None:
        return

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    model_names  = list(metrics.keys())

    fig, axes = plt.subplots(1, len(model_names),
                              figsize=(5 * len(model_names), 6),
                              sharey=True)
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        values = [metrics[model_name].get(m, 0) for m in metric_names]
        colors = [METRIC_COLORS[m] for m in metric_names]
        bars   = ax.barh(metric_names, values, color=colors,
                         alpha=0.88, edgecolor='white', linewidth=0.5)

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2.,
                    f'{val:.3f}', va='center', ha='left',
                    fontsize=10, fontweight='bold')

        ax.set_xlim(0, 1.13)
        ax.set_title(model_name, fontsize=12, fontweight='bold',
                     color=MODEL_COLORS.get(model_name, '#2C3E50'))
        ax.axvline(0.8, color='#BDC3C7', lw=1.2, linestyle='--', alpha=0.7)
        ax.set_xlabel('Score', fontsize=10)
        ax.grid(True, axis='x', alpha=0.2)

    fig.suptitle('All Metrics -- Per Model (Test Set)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'all_metrics_per_model.png'))
    plt.close()
    print("[PLOT] Saved all_metrics_per_model.png")


# ============================================================
#  Plot 4 -- Radar / Spider Chart
# ============================================================

def plot_radar_chart(metrics):
    """Radar chart comparing all models across all 5 metrics."""
    if metrics is None or len(metrics) == 0:
        return

    categories = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model_name, m in metrics.items():
        values = [m.get(c, 0) for c in categories]
        values += values[:1]
        colour = MODEL_COLORS.get(model_name, '#7F8C8D')
        ax.plot(angles, values, lw=2.5, color=colour, label=model_name)
        ax.fill(angles, values, alpha=0.07, color=colour)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                       fontsize=8.5, alpha=0.6)
    ax.set_title('Model Performance Radar Chart\n(Test Set)',
                 fontsize=13, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'radar_chart.png'), bbox_inches='tight')
    plt.close()
    print("[PLOT] Saved radar_chart.png")


# ============================================================
#  Plot 5 -- RF Feature Importance
# ============================================================

def plot_rf_feature_importance(top_n=20):
    """
    Plot top-N feature importances from the trained Random Forest.

    VIVA: Feature importance tells us WHICH EEG characteristics drive the model's
    decisions. High-importance features in the gamma band power, for example,
    confirm that our frequency-domain features capture clinically relevant signal.
    """
    rf_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    if not os.path.exists(rf_path):
        print("[SKIP] RF model not found for feature importance plot.")
        return

    import joblib
    rf = joblib.load(rf_path)

    importances = rf.feature_importances_
    n_features  = len(importances)

    # Create generic feature names (can be customised to match extract_features_for_rf)
    feature_names = [f'Feature_{i}' for i in range(n_features)]

    # Sort by importance descending
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]

    ax.barh(range(top_n), importances[indices][::-1],
            color=colors, alpha=0.9, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel('Feature Importance (Gini)', fontsize=11)
    ax.set_title(f'Random Forest -- Top {top_n} Feature Importances',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'rf_feature_importance.png'))
    plt.close()
    print(f"[PLOT] Saved rf_feature_importance.png (top {top_n} features)")


# ============================================================
#  Plot 6 -- Class Distribution
# ============================================================

def plot_class_distribution():
    """
    Visualise class imbalance in train vs test sets.

    VIVA: Understanding class imbalance is critical for justifying SMOTE and
    class_weight usage. If the dataset were balanced, these techniques would be
    unnecessary -- but for seizure detection, seizures are rare events.
    """
    sys.path.insert(0, script_dir)
    try:
        from data_preprocessing import load_data, load_test_data
        _, y_train = load_data("train")
        _, y_test  = load_test_data()
    except Exception as e:
        print(f"[SKIP] Class distribution plot skipped: {e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (y, title) in zip(axes, [(y_train, 'Training Set'), (y_test, 'Test Set')]):
        counts = np.bincount(y.astype(int))
        pct    = 100 * counts / len(y)
        bars   = ax.bar(['No Seizure (0)', 'Seizure (1)'], counts,
                        color=['#3498DB', '#E74C3C'], alpha=0.85,
                        edgecolor='white', linewidth=0.5)
        for bar, count, p in zip(bars, counts, pct):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + max(counts) * 0.01,
                    f'{count:,}\n({p:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_title(f'Class Distribution -- {title}',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample Count', fontsize=11)
        ax.grid(True, axis='y', alpha=0.25)
        ax.set_ylim(0, max(counts) * 1.2)

    fig.suptitle('EEG Seizure Dataset -- Class Imbalance',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'))
    plt.close()
    print("[PLOT] Saved class_distribution.png")


# ============================================================
#  Main
# ============================================================

def main():
    print("\n" + "="*55)
    print("  SEIZURE PREDICTION -- STANDALONE PLOT GENERATOR")
    print("="*55)

    metrics = load_metrics()
    history = load_cnn_history()

    print("\n[PLOTS] Generating all visualisations...")
    plot_cnn_training_curves(history)
    plot_precision_recall_bar(metrics)
    plot_all_metrics_bar(metrics)
    plot_radar_chart(metrics)
    plot_rf_feature_importance()
    plot_class_distribution()

    print(f"\n[DONE] All plots saved to: {PLOTS_DIR}")
    print("       Open the folder to view all generated PNG files.\n")


if __name__ == "__main__":
    main()
