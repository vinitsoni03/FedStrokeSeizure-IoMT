"""
train_models.py
===============
UNIFIED training pipeline for all three models:
  1. Random Forest   -- hyperparameter-tuned with RandomizedSearchCV (scoring=recall)
  2. XGBoost         -- scale_pos_weight class balancing + RandomizedSearchCV
  3. CNN (1D)        -- deeper architecture, class weights, EarlyStopping

DESIGN PRINCIPLES:
  - Zero data leakage:  scaler fitted ONLY on training split
  - Class imbalance:    SMOTE for RF/XGB, class_weight for CNN
  - Hyperparameter opt: RecallOptimised search for maximum clinical sensitivity
  - Model persistence:  all artifacts saved to models/

VIVA TALKING POINTS:
  - Why recall as scoring metric_  A missed seizure is life-threatening; a false alarm
    is merely inconvenient. Clinical ML systems optimise recall (sensitivity) first.
  - Why RandomizedSearchCV over GridSearchCV_  Random search explores a much larger
    hyperparameter space in the same computational budget -- equally effective in practice.
  - Why scale_pos_weight in XGBoost_  Natively tells XGBoost to weight the minority
    (seizure) gradient updates more heavily, analogous to weighted loss in neural nets.
  - Why EarlyStopping_  Prevents overfitting without manual epoch selection. Training
    stops automatically when validation loss stops improving.

Usage:
    python src/train_models.py [--skip-rf] [--skip-xgb] [--skip-cnn]
"""

import os
import sys
import json
import argparse
import numpy as np
import joblib

# -- Path setup so this file can import sibling modules ----------------------
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, script_dir)

from data_preprocessing import (
    load_data,
    load_test_data,
    normalize_signals,
    extract_features_for_rf,
    prepare_data_for_cnn,
    get_class_weights,
    apply_smote,
)

# -- Output directories -------------------------------------------------------
MODELS_DIR  = os.path.join(project_root, 'models')
OUTPUTS_DIR = os.path.join(project_root, 'outputs')
PLOTS_DIR   = os.path.join(project_root, 'outputs', 'plots')
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)


# ============================================================
# SECTION 1 -- Data Loading & Split
# ============================================================

def load_all_splits():
    """
    Load train / val / test splits, normalise them consistently.

    VIVA: The normalisation statistics (mean, std) are computed ONLY on the training
    set and then APPLIED to val and test. Computing on val/test would cause data leakage --
    the model would indirectly 'see' information from those sets during training.

    Returns:
        dict with keys: X_train_raw, X_val_raw, X_test_raw,
                        X_train_norm, X_val_norm, X_test_norm,
                        y_train, y_val, y_test, norm_stats
    """
    print("\n[1/3] Loading datasets...")
    X_train_raw, y_train = load_data("train")
    X_val_raw,   y_val   = load_data("val")
    X_test_raw,  y_test  = load_test_data()

    print(f"  Train: {X_train_raw.shape},  Seizure={100*y_train.mean():.2f}%")
    print(f"  Val  : {X_val_raw.shape},    Seizure={100*y_val.mean():.2f}%")
    print(f"  Test : {X_test_raw.shape},   Seizure={100*y_test.mean():.2f}%")

    print("\n[2/3] Normalising (fit on train only)...")
    X_train_norm, norm_stats = normalize_signals(X_train_raw)
    X_val_norm,   _          = normalize_signals(X_val_raw,  fitted_stats=norm_stats)
    X_test_norm,  _          = normalize_signals(X_test_raw, fitted_stats=norm_stats)

    # Save norm stats so evaluate_models.py can apply THE SAME normalisation
    stats_path = os.path.join(OUTPUTS_DIR, 'norm_stats.npz')
    np.savez(stats_path, mean=norm_stats[0], std=norm_stats[1])
    print(f"  Norm stats saved -> {stats_path}")

    return {
        'X_train_raw':  X_train_raw,
        'X_val_raw':    X_val_raw,
        'X_test_raw':   X_test_raw,
        'X_train_norm': X_train_norm,
        'X_val_norm':   X_val_norm,
        'X_test_norm':  X_test_norm,
        'y_train':      y_train,
        'y_val':        y_val,
        'y_test':       y_test,
        'norm_stats':   norm_stats,
    }


# ============================================================
# SECTION 2 -- Random Forest Training
# ============================================================

def train_random_forest(data):
    """
    Train a hyperparameter-tuned Random Forest with SMOTE-balanced features.

    Improvements over baseline:
      - Rich feature set (18 per channel vs 4)
      - SMOTE oversampling of seizure class
      - RandomizedSearchCV optimising RECALL (not accuracy)
      - Saves best model, prints best params

    VIVA: Why hyperparameter tuning matters -- default RF settings (100 trees, no depth
    limit) may overfit or under-learn the minority class. Tuning max_depth and
    min_samples_leaf directly controls the bias-variance tradeoff.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint

    print("\n" + "="*60)
    print("TRAINING: Random Forest (with SMOTE + Recall Tuning)")
    print("="*60)

    # -- Feature extraction (cache to disk for XGBoost reuse) -----
    train_feat_path = os.path.join(OUTPUTS_DIR, 'rf_X_train.npy')
    val_feat_path   = os.path.join(OUTPUTS_DIR, 'rf_X_val.npy')
    test_feat_path  = os.path.join(OUTPUTS_DIR, 'rf_X_test.npy')

    if os.path.exists(train_feat_path):
        print("[RF] Loading pre-extracted features from cache...")
        X_train_feat = np.load(train_feat_path)
        X_val_feat   = np.load(val_feat_path)
        X_test_feat  = np.load(test_feat_path) if os.path.exists(test_feat_path) else extract_features_for_rf(data['X_test_norm'])
    else:
        print("[RF] Extracting features for train/val/test...")
        X_train_feat = extract_features_for_rf(data['X_train_norm'])
        X_val_feat   = extract_features_for_rf(data['X_val_norm'])
        X_test_feat  = extract_features_for_rf(data['X_test_norm'])
        np.save(train_feat_path, X_train_feat)
        np.save(val_feat_path,   X_val_feat)
        np.save(test_feat_path,  X_test_feat)
        print(f"[RF] Features cached to outputs/ for XGBoost reuse.")

    # -- SMOTE oversampling (train only!) -------------------------
    print("\n[RF] Applying SMOTE to training features...")
    X_train_bal, y_train_bal = apply_smote(X_train_feat, data['y_train'])

    # -- Hyperparameter search space ----------------------------------------
    # NOTE: max_depth=None and max_features=0.5 are excluded because they cause
    # extremely long fit times (20-30 min per CV fold) on 37k samples.
    # The restricted space still covers the most impactful parameters.
    param_dist = {
        'n_estimators':      randint(100, 301),         # 100 - 300 trees
        'max_depth':         [5, 8, 10, 15, 20],        # Always bounded
        'min_samples_split': randint(2, 11),
        'min_samples_leaf':  randint(1, 6),
        'max_features':      ['sqrt', 'log2'],          # Fast options only
        'class_weight':      ['balanced', 'balanced_subsample'],
    }

    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    print("\n[RF] Running RandomizedSearchCV (n_iter=10, cv=3, scoring=recall)...")
    print("     This may take several minutes on large datasets...")
    search = RandomizedSearchCV(
        estimator    = base_rf,
        param_distributions = param_dist,
        n_iter       = 10,          # 10 random combinations
        cv           = 3,           # 3-fold cross-validation
        scoring      = 'recall',    # OPTIMISE FOR RECALL (clinically critical)
        n_jobs       = -1,
        random_state = 42,
        verbose      = 2,
    )
    search.fit(X_train_bal, y_train_bal)

    best_rf = search.best_estimator_
    print(f"\n[RF] Best params: {search.best_params_}")
    print(f"[RF] Best CV recall: {search.best_score_:.4f}")

    # -- Save model -----------------------------------------------
    rf_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    joblib.dump(best_rf, rf_path)
    print(f"[RF] Model saved -> {rf_path}")

    # -- Quick validation metrics snapshot ------------------------
    from sklearn.metrics import recall_score, f1_score
    y_val_pred = best_rf.predict(X_val_feat)
    val_recall = recall_score(data['y_val'], y_val_pred)
    val_f1     = f1_score(data['y_val'], y_val_pred)
    print(f"[RF] Val Recall={val_recall:.4f}  Val F1={val_f1:.4f}")

    return best_rf



# ============================================================
# SECTION 3 -- XGBoost Training
# ============================================================

def train_xgboost(data):
    """
    Train an XGBoost classifier with native class imbalance handling.

    WHY XGBoost_
      - Gradient boosting builds trees sequentially, each one correcting errors of
        the previous -- naturally effective on structured/tabular features.
      - scale_pos_weight = (negative samples / positive samples) tells XGBoost to
        weight seizure prediction errors more heavily during gradient computation.
      - Generally outperforms RF on tabular data when tuned properly.

    VIVA: XGBoost is the industry standard for tabular ML competitions (Kaggle winners).
    It handles missing values natively, is fast due to histogram-based tree splitting,
    and supports GPU acceleration.
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("[XGBoost] xgboost not installed. Skipping. Run: pip install xgboost")
        return None

    from sklearn.model_selection import RandomizedSearchCV

    print("\n" + "="*60)
    print("TRAINING: XGBoost (scale_pos_weight + Recall Tuning)")
    print("="*60)

    # -- Load pre-extracted or re-extract features ----------------
    train_feat_path = os.path.join(OUTPUTS_DIR, 'rf_X_train.npy')
    val_feat_path   = os.path.join(OUTPUTS_DIR, 'rf_X_val.npy')
    test_feat_path  = os.path.join(OUTPUTS_DIR, 'rf_X_test.npy')

    if os.path.exists(train_feat_path):
        print("[XGB] Loading pre-extracted features...")
        X_train_feat = np.load(train_feat_path)
        X_val_feat   = np.load(val_feat_path)
        X_test_feat  = np.load(test_feat_path)
    else:
        print("[XGB] Extracting features...")
        X_train_feat = extract_features_for_rf(data['X_train_norm'])
        X_val_feat   = extract_features_for_rf(data['X_val_norm'])
        X_test_feat  = extract_features_for_rf(data['X_test_norm'])

    y_train = data['y_train']
    y_val   = data['y_val']

    # -- Compute class imbalance ratio ----------------------------
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    spw = neg_count / max(pos_count, 1)   # scale_pos_weight
    print(f"[XGB] scale_pos_weight = {spw:.2f}  (neg={neg_count}, pos={pos_count})")

    # -- Hyperparameter search space ------------------------------
    param_dist = {
        'n_estimators':      [100, 200, 300, 400],
        'max_depth':         [3, 4, 5, 6, 7, 8],
        'learning_rate':     [0.01, 0.05, 0.1, 0.2],
        'subsample':         [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree':  [0.5, 0.6, 0.75, 1.0],
        'min_child_weight':  [1, 3, 5, 7],
        'reg_alpha':         [0, 0.01, 0.1, 1.0],
        'gamma':             [0, 0.1, 0.5, 1.0],
    }

    base_xgb = xgb.XGBClassifier(
        scale_pos_weight = spw,
        use_label_encoder = False,
        eval_metric      = 'logloss',
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    )

    print("\n[XGB] Running RandomizedSearchCV (n_iter=20, cv=3, scoring=recall)...")
    search = RandomizedSearchCV(
        estimator           = base_xgb,
        param_distributions = param_dist,
        n_iter              = 20,
        cv                  = 3,
        scoring             = 'recall',
        n_jobs              = -1,
        random_state        = 42,
        verbose             = 2,
    )
    search.fit(X_train_feat, y_train)

    best_xgb = search.best_estimator_
    print(f"\n[XGB] Best params: {search.best_params_}")
    print(f"[XGB] Best CV recall: {search.best_score_:.4f}")

    # -- Save model -----------------------------------------------
    xgb_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    joblib.dump(best_xgb, xgb_path)
    print(f"[XGB] Model saved -> {xgb_path}")

    # -- Quick validation metrics snapshot ------------------------
    from sklearn.metrics import recall_score, f1_score
    y_val_pred = best_xgb.predict(X_val_feat)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1     = f1_score(y_val, y_val_pred)
    print(f"[XGB] Val Recall={val_recall:.4f}  Val F1={val_f1:.4f}")

    return best_xgb


# ============================================================
# SECTION 4 -- CNN Training
# ============================================================

def build_improved_cnn(input_shape, num_filters_1=64, num_filters_2=128,
                        dropout_rate=0.4, dense_units=64):
    """
    Improved 1D CNN architecture:
      Input -> Conv1D(64) -> BatchNorm -> ReLU -> MaxPool
            -> Conv1D(128) -> BatchNorm -> ReLU -> GlobalAvgPool
            -> Dense(64) -> Dropout(0.4) -> Dense(1, sigmoid)

    VIVA IMPROVEMENTS over baseline:
      - Second convolutional block (128 filters) captures higher-level abstractions
      - BatchNormalization stabilises training, allows higher learning rates
      - GlobalAvgPooling instead of Flatten -- reduces parameters, less overfitting
      - Dropout(0.4) prevents co-adaptation of hidden units (regularisation)
      - kernel_size=5 in first block captures longer temporal patterns

    Args:
        input_shape   : (TimeSteps, Channels) -- Keras Conv1D format
        num_filters_1 : filters in first conv block
        num_filters_2 : filters in second conv block
        dropout_rate  : dropout probability
        dense_units   : units in dense layer before output

    Returns:
        Compiled Keras Sequential model
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D, BatchNormalization, ReLU, MaxPooling1D,
        GlobalAveragePooling1D, Flatten, Dense, Dropout, Input
    )
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=input_shape),

        # -- Block 1: Broad pattern detection ---------------------
        Conv1D(filters=num_filters_1, kernel_size=5, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling1D(pool_size=2),

        # -- Block 2: High-level feature abstraction ---------------
        Conv1D(filters=num_filters_2, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        GlobalAveragePooling1D(),   # No Flatten -- prevents param explosion

        # -- Classifier head ---------------------------------------
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid'),
    ], name='improved_cnn')

    model.compile(
        optimizer = Adam(learning_rate=1e-3),
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy'],
    )
    return model


def train_cnn(data):
    """
    Train the improved CNN with class weighting, EarlyStopping, and learning-rate decay.

    VIVA: Why class_weight in Keras_
      Setting class_weight={0: w0, 1: w1} scales the loss contribution of each sample
      by its class weight. This forces the model to learn harder from the minority
      (seizure) class, directly increasing recall without changing the data.

    VIVA: Why EarlyStopping_
      Training until validation loss stops improving prevents overfitting. The
      restore_best_weights option ensures the saved model has the BEST validation
      performance, not the last epoch (which may be overfit).
    """
    import matplotlib
    matplotlib.use('Agg')    # Non-interactive backend for server/script usage
    import matplotlib.pyplot as plt

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    print("\n" + "="*60)
    print("TRAINING: Improved 1D CNN (class_weight + EarlyStopping)")
    print("="*60)

    # -- Prepare tensors ------------------------------------------
    print("\n[CNN] Preparing CNN tensors...")
    X_train_cnn = prepare_data_for_cnn(data['X_train_norm'])
    X_val_cnn   = prepare_data_for_cnn(data['X_val_norm'])
    y_train     = data['y_train']
    y_val       = data['y_val']

    input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2])
    print(f"[CNN] Input shape: {input_shape}")

    # -- Class weights --------------------------------------------
    class_weights = get_class_weights(y_train)

    # -- Build model ----------------------------------------------
    model = build_improved_cnn(input_shape)
    model.summary()

    # -- Callbacks ------------------------------------------------
    cnn_ckpt_path = os.path.join(MODELS_DIR, 'cnn_best_checkpoint.h5')
    callbacks = [
        EarlyStopping(
            monitor              = 'val_loss',
            patience             = 7,        # stop if no improvement for 7 epochs
            restore_best_weights = True,     # use best weights, not last
            verbose              = 1,
        ),
        ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,          # halve LR when plateau detected
            patience = 3,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
        ModelCheckpoint(
            filepath           = cnn_ckpt_path,
            monitor            = 'val_loss',
            save_best_only     = True,
            verbose            = 1,
        ),
    ]

    # -- Training -------------------------------------------------
    print("\n[CNN] Training... (max 30 epochs, EarlyStopping active)")
    history = model.fit(
        X_train_cnn, y_train,
        epochs           = 30,
        batch_size       = 64,
        validation_data  = (X_val_cnn, y_val),
        class_weight     = class_weights,
        callbacks        = callbacks,
        verbose          = 1,
    )

    # -- Save final model -----------------------------------------
    cnn_path = os.path.join(MODELS_DIR, 'cnn_model.h5')
    model.save(cnn_path)
    print(f"[CNN] Model saved -> {cnn_path}")

    # -- Save training history for plot_results.py ----------------
    history_path = os.path.join(OUTPUTS_DIR, 'cnn_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=4)
    print(f"[CNN] History saved -> {history_path}")

    # -- Save training curves immediately -------------------------
    epochs_ran = range(1, len(history.history['accuracy']) + 1)

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_ran, history.history['accuracy'],     label='Train Accuracy', color='#3498DB', lw=2)
    plt.plot(epochs_ran, history.history['val_accuracy'], label='Val Accuracy',   color='#E74C3C', lw=2, linestyle='--')
    plt.title('CNN Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch',    fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cnn_accuracy_vs_epoch.png'), dpi=150)
    plt.close()

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_ran, history.history['loss'],     label='Train Loss', color='#3498DB', lw=2)
    plt.plot(epochs_ran, history.history['val_loss'], label='Val Loss',   color='#E74C3C', lw=2, linestyle='--')
    plt.title('CNN Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss',  fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cnn_loss_vs_epoch.png'), dpi=150)
    plt.close()

    print("[CNN] Training curves saved to outputs/plots/")

    # -- Quick validation metrics snapshot ------------------------
    from sklearn.metrics import recall_score, f1_score
    y_val_prob = model.predict(X_val_cnn).ravel()
    y_val_pred = (y_val_prob > 0.5).astype(int)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1     = f1_score(y_val, y_val_pred)
    print(f"[CNN] Val Recall={val_recall:.4f}  Val F1={val_f1:.4f}")

    return model


# ============================================================
# SECTION 5 -- Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train seizure prediction models')
    parser.add_argument('--skip-rf',  action='store_true', help='Skip Random Forest training')
    parser.add_argument('--skip-xgb', action='store_true', help='Skip XGBoost training')
    parser.add_argument('--skip-cnn', action='store_true', help='Skip CNN training')
    args, _ = parser.parse_known_args()

    print("\n" + "="*60)
    print("  SEIZURE PREDICTION -- UNIFIED MODEL TRAINING PIPELINE")
    print("="*60)

    # Load and normalise all splits
    data = load_all_splits()

    # -- Train models ----------------------------------------------
    if not args.skip_rf:
        train_random_forest(data)
    else:
        print("\n[SKIP] Random Forest skipped (--skip-rf flag)")

    if not args.skip_xgb:
        train_xgboost(data)
    else:
        print("\n[SKIP] XGBoost skipped (--skip-xgb flag)")

    if not args.skip_cnn:
        train_cnn(data)
    else:
        print("\n[SKIP] CNN skipped (--skip-cnn flag)")

    print("\n" + "="*60)
    print("  ALL MODELS TRAINED. Run evaluate_models.py next.")
    print("="*60)


if __name__ == "__main__":
    main()
