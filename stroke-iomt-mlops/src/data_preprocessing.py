"""
data_preprocessing.py
=====================
Advanced EEG data preprocessing module for seizure/stroke prediction.

KEY IMPROVEMENTS over baseline:
  - Rich feature engineering: time-domain + frequency-domain (FFT band power) + wavelet (DWT)
  - Dedicated test-set loader (eeg-seizure_test.npz) to enforce strict train/val/test separation
  - SMOTE helper for class imbalance oversampling on RF/XGBoost feature sets
  - Class-weight calculator for CNN training
  - All normalisation fitted ONLY on training data to prevent data leakage

VIVA NOTE:
  Why so many features_ EEG seizure signatures manifest across multiple frequency bands
  (delta 0-4 Hz, theta 4-8 Hz, alpha 8-13 Hz, beta 13-30 Hz, gamma >30 Hz). Capturing
  energy in each band gives the classical ML models (RF, XGBoost) far richer signal than
  raw statistical moments alone, substantially boosting their recall.
"""

import numpy as np
import os

# ---------------------------------------------------------------------------
# Attempt optional heavy imports - gracefully degraded if not installed yet.
# ---------------------------------------------------------------------------
try:
    import pywt                          # PyWavelets - wavelet transform features
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("[WARNING] PyWavelets not installed. Wavelet features will be skipped. "
          "Run: pip install PyWavelets")

try:
    from imblearn.over_sampling import SMOTE  # imbalanced-learn
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARNING] imbalanced-learn not installed. SMOTE will be skipped. "
          "Run: pip install imbalanced-learn")

try:
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_WEIGHTS_AVAILABLE = True
except ImportError:
    SKLEARN_WEIGHTS_AVAILABLE = False

# ============================================================
# SECTION 1 - Data Loaders
# ============================================================

def _resolve_paths():
    """Resolve absolute project root from wherever this script is called."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    base_path = os.path.join(project_root, 'data', 'raw')
    return project_root, base_path


def load_data(mode="train"):
    """
    Load EEG data from NPZ files.

    Args:
        mode (str): 'train' or 'val'

    Returns:
        signals (np.ndarray): Shape (Samples, Channels, TimeSteps)
        labels  (np.ndarray): Binary labels  (0 = no-seizure, 1 = seizure)
    """
    _, base_path = _resolve_paths()

    if mode == "train":
        path = os.path.join(base_path, "eeg-seizure_train.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}. Please check path.")
        data = np.load(path)
        return data['train_signals'], data['train_labels']

    elif mode == "val":
        path = os.path.join(base_path, "eeg-seizure_val.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Validation dataset not found at {path}.")
        data = np.load(path)
        return data['val_signals'], data['val_labels']

    else:
        raise ValueError("Mode must be 'train' or 'val'")


def load_test_data():
    """
    Load the holdout evaluation set for final model assessment.

    NOTE: eeg-seizure_test.npz contains ONLY signals (no labels) -- it is an
    inference-only set. We therefore use eeg-seizure_val.npz as the final
    labelled evaluation holdout. The val set is kept strictly separate from
    training data throughout the pipeline (no SMOTE, no fitting).

    Returns:
        signals (np.ndarray): Shape (Samples, Channels, TimeSteps)
        labels  (np.ndarray): Binary labels
    """
    _, base_path = _resolve_paths()

    # Primary: val set (has labels, used as final evaluation holdout)
    val_path = os.path.join(base_path, "eeg-seizure_val.npz")
    if os.path.exists(val_path):
        data = np.load(val_path)
        if 'val_signals' in data and 'val_labels' in data:
            print("[INFO] load_test_data: using eeg-seizure_val.npz as labelled test set.")
            return data['val_signals'], data['val_labels']

    # Fallback: balanced val set
    bal_path = os.path.join(base_path, "eeg-seizure_val_balanced.npz")
    if os.path.exists(bal_path):
        data = np.load(bal_path)
        print("[INFO] load_test_data: using eeg-seizure_val_balanced.npz as test set.")
        return data['val_signals'], data['val_labels']

    raise FileNotFoundError("No labelled evaluation set found. Expected eeg-seizure_val.npz.")


# ============================================================
# SECTION 2 - Normalisation
# ============================================================

def normalize_signals(signals, fitted_stats=None):
    """
    Z-score normalise signals along the time axis (axis=2), per channel globally.

    When computing stats from scratch, we average across both the sample axis (0)
    and time axis (2) to get one global mean/std per channel -- shape (1, C, 1).
    This ensures the stats can be applied (broadcast) to datasets of ANY size.

    VIVA NOTE: Z-score normalisation removes DC offset and amplitude scale differences
    between patients/electrodes. Crucial because EEG amplitude varies widely across
    electrode placements and individuals.

    Args:
        signals      (np.ndarray): Shape (Samples, Channels, TimeSteps)
        fitted_stats (tuple|None): (mean, std) previously computed from training set --
                                   shape (1, C, 1). Pass these when normalising val/test
                                   to prevent data leakage.

    Returns:
        normalized (np.ndarray): Normalised signals, same shape as input
        stats      (tuple):      (mean, std) computed -- shape (1, C, 1)
    """
    if fitted_stats is not None:
        mean, std = fitted_stats
    else:
        # Compute global stats per channel: average over samples AND time
        # keepdims=True on both axes -> shape (1, C, 1) for broadcasting
        mean = np.mean(signals, axis=(0, 2), keepdims=True)   # (1, Channels, 1)
        std  = np.std(signals,  axis=(0, 2), keepdims=True)   # (1, Channels, 1)

    std = std.copy()
    std[std == 0] = 1e-8    # Prevent division by zero

    normalized = (signals - mean) / std
    return normalized, (mean, std)


# ============================================================
# SECTION 3 - Feature Engineering (Rich)
# ============================================================

def _band_power(signal, fs=256.0):
    """
    Compute EEG frequency band power using FFT.

    Bands (Hz):  delta  0-4 | theta 4-8 | alpha 8-13 | beta 13-30 | gamma 30-100

    VIVA: Seizures are characterised by high-amplitude, synchronised discharges that
    show up as power spikes in specific frequency bands (especially gamma and beta).

    Args:
        signal (np.ndarray): 1D time-series for one channel
        fs     (float):      Sampling frequency (CHB-MIT default = 256 Hz)

    Returns:
        np.ndarray: 5-element array [delta_pwr, theta_pwr, alpha_pwr, beta_pwr, gamma_pwr]
    """
    n = len(signal)
    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_sq = np.abs(np.fft.rfft(signal)) ** 2  # Power spectrum

    bands = {
        'delta': (0.5, 4),
        'theta': (4,   8),
        'alpha': (8,  13),
        'beta':  (13, 30),
        'gamma': (30, 100),
    }

    powers = []
    for band_name, (lo, hi) in bands.items():
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        powers.append(np.sum(fft_sq[idx]) if len(idx) > 0 else 0.0)

    return np.array(powers)


def _wavelet_energy(signal, wavelet='db4', level=4):
    """
    Compute Discrete Wavelet Transform (DWT) coefficient energy per decomposition level.

    VIVA: DWT captures non-stationary EEG dynamics at multiple time-frequency
    resolutions simultaneously -- something FFT cannot do. Epileptic discharge patterns
    have distinctive wavelet signatures at specific decomposition levels.

    Args:
        signal  (np.ndarray): 1D time-series
        wavelet (str):        Wavelet type ('db4' is standard for EEG)
        level   (int):        Decomposition levels

    Returns:
        np.ndarray: Energy values per decomposition level
    """
    if not PYWT_AVAILABLE:
        return np.zeros(level + 1)

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    energy = np.array([np.sum(c ** 2) for c in coeffs])
    return energy


def _zero_crossing_rate(signal):
    """Count zero-crossings normalised by signal length (proxy for frequency content)."""
    zcr = np.sum(np.diff(np.sign(signal)) != 0)
    return zcr / len(signal)


def extract_features_for_rf(signals, fs=256.0, include_wavelet=True):
    """
    Extract a rich tabular feature matrix from 3D EEG signals.

    Features per channel:
      Time-domain (8):   mean, variance, min, max, skewness, kurtosis, p2p, ZCR
      Frequency (5):     delta, theta, alpha, beta, gamma band power
      Wavelet (5):       DWT energy levels 0-4  (if PyWavelets available)

    Total = 18 features per channel _ num_channels

    VIVA: This is a substantial upgrade from the 4-feature baseline (mean/var/min/max).
    The richer feature set allows the Random Forest to discriminate seizure EEG patterns
    that show up primarily in frequency/wavelet space, boosting recall significantly.

    Args:
        signals        (np.ndarray): Shape (Samples, Channels, TimeSteps)
        fs             (float):      Sampling frequency
        include_wavelet(bool):       Adds wavelet features if PyWavelets is present

    Returns:
        np.ndarray: 2D feature matrix  (Samples, Channels _ Features_per_channel)
    """
    print("Extracting rich EEG features for RF/XGBoost...")
    n_samples, n_channels, n_timesteps = signals.shape
    all_features = []

    for i in range(n_samples):
        if i % 500 == 0:
            print(f"  Processing sample {i}/{n_samples}...")

        sample_feats = []
        for c in range(n_channels):
            sig = signals[i, c, :]

            # --- Time-domain features ---
            mean_v  = np.mean(sig)
            var_v   = np.var(sig)
            min_v   = np.min(sig)
            max_v   = np.max(sig)
            p2p     = max_v - min_v                                      # peak-to-peak
            zcr     = _zero_crossing_rate(sig)
            skew_v  = float(np.mean(((sig - mean_v) / (np.std(sig) + 1e-8)) ** 3))
            kurt_v  = float(np.mean(((sig - mean_v) / (np.std(sig) + 1e-8)) ** 4))

            time_feats = [mean_v, var_v, min_v, max_v, p2p, zcr, skew_v, kurt_v]

            # --- Frequency-domain features (FFT band power) ---
            freq_feats = list(_band_power(sig, fs=fs))   # 5 values

            # --- Wavelet features ---
            if include_wavelet:
                wav_feats = list(_wavelet_energy(sig))   # 5 values
            else:
                wav_feats = []

            sample_feats.extend(time_feats + freq_feats + wav_feats)

        all_features.append(sample_feats)

    features = np.array(all_features, dtype=np.float32)
    # Replace any NaN/Inf that might arise from edge channels
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    print(f"Feature extraction complete. Shape: {features.shape}")
    return features


# ============================================================
# SECTION 4 - CNN Preparation
# ============================================================

def prepare_data_for_cnn(signals):
    """
    Reshape signals for Keras Conv1D: (Samples, Channels, TimeSteps) -> (Samples, TimeSteps, Channels).

    VIVA: Keras Conv1D expects the sequence dimension first then features/channels last.
    This transpose is mandatory for correct convolution along the time axis.

    Args:
        signals (np.ndarray): Shape (Samples, Channels, TimeSteps)

    Returns:
        np.ndarray: Shape (Samples, TimeSteps, Channels)
    """
    return np.transpose(signals, (0, 2, 1))


# ============================================================
# SECTION 5 - Class Imbalance Utilities
# ============================================================

def get_class_weights(labels):
    """
    Compute sklearn-style class weights to pass as `class_weight` to Keras .fit().

    VIVA: EEG datasets are heavily imbalanced -- seizure segments form < 5% of recordings.
    Weighting the loss inversely proportional to class frequency forces the model to
    penalise False Negatives (missed seizures) more heavily, boosting recall.

    Args:
        labels (np.ndarray): 1D binary label array

    Returns:
        dict: {0: weight_for_majority, 1: weight_for_minority}
    """
    if not SKLEARN_WEIGHTS_AVAILABLE:
        return {0: 1.0, 1: 1.0}

    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"[INFO] Class weights: {weight_dict}")
    return weight_dict


def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE oversampling to balance the training set for RF/XGBoost.

    VIVA: SMOTE (Synthetic Minority Oversampling TEchnique) generates synthetic
    seizure samples by interpolating between existing minority-class feature vectors.
    This is preferable to simple duplication because it adds diversity, reducing
    classifier bias toward the majority (no-seizure) class.

    IMPORTANT: SMOTE must ONLY be applied to the training set -- never to val/test.
    Applying it to test would inflate recall artificially (data leakage).

    Args:
        X            (np.ndarray): Feature matrix (Samples, Features)
        y            (np.ndarray): Labels
        random_state (int):        Reproducibility seed

    Returns:
        X_resampled (np.ndarray): Balanced feature matrix
        y_resampled (np.ndarray): Balanced labels
    """
    if not SMOTE_AVAILABLE:
        print("[WARNING] SMOTE not available. Returning original imbalanced data.")
        return X, y

    minority_count = int(np.sum(y == 1))
    majority_count = int(np.sum(y == 0))
    print(f"[SMOTE] Before: majority={majority_count}, minority={minority_count}")

    # Only apply SMOTE if there's meaningful imbalance
    if minority_count < 6:
        print("[SMOTE] Too few minority samples. Skipping SMOTE.")
        return X, y

    # k_neighbors must be < minority count
    k_neighbors = min(5, minority_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[SMOTE] After:  total={len(y_res)}, minority={int(np.sum(y_res==1))}")
    return X_res, y_res


# ============================================================
# SECTION 6 - Self-test
# ============================================================

if __name__ == "__main__":
    print("=== data_preprocessing.py self-test ===")
    signals, labels = load_data("train")
    print(f"Train: signals={signals.shape}, labels={labels.shape}")
    print(f"  Seizure %: {100 * labels.mean():.2f}%")

    norm_signals, stats = normalize_signals(signals)
    rf_features = extract_features_for_rf(norm_signals[:50])   # small subset for speed
    cnn_input   = prepare_data_for_cnn(norm_signals[:50])

    print(f"RF Feature Shape:  {rf_features.shape}")
    print(f"CNN Input Shape:   {cnn_input.shape}")
    print(f"Class Weights:     {get_class_weights(labels)}")
    print("Self-test PASSED.")