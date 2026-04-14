import numpy as np
import os

def load_data(mode="train"):
    """
    Loads the EEG data from NPZ files.
    
    Args:
        mode (str): 'train' or 'val' to load respective files.
        
    Returns:
        signals (np.ndarray): EEG signals of shape (Samples, Channels, TimeSteps)
        labels (np.ndarray): Binary labels indicating seizure vs no seizure
    """
    # Use robust absolute path resolution regardless of from where this is executed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    base_path = os.path.join(project_root, 'data', 'raw')
    
    if mode == "train":
        path = os.path.join(base_path, "eeg-seizure_train.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}. Please check path.")
        data = np.load(path)
        return data['train_signals'], data['train_labels']
    elif mode == "val":
        path = os.path.join(base_path, "eeg-seizure_val.npz")
        data = np.load(path)
        return data['val_signals'], data['val_labels']
    else:
        raise ValueError("Mode must be 'train' or 'val'")

def normalize_signals(signals):
    """
    Normalizes the signals using Z-score normalization per sample and channel.
    Provides basic noise resistance and standardized amplitude.
    
    Args:
        signals (np.ndarray): Shape (Samples, Channels, TimeSteps)
        
    Returns:
        np.ndarray: Normalized signals
    """
    # Normalize along the time axis (axis 2)
    mean = np.mean(signals, axis=2, keepdims=True)
    std = np.std(signals, axis=2, keepdims=True)
    
    # Add small epsilon to prevent division by zero
    std[std == 0] = 1e-8
    
    normalized = (signals - mean) / std
    return normalized

def extract_features_for_rf(signals):
    """
    Extracts tabular features from 3D signals for the Random Forest model.
    Extracts mean, variance, min, max, and sum of absolute differences.
    
    Args:
        signals (np.ndarray): Shape (Samples, Channels, TimeSteps)
        
    Returns:
        np.ndarray: 2D feature matrix of shape (Samples, Channels * Num_Features)
    """
    print("Extracting statistical features for Random Forest...")
    # Calculate statistical features across time dimension (axis=2)
    mean_feat = np.mean(signals, axis=2)
    var_feat = np.var(signals, axis=2)
    min_feat = np.min(signals, axis=2)
    max_feat = np.max(signals, axis=2)
    
    # Concatenate features: Shape becomes (Samples, Channels * 4)
    features = np.concatenate([mean_feat, var_feat, min_feat, max_feat], axis=1)
    print(f"Feature extraction complete. Shape: {features.shape}")
    return features

def prepare_data_for_cnn(signals):
    """
    Reshapes the signal data for Keras Conv1D CNN.
    Keras expects shape: (Batch, TimeSteps, Channels)
    
    Args:
        signals (np.ndarray): Shape (Samples, Channels, TimeSteps)
        
    Returns:
        np.ndarray: Shape (Samples, TimeSteps, Channels)
    """
    # Transposing signals from (Samples, Channels, TimeSteps) to (Samples, TimeSteps, Channels)
    return np.transpose(signals, (0, 2, 1))

if __name__ == "__main__":
    # Small test
    signals, labels = load_data("train")
    print(f"Loaded Train Data: Signals {signals.shape}, Labels {labels.shape}")
    
    norm_signals = normalize_signals(signals)
    rf_features = extract_features_for_rf(norm_signals)
    cnn_input = prepare_data_for_cnn(norm_signals)
    
    print(f"Random Forest Input Shape: {rf_features.shape}")
    print(f"CNN Input Shape: {cnn_input.shape}")