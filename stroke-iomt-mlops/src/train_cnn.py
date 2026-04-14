import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, ReLU, MaxPooling1D, Dense, Flatten, Input
from data_preprocessing import load_data, normalize_signals, prepare_data_for_cnn

def build_cnn(input_shape):
    """
    Builds a simple CNN for time-series extraction.
    Architecture: Conv1D -> ReLU -> Pooling -> Dense -> Sigmoid
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=3),
        ReLU(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_cnn():
    """
    Trains a simple CNN on the EEG time-series data and plost metrics.
    """
    print("Loading data...")
    X_train_raw, y_train = load_data("train")
    
    print("Normalizing signals...")
    X_train_norm = normalize_signals(X_train_raw)
    
    print("Preparing features for CNN...")
    X_train_cnn = prepare_data_for_cnn(X_train_norm)
    input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2]) # (TimeSteps, Channels)
    
    print(f"Building CNN model with input shape {input_shape}...")
    model = build_cnn(input_shape)
    model.summary()
    
    print("Training CNN...")
    # Keep epochs low for fast execution, assuming simple dataset.
    history = model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.1)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Save the model
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "cnn_model.h5")
    print(f"Saving CNN model to {model_path}...")
    model.save(model_path)
    
    # Generate and save plots
    print("Generating training plots for viva defense...")
    plots_dir = os.path.join(project_root, 'outputs', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('CNN Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'cnn_accuracy_vs_epoch.png'))
    plt.close()
    
    # Plot Loss 
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('CNN Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'cnn_loss_vs_epoch.png'))
    plt.close()
    
    print("Training complete! Plots saved in outputs/plots/")

if __name__ == "__main__":
    train_cnn()
