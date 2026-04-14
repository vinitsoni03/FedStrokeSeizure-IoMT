import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data, normalize_signals, extract_features_for_rf

def train_rf():
    """
    Trains a Random Forest classifier using extracted tabular features.
    """
    print("Loading data...")
    X_train_raw, y_train = load_data("train")
    
    print("Normalizing signals...")
    X_train_norm = normalize_signals(X_train_raw)
    
    print("Extracting features for Random Forest...")
    X_train_features = extract_features_for_rf(X_train_norm)
    
    print("Training Random Forest Classifier...")
    # Keep parameters simple but reasonable to prevent overfitting and make it explainable
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_features, y_train)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Save the model
    # Ensure models directory exists
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "random_forest.pkl")
    print(f"Saving model to {model_path}...")
    joblib.dump(rf, model_path)
    
    print("Training complete!")

if __name__ == "__main__":
    train_rf()
