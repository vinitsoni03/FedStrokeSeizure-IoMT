import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Train model
model = xgb.XGBClassifier()

model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

# Save model
joblib.dump(model, "models/stroke_model.pkl")

print("Model saved")