import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data/raw/stroke_data.csv")

# Drop id column
df = df.drop(columns=["id"])

# Fill missing BMI
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

# Fill any other missing values
df = df.fillna(df.mean(numeric_only=True))

# Encode categorical columns
categorical = ["gender","ever_married","work_type","Residence_type","smoking_status"]

le = LabelEncoder()

for col in categorical:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Split features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle class imbalance
smote = SMOTE()

X_train, y_train = smote.fit_resample(X_train, y_train)

# Save processed data
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Data preprocessing completed")