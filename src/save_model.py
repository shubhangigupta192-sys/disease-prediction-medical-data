from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_model.joblib"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop(columns="Outcome").copy()
y = df["Outcome"]

# Replace invalid zero measurements
invalid_zero_columns = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]
X[invalid_zero_columns] = X[invalid_zero_columns].replace(0, float("nan"))

# Training split (same setup used for comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build selected model pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Save trained pipeline
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("Model saved successfully!")
print(f"Saved location: {MODEL_PATH}")