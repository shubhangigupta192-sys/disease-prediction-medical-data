from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop(columns="Outcome").copy()
y = df["Outcome"]

# Replace invalid zero measurements with missing values
invalid_zero_columns = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]
X[invalid_zero_columns] = X[invalid_zero_columns].replace(0, float("nan"))

# Use the same split for all models
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Train and evaluate each model
results = []

for name, classifier in models.items():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0)
    })

# Display comparison
results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

