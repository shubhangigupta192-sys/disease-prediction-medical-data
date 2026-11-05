from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns="Outcome").copy()
y = df["Outcome"]

# Replace invalid zero measurements
invalid_zero_columns = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]
X[invalid_zero_columns] = X[invalid_zero_columns].replace(0, float("nan"))

# Split data
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

results = []
rf_predictions = None

# Train and evaluate
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

    if name == "Random Forest":
        rf_predictions = y_pred

# Model comparison chart
results_df = pd.DataFrame(results).set_index("Model")
results_df.plot(kind="bar", figsize=(10, 6), ylim=(0, 1))

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Machine Learning Model")
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()

comparison_path = OUTPUT_DIR / "model_comparison.png"
plt.savefig(comparison_path, dpi=300)
plt.show()

# Random Forest confusion matrix
cm = confusion_matrix(y_test, rf_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Class 0", "Class 1"],
    yticklabels=["Class 0", "Class 1"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

confusion_path = OUTPUT_DIR / "random_forest_confusion_matrix.png"
plt.savefig(confusion_path, dpi=300)
plt.show()

print("\nModel comparison results:")
print(results_df.round(4))

print(f"\nSaved comparison chart: {comparison_path}")
print(f"Saved confusion matrix: {confusion_path}")