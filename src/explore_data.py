from pathlib import Path
import pandas as pd

# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Locate the dataset
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"

print("Dataset path:", DATA_PATH)
print("File exists:", DATA_PATH.exists())

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print(df.head())
# Check zero values in medical measurements
columns_to_check = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

print("\nZero values in selected columns:")
print((df[columns_to_check] == 0).sum())