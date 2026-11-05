from pathlib import Path
import pandas as pd
import numpy as np

# Locate project and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Columns where zero represents an invalid/missing measurement
invalid_zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace invalid zeros with NaN
df[invalid_zero_columns] = df[invalid_zero_columns].replace(0, np.nan)

print("Missing values after replacing zeros:")
print(df.isnull().sum())

# Impute missing values using column medians
for column in invalid_zero_columns:
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)

print("\nMissing values after imputation:")
print(df.isnull().sum())

print("\nCleaned dataset:")
print(df.head())