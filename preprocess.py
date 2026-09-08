import pandas as pd
import os

# Dataset path
DATA_PATH = os.path.join("data", "students.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Make sure all columns contain numeric values
numeric_columns = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignments_score",
    "sleep_hours",
    "final_score"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows containing missing values
df = df.dropna()

# Input features
X = df[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignments_score",
        "sleep_hours"
    ]
]

# Target variable
y = df["final_score"]

print("\nAfter preprocessing:")
print("Number of rows:", len(df))
print("Number of features:", X.shape[1])

print("\nInput features:")
print(X.head())

print("\nTarget values:")
print(y.head())

print("\nPreprocessing completed successfully!")