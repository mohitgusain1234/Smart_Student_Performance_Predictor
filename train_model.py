import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = os.path.join("data", "students.csv")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Total records:", len(df))


# ==========================================
# 2. SELECT FEATURES AND TARGET
# ==========================================

features = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignments_score",
    "sleep_hours"
]

target = "final_score"

X = df[features]
y = df[target]


# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 4. CREATE ML MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining the Machine Learning model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

print("=======================================")


# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

MODEL_PATH = os.path.join("model", "student_performance_model.pkl")

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)


# ==========================================
# 9. TEST WITH SAMPLE STUDENT
# ==========================================

sample_student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [85],
    "previous_marks": [78],
    "assignments_score": [80],
    "sleep_hours": [8]
})

sample_prediction = model.predict(sample_student)[0]

print("\n========== SAMPLE PREDICTION ==========")
print("Sample Student Details:")
print(sample_student)

print(f"\nPredicted Final Score: {sample_prediction:.2f}")

print("=======================================")