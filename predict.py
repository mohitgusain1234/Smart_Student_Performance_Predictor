import pandas as pd
import joblib
import os


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = os.path.join(
    "model",
    "student_performance_model.pkl"
)

model = joblib.load(MODEL_PATH)

print("ML model loaded successfully!")


# ==========================================
# 2. PREDICTION FUNCTION
# ==========================================

def predict_student_performance(
    study_hours,
    attendance,
    previous_marks,
    assignments_score,
    sleep_hours
):
    """
    Predict the final score of a student.
    """

    # Create input DataFrame
    student_data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "previous_marks": [previous_marks],
        "assignments_score": [assignments_score],
        "sleep_hours": [sleep_hours]
    })

    # Make prediction
    prediction = model.predict(student_data)[0]

    # Keep score between 0 and 100
    prediction = max(0, min(100, prediction))

    return round(prediction, 2)


# ==========================================
# 3. TEST PREDICTION
# ==========================================

if __name__ == "__main__":

    print("\n========================================")
    print("   STUDENT PERFORMANCE PREDICTOR")
    print("========================================")

    # Sample student
    study_hours = 6
    attendance = 85
    previous_marks = 78
    assignments_score = 80
    sleep_hours = 8

    predicted_score = predict_student_performance(
        study_hours,
        attendance,
        previous_marks,
        assignments_score,
        sleep_hours
    )

    print("\nStudent Details:")
    print("Study Hours:", study_hours)
    print("Attendance:", attendance)
    print("Previous Marks:", previous_marks)
    print("Assignments Score:", assignments_score)
    print("Sleep Hours:", sleep_hours)

    print("\n----------------------------------------")
    print("Predicted Final Score:", predicted_score)
    print("----------------------------------------")

    # Performance category
    if predicted_score >= 90:
        performance = "Excellent"
    elif predicted_score >= 75:
        performance = "Very Good"
    elif predicted_score >= 60:
        performance = "Good"
    elif predicted_score >= 40:
        performance = "Average"
    else:
        performance = "Needs Improvement"

    print("Performance Level:", performance)

    print("========================================")