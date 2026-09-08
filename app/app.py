from flask import Flask, render_template, request
import joblib
import pandas as pd
import os


# ==========================================
# FLASK APP
# ==========================================

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)


# ==========================================
# LOAD TRAINED ML MODEL
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "student_performance_model.pkl"
)

model = joblib.load(MODEL_PATH)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from form
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_marks = float(request.form["previous_marks"])
        assignments_score = float(request.form["assignments_score"])
        sleep_hours = float(request.form["sleep_hours"])

        # Create DataFrame
        student_data = pd.DataFrame({
            "study_hours": [study_hours],
            "attendance": [attendance],
            "previous_marks": [previous_marks],
            "assignments_score": [assignments_score],
            "sleep_hours": [sleep_hours]
        })

        # Make prediction
        prediction = model.predict(student_data)[0]

        # Keep prediction between 0 and 100
        prediction = max(0, min(100, prediction))

        prediction = round(prediction, 2)

        # Performance category
        if prediction >= 90:
            performance = "Excellent"
        elif prediction >= 75:
            performance = "Very Good"
        elif prediction >= 60:
            performance = "Good"
        elif prediction >= 40:
            performance = "Average"
        else:
            performance = "Needs Improvement"

        return render_template(
            "result.html",
            prediction=prediction,
            performance=performance,
            study_hours=study_hours,
            attendance=attendance,
            previous_marks=previous_marks,
            assignments_score=assignments_score,
            sleep_hours=sleep_hours
        )

    except Exception as e:

        return render_template(
            "index.html",
            error="Please enter valid values."
        )


# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)