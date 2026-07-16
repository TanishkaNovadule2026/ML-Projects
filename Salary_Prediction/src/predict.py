# Predict salary of data which come in streamlit ui 
"""Input forms are : Experience, Education, Location, Jobrole, age, skill_counts 
Here Education, Location, Jobrole are : categorical data so when we give this test data so first need to apply encoding 
-> Encoding 
-> fit as test data 
-> result : salary -> put it on UI 

Use only that model which r2 score is greater then other model 

"""






"""Prediction helpers for the salary model."""
"""import os
import joblib
import pandas as pd

from src.train import X_train

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "model", "best_model.pkl"),
    os.path.join(PROJECT_ROOT, "model", "artifacts", "Ridge.joblib"),
    os.path.join(PROJECT_ROOT, "model", "artifacts", "LinearRegression.joblib"),
    os.path.join(PROJECT_ROOT, "model", "artifacts", "Lasso.joblib"),
]

model = None
for candidate in MODEL_CANDIDATES:
    if os.path.exists(candidate):
        try:
            model = joblib.load(candidate)
            break
        except Exception:
            model = None


# -------------------------------
# Build Input
# -------------------------------

def _build_input(experience, education, job_role, location, skills_count):
    #Build one-row dataframe exactly like X_train
    input_df = pd.DataFrame(
        {
            "Experience": [experience],
            "Education": [education],
            "JobRole": [job_role],
            "Location": [location],
            "SkillsCount": [skills_count],
        }
    )

    input_df = pd.get_dummies(
        input_df,
        columns=["Education", "JobRole", "Location"],
        drop_first=True,
    )
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
    return input_df


# -------------------------------
# Fallback Prediction
# -------------------------------

def fallback_predict(experience, education, job_role, location, skills_count):
    #Used if no trained model artifact is available.
    salary = 300000 + experience * 50000 + skills_count * 10000
    return round(max(salary, 0), 2), "Fallback Model"


# -------------------------------
# Predict Salary
# -------------------------------

def predict_salary(experience, education, job_role, location, skills_count):
    X_input = _build_input(experience, education, job_role, location, skills_count)

    if model is not None:
        try:
            prediction = max(float(model.predict(X_input)[0]), 0)
            return prediction, "Best Model"
        except Exception:
            pass

    return fallback_predict(experience, education, job_role, location, skills_count)"""