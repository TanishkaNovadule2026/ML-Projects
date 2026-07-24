from src.preprocess import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# ============================
# Prepare Data
# ============================

X = encoded_df.drop("loan_status", axis=1)
y = encoded_df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# Train Random Forest Model
# ============================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

# ============================
# Prediction on Test Set
# ============================

# Predict Class
y_pred = rf.predict(X_test)

# Predict Probability
probability = rf.predict_proba(X_test)

# Probability of Default (loan_status = 1)
positive_class_idx = list(rf.classes_).index(1)

risk_prob = probability[:, positive_class_idx]

# Convert to Risk Score (%)
risk_score = risk_prob * 100

# ============================
# Risk Level
# ============================

conditions = [
    risk_score < 30,
    (risk_score >= 30) & (risk_score < 70),
    risk_score >= 70
]

choices = [
    "Low Risk",
    "Medium Risk",
    "High Risk"
]

risk_level = np.select(
    conditions,
    choices,
    default="Unknown"
)

# Final Result (Test Set)

result_df = X_test.copy()

"""result_df["Actual Loan Status"] = y_test.values
result_df["Predicted Loan Status"] = y_pred
result_df["Risk Probability"] = risk_prob.round(4)
result_df["Risk Score"] = risk_score.round(2)
result_df["Risk Level"] = risk_level"""

print(result_df.head(10))

print("\nExample Prediction (Test Set)")
print("------------------")
print("Risk Probability :", risk_prob[120])
print("Risk Score       :", f"{risk_score[12]:.2f}%")
print("Risk Level       :", risk_level[120])
print(risk_score.max())


# Predict for a New / Unseen Applicant


input_df = pd.DataFrame({
    "person_age": [55],
    "person_income": [6500],
    "person_home_ownership": ["RENT"],
    "person_emp_length": [1],
    "loan_intent": ["EDUCATION"],
    "loan_grade": ["G"],
    "loan_amnt": [1200000],
    "loan_int_rate": [12.5],
    "loan_percent_income": [0.18],
    "cb_person_default_on_file": ["N"],
    "cb_person_cred_hist_length": [6]
})

# Apply the SAME encoding used in preprocess.py — must match exactly

ownership_map = {
    "RENT": 0,
    "MORTGAGE": 1,
    "OWN": 2,
    "OTHER": 3
}

grade_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6
}

default_map = {
    "N": 0,
    "Y": 1
}

input_encoded = input_df.copy()

# Ordinal / label encoding (same as training)
input_encoded["person_home_ownership"] = input_encoded["person_home_ownership"].map(ownership_map)
input_encoded["loan_grade"] = input_encoded["loan_grade"].map(grade_map)
input_encoded["cb_person_default_on_file"] = input_encoded["cb_person_default_on_file"].map(default_map)

# One-hot encoding for loan_intent (same as training)
input_encoded = pd.get_dummies(
    input_encoded,
    columns=["loan_intent"],
    drop_first=True,
    dtype=int
)

# Align columns exactly with training data — add any missing one-hot
# columns as 0, drop any extras, and keep the same column order
input_encoded = input_encoded.reindex(columns=X_train.columns, fill_value=0)

# Predict
new_prediction = rf.predict(input_encoded)
new_probability = rf.predict_proba(input_encoded)

new_risk_prob = new_probability[0][positive_class_idx]
new_risk_score = new_risk_prob * 100

if new_risk_score < 30:
    new_risk_level = "Low Risk"
elif new_risk_score < 70:
    new_risk_level = "Medium Risk"
else:
    new_risk_level = "High Risk"

print("\nNew Applicant Prediction")
print("Prediction   :", new_prediction[0])
print("Risk Score   :", f"{new_risk_score:.2f}%")
print("Risk Level   :", new_risk_level)

input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X_train.columns, fill_value=0)

prediction = rf.predict(input_encoded)
probability = rf.predict_proba(input_encoded)

