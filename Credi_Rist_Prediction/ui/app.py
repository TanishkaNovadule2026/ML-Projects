import streamlit as st
import pandas as pd

# rf, X_train, positive_class_idx aate hain trained model file se
from Model.Random_Forest import rf, X_train, positive_class_idx

st.set_page_config(
    page_title="Credit Risk Prediction System",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
    background:white;
    color:black;
}

/* Page Title */
.Title{
    text-align:center;
    background:skyblue;
    padding:15px;
    border-radius:8px;
    margin-bottom:20px;
}

/* Input */
.Input_data{
    border:2px solid #ddd;
    border-radius:8px;
    overflow:hidden;
}

.input_heading{
    background:#0b5ea8;
    color:white;
    text-align:center;
    font-size:22px;
    font-weight:bold;
    padding:10px;
}

.section{
    border:1px solid #ddd;
    margin:10px;
    padding:10px;
    border-radius:6px;
}

/* Output */

.output_heading{
    background:#2ea043;
    color:white;
    text-align:center;
    font-size:22px;
    font-weight:bold;
    padding:10px;
    border-radius:8px;
    margin-bottom:15px;
}

.box{
    border:1px solid #ddd;
    border-radius:8px;
    padding:12px;
    margin-bottom:15px;
}

.box_heading{
    background:#2ea043;
    color:white;
    text-align:center;
    padding:8px;
    border-radius:5px;
    font-weight:bold;
    margin-bottom:10px;
}

.score{
    text-align:center;
    font-size:40px;
    font-weight:bold;
}

.scale-bar{
    position:relative;
    width:100%;
    height:12px;
    border-radius:20px;
    background:linear-gradient(to right,#4CAF50,#FFD54F,#E53935);
}

.scale-pointer{
    position:absolute;
    width:18px;
    height:18px;
    border-radius:50%;
    background:white;
    border:2px solid black;
    top:-3px;
    transform:translateX(-50%);
}

.scale-label{
    display:flex;
    justify-content:space-between;
    font-size:12px;
    margin-top:5px;
}

.scale-text{
    text-align:center;
    font-size:13px;
    margin-top:5px;
}

.risk{
    font-size:22px;
    text-align:center;
    font-weight:bold;
}

.risk-low{ color:#2ea043; }
.risk-medium{ color:#e6a700; }
.risk-high{ color:#e53935; }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="Title">
<h2>Credit Risk Prediction</h2>
<p>Input & Output</p>
</div>
""", unsafe_allow_html=True)

# ============================
# Encoding maps — EXACTLY same as preprocess.py training encoding
# ============================

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

loan_intent_options = [
    "EDUCATION",
    "MEDICAL",
    "PERSONAL",
    "VENTURE",
    "HOMEIMPROVEMENT",
    "DEBTCONSOLIDATION"
]

col1, col2 = st.columns(2)

# ============================
# LEFT COLUMN — Input Form
# ============================

with col1:
    st.markdown('<div class="Input_data">', unsafe_allow_html=True)
    st.markdown('<div class="input_heading">INPUT DATA</div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><b>Demographics</b></div>', unsafe_allow_html=True)
    person_age = st.number_input("Age", min_value=18, max_value=100, value=28)

    st.markdown('<div class="section"><b>Financial Info</b></div>', unsafe_allow_html=True)
    person_income = st.number_input("Income (annual)", min_value=0, value=65000, step=1000)
    person_home_ownership = st.selectbox("Home Ownership", list(ownership_map.keys()))

    st.markdown('<div class="section"><b>Credit History</b></div>', unsafe_allow_html=True)
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=6)
    cb_person_default_on_file = st.selectbox("Prior Default on File", list(default_map.keys()))

    st.markdown('<div class="section"><b>Employment</b></div>', unsafe_allow_html=True)
    person_emp_length = st.number_input("Employment Length (years)", min_value=0, value=5)

    st.markdown('<div class="section"><b>Loan Details</b></div>', unsafe_allow_html=True)
    loan_intent = st.selectbox("Loan Intent", loan_intent_options)
    loan_grade = st.selectbox("Loan Grade", list(grade_map.keys()))
    loan_amnt = st.number_input("Loan Amount", min_value=0, value=12000, step=500)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=10.5, step=0.1)
    loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=1.0, value=0.18, step=0.01)

    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("Predict Risk", use_container_width=True)

# ============================
# RIGHT COLUMN — Prediction output for THIS applicant
# ============================

with col2:
    if predict_btn:
        # Build single-row input dataframe from form values
        input_df = pd.DataFrame({
            "person_age": [person_age],
            "person_income": [person_income],
            "person_home_ownership": [person_home_ownership],
            "person_emp_length": [person_emp_length],
            "loan_intent": [loan_intent],
            "loan_grade": [loan_grade],
            "loan_amnt": [loan_amnt],
            "loan_int_rate": [loan_int_rate],
            "loan_percent_income": [loan_percent_income],
            "cb_person_default_on_file": [cb_person_default_on_file],
            "cb_person_cred_hist_length": [cb_person_cred_hist_length]
        })

        # Apply SAME encoding as training (preprocess.py)
        input_encoded = input_df.copy()
        input_encoded["person_home_ownership"] = input_encoded["person_home_ownership"].map(ownership_map)
        input_encoded["loan_grade"] = input_encoded["loan_grade"].map(grade_map)
        input_encoded["cb_person_default_on_file"] = input_encoded["cb_person_default_on_file"].map(default_map)

        input_encoded = pd.get_dummies(
            input_encoded,
            columns=["loan_intent"],
            drop_first=True,
            dtype=int
        )

        # Align columns exactly with training data
        input_encoded = input_encoded.reindex(columns=X_train.columns, fill_value=0)

        # Predict
        new_probability = rf.predict_proba(input_encoded)
        new_risk_prob = new_probability[0][positive_class_idx]
        new_risk_score = new_risk_prob * 100

        if new_risk_score < 30:
            new_risk_level = "Low Risk"
            risk_css_class = "risk-low"
        elif new_risk_score < 70:
            new_risk_level = "Medium Risk"
            risk_css_class = "risk-medium"
        else:
            new_risk_level = "High Risk"
            risk_css_class = "risk-high"

        st.markdown(f"""
        <div class="output_heading">PREDICTED OUTPUT</div>

        <div class="box">

        <div class="box_heading">RISK SCORE</div>

        <div class="score">{new_risk_prob:.2f}</div>

        <div class="scale-bar">
            <div class="scale-pointer" style="left:{new_risk_prob*100}%"></div>
        </div>

        <div class="scale-label">
            <span>0.0</span>
            <span>1.0</span>
        </div>

        <div class="scale-text">
            Probability of Default: {new_risk_score:.2f}%
        </div>

        </div>

        <div class="box">

        <div class="box_heading">RISK LEVEL</div>

        <div class="risk {risk_css_class}">
            {"⚠ " if new_risk_level == "High Risk" else ""}{new_risk_level}
        </div>

        </div>
        """, unsafe_allow_html=True)

        # Key factors — top 3 feature importances (model-level, not per-prediction)
        importances = pd.Series(rf.feature_importances_, index=X_train.columns)
        top_factors = importances.sort_values(ascending=False).head(3).index.tolist()

        factors_html = "".join([f"<li>{f}</li>" for f in top_factors])

        st.markdown(f"""
        <div class="box">

        <div class="box_heading">KEY FACTORS</div>

        <ul>
            {factors_html}
        </ul>

        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Fill the form on the left and click **Predict Risk** to see the result here.")