"""
Salary Prediction System - Streamlit UI
Location in project: Salary_Prediction/ui/streamlit.py

Before running this, make sure the model is trained:
    python -m src.train

Then run the UI (from the Salary_Prediction/ folder):
    streamlit run ui/streamlit.py
"""

import os
import sys
import streamlit as st
# ---------------------------------------------------------------------------
# Make src/ importable (this file lives in ui/, src/ is one level up)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MODEL_AVAILABLE = True
IMPORT_ERROR = ""

try:
    from src.predict import predict_salary  # noqa: E402
except Exception as e:  # pragma: no cover
    MODEL_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Try to load dataset to populate dropdowns (optional)
DATA_FILE = os.path.join(PROJECT_ROOT, 'Data', 'Sample_HR.csv')
raw = None
if os.path.exists(DATA_FILE):
    try:
        import pandas as _pd

        raw = _pd.read_csv(DATA_FILE)
    except Exception:
        raw = None


def fallback_predict(experience, education, job_role, location, skills_count):
    """
    Used only if src/predict.py can't be imported (e.g. model not trained yet).
    Run `python -m src.train` to remove this fallback and use the real model.
    """
    edu_weight = {
        "High School": 0,
        "Bachelor's Degree": 15000,
        "Master's Degree": 30000,
        "PhD": 45000,
    }.get(education, 0)

    role_weight = {
        "Data Scientist": 20000,
        "Software Engineer": 15000,
        "Product Manager": 25000,
        "Data Analyst": 10000,
        "ML Engineer": 22000,
    }.get(job_role, 0)

    location_weight = {
        "Bangalore": 10000,
        "Mumbai": 9000,
        "Delhi": 8000,
        "Hyderabad": 7000,
        "Pune": 6000,
        "Remote": 5000,
    }.get(location, 0)

    base = 30000
    salary = (
        base
        + edu_weight
        + role_weight
        + location_weight
        + experience * 4000
        + skills_count * 1500
    )
    return round(salary, 2), "Ridge Regression (demo fallback - train.py not run yet)"


# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Salary Prediction", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f7f8fa;
            color: black;
        }
        .top-banner {
            background: linear-gradient(90deg, #ff5f6d 0%, #ffc371 100%);
            height: 8px;
            width: 100%;
            border-radius: 4px;
            margin-bottom: 1.5rem;
        }
        .main-title {
            text-align: center;
            font-size: 2rem;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            text-align: center;
            color: #6b7280;
            margin-bottom: 2rem;
        }
        .card {
            background: #ffffff;
            border-radius: 12px;
            padding: 1.5rem 1.5rem 1.8rem 1.5rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
            border: 1px solid #eef0f2;
            height: 100%;
        }
        .card-title {
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 1rem;
            color: #1a1a1a;
        }
        .model-badge {
            background: #eafaf0;
            color: #1c7c3e;
            padding: 0.5rem 0.9rem;
            border-radius: 8px;
            font-size: 0.9rem;
            display: inline-block;
            margin-bottom: 1.2rem;
        }
        .salary-label {
            color: #6b7280;
            font-size: 0.9rem;
            margin-bottom: 0.2rem;
        }
        .salary-value {
            color: #2563eb;
            font-size: 2.1rem;
            font-weight: 700;
        }
        div.stButton > button {
            background-color: #2563eb;
            color: white;
            font-weight: 600;
            padding: 0.6rem 0;
            border-radius: 8px;
            border: none;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #1d4ed8;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="top-banner"></div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Salary Prediction Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enter your details to estimate your salary.</div>',
    unsafe_allow_html=True,
)


# Layout: form (left) + result (right)
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown('<div class="card-title">Enter Employee Details</div>', unsafe_allow_html=True)

    # Use a form so submit is explicit and accessible

    with st.form(key="input_form"):
        st.markdown('<div class="text">Enter Ecperience</div>', unsafe_allow_html=True)
        experience = st.number_input(
            "Years of Experience:", min_value=0, max_value=50, value=5, step=1
        )
        st.markdown('<div class="text">Enter Education Level</div>', unsafe_allow_html=True)
        if raw is not None and 'Education' in raw.columns:
            edu_options = raw['Education'].unique().tolist()
        else:
            edu_options = ["High School", "Bachelor's Degree", "Master's Degree", "PhD"]
        education = st.selectbox(
            "Education Level:", edu_options, index=min(2, len(edu_options) - 1)
        )

        st.markdown('<div class="text">Enter JobRole</div>', unsafe_allow_html=True)
        if raw is not None and 'JobRole' in raw.columns:
            job_options = raw['JobRole'].unique().tolist()
        else:
            job_options = ["Data Scientist", "Software Engineer", "Product Manager", "Data Analyst", "ML Engineer"]
        job_role = st.selectbox("Job Role:", job_options, index=0)
        if raw is not None and 'Location' in raw.columns:
            loc_options = raw['Location'].unique().tolist()
        else:
            loc_options = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Remote"]
        st.markdown('<div class="text">Enter Location</div>', unsafe_allow_html=True)
        location = st.selectbox("Location:", loc_options, index=0)

        st.markdown('<div class="text">Enter Skill Count</div>', unsafe_allow_html=True)
        skills_count = st.number_input(
            "Skills Count:", min_value=0, max_value=50, value=6, step=1
        )
        st.markdown('<div class="text">Enter Age</div>', unsafe_allow_html=True)
        age = st.number_input(
            "Age:", min_value=18, max_value=70, value=30, step=1
        )
        submit = st.form_submit_button("Predict Salary")

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card-title">Predicted Salary</div>', unsafe_allow_html=True)

    if "salary_result" not in st.session_state:
        st.session_state.salary_result = None
        st.session_state.model_used = None

    if submit:
        try:
            if MODEL_AVAILABLE:
                salary, model_used = predict_salary(
                    experience, education, job_role, location, skills_count, age
                )
            else:
                salary, model_used = fallback_predict(
                    experience, education, job_role, location, skills_count
                )
            st.session_state.salary_result = salary
            st.session_state.model_used = model_used
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if st.session_state.salary_result is not None:
        st.markdown(
            f'<div class="model-badge"> Best Model Used: {st.session_state.model_used}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="salary-label">Estimated Salary:</div>', unsafe_allow_html=True)
        salary_display = max(float(st.session_state.salary_result), 0.0)
        st.markdown(
            f'<div class="salary-value">₹{salary_display:,.0f}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#9ca3af;">Fill the form and click '
            '"Predict Salary" to see the result here.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)