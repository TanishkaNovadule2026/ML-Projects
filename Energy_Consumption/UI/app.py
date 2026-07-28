import streamlit as st
import pandas as pd
from Model.Gradient_Boostintg import model
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Energy Consumption Prediction",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
    background:#005A9C;
    color:white;
}

h1,h2,h3,label{
    color:white !important;
}

.input-box{
    background:#0c4d87;
    padding:20px;
    border-radius:10px;
    margin-bottom:20px;
}

.prediction-box{
    background:#0c4d87;
    padding:20px;
    border-radius:10px;
}

.scale-container{
    width:100%;
    height:18px;
    background:#2b5d96;
    border:1px solid #7db4ff;
    border-radius:3px;
    overflow:hidden;
}

.scale-fill{
    height:100%;
    background:repeating-linear-gradient(
        to right,
        #7CFC00 0px,
        #7CFC00 7px,
        #9ACD32 7px,
        #9ACD32 9px
    );
}

.prediction{
    font-size:24px;
    font-weight:bold;
    color:white;
}

.value{
    font-size:42px;
    font-weight:bold;
    color:white;
}

.unit{
    font-size:22px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

st.title("⚡ Energy Consumption Prediction")

st.markdown("## Input Data")

with st.container():

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        temperature = st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=100.0,
            value=45.0
        )

    with col2:
        hour = st.number_input(
            "Hour",
            min_value=0,
            max_value=23,
            value=12
        )

    with col3:
        day = st.number_input(
            "Day Of Week",
            min_value=1,
            max_value=7,
            value=3
        )

    with col4:
        appliances = st.number_input(
            "Appliances",
            min_value=0,
            value=4
        )

if st.button("Predict Usage", use_container_width=True):

    input_df = pd.DataFrame({
        "Temperature": [temperature],
        "Hour": [hour],
        "DayOfWeek": [day],
        "Appliances": [appliances]
    })

    prediction = model.predict(input_df)
    predicted_usage = float(prediction[0])

    max_usage = 10
    percentage = min((predicted_usage / max_usage) * 100, 100)

    if predicted_usage < 3:
        suggestion = "Energy usage is very low."
        color = "#2ECC71"

    elif predicted_usage < 6:
        suggestion = "⚠️ Reduce appliance usage."
        color = "#F39C12"

    elif predicted_usage < 8:
        suggestion = "High energy consumption. Turn off unused appliances."
        color = "#E67E22"

    else:
        suggestion = "Critical! Very high energy consumption."
        color = "#E74C3C"

    bar_html = f"""
    <div class="prediction-box">
        <div class="prediction">Predicted Energy Usage</div>
        <div class="value">{predicted_usage:.2f} <span class="unit">kWh</span></div>
        <div class="scale-container">
            <div class="scale-fill" style="width:{percentage}%;"></div>
        </div>
    </div>

    <style>
    .prediction-box{{
        background:#0c4d87;
        padding:20px;
        border-radius:10px;
    }}
    .prediction{{
        font-size:24px;
        font-weight:bold;
        color:white;
    }}
    .value{{
        font-size:42px;
        font-weight:bold;
        color:white;
    }}
    .unit{{
        font-size:22px;
        color:white;
    }}
    .scale-container{{
        width:100%;
        height:18px;
        background:#2b5d96;
        border:1px solid #7db4ff;
        border-radius:3px;
        overflow:hidden;
        margin-top:10px;
    }}
    .scale-fill{{
        height:100%;
        background:repeating-linear-gradient(
            to right,
            #7CFC00 0px,
            #7CFC00 7px,
            #9ACD32 7px,
            #9ACD32 9px
        );
    }}
    </style>
    """

    components.html(bar_html, height=170)

    st.markdown(f"""
    <style>
    .suggestion-box {{
        background:{color};
        color:white;
        padding:12px;
        border-radius:8px;
        font-size:22px;
        font-weight:bold;
        text-align:center;
    }}
    </style>

    <div class="suggestion-box">
    ⚠️ Suggestion: {suggestion}
    </div>
    """, unsafe_allow_html=True)