import streamlit as st
from src import predict, preprocess, evaluate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Disease Detection System",
    layout= "wide",
    
)
st.markdown("""
<style>

.stApp{
    background:#f3f6fb;
    color:black;
}

/* Title */
.title{
    background:#1f5fbf;
    color:white;
    font-size:34px;
    font-weight:bold;
    text-align:center;
    padding:15px;
    border-radius:8px 8px 0 0;
    margin-bottom:15px;
}

/* Prediction */
.predict{
    background:white;
    border:1px solid #d9e2f2;
    border-radius:6px;
    padding:12px 18px;
    margin-bottom:12px;
    box-shadow:0px 2px 5px rgba(0,0,0,0.08);
}

.predict p{
    margin:0;
    font-size:20px;
}

.predict b{
    color:#1d3f91;
    font-size:28px;
}

/* Confidence */
.confidence{
    background:white;
    border:1px solid #d9e2f2;
    border-radius:6px;
    padding:12px 18px;
    margin-bottom:12px;
    box-shadow:0px 2px 5px rgba(0,0,0,0.08);
}

.confidence p{
    margin:0;
    font-size:20px;
}

.confidence b{
    color:#333;
    font-size:28px;
}

/* Top Diseases */
.top_disease{
    background:white;
    border:1px solid #d9e2f2;
    border-radius:6px;
    padding:12px 18px;
    margin-bottom:15px;
    box-shadow:0px 2px 5px rgba(0,0,0,0.08);
}

.top_disease strong{
    color:#1d3f91;
    font-size:19px;
}

.top_disease ol{
    margin-top:10px;
    padding-left:22px;
}

.top_disease li{
    font-size:17px;
    margin-bottom:5px;
}

/* Explanation */
.Header{
    background:white;
    border:1px solid #d9e2f2;
    border-radius:6px 6px 0 0;
    padding:12px 18px;
    font-size:22px;
    font-weight:bold;
    color:#1d3f91;
    margin-bottom:0px;
    box-shadow:0px 2px 5px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="title">Disease Prediction Dashboard</div>
""", unsafe_allow_html=True)

Disease_Prict = evaluate.predicted_disease
st.markdown(f"""
    <div class="predict">
    <p>  Disease Prediction : <b>{Disease_Prict}</b> </p>
    </div>
""", unsafe_allow_html=True)


Confidence_Level = evaluate.highest_confidence*100
st.markdown(f"""
    <div class="confidence">
    <p>  Confidence : <b>{Confidence_Level:.2f}%</b> </p>
    </div>
""", unsafe_allow_html=True)


preprocess_top = preprocess.top_disease
items_html = "\n".join(f"<li>{label}</li>" for label, count in preprocess_top.items())

st.markdown(f"""
<div class="top_disease">
    <div><strong>Top 3 Diseases:</strong></div>
    <ol>
        {items_html}
    </ol>
</div>
""", unsafe_allow_html=True)



st.markdown("""
<div class="Header">Explanation</div>
""", unsafe_allow_html=True)
# Feature importance bar plot
feature_names = preprocess.df.columns[:-1]   # Exclude target column (Disease)

# Feature importance from Random Forest
feature_importance = evaluate.rf_model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
})

# Sort and select Top 4
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(4)

# Impact Labels
impact = [
    "High Impact",
    "Medium Impact",
    "Low Impact",
    "Minimal Impact"
]

importance_df["Impact"] = impact[:len(importance_df)]

# Plot
fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    text="Impact",
    color="Impact",
    color_discrete_map={
        "High Impact": "#ff8c42",
        "Medium Impact": "#3b82f6",
        "Low Impact": "#34a853",
        "Minimal Impact": "#9ca3af"
    },
    title="Key Symptoms Influencing Prediction"
)

fig.update_traces(
    textposition="outside"
    
)

fig.update_layout(
    yaxis=dict(
        autorange="reversed",
        title=dict(text="Feature", font=dict(color="black")),
        tickfont=dict(color="black")
    ),
    xaxis=dict(
        title=dict(text="Importance", font=dict(color="black")),
        tickfont=dict(color="black"),
        zeroline=False
    ),
    showlegend=False,

    template="plotly_white",
    height=350,
    margin=dict(l=20, r=100, t=50, b=20)
)
fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(
        color="black",
        size=14
    )
)
st.plotly_chart(fig, use_container_width=True)