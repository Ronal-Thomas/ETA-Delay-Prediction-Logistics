import streamlit as st
import pandas as pd
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="ETA Delay Prediction",
    layout="wide"
)

# EY Theme Styling
st.markdown("""
<style>
body {
    background-color: #F5F5F5;
}

h1, h2, h3 {
    color: #2E2E2E;
}

.stButton>button {
    background-color:#FFE600;
    color:black;
    font-weight:bold;
    border-radius:8px;
    height:40px;
}

section[data-testid="stSidebar"] {
    background-color:#2E2E2E;
    color:white;
}

section[data-testid="stSidebar"] label {
    color:white;
}

.metric-box {
    background-color:white;
    padding:20px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.title("🚚 ETA & Delay Prediction Dashboard")
st.markdown("### Logistics Intelligence Platform")

st.divider()

# SIDEBAR INPUTS
st.sidebar.title("Shipment Details")

distance = st.sidebar.number_input("Distance (km)", 0, 5000, 100)
weight = st.sidebar.number_input("Package Weight (kg)", 0, 1000, 10)

vehicle = st.sidebar.selectbox(
    "Vehicle Type",
    ["Truck", "Van", "Bike"]
)

weather = st.sidebar.selectbox(
    "Weather Condition",
    ["Clear", "Rain", "Storm"]
)

region = st.sidebar.selectbox(
    "Region",
    ["North", "South", "East", "West"]
)

predict_button = st.sidebar.button("Predict ETA")

# DASHBOARD KPIs
st.subheader("Operational Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average ETA", "14 hrs", "2%")
col2.metric("Delay Probability", "23%", "-1%")
col3.metric("Active Shipments", "1,248", "5%")
col4.metric("Weather Risk", "Moderate")

st.divider()

# PREDICTION PANEL
st.subheader("Prediction Result")

col5, col6, col7 = st.columns(3)

col5.metric("Estimated ETA", "12 hrs")
col6.metric("Delay Risk", "Low")
col7.metric("Confidence Score", "87%")

st.divider()

# ANALYTICS SECTION
st.subheader("Logistics Analytics")

data = pd.DataFrame({
    "Lane": ["A-B", "A-C", "B-C", "C-D"],
    "Delay Risk": [0.2, 0.5, 0.3, 0.6]
})

fig = px.bar(
    data,
    x="Lane",
    y="Delay Risk",
    title="Lane Delay Risk Analysis",
    color="Delay Risk",
    color_continuous_scale="YlOrBr"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# MAP PLACEHOLDER
st.subheader("Shipment Route Heatmap")

map_data = pd.DataFrame({
    "lat": [28.6, 19.0, 13.0],
    "lon": [77.2, 72.8, 80.2]
})

st.map(map_data)

st.divider()

st.markdown("© 2026 Logistics ETA Intelligence System")