import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="ETA Delay Prediction",
    layout="wide"
)

# ---------------------------------------------------
# THEME TOGGLE
# ---------------------------------------------------

if "theme" not in st.session_state:
    st.session_state.theme = "light"

colA, colB = st.columns([9,1])

with colB:
    toggle = st.button("🌗 Theme")

if toggle:
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

# ---------------------------------------------------
# THEME COLORS
# ---------------------------------------------------

if st.session_state.theme == "dark":

    bg_color = "#0E1117"
    text_color = "#FFFFFF"
    sidebar_color = "#161A22"
    card_color = "#1F232B"
    accent = "#FFD500"

else:

    bg_color = "#F4F6FA"
    text_color = "#1F2937"
    sidebar_color = "#E5E7EB"
    card_color = "#FFFFFF"
    accent = "#FFB800"

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

MODEL_PATH = "models/best_delay_regression_model.pkl"
SCALER_PATH = "models/regression_scaler.pkl"
ENCODER_PATH = "models/regression_label_encoders.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODER_PATH)

# ---------------------------------------------------
# GLOBAL STYLE
# ---------------------------------------------------

st.markdown(f"""
<style>

.stApp {{
background:{bg_color};
color:{text_color};
}}

header {{
visibility:hidden;
}}

.block-container {{
padding-top:2rem;
}}

h1,h2,h3 {{
color:{text_color};
}}

[data-testid="stSidebar"] {{
background:{sidebar_color};
}}

[data-testid="stSidebar"] label {{
color:{text_color};
font-weight:600;
}}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] div[data-baseweb="select"] {{
background:{card_color};
color:{text_color};
}}

[data-testid="metric-container"] {{
background:{card_color};
border-radius:12px;
padding:20px;
border:1px solid rgba(0,0,0,0.05);
}}

[data-testid="stMetricValue"] {{
color:{text_color};
font-size:28px;
font-weight:700;
}}

.stButton > button {{
background:{accent};
color:black;
font-weight:700;
border-radius:8px;
height:42px;
}}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("🚚 ETA & Delay Prediction Dashboard")

st.markdown(
"""
<div style="background:#FFD500;padding:8px 16px;border-radius:8px;width:fit-content;font-weight:700;">
AI-Powered Logistics Intelligence
</div>
""",
unsafe_allow_html=True
)

st.markdown(
f"<h3 style='color:{text_color};margin-top:10px;'>Logistics Intelligence Platform</h3>",
unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("Shipment Details")

distance = st.sidebar.number_input("Distance (km)",0,5000,100)
weight = st.sidebar.number_input("Package Weight (kg)",0,1000,10)

vehicle = st.sidebar.selectbox(
"Vehicle Type",
["Truck","Van","Bike"]
)

weather = st.sidebar.selectbox(
"Weather Condition",
["Clear","Rain","Storm"]
)

region = st.sidebar.selectbox(
"Region",
["North","South","East","West"]
)

predict_button = st.sidebar.button("Predict ETA")

# ---------------------------------------------------
# VALUE MAPPING
# ---------------------------------------------------

vehicle_map = {
"Truck":"truck",
"Van":"van",
"Bike":"bike"
}

weather_map = {
"Clear":"clear",
"Rain":"rainy",
"Storm":"stormy"
}

region_map = {
"North":"north",
"South":"south",
"East":"east",
"West":"west"
}

vehicle_model = vehicle_map[vehicle]
weather_model = weather_map[weather]
region_model = region_map[region]

# ---------------------------------------------------
# INPUT DATA
# ---------------------------------------------------

input_data = pd.DataFrame({
"delivery_partner":["dhl"],
"package_type":["electronics"],
"vehicle_type":[vehicle_model],
"delivery_mode":["express"],
"region":[region_model],
"weather_condition":[weather_model],
"distance_km":[distance],
"package_weight_kg":[weight],
"hour":[12]
})

# ---------------------------------------------------
# ENCODING
# ---------------------------------------------------

for col,encoder in encoders.items():
    if col in input_data.columns:
        input_data[col] = encoder.transform(input_data[col])

# ---------------------------------------------------
# SCALING
# ---------------------------------------------------

input_scaled = scaler.transform(input_data)

# ---------------------------------------------------
# KPI PANEL
# ---------------------------------------------------

st.subheader("Operational Overview")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Average ETA","14 hrs","2%")
col2.metric("Delay Probability","23%","-1%")
col3.metric("Active Shipments","1,248","5%")
col4.metric("Weather Risk","Moderate")

st.divider()

# ---------------------------------------------------
# PREDICTION PANEL
# ---------------------------------------------------

st.subheader("Prediction Result")

col5, col6, col7 = st.columns(3)

if predict_button:

    prediction = model.predict(input_scaled)[0]

    delay_hours = max(prediction, 0)
    delay_minutes = delay_hours * 60

    # Risk classification
    if delay_hours < 0.5:
        risk = "Low"
    elif delay_hours < 1.5:
        risk = "Moderate"
    else:
        risk = "High"

    confidence = 90

    # KPI Display
    col5.metric("Estimated ETA", f"{delay_hours:.2f} hrs")
    col6.metric("Delay Risk", risk)
    col7.metric("Confidence Score", f"{confidence}%")

    st.write("Delay Severity Indicator")
    st.progress(min(delay_hours/3,1.0))

    st.success("Prediction generated using trained regression model")

    st.divider()

    # ---------------------------------------------------
    # MODEL DEBUG INFORMATION
    # ---------------------------------------------------

    st.subheader("Model Debug Information")

    st.markdown("#### Raw Model Output")
    st.code(f"{prediction:.4f} hours")

    st.markdown("#### Input Features Sent to Model")

    debug_input = pd.DataFrame({
        "Feature":[
            "Delivery Partner",
            "Package Type",
            "Vehicle Type",
            "Delivery Mode",
            "Region",
            "Weather Condition",
            "Distance (km)",
            "Package Weight (kg)",
            "Hour of Day"
        ],
        "Value":[
            "DHL",
            "Electronics",
            vehicle,
            "Express",
            region,
            weather,
            distance,
            weight,
            12
        ]
    })

    st.dataframe(debug_input, use_container_width=True)

    st.markdown("#### Encoded + Scaled Features Used by Model")

    scaled_df = pd.DataFrame(
        input_scaled,
        columns=input_data.columns
    )

    st.dataframe(scaled_df, use_container_width=True)

else:

    col5.metric("Estimated ETA", "12 hrs")
    col6.metric("Delay Risk", "Low")
    col7.metric("Confidence Score", "87%")
# ---------------------------------------------------
# ANALYTICS
# ---------------------------------------------------

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

fig.update_layout(

    # remove white background
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=card_color,

    # fix title
    title_font=dict(
        size=20,
        color=text_color
    ),

    # axis labels
    xaxis_title="Route Lane",
    yaxis_title="Delay Risk",

    font=dict(
        color=text_color,
        size=14
    ),

    coloraxis_colorbar=dict(
        title="Delay Risk",
        tickfont=dict(color=text_color),
        titlefont=dict(color=text_color)
    )
)

fig.update_xaxes(
    showgrid=False,
    tickfont=dict(color=text_color),
    title_font=dict(color=text_color)
)

fig.update_yaxes(
    gridcolor="rgba(120,120,120,0.2)",
    tickfont=dict(color=text_color),
    title_font=dict(color=text_color)
)

st.plotly_chart(fig, use_container_width=True)
# ---------------------------------------------------
# MAP
# ---------------------------------------------------

st.subheader("Shipment Route Heatmap")

map_data = pd.DataFrame({
"lat":[28.6,19.0,13.0],
"lon":[77.2,72.8,80.2]
})

st.map(map_data)

st.divider()

st.markdown("© 2026 Logistics ETA Intelligence System")