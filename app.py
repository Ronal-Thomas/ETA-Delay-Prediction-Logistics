import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

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

colA, colB = st.columns([9, 1])

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
    bg_color      = "#0E1117"
    text_color    = "#FFFFFF"
    sidebar_color = "#161A22"
    card_color    = "#1F232B"
    accent        = "#FFD500"
else:
    bg_color      = "#F4F6FA"
    text_color    = "#1F2937"
    sidebar_color = "#E5E7EB"
    card_color    = "#FFFFFF"
    accent        = "#FFB800"

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

model    = joblib.load("models/best_delay_regression_model.pkl")
scaler   = joblib.load("models/regression_scaler.pkl")
encoders = joblib.load("models/regression_label_encoders.pkl")

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

st.title("ETA & Delay Prediction Dashboard")

st.markdown(
    """
    <div style="background:#FFD500;padding:8px 16px;border-radius:8px;
    width:fit-content;font-weight:700;">
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
# SIDEBAR INPUTS
# ---------------------------------------------------

st.sidebar.title("Shipment Details")

distance = st.sidebar.number_input("Distance (km)", 1, 5000, 100)
weight   = st.sidebar.number_input("Package Weight (kg)", 1, 1000, 10)

delivery_partner = st.sidebar.selectbox(
    "Delivery Partner",
    [
        "amazon logistics",
        "blue dart",
        "delhivery",
        "dhl",
        "ecom express",
        "ekart",
        "fedex",
        "shadowfax",
        "xpressbees"
    ]
)

package_type = st.sidebar.selectbox(
    "Package Type",
    [
        "automobile parts",
        "clothing",
        "cosmetics",
        "documents",
        "electronics",
        "fragile items",
        "furniture",
        "groceries",
        "pharmacy"
    ]
)

vehicle = st.sidebar.selectbox(
    "Vehicle Type",
    [
        "ev bike",
        "bike",
        "van",
        "ev van",
        "scooter",
        "truck"
    ]
)

delivery_mode = st.sidebar.selectbox(
    "Delivery Mode",
    ["standard", "express", "same day", "two day"]
)

region = st.sidebar.selectbox(
    "Region",
    ["west", "central", "north", "east", "south"]
)

delivery_cost = st.sidebar.number_input(
    "Delivery Cost",
    min_value=1.0,
    max_value=10000.0,
    value=100.0
)

weather = st.sidebar.selectbox(
    "Weather Condition",
    ["clear", "stormy", "hot", "rainy", "cold", "foggy"]
)

hour = st.sidebar.slider("Order Hour", 0, 23, 12)

predict_button = st.sidebar.button("Predict ETA")

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

def build_reg_input(delivery_partner, package_type, vehicle,
                    delivery_mode, region, weather, distance, weight, hour):
    """
    Builds the 9-feature DataFrame expected by the regression model
    (same columns used in training — notebook Cell 8).
    """
    return pd.DataFrame({
        "delivery_partner":  [delivery_partner],
        "package_type":      [package_type],
        "vehicle_type":      [vehicle],
        "delivery_mode":     [delivery_mode],
        "region":            [region],
        "weather_condition": [weather],
        "distance_km":       [float(distance)],
        "package_weight_kg": [float(weight)],
        "hour":              [int(hour)],
    })


def encode_and_scale(df):
    """Applies saved label encoders then the regression scaler."""
    encoded = df.copy()
    for col, le in encoders.items():
        if col in encoded.columns:
            known = set(le.classes_)
            encoded[col] = encoded[col].apply(
                lambda x: le.transform([x])[0] if x in known
                else le.transform([le.classes_[0]])[0]
            )
    return scaler.transform(encoded)


def delay_to_risk(delay_hours):
    """Maps predicted delay hours to a human-readable risk label."""
    if delay_hours < 1:
        return "Low"
    elif delay_hours < 3:
        return "Moderate"
    else:
        return "High"


# ---------------------------------------------------
# RUN PREDICTION  (runs on every sidebar interaction)
# ---------------------------------------------------

reg_input    = build_reg_input(
    delivery_partner, package_type, vehicle,
    delivery_mode, region, weather, distance, weight, hour
)
scaled_input = encode_and_scale(reg_input)

delay_hours  = float(max(model.predict(scaled_input)[0], 0))
risk         = delay_to_risk(delay_hours)

# ---------------------------------------------------
# KPI PANEL  (always visible)
# ---------------------------------------------------

st.subheader("Operational Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Estimated Delay",   f"{delay_hours:.2f} hrs")
col2.metric("Risk Level",        risk)
col3.metric("Distance",          f"{distance} km")
col4.metric("Weather Condition", weather.capitalize())

st.divider()

# ---------------------------------------------------
# PREDICTION PANEL
# ---------------------------------------------------

st.subheader("Prediction Result")

col5, col6, col7 = st.columns(3)

if predict_button:

    col5.metric("Estimated Delay", f"{delay_hours:.2f} hrs")
    col6.metric("Delay Risk",      risk)
    col7.metric("Distance",        f"{distance} km")

    # Delay severity bar — normalised to a 12-hour ceiling
    progress_val = float(np.clip(delay_hours / 12.0, 0.0, 1.0))
    st.write("Delay Severity Indicator")
    st.progress(progress_val)

    st.success("Prediction generated using the XGBoost regression model.")

    st.divider()

    # -----------------------------------------------
    # PREDICTION INSIGHTS
    # -----------------------------------------------

    with st.expander("🔍 Prediction Insights"):

        st.markdown("#### Model Outputs")
        st.write(f"**Predicted Delay:** {delay_hours:.2f} hours")
        st.write(f"**Risk Level:** {risk}")

        st.markdown("#### Input Summary")

        debug_input = pd.DataFrame({
            "Feature": [
                "Delivery Partner",
                "Package Type",
                "Vehicle Type",
                "Delivery Mode",
                "Region",
                "Weather Condition",
                "Distance (km)",
                "Package Weight (kg)",
                "Order Hour"
            ],
            "Value": [
                delivery_partner,
                package_type,
                vehicle,
                delivery_mode,
                region,
                weather,
                distance,
                weight,
                hour
            ]
        })

        st.dataframe(debug_input, use_container_width=True)

else:

    col5.metric("Estimated Delay", "—")
    col6.metric("Delay Risk",      "—")
    col7.metric("Distance",        "—")

    st.info("Enter shipment details and click 'Predict ETA' to view results.")

# ---------------------------------------------------
# SCENARIO ANALYSIS
# ---------------------------------------------------

st.subheader("Scenario Delay Risk Analysis")

scenarios = ["Short Distance", "Medium Distance", "Long Distance", "Peak Hour"]

scenario_inputs = []

for scenario in scenarios:

    temp = reg_input.copy()

    if scenario == "Short Distance":
        temp["distance_km"] = max(10.0, float(distance) - 30.0)

    elif scenario == "Long Distance":
        temp["distance_km"] = float(distance) + 50.0

    elif scenario == "Peak Hour":
        temp["hour"] = (int(hour) + 5) % 24

    # "Medium Distance" keeps the base input unchanged

    scenario_inputs.append(temp)

scenario_df         = pd.concat(scenario_inputs, ignore_index=True)
scenario_scaled     = encode_and_scale(scenario_df)
scenario_delays_raw = model.predict(scenario_scaled)
scenario_delays     = [float(max(x, 0)) for x in scenario_delays_raw]

# Normalise to a 0–1 relative risk score for the bar chart
max_delay      = max(scenario_delays) if max(scenario_delays) > 0 else 1.0
scenario_risks = [round(d / max_delay, 4) for d in scenario_delays]

chart_data = pd.DataFrame({
    "Scenario":              scenarios,
    "Delay Risk":            scenario_risks,
    "Estimated Delay (hrs)": [f"{d:.2f}" for d in scenario_delays]
})

fig = px.bar(
    chart_data,
    x="Scenario",
    y="Delay Risk",
    title="Scenario-Based Delay Risk Analysis",
    color="Delay Risk",
    color_continuous_scale="YlOrBr",
    hover_data=["Estimated Delay (hrs)"]
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=card_color,
    title_font=dict(size=20, color=text_color),
    xaxis_title="Scenario",
    yaxis_title="Relative Delay Risk",
    font=dict(color=text_color, size=14),
    coloraxis_colorbar=dict(
        title=dict(text="Delay Risk", font=dict(color=text_color)),
        tickfont=dict(color=text_color)
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
    "lat": [28.6, 19.0, 13.0],
    "lon": [77.2, 72.8, 80.2]
})

st.map(map_data)

st.divider()

st.markdown("© 2026 Logistics ETA Intelligence System")