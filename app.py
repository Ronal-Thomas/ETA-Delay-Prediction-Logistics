import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from datetime import date

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
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

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

@st.cache_resource
def load_artifacts():
    model     = joblib.load("models/classification_model.pkl")
    encoders  = joblib.load("models/classification_label_encoders.pkl")
    scaler    = joblib.load("models/classification_scaler.pkl")
    features  = joblib.load("models/classification_features.pkl")
    return model, encoders, scaler, features

model, encoders, scaler, SAFE_FEATURES = load_artifacts()

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

# -- User inputs -------------------------------------------------------

delivery_partner = st.sidebar.selectbox(
    "Delivery Partner",
    ["amazon logistics", "blue dart", "delhivery", "dhl",
     "ecom express", "ekart", "fedex", "shadowfax", "xpressbees"]
)

package_type = st.sidebar.selectbox(
    "Package Type",
    ["automobile parts", "clothing", "cosmetics", "documents",
     "electronics", "fragile items", "furniture", "groceries", "pharmacy"]
)

vehicle_type = st.sidebar.selectbox(
    "Vehicle Type",
    ["ev bike", "bike", "van", "ev van", "scooter", "truck"]
)

delivery_mode = st.sidebar.selectbox(
    "Delivery Mode",
    ["standard", "express", "same day", "two day"]
)

region = st.sidebar.selectbox(
    "Region",
    ["west", "central", "north", "east", "south"]
)

weather_condition = st.sidebar.selectbox(
    "Weather Condition",
    ["clear", "stormy", "hot", "rainy", "cold", "foggy"]
)

distance_km = st.sidebar.number_input(
    "Distance (km)", min_value=1, max_value=5000, value=100
)

package_weight_kg = st.sidebar.number_input(
    "Package Weight (kg)", min_value=1, max_value=1000, value=10
)

order_date = st.sidebar.date_input(
    "Order Date", value=date.today()
)

order_hour = st.sidebar.slider("Order Hour", 0, 23, 12)

predict_button = st.sidebar.button("Predict Delivery Status")

# -- Derived features (not shown to user) ------------------------------

order_dayofweek                 = order_date.weekday()          # 0=Mon, 6=Sun
is_weekend                      = 1 if order_dayofweek >= 5 else 0
holiday_or_weekend_transit_flag = is_weekend
bad_weather_flag_api            = 1 if weather_condition in ["rainy", "stormy", "foggy"] else 0

# API weather defaults (fixed; replace with live API call if needed)
api_temperature = 28.0
api_humidity    = 65.0
api_wind_speed  = 10.0

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

def build_input():
    """Builds the input DataFrame in the exact SAFE_FEATURES column order."""
    row = {
        "delivery_partner":               delivery_partner,
        "package_type":                   package_type,
        "vehicle_type":                   vehicle_type,
        "delivery_mode":                  delivery_mode,
        "region":                         region,
        "weather_condition":              weather_condition,
        "distance_km":                    float(distance_km),
        "package_weight_kg":              float(package_weight_kg),
        "api_temperature":                api_temperature,
        "api_humidity":                   api_humidity,
        "api_wind_speed":                 api_wind_speed,
        "bad_weather_flag_api":           bad_weather_flag_api,
        "holiday_or_weekend_transit_flag":holiday_or_weekend_transit_flag,
        "order_hour":                     order_hour,
        "order_dayofweek":                order_dayofweek,
        "is_weekend":                     is_weekend,
    }
    # Enforce exact column order from training
    return pd.DataFrame([row])[SAFE_FEATURES]


def encode_and_scale(df):
    """Label-encode categoricals then apply the saved scaler."""
    encoded = df.copy()
    for col, le in encoders.items():
        if col in encoded.columns:
            known = set(le.classes_)
            encoded[col] = encoded[col].apply(
                lambda x: le.transform([x])[0] if x in known
                else le.transform([le.classes_[0]])[0]
            )
    encoded = encoded.apply(pd.to_numeric, errors="coerce").fillna(0)
    return scaler.transform(encoded)


def prob_to_risk(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Moderate"
    else:
        return "High"

# ---------------------------------------------------
# KPI PANEL  (always visible)
# ---------------------------------------------------

st.subheader("Operational Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Distance",          f"{distance_km} km")
col2.metric("Order Hour",        f"{order_hour}:00")
col3.metric("Weather Condition", weather_condition.capitalize())
col4.metric("Order Day",         order_date.strftime("%A, %d %b %Y"))

st.divider()

# ---------------------------------------------------
# PREDICTION PANEL
# ---------------------------------------------------

st.subheader("Prediction Result")

col5, col6, col7 = st.columns(3)

if predict_button:

    # Build → encode → scale → predict
    input_df     = build_input()
    scaled_input = encode_and_scale(input_df)

    prediction   = model.predict(scaled_input)[0]          # 0 = On-Time, 1 = Delayed
    delay_prob   = model.predict_proba(scaled_input)[0][1] # probability of delay
    risk         = prob_to_risk(delay_prob)
    status_label = "Delayed" if prediction == 1 else "On-Time"

    # ── Metrics ────────────────────────────────────────────────────
    col5.metric("Delivery Status",  status_label)
    col6.metric("Delay Probability", f"{delay_prob * 100:.0f}%")
    col7.metric("Risk Level",        risk)

    # ── Delay probability bar ──────────────────────────────────────
    st.write("Delay Probability Indicator")
    st.progress(float(np.clip(delay_prob, 0.0, 1.0)))

    # ── Status banner ──────────────────────────────────────────────
    if prediction == 1:
        st.error("⚠️  This shipment is likely to be **Delayed**.")
    else:
        st.success("✅  This shipment is expected to be **On-Time**.")

    st.divider()

    # ── Prediction Insights ────────────────────────────────────────
    with st.expander("🔍 Prediction Insights"):

        st.markdown("#### Model Outputs")
        st.write(f"**Delivery Status:** {status_label}")
        st.write(f"**Delay Probability:** {delay_prob * 100:.1f}%")
        st.write(f"**Risk Level:** {risk}")

        st.markdown("#### Input Summary")

        debug_df = pd.DataFrame({
            "Feature": [
                "Delivery Partner", "Package Type", "Vehicle Type",
                "Delivery Mode", "Region", "Weather Condition",
                "Distance (km)", "Package Weight (kg)",
                "Order Date", "Order Hour",
                "Day of Week", "Is Weekend",
                "Bad Weather Flag", "Holiday / Weekend Transit Flag"
            ],
            "Value": [
                delivery_partner, package_type, vehicle_type,
                delivery_mode, region, weather_condition,
                distance_km, package_weight_kg,
                order_date.strftime("%d %b %Y"), order_hour,
                order_date.strftime("%A"), "Yes" if is_weekend else "No",
                "Yes" if bad_weather_flag_api else "No",
                "Yes" if holiday_or_weekend_transit_flag else "No"
            ]
        })

        st.dataframe(debug_df, use_container_width=True)

else:

    col5.metric("Delivery Status",   "—")
    col6.metric("Delay Probability", "—")
    col7.metric("Risk Level",        "—")

    st.info("Enter shipment details in the sidebar and click **Predict Delivery Status**.")

# ---------------------------------------------------
# SCENARIO ANALYSIS
# ---------------------------------------------------

st.subheader("Scenario Delay Risk Analysis")

scenarios = ["Short Distance", "Base Case", "Long Distance", "Peak Hour"]

scenario_rows = []

for scenario in scenarios:

    row = {
        "delivery_partner":                delivery_partner,
        "package_type":                    package_type,
        "vehicle_type":                    vehicle_type,
        "delivery_mode":                   delivery_mode,
        "region":                          region,
        "weather_condition":               weather_condition,
        "distance_km":                     float(distance_km),
        "package_weight_kg":               float(package_weight_kg),
        "api_temperature":                 api_temperature,
        "api_humidity":                    api_humidity,
        "api_wind_speed":                  api_wind_speed,
        "bad_weather_flag_api":            bad_weather_flag_api,
        "holiday_or_weekend_transit_flag": holiday_or_weekend_transit_flag,
        "order_hour":                      order_hour,
        "order_dayofweek":                 order_dayofweek,
        "is_weekend":                      is_weekend,
    }

    if scenario == "Short Distance":
        row["distance_km"] = max(10.0, float(distance_km) - 30.0)

    elif scenario == "Long Distance":
        row["distance_km"] = float(distance_km) + 50.0

    elif scenario == "Peak Hour":
        row["order_hour"] = (order_hour + 5) % 24

    scenario_rows.append(row)

scenario_df     = pd.DataFrame(scenario_rows)[SAFE_FEATURES]
scenario_scaled = encode_and_scale(scenario_df)
scenario_probs  = model.predict_proba(scenario_scaled)[:, 1]

chart_data = pd.DataFrame({
    "Scenario":           scenarios,
    "Delay Probability":  [round(p, 4) for p in scenario_probs],
    "Risk":               [prob_to_risk(p) for p in scenario_probs]
})

fig = px.bar(
    chart_data,
    x="Scenario",
    y="Delay Probability",
    title="Scenario-Based Delay Probability",
    color="Delay Probability",
    color_continuous_scale="YlOrBr",
    hover_data=["Risk"],
    range_y=[0, 1]
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=card_color,
    title_font=dict(size=20, color=text_color),
    xaxis_title="Scenario",
    yaxis_title="Delay Probability",
    font=dict(color=text_color, size=14),
    coloraxis_colorbar=dict(
        title=dict(text="Probability", font=dict(color=text_color)),
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