import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import pickle
import numpy as np
from datetime import date

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="ETA Delay Prediction",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# ---------------------------------------------------
# THEME TOGGLE
# ---------------------------------------------------

if "theme" not in st.session_state:
    st.session_state.theme = "light"

colA, colB = st.columns([9, 1])
with colB:
    if st.button("🌗 Theme"):
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
# LOAD ALL MODEL ARTIFACTS
# ---------------------------------------------------

@st.cache_resource
def load_artifacts():

    # ── Classification (Random Forest, 16 features) ───────────────
    # Saved with pickle.dump in notebook 08
    with open("models/classification_model.pkl", "rb") as f:
        clf_model = pickle.load(f)
    with open("models/classification_scaler.pkl", "rb") as f:
        clf_scaler = pickle.load(f)
    with open("models/classification_label_encoders.pkl", "rb") as f:
        clf_encoders = pickle.load(f)
    with open("models/classification_features.pkl", "rb") as f:
        clf_features = pickle.load(f)

    # ── Regression (LightGBM, 14 features) ────────────────────────
    # Saved with joblib.dump in notebook 05
    reg_model    = joblib.load("models/best_delay_regression_model.pkl")
    reg_scaler   = joblib.load("models/regression_scaler.pkl")
    reg_encoders = joblib.load("models/regression_label_encoders.pkl")

    return (clf_model, clf_scaler, clf_encoders, clf_features,
            reg_model, reg_scaler, reg_encoders)

(clf_model, clf_scaler, clf_encoders, CLF_FEATURES,
 reg_model, reg_scaler, reg_encoders) = load_artifacts()

# Exact 14-feature order the regression scaler was trained on (notebook Cell 18)
REG_FEATURES = [
    "delivery_partner", "package_type", "vehicle_type",
    "delivery_mode", "region", "weather_condition",
    "distance_km", "package_weight_kg", "hour",
    "delivery_cost",
    "bad_weather_flag_api",
    "is_peak_hour", "distance_bucket", "cost_per_km"
]

# ---------------------------------------------------
# GLOBAL STYLE
# ---------------------------------------------------

st.markdown(f"""
<style>
.stApp {{
    background: {bg_color};
    color: {text_color};
}}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
.block-container {{ padding-top: 2rem; }}
h1, h2, h3 {{ color: {text_color}; }}
[data-testid="stSidebar"] {{ background: {sidebar_color}; }}
[data-testid="stSidebar"] label {{
    color: {text_color};
    font-weight: 600;
}}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] div[data-baseweb="select"] {{
    background: {card_color};
    color: {text_color};
}}
[data-testid="metric-container"] {{
    background: {card_color};
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(0,0,0,0.05);
}}
[data-testid="stMetricValue"] {{
    color: {text_color};
    font-size: 28px;
    font-weight: 700;
}}
[data-testid="stMetricLabel"] {{
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}}

/* FIX metric values */
[data-testid="stMetricValue"] {{
    color: #000000 !important;
    font-weight: 800 !important;
}}

[data-testid="stAlert"] {{
    color: #1F2937 !important;
    font-weight: 600 !important;
}}

/* specifically for info box */
[data-testid="stAlert"] div {{
    color: #1F2937 !important;
}}

.stButton > button {{
    background: {accent};
    color: black;
    font-weight: 700;
    border-radius: 8px;
    height: 42px;
}}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("ETA & Delay Prediction Dashboard")
st.markdown(
    """<div style="background:#FFD500;padding:8px 16px;border-radius:8px;
    width:fit-content;font-weight:700;">AI-Powered Logistics Intelligence</div>""",
    unsafe_allow_html=True
)
st.markdown(
    f"<h3 style='color:{text_color};margin-top:10px;'>Logistics Intelligence Platform</h3>",
    unsafe_allow_html=True
)
st.divider()

# ---------------------------------------------------
# SIDEBAR — USER INPUTS
# Only collect what the user knows; everything else is derived.
# ---------------------------------------------------

st.sidebar.title("Shipment Details")

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


# delivery_cost is a direct input needed by regression model
delivery_cost = st.sidebar.number_input(
    "Delivery Cost (₹)", min_value=1.0, max_value=10000.0, value=250.0, step=10.0
)

order_date = st.sidebar.date_input("Order Date", value=date.today())

order_hour = st.sidebar.slider("Order Hour", 0, 23, 12)

predict_button = st.sidebar.button("🔍 Predict Delivery Status")

# ---------------------------------------------------
# DERIVED FEATURES
# Computed silently from sidebar inputs — not shown to user
# ---------------------------------------------------

# Classification derived features
order_dayofweek                  = order_date.weekday()         # 0=Mon … 6=Sun
is_weekend                       = 1 if order_dayofweek >= 5 else 0
holiday_or_weekend_transit_flag  = is_weekend
bad_weather_flag_api             = 1 if weather_condition in ["rainy", "stormy", "foggy"] else 0

# Fixed API weather defaults
api_temperature = 28.0
api_humidity    = 65.0
api_wind_speed  = 10.0

# Regression derived features (from notebook Cell 5)
is_peak_hour   = 1 if (8 <= order_hour <= 11 or 17 <= order_hour <= 20) else 0

if distance_km <= 100:
    distance_bucket = 0
elif distance_km <= 300:
    distance_bucket = 1
elif distance_km <= 700:
    distance_bucket = 2
else:
    distance_bucket = 3

cost_per_km = delivery_cost / (distance_km + 1)   # +1 avoids division by zero

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

def build_clf_input():
    """
    16-feature DataFrame for classification model.
    Column order enforced from CLF_FEATURES (classification_features.pkl).
    """
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
        "bad_weather_flag_api":            float(bad_weather_flag_api),
        "holiday_or_weekend_transit_flag": float(holiday_or_weekend_transit_flag),
        "order_hour":                      int(order_hour),
        "order_dayofweek":                 int(order_dayofweek),
        "is_weekend":                      int(is_weekend),
    }
    return pd.DataFrame([row])[CLF_FEATURES]


def build_reg_input():
    """
    14-feature DataFrame for regression model.
    Exact column order from notebook Cell 18 / REG_FEATURES.
    """
    row = {
        "delivery_partner":  delivery_partner,
        "package_type":      package_type,
        "vehicle_type":      vehicle_type,
        "delivery_mode":     delivery_mode,
        "region":            region,
        "weather_condition": weather_condition,
        "distance_km":       float(distance_km),
        "package_weight_kg": float(package_weight_kg),
        "hour":              int(order_hour),
        "delivery_cost":     float(delivery_cost),
        "bad_weather_flag_api": float(bad_weather_flag_api),
        "is_peak_hour":      int(is_peak_hour),
        "distance_bucket":   int(distance_bucket),
        "cost_per_km":       float(cost_per_km),
    }
    return pd.DataFrame([row])[REG_FEATURES]


def encode_scale(df, encoders, scaler):
    """
    Label-encode categoricals then apply saved scaler.
    Handles unseen labels by falling back to first known class.
    """
    enc = df.copy()
    for col, le in encoders.items():
        if col in enc.columns:
            known = set(le.classes_)
            enc[col] = enc[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known
                else le.transform([le.classes_[0]])[0]
            )
    enc = enc.apply(pd.to_numeric, errors="coerce").fillna(0)
    return scaler.transform(enc)


def prob_to_risk(prob):
    if prob < 0.3:
        return "🟢 Low"
    elif prob < 0.6:
        return "🟡 Moderate"
    else:
        return "🔴 High"

# ---------------------------------------------------
# KPI PANEL  (always visible)
# ---------------------------------------------------

st.subheader("Operational Overview")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Distance",    f"{distance_km} km")
k2.metric("Order Hour",  f"{order_hour:02d}:00")
k3.metric("Weather",     weather_condition.capitalize())
k4.metric("Order Day",   order_date.strftime("%A, %d %b %Y"))

st.divider()

# ---------------------------------------------------
# PREDICTION PANEL
# ---------------------------------------------------

st.subheader("Prediction Result")

p1, p2, p3 = st.columns(3)

if predict_button:

    # ---------------- CLASSIFICATION ----------------
    clf_df     = build_clf_input()
    clf_scaled = encode_scale(clf_df, clf_encoders, clf_scaler)

    delay_prob   = float(clf_model.predict_proba(clf_scaled)[0][1])
    clf_pred     = clf_model.predict(clf_scaled)[0]
    status_label = "Delayed ⚠️" if clf_pred == 1 else "On-Time ✅"
    risk         = prob_to_risk(delay_prob)

    # ---------------- REGRESSION ----------------
    reg_df     = build_reg_input()
    reg_scaled = encode_scale(reg_df, reg_encoders, reg_scaler)

    delay_hours = float(max(reg_model.predict(reg_scaled)[0], 0.0))

    # ---------------- METRICS ----------------
    p1.metric("Delivery Status", status_label)
    p2.metric("Delay Probability", f"{delay_prob * 100:.1f}%")

    if clf_pred == 1:
        p3.metric("Estimated Delay", f"{delay_hours:.2f} hrs")
    else:
        p3.metric("Estimated Delay", "—")

    # ---------------- PROGRESS ----------------
    st.write(f"**Delay Probability Indicator** — Risk: {risk}")
    st.progress(float(np.clip(delay_prob, 0.0, 1.0)))

    # ---------------- STATUS BANNER ----------------
    if clf_pred == 1:
        st.error(
            f"⚠️ This shipment is likely **Delayed** by approximately "
            f"**{delay_hours:.2f} hours**."
        )
    else:
        if delay_prob > 0.3:
            st.markdown(f"""
<div style="
    background-color:#FFF3CD;
    color:#1F2937;
    padding:12px 16px;
    border-radius:8px;
    font-weight:600;
">
⚠️ Slight risk detected. If delayed, it may take around 
<b>{delay_hours:.2f} hrs</b>.
</div>
""", unsafe_allow_html=True)
        else:
            st.success("✅ This shipment is expected to be **On-Time**.")

    st.divider()

    # ---------------- INSIGHTS ----------------
    st.subheader("📌 Key Insights")

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Level", risk)
    col2.metric("Peak Hour Impact", "High 🚦" if is_peak_hour else "Low ✅")
    col3.metric("Weather Impact", "High 🌧" if bad_weather_flag_api else "Low ☀️")

    st.divider()

    # ---------------- WHY ----------------
    st.subheader("🔍 Why This Prediction?")

    factors = []

    if bad_weather_flag_api:
        factors.append("🌧 Bad weather increases delay risk")

    if is_peak_hour:
        factors.append("🚦 Peak hour traffic slows delivery")

    if distance_km > 300:
        factors.append("📏 Long distance increases delay")

    if cost_per_km < 2:
        factors.append("💸 Lower cost efficiency may increase delay")

    if vehicle_type in ["bike", "ev bike", "scooter"]:
        factors.append("🏍 Two-wheelers help reduce delay in traffic")

    if delivery_mode == "standard":
        factors.append("📦 Standard delivery is slower than express")

    if region == "south":
        factors.append("📍 Region has slightly higher delay patterns")

    if factors:
        for f in factors:
            st.write(f"- {f}")
    else:
        st.success("✅ All conditions are optimal for on-time delivery")

    st.divider()

    # ---------------- SUMMARY ----------------
    st.subheader("📊 Final Summary")

    summary_text = f"""
**Delivery Status:** {status_label}  
**Delay Probability:** {delay_prob * 100:.1f}%  
"""

    if clf_pred == 1:
        summary_text += f"**Estimated Delay:** {delay_hours:.2f} hrs"
    else:
        summary_text += "**Estimated Delay:** Not applicable (On-Time)"

    st.markdown(summary_text)

else:
    p1.metric("Delivery Status",   "—")
    p2.metric("Delay Probability", "—")
    p3.metric("Estimated Delay",   "—")

    st.info("Fill in details and click Predict.")