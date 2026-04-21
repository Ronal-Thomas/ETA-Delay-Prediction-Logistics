import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import pickle
import numpy as np
from datetime import date
import warnings
warnings.filterwarnings("ignore")
import requests


def get_weather(city):
    api_key = "4d80bee4da9b907635792d27e8575a75" 
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200 or "main" not in data:
            return 28.0, 65.0, 10.0, "clear"
        
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        weather = data["weather"][0]["main"].lower()

        return temp, humidity, wind, weather

    except:
        return 28.0, 65.0, 10.0, "clear"
    
def get_forecast_weather(city, target_date):
    api_key = "4d80bee4da9b907635792d27e8575a75"
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        forecasts = data["list"]

        closest = None

        for item in forecasts:
            forecast_time = item["dt_txt"]
            forecast_date = forecast_time.split(" ")[0]
            forecast_hour = int(forecast_time.split(" ")[1].split(":")[0])

            if forecast_date == str(target_date):
                # pick closest to noon (12 PM)
                if closest is None or abs(forecast_hour - 12) < abs(int(closest["dt_txt"].split(" ")[1].split(":")[0]) - 12):
                    closest = item

        if closest:
            temp = closest["main"]["temp"]
            humidity = closest["main"]["humidity"]
            wind = closest["wind"]["speed"]
            weather = closest["weather"][0]["main"].lower()

            return temp, humidity, wind, weather

        # fallback if date not found
        return 28.0, 65.0, 10.0, "clear"

    except:
        return 28.0, 65.0, 10.0, "clear"
    
    
def check_holiday(date):
    api_key = "fII9EVHQHfGmAQdDdLJH9ClfA4yPvsfl"
    url = f"https://calendarific.com/api/v2/holidays?api_key={api_key}&country=IN&year={date.year}&month={date.month}&day={date.day}"

    try:
        response = requests.get(url)
        data = response.json()

        holidays = data["response"]["holidays"]

        if len(holidays) > 0:
            return 1, holidays[0]["name"]
        else:
            return 0, None

    except:
        return 0, None

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
if st.sidebar.button("Bulk Prediction (CSV)"):
    st.session_state.show_bulk = True

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


mode = st.sidebar.radio(
    "Weather Mode",
    ["Manual", "Live (API)"]
)

holiday_mode = st.sidebar.radio(
    "Holiday Mode",
    ["Manual", "Live (API)"]
)

if mode == "Manual":
    region = st.sidebar.selectbox(
        "Region",
        ["north", "south", "east", "west"]
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

from datetime import timedelta

today = date.today()
max_date = today + timedelta(days=5)

if mode == "Live (API)":
    order_date = st.sidebar.date_input(
        "Order Date",
        value=today,
        min_value=today,
        max_value=max_date
    )
else:
    order_date = st.sidebar.date_input(
        "Order Date",
        value=today
    )
    
order_hour = st.sidebar.slider("Order Hour", 0, 23, 12)

# ---------------- WEATHER HANDLING ----------------

if mode == "Manual":
    weather_condition = st.sidebar.selectbox(
        "Weather Condition",
        ["clear", "stormy", "hot", "rainy", "cold", "foggy"]
    )

    api_temperature = 28.0
    api_humidity = 65.0
    api_wind_speed = 10.0

    bad_weather_flag_api = 1 if weather_condition in ["rainy", "stormy", "foggy"] else 0
    
elif mode == "Live (API)":
    city = st.sidebar.selectbox(
        "Select City",
        ["Bangalore", "Delhi", "Mumbai", "Chennai", "Kochi"]
    )

    st.sidebar.selectbox(
        "Region (Disabled in Live Mode)",
        ["north", "south", "east", "west"],
        index=1,
        disabled=True
    )

    region = "south"

    # 🔥 FIX: Weather logic INSIDE Live block
    if order_date == date.today():
        api_temperature, api_humidity, api_wind_speed, api_weather = get_weather(city)
    else:
        api_temperature, api_humidity, api_wind_speed, api_weather = get_forecast_weather(city, order_date)

    # Convert API → model format
    if api_weather in ["rain", "drizzle", "thunderstorm"]:
        weather_condition = "rainy"
    elif api_weather in ["fog", "mist"]:
        weather_condition = "foggy"
    elif api_temperature > 35:
        weather_condition = "hot"
    elif api_temperature < 15:
        weather_condition = "cold"
    else:
        weather_condition = "clear"

    bad_weather_flag_api = 1 if api_weather in ["rain", "drizzle", "thunderstorm", "fog", "mist"] else 0

    # Display
    st.sidebar.markdown("###  Live Weather")
    st.sidebar.write(f"City: {city}")
    st.sidebar.write(f"Condition: {weather_condition}")
    st.sidebar.write(f"Temp: {api_temperature}°C")
    st.sidebar.write(f"Humidity: {api_humidity}%")
    st.sidebar.write(f"Wind Speed: {api_wind_speed} m/s")



    
# ---------------- HOLIDAY HANDLING ----------------

if holiday_mode == "Manual":
    holiday_or_weekend_transit_flag = 1 if st.sidebar.checkbox("If The Order Date is a Holiday choose this option") else 0


elif holiday_mode == "Live (API)":
    is_holiday, holiday_name = check_holiday(order_date)

    is_weekend = 1 if order_date.weekday() >= 5 else 0

    holiday_or_weekend_transit_flag = 1 if (is_holiday or is_weekend) else 0

    st.sidebar.markdown("###  Holiday Info")

    if is_holiday:
        st.sidebar.success(f"Holiday: {holiday_name}")
    elif is_weekend:
        st.sidebar.info("Weekend")
    else:
        st.sidebar.write("No holiday")




predict_button = st.sidebar.button(" Predict Delivery Status")


# ---------------------------------------------------
# DERIVED FEATURES
# Computed silently from sidebar inputs — not shown to user
# ---------------------------------------------------

# Classification derived features
order_dayofweek                  = order_date.weekday()         # 0=Mon … 6=Sun
is_weekend                       = 1 if order_dayofweek >= 5 else 0


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
# 👉 Show manual section ONLY if bulk not clicked
if not st.session_state.get("show_bulk"):
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
        status_label = "Delayed !" if clf_pred == 1 else "On-Time "
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
                f"! This shipment is likely **Delayed** by approximately "
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
    ! Slight risk detected. If delayed, it may take around 
    <b>{delay_hours:.2f} hrs</b>.
    </div>
    """, unsafe_allow_html=True)
            else:
                st.success(" This shipment is expected to be **On-Time**.")

        st.divider()

        # ---------------- INSIGHTS ----------------
        st.subheader(" Key Insights")

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Level", risk)
        col2.metric("Peak Hour Impact", "High " if is_peak_hour else "Low ")
        col3.metric("Weather Impact", "High "  if bad_weather_flag_api else "Low ")

        st.divider()

        # ---------------- WHY ----------------
        st.subheader(" Why This Prediction?")

        factors = []

        if bad_weather_flag_api:
            factors.append(" Bad weather increases delay risk")

        if is_peak_hour:
            factors.append(" Peak hour traffic slows delivery")

        if distance_km > 300:
            factors.append(" Long distance increases delay")

        if cost_per_km < 2:
            factors.append(" Lower cost efficiency may increase delay")

        if vehicle_type in ["bike", "ev bike", "scooter"]:
            factors.append(" Two-wheelers help reduce delay in traffic")

        if delivery_mode == "standard":
            factors.append(" Standard delivery is slower than express")

        if region == "south":
            factors.append(" Region has slightly higher delay patterns")

        if factors:
            for f in factors:
                st.write(f"- {f}")
        else:
            st.success(" All conditions are optimal for on-time delivery")

        st.divider()

        # ---------------- SUMMARY ----------------
        st.subheader(" Final Summary")

        summary_text = f"""
    **Delivery Status:** {status_label}  
    **Delay Probability:** {delay_prob * 100:.1f}%  
    """

        if clf_pred == 1:
            summary_text += f"**Estimated Delay:** {delay_hours:.2f} hrs"
        else:
            summary_text += "**Estimated Delay:** Not applicable (On-Time)"

        st.markdown(summary_text)
        # ---------------- REMEDIES ----------------
        if clf_pred == 1:

            st.divider()
            st.subheader(" Suggested Actions to Reduce Delay")

            remedies = []

            if bad_weather_flag_api:
                remedies.append("Use weather-resistant packaging and plan alternate routes")

            if is_peak_hour:
                remedies.append("Reschedule delivery to non-peak hours if possible")

            if distance_km > 300:
                remedies.append("Split shipment or use faster transport options (air/express lanes)")

            if delivery_mode == "standard":
                remedies.append("Upgrade to express delivery for faster transit")

            if vehicle_type in ["bike", "ev bike", "scooter"]:
                remedies.append("Use larger vehicles for long-distance or heavy deliveries")

            if package_type == "fragile items":
                remedies.append("Use protective packaging and prioritize handling")

            if region in ["north", "east"]:
                remedies.append("Monitor regional logistics delays and optimize routing")

            # Display
            for r in remedies:
                st.write(f"• {r}")  
                
    else:
        p1.metric("Delivery Status",   "—")
        p2.metric("Delay Probability", "—")
        p3.metric("Estimated Delay",   "—")

        st.info("Fill in details and click Predict.")      

st.divider()

# 👉 Show bulk section only when button clicked
if st.session_state.get("show_bulk"):
    # ---------------------------------------------------
    # BULK PREDICTION (CSV UPLOAD)
    # ---------------------------------------------------
    if st.button("⬅ Back to Manual Prediction"):
        st.session_state.show_bulk = False
        st.rerun()
    
    st.subheader("Bulk Prediction (CSV Upload)")
    st.markdown("### 📄 Sample CSV Format")

    sample_data = pd.DataFrame({
        "delivery_partner": ["amazon logistics"],
        "package_type": ["electronics"],
        "vehicle_type": ["van"],
        "delivery_mode": ["standard"],
        "region": ["west"],
        "weather_condition": ["clear"],
        "distance_km": [120],
        "package_weight_kg": [5],
        "delivery_cost": [300],
        "order_hour": [14],
        "order_date": ["2024-06-15"]
    })

    st.write(sample_data)

    csv_sample = sample_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Sample CSV",
        data=csv_sample,
        file_name="sample_bulk_input.csv",
        mime="text/csv"
    )

    st.info("Please upload CSV with correct format")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data", df.head())

        # REQUIRED COLUMNS CHECK
        required_cols = ["delivery_partner","package_type","vehicle_type",
                        "delivery_mode","region","weather_condition",
                        "distance_km","package_weight_kg",
                        "delivery_cost","order_hour","order_date"]

        if not all(col in df.columns for col in required_cols):
            st.error("CSV format incorrect. Please use sample format.")
        else:

            # FEATURE ENGINEERING
            df["order_date"] = pd.to_datetime(df["order_date"])
            df["order_dayofweek"] = df["order_date"].dt.weekday
            df["is_weekend"] = (df["order_dayofweek"] >= 5).astype(int)
            df["holiday_or_weekend_transit_flag"] = df["is_weekend"]

            df["bad_weather_flag_api"] = df["weather_condition"].isin(
                ["rainy","stormy","foggy"]
            ).astype(int)

            df["api_temperature"] = 28.0
            df["api_humidity"] = 65.0
            df["api_wind_speed"] = 10.0

            # REG FEATURES
            df["is_peak_hour"] = df["order_hour"].apply(
                lambda x: 1 if (8<=x<=11 or 17<=x<=20) else 0
            )

            df["distance_bucket"] = pd.cut(
                df["distance_km"],
                bins=[0,100,300,700,9999],
                labels=[0,1,2,3]
            ).astype(float).fillna(0).astype(int)

            df["cost_per_km"] = df["delivery_cost"]/(df["distance_km"]+1)

            # ---------------- FIX: ALIGN FEATURES ----------------
            clf_input = df.reindex(columns=CLF_FEATURES, fill_value=0)
            reg_input = df.reindex(columns=REG_FEATURES, fill_value=0)

            # CLASSIFICATION
            clf_scaled = encode_scale(clf_input, clf_encoders, clf_scaler)

            df["delay_prediction"] = clf_model.predict(clf_scaled)
            df["delay_probability"] = clf_model.predict_proba(clf_scaled)[:,1]

            # REGRESSION
            reg_scaled = encode_scale(reg_input, reg_encoders, reg_scaler)

            df["delay_hours"] = reg_model.predict(reg_scaled)

            # SHOW OUTPUT
            st.write("Prediction Results", df)

            # DOWNLOAD BUTTON
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "predictions.csv")   