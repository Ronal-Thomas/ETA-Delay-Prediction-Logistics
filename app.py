import streamlit as st

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="CargoPulse",
    layout="wide"
)

# ---------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------
st.markdown("""
<style>

.header-banner{
    position: fixed;
    top: 0;
    left: 50;
    width: 100%;
    z-index: 999;
    text-align: center;
}

.header-title{
    text-align: center;
    margin: 0 auto;
}

.header-subtitle{
    text-align: center;
    margin: 0 auto;
}

.block-container{
    padding-top:140px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* PAGE BACKGROUND */

.stApp {
    background-color:#ECECEF;
}

/* Remove top margin */

.block-container {
    padding-top:0rem !important;
    padding-left:0rem;
    padding-right:0rem;
}

/* Hide Streamlit top bar gap */

[data-testid="stHeader"]{
    height:0px;
}

/* HEADER */

.header-banner {
    background:linear-gradient(90deg,#2E2E38,#3A3A44);
    padding:60px 20px;
    text-align:center;
}

.header-title{
    color:white;
    font-size:56px;
    font-weight:700;
}

.header-subtitle{
    color:#FFD500;
    font-size:22px;
    font-family:"Georgia", serif;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color:#4A4A55;
}

/* Sidebar labels */

section[data-testid="stSidebar"] label {
    color:#FFFFFF !important;
    font-size:16px !important;
    font-weight:500;
}

/* SELECTBOX */

div[data-baseweb="select"] > div {
    background-color:white !important;
    color:#000000 !important;
    font-size:15px !important;
}

/* DROPDOWN TEXT */

div[data-baseweb="select"] span {
    color:#000000 !important;
    font-size:15px !important;
}

/* NUMBER INPUT */

input {
    background-color:white !important;
    color:#000000 !important;
    font-size:15px !important;
}

/* Placeholder text */

input::placeholder {
    color:#555 !important;
}

/* BUTTON */

.stButton>button {
    background-color:#FFD500;
    color:#2E2E38;
    font-weight:600;
    font-size:15px;
}

/* MAIN PANEL */

.main-panel {
    background:#F7F7F9;
    margin:40px auto;
    padding:40px;
    border-radius:10px;
    width:85%;
    min-height:450px;
}

/* INFO BOX */

.info-box {
    background:#E3E3E8;
    padding:20px;
    border-radius:8px;
    border-left:5px solid #FFD500;
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("""
<div class="header-banner">
    <div class="header-title">CargoPulse</div>
    <div class="header-subtitle">
    The heartbeat of real-time delivery intelligence.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------

st.sidebar.markdown(
    "<h3 style='color:#FFD700; font-weight:bold;'>Shipment Details</h3>",
    unsafe_allow_html=True
)

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

vehicle_type = st.sidebar.selectbox(
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

region = st.sidebar.selectbox(
    "Region",
    ["west", "central", "north", "east", "south"]
)

weather_condition = st.sidebar.selectbox(
    "Weather Condition",
    ["clear", "stormy", "hot", "rainy", "cold", "foggy"]
)

distance_km = st.sidebar.number_input(
    "Distance (km)",
    min_value=1.0,
    max_value=1000.0,
    value=10.0,
    step=0.1,
    format="%.1f"
)

package_weight = st.sidebar.number_input(
    "Package Weight (kg)",
    min_value=0.1,
    max_value=200.0,
    value=5.0
)

delivery_cost = st.sidebar.number_input(
    "Delivery Cost",
    min_value=1.0,
    max_value=10000.0,
    value=100.0
)

predict = st.sidebar.button("Run Prediction")

# ---------------------------------------------------
# MAIN PANEL
# ---------------------------------------------------

st.markdown('<div class="main-panel">', unsafe_allow_html=True)

if predict:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Delivery Status")
        st.error("Delayed Delivery")

    with col2:
        st.subheader("Estimated Delay")
        st.metric(
            label="Predicted Delay Duration",
            value="2.4 hours"
        )

    with col3:
        st.subheader("Delay Probability")
        st.progress(0.72)
        st.write("72 % risk")

else:

    st.markdown(
        """
        <div class="info-box">
        Enter delivery parameters in the sidebar and click <b>Run Prediction</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)