# ETA Delay Prediction for Logistics Deliveries

## Project Overview
This project aims to build a Machine Learning system that predicts delivery delays in logistics operations.

The system performs two main tasks:
- Predict whether a delivery will be delayed or on time (Classification)
- Predict the expected delay duration in hours (Regression)

The model uses logistics data along with external factors such as **holidays and weather conditions** to improve prediction accuracy.

---

# Dataset Description

The dataset contains delivery information such as:

- delivery_partner
- package_type
- vehicle_type
- delivery_mode
- region
- weather_condition
- distance_km
- package_weight_kg
- delivery_cost
- delivery_rating
- order timestamps
- expected delivery timestamps
- actual delivery timestamps

Target Variables:
- **delayed_flag_recon** → Classification target
- **delay_hours_recon** → Regression target

Dataset size: **25,000 rows**

---

# Feature Engineering

Several features were engineered to improve prediction performance.

## Time Features
Derived from order timestamps:

- order_dayofweek
- order_day_name
- order_month
- order_year
- order_hour
- is_weekend
- rush_hour_flag
- night_delivery_flag

---

## Holiday Features
Holiday data was fetched using the **Calendarific API**.

Features created:
- holiday_count_transit
- holiday_names_transit
- weekend_count_transit
- holiday_or_weekend_transit_flag
- holiday_proximity_feature

These features capture delays caused by **public holidays or weekends during the delivery transit period**.

---

## Weather Features
Weather information was integrated using the **OpenWeatherMap API**.

Features created:
- api_temperature
- api_humidity
- api_wind_speed
- bad_weather_flag_api

Bad weather flag is set when:
- weather condition contains rain, storm, fog or snow
- OR wind speed > 10
- OR humidity > 85

---

# Project Structure
ETA-DELAY-PREDICTION-LOGISTICS
│
├── Data
│ ├── Delivery_Logistics_reconstructed.csv
│ ├── Added_Holiday_Features_dataset.csv
│ ├── dataset_with_weather_features.csv
│ ├── india_holidays_cached.csv
│ └── weather_cached.csv
│
├── notebooks
│ └── ETA_Delay_Prediction_Corrected.ipynb
│
├── src
│ ├── add_holidays_to_dataset.py
│ ├── weather_enrichment.py
│ └── ETA_Delay_Prediction.py
│
├── README.md
├── requirements.txt
└── FEATURE ENGINEERING SUMMARY.docx


---

# APIs Used

## Holiday API
Calendarific API  
Used to fetch national holidays in India.

## Weather API
OpenWeatherMap API  
Used to fetch weather information such as temperature, humidity, and wind speed.

---

# Model Pipeline

1. Load logistics dataset
2. Perform data cleaning
3. Feature engineering
4. Holiday API integration
5. Weather API integration
6. Train classification and regression models
7. Evaluate model performance

---

# How to Run the Project

### 1 Install dependencies
pip install -r requirements.txt


### 2 Run Holiday Feature Engineering
python src/add_holidays_to_dataset.py


### 3 Run Weather Feature Engineering
python src/weather_enrichment.py


### 4 Run Model Training
python src/ETA_Delay_Prediction.py


---

# Learning Outcomes

- Feature engineering for logistics data
- Integration of external APIs in ML pipelines
- Handling time-based features
- Building classification and regression models
- Creating structured ML project repositories

---

# Future Improvements

- Use real historical weather APIs
- Add route-level geographic features
- Build real-time ETA prediction API
- Deploy the model using FastAPI or Flask