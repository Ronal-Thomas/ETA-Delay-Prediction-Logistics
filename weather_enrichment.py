# ==========================================
# WEATHER ENRICHMENT SCRIPT
# ETA Delay Prediction Project
# ==========================================

import pandas as pd
import requests

# ------------------------------------------
# 1. API CONFIGURATION
# ------------------------------------------

API_KEY = "4d80bee4da9b907635792d27e8575a75"   # <-- paste your OpenWeatherMap key

# Default operational location (assumed)
CITY = "Bangalore"
LAT = 12.9716
LON = 77.5946

# ------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------

print("Loading dataset...")

df = pd.read_csv("dataset_with_holidays.csv")

print("Rows:", len(df))

# ------------------------------------------
# 3. WEATHER FETCH FUNCTION (FREE API)
# ------------------------------------------

def get_weather():

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "lat": LAT,
        "lon": LON,
        "appid": API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            print("API Error:", response.status_code)
            return None

    except Exception as e:
        print("Connection Error:", e)
        return None


# ------------------------------------------
# 4. FETCH WEATHER DATA
# ------------------------------------------

print("Fetching weather data...")

weather = get_weather()

if weather is None:
    raise Exception("Weather API failed. Check API key or internet.")

# Extract fields
main = weather["main"]
wind = weather["wind"]

temperature = main.get("temp")
humidity = main.get("humidity")
wind_speed = wind.get("speed")
visibility = weather.get("visibility", None)

print("Weather Snapshot:")
print("Temperature:", temperature)
print("Humidity:", humidity)
print("Wind Speed:", wind_speed)
print("Visibility:", visibility)

# ------------------------------------------
# 5. ADD WEATHER FEATURES TO DATASET
# ------------------------------------------

# Assign same snapshot to all rows (approximation)
df["api_temperature"] = temperature
df["api_humidity"] = humidity
df["api_wind_speed"] = wind_speed
df["api_visibility"] = visibility

# ------------------------------------------
# 6. WEATHER SEVERITY INDEX (ENGINEERED FEATURE)
# ------------------------------------------

print("Creating weather severity index...")

df["weather_severity_index_api"] = (
      df["api_wind_speed"] * 0.4
    + (1 / (df["api_visibility"] + 1)) * 0.6
)

# Bad weather flag
df["bad_weather_flag_api"] = (
    df["weather_severity_index_api"] > df["weather_severity_index_api"].median()
).astype(int)

# ------------------------------------------
# 7. SAVE ENRICHED DATASET
# ------------------------------------------

output_file = "dataset_weather_enriched.csv"

df.to_csv(output_file, index=False)

print(" Weather enrichment completed!")
print("Saved as:", output_file)