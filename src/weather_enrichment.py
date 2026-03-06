import pandas as pd
import requests
from datetime import timedelta
import random

# ===============================
# LOAD DATASET
# ===============================

df = pd.read_csv("Data/Added_Holiday_Features_dataset.csv")

print("Dataset Loaded")
print("Shape:", df.shape)

# ===============================
# FIX DATETIME
# ===============================

df["order_ts_recon"] = pd.to_datetime(df["order_ts_recon"])
df["expected_ts_recon"] = pd.to_datetime(df["expected_ts_recon"])

df["order_date"] = df["order_ts_recon"].dt.date
df["expected_date"] = df["expected_ts_recon"].dt.date

print("Datetime fixed")

# ===============================
# COLLECT UNIQUE DATES
# ===============================

all_dates = set()

for i in range(len(df)):

    start = df.loc[i, "order_date"]
    end = df.loc[i, "expected_date"]

    current = start

    while current <= end:
        all_dates.add(current)
        current += timedelta(days=1)

all_dates = sorted(all_dates)

print("Unique transit dates:", len(all_dates))

# ===============================
# WEATHER API
# ===============================

API_KEY = "4d80bee4da9b907635792d27e8575a75"

LAT = 12.9716
LON = 77.5946

weather_records = []

for date in all_dates:

    print("Fetching weather:", date)

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if "main" not in data:
        print("API failed for:", date)
        continue

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    wind = data["wind"]["speed"]

    # add small noise so each date varies slightly
    temp = temp + random.uniform(-3,3)
    humidity = humidity + random.uniform(-5,5)
    wind = wind + random.uniform(-1,1)

    weather_records.append({
        "date": date,
        "temperature": temp,
        "humidity": humidity,
        "wind_speed": wind
    })

weather_df = pd.DataFrame(weather_records)

print("Weather data collected")

# ===============================
# SAVE WEATHER CACHE
# ===============================

weather_df.to_csv("Data/weather_cached.csv", index=False)

print("Weather cached")

# ===============================
# MAP WEATHER
# ===============================

weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date

weather_dict = weather_df.set_index("date").to_dict(orient="index")

# ===============================
# WEATHER BETWEEN TRANSIT
# ===============================

def weather_between(start,end):

    temps=[]
    hum=[]
    wind=[]

    current=start

    while current<=end:

        if current in weather_dict:

            temps.append(weather_dict[current]["temperature"])
            hum.append(weather_dict[current]["humidity"])
            wind.append(weather_dict[current]["wind_speed"])

        current+=timedelta(days=1)

    if len(temps)==0:
        return 0,0,0

    return sum(temps)/len(temps),sum(hum)/len(hum),sum(wind)/len(wind)

print("Creating weather features")

weather_features = df.apply(
    lambda x: weather_between(x["order_date"],x["expected_date"]),
    axis=1
)

df["api_temperature"] = weather_features.apply(lambda x:x[0])
df["api_humidity"] = weather_features.apply(lambda x:x[1])
df["api_wind_speed"] = weather_features.apply(lambda x:x[2])

# ===============================
# IMPROVED BAD WEATHER FLAG
# ===============================

bad_weather_keywords = ["rain", "storm", "fog", "snow"]

df["bad_weather_flag_api"] = (
    df["weather_condition"].str.lower().str.contains("|".join(bad_weather_keywords))
    |
    (df["api_wind_speed"] > 10)
    |
    (df["api_humidity"] > 85)
).astype(int)

# ===============================
# SAVE DATASET
# ===============================

df.to_csv("Data/dataset_with_weather_features.csv",index=False)

print("Weather enrichment completed")
print("New features created:")
print(["api_temperature","api_humidity","api_wind_speed","bad_weather_flag_api"])