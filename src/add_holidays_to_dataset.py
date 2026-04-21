import pandas as pd
import requests
from datetime import timedelta

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Data/Delivery_Logistics_reconstructed.csv")

print("Dataset Loaded")
print("Shape:", df.shape)

# ==========================================
# 2. FIX DATE COLUMNS
# ==========================================

# Fix time format (13.00 -> 13:00)
df['order_ts_recon'] = df['order_ts_recon'].astype(str).str.replace('.', ':', regex=False)
df['expected_ts_recon'] = df['expected_ts_recon'].astype(str).str.replace('.', ':', regex=False)

# Convert to datetime
df['order_ts_recon'] = pd.to_datetime(df['order_ts_recon'], dayfirst=True)
df['expected_ts_recon'] = pd.to_datetime(df['expected_ts_recon'], dayfirst=True)

# Extract only date
df['order_date'] = df['order_ts_recon'].dt.date
df['expected_date'] = df['expected_ts_recon'].dt.date

print("Datetime columns fixed")

# ==========================================
# 3. DETECT YEARS IN DATASET
# ==========================================

years = df['order_ts_recon'].dt.year.unique()

print("Years present in dataset:", years)

# ==========================================
# 4. FETCH HOLIDAYS FROM API
# ==========================================

API_KEY = "fII9EVHQHfGmAQdDdLJH9ClfA4yPvsfl"

all_holidays = []

for year in years:

    print(f"Fetching holidays for {year}")

    url = f"https://calendarific.com/api/v2/holidays?api_key={API_KEY}&country=IN&year={year}"

    response = requests.get(url)
    data = response.json()

    for h in data['response']['holidays']:

        all_holidays.append({
            "date": h['date']['iso'][:10],
            "holiday_name": h['name']
        })

holiday_df = pd.DataFrame(all_holidays)

holiday_df['date'] = pd.to_datetime(holiday_df['date']).dt.date

print("Holiday data fetched")

# ==========================================
# 5. SAVE HOLIDAYS LOCALLY (CACHE)
# ==========================================

holiday_df.to_csv("Data/india_holidays_cached.csv", index=False)

print("Holiday data saved locally")

# ==========================================
# 6. CREATE HOLIDAY DICTIONARY
# ==========================================

holiday_dict = dict(zip(holiday_df['date'], holiday_df['holiday_name']))

# ==========================================
# 7. FUNCTION: HOLIDAYS BETWEEN DATES
# ==========================================

def get_transit_holidays(start, end):

    holidays_between = []
    current = start

    while current <= end:

        if current in holiday_dict:
            holidays_between.append(holiday_dict[current])

        current += timedelta(days=1)

    return holidays_between


# ==========================================
# 8. FUNCTION: WEEKEND COUNT
# ==========================================

def count_weekends(start, end):

    weekend_count = 0
    current = start

    while current <= end:

        if current.weekday() >= 5:  # Saturday or Sunday
            weekend_count += 1

        current += timedelta(days=1)

    return weekend_count


# ==========================================
# 9. FUNCTION: HOLIDAY PROXIMITY
# ==========================================

def holiday_proximity(order_date):

    min_gap = 999

    for h_date in holiday_dict.keys():

        gap = abs((h_date - order_date).days)

        if gap < min_gap:
            min_gap = gap

    return min_gap


# ==========================================
# 10. APPLY HOLIDAY FEATURES
# ==========================================

print("Creating transit holiday features")

df['transit_holidays'] = df.apply(
    lambda x: get_transit_holidays(x['order_date'], x['expected_date']),
    axis=1
)

df['holiday_count_transit'] = df['transit_holidays'].apply(len)

df['holiday_names_transit'] = df['transit_holidays'].apply(
    lambda x: ", ".join(x) if len(x) > 0 else "None"
)

# ==========================================
# 11. WEEKEND FEATURES
# ==========================================

print("Creating weekend features")

df['weekend_count_transit'] = df.apply(
    lambda x: count_weekends(x['order_date'], x['expected_date']),
    axis=1
)

df['holiday_or_weekend_transit_flag'] = (
    (df['holiday_count_transit'] > 0) |
    (df['weekend_count_transit'] > 0)
).astype(int)

# ==========================================
# 12. HOLIDAY PROXIMITY FEATURE
# ==========================================

print("Creating holiday proximity feature")

df['holiday_proximity_feature'] = df['order_date'].apply(holiday_proximity)

# ==========================================
# 13. SAVE FINAL DATASET
# ==========================================

df.to_csv("Data/Added_Holiday_Features_dataset.csv", index=False)

print("Feature engineering completed")

print("New features added:")

print([
    "holiday_count_transit",
    "holiday_names_transit",
    "weekend_count_transit",
    "holiday_or_weekend_transit_flag",
    "holiday_proximity_feature"
])