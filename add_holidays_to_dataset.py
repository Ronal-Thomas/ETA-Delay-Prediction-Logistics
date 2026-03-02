import pandas as pd
import argparse
import holidays

# -----------------------------
# Read command line arguments
# -----------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--date-cols", required=True)
parser.add_argument("--country", default="IN")
parser.add_argument("--subdivision", default=None)
parser.add_argument("--weekend-flag", action="store_true")

args = parser.parse_args()

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv(args.input)

date_cols = args.date_cols.split(",")

# -----------------------------
# Create holiday calendar
# -----------------------------
print("Creating holiday calendar...")

years = pd.to_datetime(df[date_cols[0]], dayfirst=True).dt.year.unique()

holiday_calendar = holidays.country_holidays(
    args.country,
    subdiv=args.subdivision,
    years=years
)

# -----------------------------
# Add holiday features
# -----------------------------
for col in date_cols:

    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    df[f"is_holiday_{col}"] = df[col].apply(
        lambda x: 1 if x in holiday_calendar else 0
    )

    df[f"holiday_name_{col}"] = df[col].apply(
        lambda x: holiday_calendar.get(x) if x in holiday_calendar else None
    )

    if args.weekend_flag:
        df[f"is_weekend_{col}"] = df[col].dt.dayofweek.isin([5,6]).astype(int)

# -----------------------------
# Save output
# -----------------------------
print("Saving enriched dataset...")
df.to_csv(args.output, index=False)

print(" Holiday enrichment completed!")

