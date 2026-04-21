#!/usr/bin/env python
# coding: utf-8

# # **IMPORT LIBRARIES**

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)


# # **LOAD DATASET**

# In[8]:


file_path = "Delivery_Logistics.csv"


# In[6]:


import os
print("Current directory:", os.getcwd())
print("\nFiles in current directory:")
print(os.listdir())


# In[7]:


# Check if 'data' folder exists
if os.path.exists('data'):
    print("'data' folder found!")
    print("\nFiles in 'data' folder:")
    print(os.listdir('data'))
else:
    print("No 'data' folder found in Downloads")


# In[9]:


df = pd.read_csv(file_path)
print("Dataset Loaded Successfully")
print("Shape of dataset:", df.shape)


# # **SCHEMA & STRUCTURE**

# In[10]:


print("Column Names:")
print(df.columns.tolist())


# In[11]:


print("Dataset Schema Information:")
df.info()


# In[12]:


df.head()


# # **MISSING VALUES ANALYSIS**

# In[13]:


# Calculate missing values count
missing_count = df.isnull().sum()


# In[14]:


# Calculate missing percentage
missing_percentage = (missing_count / len(df)) * 100


# In[15]:


# Missing values summary
missing_summary = pd.DataFrame({
    'Missing_Count': missing_count,
    'Missing_Percentage': missing_percentage
}).sort_values(by='Missing_Percentage', ascending=False)

missing_summary


# # **Duplicate Records Check**

# In[16]:


# Total number of duplicate rows
duplicate_count = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicate_count}")


# In[17]:


# Percentage of duplicates
duplicate_percentage = (duplicate_count / len(df)) * 100
print(f"Percentage of duplicate rows: {duplicate_percentage:.2f}%")


# # **NUMERICAL SUMMARY**

# In[18]:


numeric_summary = df.describe().T
numeric_summary


# # **NUMERICAL DISTRIBUTIONS**

# In[19]:


# Select only numeric columns from dataset
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

print("Numeric Columns:", numeric_columns)


# In[20]:


colors = ['steelblue', 'seagreen', 'darkorange', 'slateblue', 'indianred']

for i, col in enumerate(numeric_columns):
    plt.figure(figsize=(8, 5))
    plt.hist(
        df[col],
        bins=30,
        color=colors[i % len(colors)],
        edgecolor='black',
        alpha=0.8,
        label=col
    )
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# # **CATEGORICAL SUMMARY**

# In[21]:


#Percentage of total records belonging to each category.
categorical_columns = df.select_dtypes(include='object').columns

for col in categorical_columns:
    print(f"\nColumn: {col}")
    print("Unique values:", df[col].nunique())
    print(df[col].value_counts(normalize=True).head(10) * 100)


# # **CATEGORICAL DISTRIBUTIONS**

# In[22]:


categorical_columns = df.select_dtypes(include='object').columns

for col in categorical_columns:
    value_counts = df[col].value_counts()

    # BAR CHART

    plt.figure()
    value_counts.plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.show()


    # PIE CHART
    plt.figure()
    value_counts.head(6).plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'{col} – Top Categories')
    plt.ylabel('')
    plt.show()


# # **WEATHER vs DELIVERY STATUS**

# In[23]:


weather_delay = pd.crosstab(
    df['weather_condition'],
    df['delivery_status'],
    normalize='index'
) * 100

weather_delay.plot(kind='bar', figsize=(8, 5))
plt.title('Weather Impact on Delivery Status')
plt.xlabel('Weather Condition')
plt.ylabel('Percentage of Deliveries')
plt.legend(title='Delivery Status')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# # **REGION-WISE DELAY ANALYSIS**

# In[24]:


region_delay_pct = (
    df.groupby("region")["delayed"]
    .apply(lambda x: (x == "yes").mean() * 100)
    .sort_values(ascending=False)
)

print(region_delay_pct)


region_delay_pct.plot(kind="bar", figsize=(9,5), edgecolor="black")

plt.title("Region-wise Delay Percentage")
plt.xlabel("Region")
plt.ylabel("Delayed Deliveries (%)")
plt.grid(axis="y", alpha=0.3)
plt.show()


# # **VEHICLE TYPE vs DELAY**

# In[25]:


vehicle_delay = (
    df.groupby("vehicle_type")["delayed"]
    .apply(lambda x: (x == "yes").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))
vehicle_delay.plot(kind="bar", color="slateblue", edgecolor="black")
plt.title("Vehicle Type-wise Delay Percentage", fontsize=14)
plt.xlabel("Vehicle Type", fontsize=12)
plt.ylabel("Delayed Deliveries (%)", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# # **DISTANCE vs DELAY**

# 

# In[26]:


plt.figure(figsize=(7, 5))

df.boxplot(
    column='distance_km',
    by='delayed',
    grid=False
)

plt.title('Distance vs Delivery Delay')
plt.suptitle('')
plt.xlabel('Delayed (yes = delayed, no = on-time)')
plt.ylabel('Distance (km)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# # **COLUMN-WISE OUTLIER DETECTION**

# In[27]:


outlier_columns = [
    "distance_km",
    "delivery_cost",
    "delivery_rating",
    "package_weight_kg"
]

for col in outlier_columns:
    plt.figure(figsize=(8, 4))

    plt.boxplot(
        df[col],
        vert=False,
        patch_artist=True
    )

    plt.title(f"Outlier Analysis: {col}")
    plt.xlabel(col)
    plt.grid(axis="x", alpha=0.3)
    plt.show()


# # **CORRELATION HEATMAP**

# In[28]:


heatmap_columns = [
    'distance_km',
    'package_weight_kg',
    'delivery_cost',
    'delivery_rating'
]

corr_matrix = df[heatmap_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap (Numeric Features)", fontsize=14)
plt.tight_layout()
plt.show()


# # **NUMERIC vs TARGET**

# In[29]:


numeric_vs_target = [
    'delivery_cost',
    'package_weight_kg',
    'delivery_rating'
]

for col in numeric_vs_target:
    plt.figure(figsize=(7, 5))

    df.boxplot(
        column=col,
        by='delayed',   # existing column
        grid=False
    )

    plt.title(f'{col} vs Delivery Delay Status')
    plt.suptitle('')
    plt.xlabel('Delayed (yes = delayed, no = on-time)')
    plt.ylabel(col)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# # **Feature engineering**

# Create Binary Target (delayed_flag)

# In[30]:


# Clean delayed column
df["delayed"] = df["delayed"].str.strip().str.lower()

# Create delayed_flag (0 = On-time, 1 = Delayed)
df["delayed_flag"] = (df["delayed"] == "yes").astype(int)

df[["delayed", "delayed_flag"]].head()


# Create Bad Weather Flag

# In[31]:


bad_weather_list = ["rainy", "stormy", "foggy"]

df["bad_weather_flag"] = df["weather_condition"].apply(
    lambda x: 1 if x in bad_weather_list else 0
)

df[["weather_condition", "bad_weather_flag"]].head()


# Weather Severity Score

# In[32]:


weather_map = {
    "clear": 0,
    "cloudy": 1,
    "hot": 1,
    "cold": 1,
    "rainy": 2,
    "foggy": 3,
    "stormy": 4
}

df["weather_severity"] = df["weather_condition"].map(weather_map)

df[["weather_condition", "weather_severity"]].head()


# Cost Efficiency Feature (cost_per_km)

# In[33]:


df["cost_per_km"] = df["delivery_cost"] / df["distance_km"]

df[["delivery_cost", "distance_km", "cost_per_km"]].head()


# Weight Load Feature (weight_per_km)

# In[34]:


df["weight_per_km"] = df["package_weight_kg"] / df["distance_km"]

df[["package_weight_kg", "distance_km", "weight_per_km"]].head()


# Partner Delay Rate (Data-Driven Feature)

# In[35]:


partner_delay = df.groupby("delivery_partner")["delayed_flag"].mean()

df["partner_delay_rate"] = df["delivery_partner"].map(partner_delay)

df[["delivery_partner", "partner_delay_rate"]].head()


# Region Delay Rate

# In[36]:


region_delay = df.groupby("region")["delayed_flag"].mean()

df["region_delay_rate"] = df["region"].map(region_delay)

df[["region", "region_delay_rate"]].head()


# Vehicle Delay Rate

# In[37]:


vehicle_delay = df.groupby("vehicle_type")["delayed_flag"].mean()

df["vehicle_delay_rate"] = df["vehicle_type"].map(vehicle_delay)

df[["vehicle_type", "vehicle_delay_rate"]].head()


# Delivery Mode Delay Rate

# In[38]:


mode_delay = df.groupby("delivery_mode")["delayed_flag"].mean()

df["mode_delay_rate"] = df["delivery_mode"].map(mode_delay)

df[["delivery_mode", "mode_delay_rate"]].head()


# In[40]:


# Load the feature-engineered dataset
df_new = pd.read_csv("delivery_feature_engineered.csv")
print("Feature-engineered dataset loaded successfully!")
print("Shape:", df_new.shape)
df_new.head()


# In[41]:


# Check all columns and data types
print("Columns in the dataset:")
print(df_new.columns.tolist())
print("\nData types:")
print(df_new.dtypes)


# In[42]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the feature-engineered dataset
df = pd.read_csv("delivery_feature_engineered.csv")
print("✅ Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
df.head()


# In[47]:


# Basic information
print(" Dataset Info:")
print(df.info())
print("\n Missing Values:")
print(df.isnull().sum())
print("\n Target Variable (delayed_flag) Distribution:")
print(df['delayed_flag'].value_counts())
print(f"\nDelay Rate: {df['delayed_flag'].mean()*100:.2f}%")


# In[48]:


# Drop columns that shouldn't be used for prediction
# 'delayed' and 'delivery_status' are related to the target, so we drop them
columns_to_drop = ['delayed_flag', 'delayed', 'delivery_status']

X = df.drop(columns_to_drop, axis=1)
y = df['delayed_flag']

print(f" Features shape: {X.shape}")
print(f" Target shape: {y.shape}")
print(f"\nFeatures being used: {X.columns.tolist()}")


# In[49]:


# Identify categorical columns (object type)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns to encode: {categorical_cols}")

# Encode categorical variables
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le
    print(f"   ✓ Encoded {col}: {len(le.classes_)} unique values")

print("\n All categorical variables encoded!")
X_encoded.head()


# In[50]:


# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, 
    test_size=0.2,       # 20% for testing
    random_state=42,     # For reproducibility
    stratify=y           # Keep same proportion of delayed/not delayed
)

print(" Data split completed!")
print(f"\n Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

print(f"\nTraining target distribution:")
print(y_train.value_counts())
print(f"   Delay rate: {y_train.mean()*100:.2f}%")

print(f"\nTesting target distribution:")
print(y_test.value_counts())
print(f"   Delay rate: {y_test.mean()*100:.2f}%")


# In[51]:


# Scale the features (standardization: mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(" Features scaled successfully!")
print(f"Training data shape: {X_train_scaled.shape}")
print(f"Testing data shape: {X_test_scaled.shape}")
print(f"\nScaling example (first feature):")
print(f"   Before: mean={X_train.iloc[:, 0].mean():.2f}, std={X_train.iloc[:, 0].std():.2f}")
print(f"   After: mean={X_train_scaled[:, 0].mean():.2f}, std={X_train_scaled[:, 0].std():.2f}")

