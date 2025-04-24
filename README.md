Flex Living Pricing Tool - Project Summary
This document summarizes the initial development stage of the Flex Living Pricing Tool project. So far, the focus 
has been on preparing the code environment, exploring Selenium for future automation, and designing a structure for 
extracting and organizing Airbnb data. While actual scraping has not yet been implemented, significant progress has been made in planning and setup.
Notebook Breakdown
Example Code Block:
# Load CSV files with bad line handling
listings_df = pd.read_csv('listings.csv', on_bad_lines='skip')
calendar_df = pd.read_csv('calendar.csv', on_bad_lines='skip')

# Check shapes and preview data
Example Code Block:
# Clean price in listings_df
listings_df['price'] = listings_df['price'].replace('[\$,£,]', '', regex=True).astype(float)

# Clean price in calendar_df
calendar_df['price'] = calendar_df['price'].replace('[\$,£,]', '', regex=True)
Example Code Block:
# Select important features from listings_df
listings_simple = listings_df[[
    'id', 'name', 'neighbourhood_cleansed', 'latitude', 'longitude',
    'price', 'bedrooms', 'bathrooms'
]].rename(columns={'id': 'listing_id'})
Example Code Block:
# Merge on listing_id
merged_df = pd.merge(calendar_df, listings_simple, on='listing_id', how='inner')

# Drop unavailable rows
merged_df = merged_df[merged_df['available'] == 't']  # Keep only available nights
Example Code Block:
# Convert 'date' to datetime
merged_df['date'] = pd.to_datetime(merged_df['date'])

# Add new date-based features
merged_df['month'] = merged_df['date'].dt.month
Example Code Block:
# Fill any missing bedrooms/bathrooms with 0 (to avoid NaNs)
merged_df['bedrooms'] = merged_df['bedrooms'].fillna(0).astype(int)
merged_df['bathrooms'] = merged_df['bathrooms'].fillna(0).astype(int)

# Create a property class string
Example Code Block:
print("Unique property classes:", merged_df['property_class'].nunique())
display(merged_df[['property_class', 'price_x', 'month', 'is_weekend']].head())
Example Code Block:
# Features for price model
price_features = merged_df[[
    'bedrooms', 'bathrooms', 'neighbourhood_cleansed', 'month', 'day_of_week', 'is_weekend'
]]

Example Code Block:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Train/test split
Example Code Block:
import pandas as pd

# Step 1: Re-merge calendar and listings (NO filtering for availability)
merged_df_all = pd.merge(calendar_df, listings_simple, on='listing_id', how='inner')

Example Code Block:
print("✅ Full merged dataset shape:", merged_df_all.shape)
display(merged_df_all[['date', 'price_x', 'bedrooms', 'bathrooms', 'neighbourhood_cleansed', 'is_booked']].head())
Example Code Block:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Feature set
Example Code Block:
# Recreate the property_class column
merged_df_all['property_class'] = (
    merged_df_all['neighbourhood_cleansed'] + '-' +
    merged_df_all['bedrooms'].astype(str) + 'bed-' +
    merged_df_all['bathrooms'].astype(str) + 'bath'
Example Code Block:
# ✅ Only include listings with real bedroom and bathroom values
valid_rows = merged_df_all[(merged_df_all['bedrooms'] > 0) & (merged_df_all['bathrooms'] > 0)]

# ✅ Group by property class and take the first valid entry
forecast_samples = valid_rows.groupby('property_class').first().reset_index()
Example Code Block:
from datetime import datetime, timedelta

# Generate future dates (1 year ahead)
future_dates = pd.date_range(start=datetime.today(), periods=365)

Example Code Block:
# Step 1: Add time-based features
forecast_df['month'] = forecast_df['date'].dt.month
forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
forecast_df['is_weekend'] = forecast_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

Example Code Block:
# ✅ Fix date format for Excel
forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')

# ✅ Save fixed CSV
forecast_df.to_csv('12_month_forecast_results.csv', index=False)
Example Code Block:
import pandas as pd
import matplotlib.pyplot as plt

# Load forecast file
forecast_df = pd.read_csv('12_month_forecast_results.csv')
Example Code Block:
# Group by month
monthly_occ = forecast_df.groupby('month')['predicted_occupancy'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 5))
Example Code Block:
# Top 10 most expensive neighbourhoods
top_areas = (
    forecast_df.groupby('neighbourhood_cleansed')['predicted_price']
    .mean()
    .sort_values(ascending=False)
Example Code Block:
# Group by bedroom count
price_by_bed = forecast_df.groupby('bedrooms')['predicted_price'].mean().reset_index()

# Plot
plt.figure(figsize=(8, 5))
Example Code Block:
# Pick one sample listing_id
sample_id = forecast_df['listing_id'].unique()[0]

# Filter and sort by date
sample_listing = forecast_df[forecast_df['
'] == sample_id].sort_values('date')
Example Code Block:
# Plot occupancy trend for same sample listing
plt.figure(figsize=(10, 3))
plt.plot(sample_listing['date'], sample_listing['predicted_occupancy'], color='darkred')
plt.title(f"Predicted Occupancy Trend for Listing ID {sample_id}")
plt.xlabel("Date")



