import streamlit as st
import pandas as pd
import hopsworks
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("HOPSWORKS_API_KEY")

# === Connect to Hopsworks ===
project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
fs = project.get_feature_store()

# Load the predictions feature group
fg = fs.get_feature_group(name="aqi_predictions", version=1)

# Read as dataframe
df = fg.read()

# === Rename columns ===
df = df.rename(columns={
    "datetime": "forecast_date",
    "datetime_utc": "forecast_date_utc",
    "predicted_us_aqi": "us_aqi",
    "prediction_date": "prediction_time",
    "model_version": "model_version"
})

# Ensure datetime is parsed
df['forecast_date'] = pd.to_datetime(df['forecast_date'])
df['forecast_date_utc'] = pd.to_datetime(df['forecast_date_utc'])
df['prediction_time'] = pd.to_datetime(df['prediction_time'])

# === Convert to Pakistan timezone (Asia/Karachi) ===
df['forecast_date'] = df['forecast_date'].dt.tz_convert("Asia/Karachi")
df['forecast_date_utc'] = df['forecast_date_utc'].dt.tz_convert("Asia/Karachi")
df['prediction_time'] = df['prediction_time'].dt.tz_convert("Asia/Karachi")

# === Keep only the latest prediction run ===
latest_run_time = df['prediction_time'].max()
latest_preds = df[df['prediction_time'] == latest_run_time]

# Sort by forecast_date
latest_preds = latest_preds.sort_values("forecast_date")

# === Streamlit UI ===
st.title("🌍 AQI Forecast Dashboard")
st.write("Showing the latest forecast from model")

# Show AQI explanation
st.markdown("""
**Note:** The United States Air Quality Index (AQI) ranges are:
- 0-50: Good 🟢
- 51-100: Moderate 🟡
- 101-150: Unhealthy for Sensitive Groups 🟠
- 151-200: Unhealthy 🔴
- 201-300: Very Unhealthy 🟣
- 301-500: Hazardous ⚫
""")

# Show latest prediction date/time
st.info(f"Latest prediction run: {latest_run_time}")

# === Add timestamp selection ===
all_timestamps = latest_preds['forecast_date'].dt.strftime('%Y-%m-%d %H:%M:%S').unique()

col1, col2 = st.columns(2)
with col1:
    start_date = st.selectbox("Select start time:", all_timestamps)
with col2:
    end_date = st.selectbox("Select end time:", all_timestamps, index=len(all_timestamps)-1)

# Filter by user selection
mask = (latest_preds['forecast_date'] >= pd.to_datetime(start_date)) & (latest_preds['forecast_date'] <= pd.to_datetime(end_date))
filtered_preds = latest_preds[mask]

# Display table
st.dataframe(filtered_preds[['forecast_date', 'us_aqi']])

# Plot chart if filtered data exists
if not filtered_preds.empty:
    st.line_chart(filtered_preds.set_index("forecast_date")["us_aqi"])
else:
    st.warning("No data available for the selected time range.")
