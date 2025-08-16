import streamlit as st
import pandas as pd
import hopsworks
import os
from dotenv import load_dotenv

# --- Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("HOPSWORKS_API_KEY")

# --- Connect to Hopsworks ---
project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
fs = project.get_feature_store()

# Load predictions feature group
fg = fs.get_feature_group(name="aqi_predictions", version=1)

# Read as dataframe
df = fg.read()

# --- Rename columns ---
df = df.rename(columns={
    "datetime": "forecast_date",
    "datetime_utc": "forecast_date_utc",
    "predicted_us_aqi": "us_aqi",
    "prediction_date": "prediction_time",
    "model_version": "model_version",
})

# --- Parse and convert to Asia/Karachi robustly ---
# Using utc=True makes both naive and tz-aware timestamps end up tz-aware, then we convert to PKT
for col in ["forecast_date", "forecast_date_utc", "prediction_time"]:
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert("Asia/Karachi")

# --- Keep only latest prediction run ---
latest_run_time = df["prediction_time"].max()
latest_preds = df[df["prediction_time"] == latest_run_time].copy()
latest_preds = latest_preds.sort_values("forecast_date")

# --- Helper: AQI category ---
def aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"

latest_preds["category"] = latest_preds["us_aqi"].apply(aqi_category)

# --- UI ---
st.title("🌍 AQI Forecast Dashboard")
st.write("Showing the latest forecast from model (times in Asia/Karachi · PKT)")

# AQI legend
st.markdown(
    """
**Note:** The United States Air Quality Index (AQI) ranges are:
- 0–50: Good 🟢
- 51–100: Moderate 🟡
- 101–150: Unhealthy for Sensitive Groups 🟠
- 151–200: Unhealthy 🔴
- 201–300: Very Unhealthy 🟣
- 301–500: Hazardous ⚫
"""
)

# Latest run info
st.info(f"Latest prediction run: {latest_run_time}")

# --- Time range selectors (tz-aware objects, no string parsing) ---
all_ts = list(latest_preds["forecast_date"].unique())
all_ts.sort()

col1, col2 = st.columns(2)
with col1:
    start_ts = st.selectbox(
        "Start time",
        options=all_ts,
        index=0,
        format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )
with col2:
    end_ts = st.selectbox(
        "End time",
        options=all_ts,
        index=len(all_ts) - 1,
        format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )

# Ensure valid order
if start_ts > end_ts:
    st.warning("Start time is after end time — swapping them.")
    start_ts, end_ts = end_ts, start_ts

# Filter and display
mask = (latest_preds["forecast_date"] >= start_ts) & (latest_preds["forecast_date"] <= end_ts)
filtered = latest_preds.loc[mask]

st.subheader("Filtered predictions")
st.dataframe(filtered[["forecast_date", "us_aqi", "category"]])

if not filtered.empty:
    # Latest point metric (within selected range)
    latest_row = filtered.iloc[-1]
    st.metric("Latest AQI in range", f"{latest_row['us_aqi']:.0f}", help=latest_row["category"]) 

    st.line_chart(filtered.set_index("forecast_date")["us_aqi"])
else:
    st.warning("No data available for the selected time range.")
