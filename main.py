# ===================================================================
# AQI Streamlit Dashboard (aligned with optimized inference pipeline)
# ===================================================================

import streamlit as st
import pandas as pd
import hopsworks
import os
from dotenv import load_dotenv
import plotly.express as px

# --- Load environment variables ---
load_dotenv()
API_KEY = os.getenv("HOPSWORKS_API_KEY")

# --- Connect to Hopsworks ---
project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
fs = project.get_feature_store()

# Load predictions feature group
fg = fs.get_feature_group(name="aqi_predictions", version=1)

# Read as dataframe
df = fg.read()

# --- Standardize columns (as per inference pipeline) ---
df = df.rename(columns={
    "datetime_utc": "forecast_date_utc",
    "datetime": "forecast_date_local",   # already tz-aware Asia/Karachi in pipeline
    "predicted_us_aqi": "us_aqi",
    "prediction_date": "prediction_time",
    "model_version": "model_version",
})

# --- Parse datetimes safely ---
for col in ["forecast_date_utc", "forecast_date_local", "prediction_time"]:
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

# Convert to Asia/Karachi for local forecast column
df["forecast_date_local"] = df["forecast_date_utc"].dt.tz_convert("Asia/Karachi")
df["prediction_time"] = df["prediction_time"].dt.tz_convert("Asia/Karachi")

# --- Keep only latest prediction run ---
latest_run_time = df["prediction_time"].max()
latest_preds = df[df["prediction_time"] == latest_run_time].copy()
latest_preds = latest_preds.sort_values("forecast_date_utc")

# --- Ensure integer AQI ---
latest_preds["us_aqi"] = latest_preds["us_aqi"].round().astype(int)

# --- AQI category ---
def aqi_category(aqi: int) -> str:
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
st.title("🌍 Rawalpindi AQI Forecast Dashboard")
st.write("Showing the latest **74-hour forecast** from model (times in Asia/Karachi · PKT)")

# AQI legend
st.markdown(
    """
**AQI Categories:**
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

# --- Time range selectors ---
all_ts = list(latest_preds["forecast_date_local"].unique())
all_ts.sort()

if all_ts:
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

    if start_ts > end_ts:
        st.warning("Start time is after end time — swapping them.")
        start_ts, end_ts = end_ts, start_ts

    # Filter predictions
    mask = (latest_preds["forecast_date_local"] >= start_ts) & (latest_preds["forecast_date_local"] <= end_ts)
    filtered = latest_preds.loc[mask].reset_index(drop=True)
    filtered.index += 1
    filtered.index.name = "Index"

    st.subheader("Filtered Predictions")
    st.dataframe(filtered[["forecast_date_local", "us_aqi", "category"]])

    # --- Current AQI (closest to now) ---
    now = pd.Timestamp.utcnow().tz_convert("Asia/Karachi")
    latest_preds["time_diff"] = (latest_preds["forecast_date_local"] - now).abs()
    current_row = latest_preds.loc[latest_preds["time_diff"].idxmin()]

    st.metric(
        "Current Forecasted AQI",
        f"{int(current_row['us_aqi'])}",
        help=f"Forecasted for {current_row['forecast_date_local']}"
    )

    # --- Plot ---
    if not filtered.empty:
        fig = px.line(
            filtered,
            x="forecast_date_local",
            y="us_aqi",
            markers=True,
            title="74-Hour AQI Forecast (Latest Run)",
            labels={"forecast_date_local": "Forecast Time", "us_aqi": "AQI"},
            hover_data={"forecast_date_local": True, "us_aqi": True, "category": True}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected time range.")
else:
    st.warning("No predictions available.")
