# ===================================================================
# AQI Streamlit Dashboard - FIXED VERSION (No Duplicate Checkbox)
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pytz

# Set page config
st.set_page_config(
    page_title="Rawalpindi AQI Forecast",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load environment variables ---
load_dotenv()
API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not API_KEY:
    st.error("❌ HOPSWORKS_API_KEY environment variable not set!")
    st.stop()

# --- Helper Functions ---
@st.cache_data(ttl=300)
def load_predictions_data():
    try:
        project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="aqi_predictions", version=1)
        df = fg.read()
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_backup_csv():
    csv_files = [
        'predictions_20250824_204610.csv',
        'predictions.csv', 
        'latest_predictions.csv',
        'aqi_predictions.csv'
    ]
    for filename in csv_files:
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                st.info(f"📁 Loaded data from local file: {filename}")
                return df, None
        except Exception:
            continue
    return None, "No CSV backup files found"

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

def aqi_color(aqi: int) -> str:
    if aqi <= 50:
        return "#00E400"
    elif aqi <= 100:
        return "#FFFF00"
    elif aqi <= 150:
        return "#FF7E00"
    elif aqi <= 200:
        return "#FF0000"
    elif aqi <= 300:
        return "#8F3F97"
    return "#7E0023"

def format_timestamp_pkt(ts):
    if pd.isna(ts):
        return "N/A"
    pkt_tz = pytz.timezone('Asia/Karachi')
    if ts.tz is None:
        ts = pytz.utc.localize(ts)
    pkt_time = ts.astimezone(pkt_tz)
    return pkt_time.strftime("%Y-%m-%d %H:%M:%S PKT")

# --- Main App ---
def main():
    st.title("🌍 Rawalpindi AQI Forecast Dashboard")
    st.markdown("Real-time **74-hour AQI forecasts** powered by LightGBM ML model")

    # Sidebar controls
    st.sidebar.header("📊 Dashboard Controls")
    data_source = st.sidebar.selectbox(
        "📂 Data Source",
        options=["Feature Store", "Local CSV Backup"],
        index=0
    )

    # ✅ Only one checkbox here
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5 min)", value=False, key="auto_refresh")

    if auto_refresh:
        st.experimental_autorefresh(interval=5 * 60 * 1000, key="refresh_counter")

    # Load data
    with st.spinner("🔄 Loading latest predictions..."):
        if data_source == "Feature Store":
            df, error = load_predictions_data()
        else:
            df, error = load_backup_csv()

    if error:
        st.error(f"❌ Failed to load data: {error}")
        return

    if df is None or len(df) == 0:
        st.warning("⚠️ No prediction data available.")
        return

    # --- Process Data (simplified for demo) ---
    if 'us_aqi' not in df.columns:
        st.error("❌ No AQI column found!")
        return

    df['us_aqi'] = pd.to_numeric(df['us_aqi'], errors='coerce').fillna(0).round().astype(int)
    df['category'] = df['us_aqi'].apply(aqi_category)
    df['color'] = df['us_aqi'].apply(aqi_color)

    # Sort by time if present
    if 'forecast_date_utc' in df.columns:
        df['forecast_date_utc'] = pd.to_datetime(df['forecast_date_utc'], errors='coerce')
        df = df.sort_values('forecast_date_utc')

    # --- Display ---
    st.metric("📊 Total Forecasts", len(df))
    st.write(df.head())

    st.markdown("---")
    st.markdown(f"Last updated: {datetime.now(pytz.timezone('Asia/Karachi')).strftime('%Y-%m-%d %H:%M:%S PKT')}")

# Run app
if __name__ == "__main__":
    main()
