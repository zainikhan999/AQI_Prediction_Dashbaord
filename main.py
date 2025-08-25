# ===================================================================
# AQI Streamlit Dashboard - FIXED VERSION
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
    """Load predictions data from Hopsworks feature store."""
    try:
        project = hopsworks.login(api_key_value=API_KEY)
        fs = project.get_feature_store()

        fg_info = fs.get_feature_groups(name="aqi_forecast_metrics_fg")
        if not fg_info:
            return None, "Feature group 'aqi_forecast_metrics_fg' not found!"

        latest_version = max(fg.version for fg in fg_info)
        fg = fs.get_feature_group(name="aqi_forecast_metrics_fg", version=latest_version)

        if fg is None:
            return None, f"FG 'aqi_forecast_metrics_fg' (v{latest_version}) not found"

        df = fg.read()
        if df is None or len(df) == 0:
            return None, "Feature group is empty. Run inference pipeline first."

        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_backup_csv():
    """Try to load from local CSV backup files."""
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
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

def aqi_color(aqi: int) -> str:
    if aqi <= 50: return "#00E400"
    elif aqi <= 100: return "#FFFF00"
    elif aqi <= 150: return "#FF7E00"
    elif aqi <= 200: return "#FF0000"
    elif aqi <= 300: return "#8F3F97"
    return "#7E0023"

def format_timestamp_pkt(ts):
    if pd.isna(ts): return "N/A"
    pkt_tz = pytz.timezone('Asia/Karachi')
    if ts.tz is None:
        ts = pytz.utc.localize(ts)
    return ts.astimezone(pkt_tz).strftime("%Y-%m-%d %H:%M:%S PKT")

def format_timestamp_utc(ts):
    if pd.isna(ts): return "N/A"
    if ts.tz is None:
        ts = pytz.utc.localize(ts)
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")

# --- Main App ---
def main():
    st.title("🌍 Rawalpindi AQI Forecast Dashboard")
    st.markdown("Real-time **74-hour AQI forecasts** powered by LightGBM ML model")
    st.markdown("📍 **Location:** Rawalpindi, Punjab, Pakistan (33.5973°N, 73.0479°E)")

    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    data_source = st.sidebar.selectbox(
        "📂 Data Source", ["Feature Store", "Local CSV Backup"], index=0
    )
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5 min)", value=False)
    show_advanced = st.sidebar.checkbox("🔧 Advanced Options", value=False)

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
        st.info("🚀 Run the inference pipeline to generate predictions: `python inference_pipeline.py`")
        return

    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 **Records loaded:** {len(df)}")
    st.sidebar.info(f"📂 **Data source:** {data_source}")

    if st.sidebar.checkbox("🔍 Debug: Show column names"):
        st.sidebar.write(list(df.columns))

    # --- Column Mapping ---
    column_mapping = {
        'datetime': 'forecast_date_utc',
        'predicted_us_aqi': 'us_aqi',
        'us_aqi_forecast': 'us_aqi_baseline'
    }
    df = df.rename(columns=column_mapping)

    # --- Date handling ---
    if 'forecast_date_utc' in df.columns:
        df['forecast_date_utc'] = pd.to_datetime(df['forecast_date_utc'], errors='coerce')
        if df['forecast_date_utc'].dt.tz is None:
            df['forecast_date_utc'] = df['forecast_date_utc'].dt.tz_localize('UTC', nonexistent='NaT', ambiguous='NaT')
        else:
            df['forecast_date_utc'] = df['forecast_date_utc'].dt.tz_convert('UTC')
        df['forecast_date_pkt'] = df['forecast_date_utc'].dt.tz_convert('Asia/Karachi')

    # --- AQI handling ---
    if 'us_aqi' in df.columns:
        df['us_aqi'] = pd.to_numeric(df['us_aqi'], errors='coerce').fillna(0).round().astype(int)
        df['category'] = df['us_aqi'].apply(aqi_category)
        df['color'] = df['us_aqi'].apply(aqi_color)
    else:
        st.error("❌ No AQI column found in the data!")
        return

    latest_preds = df.copy()

    # Sort by forecast time
    if 'forecast_date_utc' in latest_preds.columns:
        latest_preds = latest_preds.sort_values('forecast_date_utc').reset_index(drop=True)

    # --- Dashboard Metrics ---
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    total_predictions = len(latest_preds)

    with col1:
        st.metric("📊 Total Forecasts", total_predictions)
    with col2:
        st.metric("📈 Average AQI", f"{latest_preds['us_aqi'].mean():.0f}" if total_predictions > 0 else "N/A")
    with col3:
        if total_predictions > 0:
            st.metric("🔺 AQI Range", f"{latest_preds['us_aqi'].min()} - {latest_preds['us_aqi'].max()}")
        else:
            st.metric("🔺 AQI Range", "N/A")
    with col4:
        st.metric("🤖 Model Version", latest_preds['model_version'].iloc[0] if 'model_version' in latest_preds.columns else "Unknown")

    # --- Current AQI Display ---
    if total_predictions > 0 and 'forecast_date_pkt' in latest_preds.columns:
        now = pd.Timestamp.now(tz='Asia/Karachi')
        latest_preds['time_diff'] = (latest_preds['forecast_date_pkt'] - now).abs()
        current_row = latest_preds.loc[latest_preds['time_diff'].idxmin()]

        st.markdown("---")
        st.markdown("### 🎯 Current Forecasted AQI")
        text_color = 'white' if current_row['color'] in ['#FF0000', '#8F3F97', '#7E0023'] else 'black'
        st.markdown(f"""
        <div style="background-color: {current_row['color']}; padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0; color: {text_color};">
            <h1 style="margin: 0; font-size: 4em;">{int(current_row['us_aqi'])}</h1>
            <h2 style="margin: 10px 0;">{current_row['category']}</h2>
            <p style="margin: 0; font-size: 1.2em;">Forecasted for {format_timestamp_pkt(current_row['forecast_date_pkt'])}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Forecast Chart ---
    if total_predictions > 0 and 'forecast_date_pkt' in latest_preds.columns:
        st.markdown("---")
        st.markdown("### 📈 74-Hour AQI Forecast")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=latest_preds['forecast_date_pkt'],
            y=latest_preds['us_aqi'],
            mode='lines+markers',
            name='Predicted AQI',
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=6, color=latest_preds['us_aqi'],
                        colorscale=[[0, '#00E400'], [0.2, '#FFFF00'], [0.4, '#FF7E00'],
                                    [0.6, '#FF0000'], [0.8, '#8F3F97'], [1, '#7E0023']],
                        cmin=0, cmax=300),
            text=latest_preds['category']
        ))
        st.plotly_chart(fig, use_container_width=True)

    # --- Data Table ---
    if total_predictions > 0:
        st.markdown("---")
        st.markdown("### 📋 Detailed Forecast Table")
        display_df = latest_preds[['forecast_date_pkt', 'us_aqi', 'category']].copy()
        display_df['forecast_date_pkt'] = display_df['forecast_date_pkt'].dt.strftime("%Y-%m-%d %H:%M:%S PKT")
        display_df = display_df.rename(columns={
            'forecast_date_pkt': 'Forecast Time (PKT)',
            'us_aqi': 'AQI Value',
            'category': 'AQI Category'
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv = latest_preds.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Data (CSV)",
            data=csv,
            file_name=f"aqi_forecast_rawalpindi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # --- Footer ---
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 20px;">
        <p><strong>🤖 Powered by LightGBM ML Model | 📊 Data from Hopsworks Feature Store</strong></p>
        <p><strong>🌍 Location: Rawalpindi, Punjab, Pakistan</strong> (33.5973°N, 73.0479°E)</p>
        <p>⚡ Last updated: {datetime.now(pytz.timezone('Asia/Karachi')).strftime('%Y-%m-%d %H:%M:%S PKT')}</p>
        <p>📊 Showing {total_predictions} forecast points | 🕒 All times in Pakistan Time (PKT)</p>
    </div>
    """, unsafe_allow_html=True)

# --- Run App ---
if __name__ == "__main__":
    main()
