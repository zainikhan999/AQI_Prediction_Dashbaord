# ===================================================================
# AQI Streamlit Dashboard - Enhanced for Inference Pipeline Integration
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

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
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_predictions_data():
    """Load predictions data from Hopsworks feature store."""
    try:
        # Connect to Hopsworks
        project = hopsworks.login(api_key_value=API_KEY, project="weather_aqi")
        fs = project.get_feature_store()
        
        # Load predictions feature group
        fg = fs.get_feature_group(name="aqi_predictions", version=1)
        df = fg.read()
        
        return df, None
    except Exception as e:
        return None, str(e)

def aqi_category(aqi: int) -> str:
    """Get AQI category from AQI value."""
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
    """Get color for AQI category."""
    if aqi <= 50:
        return "#00E400"  # Green
    elif aqi <= 100:
        return "#FFFF00"  # Yellow
    elif aqi <= 150:
        return "#FF7E00"  # Orange
    elif aqi <= 200:
        return "#FF0000"  # Red
    elif aqi <= 300:
        return "#8F3F97"  # Purple
    return "#7E0023"  # Maroon

def format_timestamp(ts):
    """Format timestamp for display."""
    if pd.isna(ts):
        return "N/A"
    return ts.strftime("%Y-%m-%d %H:%M:%S %Z")

# --- Main App ---
def main():
    # Header
    st.title("🌍 Rawalpindi AQI Forecast Dashboard")
    st.markdown("Real-time **74-hour AQI forecasts** powered by LightGBM ML model")
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5 min)", value=True)
    show_advanced = st.sidebar.checkbox("🔧 Advanced Options", value=False)
    
    # Load data
    with st.spinner("🔄 Loading latest predictions..."):
        df, error = load_predictions_data()
    
    if error:
        st.error(f"❌ Failed to load data: {error}")
        st.info("💡 Make sure the inference pipeline has run successfully and predictions are available.")
        return
    
    if df is None or len(df) == 0:
        st.warning("⚠️ No prediction data available.")
        st.info("🚀 Run the inference pipeline to generate predictions: `python aqi_inference_pipeline.py`")
        return
    
    # --- Data Processing ---
    # Standardize columns (matching inference pipeline output)
    df = df.rename(columns={
        "datetime_utc": "forecast_date_utc",
        "datetime": "forecast_date_local",
        "predicted_us_aqi": "us_aqi",
        "prediction_date": "prediction_time",
    })
    
    # Parse datetimes safely
    for col in ["forecast_date_utc", "forecast_date_local", "prediction_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Ensure timezone handling
    if "forecast_date_utc" in df.columns:
        df["forecast_date_utc"] = pd.to_datetime(df["forecast_date_utc"], utc=True)
        df["forecast_date_local"] = df["forecast_date_utc"].dt.tz_convert("Asia/Karachi")
    
    if "prediction_time" in df.columns:
        df["prediction_time"] = pd.to_datetime(df["prediction_time"], utc=True).dt.tz_convert("Asia/Karachi")
    
    # Keep only latest prediction run
    if "prediction_time" in df.columns:
        latest_run_time = df["prediction_time"].max()
        latest_preds = df[df["prediction_time"] == latest_run_time].copy()
    else:
        latest_preds = df.copy()
        latest_run_time = "Unknown"
    
    latest_preds = latest_preds.sort_values("forecast_date_utc").reset_index(drop=True)
    
    # Ensure integer AQI
    latest_preds["us_aqi"] = latest_preds["us_aqi"].round().astype(int)
    latest_preds["category"] = latest_preds["us_aqi"].apply(aqi_category)
    latest_preds["color"] = latest_preds["us_aqi"].apply(aqi_color)
    
    # --- Dashboard Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_predictions = len(latest_preds)
        st.metric("📊 Total Forecasts", total_predictions)
    
    with col2:
        if total_predictions > 0:
            avg_aqi = latest_preds["us_aqi"].mean()
            st.metric("📈 Average AQI", f"{avg_aqi:.0f}")
        else:
            st.metric("📈 Average AQI", "N/A")
    
    with col3:
        if total_predictions > 0:
            max_aqi = latest_preds["us_aqi"].max()
            st.metric("🔺 Peak AQI", max_aqi)
        else:
            st.metric("🔺 Peak AQI", "N/A")
    
    with col4:
        if "model_version" in latest_preds.columns:
            model_version = latest_preds["model_version"].iloc[0] if total_predictions > 0 else "Unknown"
            st.metric("🤖 Model Version", model_version)
        else:
            st.metric("🤖 Model Version", "Unknown")
    
    # --- Prediction Info ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"🕒 **Latest Prediction Run:** {format_timestamp(latest_run_time)}")
    
    with col2:
        if total_predictions > 0:
            forecast_start = latest_preds["forecast_date_local"].min()
            forecast_end = latest_preds["forecast_date_local"].max()
            st.info(f"📅 **Forecast Range:** {format_timestamp(forecast_start)} to {format_timestamp(forecast_end)}")
    
    # --- AQI Legend ---
    st.markdown("---")
    st.markdown("### 📋 AQI Categories")
    
    legend_cols = st.columns(6)
    aqi_ranges = [
        ("Good", "0-50", "#00E400"),
        ("Moderate", "51-100", "#FFFF00"),
        ("Unhealthy for Sensitive", "101-150", "#FF7E00"),
        ("Unhealthy", "151-200", "#FF0000"),
        ("Very Unhealthy", "201-300", "#8F3F97"),
        ("Hazardous", "301-500", "#7E0023")
    ]
    
    for i, (category, range_str, color) in enumerate(aqi_ranges):
        with legend_cols[i]:
            st.markdown(f"""
            <div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center; color: {'white' if color in ['#FF0000', '#8F3F97', '#7E0023'] else 'black'}">
                <strong>{category}</strong><br>{range_str}
            </div>
            """, unsafe_allow_html=True)
    
    if total_predictions == 0:
        st.warning("⚠️ No predictions to display. Please run the inference pipeline first.")
        return
    
    # --- Time Range Selection ---
    st.markdown("---")
    st.markdown("### 🕒 Time Range Selection")
    
    all_timestamps = list(latest_preds["forecast_date_local"].unique())
    all_timestamps.sort()
    
    col1, col2 = st.columns(2)
    with col1:
        start_ts = st.selectbox(
            "📅 Start Time",
            options=all_timestamps,
            index=0,
            format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S %Z"),
            key="start_time"
        )
    with col2:
        end_ts = st.selectbox(
            "📅 End Time",
            options=all_timestamps,
            index=len(all_timestamps) - 1,
            format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S %Z"),
            key="end_time"
        )
    
    if start_ts > end_ts:
        st.warning("⚠️ Start time is after end time — swapping them.")
        start_ts, end_ts = end_ts, start_ts
    
    # Filter predictions
    mask = (latest_preds["forecast_date_local"] >= start_ts) & (latest_preds["forecast_date_local"] <= end_ts)
    filtered = latest_preds.loc[mask].reset_index(drop=True)
    
    # --- Current AQI Display ---
    st.markdown("---")
    st.markdown("### 🎯 Current Forecasted AQI")
    
    now = pd.Timestamp.now(tz="Asia/Karachi")
    latest_preds["time_diff"] = (latest_preds["forecast_date_local"] - now).abs()
    current_row = latest_preds.loc[latest_preds["time_diff"].idxmin()]
    
    current_aqi = int(current_row["us_aqi"])
    current_category = current_row["category"]
    current_color = current_row["color"]
    current_time = current_row["forecast_date_local"]
    
    # Large AQI display
    st.markdown(f"""
    <div style="background-color: {current_color}; padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0; color: {'white' if current_color in ['#FF0000', '#8F3F97', '#7E0023'] else 'black'}">
        <h1 style="margin: 0; font-size: 4em;">{current_aqi}</h1>
        <h2 style="margin: 10px 0;">{current_category}</h2>
        <p style="margin: 0; font-size: 1.2em;">Forecasted for {format_timestamp(current_time)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Main Forecast Chart ---
    st.markdown("---")
    st.markdown("### 📈 74-Hour AQI Forecast")
    
    if len(filtered) > 0:
        # Create enhanced plotly chart
        fig = go.Figure()
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=filtered["forecast_date_local"],
            y=filtered["us_aqi"],
            mode='lines+markers',
            name='Predicted AQI',
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=8),
            hovertemplate='<b>Time:</b> %{x}<br><b>AQI:</b> %{y}<br><b>Category:</b> %{text}<extra></extra>',
            text=filtered["category"]
        ))
        
        # Add AQI threshold lines
        aqi_thresholds = [50, 100, 150, 200, 300]
        threshold_colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97']
        
        for i, (threshold, color) in enumerate(zip(aqi_thresholds, threshold_colors)):
            fig.add_hline(
                y=threshold, 
                line_dash="dash", 
                line_color=color, 
                opacity=0.3,
                annotation_text=f"AQI {threshold}",
                annotation_position="right"
            )
        
        # Update layout
        fig.update_layout(
            title="74-Hour AQI Forecast - Rawalpindi, Pakistan",
            xaxis_title="Forecast Time (Asia/Karachi)",
            yaxis_title="US AQI",
            height=500,
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Advanced Options ---
        if show_advanced:
            st.markdown("---")
            st.markdown("### 🔧 Advanced Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # AQI distribution
                fig_hist = px.histogram(
                    filtered, 
                    x="us_aqi", 
                    nbins=20,
                    title="AQI Distribution",
                    labels={"us_aqi": "US AQI", "count": "Frequency"}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Category breakdown
                category_counts = filtered["category"].value_counts()
                fig_pie = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="AQI Categories Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # --- Data Table ---
        st.markdown("---")
        st.markdown("### 📋 Detailed Forecast Table")
        
        # Prepare display dataframe
        display_df = filtered[["forecast_date_local", "us_aqi", "category"]].copy()
        display_df["forecast_date_local"] = display_df["forecast_date_local"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        display_df = display_df.rename(columns={
            "forecast_date_local": "Forecast Time (PKT)",
            "us_aqi": "AQI",
            "category": "Category"
        })
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "AQI": st.column_config.ProgressColumn(
                    "AQI",
                    help="US Air Quality Index",
                    min_value=0,
                    max_value=500,
                ),
            }
        )
        
        # Download button
        csv = filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Data (CSV)",
            data=csv,
            file_name=f"aqi_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("⚠️ No data available for the selected time range.")
    
    # --- Footer ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 20px;">
        <p>🤖 Powered by LightGBM ML Model | 📊 Data from Hopsworks Feature Store</p>
        <p>🌍 Location: Rawalpindi, Pakistan (33.5973°N, 73.0479°E)</p>
        <p>⚡ Auto-refreshes every 5 minutes when enabled</p>
    </div>
    """, unsafe_allow_html=True)

# --- Run App ---
if __name__ == "__main__":
    main()

# Auto-refresh functionality
if st.sidebar.checkbox("🔄 Auto-refresh (5 min)", value=False):
    import time
    time.sleep(300)  # 5 minutes
    st.rerun()