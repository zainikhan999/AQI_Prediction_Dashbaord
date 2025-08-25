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
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
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
        # 🔑 Connect to your actual project (default one with API key)
        project = hopsworks.login(api_key_value=API_KEY)
        fs = project.get_feature_store()

        # ✅ Auto-detect latest version of FG
        fg_info = fs.get_feature_groups(name="aqi_forecast_metrics_fg")
        if not fg_info:
            return None, "Feature group 'aqi_forecast_metrics_fg' not found!"

        latest_version = max(fg.version for fg in fg_info)
        fg = fs.get_feature_group(name="aqi_forecast_metrics_fg", version=latest_version)

        if fg is None:
            return None, f"FG 'aqi_forecast_metrics_fg' (v{latest_version}) not found"

        # ✅ Read into pandas dataframe
        df = fg.read()

        # Ensure pandas dataframe is not empty
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
        except Exception as e:
            continue
    
    return None, "No CSV backup files found"

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

def format_timestamp_pkt(ts):
    """Format timestamp for PKT display."""
    if pd.isna(ts):
        return "N/A"
    
    # Convert to PKT timezone
    pkt_tz = pytz.timezone('Asia/Karachi')
    if ts.tz is None:
        ts = pytz.utc.localize(ts)
    pkt_time = ts.astimezone(pkt_tz)
    return pkt_time.strftime("%Y-%m-%d %H:%M:%S PKT")

def format_timestamp_utc(ts):
    """Format timestamp for UTC display."""
    if pd.isna(ts):
        return "N/A"
    if ts.tz is None:
        ts = pytz.utc.localize(ts)
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")

# --- Main App ---
def main():
    # Header
    st.title("🌍 Rawalpindi AQI Forecast Dashboard")
    st.markdown("Real-time **74-hour AQI forecasts** powered by LightGBM ML model")
    st.markdown("📍 **Location:** Rawalpindi, Punjab, Pakistan (33.5973°N, 73.0479°E)")
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    data_source = st.sidebar.selectbox(
        "📂 Data Source",
        options=["Feature Store", "Local CSV Backup"],
        index=0
    )
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5 min)", value=False)
    show_advanced = st.sidebar.checkbox("🔧 Advanced Options", value=False)
    
    # Load data based on selected source
    with st.spinner("🔄 Loading latest predictions..."):
        if data_source == "Feature Store":
            df, error = load_predictions_data()
        else:
            df, error = load_backup_csv()
    
    if error:
        st.error(f"❌ Failed to load data: {error}")
        if data_source == "Feature Store":
            st.info("💡 Try loading from 'Local CSV Backup' or make sure the inference pipeline has run successfully.")
        else:
            st.info("💡 Make sure you have a predictions CSV file in the current directory.")
        
        # Try alternative source
        st.markdown("### 🔄 Trying Alternative Data Source...")
        if data_source == "Feature Store":
            df, error2 = load_backup_csv()
            if df is not None:
                st.success("✅ Successfully loaded from local CSV backup!")
            else:
                st.error("❌ No data available from any source.")
                return
        else:
            df, error2 = load_predictions_data()
            if df is not None:
                st.success("✅ Successfully loaded from feature store!")
            else:
                st.error("❌ No data available from any source.")
                return
    
    if df is None or len(df) == 0:
        st.warning("⚠️ No prediction data available.")
        st.info("🚀 Run the inference pipeline to generate predictions: `python inference_pipeline.py`")
        return
    
    # --- Data Processing ---
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 **Records loaded:** {len(df)}")
    st.sidebar.info(f"📂 **Data source:** {data_source}")
    
    # Display raw column names for debugging
    if st.sidebar.checkbox("🔍 Debug: Show column names"):
        st.sidebar.write("**Columns in dataset:**")
        st.sidebar.write(list(df.columns))
    
    # Standardize column names based on what we actually have
    column_mapping = {}
    
    # Try to identify datetime columns
    datetime_cols = [col for col in df.columns if 'datetime' in col.lower() or 'time' in col.lower()]
    aqi_cols = [col for col in df.columns if 'aqi' in col.lower()]
    
    # Smart column mapping
    for col in df.columns:
        if 'datetime_utc' in col.lower():
            column_mapping[col] = 'forecast_date_utc'
        elif 'datetime' in col.lower() and 'utc' not in col.lower():
            column_mapping[col] = 'forecast_date_local'
        elif 'predicted_us_aqi' in col.lower():
            column_mapping[col] = 'us_aqi'
        elif 'prediction_date' in col.lower():
            column_mapping[col] = 'prediction_time'
        elif 'model_version' in col.lower():
            column_mapping[col] = 'model_version'
        elif 'forecast_hour' in col.lower():
            column_mapping[col] = 'forecast_hour'
    
    # Apply mapping
    df = df.rename(columns=column_mapping)
    
    # Ensure we have required columns
    if 'us_aqi' not in df.columns and aqi_cols:
        df['us_aqi'] = df[aqi_cols[0]]
    
    if 'forecast_date_utc' not in df.columns and datetime_cols:
        df['forecast_date_utc'] = df[datetime_cols[0]]
    
    # Parse datetimes with better error handling
    pkt_tz = pytz.timezone('Asia/Karachi')
    utc_tz = pytz.UTC
    
    if 'forecast_date_utc' in df.columns:
        df['forecast_date_utc'] = pd.to_datetime(df['forecast_date_utc'], errors='coerce')
        # Ensure UTC timezone
        if df['forecast_date_utc'].dt.tz is None:
            df['forecast_date_utc'] = df['forecast_date_utc'].dt.tz_localize('UTC')
        else:
            df['forecast_date_utc'] = df['forecast_date_utc'].dt.tz_convert('UTC')
        
        # Create PKT version
        df['forecast_date_pkt'] = df['forecast_date_utc'].dt.tz_convert('Asia/Karachi')
    
    if 'prediction_time' in df.columns:
        df['prediction_time'] = pd.to_datetime(df['prediction_time'], errors='coerce')
        if df['prediction_time'].dt.tz is None:
            df['prediction_time'] = df['prediction_time'].dt.tz_localize('UTC')
        df['prediction_time'] = df['prediction_time'].dt.tz_convert('Asia/Karachi')
    
    # Keep only latest prediction run if we have prediction_time
    if 'prediction_time' in df.columns and not df['prediction_time'].isna().all():
        latest_run_time = df['prediction_time'].max()
        latest_preds = df[df['prediction_time'] == latest_run_time].copy()
    else:
        latest_preds = df.copy()
        latest_run_time = "Unknown"
    
    # Sort by forecast time
    if 'forecast_date_utc' in latest_preds.columns:
        latest_preds = latest_preds.sort_values('forecast_date_utc').reset_index(drop=True)
    elif 'forecast_date_pkt' in latest_preds.columns:
        latest_preds = latest_preds.sort_values('forecast_date_pkt').reset_index(drop=True)
    
    # Ensure integer AQI and add categories
    if 'us_aqi' in latest_preds.columns:
        latest_preds['us_aqi'] = pd.to_numeric(latest_preds['us_aqi'], errors='coerce').fillna(0).round().astype(int)
        latest_preds['category'] = latest_preds['us_aqi'].apply(aqi_category)
        latest_preds['color'] = latest_preds['us_aqi'].apply(aqi_color)
    else:
        st.error("❌ No AQI column found in the data!")
        return
    
    # --- Dashboard Metrics ---
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_predictions = len(latest_preds)
        st.metric("📊 Total Forecasts", total_predictions)
    
    with col2:
        if total_predictions > 0:
            avg_aqi = latest_preds['us_aqi'].mean()
            st.metric("📈 Average AQI", f"{avg_aqi:.0f}")
        else:
            st.metric("📈 Average AQI", "N/A")
    
    with col3:
        if total_predictions > 0:
            max_aqi = latest_preds['us_aqi'].max()
            min_aqi = latest_preds['us_aqi'].min()
            st.metric("🔺 AQI Range", f"{min_aqi} - {max_aqi}")
        else:
            st.metric("🔺 AQI Range", "N/A")
    
    with col4:
        if 'model_version' in latest_preds.columns:
            model_version = latest_preds['model_version'].iloc[0] if total_predictions > 0 else "Unknown"
            st.metric("🤖 Model Version", model_version)
        else:
            st.metric("🤖 Model Version", "Unknown")
    
    # --- Prediction Info ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if latest_run_time != "Unknown":
            st.info(f"🕒 **Latest Prediction Run:** {format_timestamp_pkt(latest_run_time)}")
        else:
            st.info("🕒 **Latest Prediction Run:** Unknown")
    
    with col2:
        if total_predictions > 0 and 'forecast_date_pkt' in latest_preds.columns:
            forecast_start = latest_preds['forecast_date_pkt'].min()
            forecast_end = latest_preds['forecast_date_pkt'].max()
            st.info(f"📅 **Forecast Range:** {format_timestamp_pkt(forecast_start)} to {format_timestamp_pkt(forecast_end)}")
    
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
            text_color = 'white' if color in ['#FF0000', '#8F3F97', '#7E0023'] else 'black'
            st.markdown(f"""
            <div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center; color: {text_color}; margin: 2px;">
                <strong>{category}</strong><br><small>{range_str}</small>
            </div>
            """, unsafe_allow_html=True)
    
    if total_predictions == 0:
        st.warning("⚠️ No predictions to display. Please run the inference pipeline first.")
        return
    
    # --- Current AQI Display ---
    st.markdown("---")
    st.markdown("### 🎯 Current Forecasted AQI")
    
    if 'forecast_date_pkt' in latest_preds.columns:
        # Find closest forecast to current time
        now = pd.Timestamp.now(tz='Asia/Karachi')
        latest_preds['time_diff'] = (latest_preds['forecast_date_pkt'] - now).abs()
        current_idx = latest_preds['time_diff'].idxmin()
        current_row = latest_preds.loc[current_idx]
        
        current_aqi = int(current_row['us_aqi'])
        current_category = current_row['category']
        current_color = current_row['color']
        current_time = current_row['forecast_date_pkt']
        
        # Large AQI display
        text_color = 'white' if current_color in ['#FF0000', '#8F3F97', '#7E0023'] else 'black'
        st.markdown(f"""
        <div style="background-color: {current_color}; padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0; color: {text_color}; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h1 style="margin: 0; font-size: 4em; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">{current_aqi}</h1>
            <h2 style="margin: 10px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">{current_category}</h2>
            <p style="margin: 0; font-size: 1.2em;">Forecasted for {format_timestamp_pkt(current_time)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Main Forecast Chart ---
    st.markdown("---")
    st.markdown("### 📈 74-Hour AQI Forecast")
    
    if len(latest_preds) > 0 and 'forecast_date_pkt' in latest_preds.columns:
        # Create enhanced plotly chart
        fig = go.Figure()
        
        # Add line trace
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
            hovertemplate='<b>Time:</b> %{x}<br><b>AQI:</b> %{y}<br><b>Category:</b> %{text}<extra></extra>',
            text=latest_preds['category']
        ))
        
        # Add AQI threshold lines
        aqi_thresholds = [(50, '#00E400', 'Good'), (100, '#FFFF00', 'Moderate'), 
                         (150, '#FF7E00', 'Unhealthy for Sensitive'), (200, '#FF0000', 'Unhealthy'), 
                         (300, '#8F3F97', 'Very Unhealthy')]
        
        for threshold, color, label in aqi_thresholds:
            fig.add_hline(
                y=threshold, 
                line_dash="dash", 
                line_color=color, 
                opacity=0.4,
                annotation_text=f"{label} ({threshold})",
                annotation_position="right"
            )
        
        # Update layout
        fig.update_layout(
            title="74-Hour AQI Forecast - Rawalpindi, Pakistan",
            xaxis_title="Forecast Time (Pakistan Time - PKT)",
            yaxis_title="US AQI",
            height=600,
            hovermode='x unified',
            showlegend=True,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', range=[0, max(350, latest_preds['us_aqi'].max() + 50)]),
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
                    latest_preds, 
                    x='us_aqi', 
                    nbins=20,
                    title="AQI Distribution",
                    labels={'us_aqi': 'US AQI', 'count': 'Frequency'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Category breakdown
                category_counts = latest_preds['category'].value_counts()
                fig_pie = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="AQI Categories Distribution",
                    color_discrete_map={
                        'Good': '#00E400',
                        'Moderate': '#FFFF00', 
                        'Unhealthy for Sensitive Groups': '#FF7E00',
                        'Unhealthy': '#FF0000',
                        'Very Unhealthy': '#8F3F97',
                        'Hazardous': '#7E0023'
                    }
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # --- Data Table ---
        st.markdown("---")
        st.markdown("### 📋 Detailed Forecast Table")
        
        # Prepare display dataframe - FIXED to show actual values
        display_df = latest_preds.copy()
        
        # Select and format columns for display
        display_cols = ['forecast_date_pkt', 'us_aqi', 'category']
        if 'forecast_hour' in display_df.columns:
            display_cols.insert(1, 'forecast_hour')
        
        display_df = display_df[display_cols].copy()
        display_df['forecast_date_pkt'] = display_df['forecast_date_pkt'].dt.strftime("%Y-%m-%d %H:%M:%S PKT")
        
        # Rename columns for display
        column_rename = {
            'forecast_date_pkt': 'Forecast Time (PKT)',
            'us_aqi': 'AQI Value',
            'category': 'AQI Category',
            'forecast_hour': 'Hour +'
        }
        display_df = display_df.rename(columns=column_rename)
        
        # FIXED: Use number column config instead of progress bars
        column_config = {
            'AQI Value': st.column_config.NumberColumn(
                'AQI Value',
                help='US Air Quality Index (0-500)',
                min_value=0,
                max_value=500,
                format='%d'
            )
        }
        
        if 'Hour +' in display_df.columns:
            column_config['Hour +'] = st.column_config.NumberColumn(
                'Hour +',
                help='Hours from now',
                min_value=0,
                format='%d'
            )
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )
        
        # Download button
        csv = latest_preds.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Data (CSV)",
            data=csv,
            file_name=f"aqi_forecast_rawalpindi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("⚠️ No forecast data available to display.")
    
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