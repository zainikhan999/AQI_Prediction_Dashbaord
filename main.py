import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# === Load predictions (replace with your actual file or DB fetch) ===
df = pd.read_csv("predictions.csv")

# Ensure datetime fields are parsed
df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
df['prediction_date'] = pd.to_datetime(df['prediction_date'])

# === Keep only the latest prediction run ===
latest_run_time = df['prediction_date'].max()
latest_preds = df[df['prediction_date'] == latest_run_time].copy()

# === Filter: Only show 1h to 72h ahead from now ===
now = pd.Timestamp.utcnow()
time_horizon = now + timedelta(hours=72)

filtered_preds = latest_preds[
    (latest_preds['datetime_utc'] > now) &
    (latest_preds['datetime_utc'] <= time_horizon)
].sort_values("datetime_utc")

# === Streamlit UI ===
st.title("🌍 Air Quality Forecast (72 Hours Ahead)")

st.markdown("""
This dashboard shows **AQI forecasts** for the next 72 hours, starting from the current hour.
- **Good (0–50)** 🟢  
- **Moderate (51–100)** 🟡  
- **Unhealthy for Sensitive (101–150)** 🟠  
- **Unhealthy (151–200)** 🔴  
- **Very Unhealthy (201–300)** 🟣  
- **Hazardous (301–500)** ⚫
""")

if filtered_preds.empty:
    st.warning("⚠️ No predictions available for the next 72 hours.")
else:
    # === Show table ===
    st.subheader("📊 Forecast Table (Next 72 Hours)")
    st.dataframe(
        filtered_preds[['datetime_utc', 'aqi']],
        use_container_width=True
    )

    # === Plot line chart ===
    st.subheader("📈 AQI Forecast Trend (Next 72 Hours)")
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_preds['datetime_utc'], filtered_preds['aqi'], marker='o')
    plt.axhline(50, color="green", linestyle="--", alpha=0.5)
    plt.axhline(100, color="yellow", linestyle="--", alpha=0.5)
    plt.axhline(150, color="orange", linestyle="--", alpha=0.5)
    plt.axhline(200, color="red", linestyle="--", alpha=0.5)
    plt.axhline(300, color="purple", linestyle="--", alpha=0.5)
    plt.axhline(500, color="black", linestyle="--", alpha=0.5)

    plt.xticks(rotation=45)
    plt.ylabel("AQI")
    plt.xlabel("Time (UTC)")
    plt.title("Air Quality Forecast (Next 72 Hours)")
    st.pyplot(plt)
