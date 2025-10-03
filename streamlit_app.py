import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta


# Dummy model and feature list for demonstration
def dummy_predict(features):
    return np.random.uniform(30, 150)

model_features = [
    "temperature", "relativehumidity", "windspeed",
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_7",
    "pm25_ma_3", "dayofyear"
]

st.set_page_config(page_title="PM2.5 Predictor", layout="centered")
st.title("EarthData→Action: PM2.5 short-term predictor (demo)")

indian_cities = [
    "Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru", "Hyderabad", "Ahmedabad",
    "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Patna",
    "Ludhiana", "Agra", "Nashik", "Vadodara", "Varanasi", "Srinagar", "Amritsar",
    "Ranchi", "Guwahati", "Chandigarh", "Coimbatore", "Vijayawada", "Mysuru"
]

with st.sidebar:
    st.header("Settings")
    city = st.selectbox("City", indian_cities, index=0)
    end_date = st.date_input("End date", value=date.today())
    start_date = st.date_input("Start date", value=end_date - timedelta(days=120))
    st.markdown("[GitHub Repository](https://github.com/Bunny-max-gif/vajra-env)")
    st.info("Select a city and date range, then click the button below to fetch data and predict PM2.5.")

if st.button("Fetch data & predict next day"):
    with st.spinner("Fetching data and predicting..."):
        # --- Replace this block with your real data fetching and feature engineering ---
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        pm25 = np.random.uniform(40, 120, len(dates))
        df = pd.DataFrame({"date": dates, "pm25": pm25})
        # --- End of placeholder block ---

        st.success(f"Latest observed daily PM2.5 for {city}: {df['pm25'].iloc[-1]:.1f} µg/m³")
        pred = dummy_predict(None)
        st.info(f"**Predicted next day's PM2.5:** {pred:.1f} µg/m³")

        st.line_chart(df.set_index("date")["pm25"])

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download daily PM2.5 data as CSV",
            data=csv,
            file_name=f"{city}_pm25_data.csv",
            mime='text/csv'
        )

        # Show model features
        st.markdown("#### Model Features Used")
        st.write(model_features)

st.markdown("**Notes:** Model is a demo. For production you should: (1) add more features (EO NO₂, AOD), (2) do proper cross-validation, (3) retrain frequently, (4) add uncertainty estimates.")

