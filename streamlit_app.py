import streamlit as st
import joblib
import requests
import pandas as pd
from preprocess_and_features import make_daily_features
from datetime import date, timedelta

def get_city_coordinates(city):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    resp = requests.get(geo_url).json()
    if "results" not in resp or not resp["results"]:
        return None, None
    lat = resp["results"][0]["latitude"]
    lon = resp["results"][0]["longitude"]
    return lat, lon

def fetch_pm25(city="Delhi", start_date="2024-01-01", end_date="2024-01-10"):
    lat, lon = get_city_coordinates(city)
    if lat is None:
        return pd.DataFrame()
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=pm2_5"
    )
    resp = requests.get(url).json()
    if "hourly" not in resp or "pm2_5" not in resp["hourly"]:
        return pd.DataFrame()
    df = pd.DataFrame({
        "timestamp": resp["hourly"]["time"],
        "pm25": resp["hourly"]["pm2_5"],
    })
    return df

def fetch_weather(city="Delhi", start_date="2024-01-01", end_date="2024-01-10"):
    lat, lon = get_city_coordinates(city)
    if lat is None:
        return pd.DataFrame()
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m"
    )
    resp = requests.get(url).json()
    if "hourly" not in resp or not all(k in resp["hourly"] for k in ["temperature_2m", "relative_humidity_2m", "windspeed_10m"]):
        return pd.DataFrame()
    df = pd.DataFrame({
        "timestamp": resp["hourly"]["time"],
        "temperature": resp["hourly"]["temperature_2m"],
        "relativehumidity": resp["hourly"]["relative_humidity_2m"],
        "windspeed": resp["hourly"]["windspeed_10m"]
    })
    return df

@st.cache_resource
def load_model(path='model.joblib'):
    d = joblib.load(path)
    return d['model'], d['features']

st.title("EarthData→Action: PM2.5 short-term predictor (demo)")

city = st.text_input("City", value="Delhi")
end_date = date.today()
start_date = end_date - timedelta(days=120)

if st.button("Fetch data & predict next day"):
    with st.spinner("Fetching data..."):
        pm_df = fetch_pm25(city, start_date=start_date.isoformat(), end_date=end_date.isoformat())
        if pm_df.empty:
            st.error("No PM2.5 data found for that city/time. Try different dates or city name.")
        else:
            met_df = fetch_weather(city, start_date=start_date.isoformat(), end_date=end_date.isoformat())
            if met_df.empty:
                st.error("Couldn't fetch meteorology.")
            else:
                df_feats = make_daily_features(pm_df, met_df)
                df_feats = df_feats.reset_index()
                if len(df_feats) < 10:
                    st.warning("Not enough daily rows after feature creation for stable prediction.")
                else:
                    model, feature_cols = load_model()
                    last_row = df_feats.iloc[-1].copy()
                    X_pred = {}
                    X_pred['temperature'] = last_row['temperature']
                    X_pred['relativehumidity'] = last_row['relativehumidity']
                    X_pred['windspeed'] = last_row['windspeed']
                    X_pred['pm25_lag_1'] = last_row['pm25']
                    X_pred['pm25_lag_2'] = df_feats.iloc[-2]['pm25'] if len(df_feats) >= 2 else last_row['pm25']
                    X_pred['pm25_lag_3'] = df_feats.iloc[-3]['pm25'] if len(df_feats) >= 3 else last_row['pm25']
                    X_pred['pm25_lag_7'] = df_feats.iloc[-7]['pm25'] if len(df_feats) >= 7 else last_row['pm25']
                    X_pred['pm25_ma_3'] = df_feats['pm25'].rolling(3).mean().iloc[-1]
                    X_pred['dayofyear'] = (pd.to_datetime(last_row['timestamp']) + pd.Timedelta(days=1)).dayofyear

                    # Convert to DataFrame and ensure correct column order
                    X_pred_df = pd.DataFrame([X_pred])[feature_cols]

                    # Make prediction
                    y_pred = model.predict(X_pred_df)[0]

                    st.write("Latest observed daily PM2.5 (last date):", last_row['timestamp'], f"{last_row['pm25']:.1f} µg/m³")
                    st.success(f"**Predicted next day's PM2.5:** {y_pred:.1f} µg/m³")
                    st.line_chart(df_feats.set_index('timestamp')['pm25'])

st.markdown("---")
st.markdown("**Notes:** Model is a demo. For production you should: (1) add more features (EO NO2, AOD), (2) do proper cross-validation, (3) retrain frequently, (4) add uncertainty estimates.")
