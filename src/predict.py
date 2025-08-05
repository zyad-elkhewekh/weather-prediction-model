from datetime import datetime, timedelta
from meteostat import Point, Daily
import pandas as pd
import joblib

ml_model = joblib.load("D:/pycharm--------/Weather-Prediction-Model/models/final_model.pkl")        # ML model
fallback_model = joblib.load("D:/pycharm--------/Weather-Prediction-Model/models/fallback_model.pkl")  # Simple seasonal model

ML_FEATURES = ['month', 'day', 'tsun', 'wpgt', 'pres', 'prcp', 'prcp_lag1', 'wpgt_lag3']

def fetch_weather(date: datetime, location: Point) -> pd.DataFrame:
    return Daily(location, date, date).fetch()

def build_features(target_date: str) -> pd.DataFrame | None:
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    cairo = Point(30.0444, 31.2357)

    try:
        today_data = fetch_weather(target_dt - timedelta(days=1), cairo)
        lag1_data = fetch_weather(target_dt - timedelta(days=2), cairo)
        lag3_data = fetch_weather(target_dt - timedelta(days=4), cairo)

        if today_data.empty or lag1_data.empty or lag3_data.empty:
            return None

        row = {
            'month': target_dt.month,
            'day': target_dt.weekday(),
            'tsun': today_data['tsun'].values[0],
            'wpgt': today_data['wpgt'].values[0],
            'pres': today_data['pres'].values[0],
            'prcp': today_data['prcp'].values[0],
            'prcp_lag1': lag1_data['prcp'].values[0],
            'wpgt_lag3': lag3_data['wpgt'].values[0]
        }

        return pd.DataFrame([row])
    except:
        return None

def hybrid_predict(target_date: str) -> str:
    features = build_features(target_date)

    if features is not None:
        pred = ml_model.predict(features)[0]
        method = "ML model with full weather data"
    else:
        fallback_features = pd.DataFrame([{
            'month': datetime.strptime(target_date, "%Y-%m-%d").month,
            'day': datetime.strptime(target_date, "%Y-%m-%d").weekday()
        }])
        pred = fallback_model.predict(fallback_features)[0]
        method = "fallback seasonal model (month + weekday only)"

    return f"Prediction for {target_date}: {pred:.2f} °C ({method})"
