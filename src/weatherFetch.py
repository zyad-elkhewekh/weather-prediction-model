from datetime import datetime, timedelta
from meteostat import Point, Daily
import pandas as pd
import joblib

data = pd.read_csv("D:/pycharm--------/Weather-Prediction-Model/data/cleaned-weather.csv")

date = input("Enter the date(YYYY-MM-DD): ")
date = datetime.strptime(date, "%Y-%m-%d")

cairo = Point(30.0444, 31.2357)

# today = Daily(cairo, date, date)
# today = today.fetch()

def fetch(day):
    df = Daily(cairo, day, day).fetch()
    if not df.empty:
        for col in df.columns:
            if pd.isna(df[col].values[0]):
                if col in data.columns:
                    df.at[df.index[0], col] = data[col].mean()
    return df

# print(today.head())

model = joblib.load("D:/pycharm--------/Weather-Prediction-Model/models/final_model.pkl")

today = fetch(date - timedelta(days=1))
lag1 = fetch(date - timedelta(days=2))
lag3 = fetch(date - timedelta(days=4))

if today.empty or lag1.empty or lag3.empty:
    print("Could not fetch required data.")
    exit()

# Build input features (same as training)
row = {
    'month': date.month,
    'day': date.weekday(),
    'tsun': today['tsun'].values[0],
    'wpgt': today['wpgt'].values[0],
    'pres': today['pres'].values[0],
    'prcp': today['prcp'].values[0],
    'prcp_lag1': lag1['prcp'].values[0],
    'wpgt_lag3': lag3['wpgt'].values[0]
}

X = pd.DataFrame([row])

# Predict
pred = model.predict(X)[0]
print(f"Predicted temperature for {date}: {pred:.2f} °C")

pred = model.predict(X)
print(pred)