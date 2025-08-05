import pandas as pd
from sklearn.preprocessing import StandardScaler

absolutePath = "D:/pycharm--------/Weather-Prediction-Model/data/weather.pickle"

data = pd.read_pickle(absolutePath)

print(data.head())

data.snow = data.snow.fillna(0)
data.wdir = data.wdir.fillna(0)
data.prcp = data.prcp.fillna(data.prcp.mean())
data.wpgt = data.wpgt.fillna(data.wpgt.mean())
data.tsun = data.tsun.fillna(data.tsun.mean())

print("after cleaning: ")
print(data.isnull().sum())
print(data.head())

data['month'] = data.index.month
data['day'] = data.index.day
data['year'] = data.index.year
print(data.head())


scaler = StandardScaler()
scaled = scaler.fit_transform(data[['tavg', 'prcp', 'wdir', 'wpgt', 'tsun']])
scaled_data = pd.DataFrame(scaled, columns=['tavg', 'prcp', 'wdir', 'wpgt', 'tsun'])

scaled_data.index = data.index

data[['tavg', 'prcp', 'wdir', 'wpgt', 'tsun']] = scaled_data
print(data.head())

import joblib
joblib.dump(scaler, 'D:/pycharm--------/Weather-Prediction-Model/models/scaler.pkl')
# Load later with:
# scaler = joblib.load('models/scaler.pkl')

#data.to_pickle("D:/pycharm--------/Weather-Prediction-Model/data/cleaned-weather.pickle")
#data.to_csv("D:/pycharm--------/Weather-Prediction-Model/data/cleaned-weather.csv")