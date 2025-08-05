from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd

df = pd.read_pickle("D:/pycharm--------/Weather-Prediction-Model/data/cleaned-weather.pickle")
df['target'] = df['tavg'].shift(-1)
df.dropna(inplace=True)

fallback_features = df[['month', 'day']]
fallback_target = df['target']

fallback_model = LinearRegression()
fallback_model.fit(fallback_features, fallback_target)

joblib.dump(fallback_model, "D:/pycharm--------/Weather-Prediction-Model/models/fallback_model.pkl")
