from data_loader import data

data['tavg_lag1'] = data['tavg'].shift(1)
data['prcp_lag1'] = data['prcp'].shift(1)
data['wpgt_lag3'] = data['wpgt'].shift(3)
data.dropna(inplace=True)

data.to_pickle("D:/pycharm--------/Weather-Prediction-Model/data/cleaned-weather.pickle")
data.to_csv("D:/pycharm--------/Weather-Prediction-Model/data/cleaned-weather.csv")