from sklearn import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib

data = pd.read_pickle("D:/pycharm--------/Weather-Prediction-Model/data/cleaned-weather.pickle")
#print(data.info())

data['target'] = data['tavg'].shift(-1)
data.dropna(inplace=True)
FEATURES = [
    'month', 'day', 'tsun', 'wpgt', 'pres', 'prcp',
    'prcp_lag1', 'wpgt_lag3'
]

features = data[FEATURES]
target = data.target

split_date = '2024-01-01'
x_train = features[features.index < split_date]
x_test  = features[features.index >= split_date]
y_train = target[target.index < split_date]
y_test  = target[target.index >= split_date]

#model = RandomForestRegressor(n_estimators=100, random_state=42)
#model = GradientBoostingRegressor()
#model = LinearRegression()
#model.fit(x_train, y_train)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, None]
}

grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)
grid.fit(x_train, y_train)

model = grid.best_estimator_

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")
print("Best parameters:", grid.best_params_)

#print(data.corr(numeric_only=True)['tavg'].sort_values(ascending=False))

joblib.dump(model, "D:/pycharm--------/Weather-Prediction-Model/models/final_model.pkl")
