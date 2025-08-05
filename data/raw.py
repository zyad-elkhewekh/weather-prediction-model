from datetime import datetime
from meteostat import Point, Daily
import pandas as pd

cairo1 = Point(30.0444, 31.2357)

start = datetime(2020, 1, 1)
end = datetime(2024, 12, 31)

data = Daily(cairo1, start, end)
data = data.fetch()
print(data.head())
print(data.tail())
data.info()

data.to_csv('weather.csv')
data.to_pickle('weather.pickle')