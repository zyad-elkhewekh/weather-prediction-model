from pandas import read_csv, read_pickle

#from raw import data
#clean = read_csv("weather.csv")
clean = read_pickle("weather.pickle")

print(clean.head())



clean.snow = clean.snow.fillna(0)
clean.wdir = clean.wdir.fillna(0)
clean.prcp = clean.prcp.fillna(clean.prcp.mean())
clean.wpgt = clean.wpgt.fillna(clean.wpgt.mean())
clean.tsun = clean.tsun.fillna(clean.tsun.mean())

print(f"after cleaning: ")
print(clean.isnull().sum())
print(clean.head())

clean.to_pickle("cleaned-weather.pickle")