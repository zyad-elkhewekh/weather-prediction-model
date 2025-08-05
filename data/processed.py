from raw import data

data.snow = data.snow.fillna(0)
data.wdir = data.wdir.fillna(0)
data.prcp = data.prcp.fillna(data.prcp.mean())
data.wpgt = data.wpgt.fillna(data.wpgt.mean())
data.tsun = data.tsun.fillna(data.tsun.mean())

data.isnull().sum()