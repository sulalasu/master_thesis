#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


#%%
# GET DUMMY DATA 
# from plotnine.data import economics
# print(economics.head())


#use data from https://github.com/zhouhaoyi/ETDataset
data = pd.read_csv("/media/lukas/Shared_Volume/Sattler_Master_thesis/Sattler_thesis/data/ETTh1_dummy_data.csv")#"data/ETTh1_dummy_data.csv")
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d %H:%M:%S")

#%%
# DUMMY DATA FROM SKLEARN
bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
#df = pd.DataFrame(data=bike_sharing.data, columns=bike_sharing.feature_names)
#df["target"] = bike_sharing.target
#df.head()
c = 3

#%%
#Saugeen river data (daily, univariate)
river = fetch_openml("Saugeen-River-Daily", as_frame=True) 
river = river.frame
river.head()


#%%
#Seoul bike sharing data ()
seoul = fetch_openml("seoul_bike_sharing_demand", as_frame=True)
seoul = pd.DataFrame(seoul.frame)

seoul_name_map = {
    "0": "time",
    "1": "hour",
    "2": "temperature", #C
    "3": "humidity", #%
    "4": "wind_speed", #m/s
    "5": "visibility", #10m
    "6": "dew_point_temp", #C
    "7": "solar_radiation", #Mj/m2
    "8": "rainfall", #mm,
    "9": "snowfall", #cm
    "10": "seasons", 
    "11": "holiday",
    "12": "functioning_day", #[dt: Arbeitstag]
    "13": "count" 
}
seoul.rename(mapper=seoul_name_map, axis=1, inplace=True)
seoul.head(n=100)
#seoul.info()
#seoul.describe()


#%%
#Plot seoul
plt.plot("0", "13", data=seoul)
plt.show
#seoul.head()
#%%
df = bike_sharing.frame
df.head()
#df.describe()
#df.info()

#%%
flights = sns.load_dataset("flights")
# flights.info()
# flights.head()
print("flights describ")
flights.info()

#%%
avg_weekday_demand = df.groupby(["weekday", "hour"], as_index=False)["count"].mean()
avg_weekday_demand.info()

#%%
plt.plot("weekday", "count", data=avg_weekday_demand)
plt.show()

#%%
# Show data (github) info
print(data.head())
print(data.info())
print(data.describe())

#%%
# Show data2 (SKLEARN) info:
print(data2.head())
print(data2.info())
print(data2.describe())

#%%
# USING PANDAS
data.plot(x="date", subplots=True)
# %%
# using SEARBORN
sns.lineplot(data=data, x="date", y="HUFL")
sns.lineplot(data=data, x="date", y="HULL")

