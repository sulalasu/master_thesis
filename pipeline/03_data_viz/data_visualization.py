#%%
import pandas as pd
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

# DUMMY DATA FROM SKLEARN
bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
#df = pd.DataFrame(data=bike_sharing.data, columns=bike_sharing.feature_names)
#df["target"] = bike_sharing.target
#df.head()

df = bike_sharing.frame
df.head()
df.describe()
df.info()
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

