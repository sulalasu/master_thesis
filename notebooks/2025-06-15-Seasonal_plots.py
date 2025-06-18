#%%
import os
import pandas as pd
import numpy as np
from numpy import nan
from time import time
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns


import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))


from src import load
from src import clean
from src import process
from src import viz
from src import config
from src import model


print(load.__file__)
print(clean.__file__)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# SAMPLE DATA
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#%% ------------------------------------------------------------------------------
# Get data & clean
from sklearn.datasets import fetch_openml
raw_df = fetch_openml("Electricity-hourly", as_frame=True)
raw_df = pd.DataFrame(raw_df.frame)

#%%
#Drop all except date, value_0
col = 'value_5'
df = raw_df[["date", col]]
df = df.rename(columns={col : "value"})
df['date'] = pd.to_datetime(df['date']).dt.date.astype('datetime64[ns]') #alternative: .normalize() sets all to midnight 00:00:00


#Resample daily:
df_daily = df.set_index("date")
df_daily = df_daily["value"].resample("D").sum()
df_daily = df_daily.to_frame()

#Basic plot
sns.lineplot(x='date', y='value', data=df_daily)






#%% SEASONAL PLOTS:
series = df['value']



#%% period: hours in day
col = 'value_5'
df = raw_df[["date", col]]
df = df.rename(columns={col : "value"})
df = df.set_index('date')

series = df['value']
#hourly = series.resample('h').sum()
hourly = series.reset_index()
sns.lineplot(x='date', y='value', data=df)
#%%
hourly['day'] = hourly['date'].dt.days
#hourly['week'] = hourly['date'].dt.isocalendar().week #need to make it string later
hourly['day_str'] = hourly['day'].astype(str) #need string for 'hue'
print(hourly.head())

sns.lineplot(x='day_of_week', y='value', data=hourly, hue='week_str')
period_title = 'hourly'
plt.title(f'{period_title} seasonality plot')
plt.xlabel('Hour')
plt.ylabel('value')
plt.legend([],[], frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()




#%% period: days in week
#daily = series.resample('D').sum()
daily = daily.reset_index()
daily['day_of_week'] = daily['date'].dt.day_of_week
daily['week'] = daily['date'].dt.isocalendar().week #need to make it string later
daily['week_str'] = daily['week'].astype(str) #need string for 'hue'
print(daily.head())

sns.lineplot(x='day_of_week', y='value', data=daily, hue='week_str')
period_title = 'Daily'
plt.title(f'{period_title} seasonality plot')
plt.xlabel('Day of the week')
plt.ylabel('value')
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.legend([],[], frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()

#%% period: days in month
# wsl weniger interessant, da keine? seasonality?


#%% period: weeks in year
year_of_weeks = series.resample('W').sum()
year_of_weeks = year_of_weeks.reset_index()
year_of_weeks['week_of_year'] = year_of_weeks['date'].dt.isocalendar().week
year_of_weeks['year'] = year_of_weeks['date'].dt.year
year_of_weeks['year_str'] = year_of_weeks['year'].astype('str')
print(year_of_weeks.head())

sns.lineplot(x='week_of_year', y='value', data=year_of_weeks, hue='year_str')
period_title = 'weekly'
plt.title(f'{period_title} seasonality plot')
plt.legend([],[], frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()



















#%%
df.rename(mapper=config.seoul_name_map, axis=1, inplace=True)
print("seoul head\n\n")
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df["hour"] = pd.to_datetime(df["hour"], format="%H").dt.time

load.show_info(df=df)

#%% Skip -- CLEANING -- 


# ------------------------------------------------------------------------------
# -- PROCESSING -- 

# Merge date + time --> date
# df["hour"] = pd.to_datetime(df["hour"], format="%H:%M:%S")
df["date"] = pd.to_datetime(df.date.astype(str) + " " + df.hour.astype(str))
df = df.drop(columns="hour")

# Show info/head
df.info()
df.head()

# Aggregate daily:
print(df.info())
df = df.drop(columns=["seasons", "holiday", "functioning_day"])
df = df.resample('D', on="date").sum()
#df = df.reset_index()
load.show_info(df=df)





#%%-------------------------------------------------------------------------------
# -- VISUALIZE --
# --------------------------------------------------------------------------------
# line plot ________________________________________________________________
# TODO: wrap in function (or add to Model?, because its already cleaned+processed here, so next step
# besides viz would be add to model anyway? BUT exploratory viz is done on raw data, so no specific model...
# TODO: make prettier: add title, colorchart (so i can later exchange colors), etc.


daily_average = df["count"].mean()

sns.lineplot(x="date", y="count", data=df)
plt.axhline(y=daily_average, color="black", label="average")

plt.show()

# %%


## SEASONAL PLOTS ________________________________________________________________



#%% Weekly

#%%
# def seasonal_plot(data, periods: list[str]):
"""
periods = periods to make seasonal plots for (e.g. week, months)
can be "W", "M", "Y" 
"""
data = df
periods = ["W"]

# fig, ax = plt.subplots(ncols=1, nrows=len(periods))
#Aggregate Data:
#Weekly
data = data.reset_index()
data['day_of_week'] = data['date'].dt.dayofweek
data['week'] = data['date'].dt.isocalendar().week
data['week_str'] = data['week'].astype(str)
print(data.head())

sns.lineplot(x="day_of_week", y="count", data=data, hue="week_str")
plt.title('Weekly seasonality plot')
plt.xlabel('Day of the week')
plt.ylabel('count')
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.legend([],[], frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Monthly
data = df

series = data["count"]



#%% period: days in week
daily = series.resample('D').sum()
daily = daily.reset_index()
daily['day_of_week'] = daily['date'].dt.day_of_week
daily['week'] = daily['date'].dt.isocalendar().week #need to make it string later
daily['week_str'] = daily['week'].astype(str) #need string for 'hue'
print(daily.head())

sns.lineplot(x='day_of_week', y='count', data=daily, hue='week_str')
period_title = 'Daily'
plt.title(f'{period_title} seasonality plot')
plt.xlabel('Day of the week')
plt.ylabel('count')
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.legend([],[], frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()

#%% period: days in month
# wsl weniger interessant, da keine? seasonality?


#%% period: weeks in year
year_of_weeks = series.resample('W').sum()
year_of_weeks = year_of_weeks.reset_index()
year_of_weeks['week_of_year'] = year_of_weeks['date'].dt.isocalendar().week
year_of_weeks['year'] = year_of_weeks['date'].dt.year
year_of_weeks['year_str'] = year_of_weeks['year'].astype('str')
print(year_of_weeks.head())

sns.lineplot(x='week_of_year', y='count', data=year_of_weeks, hue='year_str')
period_title = 'weekly'
plt.title(f'{period_title} seasonality plot')
plt.legend([],[], frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()



#%% monthly
monthly = series.resample('ME').sum()
monthly = monthly.reset_index()
monthly['month'] = monthly['date'].dt.month
print(monthly.head())


#%% yearly
yearly = series.resample('YE').sum()
yearly = yearly.reset_index()
yearly['year'] = yearly['date'].dt.year
print(yearly.head())


def plot_seasonal(data, target, period, period_title):
    sns.lineplot(x=period, y=target, data=data, hue=period)
    plt.title(f'{period_title} seasonality plot')
    # plt.xlabel('Day of the week')
    # plt.ylabel('count')
    # plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.legend([],[], frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_seasonal(weekly, 'count', 'week', 'Weekly')

#%%
data["monthly"] = data["count"].resample("M", on="date")
print(data.head())
print()

#%%
series = data["count"]
monthly = series.resample("M")
#monthly = monthly.dt.month
yearly = series.resample('YE').sum()
#print(series.head())
print()
print(monthly)
print()
print(monthly.info())
print()
print(yearly)

#%%

data = data.reset_index()
data['week'] = data['date'].dt.isocalendar().week
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

print(data.head())

# sns.lineplot(x="day_of_week", y="count", data=data, hue="week_str")
# plt.title('Weekly seasonality plot')
# plt.xlabel('Day of the week')
# plt.ylabel('Demand')
# plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
# plt.legend([],[], frameon=False)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#%%
#Yearly
data['month'] = data['date'].dt.isocalendar().month
data['month_str'] = data['date'].astype(str)
for period in periods:
    pass




#TODO
fig, ax = plt.subplots()
ax.plot(label = ["year"])
pass

#%%
weekly = seasonal_plot(df, ["W"])
plt.show()



# Monthly

# Yearly


