#Plotting pipeline
#TODO: make functions/object, for reuse
#TODO: func/obj for daily, weekly, monthly, n-days averages/sums
#TODO: make prettier plots: add legend, title, axis titles
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.datasets import fetch_openml
from statsmodels.tsa.seasonal import seasonal_decompose

#%%
# Get data & clean
seoul = fetch_openml("seoul_bike_sharing_demand", as_frame=True)
seoul = pd.DataFrame(seoul.frame)

seoul_name_map = {
    "0": "date",
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
print("seoul head\n\n")
seoul["date"] = pd.to_datetime(seoul["date"], format="%Y-%m-%d")
seoul["hour"] = pd.to_datetime(seoul["hour"], format="%H").dt.time

print(seoul.head(n=20))
print(seoul.info())
# print(seoul.describe())

#%%
#calculate daily values, weekly + monthly averages

s_daily = seoul.groupby("date", as_index=False)["count"].sum()
daily_average = s_daily["count"].mean()
# print(s_daily.head())

#Weekly:
s_weekly = s_daily.groupby(pd.Grouper(key='date', freq='W'))['count'].mean().reset_index()
s_weekly['week_start'] = s_weekly['date'] - pd.offsets.Week(1) + pd.offsets.Day(1)
s_weekly = s_weekly.rename({'date': 'week_end'}, axis=1)

#Monthly:
s_monthly = s_daily.groupby(pd.Grouper(key='date', freq='M'))['count'].mean().reset_index()
s_monthly['month_start'] = s_monthly['date'] - pd.offsets.Week(1) + pd.offsets.Day(1)
s_monthly = s_monthly.rename({'date': 'month_end'}, axis=1)

#group daily and plot
#matplotlib
# plt.plot("date", "count", data=s_daily)
# plt.show()


#%%
#Seaborn: plot smoothed plots

sns.lineplot(x="date", y="count", data=s_daily)
sns.lineplot(x="week_start", y="count", data=s_weekly)
sns.lineplot(x="month_start", y="count", data=s_monthly)
#Add avg
plt.axhline(y=daily_average, color="black", label="average")
plt.show()
plt.close()

 #%%
# --------------------------------------------
# DECOMPOSITION (statsmodels)
# --------------------------------------------

# Average
average = s_daily["count"].mean()

# Additive Decomposition (trend, seasonality, noise)
print(f"s_daily head: {s_daily.head()}")
s_daily = s_daily.set_index("date")
result = seasonal_decompose(s_daily, model="additive", period=7)
print("\n\ntrend results:\n\n")
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)

result.plot()
plt.show()
plt.close()


# # MULTIPLICATIVE Decomposition (trend, seasonality, noise)
# result = seasonal_decompose(s_daily, model="multiplicative", period=12)
# print("\n\ntrend results:\n\n")
# print(result.trend)
# print(result.seasonal)
# print(result.resid)
# print(result.observed)

# result.plot()
# plt.show()



#%%
# Polar seasonality plot
# print("plotting polar plot")
# fig, ax = plt.subplots(subplot_kw={"projection" : "polar"})
# ax.plot(s_daily["date"], s_daily["count"])
# ax.grid(True)

# t = mdates.date2num(s_daily.index.to_pydatetime())
# y = s_daily['count']
# print(t[:10])
# print(y[:10])

# ax = plt.subplot(projection='polar')
# ax.set_theta_direction(-1)
# ax.set_theta_zero_location("N")

# tnorm = (t-t.min())/(t.max()-t.min())*2.*np.pi
# ax.fill_between(tnorm,y ,0, alpha=0.4)
# ax.plot(tnorm,y , linewidth=0.8)
# plt.show()

# plt.show()


## Sample data: dates and values
dates = s_daily.index
values = s_daily["count"]

# Convert dates to day-of-year and then to angles
day_of_year = dates.dayofyear
years = dates.year
unique_years = np.unique(years)


# Create a polar plot
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# Set the radial limits
min_radius = 2.0  # Minimum radius where zero values will start
max_radius = 3.0  # Maximum radius
ax.set_rorigin(0)
ax.set_rmax(max_radius)
#ax.set_ylim(0, 2) 


# Plot each year separately
for year in unique_years:
    mask = years == year
    # Normalize day_of_year to a 365-day scale
    normalized_day_of_year = (day_of_year[mask] - 1) / (366 if dates[mask][0].year % 4 == 0 and (dates[mask][0].year % 100 != 0 or dates[mask][0].year % 400 == 0) else 365) * 365
    angles = normalized_day_of_year / 365.0 * 2 * np.pi
    # Scale the radii to start from min_radius
    radii = values[mask] * (max_radius - min_radius) + min_radius
    ax.plot(angles, radii, label=str(year))

# Customize the plot
ax.set_title('Multi-Year Time Series Data in Polar Plot', size=20)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rlabel_position(30)
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.show()

#TODO: Make radial offset (like donut; zero != center) for polar plot 
# %%


#TODO: plot heatmap with weeks/months on x, years on y, count as color