#%%
import pandas as pd
import numpy as np
from numpy import nan
from time import time
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns


from src import clean
from src import config
from src import data_model
from src import load
from src import model
from src import transform
from src import viz


#For developing purposes:
import importlib
print(load.__file__)
print(clean.__file__)




#%%--------------------------------------------------------------------------------
# INPUT
#----------------------------------------------------------------------------------

# Read Data
df_raw = load.load_data(path="data/01_raw/blood-data_complete.tsv")
# df_raw = load.load_data(path="data/01_raw/testdaten.tsv")

load.show_info(df=df_raw)
hidden_cols=["date", "EC_ID_I_hash", "EC_ID_O_hash", "T_ISO", "T_DE_T", "T_US", "T_DE_S", "T_US_T", "T_DE", "T_ISO_T", "T_XL"]
for col in df_raw.columns:
    if col not in hidden_cols:
        tmp = df_raw[col].astype(str).unique()
        tmp = np.array(sorted(tmp))
        print(f"{col}:\n{tmp}\n")
print(df_raw.columns)
df_raw = clean.clean_dates(df_raw) #TODO: remove here, enable again in clean_data()



#%%--------------------------------------------------------------------------------
# CLEANING 
#----------------------------------------------------------------------------------

#Runs only if no file exists at. If not existing, saves df to new file

#TODO: remove
#Sample random days (for testing purposes)
# sampled_values = df_raw.index.to_series().sample(n=500, random_state=42)
# df_raw2 = df_raw[df_raw.index.isin(sampled_values)]
#Sample random rows
#df_raw = df_raw.sample(n=90000, random_state=10) #TODO: remove
#Subset by date range:
start_date = pd.to_datetime("2016-01-01")
end_date = pd.to_datetime("2020-12-31")
df_test = df_test[start_date:end_date]
#%%
#unify dates, columns etc. rename stuff
importlib.reload(clean)

df_clean = clean.clean_data(df_raw2)
# df_clean.sort_index(inplace=True)
# #TODO: remove 5 lines:
# start_date = pd.to_datetime("2018-01-01")
# start_date = pd.to_datetime("2024-12-31")
# mask = (df_clean.index >= "2018-01-01") & (df_clean.index <= "2024-12-31")
#df_clean = df_clean.loc[mask]
#df_clean = df_clean['2018-01-01':'2024-12-31'] #only works on monotonic (==daily aggregated, no duplicate days) df

#TODO: Check what unique vals are present in df
clean.check_unique_values(df_clean.drop(["EC_ID_I_hash", "EC_ID_O_hash", "PAT_WARD"], axis=1))



#%%
# Plot frequency counts for unique values in every column
#TODO: move into viz.py
for col_name, col in df_clean.items():
    if col_name in ["EC_ID_O_hash", "EC_ID_I_hash"]:
        continue
    print(col.value_counts())
    col.value_counts()[:40].plot(kind="bar", title=col_name)
    plt.show()



#%%
df_pat_ward_daily = df_clean[df_clean['PAT_WARD'] == "901AN331"]
#df_pat_ward_daily['date'] = pd.to_datetime(df_pat_ward_daily['date'])
#df_pat_ward_daily = df_pat_ward_daily.set_index('date')

df_pat_ward_daily.info()
df_filtered = df_pat_ward_daily.groupby(df_pat_ward_daily.index.date).count()
#%%
df_filtered.head()
df_filtered['EC_BG'].plot(x='date')






#%%--------------------------------------------------------------------------------
# TRANSFORMING/PROCESSING
#----------------------------------------------------------------------------------
# remove duplicates/NAs, 
# maybe imputation, but i think i have vals for everyday, so rather check for outliers?
# There is univariate (LOCF, NOCB) and multivariate imputation (sklearn: IterativeImputer)
# make STATIONARY! (if all models need that, otherwise make it a member function)
# splitting in test/training etc. here or as extra step/model step?

#DONE: load data from csv
#df_processed = load.load_data(path="data/02_intermediate/intermediate_output.csv")

# Proces....
#add external data (holidays weather (temp, precipitation), covid/influenca cases)
#NOTE: covid/grippe muss evnetuell imputiert werden da nur wöchentlich
#NOTE: kann gut zeigen, dass wien gleichen verlauf hat wie bundesländer, daher kann ich Ö-weite Daten
# nehmen, falls es keine wien-spezifischen Daten gibt.

# make daily aggregations for categorical variables
df_processed = transform.transform_data(df_clean)
#%%


#TODO: save data to csv

#%%--------------------------------------------------------------------------------
# DATA VIZ (EXPLORATION)
#----------------------------------------------------------------------------------

#TODO: save to csv

print(df_clean.columns)
#%%

df = data_model.Data(data=df_clean)
#%%
df.print_head()
df.plot_seasonal(plot_type='daily', col_name='count')



#%%
#Boxplots
df.plot_boxplots(col_name='count')
df.plot_seasonal_subseries(col_name='count') #NOTE: i think it works, but not enough dummy data.
#TODO: check if seasonal subseries plot works with multi-year data


#%%
#Decompose
df.decompose_one(col_name='count')
#df.decompose_all("count")

# mulitple decomposition (daily + weekly)
df.multiple_decompose(col_name="count", periods=[24, 24*7])





#%%
#Time series plots (acf, pacf etc)
df.plot_autocorrelation(col_name='count')
df.plot_partial_autocorrelation(col_name='count')


#%%
df.plot_daily_heatmap(col_name='count')





#%%--------------------------------------------------------------------------------
# MODEL BUILDING
#----------------------------------------------------------------------------------


#TODO: load data from csv


#TODO: look into OOP + config.yml
#sarima = Model1(config[0])
#sarima.split([80, 20])
#sarima.run(parma1, param2, parma3...)
#sarima.predict(x_days)
#sarima.test()
#sarima.evaluate()
#sarima.

#all of the above could be grouped into sarima.run() (if certain stuff is set up before, like vals for params and split!)



# if functional:
# sarima = statsmodels.sarima(xx, xx, xx) # oder so




#TODO: save data to csv




#%%--------------------------------------------------------------------------------
# DATA VIZ (FINISHED MODEL) 
#----------------------------------------------------------------------------------
# Plot prediction vs actual

# If OOP:
# sarima.plot_time()
# sarima.plot_polar()

# lstm.plot_time()
# etc. (could even loop: for obj in [sarima, lstm]: obj.plot_time() obj.plot_polar())

# if functional:

#plot_time(sarima)  # could also usy apply or similar to use list or loop: for mod in models: plot_time(mod)

#TODO: save to csv

#%%--------------------------------------------------------------------------------
# EVALUATION
#----------------------------------------------------------------------------------

# TODO: load data from csv

# evaluate ....
# print evaluations/tests like mae, mape, etc.


# TODO: save data to csv










































#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# SAMPLE DATA
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#%% ------------------------------------------------------------------------------
# Get data & clean
from sklearn.datasets import fetch_openml
df = fetch_openml("seoul_bike_sharing_demand", as_frame=True)
df = pd.DataFrame(df.frame)

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





#---------------------------------------------------------------------------------
# -- VISUALIZE OOP --
# --------------------------------------------------------------------------------
# Daily average ________________________________________________________________
# TODO: wrap in function (or add to Model?, because its already cleaned+processed here, so next step
# besides viz would be add to model anyway? BUT exploratory viz is done on raw data, so no specific model...
# TODO: make prettier: add title, colorchart (so i can later exchange colors), etc.


#%%
# Load data as Class Data:
#TODO: rename previous 'df' to 'df_preprocessed' or something, 
# to differentiate between Data object and DataFrame object
df = data_model.Data(data=df)
#%%
df.print_head()
df.plot_seasonal(plot_type='daily', col_name='count')



#%%
#Boxplots
df.plot_boxplots(col_name='count')
df.plot_seasonal_subseries(col_name='count') #NOTE: i think it works, but not enough dummy data.
#TODO: check if seasonal subseries plot works with multi-year data


#%%
#Decompose
df.decompose_one(col_name='count')
#df.decompose_all("count")

# mulitple decomposition (daily + weekly)
df.multiple_decompose(col_name="count", periods=[24, 24*7])





#%%
#Time series plots (acf, pacf etc)
df.plot_autocorrelation(col_name='count')
df.plot_partial_autocorrelation(col_name='count')


#%%
df.plot_daily_heatmap(col_name='count')


#%%

import holidays

vie_holidays = holidays.country_holidays('Austria', subdiv='W')


print(vie_holidays)

vie_holidays.get('2024-01-01')

print(vie_holidays.is_working_day('2024-01-01'))
print(vie_holidays.is_working_day('2024-12-24'))
print(vie_holidays.is_working_day('2005-12-25'))















#%% ------------------------------------------------------------------------------
# -- PROCESSING 2 --
# --------------------------------------------------------------------------------
#%%

# Order of differencing "d" -- detrending
#detrending/plotting:
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

plot_acf(df["count"])
#%% Detrend ______________________________________________________________________
f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("No differencing")
ax1.plot(df["count"])

ax2 = f.add_subplot(122)
plot_acf(df["count"].dropna(), ax=ax2)
plt.show()


f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("1st order differencing")
ax1.plot(df["count"].diff())

ax2 = f.add_subplot(122)
plot_acf(df["count"].diff().dropna(), ax=ax2)
plt.show()


f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("2nd order differencing")
ax1.plot(df["count"].diff().diff())

ax2 = f.add_subplot(122)
plot_acf(df["count"].diff().diff().dropna(), ax=ax2)
plt.show()


f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("3rd order differencing")
ax1.plot(df["count"].diff().diff().diff())

ax2 = f.add_subplot(122)
plot_acf(df["count"].diff().diff().diff().dropna(), ax=ax2)
plt.show()


#%% Dickey-Fuller Test
res = adfuller(df["count"].dropna())
print("p-value: ", res[1])

res = adfuller(df["count"].diff().dropna())
print("p-value: ", res[1])

res = adfuller(df["count"].diff().diff().dropna())
print("p-value: ", res[1])

res = adfuller(df["count"].diff().diff().diff().dropna())
print("p-value: ", res[1])

# Results:
# p-value:  0.0936973558926073
# p-value:  1.879297433670586e-25
# p-value:  1.4284641410804533e-16
# p-value:  1.762485024258537e-20
# significance level: 0.05
# so after one diff(), is good, data is stationary. above 0.5 is not stationary
# so we assume order of differencing d = 1
# from https://www.projectpro.io/article/how-to-build-arima-model-in-python/544



#%% Computing "p" -- Order of autoregressive Model
from statsmodels.graphics.tsaplots import plot_pacf

f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("No differencing")
ax1.plot(df["count"])

ax2 = f.add_subplot(122)
plot_pacf(df["count"].dropna(), ax=ax2)
plt.show()


f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("1st order differencing")
ax1.plot(df["count"].diff())

ax2 = f.add_subplot(122)
plot_pacf(df["count"].diff().dropna(), ax=ax2)
plt.show()


f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("2nd order differencing")
ax1.plot(df["count"].diff().diff())

ax2 = f.add_subplot(122)
plot_pacf(df["count"].diff().diff().dropna(), ax=ax2)
plt.show()

#In the pacf plot, we can see the first lag to be most significant.
# so "p" = 1
# Order of "q" = 1: 
# looking at acf (not pacf), we can also see, only
# first lag is most significant 

# So d, p, q = 1


#%% Fit arima model
from statsmodels.tsa.arima.model import ARIMA

arima_model = ARIMA(df["count"], order=(1,1,2))
model = arima_model.fit()
print(model.summary())

#%% plot
from statsmodels.graphics.tsaplots import plot_predict

#Method 1
fig, ax = plt.subplots()
ax = df["count"].plot(ax=ax)
plot_predict(model, ax=ax)
plt.show()

#Method 2
pred = model.predict(dynamic=False)
plt.plot(pred)
plt.plot(df["count"])


#%% SPLIT into training and test data
# Same as above but withs split data

from statsmodels.tsa.arima.model import ARIMA

#function for getting first x% of rows:
def get_split_rows(data, perc=0.8):
    n_rows_train = int(len(data)*perc)
    n_rows_test = len(data) - n_rows_train
    return (n_rows_train, n_rows_test)

split_n_rows = get_split_rows(df)
training_set = get_first_percent_rows(df[:split_n_rows[0]])

arima_model = ARIMA(training_set["count"], order=(1,1,2))
model = arima_model.fit()
print(model.summary())

# plot
from statsmodels.graphics.tsaplots import plot_predict
fig, ax = plt.subplots()
ax = df["count"].plot(ax=ax)
plot_predict(model, ax=ax)
plt.show()

#%% Run prediction on test set:

y_pred = pd.Series(model.forecast(split_n_rows[1])[0], index=df["count"][split_n_rows[0]:].index)
y_true = df["count"][split_n_rows[0]]

print(np.array(y_pred).astype(np.uint8))
print(np.array(y_true))


#%% Now trying from https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
from sklearn.metrics import mean_squared_error
from math import sqrt

# Split in train+test:
series = df["count"]
print(series)
#%%
# series.index = series.index.to_period('D')
X = series.values
perc = 0.66
size = int(len(X) * perc)

train, test = X[0:size], X[size:len(X)]
history = [x for x in train]

predictions = list()

#Walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,2))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    print("predicted = %f, expected=%f" % (yhat, obs))

#%% evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print("Test RMSE: %.3f" % rmse)

#plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color="red")
plt.show()



#%% ------------------------------------------------------------------------------
# -- RUN MODEL --

arima = model.ModelSarima(df)

#%% 
arima.df.head()
arima.df.info()

#%%
# arima.test_class_implementation()
#%%
# arima.split()

#%%
arima.fit("count", 1, 1, 1) #p=lag von signifikanz in autocorr.

#%% Show summary
print(arima.model_fit)
# arima.fit_summary()
#arima.make_stationary()
#arima.split()
#arima.create_model()
#arima.test_model()
#arima.evaluate_model()

