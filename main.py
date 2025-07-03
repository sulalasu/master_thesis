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
from src import process
from src import viz


#For developing purposes:
from importlib import reload # python 2.7 does not require this
print(load.__file__)
print(clean.__file__)


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

daily_average = df["count"].mean()

sns.lineplot(x="date", y="count", data=df)
plt.axhline(y=daily_average, color="black", label="average")

plt.show()


# Plot autocorrelation
viz.autocorr(df["count"])

#%% Weekly average  ________________________________________________________________


#%% Monthly average  ________________________________________________________________


# Decomposition  ________________________________________________________________
# viz.decompose_one(df, model="additive", column="count", period=7)




# mulitple decomposition (daily + weekly)
viz.multiple_decompose(df, col="count", periods=[24, 24*7])


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

# %%

#%%--------------------------------------------------------------------------------
# INPUT
#----------------------------------------------------------------------------------

# Read Data
df = load.load_data(path="data/01_raw/testdaten.tsv")

load.show_info(df=df)


#%%--------------------------------------------------------------------------------
# CLEANING 
#----------------------------------------------------------------------------------
#unify dates, columns etc. rename stuff

df = clean.clean_data(df)

#Check what unique vals are present in df
clean.check_unique_values(df)

df.to_csv(path_or_buf="./data/02_intermediate/intermediate_output.csv", sep=",")




#%%--------------------------------------------------------------------------------
# PROCESSING
#----------------------------------------------------------------------------------
# remove duplicates/NAs, 
# maybe imputation, but i think i have vals for everyday, so rather check for outliers?
# There is univariate (LOCF, NOCB) and multivariate imputation (sklearn: IterativeImputer)
# make STATIONARY! (if all models need that, otherwise make it a member function)
# splitting in test/training etc. here or as extra step/model step?

#TODO: load data from csv

# Proces....


#TODO: save data to csv

#%%--------------------------------------------------------------------------------
# DATA VIZ (EXPLORATION)
#----------------------------------------------------------------------------------

#TODO: save to csv



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



