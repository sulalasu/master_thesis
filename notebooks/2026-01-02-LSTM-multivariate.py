#%%
import pandas as pd
import numpy as np
import sys
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import sklearn.metrics as metrics #error metrics (mae, mape etc)
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import MSTL #multiple seasonal decompose
from statsmodels.tools.eval_measures import rmse
import sklearn.metrics as metrics #error metrics (mae, mape etc)

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import importlib
import holidays
#---------------------------------------
#column to model !
COLUMN = "use_transfused" #column of interest (Y)
#---------------------------------------


#---------------------------------------
#Fixed values:

START_DATE = "2021-05-01"
END_DATE = "2025-05-01"
plt.rcParams['figure.dpi'] = 200



# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src import viz

#from src import load


#
#Load cleaned data
df_original = pd.read_csv("../data/03_transformed/output_transformed.csv")#../data/03_transformed/output_transformed.csv")
df_original = df_original.set_index("date")
df_original.index = pd.to_datetime(df_original.index)



#%%
# ---------------------------------------------------------#
# MARK:                   LSTM NEU                         #
# ---------------------------------------------------------#
#without exogenous (atm)

# This tutorial
# https://www.youtube.com/watch?v=94PlBzgeq90

# 3 Types of GATES:
# Forget Gate
# Input Gate
# Output Gate



#%% load dataset
data = df_original.loc[START_DATE:END_DATE, [COLUMN]]#read_csv('pollution.csv', header=0, index_col=0)
data.head()
data.info()
data.describe()

#%% Prepare, build, run, viz model (univariate LSTM)

# ------------------------
# Prepare for LSTM Model
# ------------------------

dataset = data.values # convert to np array
# dataset_feature = data[COLUMN].values

training_data_len = int(np.ceil(len(dataset) * 0.95)) # 95% of dataset

# Preprocessing stages
scaler = StandardScaler()
# scaled_data = scaler.fit_transform(dataset) #TODO: this needs to be training only,otherwise information leak into scaler
scaler = scaler.fit(dataset[:training_data_len]) #uses only train data for fitting, to avoid info leak to train data

scaled_data = scaler.transform(dataset) 
# scaled_data_feature = scaler.transform(dataset_feature.reshape(dataset_feature.shape[0], 1)) #feature = y value to predict

training_data = scaled_data[:training_data_len] #95% of our data

#----------------------------------
#prep training features
#what we actually give the model to learn
X_train, y_train = [], []

# Create sliding window for our data (60days)
sliding_size = 60
forecast_days = 0 #days more than on day ahead
for i in range(sliding_size, len(training_data)):
    X_train.append(training_data[i - sliding_size:i, 0])
    y_train.append(training_data[i + forecast_days, 0])

#convert lists to np arrays (for tensorflow, needs arrays)
X_train, y_train = np.array(X_train), np.array(y_train)
#change to 3D array, tensorflow needs this to work better
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))


#----------------------------------
#Prep test data
test_data = scaled_data[training_data_len - sliding_size : ] #[training_data_len: ]
X_test, y_test = [], dataset[training_data_len : ]

#Set the input for the test model (the "past/available" data)
for i in range(sliding_size, len(test_data)):
    X_test.append(test_data[i - sliding_size : i + forecast_days, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))





# ------------------------
# Build the model
# ------------------------

# Model with 5 Layers: 
# LSTM Layer 1 -> Layer 2 -> Layer 3 -> Dense Layer -> Dropout Layer -> Final Layer


# Actual model
memory_cells = 128 #32, 64, 128, etc
model_lstm = keras.models.Sequential()

# - - - - - 
# LAYERS 
# - - - - - 

# LSTM Layer 1
# First Layer helps understands model understand patterns
#return_sequence -> give back "ideas" after its done (list of "ideas")
model_lstm.add(keras.layers.LSTM(memory_cells, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# LSTM Layer 2
# return_sequence = False --> only give back single output
#gives back final prediciton, so dont give everything back
model_lstm.add(keras.layers.LSTM(64, return_sequences=False))

# Dense Layer
# Turns patterns into a decision (decision making)
# 128 = neurons to make final decision
# activation = relu --> non-linearity (recitvied linear unit)
# this layer helps model complicated patterns
model_lstm.add(keras.layers.Dense(128, activation="relu"))


#Dropout Layer
#0.5 --> randomly drops out 50% of neurons during traiing --> prevents overfitting
# keeps model from being to sensitive to training data
model_lstm.add(keras.layers.Dropout(0.5))

#Final Layer (Dense layer as well)
# (final decision making)
# simple layer with 1 neuron that outputs one value -- the predicted value
model_lstm.add(keras.layers.Dense(1)) 


model_lstm.summary()

#Model compilation
# puts all pieces together
# compiles moodel to tellit how to learn.
model_lstm.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])



# ------------------------
# Train the model
# ------------------------

#epochs = hwo many times is it gonna run to find the best solution
# batch size = how much data is in each batch when it runs
training = model_lstm.fit(X_train, y_train, epochs=20, batch_size=64)
plt.plot(training.history["loss"], label="uv training loss")



# ------------------------
# Prediction
# ------------------------


# Make predictions
predictions_lstm = model_lstm.predict(X_test)
predictions_lstm = scaler.inverse_transform(predictions_lstm)


# ------------------------
# Plot results
# ------------------------

train = data[ : training_data_len]
test = data[training_data_len : ]

test = test.copy()

test["Predictions"] = predictions_lstm

plt.figure(figsize=(16,8))
plt.plot(train.loc["2024-10-01": ].index, 
		 train.loc["2024-10-01": ]["use_transfused"], 
		 label="Train (Actual)", 
		 color = "blue")
plt.plot(test.index, test["use_transfused"], label="Test (Actual)", color = "orange")
plt.plot(test.index, test["Predictions"], label="Predictions", color = "red")
plt.title("EC trnasufison prediction")
plt.xlabel("Date")
plt.ylabel("number of EC")
plt.legend()



#%% ------------------------
# Get evaluation values
# ------------------------
print("LSTM")
print("RMSE:", metrics.root_mean_squared_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MAPE:", metrics.mean_absolute_percentage_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MAE: ", metrics.mean_absolute_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MdAE:", metrics.median_absolute_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MaxE:", metrics.max_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))










#%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# MARK: MULTIVARIATE LSTM
# xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX




#%% load dataset
data = df_original.loc[
    START_DATE:END_DATE, 
    [COLUMN, "use_discarded", "use_expired", 'ward_AN',
     'ward_CH', 'ward_I1', 'ward_I3', 'ward_Other', 'ward_UC', 
     "workday_enc", "holiday_enc", "day_of_week", "day_of_year", "year", "tlmin", "tlmax"]
    ]
data.head()
data.info()
data.describe()

col_Y = "use_transfused" #column to predict (y)
cols_multivariate = ["use_discarded", "used_expired"]



#%% ------------------------
# Prepare for multivariate LSTM Model
# ------------------------



dataset = data.values # convert to np array

training_data_len = int(np.ceil(len(dataset) * 0.95)) # 95% of dataset

# Preprocessing stages
scaler = StandardScaler()
# scaled_data = scaler.fit_transform(dataset[:training_data_len, :]) #TODO: this needs to be training only,otherwise information leak into scaler
scaler = scaler.fit(dataset[:training_data_len, :]) #TODO: this needs to be training only,otherwise information leak into scaler
scaled_data = scaler.transform(dataset) 



#----------------------------------
# Prep training features
#what we actually give the model to learn
# wher x train are the input variables and y train is the variable, the model uses in the training stage
# to adjust weights and baises to conform to the result (compare result in training to y_train)

training_data = scaled_data[:training_data_len, :] #95% of our data
X_train, y_train = [], []

# Create sliding window for our data (60days)
sliding_size = 60 #
forecast_days = 0 #days more than on day ahead
for i in range(sliding_size, len(training_data)):
    X_train.append(training_data[i - sliding_size:i, :])
    y_train.append(training_data[i + forecast_days, 0]) #0 = the position of our variable to forecast, is only one variable, as we only forecast COLUMN
    

#convert lists to np arrays (for tensorflow, needs arrays)
X_train, y_train = np.array(X_train), np.array(y_train)
#change to 3D array, tensorflow needs this to work better
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))


#----------------------------------
#Prep test data
# Test data again contains all input variables (multivariate) and one actual y for testing predictions
test_data = scaled_data[training_data_len - sliding_size : ]
X_test, y_test = [], dataset[training_data_len : ]

# for i in range(sliding_size, len(test_data)): #sliding_size
for i in range(sliding_size, len(test_data)): #sliding_size
    X_test.append(test_data[i - sliding_size : i + forecast_days, :])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))




#%% ------------------------
# Build the model
# ------------------------

# Model with 5 Layers: 
# LSTM Layer 1 -> Layer 2 -> Layer 3 -> Dense Layer -> Dropout Layer -> Final Layer


# Actual model
memory_cells = 64 #32, 64, 128, etc
model_lstm_mv = keras.models.Sequential() #mv = multivariate

# LSTM Layers

#this is different in multivariate (mv) lstm: the input shape
model_lstm_mv.add(keras.layers.LSTM(memory_cells, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model_lstm_mv.add(keras.layers.LSTM(64, return_sequences=False))
model_lstm_mv.add(keras.layers.Dense(128, activation="relu"))
model_lstm_mv.add(keras.layers.Dropout(0.5))
model_lstm_mv.add(keras.layers.Dense(1)) 

model_lstm_mv.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])
model_lstm_mv.summary()



#%% ------------------------
# Train the model
# ------------------------

#epochs = hwo many times is it gonna run to find the best solution
# batch size = how much data is in each batch when it runs
training = model_lstm_mv.fit(X_train, y_train, epochs=20, batch_size=32)

#Plot training:
plt.plot(training.history["loss"], label="training loss")

#%% ------------------------
# Testing + prediction
# --------------------------


# Make predictions
predictions_lstm_mv = model_lstm_mv.predict(X_test)
#has to be same shape as scaler.transform for inverse_transform, so
# just repeat and then slice.
predictions_lstm_mv_copies = np.repeat(predictions_lstm_mv, X_test.shape[2], axis=-1)
# predictions_lstm_mv = scaler.inverse_transform(predictions_lstm_mv)
predictions_lstm_mv = scaler.inverse_transform(predictions_lstm_mv_copies)[ : ,0]


#%% ------------------------
# Plot results
# ------------------------

train = data[ : training_data_len]
test = data[training_data_len : ]

test = test.copy()

test["Predictions"] = predictions_lstm_mv

plt.figure(figsize=(16,8))
plt.plot(train.loc["2024-10-01": ].index, 
		 train.loc["2024-10-01": ]["use_transfused"], 
		 label="Train (Actual)", 
		 color = "blue")
plt.plot(test.index, test["use_transfused"], label="Test (Actual)", color = "orange")
plt.plot(test.index, test["Predictions"], label="Predictions", color = "red")
plt.title("EC transfusion prediction")
plt.xlabel("Date")
plt.ylabel("number of EC")
plt.legend()



#%% ------------------------
# Get evaluation values
# ------------------------
print("LSTM")
print("RMSE:", metrics.root_mean_squared_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MAPE:", metrics.mean_absolute_percentage_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MAE: ", metrics.mean_absolute_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MdAE:", metrics.median_absolute_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))
print("MaxE:", metrics.max_error(y_pred=test["Predictions"], y_true=test["use_transfused"]))











# xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#
#
# MARK: LSTM MV PI
# Multivariate LSTM with prediction intervals
#
#
# xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

#%% load dataset
data = df_original.loc[
    START_DATE:END_DATE, 
    [COLUMN, "use_discarded", "use_expired", 'ward_AN',
     'ward_CH', 'ward_I1', 'ward_I3', 'ward_Other', 'ward_UC', 
     "workday_enc", "holiday_enc", "day_of_week", "day_of_year", "year", "tlmin", "tlmax"]
    ]


#%% ------------------------
# Prepare for multivariate LSTM Model
# ------------------------

dataset = data.values 
training_data_len = int(np.ceil(len(dataset) * 0.95)) # 95% of dataset

#Setting features (all vars incl. target var) & target var as numpy arrays:
X_raw = dataset
y_raw = data[COLUMN].values.reshape(-1, 1)


# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

# Preprocessing stages
scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X.fit(X_raw[:training_data_len, :]) #DONE: this needs to be training only,otherwise information leak into scaler
scaler_y.fit(y_raw[:training_data_len, :]) #DONE: this needs to be training only,otherwise information leak into scaler

scaled_X = scaler_X.transform(X_raw) 
scaled_y = scaler_y.transform(y_raw) 


# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

#----------------------------------
# Create sliding window for our data (60days)
sliding_size = 365 
forecast_days = 3 #days more than on day ahead

# Prep training features
X_train, y_train = [], []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
for i in range(sliding_size, training_data_len - forecast_days): #(sliding_size, training_data_len):
    X_train.append(scaled_X[i - sliding_size : i, :])
    y_train.append(scaled_y[i : i + forecast_days, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)


#convert lists to np arrays (for tensorflow, needs arrays)


# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

#----------------------------------
#Prep test data
# test_data = scaled_data[training_data_len - sliding_size : ]
X_test, y_test_raw = [], []#dataset[training_data_len : ]

for i in range(training_data_len, len(scaled_X) - forecast_days): #sliding_size, len(test_data)):
    X_test.append(scaled_X[i - sliding_size : i , :])
    y_test_raw.append(y_raw[i : i + forecast_days, 0])

X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
y_test_raw = np.array(y_test_raw)



# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

#%% ------------------------
# Build the model
# ------------------------
from keras import Input, layers, Model

memory_cells = 64 #32, 64, 128, etc
activation_fct = "relu"

#FUNCTIONAL API
# Syntax: the object inside (brackets) gets used as input into function defined beforehand like y = f(x)
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = layers.LSTM(memory_cells, return_sequences=True)(inputs)
x = layers.LSTM(memory_cells, return_sequences=False)(x)
x = layers.Dense(2*memory_cells, activation=activation_fct)(x)
#MC dropout
x = layers.Dropout(0.5)(x, training=True)

#output layer: x neurons for x days forecasting
outputs = layers.Dense(forecast_days)(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])

# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

#Training the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA


#%% 
# Run x times to get Prediction intervals:
n_iterations = 100
all_predictions = []

for _ in range(n_iterations):
    print(f"Iteration {_}")
    all_predictions.append(
        scaler_y.inverse_transform(model(X_test, training=True, verbose=0).numpy())
    )
    # all_predictions.append(
    #     scaler_y.inverse_transform(model.predict(X_test, verbose=0))
    # )

all_predictions = np.array(all_predictions)



forecast_mean = np.mean(all_predictions, axis=0)
forecast_lower = np.percentile(all_predictions, 2.5, axis=0)
forecast_upper = np.percentile(all_predictions, 97.5, axis=0)

# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA


#%%
# Test different results for different model runs (result of Dropout + training=true)

# sample = X_test[0:1]
# out1  = model(sample, training=True).numpy()
# out2  = model(sample, training=True).numpy()

# print(f"Prediction 1: {out1}")
# print(f"Prediction 2: {out2}")
# print(f"Are they exactly the same? {np.array_equal(out1, out2)}")

#%% 
# Get results into dict of DFs
# One df for every 'X days ahead', containing lower, upper limits, mean and actual values
test_idx_start = training_data_len
test_idx_end = len(dataset) - forecast_days

results = {}
for day in range(1, forecast_days + 1):
    day_label = f"Day_{day}"

    results[day_label] = pd.DataFrame(
        index = data.index[test_idx_start:test_idx_end]
    )


# Fill empty (index only) dfs:
for day in range(forecast_days):
    day_label = f"Day_{day+1}"

    day_predictions = all_predictions[:, :, day]

    results[day_label]["Actual"] = y_test_raw[:, day]
    results[day_label]["Mean"] = np.mean(day_predictions, axis=0)
    results[day_label]["Lower"] = np.percentile(day_predictions, 2.5, axis=0)
    results[day_label]["Upper"] = np.percentile(day_predictions, 97.5, axis=0)

print(results["Day_1"].head())


#%%
# Plotting:
plt.figure(figsize=(15, 7))

# Actual Future (using Day 1 actuals as the baseline)
plt.plot(results["Day_1"]["Actual"], label="Actual", color="black", alpha=0.6, linewidth=2)

# Day 1 Forecast + Prediction Interval
plt.plot(results["Day_1"]["Mean"], label="1-Day Ahead Forecast", color="blue")
plt.fill_between(results["Day_1"].index, 
                 results["Day_1"]["Lower"], 
                 results["Day_1"]["Upper"], 
                 color="blue", alpha=0.15, label="1-Day PI")

# Day 2 and 3 Forecasts (Lines only)
# plt.plot(results["Day_2"]["Mean"], label="2-Days Ahead", color="green", linestyle="--")
# plt.plot(results["Day_3"]["Mean"], label="3-Days Ahead", color="red", linestyle="--")

plt.title(f"Multivariate LSTM: Forecast Horizons for {COLUMN}")
plt.ylabel("Value")
plt.legend(loc="upper left")
plt.show()

#%%
# Error values:

print("LSTM multivariate with Prediction intervals (not shown)")
print("RMSE:", metrics.root_mean_squared_error(y_pred=results["Day_1"]["Mean"], y_true=results["Day_1"]["Actual"]))
print("MAPE:", metrics.mean_absolute_percentage_error(y_pred=results["Day_1"]["Mean"], y_true=results["Day_1"]["Actual"]))
print("MAE: ", metrics.mean_absolute_error(y_pred=results["Day_1"]["Mean"], y_true=results["Day_1"]["Actual"]))
print("MdAE:", metrics.median_absolute_error(y_pred=results["Day_1"]["Mean"], y_true=results["Day_1"]["Actual"]))
print("MaxE:", metrics.max_error(y_pred=results["Day_1"]["Mean"], y_true=results["Day_1"]["Actual"]))





# day_idx = 0
# time_steps = np.arange(forecast_days)

# plt.figure(figsize=(16,8))
# plt.plot(time_steps, y_test[day_idx], label="Actual future")
# plt.plot(time_steps, forecast_mean[day_idx], label="Mean forecast")
# plt.fill_between(time_steps,
#                  forecast_lower[day_idx],
#                  forecast_upper[day_idx],
#                  color="grey", alpha=0.2, label="95% PI"
#                  )
# plt.title("LSTM forecast with 95% Prediction interval")
# plt.xlabel("Days ahead")
# plt.ylabel(COLUMN)
# plt.legend()
# plt.show()

# SEQUENTIAL API
# LSTM Layers
# model_lstm_mv_pi = keras.models.Sequential() #mv = multivariate, pi = prediction interval
# model_lstm_mv_pi.add(LSTM(memory_cells, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model_lstm_mv_pi.add(LSTM(64, return_sequences=False))
# model_lstm_mv_pi.add(Dense(128, activation="relu"))
# model_lstm_mv_pi.add(Dropout(0.5))
# model_lstm_mv_pi.add(Dense(1)) 
# model_lstm_mv_pi.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])


# %%





#%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# MARK: LSTM UNIVAR PRED INTERVALLS
# Univariate LSTM with prediction intervals
# xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#%% load dataset
data = df_original.loc[
    START_DATE:END_DATE, 
    [COLUMN]
]


#%% ------------------------
# Prepare for multivariate LSTM Model
# ------------------------

dataset = data.values 
training_data_len = int(np.ceil(len(dataset) * 0.95)) # 95% of dataset

#Setting features (all vars incl. target var) & target var as numpy arrays:
X_raw = dataset
y_raw = data[COLUMN].values.reshape(-1, 1)

# Preprocessing stages
scaler_X = StandardScaler()
scaler_y = StandardScaler()

scaler_X.fit(X_raw[:training_data_len, :]) #TODO: this needs to be training only,otherwise information leak into scaler
scaler_y.fit(y_raw[:training_data_len, :]) #TODO: this needs to be training only,otherwise information leak into scaler

scaled_X = scaler_X.transform(X_raw) 
scaled_y = scaler_y.transform(y_raw) 



#----------------------------------
# Create sliding window for our data (60days)
sliding_size = 120 
forecast_days = 3 #days more than on day ahead

# Prep training features
X_train, y_train = [], []
for i in range(sliding_size, training_data_len - forecast_days): #(sliding_size, training_data_len):
    X_train.append(scaled_X[i - sliding_size : i, :])
    y_train.append(scaled_y[i : i + forecast_days, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)


#convert lists to np arrays (for tensorflow, needs arrays)


#----------------------------------
#Prep test data
# test_data = scaled_data[training_data_len - sliding_size : ]
X_test, y_test_raw = [], []#dataset[training_data_len : ]

for i in range(training_data_len, len(scaled_X) - forecast_days): #sliding_size, len(test_data)):
    X_test.append(scaled_X[i - sliding_size : i , :])
    y_test_raw.append(y_raw[i : i + forecast_days, 0])

X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
y_test_raw = np.array(y_test_raw)



#%% ------------------------
# Build the model
# ------------------------
from keras import Input, layers, Model

memory_cells = 64 #32, 64, 128, etc
activation_fct = "relu"

#FUNCTIONAL API
# Syntax: the object inside (brackets) gets used as input into function defined beforehand like y = f(x)
inputs2 = Input(shape=(X_train.shape[1], X_train.shape[2]))
x2 = layers.LSTM(memory_cells, return_sequences=True)(inputs2)
x2 = layers.LSTM(memory_cells, return_sequences=False)(x2)
x2 = layers.Dense(2*memory_cells, activation=activation_fct)(x2)
#MC dropout
x2 = layers.Dropout(0.5)(x2, training=True)

#output layer: x neurons for x days forecasting
outputs2 = layers.Dense(forecast_days)(x2)

model2 = keras.Model(inputs2, outputs2)
model2.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])

#Training the model
model2.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)


#%% 
# Run x times to get Prediction intervals:
n_iterations = 100
all_predictions2 = []

for _ in range(n_iterations):
    print(f"Iteration {_}")
    all_predictions2.append(
        scaler_y.inverse_transform(model2(X_test, training=True, verbose=0).numpy())
    )
    # all_predictions.append(
    #     scaler_y.inverse_transform(model.predict(X_test, verbose=0))
    # )

all_predictions2 = np.array(all_predictions2)



forecast_mean2 = np.mean(all_predictions2, axis=0)
forecast_lower2 = np.percentile(all_predictions2, 2.5, axis=0)
forecast_upper2 = np.percentile(all_predictions2, 97.5, axis=0)
#%%
# Test different results for different model runs (result of Dropout + training=true)

# sample = X_test[0:1]
# out1  = model(sample, training=True).numpy()
# out2  = model(sample, training=True).numpy()

# print(f"Prediction 1: {out1}")
# print(f"Prediction 2: {out2}")
# print(f"Are they exactly the same? {np.array_equal(out1, out2)}")

#%% 
# Get results into dict of DFs
# One df for every 'X days ahead', containing lower, upper limits, mean and actual values
test_idx_start = training_data_len
test_idx_end = len(dataset) - forecast_days

results2 = {}
for day in range(1, forecast_days + 1):
    day_label = f"Day_{day}"

    results2[day_label] = pd.DataFrame(
        index = data.index[test_idx_start:test_idx_end]
    )


# Fill empty (index only) dfs:
for day in range(forecast_days):
    day_label = f"Day_{day+1}"

    day_predictions = all_predictions2[:, :, day]

    results2[day_label]["Actual"] = y_test_raw[:, day]
    results2[day_label]["Mean"] = np.mean(day_predictions, axis=0)
    results2[day_label]["Lower"] = np.percentile(day_predictions, 2.5, axis=0)
    results2[day_label]["Upper"] = np.percentile(day_predictions, 97.5, axis=0)

print(results2["Day_1"].head())


#%%
# Plotting:
plt.figure(figsize=(15, 7))

# Actual Future (using Day 1 actuals as the baseline)
plt.plot(results2["Day_1"]["Actual"], label="Actual", color="black", alpha=0.6, linewidth=2)

# Day 1 Forecast + Prediction Interval
plt.plot(results2["Day_1"]["Mean"], label="1-Day Ahead Forecast", color="blue")
plt.fill_between(results2["Day_1"].index, 
                 results2["Day_1"]["Lower"], 
                 results2["Day_1"]["Upper"], 
                 color="blue", alpha=0.15, label="1-Day PI")

# Day 2 and 3 Forecasts (Lines only)
# plt.plot(results2["Day_2"]["Mean"], label="2-Days Ahead", color="green", linestyle="--")
# plt.plot(results2["Day_3"]["Mean"], label="3-Days Ahead", color="red", linestyle="--")

plt.title(f"Univariate LSTM: Forecast Horizons for {COLUMN}")
plt.ylabel("Value")
plt.legend(loc="upper left")
plt.show()

#%%
# Error values:

print("LSTM univariate with Prediction intervals (not shown)")
print("RMSE:", metrics.root_mean_squared_error(y_pred=results2["Day_1"]["Mean"], y_true=results2["Day_1"]["Actual"]))
print("MAPE:", metrics.mean_absolute_percentage_error(y_pred=results2["Day_1"]["Mean"], y_true=results2["Day_1"]["Actual"]))
print("MAE: ", metrics.mean_absolute_error(y_pred=results2["Day_1"]["Mean"], y_true=results2["Day_1"]["Actual"]))
print("MdAE:", metrics.median_absolute_error(y_pred=results2["Day_1"]["Mean"], y_true=results2["Day_1"]["Actual"]))
print("MaxE:", metrics.max_error(y_pred=results2["Day_1"]["Mean"], y_true=results2["Day_1"]["Actual"]))





# day_idx = 0
# time_steps = np.arange(forecast_days)

# plt.figure(figsize=(16,8))
# plt.plot(time_steps, y_test[day_idx], label="Actual future")
# plt.plot(time_steps, forecast_mean[day_idx], label="Mean forecast")
# plt.fill_between(time_steps,
#                  forecast_lower[day_idx],
#                  forecast_upper[day_idx],
#                  color="grey", alpha=0.2, label="95% PI"
#                  )
# plt.title("LSTM forecast with 95% Prediction interval")
# plt.xlabel("Days ahead")
# plt.ylabel(COLUMN)
# plt.legend()
# plt.show()

# SEQUENTIAL API
# LSTM Layers
# model_lstm_mv_pi = keras.models.Sequential() #mv = multivariate, pi = prediction interval
# model_lstm_mv_pi.add(LSTM(memory_cells, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model_lstm_mv_pi.add(LSTM(64, return_sequences=False))
# model_lstm_mv_pi.add(Dense(128, activation="relu"))
# model_lstm_mv_pi.add(Dropout(0.5))
# model_lstm_mv_pi.add(Dense(1)) 
# model_lstm_mv_pi.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])


# %%
