#%%
import os
import pandas as pd
import numpy as np
from numpy import nan
from time import time
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels
from statsmodels.tsa.arima.model import ARIMA



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

from src.model import ModelSarima

print(load.__file__)
print(clean.__file__)

#%%

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# SAMPLE DATA
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#%% ------------------------------------------------------------------------------
# Get data & clean
file_path = "../data/00_sample_data/electricity.csv"
if os.path.isfile(file_path):
    print("File exists.")
    raw_df = pd.read_csv(file_path)
    print("loaded file data")
    print(raw_df.head())
else:
    from sklearn.datasets import fetch_openml
    raw_df = fetch_openml("Electricity-hourly", as_frame=True)
    raw_df = pd.DataFrame(raw_df.frame)
    print("loaded from sklearn")
    print(raw_df.head())
    raw_df.to_csv(file_path)
    print("written file to ", file_path)
#%%
#Drop all except date, value_N
col = 'value_5'
df = raw_df[["date", col]]
df = df.rename(columns={col : "value"})
df['date'] = pd.to_datetime(df['date']).dt.date.astype('datetime64[ns]') #alternative: .normalize() sets all to midnight 00:00:00


#Resample daily:
df_daily = df.set_index("date")
df_daily = df_daily["value"].resample("D").sum()
df_daily = df_daily.to_frame()


#%% Make DATA class for dataframe
# so i can add methods for plotting etc.

class Data():
    # Class for the processed data, which contains methods for plotting
    # and transforming, which will be passed into class 'Model'
    # TODO: change parameter of model to be of 'Data' type 

    def __init__(self, data: pd.DataFrame):
        self.data = data

    #Methods:
    def plot_line(self):
        #simple line plot
        pass

    def plot_seasonal(self):
        #seasonal plot
        pass

    def plot_seasonal_subseries(self):
        #plot seasonal subseries
        pass
 
    def plot_acf(self):
        pass

    def plot_pacf(self):
        pass

    def plot_overview(self):
        # aggregate method to plot multiple functions
        pass





#%%
#fit with sliding window
class ModelSarimaExpand(ModelSarima):
    #in general:
    # model = ARIMA(series, order=(p, d, q))
    # model_fit = model.fit()
    # print(model_fit.summary())
    # output = model_fit.forecast()

    #really not needed!:
    # def set_params(self, p, d, q):
    #     self.p = p
    #     self.d = d
    #     self.q = q

    # Also not needed, since is done on rolling basis:
    # def make_model(self, p, d, q):
    #     self.model = ARIMA(self.train_data, order=(p,d,q))
    
    # def fit(self):
    #     self.model_fit = self.model.fit()


    def run(self, split_perc, p, d, q, window_size, horizon):
        """
        perc = percentage test set size. should be <0.5
        p, d, q = params for ARIMA
        window : how many observations to keep from past (in days; for rolling window). 
        horizon : how far to look ahead
        """
        if window_size < 1 or type(window_size) != int:
            raise IndexError("'window_size must be int, 1 or bigger")
        
        #generate train/test set
        self.split_by_percentage(split_perc)
        train = self.train_data
        history = train
        test = self.test_data


        output_list = []
        predictions = []

        self.forecast_expanding = []

        self.forecast_series = pd.Series()
        # Expanding window (iterate through indices, 
        # keep distance to forecast horizon at the end
        for i in range(self.split_index, len(self.data)-horizon):
            train = train.reset_index()
            print(train.head())
            train = self.data[:i]
            test = self.data[i:]

            #model fit
            model_expanding_fit = ARIMA(train, order=(p,d,q)).fit()

            #model predict
            expanding_prediction = model_expanding_fit.forecast(steps=horizon)
            print("\nexpanding pred:\n")
            print(expanding_prediction)
            print(expanding_prediction.reset_index())
            print(expanding_prediction[0])
            print(type(expanding_prediction))
            print()
            self.forecast_series = pd.concat([self.forecast_series, expanding_prediction])
            #summary/result:
            print(f"output:\n{expanding_prediction}\n\noutput[0]: {expanding_prediction[0]}\n\n")
            print(self.forecast_series)

        #model plot:
        
        print(f"forecast series: {self.forecast_series}")
        #TODO: add as kwargs:
        width = 20
        height = 10
        dpi = 300

        fig, ax = plt.subplots(figsize=(width, height), dpi = dpi)
        ax.plot(self.data)
        ax.plot(self.forecast_series, color="orange")
        plt.show()


        #iterate through test set indices, make model with train data, fit model, 
        # forecast (window size), save results
        # for i in range(len(test) - window_size + 1):
        #     print(self.data[:])
        #     model_expanding = ARIMA(self.data[:self.split_index+i], order=(p,d,q)).fit()
        #     pred_expanding = model_expanding.forecast(steps=horizon)
        #     forecast_expanding.append(pred_expanding)

        #     model_fit = ARIMA(history, order=(p,d,q)).fit()
        #     print(model_fit.summary())

        #     output = model_fit.forecast(steps = rolling_window)
        #     output_list.append(output)

        #     print(f"output:\n{output}\n\noutput[0]: {output[0]}\n\n")

        #     #predictions = predictions.append()
        # print(f"'output' has {len(output_list)} entries with a window size of {window}.\nPredictions are {[x[0] for x in output_list]}")

#%%

m = ModelSarimaExpand(df_daily)



#split
#m.split_by_percentage(0.99)
#print(m.train_data)
#print(m.test_data)


m.run(0.90, 1, 1, 1, 1, 1)




# %%
