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



#%%
#fit with sliding window
class ModelSarimaExpand(ModelSarima):
    #in general:
    # model = ARIMA(series, order=(p, d, q))
    # model_fit = model.fit()
    # print(model_fit.summary())
    # output = model_fit.forecast()

    def set_params(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q

    def make_model(self, p, d, q):
        return ARIMA(self.train_data, order=(p,d,q))
    


    def run(self, perc, p, d, q, window):
        """
        perc = percentage test set size. should be <0.5
        p, d, q = params for ARIMA
        window = size (in days) of cross validation rolling/expanding window.
        """
        if window < 1 or type(window) != int:
            raise IndexError("window must be int, 1 or bigger")
        
        #generate train/test set
        self.split_by_percentage(perc)
        train = self.train_data
        history = train
        test = self.test_data

        print(train)
        print(test)

        output_list = []
        predictions = []
        #iterate through test set, make model, fi model, forecast, save results
        for i in range(len(test) - window + 1):
            print(i)
            model = ARIMA(history, order=(p,d,q))
            
            model_fit = model.fit()
            print(model_fit.summary())
            
            output = model_fit.forecast(steps = window)
            output_list.append(output)

            print(f"output:\n{output}\n\noutput[0]: {output[0]}\n\n")

            #predictions = predictions.append()
        print(f"'output' has {len(output_list)} entries with a window size of {window}.\nPredictions are {[x[0] for x in output_list]}")

#%%

m = ModelSarimaExpand(df_daily)



#split
m.split_by_percentage(0.66)
print(m.train_data)
print(m.test_data)


#%%
m.run(0.01, 1, 1, 1, 1)


# %%
