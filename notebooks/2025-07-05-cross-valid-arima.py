#%%
import pandas as pd
import numpy as np
from numpy import nan
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels
from statsmodels.tsa.arima.model import ARIMA



import sys
from pathlib import Path
# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))


#print something
from src import load
from src import clean
from src import viz
from src import config
from src import model
from src import data_model

from src.model import ModelSarima



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
df.plot_line('count')

#---------------------------------------------------------------------------------
# -- IMPLEMENTATION OF ARIMA WITH EXPANDING WINDOW: --
# --------------------------------------------------------------------------------

#%%
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# %%
length = len(df.data)

def get_split_index_by_prct(df_len, prct: float=0.77):
    #returns index of split position

    if prct <= 0 or prct > 1:
        raise ValueError("must be between 0 and 1, not 0")
    if df_len <= 2:
        raise ValueError("df must be longer than 2 rows")
    
    l = df_len
    idx = int(l*prct)

    return idx


def rolling_window(df, train_len: int, test_len: int):
    #split df into train and test set, by rolling window (same length
    # of history 'rolling' over data): in data 11-10 with train_len=3, test_len=2:
    # [1, 2, 3][4, 5] 6, 7, 8, 9, 10 
    #  1 [2, 3, 4][5, 6] 7, 8, 9, 10 
    #  1, 2 [3, 4, 5][6, 7] 8, 9, 10
    #  1, 2, 3 [4, 5, 6][7, 8] 9, 10
    #  1, 2, 3, 4 [5, 6, 7][8, 9] 10
    #  1, 2, 3, 4, 5 [6, 7, 8][9, 10]
    start_idx = train_len 
    end_idx = len(df) - test_len + 1 

    for split_idx in range(start_idx, end_idx):

        #use iloc to get a view, not a copy (like you would get with df[n:m])
        train_set = df.iloc[split_idx-train_len : split_idx]
        test_set = df.iloc[split_idx : split_idx+test_len]

        
        # print(f"\ntrain set ({len(train_set)}):\n{train_set}")
        # print(f"\ntest_set ({len(test_set)}):\n{test_set}")

        yield train_set, test_set


def expanding_window(df, split_percent: float, test_len: int):
    #TODO: move to top of file of respective class file
    from sklearn.model_selection import TimeSeriesSplit

    #create expanding window for cross validation.
    # pass percentage for split in data (0-1), as well as pred_size, which is the number
    # of rows to look ahead.

    #index where to split/start the expanding window
    start_idx = get_split_index_by_prct(len(df), split_percent)
    end_idx = len(df) - test_len + 1

    res = []

    for split_idx in range(start_idx, end_idx):
        train_set = df.iloc[:split_idx]
        test_set = df.iloc[split_idx:split_idx+test_len]

        # print(f"\nsplit index: {split_idx}")
        # print(f"train set ({len(train_set)}):\n{train_set}")
        # print(f"\ntest_set ({len(test_set)}):\n{test_set}\n")

        # yield train_set, test_set
        res.append([train_set, test_set])
    return res

#%%
#Test expanding window:
testing_df = pd.DataFrame({'value': list(range(30))}, index=pd.date_range(start='2023-01-01', periods=30, freq='D'))
sets = expanding_window(testing_df, 0.4, 2)
print(sets[0])
print(sets[0][1])

# x = expanding_window(testing_df, 0.4, 2)
# for z in x:
#     print(z[0], "\n", z[1])
#     print("\n\n")



#%%
# Test rolling window
testing_df = pd.DataFrame({'value': list(range(10))}, index=pd.date_range(start='2023-01-01', periods=10, freq='D'))
x = rolling_window(testing_df, 3, 2)
for z in x:
    print(z[0], "\n", z[1])
    print("\n\n")


#%%
# ARIMA
look_ahead = 4
train_test_sets = expanding_window(df.data['count'], 0.8, test_len=look_ahead)


#Get date range for total prediction interval, create empty df with datetime index
# print(train_test_sets[0][1].head(1).index)
# print(train_test_sets[0][1].index.min())
res_date_range = pd.date_range(start=train_test_sets[0][1].index.min(), end=train_test_sets[-1][1].index.max())
column_names = [str(num) for num in range(1, look_ahead+1)]
res_days_ahead_df = pd.DataFrame(index=res_date_range, columns=column_names)

for i, sets in enumerate(train_test_sets):
    #train_test_sets consists of list of list with [train_set, test_set] == sets
    model = ARIMA(sets[0], order=(2,1,1))
    model_fit = model.fit()
    model_fc = model_fit.forecast(steps=look_ahead)

    for i, _ in enumerate(model_fc):
        col = str(i+1)
        row = model_fc.index[i]
        val = model_fc.iloc[i]

        res_days_ahead_df.at[row, col] = val


#%%
print(len(df.data))
#%%
fig, ax = plt.subplots(figsize=(14,8), dpi=300)
ax = plt.plot(res_days_ahead_df)
ax = plt.plot(df.data['count'])

 
#%%
print(model_fc)
print()
print(model_fc.iloc[0])
print(model_fc.index[0])

# %%
def run_ARIMA(df, split_prct: float=0.8, )