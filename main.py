#%%
import pandas as pd
import numpy as np
from numpy import nan
from time import time
from matplotlib import pyplot as plt
import seaborn as sns


from src import load
from src import clean
from src import process
from src import viz
from src import config
from src import model

print(load.__file__)
print(clean.__file__)

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
# remove duplicates/NAs, maybe imputation if necessary
# make STATIONARY!
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





#%%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# TEST DATA
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Get data & clean
from sklearn.datasets import fetch_openml
df = fetch_openml("seoul_bike_sharing_demand", as_frame=True)
df = pd.DataFrame(df.frame)

df.rename(mapper=config.seoul_name_map, axis=1, inplace=True)
print("seoul head\n\n")
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df["hour"] = pd.to_datetime(df["hour"], format="%H").dt.time

load.show_info(df=df)

# Skip -- CLEANING -- 
# Skip -- PROCESSING -- 



# -- VISUALIZE --

#%%
# Daily average
# TODO: wrap in function (or add to Model?, because its already cleaned+processed here, so next step
# besides viz would be add to model anyway? BUT exploratory viz is done on raw data, so no specific model...
# TODO: make prettier: add title, colorchart (so i can later exchange colors), etc.
daily_average = df["count"].mean()

sns.lineplot(x="date", y="count", data=df)
plt.axhline(y=daily_average, color="black", label="average")

plt.show()

#%%
# Run Arima Model

arima = model.ModelSarima(df)


arima.make_stationary()
arima.split()
arima.create_model()
arima.test_model()
arima.evaluate_model()
