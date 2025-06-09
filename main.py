#%%
import pandas as pd
import numpy as np
from numpy import nan
from time import time

from src import load
from src import clean
from src import process
from src import viz
from src import config


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
# DATA VIZ (EXPLORATION)
#----------------------------------------------------------------------------------






#%%--------------------------------------------------------------------------------
# PROCESSING
#----------------------------------------------------------------------------------
# remove duplicates/NAs, maybe imputation if necessary, splitting in test/training etc.




#%%--------------------------------------------------------------------------------
# MODEL BUILDING
#----------------------------------------------------------------------------------

#TODO: look into OOP + config.yml
#sarima = Model1(config[0])

# if functional:
# sarima = statsmodels.sarima(xx, xx, xx) # oder so


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


#%%--------------------------------------------------------------------------------
# EVALUATION
#----------------------------------------------------------------------------------



# print evaluations/tests like mae, mape, etc.