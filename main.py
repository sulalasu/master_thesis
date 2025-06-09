#%%
import pandas as pd
import numpy as np
from numpy import nan
from time import time

from src import input
from src import cleaning
from src import processing
from src import viz
from src import config


#%%--------------------------------------------------------------------------------
# INPUT
#----------------------------------------------------------------------------------

#%%
# Read Data
pd.set_option("display.max_columns", None)
df = pd.read_csv("data/01_raw/testdaten.tsv", sep="\t")
print("done reading data")




#%%--------------------------------------------------------------------------------
# CLEANING 
#----------------------------------------------------------------------------------


# %%
# Date columns: Unify and merge all date columns
# DONE Merge all date columns
# TODO: wrap in function

date_cols = [col for col in df.columns if col.startswith("T_")] # or use map: date_cols = date_format_map.keys()
for col, kwargs in config.date_format_map.items():
    df[col] = pd.to_datetime(df[col], **kwargs).dt.date

cleaning.merge_to_new_col(df, date_cols, "date")


#%%
# Transfusion status: Rename and merge
# TODO: wrap in function
# df = clean_transfusion_status()
transfusion_status_columns = [col for col in df.columns if col.startswith("ToD")]

for col, val in config.transfusion_status_map.items():
    df[col].replace(val, inplace=True) # NOTE'replace' doesnt change non-matching vals, 'map' changes them to NaN

cleaning.merge_to_new_col(df, config.transfusion_status_map.keys(), "use")
# DONE Rename to "DoT" or "Transfusion_date"


#%%
# Reformat Rh-Columns and split/merge into 2 columns: Rh, AB0
# DONE Merge/Split Rh/AB0 Columns
# TODO Reformat all Rh-Columns
# TODO: wrap in function
# df = reformat_EC_bg(df, newnames) #first apply, then merge

print(f"\n\n\noriginal:\n{df['EC_BG_RH'].unique()}\n\n")

#TODO: delete time measurements
start = time() 
df["ec_bg_col"] = df["EC_BG_RH"].apply(cleaning.extract_BG_from_detailed_notation, args=("bg", ))
end1 = time()
df["ec_rh_col"] = df["EC_BG_RH"].apply(cleaning.extract_BG_from_detailed_notation, args=("rh", ))
end2 = time() 
print(f"BG took {end1-start}s,\nRh took {end2-end1}s\ntotal of {end2-start}s.") 

#Merge the new columns bg_col, rh_col into EC_BG and EC_RH respectively; delete original col "EC_BG_RH"
cleaning.merge_cols(df=df, cols=["EC_BG", "ec_bg_col"])
cleaning.merge_cols(df=df, cols=["EC_RH", "ec_rh_col"])
df.drop("EC_BG_RH", inplace=True, axis=1)



# Save df to new file:
# TODO: leave as is, dont wrap in function
df.to_csv(path_or_buf="./data/02_intermediate/intermediate_output.csv", sep=",")


#TODO: wrap in function, keep fct call, but comment it out.
# Check (print) unique columns:
print(df.columns)
for col in df.columns:
    if col not in ["date", "EC_ID_I_hash", "EC_ID_O_hash"]:
        print(f"{col}:\n{df[col].unique()}\n")

print(df["PAT_BG_RH"])
print(df["PAT_BG"])
print(df["PAT_RH"])





#%%--------------------------------------------------------------------------------
# DATA VIZ (EXPLORATION)
#----------------------------------------------------------------------------------
# %%




#%%--------------------------------------------------------------------------------
# PROCESSING
#----------------------------------------------------------------------------------
# splitting in test/training etc.




#%%--------------------------------------------------------------------------------
# MODEL BUILDING
#----------------------------------------------------------------------------------

#TODO: look into OOP + config.yml
sarima = Model1()


#%%--------------------------------------------------------------------------------
# DATA VIZ (FINISHED MODEL) 
#----------------------------------------------------------------------------------
# Plot prediction vs actual


#%%--------------------------------------------------------------------------------
# Evaluation
#----------------------------------------------------------------------------------

# print evaluations/tests like mae, mape, etc.