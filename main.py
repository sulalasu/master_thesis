#%%
import src
import pandas as pd


# Conforming dates, etc.
#%%
# Read Data
pd.set_option("display.max_columns", None)
df = pd.read_csv("data/01_raw/testdaten.tsv", sep="\t")
print("done reading data")

# %%



# DONE: change all date formats to iso without time
# Mapping of column names with their respective format
date_format_map = {
    "T_XL" : {"unit" : "D", "origin" : "1899-12-30"},
    "T_ISO_T" : {"yearfirst" : True}, 
    "T_DE" : {"dayfirst" : True}, 
    "T_US_T" : {"format" : "%m/%d/%y %H:%M"}, #%H for 24h clock, %I for 12h clock
    "T_DE_S" : {"format" : "%d.%m.%y"}, #y for short year
    "T_US" : {"format" : "%m-%d-%y"}, 
    "T_DE_T" : {"format" : "%d.%m.%y %H:%M"},
    "T_ISO" : {"yearfirst" : True}
}
