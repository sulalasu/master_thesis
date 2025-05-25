# Conforming dates, etc.

import pandas as pd
from datetime import datetime

pd.set_option("display.max_columns", None)

df = pd.read_csv("data/01_raw_data/testdaten.tsv", sep="\t")



print("done reading data")

# TODO done: change all date formats to iso without time
# Mapping of column names with their respective format
date_format_map = {
    "T_XL" : {"unit" : "D", "origin" : "1899-12-30"},
    "T_ISO_T" : {"yearfirst" : True}, 
    "T_DE" : {"dayfirst" : True}, 
    "T_US_T" : {"format" : "%m/%d/%y %H:%M"}, #%H for 24h clock, %I for 12h clock
    "T_DE_S" : {"dayfirst" : True}, 
    "T_US" : {"format" : "%m-%d-%y"}, 
    "T_DE_T" : {"dayfirst" : True},
    "T_ISO" : {"yearfirst" : True}
}

for col, kwargs in date_format_map.items():
    df[col] = pd.to_datetime(df[col], **kwargs).dt.date


# TODO Merge all date columns
# TODO Rename to "DoT" or "Ttansfusion_date"
# Merge all date columns
#print(df["ToD_O"])

# TODO Reformat all Rh-Columns
# TODO Merge/Split Rh/AB0 Columns
# Reformat Rh-Columns and split/merge into 2 columns: Rh, AB0

#print(df.head())
print(df.columns)
print(df["EC_BG_RH"])

 
