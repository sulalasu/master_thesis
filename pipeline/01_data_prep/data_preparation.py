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
    "T_DE_S" : {"format" : "%d.%m.%y"}, #y for short year
    "T_US" : {"format" : "%m-%d-%y"}, 
    "T_DE_T" : {"format" : "%d.%m.%y %H:%M"},
    "T_ISO" : {"yearfirst" : True}
}

for col, kwargs in date_format_map.items():
    df[col] = pd.to_datetime(df[col], **kwargs).dt.date



# TODO Reformat all Rh-Columns
# TODO Merge/Split Rh/AB0 Columns
# Reformat Rh-Columns and split/merge into 2 columns: Rh, AB0

def merge_columns(df, columns_to_merge, colname):
    # Merge multiple non-overlapping columns into one with colname as the new name
    #TODO: make sure, there are no rows with overlapping values (like 2 dates)
    df[colname] = df[columns_to_merge].bfill(axis=1).iloc[:, 0]
    df = df.drop(columns_to_merge, axis=1, inplace = True)


# TODO Merge all date columns
# TODO Rename to "DoT" or "Transfusion_date"
# Merge all date columns
#print(df["ToD_O"])
date_cols = [col for col in df.columns if col.startswith("T_")]
# or use map:
# date_cols = date_format_map.keys()
merge_columns(df, date_cols, "date")



#print(df.head())
print(df.columns)
print(df)
#print(df[["ToD", "ToD_N", "ToD_O"]])

 
