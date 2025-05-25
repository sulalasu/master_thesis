# Conforming dates, etc.

import pandas as pd
from datetime import datetime

df = pd.read_csv("data/01_raw_data/testdaten.tsv", sep="\t")


print("done reading data")
print(df.head())

# print(df.head())
# print(df.info())


# print(df.describe())
# print(df.columns)

# print(f"ype: {type(df.iloc[0, 8])}, {df.iloc[:, 8]}")

# date_cols = df.filter(regex="^T_")
# print(date_cols)
# print(date_cols.columns)
# Convert all dates (strings or date types?) to iso (date only, no time)
# TODO

# iso_tf = date_cols["T_ISO_T"]
# us_tf = date_cols["T_US"]
# us_tft = date_cols["T_US_T"]

# print(iso_tf)

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
    print(df[col])
#string ISO with Time to: ISO without time
#iso_tf = pd.to_datetime(iso_tf, yearfirst=True)
#string US without time to: ISO without time

#us_tf = pd.to_datetime(us_tf, format="%m-%d-%y")
#print("us_tf: \n\n")
#print(us_tf)
#print(f"type: {type(us_tf.loc(30))}")

#convert XL (Excel) to ISO:
#t_xl = pd.to_datetime(df["T_XL"],  unit="D", origin="1899-12-30").dt.date #TODO: get citation/check for correct origin date! got it from stackoverflow 38454403
#print(t_xl)

# t_xl = t_xl.dt.date
# print("t_xl without time:\n\n")
# print(t_xl)
#string US with time to: ISO without time

#pd.to_datetime()
# print(f"iso type: {type(iso_tf.iloc[8])}")

# print(f"us type: {type(us_tf.iloc[30])}")
#def convert_ISOTstr_to_ISOdate(col):
    # Converts a pandas column/series of a string of ISO DATE + TIME string to a ISO datetime object
    # pass

# Merge all date columns
# TODO

