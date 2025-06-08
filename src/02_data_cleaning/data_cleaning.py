# Conforming dates, etc.

import pandas as pd
import time #TODO: delete, for speed measure only
from numpy import nan
from datetime import datetime

pd.set_option("display.max_columns", None)

df = pd.read_csv("data/01_raw_data/testdaten.tsv", sep="\t")



print("done reading data")

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

def merge_to_new_col(df, columns_to_merge, new_name):
    # Merge multiple non-overlapping columns into one with new_name as the new name
    #WARNING: make sure, there are no rows with overlapping values (like 2 dates)
    df[new_name] = df[columns_to_merge].bfill(axis=1).iloc[:, 0]
    df = df.drop(columns_to_merge, axis=1, inplace = True)

def merge_cols(df, cols):
    "cols=List; Merge two non-overlapping cols with values/nan into the first one in cols, REMOVES second column!"
    print(df[cols[0]])
    print(df[cols[1]])
    df[cols[0]].fillna(df[cols[1]], inplace=True)
    df.drop(cols[1], axis=1, inplace=True)

# Date columns: Unify and merge all date columns
# DONE Merge all date columns
date_cols = [col for col in df.columns if col.startswith("T_")] # or use map: date_cols = date_format_map.keys()
for col, kwargs in date_format_map.items():
    df[col] = pd.to_datetime(df[col], **kwargs).dt.date

merge_to_new_col(df, date_cols, "date")


# Transfusion status: Rename and merge
transfusion_status_columns = [col for col in df.columns if col.startswith("ToD")]
transfusion_status_map = {
    "ToD" : {
        "Transfundiert" : "transfused",
        "Entsorgt" : "discarded"
    },
    "ToD_N" : {
        "VER" : "transfused", #'Verabreicht'
        "AUS" : "discarded" # QUESTION: Does 'AUS' (='Ausgegeben') mean used or discarded or neither?
    },
    "ToD_O" : {
        "ABG" : "expired"
    }
}

for col, val in transfusion_status_map.items():
    df[col].replace(val, inplace=True) # NOTE'replace' doesnt change non-matching vals, 'map' changes them to NaN

merge_to_new_col(df, transfusion_status_map.keys(), "use")
# DONE Rename to "DoT" or "Transfusion_date"


# Reformat Rh-Columns and split/merge into 2 columns: Rh, AB0
# DONE Merge/Split Rh/AB0 Columns
# TODO Reformat all Rh-Columns
print(f"\n\n\noriginal:\n{df['EC_BG_RH'].unique()}\n\n")
#Detailed Rh phenotype notation (ccddee) Format to AB0/Rh 
def extract_BG_from_detailed_notation(original, conversion_type):
    """Use in df.apply() to extract blood group or rhesus from input string,
    depending if conversion type is bg or rh"""
    if pd.isna(original):
        return nan

    #Extract Bloodgroup and rhesus factor
    if conversion_type == "bg":
        bg = original[0:2].strip()
        return bg
    elif conversion_type == "rh":
        rh = original[2]
        if rh == "-":
            return "Rh positiv"
        elif rh == "+":
            return "Rh negativ"
        else:
            return None

#TODO: delete time measurement
start = time.time() 
df["ec_bg_col"] = df["EC_BG_RH"].apply(extract_BG_from_detailed_notation, args=("bg", ))
end1 = time.time()
df["ec_rh_col"] = df["EC_BG_RH"].apply(extract_BG_from_detailed_notation, args=("rh", ))
end2 = time.time() #TODO: delete time measurement
print(f"BG took {end1-start}s,\nRh took {end2-end1}s\ntotal of {end2-start}s.") 


#Merge the new columns bg_col, rh_col into EC_BG and EC_RH respectively; delete original col "EC_BG_RH"
merge_cols(df=df, cols=["EC_BG", "ec_bg_col"])
merge_cols(df=df, cols=["EC_RH", "ec_rh_col"])
df.drop("EC_BG_RH", inplace=True, axis=1)

# Save df to new file:
df.to_csv(path_or_buf="./pipeline/01_data_prep/output.csv", sep=",")


# Check (print) unique columns:
print(df.columns)
for col in df.columns:
    if col not in ["date", "EC_ID_I_hash", "EC_ID_O_hash"]:
        print(f"{col}:\n{df[col].unique()}\n")

print(df["PAT_BG_RH"])
print(df["PAT_BG"])
print(df["PAT_RH"])
