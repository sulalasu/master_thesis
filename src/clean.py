# Conforming dates, etc.

import pandas as pd
import datetime as dt
from time import time
from numpy import nan

from src import config


#-----------------------------------------------------------------------------
# Main wrapper function for cleaning:
#-----------------------------------------------------------------------------
def clean_data(df):

    df = clean_dates(df)

    df = clean_transfusion_status(df)

    # EC_BG_RH --> EC_BG + EC_RH
    # Reformat Rh-Columns and split/merge into 2 columns: Rh, AB0
    # DONE Merge/Split Rh/AB0 Columns
    # TODO Reformat all Rh-Columns
    # TODO: wrap in function
    # df = reformat_EC_bg(df, newnames) #first apply, then merge
    print(f"\n\n\noriginal:\n{df['EC_BG_RH'].unique()}\n\n")

    df = split_BG_RH(df, origin="EC_BG_RH", temp_cols=["ec_bg_col", "ec_rh_col"], target_cols=["EC_BG", "EC_RH"])
    df = split_BG_RH(df, origin="PAT_BG_RH", temp_cols=["pat_bg_col", "pat_rh_col"], target_cols=["PAT_BG", "PAT_RH"])

    return df

#-----------------------------------------------------------------------------
# (wrapper) functions for main steps
#-----------------------------------------------------------------------------


def clean_dates(df):
    # Date columns: Unify and merge all date columns
    # DONE Merge all date columns
    # DONE wrap in function

    date_cols = [col for col in df.columns if col.startswith("T_")] # or use map: date_cols = date_format_map.keys()
    for col, kwargs in config.date_format_map.items():
        df[col] = pd.to_datetime(df[col], **kwargs).dt.date

    df = merge_to_new_col(df, date_cols, "date")

    return df


def clean_transfusion_status(df):
    # Transfusion status: Rename and merge
    # DONE wrap in function

    for col, val in config.transfusion_status_map.items():
        df[col].replace(val, inplace=True) # NOTE'replace' doesnt change non-matching vals, 'map' changes them to NaN
    # transfusion_status_columns = [col for col in df.columns if col.startswith("ToD")]

    # DONE Rename to "DoT" or "Transfusion_date"
    df = merge_to_new_col(df, config.transfusion_status_map.keys(), "use")

    return df


def split_BG_RH(df, origin, temp_cols, target_cols):
    """splits column of PAT or EC that contains bg+rh in AB+ccddee-notation to two AB0/Rhesus factor columns,
    then drops the origin column and the two created columns after merging the extraced columns"""

    #get bloodgroup
    #TODO: delete time measurements
    start = time() 
    df[temp_cols[0]] = df[origin].apply(extract_BG_from_detailed_notation, args=("bg", ))
    end1 = time()
    #get rhesus factor
    df[temp_cols[1]] = df[origin].apply(extract_BG_from_detailed_notation, args=("rh", ))
    end2 = time() 
    print(f"BG took {end1-start}s,\nRh took {end2-end1}s\ntotal of {end2-start}s.") 

    #Merge the new columns bg_col, rh_col into EC_BG and EC_RH respectively; delete original col "EC_BG_RH"
    df = merge_cols(df=df, cols=[target_cols[0], temp_cols[0]])
    df = merge_cols(df=df, cols=[target_cols[1], temp_cols[1]])
    df = df.drop(origin, axis=1) #inplace=True


    print(f"columns of cleaned df:\n{df.columns}")
    print(f"head of cleaned df:\n{df.tail()}")
    
    #PAT_BG_RH --> PAT_BG + PAT_RH

    return df






#-----------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------


def merge_to_new_col(df, columns_to_merge, new_name):
    # Merge multiple non-overlapping columns into one with new_name as the new name
    #WARNING: make sure, there are no rows with overlapping values (like 2 dates)
    df[new_name] = df[columns_to_merge].bfill(axis=1).iloc[:, 0]
    df = df.drop(columns_to_merge, axis=1) #, inplace = True)

    return df

def merge_cols(df, cols):
    "cols=List; Merge two non-overlapping cols with values/nan into the first one in cols, REMOVES second column!"
    print(df[cols[0]])
    print(df[cols[1]])
    df[cols[0]].fillna(df[cols[1]], inplace=True)
    df = df.drop(cols[1], axis=1)#, inplace=True)

    return df

#Detailed Rh phenotype notation (ccddee) Format to AB0/Rh 
def extract_BG_from_detailed_notation(original, conversion_type):
    """Use in df.apply() to extract blood group or rhesus from input string,
    depending if conversion type is bg or rh"""
    if pd.isna(original):
        return nan

    #Extract Bloodgroup or rhesus factor
    if original[:3] == "NBN":
        return original[:3]
    
    if conversion_type == "bg":
        bg = original[0:2].strip()
        return bg
    elif conversion_type == "rh":
        try:
            rh = original[2]
        except IndexError:
            rh = "NBN"

        if rh == "-":
            return "Rh positiv"
        elif rh == "+":
            return "Rh negativ"
        elif rh == "NBN":
            return rh
        #TODO: do i need to differntiate between NBN and nan/None?
        else:
            return None



def check_unique_values(df, hidden_cols=["date", "EC_ID_I_hash", "EC_ID_O_hash"]):
    """Hidden cols for cols that have lots of unique values, like date, hash"""
    #TODO: wrap in function, keep fct call, but comment it out.
    # Check (print) unique columns:
    print(df.columns)
    for col in df.columns:
        if col not in hidden_cols:
            print(f"{col}:\n{df[col].unique()}\n")

    print(df["PAT_BG_RH"])
    print(df["PAT_BG"])
    print(df["PAT_RH"])
