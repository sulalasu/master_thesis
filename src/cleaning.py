# Conforming dates, etc.

import pandas as pd
from numpy import nan


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

