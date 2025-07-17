# Read Data
import pandas as pd

def load_data(path, sep="\t"):
    pd.set_option("display.max_columns", None)
    df = pd.read_csv(path, sep=sep)
    print("Read raw data f")

    return df


def show_info(df):
    print(df.head())
    print(df.describe())
    print(df.info())


#TODO: functions to load external data from url
#TODO: functions to check if data already present/loaded
#TODO: function to add/clean external data to main data (maybe better in clean.py)