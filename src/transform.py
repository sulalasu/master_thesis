# Functions for splitting datasets into test/training/validation etc.
import pandas as pd
from pathlib import Path



#-----------------------------------------------------------------------------
# Main wrapper function for transforming/processing:
#-----------------------------------------------------------------------------
def transform_data(df,
                   existing_file_path: str="./data/03_transformed/output_transformed.csv", 
                   new_file_path: str="./data/03_transformed/output_transformed.csv"):
    #load cleaned file if exists (so i can skip cleaning step):
    my_file = Path(existing_file_path)

    if my_file.is_file():
        df = pd.read_csv(existing_file_path, sep=",")
        #set again to datetime, because first time didnt work? (i think because of merging/missing values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()
        print(f"Reading existing transformed file at {existing_file_path}")
    else:

        #IMPUTATION
        # IMputation (removing/replacing missing values/nans) has
        # to happen before daily aggregation (or i have to decide each time,
        # how to proceed with missing values.)


        #remove unnecessary cols:
        cols_to_remove = ["EC_ID_O_hash", "EC_ID_I_hash", "PAT_WARD"] #"Unnamed: 0"
        df = df.drop(columns=cols_to_remove, axis=1)
        #colnames which i want to sum (all except 'date')
        df = df.reset_index()
        cols_to_sum = list(df.columns)
        cols_to_sum.remove('date')
        #sum all above cols and split into category columns
        df = aggregate_categorical_cols(df, cols_to_sum)


        #make wide
        #see aggregate_categorical_cols!
        #df = make_wide(df)

        #Total count of EC for that day (expired, transfused etc.)
        #df = add_daily_total(df)


        #df = add_working_days(df)


        #df = add_weather_data(df)


    return df


#-----------------------------------------------------------------------------
# (wrapper) functions for main steps
#-----------------------------------------------------------------------------

def aggregate_categorical_cols(df, cols_to_sum: dict):
    #Split columns with n categorical values into n columns.
    # Aggregate daily sum of occurences to each column.
    # cols_to_sum list of all col names with categorical values which i want to sum.  
    
    results = []

    for col in cols_to_sum:
        res = pd.crosstab(df['date'], df[col])
        res = res.add_prefix(col + "_")
        res.columns = res.columns.str.replace(" ", "_") #replace spaces

        results.append(res)

    #wide_df = results[0].join([res for res in results[1:]])
    wide_df = pd.concat(results, axis=1)


    #Add daily total amounts:
    #test_df = df.set_index('date')['use'].resample('D').count()
    total_df = df.groupby('date').size()#.reset_index(name='count')
    total_df.name = "count"
    #Add to wide_df
    wide_df = wide_df.join(total_df)#.set_index('date'))
    
    return wide_df


def add_working_days(df):
    import holidays

    holidays_aut = holidays.country_holidays(country="AT", subdiv="W", years=range(2020,2025), language="de")
    # assignment
    df["is_workday"] = df["date"].apply(holidays_aut.is_working_day)

    return df


def add_weather_data(df, weather_data_path="data/00_external_data/Wetter_Tagesdaten_InnereStadt_2005_2024.csv"):
    #add daily precipitation (mm) and the daily average temperature (°C) -- no min/max values.
    # Data from https://data.hub.geosphere.at/dataset/klima-v2-1d 
    # respectively: https://dataset.api.hub.geosphere.at/app/frontend/station/historical/klima-v2-1d
    # downloaded 03.07.2025. Data station Innere Stadt
    # 24h sum rainfall, temp max, temp min, temp I (7:00), II (14:00), III (19:00 MEZ), temp avg.

    from src import load
    weather_df = load.load_data(path=weather_data_path)
    weather_df = weather_df['time'].to_datetime().dt.day

    df = pd.merge(df, weather_df, left_on='date', right_on='time')

    return df

#Add Covid/influenca data:


#Make wide df, where every col value becomes its own col with daily freq count; prefix of original col name
def make_wide(df):
    #see aggregate_categorical_cols!

    #if date is not index
    daily_total = df.groupby("date").size()
    #if is index:
    #df = df.set_index("date")
    daily_total = df.groupby(df.index.date).size()
    daily_total.name = "total"


    res = []
    for col in df.columns:
        if col == "date":
            continue
        pivotted = pd.crosstab([df["date"]], columns=[df[col]])
        pivotted = pivotted.add_prefix(col + "_")
        res.append(pivotted)
    final = pd.concat(res, axis=1)
    final = final.join(daily_total)

    return final


#-----------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------
