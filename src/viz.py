import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import MSTL #multiple seasonal decompose
from pandas.plotting import autocorrelation_plot

from src.config import SAVE_FIGS
from src.utils import save_plots

def line_plot(data):
    # TODO
    pass


def seasonal_plot(data, periods: list[str]):
    """
    periods = periods to make seasonal plots for (e.g. week, months)
    can be "W", "M", "Y" 
    """
    fig, ax = plt.subplots(ncols=1, nrows=len(periods))

    data['day_of_week'] = data['date'].dt.dayofweek
    data['week'] = data['date'].dt.isocalendar().week
    data['week_str'] = data['week'].astype(str)
    data['month'] = data['date'].dt.isocalendar().month

    for period in periods:
        pass



    
    #TODO
    fig, ax = plt.subplots()
    ax.plot(label = ["year"])
    pass


def heatmap(data):
    #TODO
    # plt.figure(figsize=(7, 5))
    # sns.lineplot(x=data.index.month, y=data['count'], ci=None)
    # plt.xlabel('Month')
    # plt.ylabel('Number of XXX') ä#TODO
    # plt.title('Seasonal Plot')
    # plt.xticks(
    #     range(1, 13), 
    #     labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # )
    # plt.grid(True)
    # plt.show()
    # return plt
    pass




#----------------------------------------------------------------------------------------------
# DECOMPOSITION
#----------------------------------------------------------------------------------------------
#TODO: how to do it? do i want ot have a function, that iterates+decomposes every value?
# Or only the main value (total count), or main values (total count, ec_bg/rh_count, pat_bg/rh_count, etc?)


def decompose_all(df, model, period=7):
    #maybe function to decompose multiple/all columns? or just one fct, 
    # where it iterates over models (and i can pass df.columns minus date)?

    result = seasonal_decompose(df, model='additive', period=period)
    print(result.trend)
    print(result.seasonal)
    print(result.resid)
    print(result.observed)

    # Visualize:
    #TODO: tune plot
    result.plot()
    plt.show()

    return None

def decompose_one(df, model, column, period=7):
    #maybe function to target only one column to decompose?
    result = seasonal_decompose(df[column], model='additive', period=period)
    print(result.trend)
    print(result.seasonal)
    print(result.resid)
    print(result.observed)

    # Visualize:
    #TODO: tune plot
    result.plot()
    plt.show()


def multiple_decompose(df, col: str, periods: list):
    # col = col to decompose, i.e. y, for example "count"
    mstl = MSTL(df[col], periods=periods)
    res = mstl.fit()

    res.plot()
    plt.show()

    return mstl

#----------------------------------------------------------------------------------------------
# AUTOCORRELATION
#----------------------------------------------------------------------------------------------

def autocorr(column: pd.Series):
    autocorrelation_plot(column)

    plt.show()


#----------------------------------------------------------------------------------------------
# GENERAL VIZ
#----------------------------------------------------------------------------------------------

def plot_patient_wards(df: pd.DataFrame, n: int, save_figs=SAVE_FIGS, filename="pat_wards_counts", foldername="PAT_WARDS", location="./plots/02_cleaned"):
    """Plots each patients wards time series of daily transfusion count.
    Supposes that df is a wide df, where each ward has its own column with counts.
    Only prints columns that have more than n transfusions.

    Args:
        df (pd.DataFrame): wide dataframe, with separate column for each ward. Column names
        must start with 'PAT_WARD'.
        n (int): number of transfusions must be above n for the column to be plotted.
    """
    for ward in df["PAT_WARD"].unique():

        df_pat_ward_daily = df[df['PAT_WARD'] == ward]
        df_pat_ward_daily.info()
        df_filtered = df_pat_ward_daily.groupby(df_pat_ward_daily.index.date).count()
        if len(df_filtered["PAT_WARD"]) < 500:
            continue

        fig, ax = plt.subplots()
        ax.plot(df_filtered['PAT_WARD'])
        ax.set_xlabel('date')
        ax.set_title(ward)
        fig.show()

        #Old verions: delete
        # df_filtered['PAT_WARD'].plot(x='date')
        # plt.title(ward)
        # plt.show()

        if save_figs:
            save_plots(filename=filename, foldername=foldername)

