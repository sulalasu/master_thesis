import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import MSTL #multiple seasonal decompose
from pandas.plotting import autocorrelation_plot


def line_plot(df):
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