#%% MARK: libs etc

import pandas as pd
import numpy as np
from numpy import nan
from time import time
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns


from src import clean
from src import config
from src import data_model
from src import load
from src import model
from src import transform
from src import viz

import uuid


#For developing purposes:
import importlib
print(load.__file__)
print(clean.__file__)

IMAGE_PATH = "plots/2025_10_10-Plots_for_Meeting/"
RUN_ALL = False


#%%--------------------------------------------------------------------------------
# MARK: INPUT
#----------------------------------------------------------------------------------

# Read Data
df_raw = load.load_data(path="data/01_raw/blood-data_complete_2025-07-16.tsv")
# df_raw = load.load_data(path="data/01_raw/testdaten.tsv")

#%%
#TODO: remove
if RUN_ALL == True:
    load.show_info(df=df_raw)
    hidden_cols=["date", "EC_ID_I_hash", "EC_ID_O_hash", "T_ISO", "T_DE_T", "T_US", "T_DE_S", "T_US_T", "T_DE", "T_ISO_T", "T_XL"]
    for col in df_raw.columns:
        if col not in hidden_cols:
            tmp = df_raw[col].astype(str).unique()
            tmp = np.array(sorted(tmp))
            print(f"{col}:\n{tmp}\n")
    print(df_raw.columns)
    #TODO: move back to clean_data
    # df_raw = clean.clean_dates(df_raw) #TODO: remove here, enable again in clean_data()



#%%--------------------------------------------------------------------------------
# MARK: CLEANING 
#----------------------------------------------------------------------------------
#Runs only if no file exists at. If not existing, saves df to new file
#unify dates, columns etc. rename stuff
IMAGES_PATH_EXPLORATION = IMAGE_PATH + "/00-Data-Exploration/"
importlib.reload(clean)
importlib.reload(config)
df_clean = clean.clean_data(df_raw)
# df_clean.sort_index(inplace=True)
# #TODO: remove 5 lines:
# start_date = pd.to_datetime("2018-01-01")
# start_date = pd.to_datetime("2024-12-31")
# mask = (df_clean.index >= "2018-01-01") & (df_clean.index <= "2024-12-31")
#df_clean = df_clean.loc[mask]
#df_clean = df_clean['2018-01-01':'2024-12-31'] #only works on monotonic (==daily aggregated, no duplicate days) df

#%%
#TODO: remove
if RUN_ALL == True:
    #TODO: Check what unique vals are present in df
    clean.check_unique_values(df_clean.drop(["EC_ID_I_hash", "EC_ID_O_hash", "PAT_WARD"], axis=1))


#%%
# Plot frequency counts for unique values in every column
#TODO: move into viz.py

#TODO: remove
if RUN_ALL == True:
    for col_name, col in df_clean.items():
        if col_name in ["EC_ID_O_hash", "EC_ID_I_hash"]:
            continue
        print(col.value_counts())
        col.value_counts()[:40].plot(kind="bar", title=col_name,)
        plt.savefig(fname=IMAGES_PATH_EXPLORATION+f"01-barcharts-value_counts-{col_name}")
        plt.show()


    importlib.reload(viz)
    ##%%
    # Plot each patient wards transfusion counts (for wards with >500 transfusions)
    viz.plot_patient_wards(df_clean, n=500, save_figs=False, location=IMAGES_PATH_EXPLORATION, foldername="")







#%%--------------------------------------------------------------------------------
# MARK: TRANSFORMING
# /PROCESSING
#----------------------------------------------------------------------------------
# make STATIONARY! (if all models need that, otherwise make it a member function)
# splitting in test/training etc. here or as extra step/model step?



# Proces....
#add external data (holidays weather (temp, precipitation), covid/influenca cases)
#NOTE: covid/grippe muss evnetuell imputiert werden da nur wöchentlich
#NOTE: kann gut zeigen, dass wien gleichen verlauf hat wie bundesländer, daher kann ich Ö-weite Daten
# nehmen, falls es keine wien-spezifischen Daten gibt.
importlib.reload(transform)

# make daily aggregations for categorical variables
df_processed = transform.transform_data(df_clean)


#%% #Plot seasonalities daily & weekly of processed df
#TODO: remove
if RUN_ALL == True:
    importlib.reload(viz)

    BG_RH_cols = ['EC_BG_RH_0_NB']#,
        #    'EC_BG_RH_0_Rh_negative', 'EC_BG_RH_0_Rh_positive', 'EC_BG_RH_A_NB',
        #    'EC_BG_RH_A_Rh_negative', 'EC_BG_RH_A_Rh_positive', 'EC_BG_RH_AB_NB',
        #    'EC_BG_RH_AB_Rh_negative', 'EC_BG_RH_AB_Rh_positive', 'EC_BG_RH_B_NB',
        #    'EC_BG_RH_B_Rh_negative', 'EC_BG_RH_B_Rh_positive', 'PAT_BG_RH_0_NB',
        #    'PAT_BG_RH_0_Rh_negative', 'PAT_BG_RH_0_Rh_positive', 'PAT_BG_RH_A_NB',
        #    'PAT_BG_RH_A_Rh_negative', 'PAT_BG_RH_A_Rh_positive', 'PAT_BG_RH_AB_NB',
        #    'PAT_BG_RH_AB_Rh_negative', 'PAT_BG_RH_AB_Rh_positive',
        #    'PAT_BG_RH_B_NB', 'PAT_BG_RH_B_Rh_negative', 'PAT_BG_RH_B_Rh_positive',
        #    'PAT_BG_RH_NB_NB', 'PAT_BG_RH_NB_Rh_negative',
        #    'PAT_BG_RH_NB_Rh_positive', 'PAT_BG_RH_Not_applicable']
    for bg_rh in BG_RH_cols:
        # viz.seasonal_plot(df_processed, plot_type="weekly", col_name=bg_rh)
        viz.seasonal_plot(df_processed, plot_type="daily", col_name=bg_rh)



    ward_cols = ['ward_AN', 'ward_CH', 'ward_I1', 'ward_I3', 'ward_Other', 'ward_UC']
    for ward in ward_cols:
        viz.seasonal_plot(df_processed, plot_type="weekly", col_name=ward)
        viz.seasonal_plot(df_processed, plot_type="daily", col_name=ward)


    #TODO: save data to csv
    # # Plot daily/weekly cases influenza
    # fig, ax = plt.subplots(1)
    # ax.plot(df_processed["new_cases_daily"])
    # ax.plot(df_processed["new_cases_weekly"], color="red")
    # plt.show



#%%--------------------------------------------------------------------------------
# MARK: DATA VIZ 
# (EXPLORATION)
#----------------------------------------------------------------------------------
IMAGES_PATH_EXPLORATION = IMAGE_PATH + "/00-Data-Exploration/"
START_DATE_EXPLORATION = "2020-01-01"
PRE_COVID_START = "2018-01-01"
PRE_COVID_END = "2020-01-01"

#TODO: save vizualisations to csv
importlib.reload(data_model)

df = data_model.Data(data=df_processed)

#%%
#TODO: remove
if RUN_ALL == True:


    #df.print_head()
    df[START_DATE_EXPLORATION:].plot_seasonal(plot_type='daily', col_name='use_transfused', fig_location=IMAGES_PATH_EXPLORATION)
    df[START_DATE_EXPLORATION:].plot_seasonal(plot_type='weekly', col_name='use_transfused')


    ##%%
    #Boxplots
    df[START_DATE_EXPLORATION:].plot_boxplots(col_name='use_transfused')
    df[START_DATE_EXPLORATION:].plot_seasonal_subseries(col_name='use_transfused') #NOTE: i think it works, but not enough dummy data.
    #TODO: check if seasonal subseries plot works with multi-year data


    ##%%
    #Decompose
    df[PRE_COVID_START:PRE_COVID_END].decompose_one(col_name='use_transfused')


    #df.decompose_all("use_transfused")

    # mulitple decomposition (daily + weekly)
    df[PRE_COVID_START:PRE_COVID_END].multiple_decompose(col_name="use_transfused", periods=[7, 365])




    ##%%



    df[pd.to_datetime("2024-01-01"):pd.to_datetime("2024-12-31")].plot_daily_heatmap(col_name='use_transfused')


#%% Visualize counts for all plots (as of now only for those starting with 901AN) on top of each other, so that
# its visible, where naming of one ward starts/ends 
#TODO: remove
if RUN_ALL == True:
    wards = df_clean["PAT_WARD"].unique()
    fig, ax = plt.subplots(44,1, figsize=(6, 24))
    fig.set_dpi(300)
    fig.set_linewidth(5)
    sel_wards = []
    i = 0
    for ward in wards:
        if str(ward).startswith("901AN"):
            sel_wards.append(ward)
            ax[i].plot(df[f"PAT_WARD_{ward}"], label=ward, linewidth=0.15)
            # ax[i].set_title(str(ward))
            ax[i].legend(loc="upper right", prop={"size":6}, frameon=False, framealpha=0.5)
            ax[i].set_yticklabels([])
            i = i+1
            
    print(len(sel_wards))
    print(sel_wards)
    plt.savefig(fname="/".join([IMAGES_PATH_EXPLORATION + "show_inconsistency_wards"]))
    plt.show()

#%%
# Get unique Letter-combinations (i guess top-wards) from all wards:
#TODO: remove
if RUN_ALL == True:
    import re
    unique_wards = df_clean["PAT_WARD"].unique()

    pattern = re.compile(r'\d+([A-Za-z]+)\d+')  # number + letters + number

    ward_results = set()  # only unique strings

    for s in unique_wards:
        match = pattern.search(str(s))
        if match:
            ward_results.add(match.group(1))  # =letters

    print(unique_wards)
    print(len(unique_wards))
    print(ward_results)
    print(len(ward_results))

#%%--------------------------------------------------------------------------------
# MARK: STATIONARITY 
# Check for Stat./Make stationarys
#----------------------------------------------------------------------------------
#TODOLIST:
# 1. OG Data
#    Visual & statistical check: const mean, variance, no seasonal component 
#    --> then its stationary
# 1.1 Visual assessment
#    - Time series
#    - ACF, PACF
# 1.2 Statistical test
#    - Unit root test
# 
#  
#TODO: remove
if RUN_ALL == True:
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf

    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import kpss

    ##%%
    #Time series plots (acf, pacf etc)
    df.plot_autocorrelation(col_name='count')
    df.plot_partial_autocorrelation(col_name='count')

    num_differencing = 2
    i = 1
    df_diff = df_processed.copy()

    df_diff["use_transfused"].plot(lw=0.05)

    plot_acf(df_diff["use_transfused"], title="No differentiation")
    plot_pacf(df_diff["use_transfused"], title="No differentiation")

    #ADF -- Augmented dickey-fuller test
    adf_result_diff = adfuller(df_diff["use_transfused"][1:], autolag="AIC") #default
    adf_result_diff2 = adfuller(df_diff["use_transfused"][1:], autolag="BIC")
    adf_result = adfuller(df["use_transfused"], autolag="AIC") #default
    adf_result2 = adfuller(df["use_transfused"], autolag="BIC")
    adfuller(df["use_transfused"])
    plt.plot(df["use_transfused"])
    plt.plot(df_diff["use_transfused"])

    #i think i can reject the H0 in both cases (df and df_diff), for df test staticstics is -11.49, far below 
    #critical values of 1%, 5%, 10%, same  for df_diff with -21.07. therefore i reject H0 (data is 
    # non stationary, has a unit root). H1 is true, data is stationary has no unit root. so no differencing 
    # needed actually?

    #Different params for autolag:
    # no real differences in result, so doestn matter. (in gerneral AIC better for large datasets)


    #KPSS
    kpss_result_diff = kpss(df_diff["use_transfused"][1:])
    kpss_result = kpss(df["use_transfused"])
    kpss_result = kpss(df["use_transfused"])

    #kpss is one-sided test, if test statistics is greater than critical value, then H0 is rejected. 
    # H0 for KPSS is that ts is stationary, H1 ts is NOT stationary.
    # for kopss_result_diff, p-val = 0.1, test stat. = 0.0104, 0.0104 is not greater than 0.347 (10%), so H0 cant
    # be rejected, ts seems to be stationary.
    # for kpss_result, with p value = 0.01, test statistic = 1.258 is greater than 0.739 (1%), so H0 is rejected,
    # ts is not stationary. 

    # So for differentiated data (df_diff), both ADF + KPSS suggest stationarity.
    # For df (original data), ADF suggests stationarity, KPSS suggests non-stationarity.
    # with this guide: https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    # i should differntiate the original series (df), then check this for stationarity. i already did that, and
    # both tests suggest that data is stationary. so differencing was right call.

    print(adf_result[0])
    print(f"No differencing: \nadf: {adf_result[0]}\np-value: {adf_result[1]}\ncritical vals: {adf_result[4]}")
    while i <= num_differencing:
        df_diff["use_transfused"] = df_diff["use_transfused"].diff()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        df_diff["use_transfused"].plot(ax=ax1, lw=0.05)
        plot_acf(df_diff["use_transfused"].dropna(), ax=ax2)
        plot_pacf(df_diff["use_transfused"].dropna(), ax=ax3)
        fig.suptitle(f"differentiated {i}x")
        fig.show()

        adf_result = adfuller(df_diff["use_transfused"].dropna())
        print(f"No differencing: \nadf: {adf_result[0]}\np-value: {adf_result[1]}\ncritical vals: {adf_result[4]}")


        i += 1

    #seasonal differencing = subtracting value of previous season 
    # (i'll try week, so period = 7 (7 rows before in dataset))
    num_differencing = 3
    period = 365
    i = 1
    df_diff = df_processed.copy()

    df_diff["use_transfused"].plot(lw=0.05)
    plot_acf(df_diff["use_transfused"], title="No differentiation")
    plot_pacf(df_diff["use_transfused"], title="No differentiation")

    while i <= num_differencing:
        df_diff["use_transfused"] = df_diff["use_transfused"].diff(periods=period)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        df_diff["use_transfused"].plot(ax=ax1, lw=0.05)
        plot_acf(df_diff["use_transfused"].dropna(), ax=ax2)
        plot_pacf(df_diff["use_transfused"].dropna(), ax=ax3)
        fig.suptitle(f"differentiated {i}x")
        fig.show()
        i += 1




#%%--------------------------------------------------------------------------------
# MARK: MODEL BUILDING
#----------------------------------------------------------------------------------

#TODO: add multiple runs, to test for different parameters
#all of the above could be grouped into sarima.run() (if certain stuff is set up before, like vals for params and split!)
#TODO: add parameter grid, like p = 0-7, q=0, d=1-2 (for example), and then try all combinations of these params.
#TODO: Add simple forecasts: naive, naive seasonal, fixed line (like its now) as forecasts for comparison.
#TODO: add CI to plots? (Only one day stepwise ahead).
#TODO: change color scheme + x axis texts for plot_stepwise_forecast_errors
#TODO: save data to csv
#TODO: load data from csv



#%% 
# MARK: COMPARISON
#----------------------------------------------------------------------------------
importlib.reload(model)
importlib.reload(config) #for DEV_START_DATE

comp = model.ModelComparison(df)

comp.set_column()
comp.set_dates_mean()
comp.set_forecast_window()
comp.set_single_value()

comp.print_parameters()

comp.model_run()

comp.result

comp.get_error_values()


for col in comp.result.columns:
    if col == "use_transfused":
        plt.plot(comp.result.loc["2024-01-01":"2024-07-31", col], linewidth=1.5, label=col)
    else:
        plt.plot(comp.result.loc["2024-01-01":"2024-07-31", col], linewidth=0.5, label=col)
    plt.legend()
#%% 
# MARK: ARIMA
#----------------------------------------------------------------------------------

importlib.reload(model)
importlib.reload(config) #for DEV_START_DATE


arima = model.ModelArima(df)
# Test runs (it works as expected)
# arima.set_validation_expanding_window(train_percent=0.992, test_len=7, start_date="2022-01-01")
# arima.set_validation_single_split(train_percent=0.75)
arima.set_validation_rolling_window(train_percent=0.975, test_len=7, start_date=config.DEV_START_DATE) #TODO: change date/remove it

arima.set_model_parameters(7, 1, 1) #7,1,1, #TODO: add hyperparam grid
arima.model_run(col=config.COLUMN)

#Try out stepwise error measurements (now only mae):
arima.plot_stepwise(plot_type="forecast") #forecast
arima.plot_stepwise(plot_type="forecast difference") #forecast
arima.plot_stepwise(df=arima.stepwise_forecast_difference, plot_type="difference", comparison=False) #forecast difference
arima.plot_stepwise_forecast_errors()
print(arima.stepwise_forecast_errors)


#%% MARK: GRID SEARCH (ARIMA)
#TODO: ARIMA with grid search:
#dummy var names for now (partly)

#TODO: remove
if RUN_ALL == True:

    arima_gs = model.ModelArima(df)
    arima_gs_result = {}
    grid_params = {
        col: (config.COLUMN),
        p: [0, 7],
        d: [0, 3],
        q: [0, 7],
        "rolling_type" : ["expanding", "rolling"]
    }
    #TODO: convert setted grid min/max/choice to list of possible values
    #idea: min/max always as set, rest as list.
    grid_params_list = convert_gs_to_list(grid_params)


    for count, grid in enumerate(grid_params_list): #suppose this iterates over our min/max or possible values
        params = grid #store parameters

        if grid["rolling_type"] == "expanding":
            arima.set_validation_expanding(df)
        elif grid["rolling_type"] == "rolling":
            arima.set_validation_rolling(df)

        #run model with params:
        arima.set_model_parameters(7, 1, 1) #7,1,1, #TODO: add hyperparam grid
        arima.model_run(col=config.COLUMN)

        #save values
        res_fc = arima.forecastt #store forecast result
        res_og = arima.actual_values #store which actual values where used for calc errors
        res_errors = arima_gs.stepwise_forecast_errors

        #store in dict
        arima_gs_result.append({count: {
                "params": params,
                "forecast":res_fc,
                "actual":res_og,
                "errors":res_errors
            }}
        )







#%%
# MARK: SARIMAX
#----------------------------------------------------------------------------------


importlib.reload(model)
importlib.reload(config) #for DEV_START_DATE


sarima = model.ModelSarimax(df)
# Test runs (it works as expected)
# arima.set_validation_expanding_window(train_percent=0.992, test_len=7, start_date="2022-01-01")
# arima.set_validation_single_split(train_percent=0.75)
sarima.set_validation_rolling_window(train_percent=0.975, test_len=7, start_date=config.DEV_START_DATE) #TODO: change date/remove it

sarima.set_exogenous_vars(exog_cols=["tlmin", "workday_enc", "holiday_enc", "day_of_week", "day_of_year"])
sarima.set_model_parameters(p=7, d=1, q=1, P=0, D=0, Q=2, m=7) #7,1,1, #TODO: add hyperparam grid

sarima.model_run(pred_col=config.COLUMN)#, exog=["PAT_BG_0", "PAT_BG_A", "PAT_BG_AB", "PAT_BG_B"])

#Try out stepwise error measurements (now only mae):
sarima.plot_stepwise(plot_type="forecast") #forecast
sarima.plot_stepwise(df=sarima.stepwise_forecast_difference, comparison=False, plot_type="forecast difference") #forecast difference
sarima.plot_stepwise_forecast_errors()
print(sarima.stepwise_forecast_errors)



 
#%%----------------------------------------------------------------------------------
# MARK: LSTM
#----------------------------------------------------------------------------------

importlib.reload(model)
importlib.reload(config) #for DEV_START_DATE


lstm_m = model.ModelLSTM(df)


#TODO: finish grid search
#TODO: implement automatic filling of grid search var (list of dicts with all possible combinations)
#TODO: grid_search_list/params could be a df, where index is a consecutive number and columns are the test params.
# That way, would be easy to assign number to saved csv file.

#TODO: this would set the options/borders/steps for grid searching
exog_cols = ["use_discarded", "use_expired"]# TODO:put back in: , 'ward_AN', 'ward_CH', 'ward_I1', 'ward_I3', 'ward_Other', 'ward_UC', "workday_enc", "holiday_enc", "day_of_week", "day_of_year", "year", "tlmin", "tlmax"]



#TODO: grid search options/possiblilites rough idea:
# grid_search_lstm_options = {
#     "validation_type" : ["rolling", "expanding"],
#     "train_prct" : range(0.6, 0.8, step=0.1), #wouldnt it make more sense to use int of days before to train? like train_days = 365*7 or 730 or something?
#     "test_len" : [7],
#     "start_date" : [pd.to_datetime(day) for day in ["2008-01-01", "20012-01-01", "20016-01-01", "2020-01-01", "2024-01-01"]],
    
#     "memory_cell" : [32, 64, 128],
#     "epochs" : [20],
#     "batch_size" : [32],
#     "batch_size" : [0.3, 0.5, 0.7],
#     "pi_iterations" : [100, 1000],
#     "optimizer" : ["adam"],
#     "loss" : ["mae"],
#     "exog_cols" : exog_cols #this would need to be 0-all of them? or just 0 + all?
# }

# #TODO: grid search -- this is what a possible list of dicts could look like (missing exog_cols):
# grid_search_lstm = [{
#     "validation_type" : "rolling",
#     "train_prct" : 0.98,
#     "test_len" : 7,
#     "start_date" : pd.to_datetime("2020-01-01"),
    
#     "memory_cell" : 64,
#     "epochs" : 20,
#     "batch_size" : 32,
#     "batch_size" : 0.5,
#     "pi_iterations" : 100,
#     "optimizer" : "adam",
#     "loss" : "mae"
#     },
#     {
#     "validation_type" : "expanding",
#     "train_prct" : 0.97,    
#     "test_len" : 7,
#     "start_date" : pd.to_datetime("2015-01-01"),
    
#     "memory_cell" : 128,
#     "epochs" : 22,
#     "batch_size" : 36,
#     "batch_size" : 0.25,
#     "pi_iterations" : 120,
#     "optimizer" : "adam",
#     "loss" : "mae"
#     }
# ]


# #TODO: implement grid search for lstm (rought idea):
# for p in grid_search_lstm:
#     if p["validation_type"] == "rolling":
#         lstm_m.set_validation_rolling_window(train_percent=p["train_prct"], test_len=p["test_len"], start_date=p["start_date"])
#     elif p["validation_type"] == "expanding":
#         lstm_m.set_validation_expanding_window(train_percent=p["train_prct"], test_len=p["test_len"], start_date=p["start_date"])

#     lstm_m.set_model_parameters(**p) #should auto-unpack/-match params of p(grid search lstm)
#     lstm_m.model_run()
#     lstm_m.save_all() #TODO: saves model, prediction results (df), params, error values


#Simple run (for testing, beofre implementing grid search)
lstm_m.set_validation_rolling_window(
    #TODO: store validation_sets as df: index + columns train start/train end/test start/test end
    #TODO: add option to choose days for train and test period.
    train_percent=0.95,#9,#985,#975,
    test_len=3, 
    start_date="2024-12-01"
)

#%%
lstm_m.set_model_parameters(
    inner_window = 100, #365*2 #365 to capture at least 1 year, #for training length

    memory_cells=16,#64
    epochs=2,#20
    batch_size=32,
    dropout=0.9,
    pi_iterations=3, #100 #how often to run, to calculate prediction intervals
    optimizer="adam",
    loss="mae",
    activation_fct="relu",
    lower_limit=2.5,
    upper_limit=97.5
)
lstm_m.set_exogenous_cols(exog_cols = exog_cols)
lstm_m.set_prediction_column(prediction_column="use_transfused")

lstm_m.print_params()

lstm_params = lstm_m.get_params_df()
#%%
#Run model
lstm_m.model_run()

#%%
#Get error values + plotting



#%% 
# MARK: PROPHET
#----------------------------------------------------------------------------------

from prophet import Prophet

start_date = pd.to_datetime("2020-01-01")
split_date = pd.to_datetime("2023-12-31")
end_date = pd.to_datetime("2024-12-31")

pred_col = config.COLUMN
regressor_cols = ['EC_BG_0', 'EC_BG_A', 'EC_BG_AB', 'EC_BG_B', 'EC_RH_Rh_negative',
       'EC_RH_Rh_positive', 'EC_TYPE_EKF', 'EC_TYPE_EKFX', 'EC_TYPE_Other',
       'PAT_BG_0', 'use_discarded', 'use_expired', 'use_transfused']
sel_cols = [pred_col] + regressor_cols


train_df = df.loc[start_date:split_date, sel_cols]
test_df = df.loc[split_date:end_date, sel_cols]


df.info()
train_df.info()
test_df.info()


#%%
#try to run model (Prophet)
prophet_train = (
    train_df[pred_col]
    .reset_index()
    .rename(columns={"date":"ds", config.COLUMN:"y"})
    )

prophet_train.info()
print(prophet_train.head())

#%%
m = Prophet(weekly_seasonality=True, interval_width=0.95)
m.add_country_holidays(country_name="Austria")

m.fit(prophet_train)
future_dates = m.make_future_dataframe(periods=365, freq="D")

fc = m.predict(future_dates)
#%%
m.plot(fc, uncertainty=True)

fc.head()
#.loc["2024-01-01":"2024-03-01"]

# m.add_regressor()











#%%--------------------------------------------------------------------------------
# MARK: VIZ RESULTS
# DATA VIZ (FINISHED MODEL) 
#----------------------------------------------------------------------------------
# Plot prediction vs actual

# If OOP:
# sarima.plot_time()
# sarima.plot_polar()

# lstm.plot_time()
# etc. (could even loop: for obj in [sarima, lstm]: obj.plot_time() obj.plot_polar())

# if functional:

#plot_time(sarima)  # could also usy apply or similar to use list or loop: for mod in models: plot_time(mod)

#TODO: save to csv





#%%--------------------------------------------------------------------------------
# MARK: EVALUATION
#----------------------------------------------------------------------------------

# TODO: load data from csv

# evaluate ....
# print evaluations/tests like mae, mape, etc.


# TODO: save data to csv










































#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# SAMPLE DATA
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#%% ------------------------------------------------------------------------------
# Get data & clean
from sklearn.datasets import fetch_openml
df = fetch_openml("seoul_bike_sharing_demand", as_frame=True)
df = pd.DataFrame(df.frame)

df.rename(mapper=config.seoul_name_map, axis=1, inplace=True)
print("seoul head\n\n")
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df["hour"] = pd.to_datetime(df["hour"], format="%H").dt.time

load.show_info(df=df)

#%% Skip -- CLEANING -- 


# ------------------------------------------------------------------------------
# -- PROCESSING -- 

# Merge date + time --> date
# df["hour"] = pd.to_datetime(df["hour"], format="%H:%M:%S")
df["date"] = pd.to_datetime(df.date.astype(str) + " " + df.hour.astype(str))
df = df.drop(columns="hour")

# Show info/head
df.info()
df.head()

# Aggregate daily:
print(df.info())
df = df.drop(columns=["seasons", "holiday", "functioning_day"])
df = df.resample('D', on="date").sum()
#df = df.reset_index()
load.show_info(df=df)





#---------------------------------------------------------------------------------
# -- VISUALIZE OOP --
# --------------------------------------------------------------------------------
# Daily average ________________________________________________________________
# TODO: wrap in function (or add to Model?, because its already cleaned+processed here, so next step
# besides viz would be add to model anyway? BUT exploratory viz is done on raw data, so no specific model...
# TODO: make prettier: add title, colorchart (so i can later exchange colors), etc.


#%%
# Load data as Class Data:
#TODO: rename previous 'df' to 'df_preprocessed' or something, 
# to differentiate between Data object and DataFrame object
importlib.reload(data_model)
df = data_model.Data(data=df)
#%%
df.plot_seasonal(plot_type='daily', col_name='count')
df.plot_seasonal(plot_type='weekly', col_name='count')



#%%
#Boxplots
df.plot_boxplots(col_name='count')
df.plot_seasonal_subseries(col_name='count') #NOTE: i think it works, but not enough dummy data.
#TODO: check if seasonal subseries plot works with multi-year data


#%%
#Decompose
df.decompose_one(col_name='count')
#df.decompose_all("use_transfused")

# mulitple decomposition (daily + weekly)
df.multiple_decompose(col_name="use_transfused", periods=[24, 24*7])





#%%
#Time series plots (acf, pacf etc)
df.plot_autocorrelation(col_name='count')
df.plot_partial_autocorrelation(col_name='count')


#%%
df.plot_daily_heatmap(col_name='count')


#%%

import holidays

vie_holidays = holidays.country_holidays('Austria', subdiv='W')


print(vie_holidays)

vie_holidays.get('2024-01-01')

print(vie_holidays.is_working_day('2024-01-01'))
print(vie_holidays.is_working_day('2024-12-24'))
print(vie_holidays.is_working_day('2005-12-25'))















#%% ------------------------------------------------------------------------------
# -- PROCESSING 2 --
# --------------------------------------------------------------------------------
#%%

# Order of differencing "d" -- detrending
#detrending/plotting:
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

plot_acf(df["use_transfused"])
#%% Detrend ______________________________________________________________________
f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("No differencing")
ax1.plot(df["use_transfused"])

ax2 = f.add_subplot(122)
plot_acf(df["use_transfused"].dropna(), ax=ax2)
plt.show()


f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("1st order differencing")
ax1.plot(df["use_transfused"].diff())

ax2 = f.add_subplot(122)
plot_acf(df["use_transfused"].diff().dropna(), ax=ax2)
plt.show()


f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("2nd order differencing")
ax1.plot(df["use_transfused"].diff().diff())

ax2 = f.add_subplot(122)
plot_acf(df["use_transfused"].diff().diff().dropna(), ax=ax2)
plt.show()


f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("3rd order differencing")
ax1.plot(df["use_transfused"].diff().diff().diff())

ax2 = f.add_subplot(122)
plot_acf(df["use_transfused"].diff().diff().diff().dropna(), ax=ax2)
plt.show()


#%% Dickey-Fuller Test
res = adfuller(df["use_transfused"].dropna())
print("p-value: ", res[1])

res = adfuller(df["use_transfused"].diff().dropna())
print("p-value: ", res[1])

res = adfuller(df["use_transfused"].diff().diff().dropna())
print("p-value: ", res[1])

res = adfuller(df["use_transfused"].diff().diff().diff().dropna())
print("p-value: ", res[1])

# Results:
# p-value:  0.0936973558926073
# p-value:  1.879297433670586e-25
# p-value:  1.4284641410804533e-16
# p-value:  1.762485024258537e-20
# significance level: 0.05
# so after one diff(), is good, data is stationary. above 0.5 is not stationary
# so we assume order of differencing d = 1
# from https://www.projectpro.io/article/how-to-build-arima-model-in-python/544



#%% Computing "p" -- Order of autoregressive Model
from statsmodels.graphics.tsaplots import plot_pacf

f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("No differencing")
ax1.plot(df["use_transfused"])

ax2 = f.add_subplot(122)
plot_pacf(df["use_transfused"].dropna(), ax=ax2)
plt.show()


f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("1st order differencing")
ax1.plot(df["use_transfused"].diff())

ax2 = f.add_subplot(122)
plot_pacf(df["use_transfused"].diff().dropna(), ax=ax2)
plt.show()


f = plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title("2nd order differencing")
ax1.plot(df["use_transfused"].diff().diff())

ax2 = f.add_subplot(122)
plot_pacf(df["use_transfused"].diff().diff().dropna(), ax=ax2)
plt.show()

#In the pacf plot, we can see the first lag to be most significant.
# so "p" = 1
# Order of "q" = 1: 
# looking at acf (not pacf), we can also see, only
# first lag is most significant 

# So d, p, q = 1


#%% Fit arima model
from statsmodels.tsa.arima.model import ARIMA

arima_model = ARIMA(df["use_transfused"], order=(1,1,2))
model = arima_model.fit()
print(model.summary())

#%% plot
from statsmodels.graphics.tsaplots import plot_predict

#Method 1
fig, ax = plt.subplots()
ax = df["use_transfused"].plot(ax=ax)
plot_predict(model, ax=ax)
plt.show()

#Method 2
pred = model.predict(dynamic=False)
plt.plot(pred)
plt.plot(df["use_transfused"])


#%% SPLIT into training and test data
# Same as above but withs split data

from statsmodels.tsa.arima.model import ARIMA

#function for getting first x% of rows:
def get_split_rows(data, perc=0.8):
    n_rows_train = int(len(data)*perc)
    n_rows_test = len(data) - n_rows_train
    return (n_rows_train, n_rows_test)

split_n_rows = get_split_rows(df)
training_set = get_first_percent_rows(df[:split_n_rows[0]])

arima_model = ARIMA(training_set["use_transfused"], order=(1,1,2))
model = arima_model.fit()
print(model.summary())

# plot
from statsmodels.graphics.tsaplots import plot_predict
fig, ax = plt.subplots()
ax = df["use_transfused"].plot(ax=ax)
plot_predict(model, ax=ax)
plt.show()

#%% Run prediction on test set:

y_pred = pd.Series(model.forecast(split_n_rows[1])[0], index=df["use_transfused"][split_n_rows[0]:].index)
y_true = df["use_transfused"][split_n_rows[0]]

print(np.array(y_pred).astype(np.uint8))
print(np.array(y_true))


#%% Now trying from https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
from sklearn.metrics import mean_squared_error
from math import sqrt

# Split in train+test:
series = df["use_transfused"]
print(series)
#%%
# series.index = series.index.to_period('D')
X = series.values
perc = 0.66
size = int(len(X) * perc)

train, test = X[0:size], X[size:len(X)]
history = [x for x in train]

predictions = list()

#Walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,2))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    print("predicted = %f, expected=%f" % (yhat, obs))

#%% evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print("Test RMSE: %.3f" % rmse)

#plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color="red")
plt.show()


