#TODO: is it necessary to load the modules here?
from src import data_model
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm #color palette
from sklearn.model_selection import train_test_split
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import sklearn.metrics as metrics #error metrics (mae, mape etc)


# MARK: Model
class Model:
    #methods for running are in the child classes for the individual models.
    # here only general methods needed for the models


    def __init__(self, data: data_model.Data): #TODO: maybe add configuration?
        # TODO: change parameter of model to be of 'Data' type 
        self.data = data

        #Train/Test sets:
        self.split_index = None #Index at split point
        self.train_data = None
        self.test_data = None
        self.train_test_data = None #list of tuples containg train/test df index pairs

        #Config and date sets for splitting data (train/test)
        self.validation_config = {}
        self.validation_sets = None
        #save forecasts as 1-day fc, 2-day-fc, for plotting:
        # when i set test_len as 7 days, i want to save all 1-day look aheads, all for the 2nd day etc
        self.stepwise_forecasts = pd.DataFrame()

        #Model(s)
        self.models = []
        self.model_fits = []
        self.predictions = []
        #self.model_runs = list() #List of past runs
        #TODO: save model runs as well as their state (parameters etc.)
        # so they can be accessed later. save to file.

        #Forecast errors (stepwise, so for forecast 1 day ahead, 2 days ahead, etc.):
        #Values are pd.Dataframes, with stepwise errors for forecast timeframe:
        # ME -- Median error (shows model bias to be positive or negative)
        # MAE -- Mean absolute error
        # MedAE -- Median absolute error
        # MAPE -- Mean absolute percentage error
        # sMAPE -- Symmetric mean absolute percentage error
        # MASE -- Mean absolute scaled error
        # MSE -- Mean squared error
        # RMSE -- Root mean squared error
        # RMSSE -- root mean squared scaled error
        # MaxError -- maximum error
        self.stepwise_forecast_errors = pd.DataFrame()

        #Difference between stepwise forecast values to actual values
        self.stepwise_forecast_difference = pd.DataFrame()

        #Dont think is needed:
        #self.params = None #rename; prob better in base class.
        


        #Decompositions: (s.u., gehört eig. nciht hierher)
        # self.decomp = None






    #------------------------------------------------------------------------------------------------
    # Helper functions
    # MARK: HELPERS
    #------------------------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------------------------
    # SETTERS
    # MARK: Setters

    def set_validation_expanding_window(self, train_percent: float, test_len: int, start_date: str=None):
        """
        Set parameters of validation_config variable (doesnt RUN validation) and create a 
        set of indices (list of tuples) for train/test split by calling make_validation_set().
        There is only one config variable, so using setter again or for another method (rolling window)
        overwrites this setting!

        Expanding window validation means, that an initial train set is expanded, 
        by x days each iteration (here x = 1 day).


        Args:
        :param train_percent: Percentage of whole dataset, that should be used as initial training set
        :type train_percent: float 
        :param test_len: Number of days, to test into the future
        :type test_len: int
        :param start_date: Optional str in date format YYYY-MM-DD, to set start date of data set. Default None.
        :type start_date: str (, optional)
        
        :raises ValueError: if bla bla bla
        
        :return: Nothing, but sets validation_config member variable.
        :rtype: None
        """
        
        if not start_date:
            start_date = self.data.index.min()
        else:
            start_date = pd.to_datetime(start_date)

        print(start_date)

        train_len = len(self.data)

        self.validation_config = {
            "type" : "expanding window",
            "train_percent" : train_percent,
            "test_len" : test_len,
            "start_date" : start_date
        }

        self.make_validation_set()



    def set_validation_rolling_window(self, train_percent: float, test_len: int, start_date: str=None):
        """
        Set parameters of validation_config variable (doesnt RUN validation) and create a 
        set of indices (list of tuples) for train/test split by calling make_validation_set().
        There is only one config variable, so using setter again or for another method (expanding window)
        overwrites this setting!

        Rolling window validation means, a fixed number of days is used for training, which
        rolls over dataset, and the following days of number <test_len> are used as test set.

        Parameters
        ----------
        train_percent : float
            float between 0-1, sets percentage of data to assign 
            as training set.
        test_len : int
            Number of days to set as test set.
        start_data : str, optional
            Optional str in date format YYYY-MM-DD, to set 
            start date of data set. Default None.
        """
        if not start_date:
            start_date = self.data.index.min()
        elif pd.to_datetime(start_date) < pd.to_datetime(self.data.index.min()):
            raise ValueError("'start_date' must be within data")
        else:
            start_date = pd.to_datetime(start_date)

        self.validation_config = {
            "type" : "rolling window",
            "train_percent" : train_percent,
            "test_len" : test_len,
            "start_date" : start_date
        }

        self.make_validation_set()



    def set_validation_single_split(self, train_percent: float, start_date: str=None):
        """
        Set parameters of validation_config variable (doesnt RUN validation) and create a 
        set of indices (list of tuples) for train/test split by calling make_validation_set().
        There is only one config variable, so using setter again or for another method (rolling window)
        overwrites this setting!

        Single split validation means, that whole dataset is split a single time into 
        train and test sets.

        Parameters
        ----------
        train_percent : float
            float between 0-1, sets percentage of data to assign as training set
        start_data : str
            Optional str in date format YYYY-MM-DD, to set start date of data set. Default None.
        """
        if not start_date:
            start_date = self.data.index.min()
        else:
            start_date = pd.to_datetime(start_date)

        self.validation_config = {
            "type" : "single split",
            "train_percent" : train_percent,
            "start_date" : start_date
        }

        self.make_validation_set()


    #------------------------------------------------------------------------------------------------
    # General
    # MARK: General
    def make_validation_set(self, steps: int=1):
        """
        Make a list of tuples that contain (train_start, train_end, test_start, test_end), to use for validation.

        Args:
            validation_config (dict): member variable containing configs for which kind of validation, start date and length (percent) of train set
            steps (int, optional): Days to step ahead in each iteration. Defaults to 1.

        Returns nothing. Sets self.validation_sets as a list of tuples (in case of single split a single tuple)
        """
        #validation_config = self.validation_config

        # #calculate number of iterations/steps
        # train_len = train_len + validation_config["test_len"]
        # len_data = end_date - train_start #No. of days it data from start_date to the end
        # num_of_iterations = len_data - len_train_test + 1 #number of steps

        #Just with pd offset and while loop (no calculation of iterations)
        end_date = self.data.index.max() #last day in dataset

        train_start = self.validation_config["start_date"] #is (should be) pd datetime
        train_end = self.get_split_index_by_prct(start_date=self.validation_config["start_date"], prct=self.validation_config["train_percent"])
        #train_len = train_end - train_start
        
        test_start = train_end + pd.DateOffset(1)

        if self.validation_config["type"] == "single split":
            test_end = end_date
        else:
            test_end = test_start + pd.DateOffset(self.validation_config["test_len"])


        train_test_indices = []

        match self.validation_config["type"]:
            case "expanding window":
                while test_end <= end_date:
                    step_values = (train_start, train_end, test_start, test_end)
                    train_test_indices.append(step_values)

                    #Note: keep train_start always the same
                    train_end = train_end + pd.DateOffset(steps)
                    test_start = test_start + pd.DateOffset(steps)
                    test_end = test_end + pd.DateOffset(steps)


            case "rolling window":
                while test_end <= end_date:
                    step_values = (train_start, train_end, test_start, test_end)
                    train_test_indices.append(step_values)

                    train_start = train_start + pd.DateOffset(steps)
                    train_end = train_end + pd.DateOffset(steps)
                    test_start = test_start + pd.DateOffset(steps)
                    test_end = test_end + pd.DateOffset(steps)


            case "single split":
                train_test_indices.append((train_start, train_end, test_start, test_end))
        
        self.validation_sets = train_test_indices



    def add_stepwise_forecasts(self):
        """
        Add stepwise_forecasts variable: dictionary containing number of keys in the length of test_len, 
        with values as pandas Series of forecasted values for respective days to look ahead. 
        E.g. if test_len (in rolling/expanding window) 
        is 7 days, keys "1" to "7" are added, containing the predicted value for predictions 
        as far into the future as the key.
        Used for plotting and comparing, how well the prediction works into the future.

        Returns nothing, sets self.stepwise_forecasts as dataframe with days ahead as 
        columns ("Days ahead: [val]") and prediction_mean as value/columns. Datetime index
        """
        #TODO: Fix wrong results: days ahead: 1 should have data until the end

        #not run if single split validation (has no test_len):
        if self.validation_config["test_len"]:
            #Add 'step' (day x) ahead
            for step in range(1, self.validation_config["test_len"] + 1):
                step_forecasts = pd.Series()
                for pred in self.predictions:
                    step_forecasts = pd.concat([step_forecasts, pred["Prediction"].iloc[[step-1]]])
                
                step_forecasts.sort_index(inplace=True)
                if step == 1:
                    self.stepwise_forecasts = step_forecasts.to_frame()
                else:
                    self.stepwise_forecasts = pd.concat([self.stepwise_forecasts, step_forecasts], axis=1)
        
                #Rename last col to string of days to look ahead:
                self.stepwise_forecasts.columns = [*self.stepwise_forecasts.columns[:-1], f"Days ahead: {step}"]
                

    #NOTE: not in use currently
    def split_by_percentage(self, percent: float=0.33, start_date=None):
        """
        percent = percent to assign as train data. 
        """
        
        if percent >= 1 or percent <= 0:
            raise Exception(f"Training/test data ratio: test size: {percent}) must be float smaller than 1 (should be >0.5)")

        self.split_index = int(len(self.data)*percent)
        self.train_data = self.data.iloc[: self.split_index]
        self.test_data = self.data.iloc[self.split_index:]



    #------------------------------------------------------------------------------------------------
    # Getters
    # MARK: Getters
    def get_validation_type(self):
        print("Validation type:", self.validation_type["type"])



    def get_split_index_by_prct(self, start_date, prct: float=0.77):
        """
        returns datetime index of split position.
        """
        df_len = len(self.data[start_date:])
        if prct <= 0 or prct > 1:
            raise ValueError("must be between 0 and 1, not 0")
        if df_len <= 2:
            raise ValueError("df must be longer than 2 rows")
        
        idx = int(len(self.data[start_date:]) * prct)
        split_date = self.data[start_date:].index[idx]

        print(f"Split index/date with start_date ({start_date}: \n {split_date})")

        return split_date



    def add_stepwise_errors(self, col_pred: str="count"):
        """Calculate stepwise errors for stepwise forecasts and adds it to model variable. 
        That means that for x days ahead, for all predicted values, the supplied error is calculated.
        Currently Supported error metrics: MAE, MAPE, MedAE (Median absolute error), 
        MaxError, RMSE, MSE
        Args:
            col_pred (str, optional): Column name of original (test) data which is forecast, to compare preiction with
        """
        #initialize empty df with structure like stepwise_forecasts (cols, indices, no content)
        forecast_steps = self.stepwise_forecasts.columns
        errors = ["ME", "MAE", "MedAE", "MAPE", "RMSE", "MASE", "MaxError"] 

        self.stepwise_forecast_errors = pd.DataFrame(columns=errors, index=forecast_steps)

        #add error measurements for each forecast step
        for col in self.stepwise_forecasts.columns:
            min_date = self.stepwise_forecasts[col].first_valid_index()
            max_date = self.stepwise_forecasts[col].last_valid_index()

            y_pred = self.stepwise_forecasts.loc[min_date:max_date, col]
            y_true = self.data.loc[min_date:max_date, col_pred]
        
            self.stepwise_forecast_errors.loc[col, "ME"] = np.median(y_pred - y_true) #median error -- shows bias (positive or negative)
            self.stepwise_forecast_errors.loc[col, "MAE"] = metrics.mean_absolute_error(y_pred=y_pred, y_true=y_true)
            self.stepwise_forecast_errors.loc[col, "MedAE"] = metrics.median_absolute_error(y_pred=y_pred, y_true=y_true)
            self.stepwise_forecast_errors.loc[col, "MAPE"] = metrics.mean_absolute_percentage_error(y_pred=y_pred, y_true=y_true)
            self.stepwise_forecast_errors.loc[col, "MSE"] = metrics.mean_squared_error(y_pred=y_pred, y_true=y_true)
            self.stepwise_forecast_errors.loc[col, "RMSE"] = metrics.root_mean_squared_error(y_pred=y_pred, y_true=y_true)
            self.stepwise_forecast_errors.loc[col, "MaxError"] = metrics.max_error(y_pred=y_pred, y_true=y_true)

    def add_stepwise_difference(self, col_pred: str="count"):
        """Get a df with the difference between the stepwise forecasted values and 
        the actual values. By default it subtracts 'count' from the daily stepwise
        forecasted values and stores them in a new df called 'stepwise_forecast_difference'.

        Args:
            col (str, optional): Column name that should be subtracted from the forecasted values. 
            Defaults to "count".
        """
        y_true = self.data[col_pred].loc[self.stepwise_forecasts.index]
        self.stepwise_forecast_difference = self.stepwise_forecasts.sub(y_true, axis=0)





    # Gehört eigentlich nichth zum model, passiert ja vorher:
    # def decompose(self, ):
    #     self.decomp = seasonal_decompose(series, model='additive')
    #     print(self.decomp.trend)
    #     print(self.decomp.seasonal)
    #     print(self.decomp.resid)
    #     print(self.decomp.observed)

    def make_stationary(self, data):
        #make stationary (remove trend)
        #maybe move to corresping model that needs it?
        # I think actually has to happen before, in data processing, but
        # if splitting also happens here, than i could do it here, before the split?
        # but would be needed before for autocorrelatio plots (data exploration!)
        # --> move stationary/detrend + splitting as functions in PROCESSING.
        # then only pass train and test data set to Model.
        # df = xxx
        # return df
        pass


    def validate_expanding_window(self, data, w):
        """
        w = window size in days
        """
        #split data


        #run model against test data set wtih expanding window
        pass




    def evaluate_model(self):
        #run evaluations, to get values like MAE, MAPE etc.
        pass

    def run_MAE():
        pass
    
    def run_MAPE():
        pass

    def run_MSRE():
        pass


    

    #UNCLEAR: add extra plotting here or just use the functions? 

    #------------------------------------------------------------------------------------------------
    # Plotters
    # MARK: Plotters
    #TODO: implement selection for days to plot ahead
    def plot_stepwise(self, plot_type: str, df: pd.DataFrame=None, comparison=True, comparison_col="count", days=None):
        """Plot the stepwise predictions or difference against actual/true values.
        i.e. plot e.g. one-day-ahead prediction against test set, two-day-ahead, etc.
        Defaults to plot stepwise_forecasts data, but can also be used for stepwise_forecast_difference 
        (which is alos stepwise).
        """
        if df is None:
            df = self.stepwise_forecasts

        comparison_data = self.data[comparison_col]


        colors = iter(cm.rainbow(np.linspace(1, 0.6, len(df.columns))))

        #original_data = 
        plt.figure(figsize=(14,7))

        if plot_type == "forecast difference":
            plt.axhline(y= 0, linestyle = "dashed", color="lightgrey")
            for col in df.columns:
                print("original: ", col, "\n", df[col])
                df[col] = comparison_data - df[col]
                print("comparison: ", col, "\n", comparison_data)
                print("subtracted: ", col, "\n", df[col])
    

        if comparison == True:
            plot_start = df.index.min() - pd.DateOffset(60) #first element of first key
            plot_end = df.index.max()#self.stepwise_forecasts.keys()[-1][-1] #last element of last key
            plt.plot(self.data[plot_start:plot_end][comparison_col], label="original data")


        for col in df.columns:
            plt.plot(df[col], label=col, color=next(colors))

        plt.title(f"Stepwise {plot_type}; Model: {self.class_name}")
        plt.legend()
        plt.show()



    def plot_stepwise_forecast_errors(self):
        #TODO: change colors to be more different. 
        # TODO: maybe add error names directly to lines instead of having a legend 
        print(self.stepwise_forecast_errors.columns)
        print(len(self.stepwise_forecast_errors.columns))
        colors = iter(cm.gist_ncar(np.linspace(1, 0, len(self.stepwise_forecast_errors.columns))))

        for col in self.stepwise_forecast_errors.columns:
            if col == "MSE":
                continue
            plt.plot(self.stepwise_forecast_errors[col], label=col, color=next(colors))
        plt.tight_layout()
        plt.title(f"Forecast errors; Model: {self.class_name}")
        plt.legend()
        plt.show()


    



#--------------------------------------------------------------------
# Individual Models:
#--------------------------------------------------------------------

# MARK: Comparison
# Comparison Model
# single value
# naive/persistent (n-1)
# mean
# seasonal naive (n-7)
#
class ModelComparison(Model):
    class_name = "Comparison"

    def __init__(self, data, single_value: int):
        super().__init__(data)
        self.forecast_window = None
        self.result = None #Dataframe

        #individual comp. models (dataframes)
        self.single_value = None
        self.naive = None
        self.mean = None
        self.seasonal_naive = None


    def set_forecast_window(self, days=14):
        """This is for plotting purposes only. Since models have forecast
        window, with days-ahead plotting/stats, i want to keep this 
        model with the same window.

        Args:
            days (int, optional): Das ahead forecast. FOr plotting and stats. Defaults to 14.
        """

        self.forecast_window = days


    def model_run(self):
        #Run all comparison models
        pass

    def run_single_value(self):

        pass

    def run_naive(self):
        pass

    def run_mean(self):
        pass

    def run_seasonal_naive(self):
        pass

    



# ARIMA
# MARK: ARIMA
class ModelArima(Model):
    class_name = "Arima"

    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)
        self.p = None
        self.q = None
        self.d = None

    #------------------------------------------------------------------------------------------------
    # Setters
    #------------------------------------------------------------------------------------------------
    
    def set_model_parameters(self, p: int=1, d: int=1, q: int=1):
        self.p = p
        self.d = d
        self.q = q



    #composite function:
    def model_run(self, col: str="count", print_fit_summary=True, last_only=True, days=None):
        """Composite function that combines make_model, fit(), print_fit_summary(), predict(),
        add_stepwise_forecasts()

        Args:
            col (str, optional): column to make model and run prediction for. 
            Defaults to "count".
            print_fit_summary (bool, optional): If true, prints the summary for the fit().
            Defaults to True
            last_only (bool, optional): If true prints only summary for last fit(). Otherwise prints
            summary for every fit (of rolling/expanding window).
            Only relevant if print_fit_summary argument is true. defaults to True.
            days (int, optional): Manually set days to look ahead (steps). Normally supplied via
            validation_config by setter functions for rolling/expanding window or single split.
            Defaults to None, which will then use abovementioned value. 
        """

        self.make_model(col=col)
        self.fit()
        if print_fit_summary:
            self.print_fit_summary(last_only=last_only)
        self.get_prediction(days=days)

        #Get stepwise values:
        self.add_stepwise_forecasts()
        self.add_stepwise_errors(col_pred=col)
        self.add_stepwise_difference(col_pred=col) 




    def make_model(self, col: str):
        """
        create model with trainign data + (hyper)parameters
        
        Parameters
        ----------
        col : string 
            column (target) to use for for univariate forecasting 
        """
        #Important!
        self.models = []
        
        #TODO: set up split AND validation
        # i think for validation, best option to have a list of lists with train_start, train_end, test_start, test_end
        # days (datetime), which i can cycle here (make_model, fit, print_fit_summary, predict), which is just
        # inplace filtering of df, so i dont have to store multiple dfs!
        series = self.data[col]
        #TODO: !use SARIMAX instead of ARIMA!
        
        for train_set in self.validation_sets:
            #Add exogenous, check for enforce_stationarity, enforce_invertibility
            self.models.append(ARIMA(
                endog=series[train_set[0] : train_set[1]], 
                order=(self.p, self.d, self.q))) 


    def fit(self):
        #Important!
        self.model_fits = []

        for model in self.models:
            self.model_fits.append(model.fit())


    def print_fit_summary(self, last_only=True):
        if last_only:
            print(self.model_fits[-1].summary())
        else:
            for fit in self.model_fits:
                print(fit.summary())

    def predict(self, days=None):
        """Predict x days ahead, where x == 'days'

        Args:
            days (_type_, optional): Days to predict ahead. Defaults to None, then
            days will be loaded from validation_config["test_len"].
        """
        # generate forecast for x time
        # (see base class)

        #Important!
        self.predictions = []

        if days == None:
            days = self.validation_config["test_len"]

        for fit in self.model_fits:
            self.predictions.append(fit
                .get_forecast(steps=days)
                .rename(columns={"predicted_mean":"Prediction"})
            )

    def get_prediction(self, days=None, alpha: int=0.05):
        self.predictions = []

        #iterate over both model_fits and validation_sets:
        for fit, validation_set in zip(self.model_fits, self.validation_sets): #'fit' is the fitted model
            test_start = validation_set[2]
            test_end = validation_set[3]

            self.predictions.append(fit
                .get_prediction(start=test_start, end=test_end, dynamic=True)
                .summary_frame(alpha=alpha)
                .rename(columns={"mean":"Prediction"})
            )






# SARIMAX
# MARK: SARIMAX
class ModelSarimax(Model):
    class_name = "Arima"


    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)
        #ARIMA part:
        self.p = None
        self.q = None
        self.d = None
        #Seasonal part:
        self.P = None
        self.Q = None
        self.D = None
        self.m = None
        #eXogenous part:
        self.exog_cols = None

    #------------------------------------------------------------------------------------------------
    # Setters
    #------------------------------------------------------------------------------------------------
    
    def set_model_parameters(
            self, p: int=1, d: int=1, q: int=1, 
            P: int=1, D: int=1, Q: int=1, m: int=7):
        
        self.p = p
        self.d = d
        self.q = q

        self.P = P
        self.D = D
        self.Q = Q
        self.m = m

       
    def set_exogenous_vars(self, exog_cols: list):
        """Set columns to use for exogenous variables with SARIMAX. 

        Args:
            exog_cols (list): List of strings containing column names for exog vars (in self.data)

        Raises:
            ValueError: If exog_cols list is empty. Doesnt check for data type.
            ValueError: If a column from exog_cols is not present in 'df'
        """

        #Check df/exog_cols input for validity
        if len(exog_cols) == 0:
            raise ValueError("Need to pass col name present in 'df' to exog_cols")
        else:
            for col in exog_cols:
                if col not in self.data.columns:
                    raise ValueError(f"{col} not present df's columns: {self.data.columns}")
                
        self.exog_cols = exog_cols



    #composite function:
    def model_run(self, pred_col: str="count", print_fit_summary=True, last_only=True, days=None): #exog: list=None, 
        """Composite function that combines make_model, fit(), print_fit_summary(), predict(),
        add_stepwise_forecasts()

        Args:
            pred_col (str, optional): column to make model and run prediction for. 
            Defaults to "count".
            print_fit_summary (bool, optional): If true, prints the summary for the fit().
            Defaults to True
            last_only (bool, optional): If true prints only summary for last fit(). Otherwise prints
            summary for every fit (of rolling/expanding window).
            Only relevant if print_fit_summary argument is true. defaults to True.
            days (int, optional): Manually set days to look ahead (steps). Normally supplied via
            validation_config by setter functions for rolling/expanding window or single split.
            Defaults to None, which will then use abovementioned value. 
        """

        self.make_model(pred_col=pred_col)
        self.fit()
        if print_fit_summary:
            self.print_fit_summary(last_only=last_only)
        self.get_prediction(days=days)

        #Get stepwise values:
        self.add_stepwise_forecasts()
        self.add_stepwise_errors(col_pred=pred_col)
        self.add_stepwise_difference(col_pred=pred_col) 



    def make_model(self, pred_col: str):
        """
        create model with trainign data + (hyper)parameters
        
        Parameters
        ----------
        pred_col : string 
            columns (target) to use for for univariate forecasting 
        """
        #Important!
        self.models = []
        
        #TODO: set up split AND validation
        # i think for validation, best option to have a list of lists with train_start, train_end, test_start, test_end
        # days (datetime), which i can cycle here (make_model, fit, print_fit_summary, predict), which is just
        # inplace filtering of df, so i dont have to store multiple dfs!
        series = self.data[pred_col]
        #TODO: !use SARIMAX instead of ARIMA!

        
        if self.exog_cols == None:
            for train_set in self.validation_sets:
                self.models.append(SARIMAX(
                    endog=series[train_set[0] : train_set[1]], 
                    order=(self.p, self.d, self.q),
                    seasonal_order=(self.P, self.D, self.Q, self.m))) 
                
        elif self.exog_cols != None:
            # exogenous = self.data[exog]
            for train_set in self.validation_sets:
                self.models.append(SARIMAX(
                    endog=series[train_set[0] : train_set[1]], 
                    exog=self.data.loc[train_set[0]:train_set[1], self.exog_cols], 
                    # exog=exogenous[train_set[0] : train_set[1]], 
                    order=(self.p, self.d, self.q),
                    seasonal_order=(self.P, self.D, self.Q, self.m))) 



    def fit(self):
        #Important!
        self.model_fits = []

        for model in self.models:
            self.model_fits.append(model.fit())


    def print_fit_summary(self, last_only=True):
        if last_only:
            print(self.model_fits[-1].summary())
        else:
            for fit in self.model_fits:
                print(fit.summary())

    def predict(self, days=None):
        """Predict x days ahead, where x == 'days'

        Args:
            days (_type_, optional): Days to predict ahead. Defaults to None, then
            days will be loaded from validation_config["test_len"].
        """
        # generate forecast for x days ahead
        # (see base class)

        #Important!
        self.predictions = []

        if days == None:
            days = self.validation_config["test_len"]

        for fit in self.model_fits:
            self.predictions.append(fit.get_forecast(steps=days))


    def get_prediction(self, days=None, alpha: int=0.05):
        self.predictions = []

        #iterate over both model_fits and validation_sets:
        for fit, validation_set in zip(self.model_fits, self.validation_sets): #'fit' is the fitted model
            test_start = validation_set[2]
            test_end = validation_set[3]

            exog_prediction = self.data.loc[test_start:test_end, self.exog_cols]

            self.predictions.append(fit
                .get_prediction(start=test_start, end=test_end, exog=exog_prediction, dynamic=True)
                .summary_frame(alpha=alpha)
                .rename(columns={"mean":"Prediction"})
            )


    


    #------------------------------------------------------------------------------------------------
    # Setters
    #------------------------------------------------------------------------------------------------
    
    def print_params(self):
        print(f"p,d,q: {self.p}, {self.d}, {self.q}\nP,D,Q,m: {self.P},{self.D},{self.Q}{self.m}\nExogenous columns: {self.exog_cols}")




# LSTM
# MARK: LSTM

class ModelLSTM(Model):

    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)
        #LSTM-specific variable inits
        self.model = None
        self.params = None #rename; prob better in base class.

    def create_model(self, params: list, days):
        #create model with (hyper)parameters
        #params are model parameters
        # days are days to predict
        pass



# Prophet
# MARK: Prophet
class ModelProphet(Model):

    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)
        #prophet-specific variable inits
        self.model = None
        self.params = None #rename; prob better in base class.

    def create_model(self, params, days):
        #create model with (hyper)parameters
        #params are model parameters
        # days are days to predict
        pass
