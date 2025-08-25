#TODO: is it necessary to load the modules here?
from src import data_model
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm #color palette
from sklearn.model_selection import train_test_split
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX

import sklearn.metrics as metrics #error metrics (mae, mape etc)


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
        #Values are pd.Dataframes, with stepwise errors for forecast timeframe
        self.forecast_errors = {
            "MAE" : [], #Mean absolute error
            "MedAE" : [], #Median absolute error
            "MAPE" : [], #Mean absolute percentage error
            "sMAPE" : [], #Symmetric mean absolute percentage error
            "MASE" : [], #Mean absolute scaled error
            "MSE" : [], #Mean squared error
            "RMSE" : [], #Root mean squared error
            "RMSSE" : [], #root mean squared scaled error
            "MaxError" : [] #maximum error
        }

        #Dont think is needed:
        #self.params = None #rename; prob better in base class.
        


        #Decompositions: (s.u., gehört eig. nciht hierher)
        # self.decomp = None






    #------------------------------------------------------------------------------------------------
    # Helper functions
    #------------------------------------------------------------------------------------------------
    
    # #Not used anymore
    # def rolling_window(self, train_len: int, test_len: int, start_date: str=None):
    #     #split df into train and test set, by rolling window (same length
    #     # of history 'rolling' over data): in data 11-10 with train_len=3, test_len=2:
    #     # [1, 2, 3][4, 5] 6, 7, 8, 9, 10 
    #     #  1 [2, 3, 4][5, 6] 7, 8, 9, 10 
    #     #  1, 2 [3, 4, 5][6, 7] 8, 9, 10
    #     #  1, 2, 3 [4, 5, 6][7, 8] 9, 10
    #     #  1, 2, 3, 4 [5, 6, 7][8, 9] 10
    #     #  1, 2, 3, 4, 5 [6, 7, 8][9, 10]
    #     if start_date != None:
    #         start_date = pd.to_datetime(start_date)

    #     start_idx = train_len 
    #     end_idx = len(self.df) - test_len + 1 

    #     for split_idx in range(start_idx, end_idx):

    #         #use iloc to get a view, not a copy (like you would get with df[n:m])
    #         train_set = self.df.iloc[split_idx-train_len : split_idx]
    #         test_set = self.df.iloc[split_idx : split_idx+test_len]

            
    #         # print(f"\ntrain set ({len(train_set)}):\n{train_set}")
    #         # print(f"\ntest_set ({len(test_set)}):\n{test_set}")

    #         yield train_set, test_set

    # #Not used anymore
    # def expanding_window(self, train_percent: float, test_len: int):
    #     #TODO: move to top of file of respective class file
    #     from sklearn.model_selection import TimeSeriesSplit

    #     #create expanding window for cross validation.
    #     # pass percentage for split in data (0-1), which is the percentage of initially kept data for training, 
    #     # as well as test_len, which is the number of rows to look ahead.

    #     #index where to split/start the expanding window
    #     start_idx = self.get_split_index_by_prct(len(df), train_percent)
    #     end_idx = len(self.df) - test_len + 1

    #     res = []

    #     for split_idx in range(start_idx, end_idx):
    #         train_set = self.df.iloc[:split_idx]
    #         test_set = self.df.iloc[split_idx:split_idx+test_len]

    #         # print(f"\nsplit index: {split_idx}")
    #         # print(f"train set ({len(train_set)}):\n{train_set}")
    #         # print(f"\ntest_set ({len(test_set)}):\n{test_set}\n")

    #         # yield train_set, test_set
    #         res.append([train_set, test_set])
    #     return res

    # # Not used anymore:
    # def validate_rolling_window(self, data, w, sliding=False):
    #     """
    #     w : window size in days
    #     sliding : If True window slides by one day, if false window slides by window size. 
    #     """
    #     if self.test_data == None or self.train_data == None:
    #         raise ValueError("test_data and train_data cant be None. Use split_by_percentage method,"
    #         "to assign data to test and train set.")

    #     #Habe das hier implementiert, damit man nicht den split auch noch 
    #     # als argument beim fct call angeben muss. Wenn ich das hier einbaue, müsste
    #     # ich noch return zu split_by_percentage machen
    #     train = self.train_data
    #     test = self.test_data

    #     #TODO: this has to be done using for t in range(), to
    #     # account for step size of window (if sliding=True, i want
    #     # to jump 3 days, if w=3)
    #     for t in test[:len(test) - w + 1]:
    #         print(test, len(test) - w + 1)

        

    #     #run model against test data set wtih rollling window
    #     pass

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


                # Old version with for loop
                # for step in num_of_iterations:
                #     step_values = (train_start, train_end, test_start, test_end)
                #     step_test_start = train_start
                #     step_
                #     train_test_indices
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
        Add stepwise_forecasts: dictionary containing number of keys in the length of test_len, 
        with values as pandas Series of forecasted values for respective days to look ahead. 
        E.g. if test_len (in rolling/expanding window) 
        is 7 days, keys "1" to "7" are added, containing the predicted value for predictions 
        as far into the future as the key.
        Used for plotting and comparing, how well the prediction works into the future.

        Returns nothing, sets self.stepwise_forecasts as dataframe with days ahead as 
        columns ("Days ahead: [val]") and prediction_mean as value/columns. Datetime index
        """

        #not run if single split validation (has no test_len):
        if self.validation_config["test_len"]:
            #Add 'step' (day x) ahead
            for step in range(1, self.validation_config["test_len"] + 1):
                step_forecasts = pd.Series()
                for pred in self.predictions:
                    step_forecasts = pd.concat([step_forecasts, pred.predicted_mean.iloc[[step-1]]])
                
                step_forecasts.sort_index(inplace=True)
                if step == 1:
                    self.stepwise_forecasts = step_forecasts.to_frame()
                else:
                    self.stepwise_forecasts = pd.concat([self.stepwise_forecasts, step_forecasts], axis=1)
        
                #Rename last col to string of days to look ahead:
                self.stepwise_forecasts.columns = [*self.stepwise_forecasts.columns[:-1], f"Days ahead: {step}"]
                




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


    def plot_stepwise(self, days=None):
        """Plot the stepwise predictions, 
        i.e. plot e.g. one-day-ahead prediction against test set, two-day-ahead, etc.
        """
        plot_start = self.stepwise_forecasts.index.min() - pd.DateOffset(60) #first element of first key
        plot_end = self.stepwise_forecasts.index.max()#self.stepwise_forecasts.keys()[-1][-1] #last element of last key


        colors = iter(cm.rainbow(np.linspace(1, 0.6, len(self.stepwise_forecasts.columns))))

        #original_data = 
        plt.figure(figsize=(14,7))

        plt.plot(self.data[plot_start:plot_end]["count"], label="original data")
        for col in self.stepwise_forecasts.columns:
            plt.plot(self.stepwise_forecasts[col], label=col, color=next(colors))
        plt.legend()
        plt.show()



    def get_stepwise_errors(self, error: str="all"):
        """Calculate stepwise errors for stepwise forecasts. That means that for x days ahead,
        for all predicted values, the supplied error is calculated.
        Supporte

        Args:
            error (str): Abbreviation of one of the supported error metrics. Currently: MAE.
            Defaults to 'all', which gets error measures for all supported error metrics.
            Coming: MAPE; MedAE, MaxError, RMSE, MASE

        Raises:
            ValueError: If wrong 'error' is supplied
        """
        possible_error_metrics = ["MAE", "MAPE", "MedAE", "RMSE", "MaxError", "MASE"] 
        if error not in possible_error_metrics and error != "all":
            raise ValueError(f"{error} is not in the possible list of errors. Please only use {possible_error_metrics}")
        elif error == "all":
            error = possible_error_metrics

        #initialize empty df with structure like stepwise_forecasts (cols, indices, no content)
        stepwise_metric = pd.DataFrame().reindex_like(self.stepwise_forecasts)
        stepwise_metric = stepwise_metric.merge(self.data["count"], left_index=True, right_index=True) #add original 'count' as ytrue

        #get specific error measure result if is a string (single error) or for all errors (if is list)
        if type(error) == str:
            error = [error]

        # for err in error:
        #     if err == "MAE":
        #         results = self.get_mae(stepwise_metric)
        #         self.forecast_errors[err] = results
        #         print("test")

        for col in self.stepwise_forecasts.columns:
            min_date = self.stepwise_forecasts[col].first_valid_index()
            max_date = self.stepwise_forecasts[col].last_valid_index()
            y_pred = self.stepwise_forecasts.loc[min_date:max_date, col]
            y_true = self.data.loc[min_date:max_date, "count"]
        
            self.forecast_errors["ME"].append(mean(y_pred=self.stepwise_forecasts.loc[min_date:max_date, col], y_true=self.data.loc[min_date:max_date, "count"]))
            self.forecast_errors["MAE"].append(metrics.mean_absolute_error(y_pred=self.stepwise_forecasts.loc[min_date:max_date, col], y_true=self.data.loc[min_date:max_date, "count"]))
            self.forecast_errors["MedAE"].append(metrics.median_absolute_error(y_pred=self.stepwise_forecasts.loc[min_date:max_date, col], y_true=self.data.loc[min_date:max_date, "count"]))
            self.forecast_errors["MAPE"].append(metrics.mean_absolute_percentage_error(y_pred=self.stepwise_forecasts.loc[min_date:max_date, col], y_true=self.data.loc[min_date:max_date, "count"]))
            self.forecast_errors["MSE"].append(metrics.mean_squared_error(y_pred=self.stepwise_forecasts.loc[min_date:max_date, col], y_true=self.data.loc[min_date:max_date, "count"]))
            self.forecast_errors["RMSE"].append(metrics.root_mean_squared_error(y_pred=self.stepwise_forecasts.loc[min_date:max_date, col], y_true=self.data.loc[min_date:max_date, "count"]))
            self.forecast_errors["MaxError"].append(metrics.max_error(y_pred=self.stepwise_forecasts.loc[min_date:max_date, col], y_true=self.data.loc[min_date:max_date, "count"]))


        # Temp. removed, because debugger cant step into 'case'
        # for err in error:
        #     match error:
        #         case "MAE":
        #             #results = metrics.mean_absolute_error(self.data["count"], self.stepwise_forecasts)
        #             results = get_mae(stepwise_metric)
        #             self.forecast_errors[error] = results
        #         case "MAPE":
        #             pass
        #         case "MedAE":
        #             pass
        #         case "RMSE":
        #             pass
        #         case "MaxError":
        #             pass
        #         case "MASE":
        #             pass
        
        #remove y_true from df



    def get_mae(self, stepwise_metric: pd.DataFrame):
        self.stepwise_forecasts

        for col in self.stepwise_forecasts.columns:
            print(f"Calculating for {col}")
            min_date = self.stepwise_forecasts[col].first_valid_index()
            max_date = self.stepwise_forecasts[col].last_valid_index()
            y_pred = self.stepwise_forecasts.loc[min_date:max_date, col]
            y_true = self.data.loc[min_date:max_date, "count"]
            result = metrics.mean_absolute_error(y_pred=self.stepwise_forecasts.loc[min_date:max_date, col], y_true=self.data.loc[min_date:max_date, "count"])

            stepwise_metric[col] = result
            print(result)

        return stepwise_metric
        



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


    



#--------------------------------------------------------------------
# Individual Models:
#--------------------------------------------------------------------
    


# ARIMA
class ModelArima(Model):


    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)
        self.p = None
        self.q = None
        self.d = None

    #------------------------------------------------------------------------------------------------
    # Setters
    #------------------------------------------------------------------------------------------------
    
    def set_parameters(self, p: int=1, d: int=1, q: int=1):
        self.p = p
        self.d = d
        self.q = q

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
            self.models.append(ARIMA(series[train_set[0] : train_set[1]], order=(self.p, self.d, self.q))) #(1, 1, 1)))


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
            self.predictions.append(fit.get_forecast(steps=days))




# SARIMA
class ModelSarima(Model):


    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)
        self.p = None
        self.q = None
        self.d = None

    #------------------------------------------------------------------------------------------------
    # Setters
    #------------------------------------------------------------------------------------------------
    
    def set_parameters(self, p: int=1, d: int=1, q: int=1):
        self.p = p
        self.d = d
        self.q = q

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
            self.models.append(ARIMA(series[train_set[0] : train_set[1]], order=(self.p, self.d, self.q))) #(1, 1, 1)))


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





# LSTM

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
