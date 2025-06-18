#TODO: is it necessary to load the modules here?
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX

class Model:



    def __init__(self, data): #TODO: maybe add configuration?
        self.data = data

        #Train/Test sets:
        self.train_data = None
        self.test_data = None

        #Decompositions: (s.u., gehört eig. nciht hierher)
        # self.decomp = None

        #Model fit
        self.model_fit = None

        #Results:
        self.prediction = None

    def split_by_percentage(self, percent=0.33):
        """percent = percent to assign as test data. should be <0.5"""

        if percent >= 1 or percent <= 0:
            raise Exception(f"Training/test data ratio: test size: {percent}) must be float smaller than 1 (should be >0.5)")
    
        split_index = int(len(self.data)*percent)
        self.train_data = self.data.iloc[: split_index]
        self.test_data = self.data.iloc[split_index:]






    # Gehört eigentlich nichth zum model, passiert ja vorher:
    # def decompose(self, ):
    #     self.decomp = seasonal_decompose(series, model='additive')
    #     print(self.decomp.trend)
    #     print(self.decomp.seasonal)
    #     print(self.decomp.resid)
    #     print(self.decomp.observed)

    def test_class_implementation(self):
        print(self.data.head())


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

    # def model_fit(self):
    #     # fit (train) model on dataset
    #     pass
    
    def predict(self, time):
        # generate forecast for x time
        # see child class
        pass

    def validate_expanding_window(self, data, w):
        """
        w = window size in days
        """
        #split data


        #run model against test data set wtih expanding window
        pass

    def validate_rolling_window(self, data, w):
        """
        w = window size in days
        """
        if self.test_data == None or self.train_data == None:
            raise ValueError("test_data and train_data cant be None. Use split_by_percentage method,"
            "to assign data to test and train set.")

        #Habe das hier implementiert, damit man nicht den split auch noch 
        # als argument beim fct call angeben muss. Wenn ich das hier einbaue, müsste
        # ich noch return zu split_by_percentage machen
        train = self.train_data
        test = self.test_data
        for t in test[:len(test) - w + 1]:
            print(test, len(test) - w + 1)



        #run model against test data set wtih rollling window
        pass


    def evaluate_model(self):
        #run evaluations, to get values like MAE, MAPE etc.
        pass


    #UNCLEAR: add extra plotting here or just use the functions? 


    



#--------------------------------------------------------------------
# Individual Models:
#--------------------------------------------------------------------
    

# SARIMA
class ModelSarima(Model):


    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)
        self.model = None
        self.model_fit = None
        self.params = None #rename; prob better in base class.

    def fit(self, col: str, p, d, q):
        #print(self.df.head())
        # col=string for univariate forecasting column/target
        #create model with trainign data + (hyper)parameters
        #params are model parameters
        print("Series: \n\n\nSeries:")
        print(col)
        print(p, d, q)
        series = self.df[["date", col]]
        print(series)
        self.model = ARIMA(series, order=(p,d,q))
        self.model_fit = self.model.fit()

        # return self.model_fit
    
    def fit_summary(self):
        print(self.model_fit.summary())

    def predict(self, days):
        # generate forecast for x time
        # (see base class)
        pass


# LSTM

class ModelLSTM(Model):

    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)
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
        self.model = None
        self.params = None #rename; prob better in base class.

    def create_model(self, params, days):
        #create model with (hyper)parameters
        #params are model parameters
        # days are days to predict
        pass
