#TODO: is it necessary to load the modules here?
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels
from statsmodels.tsa.arima.model import ARIMA

class Model:



    def __init__(self, data): #TODO: maybe add configuration?
        self.df = data

        #Train/Test sets:
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        #Decompositions: (s.u., gehört eig. nciht hierher)
        # self.decomp = None

        #Results:
        self.prediction = None

    def split(self, train_size=0.33):
        if train_size >= 1 or train_size <= 0:
            raise Exception(f"Training data ratio: (train_size: {train_size}) must be floats smaller than 1)")
    
        x = 
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.df)
        #return self.X_train, self.X_test, self.Y_train, self.Y_test

    # Gehört eigentlich nichth zum model, passiert ja vorher:
    # def decompose(self, ):
    #     self.decomp = seasonal_decompose(series, model='additive')
    #     print(self.decomp.trend)
    #     print(self.decomp.seasonal)
    #     print(self.decomp.resid)
    #     print(self.decomp.observed)

    def test_class_implementation(self):
        print(self.df.head())

    def test_class_implementation2(self):
        self.split(self.df)

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

    def model_fit(self):
        # fit (train) model on dataset
        pass
    
    def predict(self, time):
        # generate forecast for x time
        # see child class
        pass

    def test_model(self):
        #run model against test data set
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

    def fit(self, series, p, d, q):
        #create model with trainign data + (hyper)parameters
        #params are model parameters
        self.model = ARIMA(series, order=(p,d,q))
        self.model_fit = self.model.fit()
    
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
