#TODO: is it necessary to load the modules here?
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
import statsmodels

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

    def split(self, data, ratio=(0.8, 0.2)):
        if sum(ratio) > 1 or (sum(ratio) >= 0.99 or sum(ratio) <= 1):
            raise Exception("Ratio must be floats that sum to 1 (or 0.99 for thirds)")
    
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.df)
        
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    # Gehört eigentlich nichth zum mode, passiert ja vorher:
    # def decompose(self, ):
    #     self.decomp = seasonal_decompose(series, model='additive')
    #     print(self.decomp.trend)
    #     print(self.decomp.seasonal)
    #     print(self.decomp.resid)
    #     print(self.decomp.observed)

    def make_stationary(self, data):
        #make stationary
        #maybe move to corresping model that needs it?
        # df = xxx
        # return df
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
        self.params = None #rename; prob better in base class.

    def create_model(self, params, days):
        #create model with (hyper)parameters
        #params are model parameters
        # days are days to predict
        pass




# LSTM

class ModelLSTM(Model):

    def __init__(self, data): #TODO: maybe add config, but more sense in base class imo
        super().__init__(data)
        self.model = None
        self.params = None #rename; prob better in base class.

    def create_model(self, params, days):
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
