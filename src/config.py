#Global Variables and settings
import pandas as pd
from time import time
ENABLE_TIMING = True #If true, print messages. #TODO: add logging module to do that
ENABLE_LOGGING = True


# DATE: Mapping of column names with their respective format
# DONE: change all date formats to iso without time
date_format_map = {
    "T_XL" : {"unit" : "D", "origin" : "1899-12-30"},
    "T_ISO_T" : {"yearfirst" : True}, 
    "T_DE" : {"dayfirst" : True}, 
    "T_US_T" : {"format" : "%m/%d/%y %H:%M"}, #%H for 24h clock, %I for 12h clock
    "T_DE_S" : {"format" : "%d.%m.%y"}, #y for short year
    "T_US" : {"format" : "%m-%d-%y"}, 
    "T_DE_T" : {"format" : "%d.%m.%y %H:%M"},
    "T_ISO" : {"yearfirst" : True}
}

#Columns with info about EC status ('discarded', 'expired' etc.)
transfusion_cols = ["ToD", "ToD_N", "ToD_O"]

#Mapping of transfusion status raw --> processed
#See Mail from 14.7.25
transfusion_status_map = {
        "Transfundiert" : "transfused",
        "VER" : "transfused", #'Verabreicht' = Zum patienten gekommen. Ob tatsächlich verabreicht ist unbekannt
        "Verkauft": "transfused", #verkauft an andere krankenanstalt; wie 'ausgegeben'
        "Ausgegeben": "transfused", #Zum patienten gekommen. Ob tatsächlich verabreicht ist unbekannt


        "AUS" : "discarded", # 'Ausgegeben'
        "Entsorgt" : "discarded",
        
        "ABG" : "expired", # Abgelaufen == expired
        "Abgelaufen": "expired", 
        "expired" : "expired",
        "VRN" : "discarded", #Vernichtet


         #nan : "???" ,
        "BER" : "???", #NOTE: vermutlich 'bereitgestellt' (wie ausgegeben?)
        "END" : "???", #NOTE: mapping?
        "RES" : "???", #NOTE: vermutlich 'reserviert'
        "RET" : "???" #NOTE: retourniert -- wie klassifizieren?
}

#NOTE: it would be better imo to have the cleaned value as key and original values as values,
# so that you dont repeat it so often. But then dict needs to be reversed, for replacement.
# see solution here https://stackoverflow.com/questions/35491223/inverting-a-dictionary-with-list-values (not implemented!)

#Rhesus factor (EC/PAT) mapping:
rhesus_factor_map = {
    "Rh negativ" : "Rh negative",
    "-" : "Rh negative",
    "N" : "Rh negative", #TODO: is N Negative or Nicht bestimmt?
    
    "Rh positiv" : "Rh positive",
    "+" : "Rh positive",

    "nan" : "NB",
    "NBN" : "NB",
    "Rh nicht bestimmb." : "NB",
    "KMT Rh n. bestimmb." : "NB",
    "Sonderfall" : "NB",

    "Rh D weak" : "Rh weak",
    "Rh D var" : "Rh weak",
    "Rh Du" : "Rh weak"
}

#Blood group (EC/PAT) mapping:
blood_group_map = {
    "A" : "A",
    "0" : "0",
    "0.0" : "0",
    "B" : "B",
    "AB" : "AB",

    "NB" : "NB",
    #"NBN" : "NBN",
    "BG nicht bestimmb." : "NB"
}



#EC type 
# Values to keep, all other change to "Other"
# (full bag, split bag etc) -- only two values in relevant amounts
ec_type_keep = ["EKF", "EKFX"]





#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Global functions
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def timer_func(func):
    #todo: add either own logging func+decorators or put it inside here?
    def wrap_func(*args, **kwargs):
        if ENABLE_TIMING:
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
            return result
        else:
            return func(*args, **kwargs)
    return wrap_func
