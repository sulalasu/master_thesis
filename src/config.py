#Global Variables and settings
import pandas as pd



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


         #nan : "???" ,
        "BER" : "???", #NOTE: vermutlich 'bereitgestellt' (wie ausgegeben?)
        "END" : "???", #NOTE: mapping?
        "RES" : "???", #NOTE: vermutlich 'reserviert'
        "RET" : "???", #NOTE: retourniert -- wie klassifizieren?
        "VRN" : "discarded" #Vernichtet
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

    "Rh D weak" : "Other",
    "Rh D var" : "Other"        
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
# TEST DATA
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


seoul_name_map = {
    "0": "date",
    "1": "hour",
    "2": "temperature", #C
    "3": "humidity", #%
    "4": "wind_speed", #m/s
    "5": "visibility", #10m
    "6": "dew_point_temp", #C
    "7": "solar_radiation", #Mj/m2
    "8": "rainfall", #mm,
    "9": "snowfall", #cm
    "10": "seasons", 
    "11": "holiday",
    "12": "functioning_day", #[dt: Arbeitstag]
    "13": "count" 
}

# %%
