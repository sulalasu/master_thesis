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


transfusion_status_map = {
    "ToD" : {
        "Transfundiert" : "transfused",
        "Entsorgt" : "discarded"
    },
    "ToD_N" : {
        "VER" : "transfused", #'Verabreicht'
        "AUS" : "discarded" # QUESTION: Does 'AUS' (='Ausgegeben') mean used or discarded or neither?
    },
    "ToD_O" : {
        "ABG" : "expired"
    }
}


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
