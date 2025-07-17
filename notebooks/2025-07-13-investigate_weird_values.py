#%%
import datetime

def excel_date_to_python_date(excel_date):
    # Excel date for Windows (1900 date system)
    excel_epoch = datetime.datetime(1899, 12, 31)
    # Calculate the date by adding the days and fractional days
    python_date = excel_epoch + datetime.timedelta(days=excel_date)
    return python_date

# The value you have
excel_date_values = [13349.16, 04972.16, 33421.16, 46272.15, 80400] #80400051.0

# Convert to a Python datetime object
for date in excel_date_values:
    python_date = excel_date_to_python_date(date)
    print("Converted Date:", python_date)

# Idee: Eingabe als string+number, excel convertiert es als datum, speichert das im Excel format ab.
# Daher, rückwanden zu Datum, aber siehe unten, keine sinnvolen Tage:
# Converted Date: 1936-07-19 03:50:24
# Converted Date: 1913-08-12 03:50:24
# Converted Date: 1991-07-03 03:50:24
# Converted Date: 2026-09-08 03:36:00

# %%

# Check matching EC_BG_RH daten, in denen Datum vorkommt, mit T_US.
# Weil manchmal sind die Datumswerte in EC_BG_RH übereinstimmend mit der Datumsspallte, oft nicht.
# Siehe Mail an Alex vom 13.7.25 oder Word dokument "Erklärung_von_Werten_v02" in thesis/other.


#Run in main.py, below df_raw:
df_raw_temp = df_raw.copy()
print(df_raw_temp[df_raw_temp["EC_BG_RH"] == "12-13-21"])
df_raw_temp["EC_BG_RH_dt"] = pd.to_datetime(df_raw_temp['EC_BG_RH'], format='%m-%d-%y', errors='coerce')
df_raw_temp['T_US_dt'] = pd.to_datetime(df_raw_temp['T_US'], format='%Y-%d-%m', errors='coerce')
df_raw_temp = df_raw_temp.dropna(axis="index", subset=["EC_BG_RH_dt", "T_US_dt"])
df_raw_temp.head()
matching_PAT_BG_RH_and_T_US = df_raw_temp[df_raw_temp["T_US_dt"] == df_raw_temp["EC_BG_RH_dt"]]
matching_days = list(matching_PAT_BG_RH_and_T_US["EC_BG_RH_dt"].unique().astype(str))
