# code from main.py not needed anymore.
# was after clean_data() cell

#%%
# importlib.reload(config)
pd.unique(df_clean["EC_RH"])
df_clean["PAT_RH"] = df_clean["PAT_RH"].replace(config.rhesus_factor_map)
df_clean["EC_RH"] = df_clean["EC_RH"].replace(config.rhesus_factor_map)
# pd.unique(df_clean["PAT_RH"])
# pd.unique(df_clean["PAT_RH"])
for col in df_clean.columns:
    print(pd.unique(df_clean[col]))
#%%
# Get years with date values in EC_BG_RH:
date_vals = ['11-22-21', '11-06-21', '11-01-21', '11-17-21', '12-01-21', '11-18-21',
 '11-24-21', '11-29-21', '11-21-21', '11-13-21', '11-05-21', '11-19-21'
 '11-11-21', '11-08-21', '11-07-21', '11-16-21', '11-02-21', '11-10-21'
 '11-14-21', '11-12-21', '11-23-21', '11-20-21', '11-03-21', '11-15-21'
 '11-30-21', '11-04-21', '11-27-21', '11-09-21', '11-25-21', '11-28-21'
 '12-02-21', '11-26-21', '12-03-21', '12-04-21', '12-07-21', '12-06-21'
 '12-05-21', '12-08-21', '12-10-21', '12-12-21', '12-09-21', '12-11-21'
 '12-13-21', '12-16-21', '12-17-21', '12-14-21', '12-19-21', '12-15-21'
 '12-18-21', '12-23-21', '12-25-21', '12-20-21', '12-21-21', '12-22-21'
 '12-24-21', '12-29-21', '12-28-21', '12-26-21', '12-27-21', '12-30-21'
 '12-31-21', '01-02-22', '01-01-22', '01-03-22', '01-04-22', '01-05-22'
 '01-07-22', '01-06-22', '01-08-22', '01-09-22', '01-11-22', '01-12-22'
 '01-13-22', '01-10-22', '01-14-22', '01-15-22', '01-16-22', '01-17-22'
 '01-18-22', '01-22-22', '01-19-22', '01-21-22', '01-23-22', '01-20-22'
 '01-26-22', '01-27-22', '01-29-22', '01-25-22', '01-28-22', '01-31-22'
 '01-24-22', '01-30-22', '02-01-22']
df_wrong_vals = df_clean['EC_BG_RH'].isin(date_vals).to_frame()
df_wrong_vals_complete = df_clean[df_clean['EC_BG_RH'].isin(date_vals)]
matching_rows = df_wrong_vals.index[df_wrong_vals['EC_BG_RH'] == True]
affected_dates = matching_rows.to_frame().index.unique().to_frame().reset_index(drop=True)
#unique_date = affected_dates.unique()
#matching_dates.to_
df_wrong_vals_complete.to_csv("./data/00_problems_data/EC_BG_RH_with_date_vals.csv", index=False)
affected_dates.to_csv("./data/00_problems_data/EC_BG_RH_affected_days.csv", index=False)

#%%
# Get days with '80400051' or '80400051.01' in PAT_WARD
unique_ward_vals = df_clean['PAT_WARD'].unique()
ward_vals = []
for val in unique_ward_vals:
    if not str(val).startswith("901"):
        ward_vals.append(val)
#Manually add number vals to list
ward_num_vals = [ '46272.15',  '33421.16', '04972.16', '95630001', '80400051', '80400051.0']
df_wrong_wards = df_clean[df_clean['PAT_WARD'].isin(ward_num_vals)]

df_wrong_wards.to_csv("./data/00_problems_data/PAT_WARD_with_nums.csv", index=False)
df_wrong_wards.index.unique().to_frame().to_csv("./data/00_problems_data/PAT_WARD_affected_days.csv", index=False)
