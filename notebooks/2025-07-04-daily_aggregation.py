#%% Read Data
import pandas as pd


#%%
df_raw = pd.read_csv("../data/02_intermediate/intermediate_output.csv")
df_raw['date'] = pd.to_datetime(df_raw['date'])

print(df_raw.info())
print(df_raw.head())


# %%
def aggregate_categorical_cols(df, cols_to_sum: dict):
    #Split columns with n categorical values into n columns.
    # Aggregate daily sum of occurences to each column.
    # cols_to_sum are all cols with categorical values which i want to sum.  
    
    results = []

    for col in cols_to_sum:
        res = pd.crosstab(df['date'], df[col])
        res = res.add_prefix(col + "_")
        res.columns = res.columns.str.replace(" ", "_") #replace spaces

        results.append(res)


    wide_df = results[0].join([res for res in results[1:]])

    #Add daily total amounts:
    #test_df = df.set_index('date')['use'].resample('D').count()
    total_df = df.groupby('date').size().reset_index(name='count')
    #Add to wide_df
    wide_df = wide_df.join(total_df.set_index('date'))
    
    return wide_df

#%%
# cat_cols = list(df_raw.columns)
# remove = ["Unnamed: 0", "EC_ID_O_hash", "EC_ID_I_hash", "date"]
# for rem in remove:
#     cat_cols.remove(rem)

#%%
#cols which i dont want to add the values (or dont need at all anymore):
cols_to_remove = ["Unnamed: 0", "EC_ID_O_hash", "EC_ID_I_hash"]
df_proc = df_raw.drop(columns=cols_to_remove, axis=1)

cat_cols = list(df_proc.columns)
cat_cols.remove('date')

df_proc = aggregate_categorical_cols(df_proc, cat_cols)


# %%
df_proc






# %%
df_test = df_raw.copy()

#%%
res = pd.crosstab(df_test['date'], df_test['EC_BG'])
print(res)
# %%
