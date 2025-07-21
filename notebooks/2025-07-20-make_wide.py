#Make df wide, add daily counts/freq of each column values:
# so that e.g. add 4 cols for EC_BG: EC_A, EC_B, EC_AB, EC_0
# with daily counts of frequency:
# Date  EC_A  EC_B    EC_AB   EC_0
# 07-29    2     3        2      9
# 07-30    4     5        0     17
# etc.
# and that for all cols (except ID cols and PAT_WARD, because they have to many unique vals)
#%%
import pandas as pd
import sys
from pathlib import Path
# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

#from src import load


#%%
#Load cleaned data
df_clean = pd.read_csv("../data/02_cleaned/output_cleaned.csv")

df_clean["date"] = pd.to_datetime(df_clean["date"])
df_clean = df_clean.set_index("date")
#df = df.groupby(['datetime','id','col3']).count()

#%%
#Remove unnecessary cols:
remove_cols = ["PAT_WARD", "EC_ID_O_hash", "EC_ID_I_hash"]
df_clean = df_clean.drop(remove_cols, axis="columns")

#Remove rows, where EC_BG is in 12, 11, 01 (faulty raw data)
remove_rows = ["01", "11", "12"]
df_mask = (~df_clean["EC_BG"].isin(remove_rows))
df_clean = df_clean.loc[df_mask]
df_clean.head()


#%%
#Show unique 
for col in df_clean.columns:
    print(f"{col}: {df_clean[col].unique()}")



#ACTUAL WORK:
# %%

# daily_total = df_clean["date"].groupby("date").count()
# daily_total = df_clean.groupby(df_clean.columns.tolist(),as_index=False).size()
# daily_total = df_clean.set_index("date").groupby(df_clean.index.date).count()
# daily_total = df_clean.resample('D').apply({'date':'count'})
#if date is not index
daily_total = df_clean.groupby("date").size()
#if is index:
df_clean2 = df_clean.set_index("date")
daily_total = df_clean2.groupby(df_clean2.index.date).size()
#daily_total = df_clean2.resample("D").size()
daily_total.name = "total"

#%%
df_clean.info()
one_day_df = df_clean[df_clean["date"] == pd.to_datetime("2018-01-01")]
ec_count = pd.crosstab(df_clean["date"], [df_clean["EC_BG"], df_clean["EC_RH"]])
ec_count = pd.crosstab(index=df_clean["date"], columns=[df_clean["EC_BG"], df_clean["EC_RH"], df_clean["use"]])
pat_count = df_clean.groupby("date").count()#size().unstack(fill_value=0)
ec_type = df_clean.set_index("date").groupby("EC_TYPE").resample("D").size().unstack(fill_value=0)
ec_count = df_clean.pivot_table(index="date", columns="EC_BG", values="EC_BG" , aggfunc="size", fill_value=0)

ec_count = df_clean.pivot_table(index="date", columns="EC_RH", values=["EC_BG", "EC_RH"] , aggfunc="size", fill_value=0)

res = []
for col in df_clean.columns:
    if col == "date":
        continue
    pivotted = pd.crosstab([df_clean["date"]], columns=[df_clean[col]])
    pivotted = pivotted.add_prefix(col + "_")
    res.append(pivotted)
final = pd.concat(res, axis=1)
final = final.join(daily_total)

(df_clean.set_index("date")
 .groupby(level="date")
 .apply(lambda g: g.apply(pd.value_counts))
 .unstack(level=1)
 .fillna(0))

#%%
daily_counts = df_clean.groupby('date')['EC_BG'].value_counts().unstack(fill_value=0)
daily_counts.head()
# %%
df_clean = df_clean.reset_index()
encoded_cols = []
for col in ['EC_BG', 'EC_RH', 'EC_TYPE', 'PAT_BG', 'PAT_RH', 'use']:  # replace with your actual columns
    dummies = pd.get_dummies(df_clean[col], prefix=col)
    encoded_cols.append(dummies)

# Step 3: Concatenate one-hot columns with original df
df_encoded = pd.concat([df_clean["date"], *encoded_cols], axis=1)

# Step 4: Group by date and sum to get daily counts
daily_counts = df_encoded.groupby("date").sum()

# Optional: Rename columns if you want shorter names like EC_A instead of blood_group_A
daily_counts.columns = [
    col.replace('blood_group_', 'EC_')
       .replace('rh_', 'EC_')
       .replace('use_', 'use_')  # customize as needed
    for col in daily_counts.columns
]
daily_counts.head()
#df_wide = df_clean
# %%
