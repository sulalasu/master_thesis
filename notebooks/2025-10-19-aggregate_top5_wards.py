#%%
import pandas as pd
import numpy as np

df_clean = pd.read_csv("../data/02_cleaned/output_cleaned.csv", index_col="date")
# df_transformed = pd.read_csv("../data/03_transformed/output_transformed.csv", index_col="date")
ward_map = pd.read_csv("../data/00_external_data/wards_mapping.csv", sep="\t")
#%%

ward_map["Kostenstelle"] = ward_map["Kostenstelle"].str.strip()

df = df_clean
# df = pd.merge(
#     left=df_clean.reset_index(), 
#     right=ward_map, 
#     left_on="PAT_WARD", right_on="Kostenstelle", how="left"
#     ).set_index("date")

ward_map_list_Kostenstellen = list(ward_map["ID_Kostenstelle"].unique())
len(ward_map_list_Kostenstellen)
# %%
# Get unique Letter-combinations (i guess top-wards) from all wards:
import re
unique_wards = df_clean["PAT_WARD"].unique()

# pattern = re.compile(r'\d+([A-Za-z]+)\d+')  # number + letters + number
pattern = re.compile(r'\d+(.{2})\d+')  # number + letters + number

PAT_WARD_grep_results = set()  # only unique strings

for s in unique_wards:
    match = pattern.search(str(s))
    if match:
        PAT_WARD_grep_results.add(match.group(1))  # =letters

print(unique_wards)
print(len(unique_wards))
print(PAT_WARD_grep_results)
print(len(PAT_WARD_grep_results))

# %%
in_a_not_b = list(set(PAT_WARD_grep_results)-set(ward_map_list_Kostenstellen))
in_b_not_a = list(set(ward_map_list_Kostenstellen)-set(PAT_WARD_grep_results))
non_intersecting = list(set(ward_map_list_Kostenstellen)^set(PAT_WARD_grep_results))
intersecting = list(set(ward_map_list_Kostenstellen)&set(PAT_WARD_grep_results))

df["code"] = df["PAT_WARD"].str[3:5]
df["code"].unique()
len(df["code"].unique())
df


#%%
#Extract 2 letter code
df = df_clean
df["code"] = df["PAT_WARD"].str.extract(r'^\d{3}([a-zA-Z][a-zA-Z0-9]{1})')
#less robust:
# df["code_position"] = df["PAT_WARD"].str[3:5]

#On this code, join ID_Kostenstelle
# df = pd.merge(
#     left=df.reset_index(), 
#     right=ward_map.drop("Station Kurz", axis=1), 
#     left_on="code", right_on="ID_Kostenstelle", how="left")\
#     .set_index("date") \
#     .rename(columns={"ID_Kostenstelle":"ward_code"}) \
#     .fillna({"ward_code":"Other"}, inplace=True) #fill resultung 'Missing value' with 'Other'
ward_map_unique = pd.DataFrame({"ward_code" : ward_map["ID_Kostenstelle"].unique()})
#This step 'removes' values, that are not in our map! (but also, not all map values are present in data, see above non_intersecting/intersecting)
df = pd.merge(
    left=df.reset_index(), 
    right=ward_map_unique, 
    left_on="code", right_on="ward_code", how="left")\
    .set_index("date") 
#df.rename(columns={"ID_Kostenstelle":"ward_code"}, inplace=True)
df.fillna({"ward_code":"Other"}, inplace=True) 
# df = df.rename(columns={"ID_Kostenstelle":"ward_code"}, inplace=True)

df["code"].unique()
len(df["code"].unique())
df["ward_code"].unique()
len(df["ward_code"].unique())
#%%
#Get top 5 (that are not 'Other') and make rest to 'Other'
top5_wards = df[df["ward_code"] != "Other"]\
    .groupby("ward_code")\
    .size().sort_values(ascending=False)\
    .head(5).reset_index()["ward_code"]
# %%
# Add to ward_map as new column

# ward_counts = (
#     df['ward_code']
#     .value_counts()
#     .reset_index()
#     .rename()
# )

# Use ward map with df in aggregate_cols() (in transform.py)
# while ignoring other columns: code, ward_code, PAT_WARD

#%%
#TODO:
# i think best is, to make a dataframe with all wards (i got from alex) with following columns:
# code, count, rank, label (=in this column assign: top5+other)
# but i have to be carefull because all NA are already 'Other', so either leave them as NA or 
# work around that (probably filter out 'Other', then rank, and then fill back in, but i think easier
# to leave them as NA for now.)
import numpy as np
ward_counts = (
    df[df["ward_code"] != "Other"]
    .groupby("ward_code")
    .size().to_frame("amount")
    .sort_values(by="amount", ascending=False)
    .assign(rank = lambda x: x["amount"].rank(ascending=False))
    # .reset_index()
    .assign(top_wards = lambda x: np.where(x["rank"] <= 5, x.index, "Other")) )
    #old way
    # .reset_index()
    # .assign(rank = df.index.astype(int) + 1)#.reset_index().rename(columns={"index":"index_col"})\
    # .assign(index_col=lambda x: x["index_col"] + 1)
)
#["ward_code"]
    # .assign(rank=df[df["ward_code"] != "Other"])

# then before aggregation (use 'code' instead of 'PAT_WARD' for that), join 'label' colum using 'code' column to main df,
# so only 'Other' and top5 remain in 'code' column, remove PAT_WARD (or rename code to pat_ward)





#-------------------------------------------------------------------
#%%
#All combined:
#I could get the two-letter code fromm PAT_WARD df with grep like this:
#df["code"] = df["PAT_WARD"].str.extract(r'^\d{3}([a-zA-Z][a-zA-Z0-9]{1})')
# But ill only join in the end the PAT_WARDS to Kostenstelle(n) that actually appear in ward_map.
# (Also i didnt check if theres actually PAT_WARDS with existing codes that dont appear in ward_map/Kostenstelle)

#Load data
df_clean = pd.read_csv("../data/02_cleaned/output_cleaned.csv", index_col="date")
df = df_clean.copy()
ward_map = pd.read_csv("../data/00_external_data/wards_mapping.csv", sep="\t")
ward_map["Kostenstelle"] = ward_map["Kostenstelle"].str.strip() #whitespaces around strings

#%%
# Merge short code (2-letter code) onto df
df = (
    pd.merge(
        left=df.reset_index(), #so date still persists after merge
        # right=ward_map_unique, 
        right=ward_map[["ID_Kostenstelle", "Kostenstelle"]], 
        left_on="PAT_WARD", right_on="Kostenstelle", 
        how="left")
    .set_index("date")
    .fillna({"ID_Kostenstelle":"Other"}) 
)

#%%
# Map of ID_Kostenstelle/Kostenstelle/top_wards ()
top5_df = (
    df
    .groupby("ID_Kostenstelle")
    .size().to_frame("amount")
    .sort_values(by="amount", ascending=False).reset_index()
    .assign(rank = lambda x: x["amount"].where(x["ID_Kostenstelle"] != "Other").rank(ascending=False)) #Exclude 'Other' from ranking.  Other == where no (existing) ID_Kostenstelle was matching.
    .assign(top_wards = lambda x: np.where(x["rank"] <= 5, x["ID_Kostenstelle"], "Other"))
)

#now map individual (long) 'Kostenstelle' to Top 5 + Other
top5_Kostenstelle_map = (
    pd.merge(
        left=ward_map.drop(["Station Kurz", "Station lang"], axis=1),
        right=df_top5.drop(["rank", "amount"], axis=1),
        left_on="ID_Kostenstelle",
        right_on="ID_Kostenstelle",
        how="left"
    )
)

#Test if now all PAT_WARDS are  mapped:
df_2 = (
    pd.merge(
        left=df_clean,
        right=top5_Kostenstelle_map,
        left_on="PAT_WARD",
        right_on="Kostenstelle",
        how="left"
    )
    .fillna({
        "ID_Kostenstelle":"Other",
        "Kostenstelle":"Other",
        "top_wards":"Other"
    })
)
