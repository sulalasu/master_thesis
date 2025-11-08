# THis file is just the cleaned version of the one from 2025-10-19.
# I keep the (non)-intersection part and the final part.
#%%
# Load df (data cleaned) & ward map i got from alex.

import pandas as pd
import numpy as np

df_clean = pd.read_csv("../data/02_cleaned/output_cleaned.csv", index_col="date")
df = df_clean

ward_map = pd.read_csv("../data/00_external_data/wards_mapping.csv", sep="\t")
ward_map["Kostenstelle"] = ward_map["Kostenstelle"].str.strip()


ward_map_list_Kostenstellen = list(ward_map["ID_Kostenstelle"].unique())
unique_wards = df_clean["PAT_WARD"].unique()


#----------------------------------------------------------------------------
#This part is only to check data:
# %%
# Get unique Letter-combinations (i guess top-wards) from all wards:
import re
pattern = re.compile(r'\d+(.{2})\d+')  # number + letters + number
PAT_WARD_grep_results = set()  # only unique strings

for s in unique_wards:
    match = pattern.search(str(s))
    if match:
        PAT_WARD_grep_results.add(match.group(1))  # =letters

#Show unique PAT_WARDS from original df & resulting grep matches
print(unique_wards)
print(len(unique_wards))
print(PAT_WARD_grep_results)
print(len(PAT_WARD_grep_results))


#Show (non-)intersections between parts that are there (from df_cleaned: PAT_WARD_grep_results)
# AND parts that should be there (are from ward map i got from alex)
in_a_not_b = list(set(PAT_WARD_grep_results)-set(ward_map_list_Kostenstellen))
in_b_not_a = list(set(ward_map_list_Kostenstellen)-set(PAT_WARD_grep_results))
non_intersecting = list(set(ward_map_list_Kostenstellen)^set(PAT_WARD_grep_results))
intersecting = list(set(ward_map_list_Kostenstellen)&set(PAT_WARD_grep_results))



#%%
# --Extract 2 letter code--
# --> in the final version i doont need that anymore. see text below
# (i decided to only match PAT_WARD (long codes) to the long codes that are actually
# present in the ward map file)

# df["code"] = df["PAT_WARD"].str.extract(r'^\d{3}([a-zA-Z][a-zA-Z0-9]{1})')
# less robust:
# df["code_position"] = df["PAT_WARD"].str[3:5]



#-------------------------------------------------------------------
#%%
# Working part to get TOP 5 + OTHERS

#Explanation for the steps
#1. In main df, 
# 1.1 map ID_Kostenstelle (short 2letter code) onto PAT_WARD with Kostenstelle (long code)
# 1.2 fill missing/non-matching with 'Other'
#2. Count by ID_Kostenstelle and Rank in main df
#3. Assign 'Other' to ID_Kostenstelle bigger than rank 5
# --> mapping df with ID_Kostenstelle, amount, rank and Categorization to top5 + 'other' ('pat_wards_ranking_df')
#4. Join categorization top5+Other from 'pat_wards_ranking_df' onto main df column ID_Kostenstelle
# (So first 2-letter code gets mapped onto long code, missing filled with other, then the ranking applied onto 2-letter code in main df)
# Explanation/Reasoning: This way, some information might be lost, because sometimes a valid 2-letter code could be extracted from the long code in main df,
# but i only want to map existing long codes from Alex data onto valid long codes in main df. 

# Alternative (commented out code below / 'Different approach'): 
#I could get the two-letter code fromm PAT_WARD df with grep like this:
#df["code"] = df["PAT_WARD"].str.extract(r'^\d{3}([a-zA-Z][a-zA-Z0-9]{1})')
# But ill only join in the end the PAT_WARDS to Kostenstelle(n) that actually appear in ward_map.
# (Also i didnt check if theres actually PAT_WARDS with existing codes that dont appear in ward_map/Kostenstelle)

#Load data
df_clean = pd.read_csv("../data/02_cleaned/output_cleaned.csv", index_col="date")
df = df_clean.copy()
ward_map = pd.read_csv("../data/00_external_data/wards_mapping.csv", sep="\t")
ward_map["Kostenstelle"] = ward_map["Kostenstelle"].str.strip() #remove whitespaces around strings
ward_map = pd.concat(ward_map, pd.DataFrame(["Other", "Other"]))

#%%
# Merge short code (2-letter code) onto df, fill missing values with NA
# I already need ID_Kostenstelle (the umbreella/aggregate ward) here, so i can assign the counts
# In the result, both PAT_WARD and Kostenstelle (=long code/detailled ward) are present, but Kostenstelle contains Missing value sometimes, when no match availabel with PAT_WARD (makes sense, since not in ward_map).
# and ID_Kostenstelle in that case automatically gets asigned 'Other'
df = (
    pd.merge(
        left=df.reset_index(), #so date still persists after merge
        right=ward_map[["ID_Kostenstelle", "Kostenstelle"]], 
        left_on="PAT_WARD", 
        right_on="Kostenstelle", 
        how="left")
    #.drop("Kostenstelle", axis=1)
    .set_index("date")
    .fillna({"ID_Kostenstelle":"Other"}) 
)

#%%
# Helper variable: Create count+ranking and Map of ID_Kostenstelle/Kostenstelle/top_wards
# (top_wards = top 5 ranking wards keep their names, rest get assigned as 'Other')
pat_wards_ranking_df = (
    df
    .groupby("ID_Kostenstelle")
    .size().to_frame("amount")
    .sort_values(by="amount", ascending=False).reset_index()
    .assign(rank = lambda x: x["amount"].where(x["ID_Kostenstelle"] != "Other").rank(ascending=False)) #Exclude 'Other' from ranking.  Other == where no (existing) ID_Kostenstelle was matching.
    .assign(top_wards = lambda x: np.where(
        (x["rank"] <= 5) | (x["rank"].isna()), x["ID_Kostenstelle"], "Other"))
    #.assign(top_wards = lambda x: np.where(x["ID_Kostenstelle"] == "Other", "Other"))
)


# merge directly onto ID_Kostenstelle (2-letter code)
df_test_2 = (
    pd.merge(
        left=df.drop("PAT_WARD", axis=1),
        right=pat_wards_ranking_df.drop(["amount", "rank"], axis=1),
        left_on="ID_Kostenstelle",
        right_on="ID_Kostenstelle",
        how="left"
    )
    #remove unncecessary cols
    .drop(["PAT_WARD", "ID_Kostenstelle", "Kostenstelle"], axis=1)
    .rename(columns={"top_wards":"ward"})
)

# Different approach:
# Mapping from detailed Kostenstelle to top5+other categorization.1
#create mapping between 2-letter ID_Kostenstelle+Kostenstelle to top 5 wards or Other. i.e. only detailed wards (Kostenstelle) keep the short 2-letter-code (umbrella Kostenstelle), that are
# in the top5 of transfusion. Rest gets labeled as 'Other'
# top5_Kostenstelle_map = (
#     pd.merge(
#         left=ward_map.drop(["Station Kurz", "Station lang"], axis=1),
#         right=pat_wards_ranking_df.drop(["rank", "amount"], axis=1),
#         left_on="ID_Kostenstelle",
#         right_on="ID_Kostenstelle",
#         how="left"
#     )
# )

#This maps the top5+other categorization onto detailed Kostenstelle
# #Test if now all PAT_WARDS are  mapped:
# df_only_top5_others = (
#     pd.merge(
#         left=df_clean,
#         right=top5_Kostenstelle_map,
#         left_on="PAT_WARD",
#         right_on="Kostenstelle",
#         how="left"
#     )
#     .fillna({
#         "ID_Kostenstelle":"Other",
#         "Kostenstelle":"Other",
#         "top_wards":"Other"
#     })
# )

# #or alternatively merge directly onto df:
# df_test = (
#     pd.merge(
#         left=df.drop("PAT_WARD", axis=1),
#         right=top5_Kostenstelle_map.drop("Kostenstelle", axis=1),
#         left_on="ID_Kostenstelle",
#         right_on="ID_Kostenstelle",
#         how="left"
#     )
# )


# %%
