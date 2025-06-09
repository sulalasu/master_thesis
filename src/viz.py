import pandas as pd


#----------------------------------------------------------------------------------------------
# GROUP BY DAY, WEEK, MONTH:
#----------------------------------------------------------------------------------------------
# gehört eigentlich eher zu utils

def get_daily_sum(df):
    #calculate daily values sums
    df_daily = df.groupby("date", as_index=False)["count"].sum()
    daily_average = df_daily["count"].mean()
    return df_daily


def get_daily_mean(df):
    #calculate daily values sums
    df_daily = df.groupby("date", as_index=False)["count"].sum()
    daily_average = df_daily["count"].mean()
    return daily_average

#Weekly:
def get_weekly_mean(df):
    df_weekly = df_daily.groupby(pd.Grouper(key='date', freq='W'))['count'].mean().reset_index()
    df_weekly['week_start'] = df_weekly['date'] - pd.offsets.Week(1) + pd.offsets.Day(1)
    df_weekly = df_weekly.rename({'date': 'week_end'}, axis=1)
    return df_weekly


#Monthly:
def get_monthly_mean(df):
    df_monthly = df_daily.groupby(pd.Grouper(key='date', freq='M'))['count'].mean().reset_index()
    df_monthly['month_start'] = df_monthly['date'] - pd.offsets.Week(1) + pd.offsets.Day(1)
    df_monthly = df_monthly.rename({'date': 'month_end'}, axis=1)
    return df_monthly
