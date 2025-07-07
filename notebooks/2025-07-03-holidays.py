#%%
from datetime import datetime as dt
import pandas as pd
import holidays
import numpy as np

# %%
#sample data
date_range = pd.date_range(start="01-01-2020", end="31-12-2024", freq='D')
df = pd.DataFrame(date_range, columns=['date'])
df['random_int'] = np.random.randint(1, 100, size=(len(date_range)))

#%%
#assign a var holding all holidays for AT(Wien) in 2020-2024 with german names (unnecessary)
holidays_aut = holidays.country_holidays(country="AT", subdiv="W", years=range(2020,2025), language="de")
#%%
#actual assignment
df["is_workday"] = df["date"].apply(holidays_aut.is_working_day)
print(df.info())
#%%
#stichproben
print(holidays_aut.is_working_day('2025-07-03')) #donnerstag
print(holidays_aut.is_working_day('2025-05-01'))
