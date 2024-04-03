# Economic Data Analysis using Fred API (https://fred.stlouisfed.org/docs/api/fred/)
# Free api key available on the above link

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from fredapi import Fred
pio.renderers.default = "browser"

plt.style.use('fivethirtyeight')
color_pal = plt.rcParams["axes.prop_cycle"].by_key()['color']
pd.set_option('display.max_columns', 500)

fredapi_key = "50afc3e3f77c91b124b8da9ce9259216"


# 1. Create the Fred object
fred = Fred(api_key = fredapi_key)


# 2. Search for economic data
sp_search = fred.search('S&P', order_by='popularity')


# 3. Pull Raw Data and plot
sp500 = fred.get_series(series_id = 'SP500')
sp500.plot(figsize=(10,5), title='S&P 500', lw=2)


# 4. Pull and Join Multiple Data Series
unemp_results = fred.search('unemployment')

unrate = fred.get_series('UNRATE')

# unrate.plot()

unemp_df = fred.search('unemployment rate state', filter=('frequency', 'Monthly'))
unemp_df = unemp_df.query('seasonal_adjustment == "Seasonally Adjusted" and units=="Percent"')
unemp_df = unemp_df.loc[unemp_df['title'].str.contains('Unemployment Rate in')]

all_results = []
for myid in unemp_df.index:
    results = fred.get_series(myid).to_frame(name=myid)
    all_results.append(results)    
        
unemp_results = pd.concat(all_results, axis=1).dropna(axis=1, how='all')
unemp_states = unemp_results.drop(['LASMT261982000000003', 'LASMT391746000000003'], axis=1)
unemp_states.isna().sum(axis=1).plot()
unemp_states = unemp_states.dropna()


# Retrieving names of states using ids
id_to_state = unemp_df['title'].str.replace('Unemployment Rate in ', '').to_dict()

unemp_states.columns = [id_to_state[c] for c in unemp_states.columns]

# plotting the states unemployment rate 
px.line(unemp_states)

## Pull April 2023 unemployment rate per state
ax = unemp_states.loc[unemp_states.index == '2023-04-01'].T\
        .sort_values('2023-04-01')\
        .plot(kind='barh', figsize=(8,15), width=0.7, edgecolor='black',\
              title='Unemployment Rate By State, April 2023')
ax.set_xlabel('% Unemployed')    
ax.legend().remove()
plt.show()

## Pull February 2024 unemployment rate per state
ax = unemp_states.loc[unemp_states.index == '2024-02-01'].T\
        .sort_values('2024-02-01')\
        .plot(kind='barh', figsize=(8,15), width=0.7, edgecolor='black',\
              title='Unemployment Rate By State, February 2024')
ax.set_xlabel('% Unemployed')
ax.legend().remove()
plt.show()


## Pull participation rate
part_df = fred.search('participation rate state', filter=('frequency', 'Monthly'))
part_df = part_df.query('seasonal_adjustment == "Seasonally Adjusted" and units=="Percent"')
part_df = part_df.loc[part_df['title'].str.contains('Labor Force Participation Rate for')]

all_results = []

for myid in part_df.index:
    results = fred.get_series(myid).to_frame(name=myid)
    all_results.append(results)   
    
part_states = pd.concat(all_results, axis=1).dropna(axis=1,how='all')

# Retrieving names of states using ids
part_id_to_state = part_df['title'].str.replace('Labor Force Participation Rate for ', '').to_dict()
part_states.columns = [part_id_to_state[c] for c in part_states.columns]


## Plot Unemployment vs Participation for New York state (as an example)
fig, ax = plt.subplots()
ax2 = ax.twinx()
unemp_states.query('index >=2022 and index < 2025')['New York'].plot(ax=ax, label='Unemployment')
part_states.query('index >=2022 and index < 2025')['New York'].plot(ax=ax2, label='Participation', color=color_pal[1])
ax2.grid(False)
ax.set_title('New York')

## Plot Unemployment vs Participation for all states
# finding those states names that are common in both unemp_states and part_states
import collections
common_state_names = collections.Counter(unemp_states.columns) & collections.Counter(part_states.columns)
final_states_names = list(common_state_names.elements())

# plotting those state names that are common in both dfs
fig, axs = plt.subplots(10, 5, figsize=(30,30), sharex=True)
axs = axs.flatten()

i = 0
for state in final_states_names:
    ax2 = axs[i].twinx()
    unemp_states.query('index >=2022 and index < 2025')[state].plot(ax=axs[i], label='Unemployment')
    part_states.query('index >=2022 and index < 2025')[state].plot(ax=ax2, label='Participation', color=color_pal[1])
    ax2.grid(False)
    axs[i].set_title(state)
    i += 1
    
plt.tight_layout()
plt.show()