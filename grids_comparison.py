#%% 
# compare GRID data load


#%% imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
%matplotlib inline


# %% [markdown]
# ## data loading and preparation
# %%

# Data Loading
data_path = "../blob-account-edh/challenge-data/data_parquet/"
df_consumption = pd.read_parquet(data_path + "base_consumption.parquet").set_index('timestamp')
df_pv = pd.read_parquet(data_path + "pv_profiles.parquet").set_index('timestamp')
df_ev = pd.read_parquet(data_path + "ev_profiles.parquet").set_index('timestamp')
df_hp = pd.read_parquet(data_path + "hp_profiles.parquet").set_index('timestamp')
print("Successfully loaded and indexed all profiles data.")

# Data Preparation and Timestamp Mapping
profiles = {"Consumption": df_consumption, "PV": df_pv, "EV": df_ev, "HP": df_hp}
num_periods = len(df_consumption.index)
datetime_index = pd.date_range(start='2050-01-01', periods=num_periods, freq='15min')
for df in profiles.values():
    df.index = datetime_index
print(f"\nCreated new datetime index from {datetime_index.min()} to {datetime_index.max()}")


# %%
# quick  check on nodes names 

sum('HAS' not in name for name in profiles['PV'].columns)

# %% [markdown] 

# ## quick look at profiles data
# %%


profiles.keys()
# %%
for k,v in profiles.items():
    print(v.shape)
# %%
# plot e.g. of node
node_base_name = 'HAS0000006_node' 

for key, value in profiles.items():
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the title of the current plot
    ax.set_title(key)
    for k,v in value.items():
        if node_base_name in k:
            v.plot(ax=ax,label=k, alpha=0.3)
    plt.legend()
# %% [markdown]
## reload grids 
# %%
file_path = "../blob-account-edh/challenge-data/250903_all_stations_anon.csv"
df_full = pd.read_csv(file_path, sep=";")
print("Successfully loaded network data.")

# %%
from src.functions import (
    build_and_simplify_network,
    # find_failures_with_yearly_profile,
    # suggest_grid_reinforcement,
)

# %%

my_station = 'station_1'

assert my_station in df_full.station.unique()

df_one_station = df_full[df_full['station'] == my_station].copy()

G, consumer_props, roots = build_and_simplify_network(df_one_station)


# %%
# check if all nodes in consumer_props is_consumer is True
sum(v['is_consumer'] for v in consumer_props.values()) / len(consumer_props)

print(f"Number of consumers in the grid: {sum(v['is_consumer'] for v in consumer_props.values())}")

# %%
# For every station, find all consumers and compute the average consumption across all consumers for that grid

# TO BE IMPROVED /  FIXED

# station_avg_consumption = {}

# for station in df_full['station'].unique():
#     df_station = df_full[df_full['station'] == station]
#     G, consumer_props, roots = build_and_simplify_network(df_station)
#     # Get consumer node names
#     consumer_nodes = [node for node, props in consumer_props.items() if props['is_consumer']]
#     # Filter columns in the consumption profile that match consumer nodes
#     matching_cols = [col for col in profiles['Consumption'].columns if any(node in col for node in consumer_nodes)]
#     if matching_cols:
#         avg_consumption = profiles['Consumption'][matching_cols].mean(axis=1).mean()
#         station_avg_consumption[station] = avg_consumption
#     else:
#         station_avg_consumption[station] = np.nan  # or 0, if you prefer

# # Print average consumption per station
# for station, avg in station_avg_consumption.items():
#     print(f"Station: {station}, Average Consumption: {avg:.2f}")

# %%
