# graphals_v2.py

"""
This script consolidates the logic from the '4_BatteriesAndCurtailment.ipynb' notebook.
It performs a full grid analysis for a specified substation, including:
1.  Baseline analysis to find grid failures.
2.  Allocation of optimally-sized batteries based on the Graphals sizing logic.
3.  Application of PV curtailment as a final measure.
4.  Saving all generated flexibility profiles (battery charge/discharge, curtailment) to Parquet files.
5.  Saving example visualizations as HTML files.

To run: `python graphals_v2.py`
"""

import os
import time
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Assuming the 'src' directory is in the same path
from src.functions import (
    build_and_simplify_network,
    find_failures_with_yearly_profile,
    print_analysis_results,
)

# --- GLOBAL CONSTANTS ---
# Configuration for the analysis
TARGET_STATION = 'station_1'  # Change this to analyze a different station
DATA_PATH = "../blob-account-edh/challenge-data/"
OUTPUT_PROFILES_PATH = "./results/data_parquet/"
OUTPUT_PLOTS_PATH = "./results/plots/"
NOMINAL_VOLTAGE = 400.0

# Constant from graphals_src.py for battery sizing
TIME_AT_PEAK_MAX_s = 6 * 60 * 60  # 6 hours in seconds


# --- CORE LOGIC & HELPER FUNCTIONS ---

class SimpleBattery:
    """A simple battery model that simulates energy storage with constraints."""
    def __init__(self, capacity_kwh, max_power_kw, efficiency=0.9, initial_soc_percent=5.0):
        self.capacity_kwh = float(capacity_kwh)
        self.max_power_kw = float(max_power_kw)
        self.efficiency = float(efficiency)
        self.soc_kwh = self.capacity_kwh * (initial_soc_percent / 100.0)

    def charge(self, power_kw, duration_hours):
        """Charges the battery, returning the actual power used after constraints."""
        power_to_charge = min(power_kw, self.max_power_kw)
        available_capacity_kwh = self.capacity_kwh - self.soc_kwh
        max_energy_in_kwh = available_capacity_kwh / self.efficiency
        max_power_for_duration = max_energy_in_kwh / duration_hours if duration_hours > 0 else 0
        actual_power_in = min(power_to_charge, max_power_for_duration)
        energy_added_kwh = actual_power_in * duration_hours * self.efficiency
        self.soc_kwh += energy_added_kwh
        return actual_power_in

    def discharge(self, power_kw, duration_hours):
        """Discharges the battery, returning the actual power supplied after constraints."""
        power_to_discharge = min(power_kw, self.max_power_kw)
        max_power_for_duration = self.soc_kwh / duration_hours if duration_hours > 0 else 0
        actual_power_out = min(power_to_discharge, max_power_for_duration)
        energy_removed_kwh = actual_power_out * duration_hours
        self.soc_kwh -= energy_removed_kwh
        return actual_power_out

def calculate_graphals_battery_size(peak_production_kw, export_limit_kw):
    """Calculates battery size based on the logic from graphals_src.py."""
    bat_cap_kwh, bat_power_kw = 0, 0
    if peak_production_kw > export_limit_kw:
        energy_to_store_kwh = (peak_production_kw * TIME_AT_PEAK_MAX_s) / 3600.0
        bat_cap_kwh = energy_to_store_kwh
        bat_power_kw = bat_cap_kwh
    return (bat_power_kw, bat_cap_kwh)

def create_battery_schedule(net_load_profile, battery, target_max_export_kw=0.0):
    """Creates a battery charge/discharge schedule based on the net load."""
    duration_hours = 0.25
    battery_charge_kw, battery_discharge_kw, soc_kwh_history = [], [], []
    for load_kw in net_load_profile:
        charge_for_step, discharge_for_step = 0, 0
        if load_kw < -target_max_export_kw:
            power_to_absorb = abs(load_kw) - target_max_export_kw
            charge_for_step = battery.charge(power_to_absorb, duration_hours)
        elif load_kw > 0:
            power_to_supply = load_kw
            discharge_for_step = battery.discharge(power_to_supply, duration_hours)
        
        battery_charge_kw.append(charge_for_step)
        battery_discharge_kw.append(discharge_for_step)
        soc_kwh_history.append(battery.soc_kwh)

    s_charge = pd.Series(battery_charge_kw, index=net_load_profile.index)
    s_discharge = pd.Series(battery_discharge_kw, index=net_load_profile.index)
    s_soc_kwh = pd.Series(soc_kwh_history, index=net_load_profile.index)
    return s_charge, s_discharge, s_soc_kwh

def create_dynamic_export_limits(graph, link_failures_list, consumer_props, root_node_ids, default_limit_kw=20.0, strict_limit_kw=10.0):
    """Creates customer-specific export limits based on network topology and failures."""
    print("--- Creating dynamic, location-based export limits... ---")
    customer_export_limits = {customer: default_limit_kw for customer in consumer_props.keys()}
    constrained_customers = set()
    
    if not link_failures_list:
        print("No link failures found, all customers receive the default export limit.")
        return customer_export_limits

    super_root_id = 'super_root_node'
    temp_graph = graph.copy()
    temp_graph.add_node(super_root_id)
    for root_id in root_node_ids:
        if root_id in temp_graph:
            temp_graph.add_edge(super_root_id, root_id)

    failures_df = pd.DataFrame(link_failures_list)
    for _, row in failures_df.iterrows():
        link_start, link_end = row['link']
        if not temp_graph.has_node(link_start) or not temp_graph.has_node(link_end):
            continue
        try:
            dist_start = nx.shortest_path_length(temp_graph, source=super_root_id, target=link_start)
            dist_end = nx.shortest_path_length(temp_graph, source=super_root_id, target=link_end)
            downstream_node = link_end if dist_end > dist_start else link_start
            subgraph_nodes = nx.dfs_tree(temp_graph, source=downstream_node).nodes()
            for node in subgraph_nodes:
                if node in consumer_props:
                    constrained_customers.add(node)
        except nx.NetworkXNoPath:
            continue
            
    for customer in constrained_customers:
        customer_export_limits[customer] = strict_limit_kw
        
    print(f"Identified {len(constrained_customers)} customers on constrained lines requiring strict limits ({strict_limit_kw} kW).")
    return customer_export_limits

def apply_curtailment(net_load_after_battery_df, customer_export_limits):
    """Applies PV curtailment to cap exports at the specified limits."""
    print("\n--- Applying Final PV Curtailment as a Safety Measure ---")
    curtailed_net_load = net_load_after_battery_df.copy()
    duration_hours = 0.25
    curtailed_energy_kwh = pd.DataFrame(0.0, index=net_load_after_battery_df.index, columns=net_load_after_battery_df.columns)

    for customer_id, limit_kw in customer_export_limits.items():
        if customer_id not in curtailed_net_load.columns: continue
        export_limit = -abs(limit_kw)
        customer_profile = curtailed_net_load[customer_id]
        violation_mask = customer_profile < export_limit
        
        if violation_mask.any():
            curtailed_power = export_limit - customer_profile[violation_mask]
            curtailed_energy_kwh.loc[violation_mask, customer_id] = curtailed_power * duration_hours
            curtailed_net_load.loc[violation_mask, customer_id] = export_limit
            
    total_curtailed = curtailed_energy_kwh.sum().sum()
    print(f"Curtailment applied. Total energy curtailed: {total_curtailed:,.2f} kWh/year.")
    return curtailed_net_load, curtailed_energy_kwh

def update_and_save_parquet(new_data_df, file_path, customers_to_update):
    """Saves or updates a Parquet file with new profile data."""
    relevant_new_data = new_data_df[new_data_df.columns.intersection(customers_to_update)].copy()
    if relevant_new_data.empty:
        print(f"  No new data for specified customers in {os.path.basename(file_path)}. Skipping.")
        return

    if os.path.exists(file_path):
        print(f"File '{os.path.basename(file_path)}' exists. Loading and updating...")
        existing_df = pd.read_parquet(file_path)
        # Drop old columns for customers being updated and merge new ones
        cols_to_drop = existing_df.columns.intersection(relevant_new_data.columns)
        final_df = existing_df.drop(columns=cols_to_drop).join(relevant_new_data, how='outer')
    else:
        print(f"File '{os.path.basename(file_path)}' does not exist. Creating new file...")
        final_df = relevant_new_data
    
    final_df.to_parquet(file_path, index=True)
    print(f"Successfully saved data to '{os.path.basename(file_path)}'")
    
def save_customer_battery_plot(customer_id, df_net_load, df_battery_in, df_battery_out, df_battery_soc_kwh, customer_export_limits, file_path):
    """Creates and saves a plot showing battery operation for one customer."""
    original_net_load = df_net_load[customer_id]
    battery_charge = df_battery_in[customer_id]
    battery_discharge = df_battery_out[customer_id]
    battery_soc_kwh = df_battery_soc_kwh[customer_id]
    
    battery_power = battery_discharge - battery_charge
    final_net_load = original_net_load - battery_power
    capacity_kwh = battery_soc_kwh.max() / 0.95 # Estimate capacity
    soc_percent = (battery_soc_kwh / capacity_kwh) * 100 if capacity_kwh > 0 else 0

    worst_day_index = original_net_load.idxmin().dayofyear - 1
    start_idx, end_idx = worst_day_index * 96, (worst_day_index + 7) * 96
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=original_net_load.index[start_idx:end_idx], y=original_net_load.iloc[start_idx:end_idx], name='Original Net Load (kW)', line=dict(color='royalblue', dash='dash')), secondary_y=False)
    fig.add_trace(go.Scatter(x=final_net_load.index[start_idx:end_idx], y=final_net_load.iloc[start_idx:end_idx], name='Final Net Load (kW)', line=dict(color='green', width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=battery_power.index[start_idx:end_idx], y=battery_power.iloc[start_idx:end_idx], name='Battery Power (kW)', fill='tozeroy', marker_color='crimson'), secondary_y=False)
    fig.add_trace(go.Scatter(x=soc_percent.index[start_idx:end_idx], y=soc_percent.iloc[start_idx:end_idx], name='Battery SoC (%)', line=dict(color='purple')), secondary_y=True)
    
    fig.update_layout(title_text=f'Battery Operation for {customer_id}', legend_title_text='Metric')
    fig.update_yaxes(title_text="Power (kW)", secondary_y=False)
    fig.update_yaxes(title_text="State of Charge (%)", range=[0, 100.5], secondary_y=True)
    fig.write_html(file_path)
    print(f"Saved battery plot to {file_path}")


# --- MAIN EXECUTION SCRIPT ---

def main():
    """Main function to run the entire analysis pipeline."""
    print("--- 1. Setting up and Loading Data ---")
    
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_PROFILES_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PLOTS_PATH, exist_ok=True)
    
    # Load network topology data
    df_full = pd.read_csv(DATA_PATH + "250903_all_stations_anon.csv", sep=";")
    
    # Load and index all yearly profiles
    profiles_path = DATA_PATH + "data_parquet/"
    df_consumption = pd.read_parquet(profiles_path + "base_consumption.parquet")
    df_pv = pd.read_parquet(profiles_path + "pv_profiles.parquet")
    df_ev = pd.read_parquet(profiles_path + "ev_profiles.parquet")
    df_hp = pd.read_parquet(profiles_path + "hp_profiles.parquet")
    
    # Data Preparation and Timestamp Mapping
    datetime_index = pd.date_range(start='2050-01-01', periods=len(df_consumption), freq='15min')
    for df in [df_consumption, df_pv, df_ev, df_hp]:
        df.index = datetime_index

    df_net_load = df_consumption.add(df_ev, fill_value=0).add(df_hp, fill_value=0).subtract(df_pv, fill_value=0)
    customer_peak_generation_kw = -df_net_load.min()
    print("Data loaded and prepared successfully.\n")

    # --- 2. Baseline Analysis ---
    print(f"--- 2. Running Full Baseline Analysis for Station: '{TARGET_STATION}' ---")
    df_one_station = df_full[df_full['station'] == TARGET_STATION].copy()
    G, consumer_props, roots = build_and_simplify_network(df_one_station)
    
    initial_results = find_failures_with_yearly_profile(
        graph=G, net_profile_df=df_net_load, consumer_props=consumer_props,
        root_node_ids=roots, nominal_voltage=NOMINAL_VOLTAGE
    )
    print_analysis_results("RESULTS: Baseline Yearly Profile Analysis", initial_results)

    # --- 3. Scenario 1: Deploying Batteries ---
    print("\n--- 3. Simulating Battery Deployment ---")
    customer_export_limits = create_dynamic_export_limits(
        graph=G, link_failures_list=initial_results['link_failures'],
        consumer_props=consumer_props, root_node_ids=roots,
        default_limit_kw=20.0, strict_limit_kw=10.0
    )

    df_battery_in = pd.DataFrame(0, index=df_net_load.index, columns=df_net_load.columns)
    df_battery_out = pd.DataFrame(0, index=df_net_load.index, columns=df_net_load.columns)
    df_battery_soc_kwh = pd.DataFrame(0, index=df_net_load.index, columns=df_net_load.columns)

    customers_in_station = list(consumer_props.keys())
    pv_customers_in_station = customer_peak_generation_kw.index.intersection(customers_in_station)
    
    start_time = time.time()
    for customer_id in pv_customers_in_station:
        peak_gen_kw = customer_peak_generation_kw.get(customer_id, 0)
        target_kw = customer_export_limits.get(customer_id, 20.0)
        
        power_kw, capacity_kwh = calculate_graphals_battery_size(
            peak_production_kw=peak_gen_kw, export_limit_kw=target_kw
        )
        if capacity_kwh <= 0: continue
            
        customer_battery = SimpleBattery(capacity_kwh, power_kw)
        charge, discharge, soc = create_battery_schedule(df_net_load[customer_id], customer_battery, target_kw)
        
        df_battery_in[customer_id] = charge
        df_battery_out[customer_id] = discharge
        df_battery_soc_kwh[customer_id] = soc

    print(f"Battery simulation completed in {time.time() - start_time:.2f} seconds.")

    df_net_load_with_batteries = df_net_load.add(df_battery_in, fill_value=0).subtract(df_battery_out, fill_value=0)
    results_with_batteries = find_failures_with_yearly_profile(
        graph=G, net_profile_df=df_net_load_with_batteries, consumer_props=consumer_props,
        root_node_ids=roots, nominal_voltage=NOMINAL_VOLTAGE
    )
    print_analysis_results("RESULTS: After Adding Batteries", results_with_batteries)

    # --- 4. Scenario 2: Adding Curtailment ---
    df_net_load_final, df_curtailed_energy = apply_curtailment(df_net_load_with_batteries, customer_export_limits)
    definitive_results = find_failures_with_yearly_profile(
        graph=G, net_profile_df=df_net_load_final, consumer_props=consumer_props,
        root_node_ids=roots, nominal_voltage=NOMINAL_VOLTAGE
    )
    print_analysis_results("DEFINITIVE RESULTS (Batteries + Curtailment)", definitive_results)

    # --- 5. Saving Visualizations ---
    print("\n--- 5. Generating and Saving Example Plots ---")
    # Find a customer who has a battery to visualize
    customer_with_battery = df_battery_in[pv_customers_in_station].sum()
    if not customer_with_battery.empty:
        customer_to_plot = customer_with_battery.idxmax()
        plot_filepath = os.path.join(OUTPUT_PLOTS_PATH, f"battery_plot_{customer_to_plot}.html")
        save_customer_battery_plot(customer_to_plot, df_net_load, df_battery_in, df_battery_out, df_battery_soc_kwh, customer_export_limits, plot_filepath)
    else:
        print("No batteries were allocated, skipping battery plot.")

    # --- 6. Saving Flexibility Profiles ---
    print("\n--- 6. Saving All Flexibility Profiles to Parquet Files ---")
    profiles_to_save = [
        {"df": df_battery_in, "filename": "battery_in_profiles.parquet"},
        {"df": df_battery_out, "filename": "battery_out_profiles.parquet"},
        {"df": df_battery_soc_kwh, "filename": "battery_soc_profiles.parquet"},
        {"df": df_curtailed_energy, "filename": "curtailed_energy_profiles.parquet"},
    ]
    for p_info in profiles_to_save:
        update_and_save_parquet(
            new_data_df=p_info["df"],
            file_path=os.path.join(OUTPUT_PROFILES_PATH, p_info["filename"]),
            customers_to_update=customers_in_station
        )
    
    print("\n--- Analysis and Save Complete. ---")

if __name__ == "__main__":
    main()
