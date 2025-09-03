# EDH-Hackathon/src/functions.py

"""
This module provides functions for analyzing a low-voltage electrical grid.
It includes capabilities to:
1.  Build and simplify a network graph from raw CSV data.
2.  Simulate grid load over a yearly time-series profile to identify overloads.
"""

import pandas as pd
import networkx as nx
import numpy as np
import re
import operator
import time

# ==============================================================================
# --- NETWORK BUILDING AND SIMPLIFICATION (MULTI-TRANSFORMER SUPPORT) ---
# ==============================================================================

def build_and_simplify_network(df_station: pd.DataFrame) -> tuple:
    """
    Processes raw network data for a SINGLE STATION, builds a simplified graph,
    and prunes dangling edges.

    MODIFICATION: Added coordinate columns to the numeric conversion step to
                  prevent TypeError during visualization.

    Args:
        df_station (pd.DataFrame): DataFrame for a single station.

    Returns:
        tuple: (G_simplified, consumer_properties, root_node_ids)
    """
    station_name = df_station['station'].iloc[0] if not df_station.empty else "Unknown"
    print(f"\n--- Building and simplifying network for: {station_name} ---")

    df_raw = df_station.copy()

    # Step 1 is unchanged...
    # ==============================================================================
    # === Step 1: Normalize Consumer Connection Direction ===
    # ==============================================================================
    print("Step 1: Normalizing consumer connection directions...")
    reversed_mask = df_raw['to'].str.startswith('HAS', na=False) & \
                    ~df_raw['from'].str.startswith('HAS', na=False)
    df_raw.loc[reversed_mask, ['from', 'to']] = \
        df_raw.loc[reversed_mask, ['to', 'from']].values
    
    # ==============================================================================
    # --- Step 2: Convert columns to appropriate data types ---
    # <<< FIX APPLIED HERE >>>
    # ==============================================================================
    print("Step 2: Converting columns to appropriate data types...")
    # Add the coordinate columns to the list of columns to be converted to numeric
    numeric_cols = [
        'length', 'ratedCurrent', 'Irmax_hoch', 'X', 'R', 'X0', 'R0', 
        'C', 'G', 'C0', 'x1', 'y1', 'x2', 'y2'
    ]
    string_cols = ['from', 'to', 'id_equ', 'name', 'station']
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    for col in string_cols:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].astype('string')
    if 'normalOpen' in df_raw.columns:
        df_raw['normalOpen'] = df_raw['normalOpen'].astype(bool)

    # ==============================================================================
    # --- Step 2.5: Extract Node Coordinates ---
    # ==============================================================================
    print("Step 2.5: Extracting node coordinates...")
    node_coordinates = {}
    coord_cols = ['x1', 'y1', 'x2', 'y2']
    
    if all(col in df_raw.columns for col in coord_cols):
        from_nodes = df_raw[['from', 'x1', 'y1']].rename(
            columns={'from': 'node_id', 'x1': 'x', 'y1': 'y'}
        )
        to_nodes = df_raw[['to', 'x2', 'y2']].rename(
            columns={'to': 'node_id', 'x2': 'x', 'y2': 'y'}
        )
        all_nodes_df = pd.concat([from_nodes, to_nodes]).dropna().drop_duplicates(subset=['node_id'])
        node_coordinates = {row.node_id: (row.x, row.y) for row in all_nodes_df.itertuples()}
        print(f"  -> Extracted coordinates for {len(node_coordinates)} unique nodes.")
    else:
        print("  -> Coordinate columns ('x1', 'y1', 'x2', 'y2') not found. Skipping coordinate extraction.")

    # The rest of the function remains unchanged...
    # ==============================================================================
    # --- Step 3: Segregating network data ---
    # ==============================================================================
    print("Step 3: Segregating network data...")
    is_consumer_row = df_raw['from'].str.startswith('HAS', na=False)
    is_transformer_row = df_raw['clazz'] == 'PowerTransformer'
    df_consumers = df_raw[is_consumer_row].copy()
    df_transformers = df_raw[is_transformer_row].copy()
    df_edges = df_raw[~is_transformer_row].copy()
    transformer_pins = set()
    if not df_transformers.empty:
        transformer_pins = set(df_transformers['from']).union(set(df_transformers['to']))
        
    # ==============================================================================
    # --- Step 4: Infer and Store Consumer Properties ---
    # ==============================================================================
    print("Step 4: Parsing and storing consumer properties...")
    STANDARD_FUSE_SIZES = sorted([35, 50, 63, 80, 100, 125, 160, 200, 250], reverse=True)
    def _infer_fuse_rating(name_str: str, cable_imax: float) -> float:
        if pd.notna(name_str):
            match = re.search(r'(\d+)\s*A', str(name_str))
            if match: return float(match.group(1))
        if cable_imax > 0:
            for fuse_size in STANDARD_FUSE_SIZES:
                if fuse_size <= cable_imax: return float(fuse_size)
        return 63.0

    consumer_properties = {}
    for consumer_id, group in df_consumers.groupby('from'):
        consumer_row = group.iloc[0]
        cable_imax = consumer_row.get('Irmax_hoch', 0)
        consumer_properties[consumer_id] = {
            'is_consumer': True, 'consumer_Imax': cable_imax,
            'consumer_fuse_A': _infer_fuse_rating(consumer_row['name'], cable_imax),
            'consumer_R': consumer_row.get('R', 0), 'consumer_X': consumer_row.get('X', 0),
        }

    # ==============================================================================
    # --- Step 5: Simplifying Network Topology ---
    # ==============================================================================
    print("Step 5: Simplifying network topology (merging nodes and parallel edges)...")
    G_multi = nx.from_pandas_edgelist(df_edges, source='from', target='to', edge_attr=True, create_using=nx.MultiGraph)
    df_zero_length_edges = df_edges[df_edges['length'] == 0]
    G_contracted = G_multi.copy()
    node_to_representative = {}

    if not df_zero_length_edges.empty:
        G_zero_length = nx.from_pandas_edgelist(df_zero_length_edges, 'from', 'to')
        components_to_merge = list(nx.connected_components(G_zero_length))
        for component in components_to_merge:
            representative_node = sorted(list(component), key=lambda x: (x not in transformer_pins, x.startswith('HAS'), x))[0]
            for node in component:
                node_to_representative[node] = representative_node
                if node != representative_node and G_contracted.has_node(node):
                    G_contracted = nx.contracted_nodes(G_contracted, representative_node, node, self_loops=False)
    
    G_simplified = nx.Graph()
    for u, v in G_contracted.edges():
        if G_simplified.has_edge(u, v): continue
        parallel_edges = list(G_contracted.get_edge_data(u, v).values())
        agg_imax = sum(edge.get('Irmax_hoch', 0) for edge in parallel_edges)
        agg_length = np.mean([edge.get('length', 0) for edge in parallel_edges])
        total_conductance_G, total_susceptance_B = 0, 0
        for edge in parallel_edges:
            R, X = edge.get('R', 0), edge.get('X', 0)
            z_squared = R**2 + X**2
            if z_squared > 0:
                total_conductance_G += R / z_squared
                total_susceptance_B += -X / z_squared
        y_squared = total_conductance_G**2 + total_susceptance_B**2
        agg_r = total_conductance_G / y_squared if y_squared > 0 else 0
        agg_x = -total_susceptance_B / y_squared if y_squared > 0 else 0
        G_simplified.add_edge(u, v, Irmax_hoch=agg_imax, length=agg_length, R=agg_r, X=agg_x, parallel_count=len(parallel_edges))

    for original_node, props in consumer_properties.items():
        final_node_name = node_to_representative.get(original_node, original_node)
        if not G_simplified.has_node(final_node_name): G_simplified.add_node(final_node_name)
        if 'contained_consumers' not in G_simplified.nodes[final_node_name]:
            G_simplified.nodes[final_node_name].update({'contained_consumers': [], 'is_consumer_connection': True})
        G_simplified.nodes[final_node_name]['contained_consumers'].append(original_node)
        
    if node_coordinates:
        print("  -> Attaching coordinates to simplified graph nodes...")
        nodes_with_coords = 0
        for node in G_simplified.nodes():
            coords = node_coordinates.get(node)
            if coords:
                G_simplified.nodes[node]['pos'] = coords
                nodes_with_coords += 1
        print(f"  -> Successfully attached coordinates to {nodes_with_coords}/{G_simplified.number_of_nodes()} nodes.")

    # ==============================================================================
    # --- Step 6: Detecting all root nodes (transformers) ---
    # ==============================================================================
    print("Step 6: Detecting root nodes (transformer LV-sides)...")
    root_node_ids = set()
    if not df_transformers.empty:
        for _, row in df_transformers.iterrows():
            from_node, to_node = row['from'], row['to']
            rep_from = node_to_representative.get(from_node, from_node)
            rep_to = node_to_representative.get(to_node, to_node)
            degree_from = G_simplified.degree(rep_from) if G_simplified.has_node(rep_from) else 0
            degree_to = G_simplified.degree(rep_to) if G_simplified.has_node(rep_to) else 0
            if degree_to > degree_from: root_node_ids.add(rep_to)
            elif degree_from > degree_to: root_node_ids.add(rep_from)
            elif degree_to > 0: root_node_ids.add(rep_to)
            else: print(f"  -> Warning: Transformer between {from_node} and {to_node} seems disconnected.")
    root_node_ids = list(root_node_ids)

    if not root_node_ids and G_simplified.number_of_nodes() > 0:
        print("  -> Warning: No valid 'PowerTransformer' elements found. Falling back to centrality.")
        centrality = nx.betweenness_centrality(G_simplified)
        single_root = max(centrality, key=centrality.get)
        root_node_ids.append(single_root)
        print(f"  -> Selected '{single_root}' as the single root based on fallback method.")
    for root_id in root_node_ids:
        if G_simplified.has_node(root_id):
            G_simplified.nodes[root_id]['is_transformer'] = True

    # ==============================================================================
    # --- Step 7: Prune Dangling Edges ---
    # ==============================================================================
    print("Step 7: Pruning dangling edges without consumers...")
    nodes_pruned_total = 0
    while True:
        nodes_to_remove = []
        for node in G_simplified.nodes():
            is_leaf = G_simplified.degree(node) == 1
            is_consumer = G_simplified.nodes[node].get('is_consumer_connection', False)
            is_root = node in root_node_ids

            if is_leaf and not is_consumer and not is_root:
                nodes_to_remove.append(node)
        
        if not nodes_to_remove:
            break
        else:
            G_simplified.remove_nodes_from(nodes_to_remove)
            nodes_pruned_total += len(nodes_to_remove)

    if nodes_pruned_total > 0:
        print(f"  -> Finished pruning. A total of {nodes_pruned_total} nodes were removed.")
    else:
        print("  -> No dangling non-consumer edges found to prune.")
    
    print(f"✅ Network build complete for {station_name}. Detected {len(root_node_ids)} root node(s): {root_node_ids}")

    return G_simplified, consumer_properties, root_node_ids

# ==============================================================================
# --- NETWORK FAILURE ANALYSIS (MULTI-TRANSFORMER SUPPORT) ---
# ==============================================================================

def find_failures_with_yearly_profile(
    graph: nx.Graph,
    net_profile_df: pd.DataFrame,
    consumer_props: dict,
    root_node_ids: list[str], # MODIFIED: Now accepts a list of root nodes
    nominal_voltage: float = 230.0,
    power_factor: float = 1.0
) -> dict:
    """
    Finds network failures by simulating power flow for a yearly profile.
    
    This version is updated to handle networks with multiple transformers (root nodes)
    by using a "super source" simulation method.

    This analysis is performed in two parts:
    1.  Fuse Failures: A worst-case check on each consumer's max injection. (Unchanged)
    2.  Link Failures: A time-series simulation. It iterates through every
        timestep, calculates the current flowing through each cable towards all
        sources, and records the maximum current to check against thermal limits.

    Args:
        graph (nx.Graph): The simplified network graph.
        net_profile_df (pd.DataFrame): DataFrame with consumer IDs as columns and
                                       net power (kW) at each timestep as rows.
        consumer_props (dict): Dictionary of consumer properties.
        root_node_ids (list[str]): A list of IDs for all transformer/grid connection points.
        nominal_voltage (float): The nominal phase-to-neutral voltage (V).
        power_factor (float): The power factor for converting power to current.

    Returns:
        dict: A dictionary containing 'fuse_failures', 'link_failures', and the
              graph with analysis results attached ('graph_analysis').
    """
    t_start_total = time.time()
    print("\n--- Starting Yearly Profile-Based Network Analysis (Multi-Transformer Mode) ---")

    # --- Input Validation for Multiple Roots ---
    if not isinstance(root_node_ids, list) or not root_node_ids:
        raise ValueError("`root_node_ids` must be a non-empty list.")
    for root_id in root_node_ids:
        if not graph.has_node(root_id):
            raise ValueError(f"Root node '{root_id}' from the list was not found in the graph.")

    g_analysis = graph.copy()

    # --- Part A: Check Consumer Fuse Limits (Worst-Case Injection) ---
    # This part is independent of network topology and remains unchanged.
    print("Step A: Checking for fuse failures based on max yearly injection...")
    t_start_fuse = time.time()
    fuse_failures = []
    max_injection_kw = net_profile_df.min()

    for consumer_id, injection_kw in max_injection_kw.items():
        if injection_kw >= 0:
            continue
        if consumer_id not in consumer_props:
            continue
        
        # P = V * I * pf  =>  I = P / (V * pf)
        # Using phase-to-neutral voltage, so no sqrt(3) is needed.
        power_watts = abs(injection_kw) * 1000
        generated_current = power_watts / (nominal_voltage * power_factor)
        fuse_limit = consumer_props[consumer_id].get('consumer_fuse_A', np.inf)

        if generated_current > fuse_limit:
            overload = ((generated_current - fuse_limit) / fuse_limit) * 100
            fuse_failures.append({
                'consumer_id': consumer_id,
                'fuse_limit_A': fuse_limit,
                'generated_current_A': round(generated_current, 2),
                'overload_percentage': round(overload, 1)
            })
    t_end_fuse = time.time()
    print(f"  -> Fuse check completed in {t_end_fuse - t_start_fuse:.2f} seconds.")


    # --- Part B: Check Network Link Overloads (Time-Series Simulation) ---
    print("Step B: Simulating power flow using the 'Super Source' method...")
    t_start_simulation = time.time()

    # --- Setup the Super Source for simulation ---
    SUPER_SOURCE_ID = 'SUPER_SOURCE'
    g_analysis.add_node(SUPER_SOURCE_ID)
    for root_id in root_node_ids:
        # Add a zero-impedance, infinite-capacity link from each real root to the super source
        g_analysis.add_edge(root_id, SUPER_SOURCE_ID, Irmax_hoch=np.inf, R=0, X=0, length=0)
    
    # Initialize a "high-water mark" for current on each edge.
    nx.set_edge_attributes(g_analysis, 0.0, 'max_observed_current_A')

    # The recursive function is unchanged, but it will now trace paths to the SUPER_SOURCE.
    def _calculate_upstream_flow(node_id, visited_nodes):
        visited_nodes.add(node_id)
        
        power_watts = g_analysis.nodes[node_id].get('current_timestep_kw', 0) * -1000
        current_from_this_node = power_watts / (nominal_voltage * power_factor)
        
        total_downstream_current = 0
        for neighbor_id in g_analysis.neighbors(node_id):
            if neighbor_id not in visited_nodes:
                child_current = _calculate_upstream_flow(neighbor_id, visited_nodes)
                g_analysis.edges[node_id, neighbor_id]['calculated_current'] = child_current
                total_downstream_current += child_current
        
        return current_from_this_node + total_downstream_current

    # Main simulation loop
    num_timesteps = len(net_profile_df)
    for i, (timestamp, series) in enumerate(net_profile_df.iterrows()):
        if (i + 1) % 5000 == 0:
            print(f"  ...processed {i+1} of {num_timesteps} timesteps...")

        # 1. Update node power for the current timestep
        nx.set_node_attributes(g_analysis, 0, 'current_timestep_kw')
        for node, data in g_analysis.nodes(data=True):
            if 'contained_consumers' in data:
                total_kw = sum(series.get(cons_id, 0) for cons_id in data['contained_consumers'])
                g_analysis.nodes[node]['current_timestep_kw'] = total_kw
    
        # 2. Run the power flow calculation starting from the SUPER_SOURCE.
        _calculate_upstream_flow(SUPER_SOURCE_ID, visited_nodes=set())

        # 3. Update the max_observed_current for each edge.
        for u, v, data in g_analysis.edges(data=True):
            current_now = abs(data.get('calculated_current', 0))
            if current_now > data['max_observed_current_A']:
                g_analysis.edges[u, v]['max_observed_current_A'] = current_now

    t_end_simulation = time.time()
    print("  ...simulation complete.")
    print(f"  -> Time-series simulation completed in {t_end_simulation - t_start_simulation:.2f} seconds.")


    # 4. Final Assessment: Find failures, ignoring the artificial super source links.
    link_failures = []
    # Iterate over original graph edges to avoid including the artificial ones.
    for u, v, data in graph.edges(data=True):
        # Get the max observed current from the analysis graph
        max_observed = g_analysis.edges[u, v].get('max_observed_current_A', 0)
        max_allowed = data.get('Irmax_hoch', 0)

        if max_allowed > 0 and max_observed > max_allowed:
            overload = ((max_observed - max_allowed) / max_allowed) * 100
            link_failures.append({
                'link': tuple(sorted((u, v))),
                'max_allowed_current_A': round(max_allowed, 2),
                'calculated_current_A': round(max_observed, 2),
                'overload_percentage': round(overload, 1)
            })

    # Clean up the analysis graph by removing the super source for a cleaner output
    g_analysis.remove_node(SUPER_SOURCE_ID)
    
    t_end_total = time.time()
    print(f"\n--- Analysis Finished. Total elapsed time: {t_end_total - t_start_total:.2f} seconds. ---")
            
    return {'fuse_failures': fuse_failures, 'link_failures': link_failures, "graph_analysis": g_analysis}



# ==============================================================================
# --- NETWORK FAILURE ANALYSIS (MULTI-TRANSFORMER SUPPORT) ---
# ==============================================================================

def suggest_grid_reinforcement(
    initial_graph: nx.Graph,
    initial_results: dict,
    reinforcement_costs_df: pd.DataFrame,
    # Parameters needed for re-analysis
    net_profile_df: pd.DataFrame,
    consumer_props: dict,
    root_node_ids: list,
    nominal_voltage: float,
    max_iterations: int = 50 # Kept for signature compatibility, but not used in logic
) -> dict:
    """
    Analyzes grid failures and suggests a cost-effective reinforcement plan by
    applying all necessary changes in a single, efficient run.

    This method works by:
    1. Identifying ALL overloaded links from the initial failure analysis.
    2. For each failed link, determining the cheapest cable upgrade that meets its peak demand.
    3. Aggregating all upgrades into a single reinforcement plan.
    4. Applying all changes to the graph model.
    5. Running a final verification analysis to confirm the solution is effective.

    Args:
        initial_graph: The original simplified graph.
        initial_results: The results from the first failure analysis run.
        reinforcement_costs_df: DataFrame with costs and specs for upgrades.
        net_profile_df, consumer_props, root_node_ids, nominal_voltage:
            Parameters required to re-run the failure analysis for verification.
        max_iterations: Safeguard parameter, not used in this single-run approach.

    Returns:
        A dictionary containing the final status, total cost, and reinforcement plan.
    """
    print("\n" + "="*20 + "\n--- Starting Grid Reinforcement (Single Run Mode) ---\n" + "="*20)
    
    # --- 1. Prepare Inputs ---
    g_reinforced = initial_graph.copy()
    initial_failures = initial_results['link_failures']
    
    if not initial_failures:
        print("✅ No link failures were found in the initial analysis. No reinforcement needed.")
        return {
            'status': 'Success',
            'total_cost_CHF': 0.0,
            'reinforcement_plan': pd.DataFrame(),
            'reinforced_graph': g_reinforced
        }

    # Pre-process the reinforcement options for quick lookups
    df_lines = reinforcement_costs_df[
        reinforcement_costs_df['material'] == 'line'
    ].sort_values('cost').copy()
    df_lines['Irmax'] = pd.to_numeric(df_lines['Irmax'], errors='coerce')
    
    non_repairable_cost_row = reinforcement_costs_df[reinforcement_costs_df['type'] == 'large']
    NON_REPAIRABLE_COST = non_repairable_cost_row['cost'].iloc[0] if not non_repairable_cost_row.empty else 244728.1650

    total_cost = 0.0
    reinforcement_plan = []
    
    print(f"Found {len(initial_failures)} overloaded links to fix. Planning all upgrades now.")

    # --- 2. Plan All Reinforcements Based on Initial Analysis ---
    for i, failure in enumerate(initial_failures):
        link_to_fix = failure['link']
        required_current = failure['calculated_current_A']
        u, v = link_to_fix
        
        # Find the cheapest valid cable upgrade for this specific link
        possible_upgrades = df_lines[df_lines['Irmax'] > required_current]
        
        if possible_upgrades.empty:
            print(f"\n🚨 CRITICAL FAILURE: Link {link_to_fix} requires > {required_current:.2f} A.")
            print(f"   The maximum available cable capacity is {df_lines['Irmax'].max()} A.")
            print("   The grid is considered non-repairable with the available materials.")
            return {
                'status': 'Non-Repairable',
                'total_cost_CHF': NON_REPAIRABLE_COST,
                'reason': f"Link {link_to_fix} requires > {required_current:.2f} A, but max available is {df_lines['Irmax'].max()} A.",
                'reinforcement_plan': pd.DataFrame(reinforcement_plan)
            }
        
        cheapest_upgrade = possible_upgrades.iloc[0]
        
        # Calculate cost and prepare the action log
        link_data = g_reinforced.edges[u, v]
        link_length = link_data.get('length', 0)
        
        fix_cost = 0
        if link_length > 0:
            fix_cost = cheapest_upgrade['cost'] * link_length
        
        total_cost += fix_cost
        
        action = {
            'fix_order': i + 1,
            'fixed_link': link_to_fix,
            'overload_percentage': failure['overload_percentage'],
            'original_Imax_A': link_data.get('Irmax_hoch'),
            'required_Imax_A': required_current,
            'new_cable_type': cheapest_upgrade['type'],
            'new_cable_Imax_A': cheapest_upgrade['Irmax'],
            'cost_CHF': round(fix_cost, 2)
        }
        reinforcement_plan.append(action)
        
        # Apply the fix to the graph model in memory
        g_reinforced.edges[u, v]['Irmax_hoch'] = cheapest_upgrade['Irmax']
        g_reinforced.edges[u, v]['reinforcement_type'] = cheapest_upgrade['type']

    print("\n--- Summary of Planned Reinforcements ---")
    print(pd.DataFrame(reinforcement_plan).to_string(index=False))
    print(f"\nTotal estimated cost: {total_cost:.2f} CHF")
    
    # --- 3. Verification Step ---
    print("\n--- Verifying the Complete Reinforcement Plan ---")
    print("-> Re-running failure analysis on the fully modified grid...")
    
    final_results = find_failures_with_yearly_profile(
        graph=g_reinforced,
        net_profile_df=net_profile_df,
        consumer_props=consumer_props,
        root_node_ids=root_node_ids,
        nominal_voltage=nominal_voltage
    )
    
    # --- 4. Finalize and Return Results ---
    if final_results['link_failures']:
        print("\n" + "="*20 + "\n--- Reinforcement Verification FAILED ---\n" + "="*20)
        print("🚨 After applying all upgrades, new failures were detected. This indicates a complex network effect.")
        print("   This can happen if changing cable impedances (not modeled here) reroutes current in unexpected ways.")
        print("   The proposed plan is not a complete solution.")
        
        return {
            'status': 'Failed (Verification Check)',
            'total_cost_CHF': round(total_cost, 2),
            'reinforcement_plan': pd.DataFrame(reinforcement_plan),
            'remaining_failures_after_fix': final_results['link_failures']
        }

    print("\n" + "="*20 + "\n--- Reinforcement Complete and Verified! ---\n" + "="*20)
    print("✅ All link failures have been successfully resolved with the proposed plan.")
    
    return {
        'status': 'Success',
        'total_cost_CHF': round(total_cost, 2),
        'reinforcement_plan': pd.DataFrame(reinforcement_plan),
        'reinforced_graph': g_reinforced
    }


# --- Helper function for printing results consistently ---
def print_analysis_results(title, results):
    print(f"\n{'='*20}\n--- {title} ---\n{'='*20}")
    fuse_failures = results['fuse_failures']
    link_failures = results['link_failures']

    if not fuse_failures and not link_failures:
        print("\n✅ SUCCESS: The network is robust under these conditions. No overloads detected.")
        return

    if fuse_failures:
        print(f"\n🚨 FUSE FAILURES: Found {len(fuse_failures)} overloaded consumer fuses.")
        display(pd.DataFrame(fuse_failures))
    else:
        print("\n✅ No fuse failures were detected.")

    if link_failures:
        print(f"\n🚨 LINK FAILURES: Found {len(link_failures)} overloaded cables.")
        display(pd.DataFrame(link_failures))
    else:
        print("\n✅ No link/cable failures were detected.")
        
        
        

import os

def update_and_save_parquet(new_data_df, file_path, customers_to_update):
    """
    Saves or updates a Parquet file with new profile data for a specific set of customers.

    Args:
        new_data_df (pd.DataFrame): DataFrame containing the new profile data. 
                                    It can be a large frame, but only data from 
                                    'customers_to_update' will be used.
        file_path (str): The full path to the Parquet file to be saved.
        customers_to_update (list): A list of customer IDs whose data should be
                                    updated or added to the file.
    """
    # Filter the new data to only include columns for the customers we just analyzed
    relevant_new_data = new_data_df[customers_to_update]

    if os.path.exists(file_path):
        print(f"File '{os.path.basename(file_path)}' exists. Loading and updating...")
        try:
            existing_df = pd.read_parquet(file_path)
            
            # Update existing columns and add new ones from the relevant new data
            for col in relevant_new_data.columns:
                existing_df[col] = relevant_new_data[col]
            
            final_df = existing_df
            print(f"Updated data for {len(customers_to_update)} customers.")

        except Exception as e:
            print(f"Error reading existing file {file_path}: {e}. Overwriting with new data.")
            final_df = relevant_new_data
    else:
        print(f"File '{os.path.basename(file_path)}' does not exist. Creating new file...")
        final_df = relevant_new_data

    try:
        final_df.to_parquet(file_path, index=True)
        print(f"Successfully saved data to '{file_path}'")
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")

