# src/visualization.py

import plotly.graph_objects as go
import networkx as nx
import uuid
from typing import List, Dict, Optional
import pandas as pd

# ==============================================================================
# --- HELPER FUNCTIONS FOR CODE REUSABILITY ---
# ==============================================================================
def _compute_layout(graph: nx.Graph, root_node_ids: List[str], optimize_space: bool=False) -> Dict:
    """
    Computes the network layout. Prioritizes using pre-defined 'pos' attributes
    on nodes. If not available, or if optimize_space is True, falls back to
    abstract layout algorithms.
    """
    # --- Attempt to use pre-defined coordinates first ---
    pos_attributes = nx.get_node_attributes(graph, 'pos')
    
    # Check if all nodes have a 'pos' attribute and we are NOT optimizing for space.
    if len(pos_attributes) == graph.number_of_nodes() and graph.number_of_nodes() > 0 and not optimize_space:
        print("Computing layout: Using pre-defined geographic coordinates from graph nodes.")
        return pos_attributes # Use the coordinates stored in the graph
    
    # --- Fallback to abstract layout if coordinates are missing or if space optimization is requested ---
    if optimize_space:
        print("`optimize_space` is True. Falling back to abstract layout for a compact view.")
    else:
        print("Coordinates not found on graph nodes. Falling back to abstract layout.")
        
    print("Computing layout (pinning root nodes for clarity)...")
    try:
        # Pin root nodes to the top for a hierarchical view
        fixed_nodes = {node: (i * 2, 1) for i, node in enumerate(root_node_ids)}
        return nx.spring_layout(graph, pos=fixed_nodes, fixed=list(fixed_nodes.keys()), k=0.2, iterations=100, seed=42)
    except Exception:
        print("  -> Pinned spring_layout failed, falling back to Kamada-Kawai.")
        try:
            return nx.kamada_kawai_layout(graph)
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            print("  -> Kamada-Kawai layout failed, falling back to simple spring_layout.")
            return nx.spring_layout(graph, k=0.15, iterations=70, seed=42)


def _prepare_failure_maps(link_failures: List[Dict], fuse_failures: List[Dict], graph: nx.Graph) -> tuple:
    """Prepares dictionaries for quick lookup of failure data."""
    failed_link_set = {failure['link'] for failure in link_failures}
    failed_link_details = {failure['link']: failure for failure in link_failures}
    
    failed_node_map = {}
    for failure in fuse_failures:
        consumer_id = failure['consumer_id']
        for node, data in graph.nodes(data=True):
            if data.get('contained_consumers') and consumer_id in data['contained_consumers']:
                failed_node_map.setdefault(node, []).append(failure)
                break
    return failed_link_set, failed_link_details, failed_node_map

# ==============================================================================
# --- CORE VISUALIZATION LOGIC (COMPLETED & CORRECTED) ---
# ==============================================================================

def _visualize_network_core(
    graph: nx.Graph,
    root_node_ids: List[str],
    title: str,
    link_failures: Optional[List[Dict]] = None,
    fuse_failures: Optional[List[Dict]] = None,
    upgraded_link_details: Optional[Dict] = None,
    optimize_space: bool = False,
):
    """Core function to generate and display the network visualization."""
    link_failures = link_failures or []
    fuse_failures = fuse_failures or []
    upgraded_link_details = upgraded_link_details or {}

    if upgraded_link_details:
        link_failures = []

    # --- FIX: Pass the optimize_space flag to the layout function ---
    pos = _compute_layout(graph, root_node_ids, optimize_space=optimize_space)
    
    failed_link_set, failed_link_details, failed_node_map = _prepare_failure_maps(link_failures, fuse_failures, graph)

    traces = []
    edge_traces = {
        'normal': go.Scatter(x=[], y=[], mode='lines', line=dict(width=2, color='grey'), hoverinfo='none', name='Link (Unchanged)'),
        'failed': go.Scatter(x=[], y=[], mode='lines', line=dict(width=6, color='red'), hoverinfo='none', name='Link (Overloaded)'),
        'upgraded': go.Scatter(x=[], y=[], mode='lines', line=dict(width=5, color='blue'), hoverinfo='none', name='Link (Reinforced)')
    }
    em_x, em_y, em_text = [], [], []

    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_tuple = tuple(sorted((u, v)))
        
        link_length = data.get('length', 0.0)
        length_str = f"Link Length: {link_length:.1f} m<br>"

        hover_text = ""
        trace_key = 'normal'
        
        if edge_tuple in failed_link_set:
            trace_key = 'failed'
            details = failed_link_details[edge_tuple]
            hover_text = (f"<b>🚨 OVERLOADED LINK 🚨</b><br>Connection: {u}-{v}<br>{length_str}<br>"
                          f"Max Allowed: {details['max_allowed_current_A']:.2f} A<br>"
                          f"Calculated Peak: {details['calculated_current_A']:.2f} A<br>"
                          f"<b>Overload: {details['overload_percentage']}%</b>")
        
        elif edge_tuple in upgraded_link_details:
            trace_key = 'upgraded'
            upgrade_info = upgraded_link_details[edge_tuple]
            cost = upgrade_info.get('cost_CHF', 0.0)
            hover_text = (f"<b>🔧 REINFORCED LINK 🔧</b><br>Connection: {u}-{v}<br>{length_str}<br>"
                          f"New Cable Type: {data.get('reinforcement_type', 'N/A')}<br>"
                          f"<b>New Max Current: {data.get('Irmax_hoch', 0):.2f} A</b><br>"
                          f"<b>Cost of Fix: {cost:,.2f} CHF</b>")
        else:
            trace_key = 'normal'
            hover_text = (f"Connection: {u}-{v}<br>{length_str}"
                          f"Aggregated Imax: {data.get('Irmax_hoch', 0):.2f} A")
        
        edge_traces[trace_key]['x'] += (x0, x1, None)
        edge_traces[trace_key]['y'] += (y0, y1, None)
        
        em_x.append((x0 + x1) / 2)
        em_y.append((y0 + y1) / 2)
        em_text.append(hover_text)
    
    for trace in edge_traces.values():
        if trace['x']:
            traces.append(trace)
            
    traces.append(go.Scatter(x=em_x, y=em_y, mode='markers', hoverinfo='text', text=em_text, marker=dict(size=15, opacity=0), showlegend=False))    
    node_data = {
        'Root': {'x': [], 'y': [], 'text': [], 'marker': {'size': 22, 'color': '#FFD700', 'symbol': 'star'}},
        'Junction': {'x': [], 'y': [], 'text': [], 'marker': {'size': 10, 'color': 'grey'}},
        'Consumer (OK)': {'x': [], 'y': [], 'text': [], 'marker': {'size': [], 'color': 'green', 'symbol': 'square'}},
        'Consumer (Fuse Failure)': {'x': [], 'y': [], 'text': [], 'marker': {'size': 18, 'color': '#FFA500', 'symbol': 'square-open'}}
    }
    root_node_set = set(root_node_ids)
    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        if node in root_node_set:
            cat = 'Root'
            node_data[cat]['text'].append(f"<b>ID: {node}</b><br>Type: Network Root")
        elif 'contained_consumers' in attrs:
            num = len(attrs['contained_consumers'])
            if node in failed_node_map:
                cat = 'Consumer (Fuse Failure)'
                fails = "<br>".join([f" - {f['consumer_id']}: {f['overload_percentage']:.1f}% overload" for f in failed_node_map[node]])
                node_data[cat]['text'].append(f"<b>ID: {node}</b> ({num} consumers)<br><b>🚨 FUSE FAILURE(S) 🚨</b><br>{fails}")
            else:
                cat = 'Consumer (OK)'
                node_data[cat]['text'].append(f"<b>ID: {node}</b><br>Type: Consumer Node<br>Contained Consumers: {num}")
                node_data[cat]['marker']['size'].append(12 + num * 2)
        else:
            cat = 'Junction'
            node_data[cat]['text'].append(f"<b>ID: {node}</b><br>Type: Junction Point")
        node_data[cat]['x'].append(x)
        node_data[cat]['y'].append(y)
    for name, data in node_data.items():
        if not data['x']: continue
        traces.append(go.Scatter(name=name, x=data['x'], y=data['y'], text=data['text'], mode='markers', hoverinfo='text', marker=data['marker']))
    print("Assembling Plotly figure...")
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=title, showlegend=True, hovermode='closest',
            margin=dict(b=10, l=10, r=10, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_dark',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    )
    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"network_visualization_{unique_id}.html"
    #fig.write_html(output_filename)
    print(f"--- Successfully saved visualization to '{output_filename}' ---")
    fig.show()
    
    
# ==============================================================================
# --- PUBLIC-FACING WRAPPER FUNCTIONS ---
# ==============================================================================

def visualize_network_topology(graph: nx.Graph, root_node_ids: List[str], optimize_space: bool = False):
    """
    Generates an interactive Plotly visualization of the base network topology.
    """
    print("\n--- Creating interactive visualization of the network topology ---")
    # --- FIX: Pass optimize_space as a keyword argument ---
    _visualize_network_core(
        graph=graph,
        root_node_ids=root_node_ids,
        title='Interactive Visualization of Network Topology',
        optimize_space=optimize_space
    )

def visualize_network_with_failures(
    graph: nx.Graph,
    root_node_ids: List[str],
    link_failures: List[Dict],
    fuse_failures: List[Dict],
    optimize_space: bool = False
):
    """
    Generates an interactive visualization of the network, highlighting failures.
    """
    print("\n--- Creating interactive visualization with failure highlighting ---")
    # --- FIX: Added optimize_space parameter for consistency ---
    _visualize_network_core(
        graph=graph,
        root_node_ids=root_node_ids,
        title='Network Capacity Analysis - Visualization of Failures',
        link_failures=link_failures,
        fuse_failures=fuse_failures,
        optimize_space=optimize_space
    )
    


def visualize_reinforced_network(
    reinforced_graph: nx.Graph,
    root_node_ids: list[str],
    reinforcement_plan: pd.DataFrame,
    total_cost: float,
    optimize_space: bool = False
):
    """
    Generates an interactive visualization of the reinforced network topology.
    This function highlights the specific links that were upgraded.
    """
    print("\n--- Creating visualization of the reinforced network ---")
    
    upgraded_link_details = {}
    if not reinforcement_plan.empty and 'fixed_link' in reinforcement_plan.columns:
        plan_copy = reinforcement_plan.copy()
        plan_copy['sorted_link'] = plan_copy['fixed_link'].apply(lambda x: tuple(sorted(x)))
        upgraded_link_details = plan_copy.set_index('sorted_link').to_dict(orient='index')

    title = f"Reinforced Network Topology - Total Cost: {total_cost:,.2f} CHF"
    
    # --- FIX: Added optimize_space parameter for consistency ---
    _visualize_network_core(
        graph=reinforced_graph,
        root_node_ids=root_node_ids,
        title=title,
        upgraded_link_details=upgraded_link_details,
        optimize_space=optimize_space
    )
    

def visualize_grid_improvement(
    initial_graph: nx.Graph,
    initial_failures: list[dict],
    reinforced_graph: nx.Graph,
    reinforcement_plan: pd.DataFrame,
    root_node_ids: list[str],
    total_cost: float,
    optimize_space: bool = False
):
    """
    Creates a powerful "before and after" visualization of the grid reinforcement.
    """
    print("\n--- Creating 'Before & After' visualization of grid improvements ---")

    upgraded_link_details = {}
    if not reinforcement_plan.empty:
        plan_copy = reinforcement_plan.copy()
        plan_copy['sorted_link'] = plan_copy['fixed_link'].apply(lambda x: tuple(sorted(x)))
        upgraded_link_details = plan_copy.set_index('sorted_link').to_dict(orient='index')
        
    initial_failure_links = [
        {'link': tuple(sorted(f['link'])), **f} for f in initial_failures
    ]

    title = f"Grid Reinforcement Analysis: {len(upgraded_link_details)} Upgrades for {total_cost:,.2f} CHF"

    # --- FIX: Added optimize_space parameter for consistency ---
    _visualize_network_core(
        graph=reinforced_graph,
        root_node_ids=root_node_ids,
        title=title,
        link_failures=initial_failure_links,
        upgraded_link_details=upgraded_link_details,
        optimize_space=optimize_space
    )