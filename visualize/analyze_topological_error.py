#!/usr/bin/env python3
"""
3D Error Localization Pipeline: Topological Analysis
Computes Betweenness Centrality and Degree to correlate with GNN prediction error.
Note: Betweenness Centrality is computationally expensive for large graphs.
We focus on a single high-strain timestep (2.5% strain -> timestep 25).
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import vtk
    from vtk.util import numpy_support
except ImportError:
    print("Error: VTK not installed.")
    sys.exit(1)

ASSEMBLY_ID = 58
TARGET_TS = 25  # 2.5% Strain
DATA_DIR = "data_sets(0-72)"
OUT_DIR = "Statistical_Error_Plots"
VTK_DIR = "Error_VTK_particles_single_run"

# Optional paths check
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(PROJECT_DIR, DATA_DIR)
if not os.path.exists(VTK_DIR):
    VTK_DIR = os.path.join(PROJECT_DIR, VTK_DIR)


def read_tab_file(fp):
    data = []
    try:
        with open(fp) as f:
            for i, line in enumerate(f):
                if i < 2:  # Skip headers
                    continue
                parts = re.split(r"[\s,]+", line.strip())
                if len(parts) >= 2 and parts[0]:
                    data.append(int(float(parts[1])))
    except Exception as e:
        print(f"Error reading {fp}: {e}")
    return data

def build_graph(gid, ts):
    prefix = f"{gid}_{ts}"
    end1 = read_tab_file(os.path.join(DATA_DIR, f"{prefix}_contact_end1.tab"))
    end2 = read_tab_file(os.path.join(DATA_DIR, f"{prefix}_contact_end2.tab"))
    
    if not end1 or not end2 or len(end1) != len(end2):
        print("Error loading contact edges.")
        return None
        
    G = nx.Graph()
    edges = list(zip(end1, end2))
    G.add_edges_from(edges)
    return G

def read_vtk_errors(gid, ts):
    fname = f"particle_error_graph{gid:02d}_t{ts:03d}.vtk"
    fpath = os.path.join(VTK_DIR, fname)
    if not os.path.exists(fpath):
        print(f"Error: {fpath} not found.")
        return None
        
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fpath)
    reader.Update()
    polydata = reader.GetOutput()
    
    pd_data = polydata.GetPointData()
    pids = numpy_support.vtk_to_numpy(pd_data.GetArray("particle_id"))
    errs = numpy_support.vtk_to_numpy(pd_data.GetArray("particle_force_absolute_error"))
    
    return {int(p): float(e) for p, e in zip(pids, errs)}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Building Graph for Assembly {ASSEMBLY_ID}, Timestep {TARGET_TS}...")
    G = build_graph(ASSEMBLY_ID, TARGET_TS)
    if not G:
        sys.exit(1)
        
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    print("Computing Betweenness Centrality (this may take a few minutes for 10k+ nodes)...")
    # To save time, we compute approximate betweenness if the graph is huge
    if G.number_of_nodes() > 5000:
        print("Graph is large, using approximation (k=1000)...")
        betweenness = nx.betweenness_centrality(G, k=1000, normalized=True)
    else:
        betweenness = nx.betweenness_centrality(G, normalized=True)
        
    print("Loading Prediction Errors...")
    errors = read_vtk_errors(ASSEMBLY_ID, TARGET_TS)
    if not errors:
        sys.exit(1)
        
    data = []
    for node in G.nodes():
        if node in errors:
            data.append({
                "particle_id": node,
                "degree": G.degree[node],
                "betweenness": betweenness[node],
                "abs_error": errors[node]
            })
            
    df = pd.DataFrame(data)
    
    print("Plotting Topological Analysis...")
    # Plot Betweenness vs Error
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='betweenness', y='abs_error', alpha=0.6, color='purple')
    plt.title("Figure 9: Prediction Error vs. Betweenness Centrality")
    plt.xlabel("Betweenness Centrality (Log Scale)")
    plt.ylabel("Absolute Prediction Error")
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Figure9_Error_vs_Betweenness.png")
    plt.savefig(out, dpi=300)
    print(f"✅ Saved: {out}")
    plt.close()
    
    print("\n✅ Topological Analysis Complete!")

if __name__ == "__main__":
    main()
