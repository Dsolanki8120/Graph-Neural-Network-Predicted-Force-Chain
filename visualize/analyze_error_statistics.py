#!/usr/bin/env python3
"""
3D Error Localization Pipeline: Statistical Analysis
Generates statistical plots from Error VTKs (Figure 5, Figure 6, Figure 7).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    import vtk
    from vtk.util import numpy_support
except ImportError:
    print("Error: VTK not installed.")
    sys.exit(1)

ASSEMBLY_ID = 58
# Target timesteps mapping to strain (0.1% per timestep)
TARGET_TIMESTEPS = {5: "0.5%", 10: "1.0%", 15: "1.5%", 20: "2.0%", 25: "2.5%"}

POSSIBLE_DIRS = [
    "Error_VTK_particles_single_run",
    "../Error_VTK_particles_single_run",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Error_VTK_particles_single_run")
]
OUT_DIR = "Statistical_Error_Plots"

def find_vtk_directory():
    for d in POSSIBLE_DIRS:
        if os.path.exists(d):
            return d
    return None

def read_vtk_data(filepath):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    polydata = reader.GetOutput()
    
    points = polydata.GetPoints()
    if not points:
        return None
        
    positions = numpy_support.vtk_to_numpy(points.GetData())
    pd_data = polydata.GetPointData()
    
    def get_arr(name):
        arr = pd_data.GetArray(name)
        return numpy_support.vtk_to_numpy(arr) if arr else None

    return {
        "positions": positions,
        "pid": get_arr("particle_id"),
        "actual": get_arr("actual_npmncf"),
        "predicted": get_arr("predicted_npmncf"),
        "abs_error": get_arr("particle_force_absolute_error"),
        "cn": get_arr("coordination_number")
    }

def plot_error_vs_cn(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='coordination_number', y='abs_error', palette='viridis')
    plt.title("Figure 5: Error vs Coordination Number (Combined Strains)")
    plt.xlabel("Coordination Number")
    plt.ylabel("Absolute Prediction Error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Figure5_Error_vs_CN.png")
    plt.savefig(out, dpi=300)
    print(f"✅ Saved: {out}")
    plt.close()

def plot_error_vs_force(df):
    plt.figure(figsize=(10, 6))
    # Sample down if too many points to avoid crowding
    sample_df = df.sample(min(len(df), 10000)) if len(df) > 10000 else df
    sns.scatterplot(data=sample_df, x='actual_force', y='abs_error', hue='strain_pct', alpha=0.5, s=15, palette='plasma')
    plt.title("Figure 6: Error vs True Force Magnitude")
    plt.xlabel("True Normalized Force (NPMNCF)")
    plt.ylabel("Absolute Prediction Error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Figure6_Error_vs_Force.png")
    plt.savefig(out, dpi=300)
    print(f"✅ Saved: {out}")
    plt.close()

def plot_layer_wise_error(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='layer', y='abs_error', hue='strain_pct', palette='coolwarm')
    plt.title("Figure 7: Layer-wise Average Error across Strains")
    plt.xlabel("Sample Layer (Z-axis)")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Figure7_Layer_Error.png")
    plt.savefig(out, dpi=300)
    print(f"✅ Saved: {out}")
    plt.close()

def main():
    vtk_dir = find_vtk_directory()
    if not vtk_dir:
        print("Error: Could not find VTK directory.")
        sys.exit(1)
        
    os.makedirs(OUT_DIR, exist_ok=True)
    all_data = []

    for ts, strain_label in TARGET_TIMESTEPS.items():
        fname = f"particle_error_graph{ASSEMBLY_ID:02d}_t{ts:03d}.vtk"
        fpath = os.path.join(vtk_dir, fname)
        if not os.path.exists(fpath):
            print(f"⚠️ Missing file: {fname}")
            continue
            
        print(f"Processing {fname} ({strain_label} Strain)")
        data = read_vtk_data(fpath)
        if not data or data['abs_error'] is None:
            continue
            
        z_coords = data['positions'][:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        z_range = z_max - z_min
        
        # Determine layers
        layers = []
        for z in z_coords:
            rel_z = (z - z_min) / z_range
            if rel_z <= 0.25:
                layers.append("Bottom 25%")
            elif rel_z >= 0.75:
                layers.append("Top 25%")
            else:
                layers.append("Middle 50%")
                
        df = pd.DataFrame({
            'abs_error': data['abs_error'],
            'actual_force': data['actual'],
            'coordination_number': data['cn'] if data['cn'] is not None else 0,
            'z_coord': z_coords,
            'layer': layers,
            'strain_pct': strain_label
        })
        all_data.append(df)
        
    if not all_data:
        print("Error: No data successfully loaded.")
        sys.exit(1)
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print("\nGenerating Plots...")
    plot_error_vs_cn(combined_df)
    plot_error_vs_force(combined_df)
    plot_layer_wise_error(combined_df)
    
    print("\n✅ Statistical Analysis Complete!")

if __name__ == "__main__":
    main()
