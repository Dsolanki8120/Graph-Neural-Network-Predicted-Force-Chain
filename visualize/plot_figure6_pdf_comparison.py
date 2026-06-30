#!/usr/bin/env python3
"""
Generate Figure 6 style PDF comparison plots: 2x2 layout
- 6a: DEM Linear Scale
- 6b: GNN Linear Scale  
- 6c: DEM Semi-Log Scale
- 6d: GNN Semi-Log Scale

All strain levels (0.5%, 1.5%, 3.0%, 4.5%, 6.0%, 7.5%, 8.0%) plotted together on each axis.
Averaged across test assemblies 58-72.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import vtk

# Configuration
ERROR_VTK_DIR = "Error_VTK_particles"
ASSEMBLY_MIN = 58
ASSEMBLY_MAX = 72
NBINS = 100

# All strain levels to plot
STRAIN_LEVELS = [0.5, 1.5, 3.0, 4.5, 6.0, 7.5, 8.0]

# Color scheme for strains
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

def read_vtk_polydata(filepath):
    """Read VTK PolyData file and extract force values."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    polydata = reader.GetOutput()
    
    num_arrays = polydata.GetPointData().GetNumberOfArrays()
    
    actual_npmncf = None
    predicted_npmncf = None
    
    for i in range(num_arrays):
        array = polydata.GetPointData().GetArray(i)
        array_name = array.GetName() if array.GetName() else f"Array_{i}"
        data = np.array([array.GetValue(j) for j in range(array.GetNumberOfTuples())])
        
        if 'actual' in array_name.lower():
            actual_npmncf = data
        elif 'predicted' in array_name.lower():
            predicted_npmncf = data
    
    if actual_npmncf is None or predicted_npmncf is None:
        if polydata.GetPointData().GetNumberOfArrays() >= 2:
            actual_npmncf = np.array([polydata.GetPointData().GetArray(0).GetValue(j) 
                                     for j in range(polydata.GetPointData().GetArray(0).GetNumberOfTuples())])
            predicted_npmncf = np.array([polydata.GetPointData().GetArray(1).GetValue(j) 
                                        for j in range(polydata.GetPointData().GetArray(1).GetNumberOfTuples())])
    
    return actual_npmncf, predicted_npmncf

def extract_assembly_and_timestep(filename):
    """Extract assembly ID and timestep from filename."""
    match = re.search(r'graph(\d+)_t(\d+)', filename)
    if match:
        assembly_id = int(match.group(1))
        timestep = int(match.group(2))
        return assembly_id, timestep
    return None, None

def find_all_vtk_files():
    """Find all VTK files in Error_VTK_particles directory."""
    vtk_files = []
    if os.path.exists(ERROR_VTK_DIR):
        for file in os.listdir(ERROR_VTK_DIR):
            if file.endswith('.vtk'):
                vtk_files.append(os.path.join(ERROR_VTK_DIR, file))
    return sorted(vtk_files)

def timestep_to_strain(timestep):
    """Convert timestep to axial strain percentage."""
    return timestep * 0.1

def calculate_pdf(values, nbins=100):
    """Calculate PDF using histogram normalization."""
    values_clean = values[~np.isnan(values)]
    if len(values_clean) == 0:
        return None, None, None
    
    hist, bin_edges = np.histogram(values_clean, bins=nbins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    pdf = hist / (len(values_clean) * bin_width)
    
    return bin_centers, pdf, len(values_clean)

def process_strain_data(strain_level, tolerance=0.05):
    """
    Process all test assemblies for a given strain level.
    Returns actual and predicted force distributions.
    """
    vtk_files = find_all_vtk_files()
    strain_data_actual = []
    strain_data_predicted = []
    
    for vtk_file in vtk_files:
        filename = os.path.basename(vtk_file)
        assembly_id, timestep = extract_assembly_and_timestep(filename)
        
        if assembly_id is None or timestep is None:
            continue
        
        if assembly_id < ASSEMBLY_MIN or assembly_id > ASSEMBLY_MAX:
            continue
        
        file_strain = timestep_to_strain(timestep)
        if abs(file_strain - strain_level) > tolerance:
            continue
        
        try:
            actual, predicted = read_vtk_polydata(vtk_file)
            if actual is not None and predicted is not None:
                strain_data_actual.extend(actual)
                strain_data_predicted.extend(predicted)
        except Exception as e:
            continue
    
    return np.array(strain_data_actual), np.array(strain_data_predicted)

def create_figure6_plots():
    """Create 2x2 Figure 6 style plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax_6a = axes[0, 0]  # DEM Linear
    ax_6b = axes[0, 1]  # GNN Linear
    ax_6c = axes[1, 0]  # DEM Semi-Log
    ax_6d = axes[1, 1]  # GNN Semi-Log
    
    print(f"Processing {len(STRAIN_LEVELS)} strain levels...\n")
    
    strain_data_all = {}
    
    # Process all strains first
    for strain_level in STRAIN_LEVELS:
        print(f"Strain {strain_level}%...", end="", flush=True)
        actual_data, predicted_data = process_strain_data(strain_level)
        
        if len(actual_data) == 0 or len(predicted_data) == 0:
            print(f" SKIPPED (no data)")
            continue
        
        strain_data_all[strain_level] = {
            'actual': actual_data,
            'predicted': predicted_data,
            'actual_count': len(actual_data),
            'predicted_count': len(predicted_data)
        }
        print(f" OK ({len(actual_data)} particles)")
    
    # Plot on all four axes
    print("\nPlotting on axes...")
    
    # Track max PDF values for linear scale synchronization
    max_pdf_dem_linear = 0
    max_pdf_gnn_linear = 0
    
    # First pass: collect PDF data and find max values
    pdf_data_cache = {}
    for idx, strain_level in enumerate(STRAIN_LEVELS):
        if strain_level not in strain_data_all:
            continue
        
        data = strain_data_all[strain_level]
        
        # DEM data
        bins_dem, pdf_dem, _ = calculate_pdf(data['actual'], NBINS)
        # GNN data
        bins_gnn, pdf_gnn, _ = calculate_pdf(data['predicted'], NBINS)
        
        if bins_dem is not None and pdf_dem is not None:
            max_pdf_dem_linear = max(max_pdf_dem_linear, np.max(pdf_dem))
        
        if bins_gnn is not None and pdf_gnn is not None:
            max_pdf_gnn_linear = max(max_pdf_gnn_linear, np.max(pdf_gnn))
        
        pdf_data_cache[strain_level] = {
            'bins_dem': bins_dem,
            'pdf_dem': pdf_dem,
            'bins_gnn': bins_gnn,
            'pdf_gnn': pdf_gnn
        }
    
    # Second pass: plot with shared max values
    for idx, strain_level in enumerate(STRAIN_LEVELS):
        if strain_level not in pdf_data_cache:
            continue
        
        color = COLORS[idx]
        cached = pdf_data_cache[strain_level]
        
        # ========== 6a: DEM Linear ==========
        if cached['bins_dem'] is not None:
            ax_6a.plot(cached['bins_dem'], cached['pdf_dem'], color=color, linewidth=2.0, 
                      marker='o', markersize=5, markevery=4,
                      label=f'{strain_level}%', alpha=0.85)
        
        # ========== 6b: GNN Linear ==========
        if cached['bins_gnn'] is not None:
            ax_6b.plot(cached['bins_gnn'], cached['pdf_gnn'], color=color, linewidth=2.0, 
                      marker='s', markersize=5, markevery=4,
                      label=f'{strain_level}%', alpha=0.85)
        
        # ========== 6c: DEM Semi-Log ==========
        if cached['bins_dem'] is not None:
            ax_6c.semilogy(cached['bins_dem'], cached['pdf_dem'], color=color, linewidth=2.0,
                          marker='o', markersize=5, markevery=4,
                          label=f'{strain_level}%', alpha=0.85)
        
        # ========== 6d: GNN Semi-Log ==========
        if cached['bins_gnn'] is not None:
            ax_6d.semilogy(cached['bins_gnn'], cached['pdf_gnn'], color=color, linewidth=2.0,
                          marker='s', markersize=5, markevery=4,
                          label=f'{strain_level}%', alpha=0.85)
    
    # ========== Format 6a: DEM Linear ==========
    ax_6a.set_xlabel('Normalized Particle Maximum Normal Contact Force (NPMNCF)', 
                    fontsize=11, fontweight='bold')
    ax_6a.set_ylabel('Probability Density Function (PDF)', fontsize=11, fontweight='bold')
    ax_6a.set_title('(a) DEM - Linear Scale', fontsize=12, fontweight='bold', loc='left')
    ax_6a.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax_6a.legend(fontsize=9, loc='upper right', ncol=2, framealpha=0.92)
    ax_6a.tick_params(labelsize=10)
    # Set y-axis limit to max of both DEM and GNN for comparison
    ax_6a.set_ylim(0, max(max_pdf_dem_linear, max_pdf_gnn_linear) * 1.1)
    
    # ========== Format 6b: GNN Linear ==========
    ax_6b.set_xlabel('Normalized Particle Maximum Normal Contact Force (NPMNCF)', 
                    fontsize=11, fontweight='bold')
    ax_6b.set_ylabel('Probability Density Function (PDF)', fontsize=11, fontweight='bold')
    ax_6b.set_title('(b) GNN - Linear Scale', fontsize=12, fontweight='bold', loc='left')
    ax_6b.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax_6b.legend(fontsize=9, loc='upper right', ncol=2, framealpha=0.92)
    ax_6b.tick_params(labelsize=10)
    # Set same y-axis limit as 6a for comparison
    ax_6b.set_ylim(0, max(max_pdf_dem_linear, max_pdf_gnn_linear) * 1.1)
    
    # ========== Format 6c: DEM Semi-Log ==========
    ax_6c.set_xlabel('Normalized Particle Maximum Normal Contact Force (NPMNCF)', 
                    fontsize=11, fontweight='bold')
    ax_6c.set_ylabel('Probability Density Function (PDF, log scale)', fontsize=11, fontweight='bold')
    ax_6c.set_title('(c) DEM - Semi-Log Scale', fontsize=12, fontweight='bold', loc='left')
    ax_6c.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, which='both')
    ax_6c.legend(fontsize=9, loc='upper right', ncol=2, framealpha=0.92)
    ax_6c.tick_params(labelsize=10)
    
    # ========== Format 6d: GNN Semi-Log ==========
    ax_6d.set_xlabel('Normalized Particle Maximum Normal Contact Force (NPMNCF)', 
                    fontsize=11, fontweight='bold')
    ax_6d.set_ylabel('Probability Density Function (PDF, log scale)', fontsize=11, fontweight='bold')
    ax_6d.set_title('(d) GNN - Semi-Log Scale', fontsize=12, fontweight='bold', loc='left')
    ax_6d.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, which='both')
    ax_6d.legend(fontsize=9, loc='upper right', ncol=2, framealpha=0.92)
    ax_6d.tick_params(labelsize=10)
    
    # Overall title
    fig.suptitle('PDF Distribution Comparison: DEM vs GNN Predictions ', 
                fontsize=14, fontweight='bold', y=0.995)
    
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_file = "Figure6_PDF_Comparison_DEM_vs_GNN_Assembly58-72.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_file}")
    
    plt.close(fig)
    
    # Print summary
    print("\n" + "="*70)
    print("FIGURE 6 SUMMARY")
    print("="*70)
    for strain_level in sorted(strain_data_all.keys()):
        data = strain_data_all[strain_level]
        print(f"  {strain_level}%: DEM={data['actual_count']} particles, GNN={data['predicted_count']} particles")
    print("="*70)

if __name__ == "__main__":
    print("=" * 70)
    print("Generating Figure 6: PDF Comparison (2x2 Layout)")
    print(f"Test Assemblies: {ASSEMBLY_MIN}-{ASSEMBLY_MAX}")
    print(f"Strain Levels: {STRAIN_LEVELS}")
    print("=" * 70)
    print()
    
    create_figure6_plots()
    
    print("\n✅ Process complete!")
