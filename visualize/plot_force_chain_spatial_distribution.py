"""
Plot Spatial Distribution of Force Chains (Figure 8 Style)
Shows force chain evolution at XZ and YZ planes for multiple strain levels.
Replicates Figure 8 from: "Estimation of contact forces of granular materials 
under uniaxial compression based on a machine learning model"
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import vtk
from vtk.util import numpy_support


def read_vtk_polydata(vtk_file_path):
    """
    Read VTK polydata file and extract particle positions and forces.
    
    Returns:
        positions: Nx3 array of particle positions
        actual_forces: N array of actual forces (normalized)
        predicted_forces: N array of predicted forces (normalized)
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    
    poly_data = reader.GetOutput()
    
    # Extract point positions
    points = poly_data.GetPoints()
    positions = numpy_support.vtk_to_numpy(points.GetData())
    
    # Extract force arrays
    point_data = poly_data.GetPointData()
    
    actual_forces = numpy_support.vtk_to_numpy(point_data.GetArray('actual_npmncf'))
    predicted_forces = numpy_support.vtk_to_numpy(point_data.GetArray('predicted_npmncf'))
    
    return positions, actual_forces, predicted_forces


def find_vtk_file(vtk_directory, assembly_id, timestep_str):
    """Find VTK file matching the pattern."""
    vtk_dir = Path(vtk_directory)
    pattern = f"particle_error_graph{assembly_id}_{timestep_str}.vtk"
    
    for file in vtk_dir.glob("*.vtk"):
        if pattern in str(file):
            return str(file)
    
    return None


def get_plane_slice(positions, forces, plane='XZ', thickness=0.02):
    """
    Extract particles near a plane passing through centroid (per paper Figure 8).
    
    According to paper: "The XZ and YZ planes contain the centroid of the granular 
    system and are vertical to the y-axis and x-axis, respectively."
    
    - XZ Plane: Contains centroid, vertical (perpendicular) to y-axis
              → Extract particles where |y - centroid_y| < thickness
              → Display X and Z coordinates
    
    - YZ Plane: Contains centroid, vertical (perpendicular) to x-axis
              → Extract particles where |x - centroid_x| < thickness
              → Display Y and Z coordinates
    
    plane: 'XZ' or 'YZ'
    thickness: thickness of slice around center plane
    
    Returns:
        slice_positions: Nx2 coordinates on the plane
        slice_forces: N array of forces
        mask: Boolean mask of selected particles
    """
    centroid = np.mean(positions, axis=0)  # Center of granular system
    
    if plane == 'XZ':
        # XZ plane passes through centroid, perpendicular to y-axis
        # Extract particles near Y = centroid[1]
        centroid_y = centroid[1]
        mask = np.abs(positions[:, 1] - centroid_y) < thickness
        plane_coords = positions[mask][:, [0, 2]]  # X, Z coordinates
    else:  # YZ plane
        # YZ plane passes through centroid, perpendicular to x-axis
        # Extract particles near X = centroid[0]
        centroid_x = centroid[0]
        mask = np.abs(positions[:, 0] - centroid_x) < thickness
        plane_coords = positions[mask][:, [1, 2]]  # Y, Z coordinates
    
    slice_forces = forces[mask]
    
    return plane_coords, slice_forces, mask


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("FORCE CHAIN SPATIAL DISTRIBUTION ANALYSIS (Figure 8 Style)")
    print("="*80 + "\n")
    
    # Configuration
    VTK_DIRECTORY = "Error_VTK_particles_single_run"
    ASSEMBLY_ID = 58
    
    # Check if directory exists
    if not Path(VTK_DIRECTORY).exists():
        alt_dirs = ["Error_VTK_particles", "error_vtk_single_run"]
        for alt_dir in alt_dirs:
            if Path(alt_dir).exists():
                VTK_DIRECTORY = alt_dir
                print(f"✅ Using directory: {VTK_DIRECTORY}\n")
                break
    
    # Define strain levels with correct timesteps
    strain_levels = [
        (15, 1.5),   # t015 = 1.5% strain
        (30, 3.0),   # t030 = 3.0% strain
        (45, 4.5),   # t045 = 4.5% strain
        (60, 6.0),   # t060 = 6.0% strain
        (75, 7.5),   # t075 = 7.5% strain
    ]
    
    print(f"Strain Levels Analysis:")
    for timestep, strain in strain_levels:
        print(f"  Timestep t{timestep:03d} → {strain}% axial strain")
    print()
    
    # Create figure with 5 rows (strains) x 4 columns (XZ actual, XZ predicted, YZ actual, YZ predicted)
    fig, axes = plt.subplots(len(strain_levels), 4, figsize=(16, 4*len(strain_levels)))
    
    fig.suptitle(f'Force Chain Evolution - Assembly {ASSEMBLY_ID}\n(XZ and YZ Planes at Different Strain Levels)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    for row_idx, (timestep, strain) in enumerate(strain_levels):
        ts_str = f"t{timestep:03d}"
        vtk_file = find_vtk_file(VTK_DIRECTORY, ASSEMBLY_ID, ts_str)
        
        if not vtk_file:
            print(f"⚠️  VTK file not found for {ts_str} ({strain}% strain)")
            continue
        
        print(f"[{row_idx+1}/{len(strain_levels)}] Processing {ts_str} ({strain}% strain)...")
        print(f"  File: {vtk_file}")
        
        try:
            positions, actual_forces, predicted_forces = read_vtk_polydata(vtk_file)
            print(f"  ✅ Loaded {len(actual_forces)} particles")
            
            # Get XZ plane slices (reduced thickness for better plane accuracy)
            xz_actual_coords, xz_actual_forces, _ = get_plane_slice(
                positions, actual_forces, plane='XZ', thickness=0.005
            )
            xz_pred_coords, xz_pred_forces, _ = get_plane_slice(
                positions, predicted_forces, plane='XZ', thickness=0.005
            )
            
            # Get YZ plane slices (reduced thickness for better plane accuracy)
            yz_actual_coords, yz_actual_forces, _ = get_plane_slice(
                positions, actual_forces, plane='YZ', thickness=0.005
            )
            yz_pred_coords, yz_pred_forces, _ = get_plane_slice(
                positions, predicted_forces, plane='YZ', thickness=0.005
            )
            
            # Determine force limits for consistent colormap across all plots
            force_min = min(xz_actual_forces.min(), xz_pred_forces.min(), 
                          yz_actual_forces.min(), yz_pred_forces.min())
            force_max = max(xz_actual_forces.max(), xz_pred_forces.max(),
                          yz_actual_forces.max(), yz_pred_forces.max())
            
            print(f"  Force range: [{force_min:.3f}, {force_max:.3f}]")
            print(f"  XZ plane: {len(xz_actual_forces)} particles (actual), {len(xz_pred_forces)} (predicted)")
            print(f"  YZ plane: {len(yz_actual_forces)} particles (actual), {len(yz_pred_forces)} (predicted)\n")
            
            # Plot XZ Actual (column 0)
            ax = axes[row_idx, 0]
            scatter1 = ax.scatter(xz_actual_coords[:, 0], xz_actual_coords[:, 1], 
                                 c=xz_actual_forces, s=50, cmap='hot', 
                                 vmin=force_min, vmax=force_max, alpha=0.8, edgecolors='k', linewidth=0.5)
            ax.set_title(f'{strain}% - XZ Plane (Actual)', fontsize=11, fontweight='bold')
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Z', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            # Plot XZ Predicted (column 1)
            ax = axes[row_idx, 1]
            scatter2 = ax.scatter(xz_pred_coords[:, 0], xz_pred_coords[:, 1], 
                                 c=xz_pred_forces, s=50, cmap='hot', 
                                 vmin=force_min, vmax=force_max, alpha=0.8, edgecolors='k', linewidth=0.5)
            ax.set_title(f'{strain}% - XZ Plane (Predicted)', fontsize=11, fontweight='bold')
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Z', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            # Plot YZ Actual (column 2)
            ax = axes[row_idx, 2]
            scatter3 = ax.scatter(yz_actual_coords[:, 0], yz_actual_coords[:, 1], 
                                 c=yz_actual_forces, s=50, cmap='hot', 
                                 vmin=force_min, vmax=force_max, alpha=0.8, edgecolors='k', linewidth=0.5)
            ax.set_title(f'{strain}% - YZ Plane (Actual)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Y', fontsize=10)
            ax.set_ylabel('Z', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            # Plot YZ Predicted (column 3)
            ax = axes[row_idx, 3]
            scatter4 = ax.scatter(yz_pred_coords[:, 0], yz_pred_coords[:, 1], 
                                 c=yz_pred_forces, s=50, cmap='hot', 
                                 vmin=force_min, vmax=force_max, alpha=0.8, edgecolors='k', linewidth=0.5)
            ax.set_title(f'{strain}% - YZ Plane (Predicted)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Y', fontsize=10)
            ax.set_ylabel('Z', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            # Add colorbar to the last plot in the row
            cbar = plt.colorbar(scatter4, ax=ax, pad=0.02)
            cbar.set_label('NPMNCF', fontsize=9)
        
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"Fig8_Force_Chain_Spatial_Distribution_assembly{ASSEMBLY_ID}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file} (300 DPI)")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nFigure Details:")
    print(f"  • 5 rows: 1.5%, 3.0%, 4.5%, 6.0%, 7.5% strain levels")
    print(f"  • 4 columns: XZ Actual, XZ Predicted, YZ Actual, YZ Predicted")
    print(f"  • Color: Hot colormap representing NPMNCF values")
    print(f"  • Planes: Through center of mass, perpendicular to compression axis")
    print("\n" + "="*80 + "\n")
