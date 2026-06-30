#!/usr/bin/env python3
"""
Export Persistent Error VTK.
Calculates the time-averaged absolute error, actual/predicted NPMNCF, and 
coordination number for each particle across all timesteps
to identify "persistently difficult" regions (Figure 8).
"""

import os
import sys
import glob
from pathlib import Path

import numpy as np

try:
    import vtk
    from vtk.util import numpy_support
except ImportError:
    print("Error: VTK not installed. Please install vtk (pip install vtk)")
    sys.exit(1)

# ===== CONFIGURATION =====
ASSEMBLY_ID = 72
# Try to find the directory containing the single run VTKs
POSSIBLE_DIRS = [
    "../Error_VTK_particles",
    "Error_VTK_particles",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Error_VTK_particles_single_run")
]
OUT_DIR = "Persistent_Error_Output"
# ========================


def find_vtk_directory():
    for d in POSSIBLE_DIRS:
        if os.path.exists(d):
            return d
    return None


def read_vtk_polydata(vtk_file_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    return reader.GetOutput()


def add_point_array(polydata, values, name, array_type=vtk.VTK_FLOAT):
    arr = numpy_support.numpy_to_vtk(values, deep=True, array_type=array_type)
    arr.SetName(name)
    polydata.GetPointData().AddArray(arr)
    return arr


def main():
    vtk_dir = find_vtk_directory()
    if not vtk_dir:
        print("Error: Could not find 'Error_VTK_particles_single_run' directory.")
        print("Run export_single_test_error_vtk.py first.")
        sys.exit(1)

    vtk_files = sorted(glob.glob(os.path.join(vtk_dir, f"particle_error_graph{ASSEMBLY_ID:02d}_*.vtk")))
    if not vtk_files:
        print(f"Error: No VTK files found for Assembly {ASSEMBLY_ID} in {vtk_dir}")
        sys.exit(1)

    print(f"Found {len(vtk_files)} timesteps for Assembly {ASSEMBLY_ID}.")

    # Dictionaries to accumulate per-particle data across timesteps
    particle_error_sum = {}
    particle_actual_sum = {}
    particle_predicted_sum = {}
    particle_cn_sum = {}
    particle_counts = {}
    particle_positions = {}
    
    print("Accumulating data across timesteps...")
    for i, file_path in enumerate(vtk_files):
        if i % 10 == 0:
            print(f"  Processing file {i+1}/{len(vtk_files)}: {os.path.basename(file_path)}")
        
        polydata = read_vtk_polydata(file_path)
        points = polydata.GetPoints()
        if not points:
            continue
            
        positions = numpy_support.vtk_to_numpy(points.GetData())
        point_data = polydata.GetPointData()
        
        pids_arr = point_data.GetArray("particle_id")
        errs_arr = point_data.GetArray("particle_force_absolute_error")
        actual_arr = point_data.GetArray("actual_npmncf")
        predicted_arr = point_data.GetArray("predicted_npmncf")
        cn_arr = point_data.GetArray("coordination_number")
        
        if not pids_arr or not errs_arr:
            continue
            
        pids = numpy_support.vtk_to_numpy(pids_arr)
        errs = numpy_support.vtk_to_numpy(errs_arr)
        
        actuals = numpy_support.vtk_to_numpy(actual_arr) if actual_arr else None
        predicteds = numpy_support.vtk_to_numpy(predicted_arr) if predicted_arr else None
        cns = numpy_support.vtk_to_numpy(cn_arr) if cn_arr else None

        for idx, pid in enumerate(pids):
            pid = int(pid)
            particle_error_sum[pid] = particle_error_sum.get(pid, 0.0) + errs[idx]
            particle_counts[pid] = particle_counts.get(pid, 0) + 1
            if actuals is not None:
                particle_actual_sum[pid] = particle_actual_sum.get(pid, 0.0) + actuals[idx]
            if predicteds is not None:
                particle_predicted_sum[pid] = particle_predicted_sum.get(pid, 0.0) + predicteds[idx]
            if cns is not None:
                particle_cn_sum[pid] = particle_cn_sum.get(pid, 0.0) + cns[idx]
            # Store latest position (positions change slightly with strain)
            particle_positions[pid] = positions[idx]

    if not particle_positions:
        print("Error: No particles found.")
        sys.exit(1)

    # Compute time-averaged values
    pids_sorted = sorted(particle_positions.keys())
    avg_errors = np.array([particle_error_sum[p] / particle_counts[p] for p in pids_sorted], dtype=np.float32)
    avg_actual = np.array([particle_actual_sum.get(p, 0.0) / particle_counts[p] for p in pids_sorted], dtype=np.float32)
    avg_predicted = np.array([particle_predicted_sum.get(p, 0.0) / particle_counts[p] for p in pids_sorted], dtype=np.float32)
    avg_cn = np.array([particle_cn_sum.get(p, 0.0) / particle_counts[p] for p in pids_sorted], dtype=np.float32)
    cn_rounded = np.round(avg_cn).astype(np.int32)
    counts = np.array([particle_counts[p] for p in pids_sorted], dtype=np.int32)
    
    # Compute derived fields from time-averaged values
    avg_signed_error = avg_predicted - avg_actual
    avg_relative_error = np.abs(avg_signed_error) / (np.abs(avg_actual) + 1e-6)
    
    # Print summary statistics
    print(f"\n--- Summary Statistics ---")
    print(f"  Particles:        {len(pids_sorted)}")
    print(f"  Avg abs error:    {avg_errors.mean():.4f} (std={avg_errors.std():.4f})")
    print(f"  Avg actual NPMNCF:    {avg_actual.mean():.4f}")
    print(f"  Avg predicted NPMNCF: {avg_predicted.mean():.4f}")
    print(f"  CN distribution:  min={cn_rounded.min()}, max={cn_rounded.max()}, mean={avg_cn.mean():.1f}")
    
    # Create new polydata
    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(len(pids_sorted))
    cells = vtk.vtkCellArray()
    
    for i, pid in enumerate(pids_sorted):
        pos = particle_positions[pid]
        pts.SetPoint(i, float(pos[0]), float(pos[1]), float(pos[2]))
        
        cell = vtk.vtkVertex()
        cell.GetPointIds().SetId(0, i)
        cells.InsertNextCell(cell)

    out_polydata = vtk.vtkPolyData()
    out_polydata.SetPoints(pts)
    out_polydata.SetVerts(cells)

    # Add arrays — all with real time-averaged data
    add_point_array(out_polydata, np.array(pids_sorted, dtype=np.int32), "particle_id", vtk.VTK_INT)
    add_point_array(out_polydata, avg_errors, "persistent_absolute_error", vtk.VTK_FLOAT)
    add_point_array(out_polydata, avg_actual, "avg_actual_npmncf", vtk.VTK_FLOAT)
    add_point_array(out_polydata, avg_predicted, "avg_predicted_npmncf", vtk.VTK_FLOAT)
    add_point_array(out_polydata, avg_signed_error.astype(np.float32), "avg_signed_error", vtk.VTK_FLOAT)
    add_point_array(out_polydata, avg_relative_error.astype(np.float32), "avg_relative_error", vtk.VTK_FLOAT)
    add_point_array(out_polydata, cn_rounded, "coordination_number", vtk.VTK_INT)
    add_point_array(out_polydata, counts, "timestep_count", vtk.VTK_INT)
    
    # Set active scalar
    out_polydata.GetPointData().SetScalars(out_polydata.GetPointData().GetArray("persistent_absolute_error"))

    # Ensure output dir exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    out_path = os.path.join(project_dir, OUT_DIR)
    os.makedirs(out_path, exist_ok=True)
    
    out_file = os.path.join(out_path, f"persistent_error_graph{ASSEMBLY_ID:02d}.vtk")
    
    print(f"\nWriting persistent error to: {out_file}")
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_file)
    writer.SetInputData(out_polydata)
    writer.Write()
    
    print(f"✅ Export Complete! Processed {len(pids_sorted)} unique particles.")



if __name__ == "__main__":
    main()
