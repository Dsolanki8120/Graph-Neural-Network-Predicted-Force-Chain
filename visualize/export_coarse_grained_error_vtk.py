#!/usr/bin/env python3
"""
Export coarse-grained error VTK files for a single graph (assembly).
Applies hard kernel spatial binning to reduce data and smooth error field.
Processes all timesteps for the specified assembly.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import vtk
    from vtk.util import numpy_support
    from scipy.interpolate import griddata
except Exception:
    print("Error: VTK or scipy not installed")
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR = "data_sets(0-72)"
INFERENCE_DIR_TRAIN = "New_inference_results_train"
INFERENCE_DIR_TEST = "New_inference_results_test"

OUT_EP_COARSE = "Error_VTK_particles_coarse_grained"
OUT_EP_CONTINUOUS = "Error_VTK_particles_continuous"

EPS = 1e-6

# ===== CONFIGURATION =====
GRAPH_ID = 58                           # Change to 1-73 (1-57=train, 58-73=test)
START_TIMESTEP = 0                      # Change as needed (0-80)
END_TIMESTEP = 80                       # 0-80 = 81 total timesteps
COARSE_GRID_CELL_SIZE = 10              # Size of coarse cells (try 5, 10, 15, 20, 25)
AGGREGATION_METHOD = "mean"             # Options: "mean", "max" (for error peaks)
FINE_GRID_CELL_SIZE = 2                 # Fine interpolated grid cell size (smaller = higher resolution, but slower)
# ========================


def resolve_project_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_DIR, path)


DATA_PATH = resolve_project_path(DATA_DIR)

# Determine split and inference directory
if 1 <= GRAPH_ID <= 57:
    SPLIT = "train"
    INFERENCE_DIR_FALLBACKS = [INFERENCE_DIR_TRAIN, "New_inference_results_train"]
elif 58 <= GRAPH_ID <= 73:
    SPLIT = "test"
    INFERENCE_DIR_FALLBACKS = [INFERENCE_DIR_TEST, "New_inference_results_test"]
else:
    print(f"ERROR: Invalid GRAPH_ID {GRAPH_ID}. Must be 1-73")
    sys.exit(1)

INFERENCE_PATH = None
for candidate in INFERENCE_DIR_FALLBACKS:
    path = resolve_project_path(candidate)
    if os.path.isdir(path):
        INFERENCE_PATH = path
        break

if INFERENCE_PATH is None:
    print(f"ERROR: Could not find inference directory")
    sys.exit(1)

print(f"✓ Using inference path: {INFERENCE_PATH}", flush=True)


def read_tab_file(fp):
    """Read tab-separated data file."""
    data = {}
    try:
        with open(fp) as f:
            for i, line in enumerate(f):
                if i < 2:
                    continue
                parts = re.split(r"[\s,]+", line.strip())
                if len(parts) >= 2 and parts[0]:
                    data[int(float(parts[0]))] = float(parts[1])
    except Exception:
        pass
    return data


def load_coordination_number(gid, ts):
    """Load coordination number with multiple format support."""
    cn = {}
    
    cn_file_patterns = [
        f"{gid}_{ts}_ball_cn.tab",
        f"graph_{gid}_{ts}_cn.tab",
        f"{gid}_{ts}_cn.tab",
        f"graph_{gid:02d}_{ts:02d}_cn.tab",
        f"{gid:02d}_{ts:02d}_cn.tab",
        f"{gid:02d}_{ts:02d}_ball_cn.tab",
        f"graph_{gid:02d}_{ts:03d}_cn.tab",
        f"{gid:02d}_{ts:03d}_cn.tab",
        f"{gid:02d}_{ts:03d}_ball_cn.tab",
    ]
    
    for pattern in cn_file_patterns:
        fp = os.path.join(DATA_PATH, pattern)
        if os.path.exists(fp):
            try:
                with open(fp) as f:
                    for i, line in enumerate(f):
                        if i < 2:
                            continue
                        line = line.strip()
                        if not line:
                            continue
                        parts = re.split(r"[\s,]+", line)
                        if len(parts) >= 2 and parts[0]:
                            try:
                                ball_id = int(float(parts[0]))
                                coord_num = int(float(parts[1]))
                                cn[ball_id] = coord_num
                            except (ValueError, IndexError):
                                continue
                if cn:
                    return cn
            except Exception:
                pass
    
    return cn


def load_particle_data(gid, ts):
    """Load particle positions and displacements."""
    prefix = f"{gid}_{ts}"
    px = read_tab_file(os.path.join(DATA_PATH, f"{prefix}_ball_pos_x.tab"))
    py = read_tab_file(os.path.join(DATA_PATH, f"{prefix}_ball_pos_y.tab"))
    pz = read_tab_file(os.path.join(DATA_PATH, f"{prefix}_ball_pos_z.tab"))
    pos = {
        pid: np.array([px[pid], py[pid], pz[pid]], dtype=np.float32)
        for pid in px
        if pid in py and pid in pz
    }

    dx = read_tab_file(os.path.join(DATA_PATH, f"{prefix}_ball_disp_x.tab"))
    dy = read_tab_file(os.path.join(DATA_PATH, f"{prefix}_ball_disp_y.tab"))
    dz = read_tab_file(os.path.join(DATA_PATH, f"{prefix}_ball_disp_z.tab"))
    dis = {
        pid: np.array([dx[pid], dy[pid], dz[pid]], dtype=np.float32)
        for pid in dx
        if pid in dy and pid in dz
    }

    cn = load_coordination_number(gid, ts)
    
    return pos, dis, cn


def load_pred_particle_forces(gid, ts):
    """Load predicted and actual NPMNCF from CSV."""
    file_candidates = [
        os.path.join(INFERENCE_PATH, "predicted_npmncf_files", f"assembly_{gid:02d}_timestep_{ts:02d}_predicted_npmncf.csv"),
        os.path.join(INFERENCE_PATH, "predicted_npmncf_files", f"assembly_{gid:02d}_timestep_{ts:03d}_predicted_npmncf.csv"),
        os.path.join(INFERENCE_PATH, "predicted_npmncf_files", f"assembly_{gid:02d}_timestep_{ts}_predicted_npmncf.csv"),
    ]
    
    for fp in file_candidates:
        if os.path.exists(fp):
            predicted, actual = {}, {}
            try:
                df = pd.read_csv(fp)
                for _, row in df.iterrows():
                    pid = int(row["ball_id"])
                    predicted[pid] = float(row["predicted_npmncf"])
                    if "actual_npmncf" in df.columns:
                        actual[pid] = float(row["actual_npmncf"])
                return predicted, actual
            except Exception:
                return {}, {}
    
    return {}, {}


def add_point_array(polydata, values, name, array_type=vtk.VTK_FLOAT):
    """Add array to polydata points."""
    arr = numpy_support.numpy_to_vtk(values, deep=True, array_type=array_type)
    arr.SetName(name)
    polydata.GetPointData().AddArray(arr)
    return arr


def make_coarse_polydata(coarse_pos):
    """Create VTK polydata from coarse-grained positions."""
    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(len(coarse_pos))
    cell_indices = sorted(coarse_pos.keys())
    
    for i, cell_idx in enumerate(cell_indices):
        pp = coarse_pos[cell_idx]
        pts.SetPoint(i, float(pp[0]), float(pp[1]), float(pp[2]))
    
    cells = vtk.vtkCellArray()
    for i in range(len(coarse_pos)):
        cell = vtk.vtkVertex()
        cell.GetPointIds().SetId(0, i)
        cells.InsertNextCell(cell)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pts)
    polydata.SetVerts(cells)
    
    return polydata, cell_indices


def write_polydata(polydata, out, label):
    """Write polydata to VTK file."""
    print(f"  → exporting {label}: {out}", flush=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out)
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        raise RuntimeError(f"VTK writer failed for {out}")
    print(f"  ✓ exported {label}", flush=True)


def coarse_grain_error_field(pos, actual_forces, predicted_forces, coord_num, 
                             grid_cell_size=10, aggregation="mean"):
    """
    Hard kernel spatial binning coarse graining for error field.
    
    Assigns each particle to exactly ONE grid cell based on floor division.
    Aggregates forces and errors within each cell.
    
    Args:
        pos: dict {particle_id: [x, y, z]}
        actual_forces: dict {particle_id: force_value}
        predicted_forces: dict {particle_id: force_value}
        coord_num: dict {particle_id: coordination_number}
        grid_cell_size: size of coarse cells
        aggregation: "mean" or "max" (max for error peaks)
    
    Returns:
        Dictionary with coarse-grained data indexed by cell
    """
    cells = {}
    
    # Step 1: Assign particles to grid cells (hard kernel)
    for pid, particle_pos in pos.items():
        cell_idx = tuple(np.floor(particle_pos / grid_cell_size).astype(int))
        if cell_idx not in cells:
            cells[cell_idx] = []
        cells[cell_idx].append(pid)
    
    # Step 2: Aggregate data for each cell
    coarse_data = {}
    
    for cell_idx, particle_ids in cells.items():
        # Average position
        positions = np.array([pos[pid] for pid in particle_ids])
        cell_pos = np.mean(positions, axis=0)
        
        # Aggregate forces
        actual_vals = np.array([actual_forces.get(pid, 0.0) for pid in particle_ids])
        predicted_vals = np.array([predicted_forces.get(pid, 0.0) for pid in particle_ids])
        
        if aggregation == "mean":
            cell_actual = np.mean(actual_vals)
            cell_predicted = np.mean(predicted_vals)
        elif aggregation == "max":  # For error peaks
            cell_actual = np.max(np.abs(actual_vals))
            cell_predicted = np.max(np.abs(predicted_vals))
        else:
            cell_actual = np.mean(actual_vals)
            cell_predicted = np.mean(predicted_vals)
        
        # Calculate errors
        signed_error = predicted_vals - actual_vals
        absolute_error = np.abs(signed_error)
        relative_error = absolute_error / (np.abs(actual_vals) + EPS)
        
        if aggregation == "mean":
            cell_signed_error = np.mean(signed_error)
            cell_absolute_error = np.mean(absolute_error)
            cell_relative_error = np.mean(relative_error)
        else:  # max
            cell_signed_error = signed_error[np.argmax(absolute_error)]
            cell_absolute_error = np.max(absolute_error)
            cell_relative_error = np.max(relative_error)
        
        cell_relative_error_percent = cell_relative_error * 100.0
        
        # Coordination number
        cn_vals = np.array([coord_num.get(pid, 0) for pid in particle_ids])
        cell_cn = int(np.mean(cn_vals))
        
        # Store all data for this cell
        coarse_data[cell_idx] = {
            'pos': cell_pos,
            'actual': cell_actual,
            'predicted': cell_predicted,
            'signed_error': cell_signed_error,
            'absolute_error': cell_absolute_error,
            'relative_error': cell_relative_error,
            'relative_error_percent': cell_relative_error_percent,
            'coordination_number': cell_cn,
            'particle_count': len(particle_ids),
        }
    
    return coarse_data


def export_coarse_grained_error(coarse_data, out, label, grid_cell_size):
    """Export coarse-grained error data to VTK."""
    if not coarse_data:
        return False
    
    try:
        # Extract data in consistent order
        cell_indices = sorted(coarse_data.keys())
        coarse_positions = np.array([coarse_data[idx]['pos'] for idx in cell_indices], dtype=np.float32)
        
        # Create polydata
        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(len(cell_indices))
        for i, pos in enumerate(coarse_positions):
            pts.SetPoint(i, float(pos[0]), float(pos[1]), float(pos[2]))
        
        cells = vtk.vtkCellArray()
        for i in range(len(cell_indices)):
            cell = vtk.vtkVertex()
            cell.GetPointIds().SetId(0, i)
            cells.InsertNextCell(cell)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(pts)
        polydata.SetVerts(cells)
        
        # Add arrays
        actual_arr = np.array([coarse_data[idx]['actual'] for idx in cell_indices], dtype=np.float32)
        predicted_arr = np.array([coarse_data[idx]['predicted'] for idx in cell_indices], dtype=np.float32)
        signed_error_arr = np.array([coarse_data[idx]['signed_error'] for idx in cell_indices], dtype=np.float32)
        absolute_error_arr = np.array([coarse_data[idx]['absolute_error'] for idx in cell_indices], dtype=np.float32)
        relative_error_arr = np.array([coarse_data[idx]['relative_error'] for idx in cell_indices], dtype=np.float32)
        relative_error_percent_arr = np.array([coarse_data[idx]['relative_error_percent'] for idx in cell_indices], dtype=np.float32)
        cn_arr = np.array([coarse_data[idx]['coordination_number'] for idx in cell_indices], dtype=np.int32)
        particle_count_arr = np.array([coarse_data[idx]['particle_count'] for idx in cell_indices], dtype=np.int32)
        
        # Add to polydata
        add_point_array(polydata, actual_arr, "coarse_actual_npmncf")
        add_point_array(polydata, predicted_arr, "coarse_predicted_npmncf")
        add_point_array(polydata, signed_error_arr, "coarse_signed_error")
        add_point_array(polydata, absolute_error_arr, "coarse_absolute_error")
        add_point_array(polydata, relative_error_arr, "coarse_relative_error")
        add_point_array(polydata, relative_error_percent_arr, "coarse_relative_error_percent")
        add_point_array(polydata, cn_arr, "coarse_avg_coordination_number", vtk.VTK_INT)
        add_point_array(polydata, particle_count_arr, "coarse_particles_in_cell", vtk.VTK_INT)
        
        # Set scalars
        polydata.GetPointData().SetScalars(
            polydata.GetPointData().GetArray("coarse_absolute_error")
        )
        
        # Calculate and print statistics
        n_over = np.sum(signed_error_arr > EPS)
        n_under = np.sum(signed_error_arr < -EPS)
        n_perfect = np.sum(np.abs(signed_error_arr) <= EPS)
        n_cells = len(cell_indices)
        mean_signed = np.mean(signed_error_arr)
        mean_abs = np.mean(absolute_error_arr)
        total_particles = np.sum(particle_count_arr)
        
        print(f"  Coarse Stats: {n_cells} cells (from {total_particles} particles, cell_size={grid_cell_size})", flush=True)
        print(f"         Over={n_over:5d} ({100*n_over/n_cells:5.1f}%) | Under={n_under:5d} ({100*n_under/n_cells:5.1f}%) | Perfect={n_perfect:5d} ({100*n_perfect/n_cells:5.1f}%)", flush=True)
        print(f"         Mean_Signed={mean_signed:8.5f} | Mean_Abs={mean_abs:8.5f}", flush=True)
        
        write_polydata(polydata, out, f"{label} (coarse-grained, grid={grid_cell_size})")
        return True
    except Exception as e:
        print(f"  ✗ {label} error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def export_continuous_error_field_imagedata(coarse_data, out, label, fine_grid_cell_size):
    """
    Export continuous interpolated error field as VTK ImageData (structured grid).
    
    Interpolates coarse-grained data to a fine regular grid using linear interpolation.
    Perfect for visualizing continuous force fields in granular materials.
    
    Args:
        coarse_data: dict {cell_idx: {pos, actual, predicted, errors, ...}}
        out: output file path
        label: description label
        fine_grid_cell_size: size of fine grid cells (smaller = higher resolution)
    """
    if not coarse_data:
        return False
    
    try:
        # Extract coarse points and scalar data
        coarse_points = np.array([coarse_data[idx]['pos'] for idx in sorted(coarse_data.keys())], dtype=np.float32)
        
        # Get scalar fields to interpolate
        scalar_fields = {
            'actual_npmncf': np.array([coarse_data[idx]['actual'] for idx in sorted(coarse_data.keys())], dtype=np.float32),
            'predicted_npmncf': np.array([coarse_data[idx]['predicted'] for idx in sorted(coarse_data.keys())], dtype=np.float32),
            'signed_error': np.array([coarse_data[idx]['signed_error'] for idx in sorted(coarse_data.keys())], dtype=np.float32),
            'absolute_error': np.array([coarse_data[idx]['absolute_error'] for idx in sorted(coarse_data.keys())], dtype=np.float32),
            'relative_error': np.array([coarse_data[idx]['relative_error'] for idx in sorted(coarse_data.keys())], dtype=np.float32),
            'relative_error_percent': np.array([coarse_data[idx]['relative_error_percent'] for idx in sorted(coarse_data.keys())], dtype=np.float32),
            'coordination_number': np.array([float(coarse_data[idx]['coordination_number']) for idx in sorted(coarse_data.keys())], dtype=np.float32),
        }
        
        # Create fine grid
        min_coords = np.min(coarse_points, axis=0)
        max_coords = np.max(coarse_points, axis=0)
        
        # Add padding (10% of range) to avoid edge effects
        padding = (max_coords - min_coords) * 0.1
        grid_min = min_coords - padding
        grid_max = max_coords + padding
        
        # Create grid points
        x = np.arange(grid_min[0], grid_max[0] + fine_grid_cell_size, fine_grid_cell_size, dtype=np.float32)
        y = np.arange(grid_min[1], grid_max[1] + fine_grid_cell_size, fine_grid_cell_size, dtype=np.float32)
        z = np.arange(grid_min[2], grid_max[2] + fine_grid_cell_size, fine_grid_cell_size, dtype=np.float32)
        
        nx, ny, nz = len(x), len(y), len(z)
        
        # Create mesh grid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        fine_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Interpolate all scalar fields using linear interpolation
        interpolated_fields = {}
        for field_name, field_values in scalar_fields.items():
            try:
                # Linear interpolation
                interp_values = griddata(
                    coarse_points, 
                    field_values, 
                    fine_points, 
                    method='linear',
                    fill_value=np.nanmean(field_values)  # Fill NaNs with mean
                )
                # Replace any NaNs with nearest neighbor fallback
                if np.any(np.isnan(interp_values)):
                    nan_mask = np.isnan(interp_values)
                    fallback = griddata(
                        coarse_points,
                        field_values,
                        fine_points[nan_mask],
                        method='nearest'
                    )
                    interp_values[nan_mask] = fallback
                interpolated_fields[field_name] = interp_values.reshape(nx, ny, nz)
            except Exception as e:
                print(f"    Warning: Interpolation failed for {field_name}: {e}", flush=True)
                # Use nearest neighbor fallback
                nn_values = griddata(coarse_points, field_values, fine_points, method='nearest')
                interpolated_fields[field_name] = nn_values.reshape(nx, ny, nz)
        
        # Create VTK ImageData
        image = vtk.vtkImageData()
        
        # Set dimensions (nx, ny, nz)
        image.SetDimensions(nx, ny, nz)
        
        # Set spacing (grid cell size)
        image.SetSpacing(fine_grid_cell_size, fine_grid_cell_size, fine_grid_cell_size)
        
        # Set origin
        image.SetOrigin(float(grid_min[0]), float(grid_min[1]), float(grid_min[2]))
        
        # Add interpolated scalar arrays
        for field_name, field_data in interpolated_fields.items():
            arr = numpy_support.numpy_to_vtk(field_data.ravel().astype(np.float32), deep=True)
            arr.SetName(field_name)
            image.GetPointData().AddArray(arr)
        
        # Set absolute_error as default scalar
        if 'absolute_error' in interpolated_fields:
            arr = image.GetPointData().GetArray('absolute_error')
            image.GetPointData().SetScalars(arr)
        
        # Write to file
        print(f"  → exporting {label}: {out}", flush=True)
        writer = vtk.vtkImageDataGeometryFilter()
        writer_poly = vtk.vtkPolyDataWriter()
        
        # Actually use vtkStructuredPointsWriter for ImageData
        writer_img = vtk.vtkStructuredPointsWriter()
        writer_img.SetFileName(out)
        writer_img.SetInputData(image)
        
        if writer_img.Write() != 1:
            raise RuntimeError(f"VTK writer failed for {out}")
        
        # Calculate statistics
        abs_error = interpolated_fields.get('absolute_error', None)
        signed_error = interpolated_fields.get('signed_error', None)
        
        if abs_error is not None and signed_error is not None:
            n_over = np.sum(signed_error > EPS)
            n_under = np.sum(signed_error < -EPS)
            n_perfect = np.sum(np.abs(signed_error) <= EPS)
            n_total = signed_error.size
            
            print(f"  Continuous Stats: {nx}×{ny}×{nz}={n_total:,} voxels (cell_size={fine_grid_cell_size})", flush=True)
            print(f"         Over={100*n_over/n_total:5.1f}% | Under={100*n_under/n_total:5.1f}% | Perfect={100*n_perfect/n_total:5.1f}%", flush=True)
            print(f"         Mean_Abs={np.nanmean(abs_error):8.5f} | Max_Abs={np.nanmax(abs_error):8.5f}", flush=True)
        
        print(f"  ✓ exported continuous field", flush=True)
        return True
        
    except Exception as e:
        print(f"  ✗ continuous field error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def main():
    os.makedirs(resolve_project_path(OUT_EP_COARSE), exist_ok=True)
    os.makedirs(resolve_project_path(OUT_EP_CONTINUOUS), exist_ok=True)
    
    print("\n" + "=" * 100, flush=True)
    print(f"CONTINUOUS ERROR FIELD VTK EXPORT - Graph {GRAPH_ID:02d} ({SPLIT})", flush=True)
    print(f"Timesteps: {START_TIMESTEP} to {END_TIMESTEP} (total: {END_TIMESTEP - START_TIMESTEP + 1})", flush=True)
    print(f"Coarse Grid Cell Size: {COARSE_GRID_CELL_SIZE}", flush=True)
    print(f"Fine Grid Cell Size (interpolated): {FINE_GRID_CELL_SIZE}", flush=True)
    print(f"Aggregation Method: {AGGREGATION_METHOD}", flush=True)
    print(f"Output Directories:", flush=True)
    print(f"  - Discrete: {OUT_EP_COARSE}", flush=True)
    print(f"  - Continuous: {OUT_EP_CONTINUOUS}", flush=True)
    print("=" * 100 + "\n", flush=True)
    
    gid = GRAPH_ID
    success_count = 0
    fail_count = 0
    
    print(f">>> START GRAPH {gid:02d}", flush=True)
    
    for ts in range(START_TIMESTEP, END_TIMESTEP + 1):
        try:
            # Load data
            pred_npmf, csv_actual_npmf = load_pred_particle_forces(gid, ts)
            if not pred_npmf or not csv_actual_npmf:
                print(f"  [{gid:02d}:{ts:02d}] No predicted or actual data", flush=True)
                fail_count += 1
                continue
            
            # Load particle data
            pos, dis, cn = load_particle_data(gid, ts)
            if not pos or not dis:
                print(f"  [{gid:02d}:{ts:02d}] No position/displacement data", flush=True)
                fail_count += 1
                continue
            
            # Filter to valid particles
            valid_ball_ids = set(csv_actual_npmf.keys()) & set(pos.keys()) & set(pred_npmf.keys())
            if not valid_ball_ids:
                print(f"  [{gid:02d}:{ts:02d}] No valid particles", flush=True)
                fail_count += 1
                continue
            
            pos_filtered = {bid: pos[bid] for bid in valid_ball_ids}
            cn_filtered = {bid: cn[bid] for bid in valid_ball_ids if bid in cn}
            actual_npmf = {bid: csv_actual_npmf[bid] for bid in valid_ball_ids}
            pred_npmf_filtered = {bid: pred_npmf[bid] for bid in valid_ball_ids}
            
            # Apply coarse graining
            coarse_data = coarse_grain_error_field(
                pos_filtered, actual_npmf, pred_npmf_filtered, cn_filtered,
                grid_cell_size=COARSE_GRID_CELL_SIZE,
                aggregation=AGGREGATION_METHOD
            )
            
            # Export discrete coarse-grained error VTK
            suffix = f"graph{gid:02d}_t{ts:03d}_cg{COARSE_GRID_CELL_SIZE}.vtk"
            result_discrete = export_coarse_grained_error(
                coarse_data,
                os.path.join(resolve_project_path(OUT_EP_COARSE), f"particle_error_{suffix}"),
                f"Graph {gid:02d} Timestep {ts:02d}",
                COARSE_GRID_CELL_SIZE
            )
            
            # Export continuous interpolated error field as ImageData
            suffix_continuous = f"graph{gid:02d}_t{ts:03d}_continuous_fg{FINE_GRID_CELL_SIZE}.vtk"
            result_continuous = export_continuous_error_field_imagedata(
                coarse_data,
                os.path.join(resolve_project_path(OUT_EP_CONTINUOUS), f"error_field_{suffix_continuous}"),
                f"Graph {gid:02d} Timestep {ts:02d}",
                FINE_GRID_CELL_SIZE
            )
            
            if result_discrete or result_continuous:
                success_count += 1
                print(f"  ✓ [{gid:02d}:{ts:02d}]", flush=True)
            else:
                fail_count += 1
                print(f"  ✗ [{gid:02d}:{ts:02d}]", flush=True)
        
        except Exception as e:
            print(f"  ✗ [{gid:02d}:{ts:02d}] Exception: {e}", flush=True)
            fail_count += 1
    
    print(f"\n<<< END GRAPH {gid:02d}", flush=True)
    print("\n" + "=" * 100, flush=True)
    print(f"COMPLETE: {success_count} success, {fail_count} failed", flush=True)
    print("=" * 100 + "\n", flush=True)


if __name__ == "__main__":
    main()
