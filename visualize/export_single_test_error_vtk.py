#!/usr/bin/env python3
"""Export error VTK file for a single test assembly (only error data, no other exports)."""
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import vtk
    from vtk.util import numpy_support
except Exception:
    print("Error: VTK not installed")
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR = "data_sets(0-72)"
INFERENCE_DIR_TEST = "New_inference_results_test"
OUT_EP = "Error_VTK_particles_single_run"

EPS = 1e-6

# ===== CONFIGURATION =====
TEST_ASSEMBLY = 58  # Change to 58-72 as needed
START_TIMESTEP = 0  # Change as needed
END_TIMESTEP = 80   # 0-80 = 81 total timesteps
# ========================


def resolve_project_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_DIR, path)


DATA_PATH = resolve_project_path(DATA_DIR)
INFERENCE_PATH = resolve_project_path(INFERENCE_DIR_TEST)


def read_tab_file(fp):
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
            except Exception as e:
                print(f"  ERROR reading {fp}: {e}")
                return {}, {}
    
    return {}, {}


def add_point_array(polydata, values, name, array_type=vtk.VTK_FLOAT):
    arr = numpy_support.numpy_to_vtk(values, deep=True, array_type=array_type)
    arr.SetName(name)
    polydata.GetPointData().AddArray(arr)
    return arr


def make_particle_polydata(pos, dis):
    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(len(pos))
    pids = sorted(pos.keys())
    for i, pid in enumerate(pids):
        pp = pos[pid]
        pts.SetPoint(i, float(pp[0]), float(pp[1]), float(pp[2]))

    cells = vtk.vtkCellArray()
    for i in range(len(pos)):
        cell = vtk.vtkVertex()
        cell.GetPointIds().SetId(0, i)
        cells.InsertNextCell(cell)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pts)
    polydata.SetVerts(cells)

    displacement = np.zeros((len(pos), 3), dtype=np.float32)
    for i, pid in enumerate(pids):
        if pid in dis:
            displacement[i] = dis[pid]

    vtk_displacement = numpy_support.numpy_to_vtk(
        displacement.ravel(), deep=True, array_type=vtk.VTK_FLOAT
    )
    vtk_displacement.SetNumberOfComponents(3)
    vtk_displacement.SetName("displacement")
    polydata.GetPointData().AddArray(vtk_displacement)
    return polydata, pids


def write_polydata(polydata, out, label):
    print(f"→ exporting {label}: {out}", flush=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out)
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        raise RuntimeError(f"VTK writer failed for {out}")
    print(f"✓ exported {label}: {out}", flush=True)


def export_particle_errors(pos, dis, actual_forces, predicted_forces, out, label, coord_num=None):
    """Export ONLY error data (no predicted/actual particles)."""
    if not pos or not predicted_forces:
        return False
    try:
        polydata, pids = make_particle_polydata(pos, dis)
        particle_ids = np.array(pids, dtype=np.int32)
        actual = np.array([actual_forces.get(pid, 0.0) for pid in pids], dtype=np.float32)
        predicted = np.array([predicted_forces.get(pid, 0.0) for pid in pids], dtype=np.float32)

        signed_error = predicted - actual
        absolute_error = np.abs(signed_error)
        relative_error = absolute_error / (np.abs(actual) + EPS)
        relative_error_percent = relative_error * 100.0

        # Add all data arrays
        add_point_array(polydata, particle_ids, "particle_id", vtk.VTK_INT)
        add_point_array(polydata, actual, "actual_npmncf")
        add_point_array(polydata, predicted, "predicted_npmncf")
        add_point_array(polydata, signed_error.astype(np.float32), "particle_force_error")
        add_point_array(polydata, absolute_error.astype(np.float32), "particle_force_absolute_error")
        add_point_array(polydata, relative_error.astype(np.float32), "particle_force_relative_error")
        add_point_array(
            polydata,
            relative_error_percent.astype(np.float32),
            "particle_force_relative_error_percent",
        )
        
        # Add coordination number
        if coord_num is None:
            coord_num = {}
        cn_values = np.array([coord_num.get(pid, 0) for pid in pids], dtype=np.int32)
        add_point_array(polydata, cn_values, "coordination_number", vtk.VTK_INT)
        
        # Calculate statistics
        n_over = np.sum(signed_error > EPS)
        n_under = np.sum(signed_error < -EPS)
        n_perfect = np.sum(np.abs(signed_error) <= EPS)
        n_total = len(pids)
        mean_signed = np.mean(signed_error)
        mean_abs = np.mean(absolute_error)
        
        print(f"  Stats: Over={n_over:5d} ({100*n_over/n_total:5.1f}%) | Under={n_under:5d} ({100*n_under/n_total:5.1f}%) | Perfect={n_perfect:5d} ({100*n_perfect/n_total:5.1f}%)")
        print(f"         Mean_Signed={mean_signed:8.5f} | Mean_Abs={mean_abs:8.5f}")
        
        polydata.GetPointData().SetScalars(
            polydata.GetPointData().GetArray("particle_force_absolute_error")
        )
        write_polydata(polydata, out, f"{label} error")
        print(f"✓ {label} error")
        return True
    except Exception as e:
        print(f"✗ {label} error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    os.makedirs(resolve_project_path(OUT_EP), exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"ERROR VTK EXPORT - Single Test Assembly: {TEST_ASSEMBLY}")
    print(f"Timesteps: {START_TIMESTEP} to {END_TIMESTEP}")
    print("=" * 80 + "\n")
    
    success_count = 0
    fail_count = 0
    
    gid = TEST_ASSEMBLY
    print(f"\n>>> START ASSEMBLY {gid:02d}", flush=True)
    
    for ts in range(START_TIMESTEP, END_TIMESTEP + 1):
        try:
            # Load data
            pred_npmf, csv_actual_npmf = load_pred_particle_forces(gid, ts)
            if not pred_npmf or not csv_actual_npmf:
                print(f"  [{gid:02d}:{ts:02d}] No predicted or actual data")
                fail_count += 1
                continue
            
            # Load particle positions and displacements
            pos, dis, cn = load_particle_data(gid, ts)
            if not pos or not dis:
                print(f"  [{gid:02d}:{ts:02d}] No position/displacement data")
                fail_count += 1
                continue
            
            # Filter to valid particles
            valid_ball_ids = set(csv_actual_npmf.keys()) & set(pos.keys()) & set(pred_npmf.keys())
            if not valid_ball_ids:
                print(f"  [{gid:02d}:{ts:02d}] No valid particles")
                fail_count += 1
                continue
            
            pos_filtered = {bid: pos[bid] for bid in valid_ball_ids}
            dis_filtered = {bid: dis[bid] for bid in valid_ball_ids if bid in dis}
            cn_filtered = {bid: cn[bid] for bid in valid_ball_ids if bid in cn}
            actual_npmf = {bid: csv_actual_npmf[bid] for bid in valid_ball_ids}
            pred_npmf_filtered = {bid: pred_npmf[bid] for bid in valid_ball_ids}
            
            # Export error VTK
            suffix = f"graph{gid:02d}_t{ts:03d}.vtk"
            result = export_particle_errors(
                pos_filtered,
                dis_filtered,
                actual_npmf,
                pred_npmf_filtered,
                os.path.join(resolve_project_path(OUT_EP), f"particle_error_{suffix}"),
                f"Test Assembly {gid:02d} Timestep {ts:02d}",
                cn_filtered,
            )
            
            if result:
                success_count += 1
                print(f"  ✓ [{gid:02d}:{ts:02d}]", flush=True)
            else:
                fail_count += 1
                print(f"  ✗ [{gid:02d}:{ts:02d}]", flush=True)
        
        except Exception as e:
            print(f"  ✗ [{gid:02d}:{ts:02d}] Exception: {e}")
            fail_count += 1
    
    print(f"\n<<< END ASSEMBLY {gid:02d}")
    print("\n" + "=" * 80)
    print(f"COMPLETE: {success_count} success, {fail_count} failed")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
