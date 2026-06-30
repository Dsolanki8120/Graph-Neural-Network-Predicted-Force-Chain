#!/usr/bin/env python3
"""Export actual, predicted, and per-particle error force data to VTK."""
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
INFERENCE_DIR_TRAIN = "inference_results_train"
INFERENCE_DIR_TEST = "inference_results_test"

OUT_AP = "Actual_VTK_particles"
OUT_AC = "Actual_VTK_contacts"
OUT_PP = "Predicted_VTK_particles"
OUT_PC = "Predicted_VTK_contacts"
OUT_EP = "Error_VTK_particles"

EPS = 1e-6


def resolve_project_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_DIR, path)


DATA_PATH = resolve_project_path(DATA_DIR)
INFERENCE_DIR_FALLBACKS = {
    "train": [INFERENCE_DIR_TRAIN, "New_inference_results_train"],
    "test": [INFERENCE_DIR_TEST, "New_inference_results_test"],
}


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
    """Load coordination number from data_sets folder with multiple format support."""
    cn = {}
    
    # Try different file naming patterns
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
                        if i < 2:  # Skip header
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
                    print(f"  Loaded CN from: {pattern} ({len(cn)} particles)", flush=True)
                    return cn
            except Exception as e:
                pass
    
    # If no CN found, return empty dict (will default to 0)
    return cn


def load_particle_data(gid, ts):
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

    npmf = read_tab_file(os.path.join(DATA_PATH, f"{prefix}_ball_NPMNCF.tab"))
    if not npmf:
        npmf = read_tab_file(os.path.join(DATA_PATH, f"{prefix}_ball_npmncf.tab"))
    
    # Load coordination number using dedicated function with multiple format support
    cn = load_coordination_number(gid, ts)
    
    return pos, dis, npmf, cn


def load_actual_contacts(gid, ts):
    prefix = f"{gid}_{ts}"
    c1 = read_tab_file(os.path.join(DATA_PATH, f"{prefix}_contact_end1.tab"))
    c2 = read_tab_file(os.path.join(DATA_PATH, f"{prefix}_contact_end2.tab"))
    return [(int(c1[idx]), int(c2[idx])) for idx in c1 if idx in c2]


def get_inference_dir_for_graph(gid):
    """Route graph ID to correct inference directory.
    
    Graph 1-57: New_inference_results_train
    Graph 58-73: New_inference_results_test
    """
    if 1 <= gid <= 57:
        split = "train"
    elif 58 <= gid <= 73:
        split = "test"
    else:
        return None, None
    
    candidates = INFERENCE_DIR_FALLBACKS.get(split, [])
    for candidate in candidates:
        path = resolve_project_path(candidate)
        if os.path.isdir(path):
            return split, path
    return None, None


def existing_inference_dirs():
    dirs = []
    for split, candidates in INFERENCE_DIR_FALLBACKS.items():
        for candidate in candidates:
            path = resolve_project_path(candidate)
            if os.path.isdir(path):
                dirs.append((split, path))
                break
    return dirs


def first_existing_file(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def inference_file_candidates(inference_dir, folder, gid, ts, suffix):
    return [
        os.path.join(inference_dir, folder, f"assembly_{gid:02d}_timestep_{ts:02d}_{suffix}"),
        os.path.join(inference_dir, folder, f"assembly_{gid:02d}_timestep_{ts:03d}_{suffix}"),
        os.path.join(inference_dir, folder, f"assembly_{gid:02d}_timestep_{ts}_{suffix}"),
    ]


def load_pred_contacts(inference_dir, gid, ts):
    contact_file = first_existing_file(
        inference_file_candidates(inference_dir, "contact_files", gid, ts, "contact_pairs.csv")
    )
    force_file = first_existing_file(
        inference_file_candidates(
            inference_dir,
            "contact_forces_folder",
            gid,
            ts,
            "predicted_contact_force.csv",
        )
    )

    contacts, forces = [], {}
    if os.path.exists(contact_file):
        try:
            for _, row in pd.read_csv(contact_file).iterrows():
                contacts.append((int(row["sender_id"]), int(row["receiver_id"])))
        except Exception:
            pass

    if os.path.exists(force_file):
        try:
            for idx, row in pd.read_csv(force_file).iterrows():
                forces[idx] = float(row["contact_force"])
        except Exception:
            pass
    return contacts, forces


def load_pred_particle_forces(inference_dir, gid, ts):
    fp = first_existing_file(
        inference_file_candidates(
            inference_dir,
            "predicted_npmncf_files",
            gid,
            ts,
            "predicted_npmncf.csv",
        )
    )
    predicted, actual = {}, {}
    if not os.path.exists(fp):
        return predicted, actual

    try:
        df = pd.read_csv(fp)
        for _, row in df.iterrows():
            pid = int(row["ball_id"])
            predicted[pid] = float(row["predicted_npmncf"])
            if "actual_npmncf" in df.columns:
                actual[pid] = float(row["actual_npmncf"])
    except Exception:
        return {}, {}
    return predicted, actual





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


def export_particles(pos, dis, forces, out, label, coord_num=None, actual_forces=None, predicted_forces=None):
    if not pos:
        return False
    try:
        polydata, pids = make_particle_polydata(pos, dis)
        
        # Add particle IDs
        particle_ids = np.array(pids, dtype=np.int32)
        add_point_array(polydata, particle_ids, "particle_id", vtk.VTK_INT)
        
        # Add force values aligned with particle IDs
        force_values = np.array([forces.get(pid, 0.0) for pid in pids], dtype=np.float32)
        add_point_array(polydata, force_values, "particle_force")
        
        # Add actual and predicted forces if provided (for alignment checking)
        if actual_forces is not None:
            actual = np.array([actual_forces.get(pid, 0.0) for pid in pids], dtype=np.float32)
            add_point_array(polydata, actual, "actual_npmncf")
        
        if predicted_forces is not None:
            predicted = np.array([predicted_forces.get(pid, 0.0) for pid in pids], dtype=np.float32)
            add_point_array(polydata, predicted, "predicted_npmncf")
        
        # Always add coordination number (handle None case)
        if coord_num is None:
            coord_num = {}
        cn_values = np.array([coord_num.get(pid, 0) for pid in pids], dtype=np.int32)
        add_point_array(polydata, cn_values, "coordination_number", vtk.VTK_INT)
        
        polydata.GetPointData().SetScalars(polydata.GetPointData().GetArray("particle_force"))
        write_polydata(polydata, out, f"{label} particles")
        print(f"✓ {label} particles")
        return True
    except Exception as e:
        print(f"✗ {label} particles: {e}")
        return False


def export_particle_errors(pos, dis, actual_forces, predicted_forces, out, label, coord_num=None):
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
        
        # Add coordination number (required)
        if coord_num is None:
            coord_num = {}
        cn_values = np.array([coord_num.get(pid, 0) for pid in pids], dtype=np.int32)
        add_point_array(polydata, cn_values, "coordination_number", vtk.VTK_INT)
        
        polydata.GetPointData().SetScalars(
            polydata.GetPointData().GetArray("particle_force_absolute_error")
        )
        write_polydata(polydata, out, f"{label} particle errors")
        print(f"✓ {label} particle errors")
        return True
    except Exception as e:
        print(f"✗ {label} particle errors: {e}")
        return False


def export_contacts(pos, dis, contacts, forces, out, label):
    if not contacts:
        return False
    try:
        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(len(pos))
        pids = sorted(pos.keys())
        p2i = {p: i for i, p in enumerate(pids)}
        for i, pid in enumerate(pids):
            pp = pos[pid]
            pts.SetPoint(i, float(pp[0]), float(pp[1]), float(pp[2]))

        lines = vtk.vtkCellArray()
        valid_contact_indices = []
        for idx, (p1, p2) in enumerate(contacts):
            i1, i2 = p2i.get(p1), p2i.get(p2)
            if i1 is None or i2 is None:
                continue
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i1)
            line.GetPointIds().SetId(1, i2)
            lines.InsertNextCell(line)
            valid_contact_indices.append(idx)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(pts)
        polydata.SetLines(lines)

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

        contact_force = np.array(
            [forces.get(idx, 0.0) for idx in valid_contact_indices], dtype=np.float32
        )
        vtk_contact_force = numpy_support.numpy_to_vtk(
            contact_force, deep=True, array_type=vtk.VTK_FLOAT
        )
        vtk_contact_force.SetName("contact_force")
        polydata.GetCellData().SetScalars(vtk_contact_force)

        write_polydata(polydata, out, f"{label} contacts")
        print(f"✓ {label} contacts")
        return True
    except Exception as e:
        print(f"✗ {label} contacts: {e}")
        return False


def process_pair(split, inference_dir, gid, ts):
    try:
        # Load predicted and actual values from inference results CSV (NORMALIZED)
        pred_npmf, csv_actual_npmf = load_pred_particle_forces(inference_dir, gid, ts)
        cp, cf_pred = load_pred_contacts(inference_dir, gid, ts)
        if not pred_npmf and not cp:
            print(f"  [{gid:02d}:{ts:02d}] No data", flush=True)
            return False

        # Load particle position, displacement, and coordination number from data_sets
        # DO NOT use npmf from data_sets (it's not normalized)
        pos, dis, _, cn = load_particle_data(gid, ts)
        if not pos or not dis:
            print(f"  [{gid:02d}:{ts:02d}] No position/displacement data", flush=True)
            return False
        
        # Use ONLY normalized actual values from CSV inference results
        # csv_actual_npmf contains normalized NPMNCF values (particle_id -> force)
        actual_npmf = csv_actual_npmf
        if not actual_npmf:
            # Skip if no actual values available
            return False
        
        # Filter pos, dis, cn to only include ball_ids that exist in normalized force data
        # This ensures correct mapping between positions/displacements and normalized forces
        valid_ball_ids = set(actual_npmf.keys()) & set(pos.keys())
        pos = {bid: pos[bid] for bid in valid_ball_ids}
        dis = {bid: dis[bid] for bid in valid_ball_ids if bid in dis}
        cn = {bid: cn[bid] for bid in valid_ball_ids if bid in cn}
        
        if not pos:
            return False

        # Load actual contacts and create contact forces from normalized actual particle forces
        ca = load_actual_contacts(gid, ts)
        ca = [(p1, p2) for p1, p2 in ca if p1 in actual_npmf and p2 in actual_npmf]
        cfa = {i: (actual_npmf.get(p1, 0.0) + actual_npmf.get(p2, 0.0)) / 2 for i, (p1, p2) in enumerate(ca)}
        
        # Create contact forces from normalized predicted particle forces
        cp = [(p1, p2) for p1, p2 in cp if p1 in pred_npmf and p2 in pred_npmf]
        cf_pred = {i: (pred_npmf.get(p1, 0.0) + pred_npmf.get(p2, 0.0)) / 2 for i, (p1, p2) in enumerate(cp)}
        
        suffix = f"graph{gid:02d}_t{ts:03d}.vtk"

        # Filter pred_npmf to only include valid ball_ids (with position and actual data)
        pred_npmf_filtered = {bid: pred_npmf[bid] for bid in valid_ball_ids if bid in pred_npmf}

        # Export actual particles with both actual and predicted forces from CSV
        export_particles(
            pos,
            dis,
            actual_npmf,
            os.path.join(resolve_project_path(OUT_AP), f"particles_{suffix}"),
            f"{split} Actual",
            cn,
            actual_forces=actual_npmf,
            predicted_forces=pred_npmf_filtered,
        )
        if ca:
            export_contacts(
                pos,
                dis,
                ca,
                cfa,
                os.path.join(resolve_project_path(OUT_AC), f"contacts_{suffix}"),
                f"{split} Actual",
            )

        # Export predicted particles with both actual and predicted forces from CSV
        export_particles(
            pos,
            dis,
            pred_npmf_filtered,
            os.path.join(resolve_project_path(OUT_PP), f"particles_{suffix}"),
            f"{split} Predicted",
            cn,
            actual_forces=actual_npmf,
            predicted_forces=pred_npmf_filtered,
        )
        export_particle_errors(
            pos,
            dis,
            actual_npmf,
            pred_npmf_filtered,
            os.path.join(resolve_project_path(OUT_EP), f"particle_error_{suffix}"),
            split,
            cn,
        )
        if cp:
            export_contacts(
                pos,
                dis,
                cp,
                cf_pred,
                os.path.join(resolve_project_path(OUT_PC), f"contacts_{suffix}"),
                f"{split} Predicted",
            )
        return True
    except Exception as e:
        import traceback
        print(f"\n✗✗✗ ERROR [{gid:02d}:{ts:02d}] {type(e).__name__}: {str(e)}", flush=True)
        print(f"Traceback:\n{traceback.format_exc()}", flush=True)
        return False


def main():
    import sys
    
    for output_dir in [OUT_AP, OUT_AC, OUT_PP, OUT_PC, OUT_EP]:
        os.makedirs(resolve_project_path(output_dir), exist_ok=True)
    
    # Open log file for error tracking
    log_file = open(os.path.join(PROJECT_DIR, "vtk_export_errors.log"), "w")
    sys.stderr = log_file  # Redirect stderr to log file

    print("\n" + "=" * 80, flush=True)
    print("VTK Export Started - Processing Graphs 1-73 (Sequential Order)", flush=True)
    print(f"Log file: {os.path.join(PROJECT_DIR, 'vtk_export_errors.log')}", flush=True)
    print("=" * 80, flush=True)
    success_count, fail_count, skip_count = 0, 0, 0

    for gid in range(1, 74):  # Graphs 1-73 IN ORDER
        split, inference_dir = get_inference_dir_for_graph(gid)
        if not inference_dir:
            print(f"\n✗ GRAPH {gid:02d}: inference directory not found", flush=True)
            fail_count += 81  # 81 timesteps
            continue
        
        print(f"\n>>> START GRAPH {gid:02d} (81 timesteps) ({split})", flush=True)
        ts_success, ts_fail, ts_skip = 0, 0, 0
        for ts in range(0, 81):
            result = process_pair(split, inference_dir, gid, ts)
            if result is True:
                ts_success += 1
                success_count += 1
                print(f"  ✓ [{gid:02d}:{ts:02d}]", flush=True, end=" ")
            elif result is False:
                ts_fail += 1
                fail_count += 1
                print(f"  ✗ [{gid:02d}:{ts:02d}]", flush=True, end=" ")
            else:
                ts_skip += 1
                skip_count += 1
                print(f"  - [{gid:02d}:{ts:02d}]", flush=True, end=" ")
            
            if (ts + 1) % 10 == 0:
                print(flush=True)
        
        print(flush=True)
        print(f"<<< END GRAPH {gid:02d}: {ts_success}/81 success, {ts_fail} failed, {ts_skip} skipped", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"COMPLETE: {success_count} success, {fail_count} failed, {skip_count} skipped", flush=True)
    print("=" * 80 + "\n", flush=True)
    
    log_file.close()


if __name__ == "__main__":
    main()
