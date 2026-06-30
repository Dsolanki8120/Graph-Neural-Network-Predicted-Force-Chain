"""
ParaView macro: analyze particle error VTK files.

Run from ParaView:
  Tools -> Python Shell -> Run Script
or save as a ParaView macro.

Outputs CSV files in Transforme_GNN/paraview_error_analysis:
  - error_by_timestep.csv
  - error_by_coordination_number.csv
  - error_particles_sample.csv
"""
from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "paraview" else SCRIPT_DIR
INPUT_DIR = os.path.join(PROJECT_DIR, "Error_VTK_particles")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "paraview_error_analysis")

MAX_SAMPLE_ROWS_PER_FILE = 200


def parse_graph_timestep(path):
    name = os.path.basename(path)
    match = re.search(r"graph(\d+)_t(\d+)", name)
    if not match:
        return -1, -1
    return int(match.group(1)), int(match.group(2))


def get_point_array(polydata, name, default=None):
    arr = polydata.GetPointData().GetArray(name)
    if arr is None:
        if default is None:
            raise KeyError(f"Missing point array '{name}'")
        return default
    return vtk_to_numpy(arr)


def safe_corr(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    value = np.corrcoef(a, b)[0, 1]
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(
        os.path.join(INPUT_DIR, name)
        for name in os.listdir(INPUT_DIR)
        if name.endswith(".vtk")
    )
    if not files:
        print(f"No VTK files found in {INPUT_DIR}")
        return

    by_timestep_rows = []
    sample_rows = []
    cn_bins = defaultdict(lambda: {"count": 0, "abs": [], "signed": [], "rel_pct": [], "actual": []})

    print(f"Reading {len(files)} particle error VTK files from {INPUT_DIR}")

    for file_index, vtk_file in enumerate(files, start=1):
        graph_id, timestep = parse_graph_timestep(vtk_file)
        print(f"[{file_index}/{len(files)}] analyzing {os.path.basename(vtk_file)}", flush=True)

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(vtk_file)
        reader.Update()
        polydata = reader.GetOutput()

        n = polydata.GetNumberOfPoints()
        zeros = np.zeros(n, dtype=np.float64)

        particle_id = get_point_array(polydata, "particle_id", np.arange(n)).astype(np.int64)
        cn = get_point_array(polydata, "coordination_number", zeros).astype(np.int64)
        actual = get_point_array(polydata, "actual_npmncf", zeros).astype(np.float64)
        predicted = get_point_array(polydata, "predicted_npmncf", zeros).astype(np.float64)
        signed_error = get_point_array(polydata, "particle_force_error", predicted - actual).astype(np.float64)
        abs_error = get_point_array(polydata, "particle_force_absolute_error", np.abs(signed_error)).astype(np.float64)
        rel_pct = get_point_array(polydata, "particle_force_relative_error_percent", zeros).astype(np.float64)

        by_timestep_rows.append(
            {
                "graph": graph_id,
                "timestep": timestep,
                "num_particles": n,
                "mean_actual_npmncf": float(np.mean(actual)),
                "mean_predicted_npmncf": float(np.mean(predicted)),
                "mae": float(np.mean(abs_error)),
                "rmse": float(np.sqrt(np.mean(signed_error**2))),
                "bias_mean_signed_error": float(np.mean(signed_error)),
                "max_abs_error": float(np.max(abs_error)),
                "mean_relative_error_percent": float(np.mean(rel_pct)),
                "corr_actual_predicted": safe_corr(actual, predicted),
                "corr_coordination_abs_error": safe_corr(cn, abs_error),
            }
        )

        for value in np.unique(cn):
            mask = cn == value
            bucket = cn_bins[int(value)]
            bucket["count"] += int(np.sum(mask))
            bucket["abs"].append(float(np.mean(abs_error[mask])))
            bucket["signed"].append(float(np.mean(signed_error[mask])))
            bucket["rel_pct"].append(float(np.mean(rel_pct[mask])))
            bucket["actual"].append(float(np.mean(actual[mask])))

        step = max(1, n // MAX_SAMPLE_ROWS_PER_FILE)
        for i in range(0, n, step):
            sample_rows.append(
                {
                    "graph": graph_id,
                    "timestep": timestep,
                    "particle_id": int(particle_id[i]),
                    "coordination_number": int(cn[i]),
                    "actual_npmncf": float(actual[i]),
                    "predicted_npmncf": float(predicted[i]),
                    "particle_force_error": float(signed_error[i]),
                    "particle_force_absolute_error": float(abs_error[i]),
                    "particle_force_relative_error_percent": float(rel_pct[i]),
                }
            )

    by_cn_rows = []
    for cn_value, bucket in sorted(cn_bins.items()):
        by_cn_rows.append(
            {
                "coordination_number": cn_value,
                "particle_count": bucket["count"],
                "mean_abs_error": mean(bucket["abs"]),
                "mean_signed_error": mean(bucket["signed"]),
                "mean_relative_error_percent": mean(bucket["rel_pct"]),
                "mean_actual_npmncf": mean(bucket["actual"]),
            }
        )

    write_csv(os.path.join(OUTPUT_DIR, "error_by_timestep.csv"), by_timestep_rows)
    write_csv(os.path.join(OUTPUT_DIR, "error_by_coordination_number.csv"), by_cn_rows)
    write_csv(os.path.join(OUTPUT_DIR, "error_particles_sample.csv"), sample_rows)

    print(f"Done. CSV output folder: {OUTPUT_DIR}")
    print("Open the CSV files in ParaView with CSV Reader, then use Plot Data or Chart View.")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


main()
