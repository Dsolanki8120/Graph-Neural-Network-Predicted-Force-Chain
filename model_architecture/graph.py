# graph_fixed.py - UPDATED VERSION (Normalized NPMNCF per paper Cheng et al. 2023)
# Changes:
# 1. RESTORED: load_targets normalizes NPMNCF to Mean = 1.0 (per paper requirement)
# 2. Uses UNDIRECTED edges (bidirectional)
# 3. Proper displacement sign handling
# 4. Global max radius and CN for node feature normalization

import torch
import numpy as np
import pandas as pd
import os

# ----------------------------------------------------------------------
# Global configuration variables
# ----------------------------------------------------------------------
_DATA_PATH = None
_DEVICE = None
_TIMESTEPS = None


def set_global_paths_and_device(data_path, device, timesteps):
    global _DATA_PATH, _DEVICE, _TIMESTEPS
    _DATA_PATH = data_path
    _DEVICE = device
    _TIMESTEPS = timesteps
    print(f"Graph utilities: Data path set to {_DATA_PATH}")
    print(f"Graph utilities: Graphs will be constructed on CPU (moved to GPU during training)")


# ----------------------------------------------------------------------
# Data loading utilities
# ----------------------------------------------------------------------

def load_targets(assembly_id, t):
    """
    Reads the NPMNCF file and normalizes according to Cheng et al. (2023) paper.
    
    Paper Definition: NPMNCF = Force / Mean(Force)
    
    "normalized particle maximum normal contact forces (NPMNCFs, i.e., the 
    particle maximum normal contact forces divided by the mean value over 
    the granular system)"
    
    This ensures each assembly's target mean = 1.0
    """
    if _DATA_PATH is None:
        raise RuntimeError("Global data path not set.")
    
    fname = os.path.join(_DATA_PATH, f"{assembly_id}_{t}_ball_NPMNCF.tab")
    df = pd.read_csv(fname, sep=r"\s+", header=None, skiprows=2, names=["ball_id", "npmncf"])
    df.sort_values("ball_id", inplace=True)
    
    # Convert to numeric and clean
    df["ball_id"] = pd.to_numeric(df["ball_id"], errors="coerce")
    df.dropna(subset=["ball_id"], inplace=True)
    df["ball_id"] = df["ball_id"].astype(int)
    df["npmncf"] = pd.to_numeric(df["npmncf"], errors="coerce").fillna(0.0)
    
    npmncf_values = df["npmncf"].values.astype(np.float32)
    
    # =========================================================================
    # NORMALIZATION PER PAPER: Divide by mean to ensure Mean = 1.0
    # =========================================================================
    current_mean = npmncf_values.mean()
    if current_mean > 1e-9:
        # Formula: NPMNCF_normalized = Force / Mean(Force)
        npmncf_values = npmncf_values / current_mean
    # =========================================================================
    
    return df, npmncf_values


def load_scalar_feat(assembly_id, name, t):
    return pd.read_csv(
        f"{_DATA_PATH}/{assembly_id}_{t}_{name}.tab",
        header=None,
        names=["ball_id", name],
        sep=r"\s+",
        skiprows=2
    )


def load_all_node_data(assembly_id, t):
    df = load_scalar_feat(assembly_id, "ball_rad", t)
    df = df.merge(load_scalar_feat(assembly_id, "ball_cn", t), on="ball_id")
    for name in ["ball_disp_x", "ball_disp_y", "ball_disp_z",
                 "ball_pos_x", "ball_pos_y", "ball_pos_z"]:
        df = df.merge(load_scalar_feat(assembly_id, name, t), on="ball_id")
    df.sort_values("ball_id", inplace=True)
    return df


def load_edges(assembly_id, t, id_to_idx):
    e1 = pd.read_csv(f"{_DATA_PATH}/{assembly_id}_{t}_contact_end1.tab", sep=r"\s+", header=None, skiprows=2)
    e2 = pd.read_csv(f"{_DATA_PATH}/{assembly_id}_{t}_contact_end2.tab", sep=r"\s+", header=None, skiprows=2)

    e1_ids = pd.to_numeric(e1.iloc[:, 1], errors="coerce").dropna().astype(int)
    e2_ids = pd.to_numeric(e2.iloc[:, 1], errors="coerce").dropna().astype(int)

    valid_edges = []
    for a, b in zip(e1_ids, e2_ids):
        if a in id_to_idx and b in id_to_idx:
            edge = tuple(sorted([id_to_idx[a], id_to_idx[b]]))
            valid_edges.append(edge)
    
    return set(valid_edges)


# ----------------------------------------------------------------------
# Feature computation
# ----------------------------------------------------------------------

def compute_displacement_features(pos, disp, i, j, max_r):
    rel_pos = pos[i] - pos[j]
    rel_disp = disp[i] - disp[j]
    norm_rel_pos = np.linalg.norm(rel_pos)
    
    if norm_rel_pos < 1e-9:
        return [0.0, 0.0]
    
    contact_dir = rel_pos / norm_rel_pos
    disp_normal = np.dot(rel_disp, contact_dir)
    disp_perp_vec = rel_disp - (disp_normal * contact_dir)
    disp_perp = np.linalg.norm(disp_perp_vec)

    return [disp_perp / max_r, disp_normal / max_r]


def compute_global_max_radius(assembly_ids, timesteps):
    global_max_r = 0.0
    for assembly_id in assembly_ids:
        fname = os.path.join(_DATA_PATH, f"{assembly_id}_0_ball_rad.tab")
        df = pd.read_csv(fname, sep=r"\s+", header=None, skiprows=2, names=["ball_id", "ball_rad"])
        df["ball_rad"] = pd.to_numeric(df["ball_rad"], errors="coerce")
        max_r = df["ball_rad"].max()
        if not np.isnan(max_r) and max_r > global_max_r:
            global_max_r = max_r
    return global_max_r


def compute_global_max_cn(assembly_ids, timesteps):
    global_max_cn = 0.0
    for assembly_id in assembly_ids:
        for t in [0, 40, 80]:  # Check samples to save time
            try:
                fname = os.path.join(_DATA_PATH, f"{assembly_id}_{t}_ball_cn.tab")
                df = pd.read_csv(fname, sep=r"\s+", header=None, skiprows=2, names=["ball_id", "ball_cn"])
                df["ball_cn"] = pd.to_numeric(df["ball_cn"], errors="coerce")
                max_cn = df["ball_cn"].max()
                if not np.isnan(max_cn) and max_cn > global_max_cn:
                    global_max_cn = max_cn
            except: continue
    return global_max_cn


# ----------------------------------------------------------------------
# Graph creation with BIDIRECTIONAL edges
# ----------------------------------------------------------------------

def create_all_graphs(assembly_ids):
    if _DATA_PATH is None or _DEVICE is None or _TIMESTEPS is None:
        raise RuntimeError("Global settings not set.")

    print("🔍 Computing global normalization factors...")
    GLOBAL_MAX_R = compute_global_max_radius(assembly_ids, range(_TIMESTEPS))
    GLOBAL_MAX_CN = compute_global_max_cn(assembly_ids, range(_TIMESTEPS))

    all_graphs = []
    all_targets = []

    for idx, assembly_id in enumerate(assembly_ids):
        print(f"Processing assembly {assembly_id}... ({idx+1}/{len(assembly_ids)})")
        
        for t in range(1, _TIMESTEPS):
            # Node features
            node_data_t = load_all_node_data(assembly_id, t)
            id_to_idx = {bid: i for i, bid in enumerate(node_data_t["ball_id"])}
            
            radius_norm = (node_data_t["ball_rad"].astype(float) / GLOBAL_MAX_R).values.reshape(-1, 1)
            cn_norm = (node_data_t["ball_cn"].astype(float) / GLOBAL_MAX_CN).values.reshape(-1, 1)
            nodes = torch.tensor(np.hstack([radius_norm, cn_norm]), dtype=torch.float32)

            # Edge features (Bidirectional)
            edges_prev = load_edges(assembly_id, max(0, t-1), id_to_idx)
            edges_curr = load_edges(assembly_id, t, id_to_idx)
            all_edge_ids = edges_prev.union(edges_curr)

            pos = node_data_t[["ball_pos_x", "ball_pos_y", "ball_pos_z"]].values
            disp = node_data_t[["ball_disp_x", "ball_disp_y", "ball_disp_z"]].values

            edge_feats, senders, receivers = [], [], []
            for (i, j) in all_edge_ids:
                status = 0.0 if ((i,j) in edges_prev and (i,j) in edges_curr) else (-1.0 if (i,j) in edges_prev else 1.0)
                disp_feat = compute_displacement_features(pos, disp, i, j, GLOBAL_MAX_R)
                
                # Bidirectional i<->j
                for s, r in [(i, j), (j, i)]:
                    senders.append(s)
                    receivers.append(r)
                    edge_feats.append([status, disp_feat[0], disp_feat[1]])

            all_graphs.append({
                "nodes": nodes,
                "edges": torch.tensor(edge_feats, dtype=torch.float32) if edge_feats else torch.empty(0, 3),
                "senders": torch.tensor(senders, dtype=torch.int64),
                "receivers": torch.tensor(receivers, dtype=torch.int64),
            })

            # Targets (FIXED NORMALIZATION)
            _, npmncf_values = load_targets(assembly_id, t)
            all_targets.append(torch.tensor(npmncf_values.reshape(-1, 1), dtype=torch.float32))

    return all_graphs, all_targets


def data_dicts_to_graphs_tuple_pytorch(graph_dicts_list):
    batch_nodes, batch_edges, batch_senders, batch_receivers = [], [], [], []
    node_offset = 0
    for g in graph_dicts_list:
        batch_nodes.append(g["nodes"])
        batch_edges.append(g["edges"])
        batch_senders.append(g["senders"] + node_offset)
        batch_receivers.append(g["receivers"] + node_offset)
        node_offset += g["nodes"].shape[0]
    return {
        "nodes": torch.cat(batch_nodes, dim=0),
        "edges": torch.cat(batch_edges, dim=0),
        "senders": torch.cat(batch_senders, dim=0),
        "receivers": torch.cat(batch_receivers, dim=0),
    }


def pearson_corr_pytorch(x, y):
    xm, ym = x - torch.mean(x), y - torch.mean(y)
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2)) * torch.sqrt(torch.sum(ym ** 2))
    return r_num / (r_den + 1e-8)