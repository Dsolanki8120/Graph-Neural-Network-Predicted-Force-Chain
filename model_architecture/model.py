# model_fixed.py - RESIDUAL CONNECTION VERSION
# Changes:
# 1. Removed G_0 skip connections (concatenation).
# 2. Added Residual Connections (latent = latent + update).
# 3. Adjusted MLP input dimensions to match the removed skip features.

import torch
import torch.nn as nn

_DEVICE = None

def set_device_for_model(device):
    """Sets the global device for operations within this module."""
    global _DEVICE
    _DEVICE = device

# -------------------------------------------------
# 1. MLP builder - Exact paper spec
# -------------------------------------------------
def make_mlp_model(input_dim, output_dim, hidden_size=64, num_hidden_layers=2, final_relu=True):
    """
    Standard MLP: 2 hidden layers, 64 neurons, ReLU activations.
    """
    layers = []
    # Input to first hidden layer
    layers.append(nn.Linear(input_dim, hidden_size))
    layers.append(nn.ReLU())
    
    # Additional hidden layers
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    
    # Output layer
    layers.append(nn.Linear(hidden_size, output_dim))
    if final_relu:
        layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)

# -------------------------------------------------
# 2. Edge Update Block (Residual Version)
# -------------------------------------------------
class EdgeUpdate(nn.Module):
    """
    Edge update function φ_e.
    Input: [receiver ; sender ; current_edge]
    Output is added to the previous edge state (Residual).
    """
    def __init__(self, node_dim, edge_dim, hidden_dim=64):
        super().__init__()
        # Input: receiver features + sender features + edge features
        input_dim = 2 * node_dim + edge_dim
        self.mlp = make_mlp_model(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, edge_features, senders, receivers):
        sender_feats = node_features[senders]
        receiver_feats = node_features[receivers]
        
        # Concatenate: [receiver ; sender ; edge]
        edge_input = torch.cat([receiver_feats, sender_feats, edge_features], dim=-1)
        
        # Calculate Delta and add to original (Residual)
        delta_edges = self.mlp(edge_input)
        return self.norm(edge_features + delta_edges)

# -------------------------------------------------
# 3. Node Update Block (Residual Version)
# -------------------------------------------------
class NodeUpdate(nn.Module):
    """
    Node update function φ_n.
    Input: [current_node ; aggregated_edges]
    Output is added to the previous node state (Residual).
    """
    def __init__(self, node_dim, edge_dim, hidden_dim=64):
        super().__init__()
        # Input: current node features + aggregated edge features
        input_dim = node_dim + edge_dim
        self.mlp = make_mlp_model(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, edge_features, receivers, num_nodes):
        # Aggregate edges to nodes (Mean Aggregation)
        agg_edges = torch.zeros(num_nodes, edge_features.shape[1], 
                                device=_DEVICE, dtype=edge_features.dtype)
        agg_edges.index_add_(0, receivers, edge_features)

        deg = torch.bincount(receivers, minlength=num_nodes).clamp(min=1).unsqueeze(-1)
        agg_edges = agg_edges / deg.float()

        # Concatenate: [node ; agg_edges]
        node_input = torch.cat([node_features, agg_edges], dim=-1)
        
        # Calculate Delta and add to original (Residual)
        delta_nodes = self.mlp(node_input)
        return self.norm(node_features + delta_nodes)

# -------------------------------------------------
# 4. Encode–Process–Decode Network
# -------------------------------------------------
class EncodeProcessDecode(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, node_output_size,
                 hidden_dim=64, num_processing_steps=7):
        super().__init__()
        self.num_processing_steps = num_processing_steps
        self.hidden_dim = hidden_dim

        # --- ENCODER ---
        self.edge_encoder = make_mlp_model(2 * node_input_dim + edge_input_dim, hidden_dim)
        self.node_encoder = make_mlp_model(node_input_dim + hidden_dim, hidden_dim)
        
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

        # --- CORE (Residual Processing) ---
        self.edge_core = EdgeUpdate(hidden_dim, hidden_dim, hidden_dim)
        self.node_core = NodeUpdate(hidden_dim, hidden_dim, hidden_dim)

        # --- DECODER ---
        self.node_decoder = make_mlp_model(hidden_dim, node_output_size, final_relu=False)

    def forward(self, graph_dict, num_processing_steps_tensor):
        num_processing_steps = num_processing_steps_tensor.item()
        
        nodes = graph_dict["nodes"]
        edges = graph_dict["edges"]
        senders = graph_dict["senders"]
        receivers = graph_dict["receivers"]
        
        num_nodes = nodes.shape[0]
        num_edges = edges.shape[0]

        # 1. ENCODE
        sender_nodes = nodes[senders]
        receiver_nodes = nodes[receivers]
        edge_input = torch.cat([receiver_nodes, sender_nodes, edges], dim=-1)
        latent_edges = self.edge_norm(self.edge_encoder(edge_input))

        agg_edges = torch.zeros(num_nodes, self.hidden_dim, device=_DEVICE)
        if num_edges > 0:
            agg_edges.index_add_(0, receivers, latent_edges)
            deg = torch.bincount(receivers, minlength=num_nodes).clamp(min=1).unsqueeze(-1)
            agg_edges = agg_edges / deg.float()

        node_input = torch.cat([nodes, agg_edges], dim=-1)
        latent_nodes = self.node_norm(self.node_encoder(node_input))

        # 2. PROCESS (Residual Steps: G_n = G_{n-1} + ΔG)
        for _ in range(num_processing_steps):
            if num_edges > 0:
                latent_edges = self.edge_core(latent_nodes, latent_edges, senders, receivers)
                latent_nodes = self.node_core(latent_nodes, latent_edges, receivers, num_nodes)

        # 3. DECODE
        decoded_nodes = self.node_decoder(latent_nodes)
        return [{"nodes": decoded_nodes}]