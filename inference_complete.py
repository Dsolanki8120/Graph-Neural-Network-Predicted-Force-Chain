#!/usr/bin/env python3
"""
Complete Inference Pipeline
Loads pretrained GNN model and generates inference results with proper folder structure

Input:
  - Pretrained model: gnn_model_final.pth
  - Preprocessed data: processed_train_data(0-72).pt

Output Structure:
  inference_results/
  ├── contact_files/
  │   └── assembly_{id:02d}_timestep_{ts:02d}_contact_pairs.csv
  ├── contact_forces_folder/
  │   └── assembly_{id:02d}_timestep_{ts:02d}_predicted_contact_force.csv
  └── predicted_npmncf_files/
      └── assembly_{id:02d}_timestep_{ts:02d}_predicted_npmncf.csv
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import model as gnn_model
import os
from datetime import datetime
import csv

# Check GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Directory structure
OUTPUT_BASE_DIR = "New_inference_results_train"
CONTACT_FILES_DIR = os.path.join(OUTPUT_BASE_DIR, "contact_files")
CONTACT_FORCES_DIR = os.path.join(OUTPUT_BASE_DIR, "contact_forces_folder")
PREDICTED_NPMNCF_DIR = os.path.join(OUTPUT_BASE_DIR, "predicted_npmncf_files")


# ============================================================================
# STEP 1: Load Pretrained Model
# ============================================================================

def load_pretrained_model(model_path='gnn_model_normalization_final_mse_losss.pth'):
    """Load pretrained GNN model"""
    print("\n" + "="*80)
    print("LOADING PRETRAINED GNN MODEL")
    print("="*80)
    
    model = gnn_model.EncodeProcessDecode(
        node_input_dim=2,
        edge_input_dim=3,
        node_output_size=1,
        hidden_dim=64,
        num_processing_steps=7
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    gnn_model.set_device_for_model(DEVICE)
    
    print(f"✓ Model loaded from '{model_path}'")
    print(f"✓ Model on device: {DEVICE}")
    print(f"✓ Model in evaluation mode")
    print("="*80)
    
    return model


# ============================================================================
# STEP 2: Load Preprocessed Graph Data
# ============================================================================

def load_preprocessed_data(data_file='processed_test_normalization_fixed.pt'):
    """Load preprocessed graph data from .pt file"""
    print(f"\nLoading preprocessed data from {data_file}...")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    data = torch.load(data_file, map_location='cpu')
    
    if isinstance(data, tuple) and len(data) == 2:
        graphs_list = data[0]
        targets_list = data[1]
        print(f"✓ Loaded {len(graphs_list)} graphs")
        return graphs_list, targets_list
    else:
        raise ValueError(f"Unexpected data format. Got {type(data)}")


# ============================================================================
# STEP 3: Create Output Directories
# ============================================================================

def create_output_directories():
    """Create output folder structure"""
    print("\nCreating output directories...")
    os.makedirs(CONTACT_FILES_DIR, exist_ok=True)
    os.makedirs(CONTACT_FORCES_DIR, exist_ok=True)
    os.makedirs(PREDICTED_NPMNCF_DIR, exist_ok=True)
    print(f"✓ Created directory structure:")
    print(f"  - {CONTACT_FILES_DIR}/")
    print(f"  - {CONTACT_FORCES_DIR}/")
    print(f"  - {PREDICTED_NPMNCF_DIR}/")


# ============================================================================
# STEP 4: Run Inference and Save Results
# ============================================================================

def run_inference_and_save(model, graphs_list, targets_list, batch_size=50):
    """
    Run inference on all graphs and save results in proper structure
    
    Args:
        model: trained GNN model
        graphs_list: list of graph dictionaries
        targets_list: list of target tensors
        batch_size: graphs to process before saving
    """
    
    print("\n" + "="*80)
    print(f"RUNNING INFERENCE ON {len(graphs_list)} GRAPHS")
    print("="*80)
    
    total_graphs = len(graphs_list)
    total_particles = 0
    total_contacts = 0
    
    # Process each graph
    for graph_idx, (graph, target) in enumerate(zip(graphs_list, targets_list)):
        
        # Extract graph ID and timestep from graph (if available in metadata)
        # For now, we use sequential indexing: graph_idx maps to assembly and timestep
        assembly_id = (graph_idx // 81) + 1  # 81 timesteps per assembly, 1-based indexing
        timestep = graph_idx % 81
        
        if graph_idx % 50 == 0:
            print(f"\nProcessing graph {graph_idx+1}/{total_graphs} (Assembly {assembly_id:02d}, Timestep {timestep:02d})")
        
        # Move graph to device
        graph_device = {
            'nodes': graph['nodes'].to(DEVICE),
            'edges': graph['edges'].to(DEVICE),
            'senders': graph['senders'].to(DEVICE),
            'receivers': graph['receivers'].to(DEVICE)
        }
        
        num_nodes = graph_device['nodes'].shape[0]
        num_edges = graph_device['edges'].shape[0]
        
        # Run inference
        with torch.no_grad():
            num_steps_tensor = torch.tensor(7, device=DEVICE)
            output = model(graph_device, num_steps_tensor)
        
        # Extract predictions
        predictions = output[0]['nodes'].squeeze(-1).detach().cpu().numpy()
        target_npmncf = target.squeeze(-1).cpu().numpy() if target is not None else None
        
        senders = graph_device['senders'].cpu().numpy()
        receivers = graph_device['receivers'].cpu().numpy()
        
        # ================================================================
        # Save 1: Predicted NPMNCF per particle
        # ================================================================
        npmncf_file = os.path.join(
            PREDICTED_NPMNCF_DIR,
            f"assembly_{assembly_id:02d}_timestep_{timestep:02d}_predicted_npmncf.csv"
        )
        
        with open(npmncf_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ball_id', 'predicted_npmncf', 'actual_npmncf'])
            
            for particle_idx, pred_value in enumerate(predictions):
                # Clip negative values to 0 (physically unrealistic)
                pred_value = max(float(pred_value), 0.0)
                
                actual_value = float(target_npmncf[particle_idx]) if target_npmncf is not None else 0.0
                ball_id = particle_idx + 1  # Convert to 1-based indexing to match data_sets files
                writer.writerow([ball_id, pred_value, actual_value])
        
        total_particles += num_nodes
        
        # ================================================================
        # Save 2: Contact pairs (graph structure)
        # ================================================================
        contact_pairs_file = os.path.join(
            CONTACT_FILES_DIR,
            f"assembly_{assembly_id:02d}_timestep_{timestep:02d}_contact_pairs.csv"
        )
        
        with open(contact_pairs_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sender_id', 'receiver_id'])
            
            for edge_idx in range(num_edges):
                sender_id = int(senders[edge_idx]) + 1  # Convert to 1-based indexing
                receiver_id = int(receivers[edge_idx]) + 1  # Convert to 1-based indexing
                writer.writerow([sender_id, receiver_id])
        
        # ================================================================
        # Save 3: Contact forces (predicted)
        # ================================================================
        contact_forces_file = os.path.join(
            CONTACT_FORCES_DIR,
            f"assembly_{assembly_id:02d}_timestep_{timestep:02d}_predicted_contact_force.csv"
        )
        
        with open(contact_forces_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sender_id', 'receiver_id', 'sender_force', 'receiver_force', 'contact_force'])
            
            for edge_idx in range(num_edges):
                sender_id = int(senders[edge_idx]) + 1  # Convert to 1-based indexing
                receiver_id = int(receivers[edge_idx]) + 1  # Convert to 1-based indexing
                
                # Get particle forces
                sender_force = max(float(predictions[int(senders[edge_idx])]), 0.0)  # Clip negative to 0
                receiver_force = max(float(predictions[int(receivers[edge_idx])]), 0.0)  # Clip negative to 0
                
                # Contact force = average
                contact_force = (sender_force + receiver_force) / 2.0
                
                writer.writerow([sender_id, receiver_id, sender_force, receiver_force, contact_force])
        
        total_contacts += num_edges
        
        if graph_idx % 50 == 0:
            print(f"  ✓ Saved {num_nodes} particles, {num_edges} contacts")
    
    print("\n" + "="*80)
    print(f"INFERENCE COMPLETE!")
    print(f"Total graphs processed: {total_graphs}")
    print(f"Total particles saved: {total_particles}")
    print(f"Total contacts saved: {total_contacts}")
    print("="*80)
    print(f"\nOutput directories:")
    print(f"  {CONTACT_FILES_DIR}/")
    print(f"  {CONTACT_FORCES_DIR}/")
    print(f"  {PREDICTED_NPMNCF_DIR}/")
    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    
    print("\n" + "="*80)
    print("COMPLETE INFERENCE PIPELINE")
    print("="*80)
    
    try:
        # Load model
        model = load_pretrained_model('gnn_model_normalization_final_mse_loss.pth')
        
        # Load data
        graphs_list, targets_list = load_preprocessed_data('processed_train_normalization_fixed.pt')
        
        # Create output structure
        create_output_directories()
        
        # Run inference and save
        run_inference_and_save(model, graphs_list, targets_list, batch_size=50)
        
        print("\n✓ Inference pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
