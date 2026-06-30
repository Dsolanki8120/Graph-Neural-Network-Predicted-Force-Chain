import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os, re
from datetime import datetime
from scipy.stats import pearsonr

# Import local modules
import graph as graph      
import model as gnn_model

# ----------------------------------------------------------------------
# 1. Global Configuration & Device Setup
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CURRENT_DIR = os.getcwd()
DATA_PATH = os.path.join(CURRENT_DIR, "data_sets(0-72)")

# Output folders for plots and models
OUTPUT_FOLDER = "plots_bidirectional_3"
PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")

os.makedirs(PLOTS_FOLDER, exist_ok=True)


# ======================================================================
# 2. Data Discovery & Splitting (Dynamic Logic)
# ======================================================================
def get_data_info(data_path):
    """Scans the data directory to find unique assembly IDs and timesteps."""
    assembly_ids = set()
    timesteps = set()
    pattern = re.compile(r"(\d+)_(\d+)_ball_NPMNCF\.tab")
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"WARNING: Data path '{data_path}' created but is empty.")
        return [], 0

    for fname in os.listdir(data_path):
        match = pattern.match(fname)
        if match:
            assembly_ids.add(int(match.group(1)))
            timesteps.add(int(match.group(2)))
            
    return sorted(list(assembly_ids)), len(timesteps)

ASSEMBLY_IDS, TIMESTEPS = get_data_info(DATA_PATH)
total_assemblies = len(ASSEMBLY_IDS)
print(f"Discovered {total_assemblies} assemblies with {TIMESTEPS} timesteps each.")

if total_assemblies < 1:
    TRAIN_ASSEMBLY_IDS = []
    TEST_ASSEMBLY_IDS = []
else:
    # --- Dynamic Split Logic (80% Train / 20% Test) ---
    train_ratio = 0.8
    split_index = int(total_assemblies * train_ratio)
    
    if split_index >= total_assemblies and total_assemblies > 1:
        split_index = total_assemblies - 1

    TRAIN_ASSEMBLY_IDS = ASSEMBLY_IDS[:split_index]
    TEST_ASSEMBLY_IDS = ASSEMBLY_IDS[split_index:]

    print(f"\n--- Split Configuration ({int(train_ratio*100)}/{100-int(train_ratio*100)}) ---")
    print(f"Training Assemblies ({len(TRAIN_ASSEMBLY_IDS)}): {TRAIN_ASSEMBLY_IDS}")
    print(f"Testing Assemblies  ({len(TEST_ASSEMBLY_IDS)}): {TEST_ASSEMBLY_IDS}")

# Set global variables in graph module
graph.set_global_paths_and_device(DATA_PATH, DEVICE, TIMESTEPS)
gnn_model.set_device_for_model(DEVICE)

# ----------------------------------------------------------------------
# 3. Data Loading / Graph Generation & Saving
# ----------------------------------------------------------------------
# CHANGED: Renamed cache files to ensure the new Normalization (Mean=1.0) is used
PROCESSED_TRAIN_FILE = "processed_train_normalization_fixed.pt"
PROCESSED_TEST_FILE = "processed_test_normalization_fixed.pt" 

# Check if cache exists
if os.path.exists(PROCESSED_TRAIN_FILE) and os.path.exists(PROCESSED_TEST_FILE):
    print("\nFound cached graph data (Residual/Normalized). Loading...")
    train_graph_dicts, train_targets = torch.load(PROCESSED_TRAIN_FILE)
    test_graph_dicts, test_targets = torch.load(PROCESSED_TEST_FILE)
    
    if len(test_graph_dicts) == 0 and len(TEST_ASSEMBLY_IDS) > 0:
        print("Warning: Cached test data is empty. Re-generating...")
        train_graph_dicts, train_targets = graph.create_all_graphs(TRAIN_ASSEMBLY_IDS)
        test_graph_dicts, test_targets = graph.create_all_graphs(TEST_ASSEMBLY_IDS)
        torch.save((train_graph_dicts, train_targets), PROCESSED_TRAIN_FILE)
        torch.save((test_graph_dicts, test_targets), PROCESSED_TEST_FILE)
    else:
        print("Data loaded successfully.")
else:
    print("\nCached data not found. Generating fresh normalized graphs for Residual Model...")
    
    if len(TRAIN_ASSEMBLY_IDS) == 0:
        raise RuntimeError("No raw data found. Cannot proceed.")

    print(f"\n--- Generating TRAINING graphs ---")
    train_graph_dicts, train_targets = graph.create_all_graphs(TRAIN_ASSEMBLY_IDS)

    print(f"\n--- Generating TESTING graphs ---")
    test_graph_dicts, test_targets = graph.create_all_graphs(TEST_ASSEMBLY_IDS)

    print("\nSaving constructed graphs to disk...")
    torch.save((train_graph_dicts, train_targets), PROCESSED_TRAIN_FILE)
    torch.save((test_graph_dicts, test_targets), PROCESSED_TEST_FILE)

print(f"\nTotal Training Samples (Graphs): {len(train_graph_dicts)}")
print(f"Total Testing Samples (Graphs):  {len(test_graph_dicts)}")

if len(test_graph_dicts) == 0:
    print(" ERROR: Test set is empty.")
    exit()

def move_graphs_tuple_to_device(graphs_tuple, device):
    return {
        "nodes": graphs_tuple["nodes"].to(device),
        "edges": graphs_tuple["edges"].to(device),
        "senders": graphs_tuple["senders"].to(device),
        "receivers": graphs_tuple["receivers"].to(device),
    }

# ----------------------------------------------------------------------
# 4. Model Initialization
# ----------------------------------------------------------------------
num_epochs = 100
batch_size = 1  
num_processing_steps = 7
clip_norm = 5.0
learning_rate = 1e-4

model = gnn_model.EncodeProcessDecode(
    node_input_dim=2, 
    edge_input_dim=3, 
    node_output_size=1, 
    hidden_dim=64,
    num_processing_steps=num_processing_steps
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_processing_steps_tensor = torch.tensor(num_processing_steps, dtype=torch.int32).to(DEVICE)

# ============================================================================
# MSE LOSS FUNCTION
# ============================================================================
def mse_loss(predictions, targets):
    """
    Mean Squared Error Loss Function for Contact Force Prediction
    
    Args:
        predictions: Model output (batch_size, 1)
        targets: Ground truth (batch_size, 1)
    
    Returns:
        MSE loss value
    """
    return F.mse_loss(predictions, targets)

# ============================================================================

# ----------------------------------------------------------------------
# 5. Training Loop
# ----------------------------------------------------------------------
l1_train_hist, l2_train_hist, rho_train_hist = [], [], []
mse_train_hist, mse_test_hist = [], []  # For MSE loss tracking
l1_test_hist, l2_test_hist, rho_test_hist = [], [], []

all_predictions_train_list = []
all_targets_train_list = []
all_predictions_test_list = []
all_targets_test_list = []

print("\nStarting training (Residual Connection Version)...")

for epoch in range(num_epochs):
    # --- TRAIN ---
    model.train()
    l1s, l2s, rhos = [], [], []
    epoch_preds_train = []
    epoch_targets_train = []

    perm = np.random.permutation(len(train_graph_dicts))
    
    for i in range(0, len(train_graph_dicts), batch_size):
        idx = perm[i:i + batch_size]
        batch_graphs = [train_graph_dicts[k] for k in idx]
        batch_targets = [train_targets[k] for k in idx]

        if not batch_graphs: continue

        graphs_tuple = graph.data_dicts_to_graphs_tuple_pytorch(batch_graphs)
        graphs_tuple = move_graphs_tuple_to_device(graphs_tuple, DEVICE)
        targets_concat = torch.cat(batch_targets, dim=0).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(graphs_tuple, num_processing_steps_tensor)
        preds = outputs[-1]["nodes"] 

        loss_l1 = F.l1_loss(preds, targets_concat)
        loss_mse = mse_loss(preds, targets_concat)
        rho = graph.pearson_corr_pytorch(preds, targets_concat)

        loss_mse.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        l1s.append(loss_l1.item())
        l2s.append(loss_mse.item())  # Store MSE loss in l2s for plotting
        rhos.append(rho.item())
        
        epoch_preds_train.append(preds.detach().cpu().numpy())
        epoch_targets_train.append(targets_concat.detach().cpu().numpy())
        
        del preds, targets_concat, graphs_tuple
        torch.cuda.empty_cache()

    L1_train = np.mean(l1s)
    L2_train = np.mean(l2s)  # This is now MSE loss
    rho_train = np.mean(rhos)
    l1_train_hist.append(L1_train)
    l2_train_hist.append(L2_train)
    mse_train_hist.append(L2_train)  # Track MSE loss
    rho_train_hist.append(rho_train)
    
    if epoch_preds_train:
        all_predictions_train_list.append(np.concatenate(epoch_preds_train))
        all_targets_train_list.append(np.concatenate(epoch_targets_train))
        del epoch_preds_train, epoch_targets_train
        epoch_preds_train = []
        epoch_targets_train = []

    # --- TEST ---
    model.eval()
    l1s_t, l2s_t, rhos_t = [], [], []
    epoch_preds_test = []
    epoch_targets_test = []
    
    with torch.no_grad():
        for i in range(0, len(test_graph_dicts), batch_size):
            batch_graphs = test_graph_dicts[i:i + batch_size]
            batch_targets = test_targets[i:i + batch_size]
            
            if not batch_graphs: continue

            graphs_tuple = graph.data_dicts_to_graphs_tuple_pytorch(batch_graphs)
            graphs_tuple = move_graphs_tuple_to_device(graphs_tuple, DEVICE)
            targets_concat = torch.cat(batch_targets, dim=0).to(DEVICE)

            outputs = model(graphs_tuple, num_processing_steps_tensor)
            preds = outputs[-1]["nodes"]

            l1s_t.append(F.l1_loss(preds, targets_concat).item())
            l2s_t.append(mse_loss(preds, targets_concat).item())
            rhos_t.append(graph.pearson_corr_pytorch(preds, targets_concat).item())
            
            epoch_preds_test.append(preds.detach().cpu().numpy())
            epoch_targets_test.append(targets_concat.detach().cpu().numpy())
            
            del preds, targets_concat, graphs_tuple
            torch.cuda.empty_cache()

    L1_test = np.mean(l1s_t)
    L2_test = np.mean(l2s_t)  # This is now MSE loss
    rho_test = np.mean(rhos_t)
    l1_test_hist.append(L1_test)
    l2_test_hist.append(L2_test)
    mse_test_hist.append(L2_test)  # Track MSE loss
    rho_test_hist.append(rho_test)
    
    if epoch_preds_test:
        all_predictions_test_list.append(np.concatenate(epoch_preds_test))
        all_targets_test_list.append(np.concatenate(epoch_targets_test))
        del epoch_preds_test, epoch_targets_test
        epoch_preds_test = []
        epoch_targets_test = []

    print(f"Epoch {epoch+1:03d} | "
          f"L1: Train={L1_train:.6f}, Test={L1_test:.6f} | "
          f"MSE: Train={L2_train:.6f}, Test={L2_test:.6f} | "
          f"Rho: Train={rho_train:.4f}, Test={rho_test:.4f}")
    
    # Save Loss Plot Every Epoch
    epochs_range = np.arange(1, epoch + 2)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, l1_train_hist, label='Train L1 (MAE)', linewidth=2.5, color='green')
    plt.plot(epochs_range, l1_test_hist, label='Test L1 (MAE)', linewidth=2.5, linestyle='--', color='darkgreen')
    plt.xlabel('Epoch'); plt.ylabel('L1 Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, rho_train_hist, label='Train rho (Pearson)', linewidth=2.5, color='blue')
    plt.plot(epochs_range, rho_test_hist, label='Test rho (Pearson)', linewidth=2.5, linestyle='--', color='darkblue')
    plt.xlabel('Epoch'); plt.ylabel('Correlation (Rho)'); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, mse_train_hist, label='Train MSE Loss', linewidth=2, color='orange', alpha=0.7)
    plt.plot(epochs_range, mse_test_hist, label='Test MSE Loss', linewidth=2, linestyle='--', color='darkorange', alpha=0.7)
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, f"loss_plot_epoch_{epoch+1:03d}.png"))
    plt.close()

    all_predictions_train_list.clear(); all_targets_train_list.clear()
    all_predictions_test_list.clear(); all_targets_test_list.clear()
    torch.cuda.empty_cache()

print(f"\n✅ Training Completed. Final Test ρ: {rho_test:.4f}")
print(f"📊 Final Test MSE Loss: {L2_test:.6f}")

# ----------------------------------------------------------------------
# 6. Final Results
# ----------------------------------------------------------------------
epochs = np.arange(1, num_epochs + 1)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(epochs, l1_train_hist, label='Train L1 (MAE)', color='green', linewidth=2)
plt.plot(epochs, l1_test_hist, label='Test L1 (MAE)', linestyle='--', color='darkgreen', linewidth=2)
plt.title('MAE - Mean Absolute Error'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(epochs, rho_train_hist, label='Train ρ (Pearson)', color='blue', linewidth=2)
plt.plot(epochs, rho_test_hist, label='Test ρ (Pearson)', linestyle='--', color='darkblue', linewidth=2)
plt.title('Pearson Correlation Coefficient (ρ)'); plt.xlabel('Epoch'); plt.ylabel('Correlation'); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(epochs, mse_train_hist, label='Train MSE Loss', color='orange', linewidth=2)
plt.plot(epochs, mse_test_hist, label='Test MSE Loss', linestyle='--', color='darkorange', linewidth=2)
plt.title('Mean Squared Error Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("final_training_results_normalization.png", dpi=300)
plt.show()

torch.save(model.state_dict(), "gnn_model_normalization_final_mse_loss.pth")