# 3D Visualization and Quantitative Evaluation of Graph Neural Network Predicted Force Chains in Granular Materials

**M.Tech Thesis Project** | Indian Institute of Science, Bangalore  
**Author:** Deepak Solanki | **Supervisor:** Prof. Vijay Natarajan  
**Department:** Computer Science and Automation

---

## Overview

Force chains are localized chain-like structures that govern stress transmission in granular materials. While DEM simulations can compute inter-particle forces, they are computationally expensive. X-ray CT imaging can capture 3D particle geometry but cannot measure forces directly.

This project implements a **Graph Neural Network (GNN)** with an **Encode–Process–Decode** architecture to predict particle-scale contact forces from measurable kinematic quantities, enabling 3D force-chain identification.

### Key Contributions
- **Persistent error analysis** — identified 6 boundary failure mechanisms
- **3D visualization** — spatial comparison of predicted vs DEM force chains
- **Multi-metric evaluation** — correlation (0.79–0.85), force distribution comparison on linear and semi-log scales

---

## Project Structure

```
├── model_architecture/                        # Core GNN model code
│   ├── model.py                               # Encode-Process-Decode GNN architecture
│   ├── graph.py                               # Graph construction from DEM data
│   ├── main.py                                # Training pipeline
│   └── inference_complete.py                  # Inference on test assemblies
│
├── visualize/                                 # Visualization scripts
│   ├── export_persistent_error_vtk.py         # Persistent error VTK export for ParaView
│   └── plot_figure6_pdf_comparison.py         # NPMNCF PDF comparison (linear + semi-log)
│
└── README.md
```

---

## GNN Architecture

The model follows the **Encode–Process–Decode** paradigm:

| Component | Description |
|-----------|-------------|
| **Encoder** | Maps node features (R²) and edge features (R³) into 64-dim latent space |
| **Processor** | 7 rounds of message passing with residual connections |
| **Decoder** | Transforms final node embedding → predicted NPMNCF (scalar) |

### Input Features

| Type | Features | Dimension |
|------|----------|-----------|
| **Node** (Particle) | Normalized radius, Coordination number | R² |
| **Edge** (Contact) | Contact status, Normal displacement, Tangential displacement | R³ |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 10⁻⁴ |
| Loss function | MSE |
| Batch size | 1 |
| Epochs | 100 |
| Message-passing steps | 7 |

---

## Results

- **Correlation:** 0.79–0.85 across 3 unseen test assemblies
- **MAE:** ~0.29
- **Force distributions** match DEM on both linear and semi-log scales
- **3D visualization** confirms correct spatial location of force chains

### Persistent Error Analysis

Prediction failures concentrate at specimen boundaries due to:
1. Incomplete neighborhoods (CN ≤ 3)
2. Force chain termination gradients
3. Truncated receptive fields
4. Mean-aggregation bottleneck
5. Training data imbalance (~80% interior particles)
6. Evolving contact topology

---

## Usage

### Training
```bash
cd model_architecture
python main.py
```

### Inference
```bash
cd model_architecture
python inference_complete.py
```

### Visualization
```bash
cd visualize
python plot_figure6_pdf_comparison.py
python export_persistent_error_vtk.py
```

---

## Dependencies

- Python 3.x
- PyTorch
- PyTorch Geometric
- NumPy, Matplotlib, SciPy
- VTK

---

## References

1. Radjai et al., "Bimodal character of stress transmission in granular packings," *Physical Review Letters*, 1998.
2. Cheng & Wang, "Experimental investigation of inter-particle contact evolution of sheared granular materials," *Soils and Foundations*, 2018.
3. Cheng et al., "A machine learning-based strategy for experimentally estimating force chains of granular materials," *Géotechnique*, 2024.
4. Cheng & Wang, "Estimation of contact forces of granular materials under uniaxial compression," *Granular Matter*, 2022.
