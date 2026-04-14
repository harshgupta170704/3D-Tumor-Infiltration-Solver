# 🧠 Physics-Constrained ResNet for Brain Tumor Segmentation & Biophysical Parameter Estimation

A deep learning model that combines **MONAI 3D ResNet** with **Fisher-KPP reaction-diffusion physics constraints** for simultaneous brain tumor segmentation and glioma growth parameter estimation.

> **Scientific Framing Note**: This model is a **physics-regularized (physics-constrained) deep learning** approach — NOT a classical PINN (Physics-Informed Neural Network). Classical PINNs use MLP networks with collocation point sampling. Here, a 3D CNN processes a voxel grid with soft PDE constraints enforced as loss penalties. This distinction is important for accurate academic framing.

---

## Scientific Background

### Physics Model — Fisher-KPP Reaction-Diffusion (Swanson et al. 2000)

```
∂u/∂t = ∇·(D(x)∇u) + ρ·u·(1 − u)
```

| Symbol | Meaning | Biophysical range |
|--------|---------|-------------------|
| `u(x,t)` | Normalized tumor cell density ∈ [0,1] | — |
| `D(x)` | Spatially-varying diffusion coefficient | 0.001–0.10 mm²/day (Harpold 2007) |
| `ρ(x)` | Local proliferation rate | 0.012–0.025 /day (Swanson 2000) |

Also supports **Gompertz growth**: `∂u/∂t = ∇·(D(x)∇u) + ρ·u·ln(K/(u+ε))`

### Tissue-Aware Diffusion

White matter is **5–10× more diffusive** than grey matter for glioma spread (Harpold et al. 2007):

- `D_wm ≈ 0.08 mm²/day` (white matter)
- `D_gm ≈ 0.016 mm²/day` (grey matter)

### Tumor Density Ground Truth Construction

Since MRI does not directly measure cell density, we construct `u(x, t₀)` from BraTS segmentation labels using a **biophysically grounded mapping** (Swanson 2000; Hormuth 2021):

| BraTS Region | Label | Assigned Density `u` |
|---|---|---|
| Background | 0 | 0.00 |
| Edema (ED) | 2 | 0.20 (sparse infiltration) |
| Necrotic Core (NCR) | 1 | 0.60 (cell debris) |
| Enhancing Tumor (ET) | 3 | 1.00 (highest cell density) |

### Synthetic Longitudinal Data

BraTS 2021 is **cross-sectional** (single time point per patient). To provide temporal supervision for the PDE constraint (`∂u/∂t`), we **numerically simulate** tumor growth using the Fisher-KPP PDE forward in time (RK4 integrator) to generate synthetic `u(t₂)`.

> **This must be stated clearly in any paper**: results use synthetically generated longitudinal pairs, not real multi-timepoint patient data.

This approach follows established computational oncology methods: Mang et al. (2019), Lipkova et al. (2019).

---

## Architecture

```
MRI Input [B, 4, 64, 64, 64] (T1, T1ce, T2, FLAIR)
         │
    ┌────▼─────┐
    │  MONAI   │  ← resnet10/18/50 (3D)
    │  ResNet  │  ← Multi-scale skip features
    └──┬─┬─┬───┘
       │ │ │
  ┌────┘ │ └────┐
  ▼      ▼      ▼
┌────┐ ┌────┐ ┌────┐
│ u  │ │D, ρ│ │Seg │
│    │ │    │ │    │  ← Three output heads
└──┬─┘ └──┬─┘ └──┬─┘

Physics Loss: L = λ_data·L_data + λ_PDE·L_PDE + λ_IC·L_IC + λ_BC·L_BC + λ_reg·L_reg
```

---

## Loss Function

| Loss | Formula | Scientific Purpose |
|------|---------|-------------------|
| `L_data` | MSE(u_pred, u_target) | Match biophysical density from segmentation |
| `L_PDE` | ‖∂u/∂t − ∇·(D∇u) − ρu(1−u)‖² | Enforce Fisher-KPP equation |
| `L_IC` | ‖u(t₀) − u₀‖² | Match initial tumor state |
| `L_BC` | ‖∇u·n‖² at brain boundary | No-flux Neumann BC (tumor stays in brain) |
| `L_reg` | Non-negativity + boundedness + smoothness | Physical constraints on u, D, ρ |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Scientific Correctness

```bash
python validate_physics.py
```

All tests must pass before training.

### 📦 3. Data Preparation

Because BraTS 2021 requires registration and an API key that frequently changes, this repository uses **Medical Segmentation Decathlon (MSD) Task01_BrainTumour**, which is scientifically identical to BraTS (4 modalities: T1, T1ce, T2, FLAIR + identical expert segmentation formats) but allows direct, automatic downloading.

**Important Label Note**: The MSD maps enhancing tumor to label `3` (whereas BraTS uses `4`). Our `preprocessing.py` logic safely handles both formats automatically.

It takes ~15 minutes to download/extract (~4.6 GB).

```bash
python download_data.py
```

This will create a `data/Task01_BrainTumour/` directory with `imagesTr` and `labelsTr`.

### Scale & VRAM Note
By default, the project config is set to use a `resnet10` backbone with a `(64, 64, 64)` spatial crop. This requires only **~4-6GB VRAM**, meaning it will run easily on your local machine.

*For final research/paper runs:* you should update `config.py` to use `(128, 128, 128)` and `resnet50`, which will require a cloud GPU (e.g., A100 or Kaggle P100) and take ~40-80GB VRAM.

### 4. Train

The code utilizes **Two-Phase Training**, an established optimization strategy for physics-informed neural networks.

### Phase 1: Pretraining (Data-Driven)
Focuses entirely on representation learning. The `resnet10` encoder and segmentation head learn to extract dense tumor features.

```bash
python train.py --data_root ./data/Task01_BrainTumour --phase pretrain --epochs 50
```

### Phase 2: Fine-Tuning (Physics-Informed)
Turns on the `Tumor Density`, `Diffusion`, and `Proliferation` decoder heads. The PDE loss is activated. Since MSD is cross-sectional, the `FisherKPPSolver` dynamically generates synthetic temporal evolution tuples $(\Delta t, u_{t_2})$ on the fly to supervise the physical constraint without requiring real longitudinal scanning.

```bash
python train.py --data_root ./data/Task01_BrainTumour --phase finetune --resume checkpoints/pretrain_best.pth --epochs 200
```

**For paper experiments (cloud GPU — A100 recommended):**
```bash
python train.py --data_root ./data/Task01_BrainTumour --phase joint --backbone resnet50 --epochs 300 --spatial_size 128 128 128 --batch_size 2
```

### 5. Predict

```bash
python predict.py \
    --input_dir ./data/BraTS2021/BraTS2021_00000 \
    --checkpoint checkpoints/finetune_best.pth \
    --simulate_days 180 \
    --output_dir ./predictions
```

---

## Scale Guide (VRAM Requirements)

| Backbone | Volume | Batch | VRAM | Hardware |
|----------|--------|-------|------|----------|
| resnet10 | 64³ | 1 | ~4–6 GB | Any gaming GPU |
| resnet18 | 64³ | 2 | ~8–10 GB | RTX 3070+ |
| resnet50 | 128³ | 2 | ~40–80 GB | A100 (cloud) |

**Default is resnet10 + 64³ for local development.**

---

## Project Structure

```
├── config.py                      # All hyperparameters (scale guide included)
├── train.py                       # Training script (2-phase)
├── predict.py                     # Inference + growth simulation
├── validate_physics.py            # ★ Scientific validation tests (run first!)
├── requirements.txt               # Dependencies
│
├── models/
│   ├── resnet_backbone.py         # MONAI ResNet encoder (multi-scale features)
│   ├── decoder.py                 # Density decoder + physics head + seg head
│   ├── attention.py               # Attention gates (Oktay 2018)
│   └── hybrid_model.py            # Complete PhysicsConstrainedTumorNet
│
├── losses/
│   ├── physics_loss.py            # PDE residual, IC, BC, tissue-aware diffusion
│   ├── data_loss.py               # Biophysical seg→density + segmentation losses
│   └── combined_loss.py           # Hybrid loss with synthetic du/dt generation
│
├── data/
│   ├── dataset.py                 # BraTS dataset loader
│   ├── preprocessing.py           # MONAI transforms pipeline
│   └── synthetic_longitudinal.py  # ★ Fisher-KPP PDE simulator (critical fix!)
│
└── utils/
    ├── spatial_ops.py             # 3D finite-difference gradients & Laplacian
    └── metrics.py                 # Dice, Hausdorff, growth error, PDE residual
```

---

## Evaluation Metrics

- **Dice Score (WT, TC, ET)**: Segmentation quality
- **Hausdorff Distance (95%)**: Surface-to-surface segmentation error
- **Growth MSE/MAE**: Density prediction accuracy vs. synthetic t₂
- **PDE Residual L²**: How well predictions satisfy the Fisher-KPP equation
- **Volume Error**: Predicted vs. actual tumor volume

---

## Known Limitations (Required for Scientific Honesty)

> ⚠️ **Longitudinal data**: BraTS 2021 is cross-sectional. Temporal supervision uses numerically simulated `u(t₂)` — not real longitudinal MRI. All results must be framed as "physics-constrained" estimation, not growth prediction from observed pairs.

> ⚠️ **Density observability**: MRI does not directly measure cell density. The density field is a theoretical construct inferred from segmentation subregions. Uncertainty quantification of density estimates is a future direction.

> ⚠️ **Tissue segmentation**: Tissue-aware diffusion uses a tumor-derived proxy for white-matter segmentation. For full accuracy, dedicated tissue segmentation (e.g., FSL FAST) should be used.

---

## Configuration

Key settings in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone` | resnet10 | ResNet variant (change to resnet50 for paper) |
| `spatial_size` | (64,64,64) | Volume dimensions (128³ for paper) |
| `pde_model` | fisher_kpp | PDE type |
| `lambda_pde` | 0.1 | PDE loss weight |
| `batch_size` | 1 | Training batch size |
| `default_diffusion` | 0.05 | D default mm²/day (Swanson 2000) |
| `default_proliferation` | 0.02 | ρ default 1/day (Swanson 2000) |

---

## References

1. Swanson, K.R. et al. (2000). *A mathematical modelling tool for predicting survival of individual patients following resection of glioblastoma.* Br. J. Cancer.
2. Harpold, H.L. et al. (2007). *The evolution of mathematical modeling of glioma proliferation and invasion.* J. Neuropathol. Exp. Neurol.
3. Lipkova, J. et al. (2019). *Personalized Radiotherapy Design for Glioblastoma.* IEEE Trans. Med. Imag.
4. Mang, A. et al. (2019). *CLAIRE: A distributed-memory solver for constrained large diffeomorphic image registration.* SIAM J. Sci. Comput.
5. Oktay, O. et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas.* MIDL.
6. Raissi, M. et al. (2019). *Physics-informed neural networks.* J. Comput. Phys.
7. Menze, B. et al. (2015). *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).* IEEE Trans. Med. Imag.
