"""
Configuration for Physics-Constrained ResNet Brain Tumor Model.

All hyperparameters, paths, and settings centralized here.

SCALE GUIDE (VRAM requirements):
    resnet10  + spatial=(64,64,64)  + batch=1 → ~4–6 GB   (any gaming GPU)
    resnet18  + spatial=(64,64,64)  + batch=2 → ~8–10 GB  (RTX 3070+)
    resnet50  + spatial=(128,128,128)+ batch=2 → ~40–80 GB (A100 only)

For LOCAL DEVELOPMENT use the defaults (resnet10, 64³).
Change backbone to resnet50 and spatial_size to (128,128,128) only for
final paper experiments on cloud GPU (Colab A100, Kaggle P100).
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""
    # --- Paths ---
    data_root: str = "./data/Task01_BrainTumour"
    train_csv: str = "./data/train.csv"
    val_csv: str = "./data/val.csv"
    test_csv: str = "./data/test.csv"

    # --- MRI Modalities ---
    modalities: List[str] = field(default_factory=lambda: ["t1", "t1ce", "t2", "flair"])
    num_channels: int = 4  # Number of input MRI modalities

    # --- Volume dimensions ---
    # DEFAULT: 64³ for local GPU training (4–6 GB VRAM)
    # For paper results, change to (128, 128, 128) on cloud GPU
    spatial_size: Tuple[int, int, int] = (64, 64, 64)  # Crop/resize target
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # mm

    # --- Segmentation labels (BraTS) ---
    num_seg_classes: int = 4  # Background, NCR/NET, ED, ET
    # BraTS label mapping: 0=background, 1=NCR/NET, 2=ED, 4=ET
    label_mapping: dict = field(default_factory=lambda: {0: 0, 1: 1, 2: 2, 4: 3})

    # --- Augmentation ---
    rand_flip_prob: float = 0.5
    rand_rotate_prob: float = 0.3
    rand_scale_prob: float = 0.3
    rand_noise_std: float = 0.1
    intensity_shift_range: float = 0.1
    intensity_scale_range: float = 0.1


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # --- Backbone ---
    # DEFAULT: resnet10 for local GPU training (~1.7M params)
    # Change to resnet50 for paper experiments on cloud GPU (~46M params)
    backbone: str = "resnet10"  # resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200
    pretrained: bool = True
    pretrained_weights_path: Optional[str] = None  # Path to MedicalNet weights

    # --- Feature dimensions ---
    # MUST match backbone: resnet10/18/34 → 512, resnet50/101/152/200 → 2048
    feature_dim: int = 512   # resnet10 output features
    hidden_dim: int = 256

    # --- Tumor Density Decoder ---
    # Scaled down for resnet10; use [256,128,64,32,16] for resnet50
    decoder_channels: List[int] = field(default_factory=lambda: [128, 64, 32, 16, 8])
    decoder_use_skip: bool = True  # Use skip connections from encoder

    # --- Physics Parameter Head ---
    predict_diffusion: bool = True  # Predict D(x) spatially
    predict_proliferation: bool = True  # Predict ρ
    diffusion_range: Tuple[float, float] = (0.0001, 0.5)  # mm²/day
    proliferation_range: Tuple[float, float] = (0.001, 0.5)  # 1/day
    carrying_capacity: float = 1.0  # K in logistic growth

    # --- Segmentation head (for pretraining) ---
    use_seg_head: bool = True
    num_seg_classes: int = 4

    # --- Dropout ---
    dropout_rate: float = 0.3


@dataclass
class PhysicsConfig:
    """
    Physics-constrained loss configuration.

    PDE: Fisher-KPP reaction-diffusion equation (Swanson et al. 2000)
        ∂u/∂t = ∇·(D(x)∇u) + ρ·u·(1 − u)

    Biophysically grounded default parameters:
        D_wm  ≈ 0.05–0.10 mm²/day (white matter, Harpold et al. 2007)
        D_gm  ≈ 0.01–0.02 mm²/day (grey matter, ≈5x lower)
        ρ     ≈ 0.012–0.025 /day  (Swanson et al. 2000)
    """
    # --- PDE Model ---
    pde_model: str = "fisher_kpp"  # "fisher_kpp" or "gompertz"

    # --- Default physics parameters (used if not predicted by network) ---
    # Source: Swanson et al. (2000), Harpold et al. (2007)
    default_diffusion: float = 0.05   # mm²/day — conservative/stable default
    default_proliferation: float = 0.02  # 1/day
    default_carrying_capacity: float = 1.0

    # --- Spatial resolution for finite differences ---
    dx: float = 1.0  # voxel spacing in mm (x)
    dy: float = 1.0  # voxel spacing in mm (y)
    dz: float = 1.0  # voxel spacing in mm (z)

    # --- Temporal ---
    # Default Δt for synthetic longitudinal pairs (days)
    dt: float = 90.0  # ~3 months — typical glioma re-scan interval
    synthetic_delta_t: float = 90.0  # days for PDE simulation in training

    # --- Tissue-aware diffusion (Harpold et al. 2007) ---
    # White matter is ~5-10x more diffusive than grey matter for glioma
    tissue_aware_diffusion: bool = True
    D_white_matter: float = 0.08   # mm²/day (white matter diffusion)
    D_grey_matter: float = 0.016   # mm²/day (grey matter diffusion)
    # Ratio: D_wm / D_gm ≈ 5 (biophysically validated)
    diffusion_ratio_wm_gm: float = 5.0

    # --- Boundary conditions ---
    boundary_type: str = "neumann"  # "neumann" (no-flux) — standard for glioma

    # --- Constraints ---
    enforce_non_negativity: bool = True
    enforce_upper_bound: bool = True
    upper_bound_value: float = 1.0
    epsilon: float = 1e-7  # Small constant to avoid log(0) in Gompertz


@dataclass
class LossConfig:
    """Loss function weights and configuration."""
    # --- Loss weights ---
    lambda_data: float = 1.0       # Data fidelity loss
    lambda_pde: float = 0.1        # PDE residual loss
    lambda_ic: float = 0.05        # Initial condition loss
    lambda_bc: float = 0.01        # Boundary condition loss
    lambda_reg: float = 0.01       # Regularization loss
    lambda_seg: float = 1.0        # Segmentation loss (pretraining)

    # --- Adaptive weighting ---
    use_adaptive_weights: bool = True
    adaptive_method: str = "grad_norm"  # "grad_norm", "uncertainty", "manual"

    # --- Data loss type ---
    data_loss_type: str = "mse"  # "mse", "l1", "huber"
    seg_loss_type: str = "dice_ce"  # "dice", "ce", "dice_ce", "focal"

    # --- Regularization ---
    smoothness_weight: float = 0.001
    l2_weight_decay: float = 1e-5


@dataclass
class TrainConfig:
    """Training configuration."""
    # --- Phase ---
    phase: str = "pretrain"  # "pretrain" (seg only), "finetune" (physics), "joint"

    # --- Optimization ---
    # batch_size=1 for local (resnet10+64³); increase to 2 on cloud GPU (resnet50+128³)
    batch_size: int = 1
    accumulation_steps: int = 4  # Simulates larger batch size (1 * 4 = effect of batch 4)
    num_epochs_pretrain: int = 1
    num_epochs_finetune: int = 1
    learning_rate: float = 1e-4
    lr_backbone: float = 1e-5  # Lower LR for pretrained backbone
    weight_decay: float = 1e-5
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"

    # --- Scheduler ---
    scheduler: str = "cosine"  # "cosine", "step", "plateau", "warmup_cosine", "cosine_warm_restarts"
    warmup_epochs: int = 10
    min_lr: float = 1e-7

    # --- Mixed precision ---
    use_amp: bool = True
    grad_clip_norm: float = 1.0

    # --- Checkpointing ---
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10
    resume_from: Optional[str] = None

    # --- Logging ---
    log_dir: str = "./logs"
    use_wandb: bool = False
    wandb_project: str = "brain-tumor-pinn"

    # --- Validation ---
    val_every: int = 5
    early_stopping_patience: int = 30

    # --- Device ---
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True

    # --- Reproducibility ---
    seed: int = 42


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.train.checkpoint_dir, exist_ok=True)
        os.makedirs(self.train.log_dir, exist_ok=True)


def get_config(**overrides) -> Config:
    """Get configuration with optional overrides."""
    config = Config()
    for key, value in overrides.items():
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return config
