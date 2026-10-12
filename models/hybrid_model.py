"""
HybridTumorNet — The complete hybrid MONAI ResNet + Physics-Informed model.

Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │  Input: 4-channel MRI [B, 4, 128, 128, 128]              │
    │         (T1, T1ce, T2, FLAIR)                             │
    │                   │                                       │
    │      ┌────────────▼────────────┐                          │
    │      │   MONAI 3D ResNet-50    │                          │
    │      │   (Encoder/Backbone)    │                          │
    │      │      + Skip features    │                          │
    │      └──┬──────┬──────┬───────┘                           │
    │         │      │      │                                   │
    │    ┌────▼──┐ ┌─▼────┐ ┌▼─────────┐                       │
    │    │Density│ │Phys. │ │Segment.  │                        │
    │    │Decoder│ │Params│ │Head      │                        │
    │    │       │ │D(x),ρ│ │(pretrain)│                        │
    │    └───┬───┘ └──┬───┘ └────┬─────┘                        │
    │        │        │          │                               │
    │   u(x,t)   D(x), ρ(x)   seg_logits                       │
    │   [0,1]    positive      [num_cls]                        │
    └───────────────────────────────────────────────────────────┘

Output dict:
    - "tumor_density": u(x,t) ∈ [0,1]   [B, 1, D, H, W]
    - "diffusion":     D(x) > 0          [B, 1, D, H, W]
    - "proliferation": ρ(x) > 0          [B, 1, D, H, W]
    - "segmentation":  logits            [B, C, D, H, W]  (optional)
"""

import torch
import torch.nn as nn

from .resnet_backbone import ResNetEncoder
from .decoder import TumorDensityDecoder, PhysicsParameterHead, SegmentationHead


class HybridTumorNet(nn.Module):
    """
    Hybrid deep learning + physics-informed model for brain tumor analysis.
    
    Combines:
        1. MONAI 3D ResNet encoder for feature extraction from multi-modal MRI
        2. Dense decoder for tumor density u(x,t) prediction
        3. Physics parameter prediction (spatially varying D(x), ρ(x))
        4. Optional segmentation head for BraTS pretraining
    
    The model outputs are fed into physics-informed losses that enforce
    the reaction–diffusion tumor growth equation.
    """

    def __init__(self, 
                 resnet_variant: str = "resnet50",
                 in_channels: int = 4,
                 num_seg_classes: int = 4,
                 pretrained_path: str = None,
                 freeze_encoder_layers: int = 0,
                 decoder_channels: tuple = (256, 128, 64, 32, 16),
                 dropout: float = 0.1,
                 use_seg_head: bool = True,
                 predict_physics_params: bool = True,
                 diffusion_range: tuple = (0.0001, 0.5),
                 proliferation_range: tuple = (0.001, 0.5)):
        """
        Args:
            resnet_variant: Which ResNet to use ("resnet18", "resnet50", etc.).
            in_channels: Number of MRI modalities (default 4).
            num_seg_classes: Number of segmentation classes (BraTS = 4).
            pretrained_path: Path to pretrained weight file (.pth).
            freeze_encoder_layers: Number of encoder layers to freeze (0=none).
            decoder_channels: Channel dimensions for decoder levels.
            dropout: Dropout rate.
            use_seg_head: Whether to include segmentation head.
            predict_physics_params: Whether to predict D(x), ρ(x) spatially.
            diffusion_range: (min, max) for diffusion coefficient.
            proliferation_range: (min, max) for proliferation rate.
        """
        super().__init__()

        self.use_seg_head = use_seg_head
        self.predict_physics_params = predict_physics_params

        # === ENCODER ===
        self.encoder = ResNetEncoder(
            variant=resnet_variant,
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            freeze_layers=freeze_encoder_layers,
        )
        enc_channels = self.encoder.get_channels()

        # === TUMOR DENSITY DECODER ===
        self.density_decoder = TumorDensityDecoder(
            encoder_channels=enc_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )

        # === PHYSICS PARAMETER HEAD ===
        if predict_physics_params:
            self.physics_head = PhysicsParameterHead(
                encoder_channels=enc_channels,
                diffusion_range=diffusion_range,
                proliferation_range=proliferation_range,
            )
        else:
            self.physics_head = None

        # === SEGMENTATION HEAD (for pretraining) ===
        if use_seg_head:
            self.seg_head = SegmentationHead(
                encoder_channels=enc_channels,
                num_classes=num_seg_classes,
                decoder_channels=decoder_channels,
                dropout=dropout,
            )
        else:
            self.seg_head = None

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        """
        Forward pass through the hybrid model.

        Args:
            x: Input MRI tensor [B, 4, D, H, W].
            return_features: If True, include encoder features in output.

        Returns:
            Dictionary with keys:
                "tumor_density":  [B, 1, D, H, W] ∈ [0, 1]
                "diffusion":      [B, 1, D, H, W] > 0  (if predict_physics_params)
                "proliferation":  [B, 1, D, H, W] > 0  (if predict_physics_params)
                "segmentation":   [B, C, D, H, W]       (if use_seg_head)
                "features":       dict of encoder features (if return_features)
        """
        # Encode
        features = self.encoder(x)

        output = {}
        
        return_ds = self.training

        # Tumor density prediction
        tumor_density = self.density_decoder(features, return_ds=return_ds)
        output["tumor_density"] = tumor_density

        # Physics parameters
        if self.predict_physics_params and self.physics_head is not None:
            td_base = tumor_density[0] if isinstance(tumor_density, list) else tumor_density
            target_size = td_base.shape[2:]  # (D, H, W)
            D, rho = self.physics_head(features, target_size=target_size)
            output["diffusion"] = D
            output["proliferation"] = rho

        # Segmentation
        if self.use_seg_head and self.seg_head is not None:
            seg_logits = self.seg_head(features, return_ds=return_ds)
            output["segmentation"] = seg_logits

        # Optional: return encoder features
        if return_features:
            output["features"] = features

        return output

    def get_parameter_groups(self, lr_backbone: float = 1e-5, lr_head: float = 1e-4):
        """
        Get parameter groups with different learning rates.
        Lower LR for pretrained backbone, higher for new heads.

        Returns:
            List of parameter group dicts for optimizer.
        """
        backbone_params = list(self.encoder.parameters())
        head_params = []
        head_params.extend(list(self.density_decoder.parameters()))
        if self.physics_head is not None:
            head_params.extend(list(self.physics_head.parameters()))
        if self.seg_head is not None:
            head_params.extend(list(self.seg_head.parameters()))

        return [
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": head_params, "lr": lr_head, "name": "heads"},
        ]

    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def set_phase(self, phase: str):
        """
        Configure model for a specific training phase.

        Args:
            phase: "pretrain" (seg only), "finetune" (physics), or "joint" (all).
        """
        if phase == "pretrain":
            # Only segmentation head active, freeze physics parts
            self.use_seg_head = True
            if self.seg_head:
                for p in self.seg_head.parameters():
                    p.requires_grad = True
            if self.physics_head:
                for p in self.physics_head.parameters():
                    p.requires_grad = False
            for p in self.density_decoder.parameters():
                p.requires_grad = False

        elif phase == "finetune":
            # Physics-informed training, freeze seg head
            self.unfreeze_encoder()
            for p in self.density_decoder.parameters():
                p.requires_grad = True
            if self.physics_head:
                for p in self.physics_head.parameters():
                    p.requires_grad = True
            if self.seg_head:
                for p in self.seg_head.parameters():
                    p.requires_grad = False

        elif phase == "joint":
            # Everything trainable
            self.unfreeze_encoder()
            for p in self.parameters():
                p.requires_grad = True

        else:
            raise ValueError(f"Unknown phase: {phase}")

    @torch.no_grad()
    def predict_tumor_growth(self, x: torch.Tensor, 
                              num_steps: int = 10,
                              dt: float = 1.0,
                              integration_method: str = "rk4") -> list:
        """
        Predict tumor evolution over time using the learned parameters.
        
        Uses RK4 or explicit Euler integration of the reaction-diffusion PDE:
            ∂u/∂t = ∇·(D∇u) + ρu(1-u)

        Args:
            x: Input MRI [B, 4, D, H, W].
            num_steps: Number of time steps to simulate.
            dt: Time step size (days).
            integration_method: "rk4" or "euler"

        Returns:
            List of tumor density fields [B, 1, D, H, W] at each time step.
        """
        from utils.spatial_ops import SpatialGradients3D
        
        self.eval()
        output = self.forward(x)
        
        u = output["tumor_density"]
        D = output.get("diffusion", torch.full_like(u, 0.1))
        rho = output.get("proliferation", torch.full_like(u, 0.025))

        spatial_ops = SpatialGradients3D(dx=1.0, dy=1.0, dz=1.0).to(x.device)

        def get_du_dt(u_current):
            diffusion = spatial_ops.divergence_of_flux(u_current, D)
            reaction = rho * u_current * (1.0 - u_current)
            return diffusion + reaction

        trajectory = [u.clone()]

        for step in range(num_steps):
            if integration_method == "rk4":
                k1 = get_du_dt(u)
                k2 = get_du_dt(u + 0.5 * dt * k1)
                k3 = get_du_dt(u + 0.5 * dt * k2)
                k4 = get_du_dt(u + dt * k3)
                u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                # Euler step
                u = u + dt * get_du_dt(u)
            
            # Enforce constraints
            u = torch.clamp(u, 0.0, 1.0)
            
            trajectory.append(u.clone())

        return trajectory

    def count_parameters(self) -> dict:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.density_decoder.parameters())
        
        info = {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "encoder": encoder_params,
            "density_decoder": decoder_params,
        }
        
        if self.physics_head:
            info["physics_head"] = sum(p.numel() for p in self.physics_head.parameters())
        if self.seg_head:
            info["seg_head"] = sum(p.numel() for p in self.seg_head.parameters())
        
        return info
