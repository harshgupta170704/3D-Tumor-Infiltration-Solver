"""
Data-driven loss functions for supervised learning.
Segmentation losses, density regression, and seg-to-density conversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, DiceCELoss, FocalLoss


class SegmentationLoss(nn.Module):
    """Segmentation loss: Dice, CE, DiceCE, or Focal."""

    def __init__(self, loss_type="dice_ce", num_classes=4,
                 include_background=False):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "dice":
            self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True,
                                     include_background=include_background)
        elif loss_type == "ce":
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == "dice_ce":
            self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True,
                                       include_background=include_background)
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(to_onehot_y=True, gamma=2.0,
                                      include_background=include_background)

    def forward(self, pred, target):
        """pred: [B,C,D,H,W] logits, target: [B,1,D,H,W] labels."""
        if self.loss_type == "ce":
            return self.loss_fn(pred, target.squeeze(1).long())
        return self.loss_fn(pred, target.long())


class DensityRegressionLoss(nn.Module):
    """MSE/L1/Huber loss for tumor density regression."""

    def __init__(self, loss_type="mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, mask=None):
        """pred, target: [B,1,D,H,W]. mask: optional brain mask."""
        diff = pred - target
        if mask is not None:
            diff = diff * mask
            n = mask.sum().clamp(min=1)
        else:
            n = diff.numel()

        if self.loss_type == "mse":
            return (diff ** 2).sum() / n
        elif self.loss_type == "l1":
            return diff.abs().sum() / n
        elif self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target, reduction="mean")
        return (diff ** 2).sum() / n


def seg_to_density(seg_mask, sigma: float = 1.5, smooth: bool = True):
    """
    Convert BraTS segmentation mask to a continuous tumor cell density field.

    SCIENTIFIC BASIS (Swanson et al. 2000; Harpold et al. 2007):
    ---------------------------------------------------------------
    BraTS tumor regions have different expected cell densities:

        - ET  (Enhanced Tumor,  class 3): u ≈ 1.0
          Contrast-enhancing region: highest intact cell density,
          active tumor growth, high vascularization.

        - NCR (Necrotic Core,   class 1): u ≈ 0.6
          Necrotic/Non-enhancing core: high initial cell mass, but
          cells are dead/dying. We assign moderate density (cell debris).

        - ED  (Edema,           class 2): u ≈ 0.2
          Infiltrative peritumoral edema: sparse tumor cell invasion
          into surrounding brain tissue. Cells are present but sparse.

        - BG  (Background,      class 0): u = 0.0
          No tumor cells.

    This mapping is used as the INITIAL CONDITION u(x, t₀) for the PDE.
    Optional Gaussian smoothing creates a spatially smooth density field
    (avoids sharp discontinuities that cause large PDE residuals at borders).

    Args:
        seg_mask: [B, 1, D, H, W] integer labels ∈ {0, 1, 2, 3}.
                  (After BraTS label remapping: 0=BG, 1=NCR, 2=ED, 3=ET)
        sigma: Gaussian smoothing kernel width (mm). Set 0 to disable.
        smooth: Whether to apply Gaussian smoothing at region boundaries.

    Returns:
        [B, 1, D, H, W] float ∈ [0, 1] — tumor cell density.
    """
    seg = seg_mask.float()
    device = seg.device

    # Biophysically grounded density mapping
    density = torch.zeros_like(seg)
    density = torch.where(seg == 2, torch.tensor(0.2, device=device), density)  # ED
    density = torch.where(seg == 1, torch.tensor(0.6, device=device), density)  # NCR
    density = torch.where(seg == 3, torch.tensor(1.0, device=device), density)  # ET

    # Optional: Gaussian smoothing at region boundaries to avoid sharp jumps
    # This is a SMOOTHING operation, NOT normalization
    if smooth and sigma > 0:
        ks = max(3, int(4 * sigma + 1))
        if ks % 2 == 0:
            ks += 1
        pad = ks // 2

        x = torch.arange(ks, dtype=torch.float32, device=device)
        g = torch.exp(-(x - pad) ** 2 / (2.0 * sigma ** 2))
        g = g / g.sum()  # Normalize kernel (not the density)

        density = F.conv3d(density, g.view(1, 1, -1, 1, 1), padding=(pad, 0, 0))
        density = F.conv3d(density, g.view(1, 1, 1, -1, 1), padding=(0, pad, 0))
        density = F.conv3d(density, g.view(1, 1, 1, 1, -1), padding=(0, 0, pad))
        # Clamp: smoothing can slightly exceed [0,1] at boundaries
        density = torch.clamp(density, 0.0, 1.0)

    return density
