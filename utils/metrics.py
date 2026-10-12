"""
Evaluation metrics for brain tumor analysis.

Includes:
    - Dice Score (per-class and mean)
    - Hausdorff Distance (95th percentile)
    - Growth Prediction Error (MSE, MAE on tumor density)
    - Physics Residual Error (PDE residual norm)
    - Volume-based metrics
"""

import numpy as np
import torch
from scipy import ndimage


def compute_dice(pred: torch.Tensor, target: torch.Tensor, 
                 num_classes: int = 4, include_background: bool = False,
                 smooth: float = 1e-5) -> dict:
    """
    Compute Dice similarity coefficient per class.

    Args:
        pred: Predicted segmentation [B, num_classes, D, H, W] (probabilities or logits).
        target: Ground truth segmentation [B, 1, D, H, W] (integer labels) or one-hot.
        num_classes: Number of segmentation classes.
        include_background: Whether to include background in mean Dice.
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Dictionary with per-class Dice and mean Dice.
    """
    if pred.dim() == 5 and pred.shape[1] > 1:
        pred = torch.argmax(pred, dim=1)  # [B, D, H, W]
    elif pred.dim() == 5:
        pred = pred.squeeze(1)

    if target.dim() == 5:
        target = target.squeeze(1)

    dice_scores = {}
    start_class = 0 if include_background else 1

    for c in range(start_class, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores[f"dice_class_{c}"] = dice.item()

    # BraTS-specific regions
    if num_classes == 4:
        # Whole Tumor (WT): classes 1, 2, 3 (NCR/NET + ED + ET)
        pred_wt = ((pred == 1) | (pred == 2) | (pred == 3)).float()
        target_wt = ((target == 1) | (target == 2) | (target == 3)).float()
        dice_wt = (2.0 * (pred_wt * target_wt).sum() + smooth) / (pred_wt.sum() + target_wt.sum() + smooth)
        dice_scores["dice_WT"] = dice_wt.item()

        # Tumor Core (TC): classes 1, 3 (NCR/NET + ET)
        pred_tc = ((pred == 1) | (pred == 3)).float()
        target_tc = ((target == 1) | (target == 3)).float()
        dice_tc = (2.0 * (pred_tc * target_tc).sum() + smooth) / (pred_tc.sum() + target_tc.sum() + smooth)
        dice_scores["dice_TC"] = dice_tc.item()

        # Enhancing Tumor (ET): class 3
        pred_et = (pred == 3).float()
        target_et = (target == 3).float()
        dice_et = (2.0 * (pred_et * target_et).sum() + smooth) / (pred_et.sum() + target_et.sum() + smooth)
        dice_scores["dice_ET"] = dice_et.item()

        dice_scores["dice_mean"] = np.mean([dice_wt.item(), dice_tc.item(), dice_et.item()])
    else:
        vals = [v for k, v in dice_scores.items() if k.startswith("dice_class")]
        dice_scores["dice_mean"] = np.mean(vals) if vals else 0.0

    return dice_scores


def compute_hausdorff(pred: np.ndarray, target: np.ndarray, 
                      percentile: float = 95.0,
                      voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """
    Compute the Hausdorff distance (at a given percentile) between
    predicted and target binary masks.

    Args:
        pred: Binary prediction mask [D, H, W].
        target: Binary ground truth mask [D, H, W].
        percentile: Percentile for robust Hausdorff (default 95).
        voxel_spacing: Physical voxel spacing in mm.

    Returns:
        Hausdorff distance in mm.
    """
    if np.sum(pred) == 0 and np.sum(target) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return np.inf

    # Compute distance transforms
    pred_border = _get_surface_points(pred)
    target_border = _get_surface_points(target)

    # Get surface point coordinates
    pred_coords = np.argwhere(pred_border) * np.array(voxel_spacing)
    target_coords = np.argwhere(target_border) * np.array(voxel_spacing)

    if len(pred_coords) == 0 or len(target_coords) == 0:
        return np.inf

    # Compute distances from pred surface to target surface and vice versa
    from scipy.spatial.distance import cdist
    d_pred_to_target = cdist(pred_coords, target_coords).min(axis=1)
    d_target_to_pred = cdist(target_coords, pred_coords).min(axis=1)

    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return np.percentile(all_distances, percentile)


def _get_surface_points(mask: np.ndarray) -> np.ndarray:
    """Extract surface voxels of a binary mask using erosion."""
    struct = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask, structure=struct)
    return mask.astype(bool) ^ eroded.astype(bool)


def compute_growth_prediction_error(pred_density: torch.Tensor, 
                                     target_density: torch.Tensor,
                                     mask: torch.Tensor = None) -> dict:
    """
    Compute tumor growth prediction errors.

    Args:
        pred_density: Predicted tumor density u(x,t₂) [B, 1, D, H, W].
        target_density: Observed tumor density at t₂ [B, 1, D, H, W].
        mask: Optional brain mask [B, 1, D, H, W].

    Returns:
        Dictionary with MSE, MAE, volume error, and spatial correlation.
    """
    if mask is not None:
        pred_masked = pred_density * mask
        target_masked = target_density * mask
        n_voxels = mask.sum().clamp(min=1)
    else:
        pred_masked = pred_density
        target_masked = target_density
        n_voxels = pred_density.numel()

    # MSE
    mse = ((pred_masked - target_masked) ** 2).sum() / n_voxels

    # MAE
    mae = (pred_masked - target_masked).abs().sum() / n_voxels

    # Volume error (in voxels)
    pred_volume = (pred_masked > 0.5).float().sum()
    target_volume = (target_masked > 0.5).float().sum()
    volume_error = (pred_volume - target_volume).abs()
    volume_relative_error = volume_error / target_volume.clamp(min=1)

    # Spatial Pearson correlation
    pred_flat = pred_masked.flatten()
    target_flat = target_masked.flatten()
    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()
    correlation = (pred_centered * target_centered).sum() / (
        pred_centered.norm() * target_centered.norm() + 1e-8
    )

    return {
        "growth_mse": mse.item(),
        "growth_mae": mae.item(),
        "volume_error_voxels": volume_error.item(),
        "volume_relative_error": volume_relative_error.item(),
        "spatial_correlation": correlation.item(),
    }


def compute_physics_residual(u: torch.Tensor, D: torch.Tensor, rho: torch.Tensor,
                              spatial_ops, du_dt: torch.Tensor = None,
                              pde_model: str = "fisher_kpp",
                              mask: torch.Tensor = None) -> dict:
    """
    Compute how well the predicted tumor density satisfies the PDE.

    Args:
        u: Predicted tumor density [B, 1, D, H, W].
        D: Diffusion coefficient [B, 1, D, H, W].
        rho: Proliferation rate [B, 1, D, H, W] or scalar.
        spatial_ops: SpatialGradients3D instance.
        du_dt: Temporal derivative [B, 1, D, H, W], or None (assumes steady state).
        pde_model: "fisher_kpp" or "gompertz".
        mask: Optional brain mask.

    Returns:
        Dictionary with mean and max PDE residual.
    """
    # Compute diffusion term: ∇·(D∇u)
    diffusion_term = spatial_ops.divergence_of_flux(u, D)

    # Compute reaction term
    if pde_model == "fisher_kpp":
        reaction_term = rho * u * (1.0 - u)
    elif pde_model == "gompertz":
        eps = 1e-7
        reaction_term = rho * u * torch.log((1.0) / (u + eps))
    else:
        raise ValueError(f"Unknown PDE model: {pde_model}")

    # PDE: ∂u/∂t = ∇·(D∇u) + reaction
    rhs = diffusion_term + reaction_term

    if du_dt is not None:
        residual = du_dt - rhs
    else:
        # Steady-state assumption: ∂u/∂t ≈ 0
        residual = -rhs

    if mask is not None:
        residual = residual * mask
        n = mask.sum().clamp(min=1)
    else:
        n = residual.numel()

    mean_residual = (residual ** 2).sum() / n
    max_residual = residual.abs().max()

    return {
        "pde_residual_mean": mean_residual.item(),
        "pde_residual_max": max_residual.item(),
        "pde_residual_l1": (residual.abs().sum() / n).item(),
    }


def compute_all_metrics(pred_seg, target_seg, pred_density, target_density,
                         u, D, rho, spatial_ops, du_dt=None,
                         pde_model="fisher_kpp", mask=None,
                         num_classes=4) -> dict:
    """Compute all evaluation metrics in one call."""
    metrics = {}

    # Segmentation metrics
    if pred_seg is not None and target_seg is not None:
        dice = compute_dice(pred_seg, target_seg, num_classes=num_classes)
        metrics.update(dice)

    # Growth prediction metrics
    if pred_density is not None and target_density is not None:
        growth = compute_growth_prediction_error(pred_density, target_density, mask)
        metrics.update(growth)

    # Physics residual
    if u is not None and D is not None and rho is not None:
        physics = compute_physics_residual(u, D, rho, spatial_ops, du_dt, pde_model, mask)
        metrics.update(physics)

    return metrics
