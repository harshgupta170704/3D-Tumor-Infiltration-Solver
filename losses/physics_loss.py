"""
Physics-Informed Loss Functions for Tumor Growth.

Implements the PDE residual and constraint losses:

    L_PDE = ||∂u/∂t − ∇·(D(x)∇u) − ρu(1−u)||²     (Fisher-KPP)
    L_PDE = ||∂u/∂t − ∇·(D(x)∇u) − ρu·ln(K/(u+ε))||²  (Gompertz)
    
    L_IC  = ||u(x, t₀) − u₀(x)||²                    (Initial condition)
    L_BC  = ||∇u · n||²  at boundaries                 (No-flux Neumann BC)
    L_reg = λ_pos·||ReLU(−u)||² + λ_bound·||ReLU(u−1)||² + λ_smooth·||∇u||²

All losses work on 3D volume tensors [B, 1, D, H, W].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.spatial_ops import SpatialGradients3D, compute_temporal_derivative


class PDEResidualLoss(nn.Module):
    """
    PDE residual loss enforcing the Fisher-KPP reaction-diffusion equation.

    Residual R = ∂u/∂t − ∇·(D(x)∇u) − ρ·u·(1 − u)
    Loss: L_PDE = (1/N) Σ R²  [mean squared residual over domain]

    Temporal derivative (∂u/∂t):
        - If du_dt is provided (from synthetic longitudinal pairs): used directly.
        - If du_dt is None: quasi-steady-state mode — penalizes |RHS| directly.
          This is valid when the tumor is assumed to be at a slowly-changing state.
          NOTE: quasi-steady-state is a WEAKER constraint — state clearly in paper.

    Tissue-aware diffusion (Harpold et al. 2007):
        If tissue_mask is provided, D(x) is scaled:
            D_effective(x) = D_pred(x) * [D_wm in WM, D_gm in GM]
        This ensures biophysically realistic diffusion anisotropy.
    """

    def __init__(
        self,
        pde_model: str = "fisher_kpp",
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        carrying_capacity: float = 1.0,
        epsilon: float = 1e-7,
        D_white_matter: float = 0.08,
        D_grey_matter: float = 0.016,
    ):
        """
        Args:
            pde_model: "fisher_kpp" or "gompertz".
            dx, dy, dz: Voxel spacing in mm.
            carrying_capacity: K in Gompertz model.
            epsilon: Small constant for numerical stability.
            D_white_matter: Baseline WM diffusion (mm²/day). Harpold et al. 2007.
            D_grey_matter: Baseline GM diffusion (mm²/day). Harpold et al. 2007.
        """
        super().__init__()
        self.pde_model = pde_model
        self.K = carrying_capacity
        self.eps = epsilon
        self.D_wm = D_white_matter
        self.D_gm = D_grey_matter
        self.spatial_ops = SpatialGradients3D(dx, dy, dz)

    def _apply_tissue_diffusion(
        self,
        D: torch.Tensor,
        tissue_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scale predicted D(x) by tissue-specific baseline values.

        Strategy: The network learns a RELATIVE diffusion field D_pred ∈ (0,1].
        We then scale it: D_effective = D_pred * D_tissue(x)
        where D_tissue = D_wm inside white matter, D_gm otherwise.

        Args:
            D: Predicted diffusion [B, 1, D, H, W] (positive, raw network output).
            tissue_mask: White-matter probability map [B, 1, D, H, W] ∈ [0, 1].
                         1.0 = fully white matter, 0.0 = grey matter / background.

        Returns:
            D_effective [B, 1, D, H, W] with tissue-appropriate magnitude.
        """
        D_tissue = tissue_mask * self.D_wm + (1.0 - tissue_mask) * self.D_gm
        return D * D_tissue

    def forward(
        self,
        u: torch.Tensor,
        D: torch.Tensor,
        rho: torch.Tensor,
        du_dt: torch.Tensor = None,
        mask: torch.Tensor = None,
        tissue_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the PDE residual loss.

        Args:
            u: Tumor density [B, 1, D, H, W] ∈ [0, 1].
            D: Predicted diffusion coefficient [B, 1, D, H, W] > 0.
            rho: Proliferation rate [B, 1, D, H, W] or scalar tensor.
            du_dt: Temporal derivative [B, 1, D, H, W] (from synthetic pairs
                   or real longitudinal data). If None, quasi-steady-state mode.
            mask: Optional brain mask [B, 1, D, H, W]. Loss computed only in mask.
            tissue_mask: Optional white-matter map [B, 1, D, H, W] ∈ [0,1].
                         If provided, applies tissue-aware diffusion scaling.

        Returns:
            Scalar PDE residual loss (non-negative).
        """
        # Move spatial ops to same device
        self.spatial_ops = self.spatial_ops.to(u.device)

        # Apply tissue-aware diffusion scaling if white-matter mask provided
        if tissue_mask is not None:
            D_effective = self._apply_tissue_diffusion(D, tissue_mask.to(u.device))
        else:
            D_effective = D

        # --- Diffusion term: ∇·(D(x)∇u) ---
        diffusion_term = self.spatial_ops.divergence_of_flux(u, D_effective)

        # --- Reaction term ---
        if self.pde_model == "fisher_kpp":
            reaction_term = rho * u * (1.0 - u)
        elif self.pde_model == "gompertz":
            reaction_term = rho * u * torch.log(self.K / (u + self.eps))
        else:
            raise ValueError(f"Unknown PDE model: {self.pde_model}")

        # --- PDE Residual ---
        rhs = diffusion_term + reaction_term

        if du_dt is not None:
            # Full PDE residual: R = ∂u/∂t - RHS
            residual = du_dt - rhs
        else:
            # Quasi-steady-state: penalize |RHS|  (assumes ∂u/∂t ≈ 0)
            # WEAKER than full PDE — state this limitation explicitly in any paper
            residual = rhs

        # --- Apply brain mask ---
        if mask is not None:
            residual = residual * mask
            n_voxels = mask.sum().clamp(min=1)
        else:
            n_voxels = residual.numel()

        return (residual ** 2).sum() / n_voxels


class InitialConditionLoss(nn.Module):
    """
    Initial condition loss: L_IC = ||u(x, t₀) − u₀(x)||².
    
    u₀ can be derived from the earliest available segmentation mask
    or from a known initial tumor state.
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, u_pred: torch.Tensor, u_initial: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            u_pred: Predicted density at initial time [B, 1, D, H, W].
            u_initial: Observed initial density [B, 1, D, H, W].
                       Can be derived from segmentation: u₀ = seg > 0.
            mask: Optional brain mask.

        Returns:
            Scalar IC loss.
        """
        diff = u_pred - u_initial

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
            return F.smooth_l1_loss(u_pred * (mask if mask is not None else 1),
                                     u_initial * (mask if mask is not None else 1))
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class BoundaryConditionLoss(nn.Module):
    """
    No-flux Neumann boundary condition: ∇u · n = 0 at the boundary.
    
    L_BC = (1/N_boundary) Σ |∇u · n|²
    
    The boundary is either the edge of the brain mask or the edge of the volume.
    """

    def __init__(self, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0):
        super().__init__()
        self.spatial_ops = SpatialGradients3D(dx, dy, dz)

    def forward(self, u: torch.Tensor, brain_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            u: Tumor density [B, 1, D, H, W].
            brain_mask: Brain mask [B, 1, D, H, W]. Boundary = edge of mask.

        Returns:
            Scalar BC loss.
        """
        self.spatial_ops = self.spatial_ops.to(u.device)
        
        # Get normal gradient at boundary
        boundary_grad = self.spatial_ops.normal_gradient_at_boundary(u, brain_mask)

        # Penalize non-zero normal gradient at boundary
        n_boundary = (boundary_grad > 0).float().sum().clamp(min=1)
        return (boundary_grad ** 2).sum() / n_boundary


class RegularizationLoss(nn.Module):
    """
    Regularization losses to enforce physical constraints.
    
    L_reg = λ_pos · ||ReLU(−u)||²           (Non-negativity: u ≥ 0)
          + λ_bound · ||ReLU(u − 1)||²      (Upper bound: u ≤ 1)  
          + λ_smooth · ||∇u||²              (Spatial smoothness)
          + λ_param · (||D||² + ||ρ||²)      (Parameter regularization)
    """

    def __init__(self, lambda_positivity: float = 10.0,
                 lambda_boundedness: float = 10.0,
                 lambda_smoothness: float = 0.001,
                 lambda_param_reg: float = 0.001,
                 dx: float = 1.0, dy: float = 1.0, dz: float = 1.0):
        super().__init__()
        self.lambda_pos = lambda_positivity
        self.lambda_bound = lambda_boundedness
        self.lambda_smooth = lambda_smoothness
        self.lambda_param = lambda_param_reg
        self.spatial_ops = SpatialGradients3D(dx, dy, dz)

    def forward(self, u: torch.Tensor, D: torch.Tensor = None, 
                rho: torch.Tensor = None, mask: torch.Tensor = None) -> dict:
        """
        Args:
            u: Tumor density [B, 1, D, H, W].
            D: Diffusion coefficient [B, 1, D, H, W] (optional).
            rho: Proliferation rate [B, 1, D, H, W] (optional).
            mask: Brain mask [B, 1, D, H, W] (optional).

        Returns:
            Dictionary with individual regularization terms and total.
        """
        self.spatial_ops = self.spatial_ops.to(u.device)
        
        if mask is not None:
            n = mask.sum().clamp(min=1)
        else:
            n = u.numel()

        losses = {}

        # --- Non-negativity: u ≥ 0 ---
        neg_violation = F.relu(-u)
        if mask is not None:
            neg_violation = neg_violation * mask
        losses["positivity"] = self.lambda_pos * (neg_violation ** 2).sum() / n

        # --- Upper bound: u ≤ 1 ---
        upper_violation = F.relu(u - 1.0)
        if mask is not None:
            upper_violation = upper_violation * mask
        losses["boundedness"] = self.lambda_bound * (upper_violation ** 2).sum() / n

        # --- Spatial smoothness: ||∇u||² ---
        grad_mag = self.spatial_ops.gradient_magnitude(u)
        if mask is not None:
            grad_mag = grad_mag * mask
        losses["smoothness"] = self.lambda_smooth * (grad_mag ** 2).sum() / n

        # --- Parameter regularization ---
        param_loss = torch.tensor(0.0, device=u.device)
        if D is not None:
            param_loss = param_loss + D.mean()
        if rho is not None:
            param_loss = param_loss + rho.mean()
        losses["param_reg"] = self.lambda_param * param_loss

        # Total
        losses["total_reg"] = sum(losses.values())

        return losses


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency between predictions at different time points.
    
    Penalizes large jumps in tumor density that violate physical continuity:
        L_temporal = ||u(t₂) − [u(t₁) + Δt · f(u(t₁))]||²
    
    where f(u) = ∇·(D∇u) + ρu(1−u) is the PDE right-hand side.
    """

    def __init__(self, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
                 pde_model: str = "fisher_kpp"):
        super().__init__()
        self.spatial_ops = SpatialGradients3D(dx, dy, dz)
        self.pde_model = pde_model

    def forward(self, u_t1: torch.Tensor, u_t2: torch.Tensor,
                D: torch.Tensor, rho: torch.Tensor,
                delta_t: float, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            u_t1: Tumor density at t₁ [B, 1, D, H, W].
            u_t2: Tumor density at t₂ [B, 1, D, H, W].
            D: Diffusion coefficient.
            rho: Proliferation rate.
            delta_t: Time between scans (days).
            mask: Brain mask.

        Returns:
            Scalar temporal consistency loss.
        """
        self.spatial_ops = self.spatial_ops.to(u_t1.device)

        # Compute PDE RHS at t₁
        diffusion = self.spatial_ops.divergence_of_flux(u_t1, D)
        
        if self.pde_model == "fisher_kpp":
            reaction = rho * u_t1 * (1.0 - u_t1)
        else:
            reaction = rho * u_t1 * torch.log(1.0 / (u_t1 + 1e-7))

        rhs = diffusion + reaction

        # Forward Euler prediction
        u_predicted = u_t1 + delta_t * rhs
        u_predicted = torch.clamp(u_predicted, 0.0, 1.0)

        # Consistency error
        error = u_t2 - u_predicted

        if mask is not None:
            error = error * mask
            n = mask.sum().clamp(min=1)
        else:
            n = error.numel()

        return (error ** 2).sum() / n
