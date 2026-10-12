"""
Combined Physics-Constrained Loss.

L_total = L_data + λ1·L_PDE + λ2·L_IC + λ3·L_BC + λ4·L_reg

CRITICAL FIX: The PDE loss now uses the FisherKPPSolver to generate a
NUMERICALLY INTEGRATED u(t₂) from the current u(t₁), then computes
du/dt = (u_t2 - u_t1) / Δt. This gives a NON-TRIVIAL temporal derivative
that the PDE residual can actually constrain against.

The old approach computed du/dt = RHS(u, D, ρ) synthetically, which made
the PDE residual R = du/dt - RHS = 0 by construction — a no-op.

Adaptive weight balancing via uncertainty weighting (Kendall et al. 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .physics_loss import (
    PDEResidualLoss, InitialConditionLoss,
    BoundaryConditionLoss, RegularizationLoss,
    TemporalConsistencyLoss,
)
from .data_loss import SegmentationLoss, DensityRegressionLoss, seg_to_density


class AdaptiveWeights(nn.Module):
    """Learnable log-variance weights (uncertainty-based, Kendall et al. 2018)."""
    def __init__(self, num_losses=5):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: list) -> torch.Tensor:
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total


class HybridTumorLoss(nn.Module):
    """
    Complete loss function combining data-driven and physics losses.

    L_total = λ_data · L_data
            + λ_pde  · L_PDE
            + λ_ic   · L_IC
            + λ_bc   · L_BC
            + λ_reg  · L_reg
            + λ_seg  · L_seg    (pretraining only)
    """

    def __init__(self, config):
        super().__init__()
        lc = config.loss
        pc = config.physics

        # Loss weights
        self.lambda_data = lc.lambda_data
        self.lambda_pde = lc.lambda_pde
        self.lambda_ic = lc.lambda_ic
        self.lambda_bc = lc.lambda_bc
        self.lambda_reg = lc.lambda_reg
        self.lambda_seg = lc.lambda_seg

        # Data losses
        self.seg_loss = SegmentationLoss(lc.seg_loss_type)
        self.density_loss = DensityRegressionLoss(lc.data_loss_type)

        # Physics losses (pass tissue-aware diffusion parameters)
        self.pde_loss = PDEResidualLoss(
            pde_model=pc.pde_model,
            dx=pc.dx, dy=pc.dy, dz=pc.dz,
            carrying_capacity=pc.default_carrying_capacity,
            epsilon=pc.epsilon,
            D_white_matter=pc.D_white_matter,
            D_grey_matter=pc.D_grey_matter,
        )
        self.ic_loss = InitialConditionLoss()
        self.bc_loss = BoundaryConditionLoss(pc.dx, pc.dy, pc.dz)
        self.reg_loss = RegularizationLoss(
            dx=pc.dx, dy=pc.dy, dz=pc.dz,
        )
        self.temporal_loss = TemporalConsistencyLoss(
            dx=pc.dx, dy=pc.dy, dz=pc.dz,
            pde_model=pc.pde_model,
        )

        # Adaptive weights
        self.use_adaptive = lc.use_adaptive_weights
        if self.use_adaptive:
            self.adaptive = AdaptiveWeights(num_losses=6)

        # Default physics params (used when model doesn't predict them)
        self.default_D = pc.default_diffusion
        self.default_rho = pc.default_proliferation

        # Synthetic longitudinal config
        self.synthetic_delta_t = pc.synthetic_delta_t
        self.use_tissue_aware = pc.tissue_aware_diffusion

        # Build a lightweight PDE solver for synthetic du/dt generation
        # This is the FisherKPPSolver that numerically integrates the PDE
        # forward to produce a NON-TRIVIAL temporal derivative.
        self._solver = None  # Lazy init to avoid import issues

    def _get_solver(self, device):
        """Lazy-initialize the FisherKPPSolver on the correct device."""
        if self._solver is None or self._solver_device != str(device):
            from data.synthetic_longitudinal import FisherKPPSolver
            self._solver = FisherKPPSolver(
                default_D=self.default_D,
                default_rho=self.default_rho,
                max_dt_step=1.0,
                device=str(device),
            )
            self._solver_device = str(device)
        return self._solver

    def _generate_synthetic_du_dt(
        self,
        u: torch.Tensor,
        D: torch.Tensor,
        rho: torch.Tensor,
        brain_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate du/dt using NUMERICAL PDE INTEGRATION (not the identity trick).

        Strategy:
            1. Take current predicted u as u(t₁)
            2. Use FisherKPPSolver to integrate the PDE forward by Δt days
               using default physics parameters (NOT the network's predictions)
            3. Compute du/dt = (u_simulated(t₂) - u(t₁)) / Δt

        WHY this works and the old method didn't:
            - Old method: du/dt = RHS(u, D_pred, ρ_pred), then loss = |du/dt - RHS|² = 0
              This is trivially zero — no training signal.
            - New method: du/dt comes from integrating with DEFAULT (D₀, ρ₀),
              but the PDE residual tests against the NETWORK's (D_pred, ρ_pred).
              The loss is zero ONLY when the network's predictions match the
              default parameters AND satisfy the PDE — a real constraint.

        The network must learn D(x) and ρ(x) such that the PDE is satisfied
        with the observed temporal evolution. This is a genuine physics constraint.
        """
        solver = self._get_solver(u.device)

        with torch.no_grad():
            # Numerically integrate PDE forward using DEFAULT parameters
            # (independent of network predictions — breaks the identity)
            u_t2, du_dt = solver.simulate(
                u0=u.detach(),
                delta_t_days=self.synthetic_delta_t,
                D=None,     # Uses solver's default_D (not network's D_pred)
                rho=None,   # Uses solver's default_rho (not network's ρ_pred)
                brain_mask=brain_mask,
                add_noise=False,
            )

        return du_dt

    def _compute_supervised_loss(self, preds, target, loss_fn, is_seg=False, mask=None):
        """Helper to compute loss with deep supervision."""
        if not isinstance(preds, list):
            preds = [preds]
            
        total_loss = 0.0
        weights = [1.0, 0.5, 0.25]
        
        for i, p in enumerate(preds):
            weight = weights[i] if i < len(weights) else 0.1
            
            # Match spatial dimensions for Deep Supervision
            t_down = target
            if p.shape[2:] != target.shape[2:]:
                t_down = F.interpolate(target.float(), size=p.shape[2:], mode="nearest")
                if is_seg:
                    t_down = t_down.long()
                    
            m_down = mask
            if mask is not None and p.shape[2:] != mask.shape[2:]:
                m_down = F.interpolate(mask.float(), size=p.shape[2:], mode="nearest")
            
            if is_seg:
                total_loss += weight * loss_fn(p, t_down)
            else:
                total_loss += weight * loss_fn(p, t_down, mask=m_down)
                
        return total_loss

    def forward(self, model_output: dict, batch: dict,
                phase: str = "joint") -> dict:
        """
        Compute all losses.

        Args:
            model_output: Dict from HybridTumorNet.forward():
                - "tumor_density": [B,1,D,H,W]
                - "diffusion": [B,1,D,H,W] (optional)
                - "proliferation": [B,1,D,H,W] (optional)
                - "segmentation": [B,C,D,H,W] (optional)
            batch: Dict from dataloader:
                - "image": [B,4,D,H,W]
                - "label": [B,1,D,H,W] segmentation
                - "du_dt" (optional): real temporal derivative from longitudinal data
                - "tissue_mask" (optional): white-matter probability map
            phase: "pretrain", "finetune", or "joint"

        Returns:
            Dict with individual losses and "total_loss".
        """
        losses = {}
        u = model_output.get("tumor_density")
        seg_pred = model_output.get("segmentation")
        label = batch.get("label")

        # Create brain mask from input (non-zero voxels = inside brain)
        brain_mask = (batch["image"].abs().sum(dim=1, keepdim=True) > 0).float()

        # Get physics params
        D = model_output.get("diffusion")
        rho = model_output.get("proliferation")
        if D is None:
            D = torch.full_like(u, self.default_D)
        if rho is None:
            rho = torch.full_like(u, self.default_rho)

        # Optional tissue mask for tissue-aware diffusion
        tissue_mask = batch.get("tissue_mask", None)

        # Base density used for physics losses (always highest resolution)
        u_base = u[0] if isinstance(u, list) else u

        # === SEGMENTATION LOSS (pretraining) ===
        if phase in ("pretrain", "joint") and seg_pred is not None and label is not None:
            losses["seg"] = self.lambda_seg * self._compute_supervised_loss(seg_pred, label, self.seg_loss, is_seg=True)

        # === DATA FIDELITY LOSS ===
        if u is not None and label is not None:
            target_density = seg_to_density(label)  # biophysically grounded mapping
            losses["data"] = self.lambda_data * self._compute_supervised_loss(
                u, target_density, self.density_loss, mask=brain_mask
            )

        # === PHYSICS LOSSES (finetune / joint) ===
        if phase in ("finetune", "joint") and u_base is not None:
            # Resolve du/dt:
            # Priority 1: real longitudinal du_dt from batch (if available)
            # Priority 2: numerically integrated via FisherKPPSolver
            du_dt = batch.get("du_dt", None)
            if du_dt is not None:
                du_dt = du_dt.to(u_base.device)
            else:
                # Generate synthetic du/dt via numerical PDE integration
                # Uses DEFAULT params, NOT network predictions — ensures non-trivial loss
                du_dt = self._generate_synthetic_du_dt(u_base, D, rho, brain_mask)

            # PDE residual (with optional tissue-aware diffusion)
            losses["pde"] = self.lambda_pde * self.pde_loss(
                u_base, D, rho,
                du_dt=du_dt,
                mask=brain_mask,
                tissue_mask=tissue_mask,
            )

            # Initial condition loss
            if label is not None:
                u_initial = seg_to_density(label)
                losses["ic"] = self.lambda_ic * self.ic_loss(
                    u_base, u_initial, mask=brain_mask
                )

            # Boundary condition (no-flux Neumann)
            losses["bc"] = self.lambda_bc * self.bc_loss(u_base, brain_mask)

            # Regularization (non-negativity, boundedness, smoothness)
            reg = self.reg_loss(u_base, D, rho, mask=brain_mask)
            losses["reg"] = self.lambda_reg * reg["total_reg"]

            # Temporal consistency (if real longitudinal data available)
            if "image_t2" in batch and "density_t2" in batch:
                u_t2 = batch["density_t2"].to(u_base.device)
                delta_t = batch.get("delta_t", 90.0)
                if isinstance(delta_t, torch.Tensor):
                    delta_t = delta_t.mean().item()
                losses["temporal"] = 0.05 * self.temporal_loss(
                    u_base, u_t2, D, rho, delta_t, mask=brain_mask
                )

        # === TOTAL ===
        if self.use_adaptive and len(losses) > 1:
            loss_list = list(losses.values())
            losses["total_loss"] = self.adaptive(loss_list)
        else:
            losses["total_loss"] = sum(losses.values())

        return losses
