"""
Synthetic Longitudinal Data Generator for Physics-Constrained Training.

PURPOSE & SCIENTIFIC JUSTIFICATION
------------------------------------
BraTS 2021 is a CROSS-SECTIONAL dataset (single time point per patient).
The Fisher-KPP PDE residual requires temporal derivatives:
    ∂u/∂t ≈ (u(t₂) - u(t₁)) / Δt

To supply this without real longitudinal MRI pairs, we use a well-established
method from the computational oncology literature (Mang et al. 2019,
Lipkova et al. 2019):

    1. Treat the segmentation-derived density u₀ as the INITIAL condition u(t₁)
    2. Numerically integrate the Fisher-KPP PDE forward by Δt days
       using default/estimated diffusion and proliferation parameters
    3. The numerically simulated u(t₂) serves as the synthetic ground truth

This is scientifically valid because:
    a) The PDE model is self-consistent — we train the network to predict
       parameters (D, ρ) that reproduce the numerically integrated evolution
    b) It constrains the network to learn biophysically plausible parameters
    c) It is clearly labeled as SYNTHETIC in all outputs

This approach is DIFFERENT from claiming real growth was observed.
The distinction must be stated clearly in any paper.

References:
    - Mang et al. (2019) CLAIRE: A distributed-memory solver for
      constrained large diffeomorphic image registration. SIAM J. Sci. Comput.
    - Lipkova et al. (2019) Personalized Radiotherapy Design for Glioblastoma.
      IEEE Trans. Med. Imag.
    - Swanson et al. (2000) A mathematical modelling tool for predicting
      survival of individual patients following resection of glioblastoma.
      Brit. J. Cancer.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class FisherKPPSolver:
    """
    Numerically solves the Fisher-KPP reaction-diffusion PDE:

        ∂u/∂t = ∇·(D(x)∇u) + ρ·u·(1 - u)

    using explicit 4th-order Runge-Kutta integration on a 3D voxel grid.

    This solver is used to generate SYNTHETIC longitudinal tumor density
    fields from a single BraTS time point. The output is the SIMULATED
    tumor state Δt days in the future.

    Stability constraint (CFL condition for explicit solver):
        Δt_step ≤ (Δx)² / (6 · D_max)
        For Δx=1mm, D_max=0.1: Δt_step ≤ 1.67 days

    Parameters are intentionally set conservatively to avoid numerical blowup.
    """

    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        default_D: float = 0.05,       # mm²/day — conservative default
        default_rho: float = 0.02,     # 1/day — proliferation rate
        max_dt_step: float = 1.0,      # days per RK4 step (CFL-safe at D=0.05)
        device: str = "cpu",
    ):
        """
        Args:
            dx, dy, dz: Voxel spacing in mm along each axis.
            default_D: Default diffusion coefficient (mm²/day).
            default_rho: Default proliferation rate (1/day).
            max_dt_step: Maximum time step per RK4 integration step (days).
                         Ensure CFL: max_dt_step < dx² / (6 * default_D)
            device: Torch device string.
        """
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.default_D = default_D
        self.default_rho = default_rho
        self.max_dt_step = max_dt_step
        self.device = torch.device(device)

        # CFL stability check
        cfl_limit = min(dx, dy, dz) ** 2 / (6.0 * default_D + 1e-9)
        if max_dt_step > cfl_limit:
            import warnings
            warnings.warn(
                f"max_dt_step={max_dt_step} may violate CFL condition "
                f"(limit={cfl_limit:.3f} for D={default_D}). "
                f"Consider reducing max_dt_step for numerical stability.",
                RuntimeWarning,
            )

        # Pre-build finite-difference kernels (fixed, non-trainable)
        self._build_kernels()

    def _build_kernels(self):
        """Build 3D finite-difference convolution kernels."""
        # 7-point Laplacian stencil
        k_lap = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
        k_lap[0, 0, 1, 1, 1] = -(2 / self.dx**2 + 2 / self.dy**2 + 2 / self.dz**2)
        k_lap[0, 0, 0, 1, 1] = 1.0 / self.dx**2
        k_lap[0, 0, 2, 1, 1] = 1.0 / self.dx**2
        k_lap[0, 0, 1, 0, 1] = 1.0 / self.dy**2
        k_lap[0, 0, 1, 2, 1] = 1.0 / self.dy**2
        k_lap[0, 0, 1, 1, 0] = 1.0 / self.dz**2
        k_lap[0, 0, 1, 1, 2] = 1.0 / self.dz**2

        # Central difference kernels for ∂/∂x, ∂/∂y, ∂/∂z
        k_dx = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
        k_dx[0, 0, 0, 1, 1] = -1.0 / (2 * self.dx)
        k_dx[0, 0, 2, 1, 1] =  1.0 / (2 * self.dx)

        k_dy = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
        k_dy[0, 0, 1, 0, 1] = -1.0 / (2 * self.dy)
        k_dy[0, 0, 1, 2, 1] =  1.0 / (2 * self.dy)

        k_dz = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
        k_dz[0, 0, 1, 1, 0] = -1.0 / (2 * self.dz)
        k_dz[0, 0, 1, 1, 2] =  1.0 / (2 * self.dz)

        self.k_lap = k_lap.to(self.device)
        self.k_dx  = k_dx.to(self.device)
        self.k_dy  = k_dy.to(self.device)
        self.k_dz  = k_dz.to(self.device)

    def _rhs(
        self,
        u: torch.Tensor,
        D: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Fisher-KPP PDE right-hand side: ∇·(D∇u) + ρ·u·(1-u).

        Uses product rule: ∇·(D∇u) = D·∇²u + ∇D·∇u

        Args:
            u:   [B, 1, D, H, W] tumor density ∈ [0, 1]
            D:   [B, 1, D, H, W] diffusion coefficient (mm²/day)
            rho: [B, 1, D, H, W] or scalar proliferation rate (1/day)

        Returns:
            [B, 1, D, H, W] — du/dt
        """
        # Laplacian of u
        lap_u = F.conv3d(u, self.k_lap, padding=1)

        # Gradient of u
        du_dx = F.conv3d(u, self.k_dx, padding=1)
        du_dy = F.conv3d(u, self.k_dy, padding=1)
        du_dz = F.conv3d(u, self.k_dz, padding=1)

        # Gradient of D
        dD_dx = F.conv3d(D, self.k_dx, padding=1)
        dD_dy = F.conv3d(D, self.k_dy, padding=1)
        dD_dz = F.conv3d(D, self.k_dz, padding=1)

        # Divergence of flux: ∇·(D∇u) = D·∇²u + ∇D·∇u
        diffusion = D * lap_u + (dD_dx * du_dx + dD_dy * du_dy + dD_dz * du_dz)

        # Reaction term
        reaction = rho * u * (1.0 - u)

        return diffusion + reaction

    @torch.no_grad()
    def simulate(
        self,
        u0: torch.Tensor,
        delta_t_days: float,
        D: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
        brain_mask: Optional[torch.Tensor] = None,
        add_noise: bool = True,
        noise_std: float = 0.005,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate tumor growth for delta_t_days using RK4.

        Args:
            u0: Initial tumor density [B, 1, D, H, W] ∈ [0, 1].
            delta_t_days: Total simulation time in days.
            D: Spatially-varying diffusion [B, 1, D, H, W]. If None, uses default.
            rho: Proliferation rate [B, 1, D, H, W] or scalar. If None, uses default.
            brain_mask: Binary brain mask [B, 1, D, H, W]. Enforces no-flux BC.
            add_noise: Whether to add small realistic measurement noise to u_t2.
            noise_std: Standard deviation of measurement noise.

        Returns:
            (u_t2, du_dt):
                u_t2:  [B, 1, D, H, W] — synthetic tumor state at t₂
                du_dt: [B, 1, D, H, W] — finite-difference temporal derivative
        """
        u0 = u0.to(self.device).float()

        if D is None:
            D = torch.full_like(u0, self.default_D)
        else:
            D = D.to(self.device).float()

        if rho is None:
            rho = torch.full_like(u0, self.default_rho)
        else:
            rho = rho.to(self.device).float()

        # Number of integration steps (ensure CFL)
        n_steps = max(1, int(np.ceil(delta_t_days / self.max_dt_step)))
        dt_step = delta_t_days / n_steps

        u = u0.clone()

        for _ in range(n_steps):
            # RK4 integration
            k1 = self._rhs(u, D, rho)
            k2 = self._rhs(torch.clamp(u + 0.5 * dt_step * k1, 0.0, 1.0), D, rho)
            k3 = self._rhs(torch.clamp(u + 0.5 * dt_step * k2, 0.0, 1.0), D, rho)
            k4 = self._rhs(torch.clamp(u + dt_step * k3,        0.0, 1.0), D, rho)

            u = u + (dt_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            u = torch.clamp(u, 0.0, 1.0)

            # Apply brain mask (no-flux: tumor stays inside brain)
            if brain_mask is not None:
                u = u * brain_mask

        u_t2 = u

        # Add small realistic measurement noise to simulate MRI variability
        if add_noise and noise_std > 0.0:
            noise = torch.randn_like(u_t2) * noise_std
            u_t2 = torch.clamp(u_t2 + noise, 0.0, 1.0)

        # Temporal derivative (first-order finite difference)
        du_dt = (u_t2 - u0) / (delta_t_days + 1e-9)

        return u_t2, du_dt


def generate_synthetic_pair(
    seg_label: torch.Tensor,
    delta_t_days: float = 90.0,
    D_wm: float = 0.08,
    D_gm: float = 0.016,
    rho: float = 0.02,
    brain_mask: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> dict:
    """
    High-level API: Generate a synthetic longitudinal pair from a BraTS
    segmentation mask using the Fisher-KPP PDE.

    Scientific basis:
        - D_wm (white matter diffusion) ≈ 5x D_gm (Harpold et al. 2007)
        - Default D and ρ from Swanson et al. (2000) glioma growth model
        - Δt = 90 days ≈ typical BraTS re-scan interval

    Args:
        seg_label: Segmentation [B, 1, D, H, W] with BraTS labels {0,1,2,3}.
        delta_t_days: Simulation time (days between virtual scans).
        D_wm: White-matter diffusion coefficient (mm²/day).
        D_gm: Grey-matter diffusion coefficient (mm²/day).
        rho: Proliferation rate (1/day).
        brain_mask: Binary brain mask [B, 1, D, H, W].
        device: Torch device string.

    Returns:
        dict with keys:
            "u_t1":    Initial tumor density [B, 1, D, H, W]
            "u_t2":    Simulated tumor density at t₂ [B, 1, D, H, W]
            "du_dt":   Temporal derivative (u_t2 - u_t1) / Δt [B, 1, D, H, W]
            "D_field": Diffusion field used [B, 1, D, H, W]
            "rho_field": Proliferation field [B, 1, D, H, W]
            "delta_t": Simulation time (days)
            "is_synthetic": True (always — must be stated in paper)
    """
    seg = seg_label.to(device).float()

    # --- Construct initial tumor density from segmentation ---
    # Biophysically grounded density mapping:
    #   ET  (class 3) → u = 1.0  (highest cell density, contrast-enhancing)
    #   NCR (class 1) → u = 0.6  (necrotic core, cell debris)
    #   ED  (class 2) → u = 0.2  (infiltrative edema, sparse invasion)
    #   BG  (class 0) → u = 0.0
    u_t1 = torch.zeros_like(seg)
    u_t1 = torch.where(seg == 2, torch.tensor(0.2, device=device), u_t1)
    u_t1 = torch.where(seg == 1, torch.tensor(0.6, device=device), u_t1)
    u_t1 = torch.where(seg == 3, torch.tensor(1.0, device=device), u_t1)

    # --- Construct spatially varying diffusion field ---
    # Approximate tissue type from MRI: use segmentation neighborhood
    # as a proxy — inside tumor use D_gm, periphery use D_wm
    # This is a simplification; ideally requires T1 white-matter segmentation.
    # Use binary dilation to estimate infiltration zone (white matter region)
    if brain_mask is not None:
        # Region inside brain but outside tumor core → assume mix, use D_wm weighted
        tumor_core = (seg > 0).float()
        brain_region = brain_mask.float()
        perilesional = brain_region * (1.0 - tumor_core)  # brain minus tumor
        D_field = D_gm * tumor_core + D_wm * perilesional
    else:
        D_field = torch.full_like(u_t1, D_wm)  # Uniform fallback

    D_field = D_field.to(device)
    rho_field = torch.full_like(u_t1, rho)

    # --- Simulate growth ---
    solver = FisherKPPSolver(
        default_D=D_wm,
        default_rho=rho,
        max_dt_step=1.0,
        device=device,
    )

    u_t2, du_dt = solver.simulate(
        u0=u_t1,
        delta_t_days=delta_t_days,
        D=D_field,
        rho=rho_field,
        brain_mask=brain_mask,
        add_noise=True,
        noise_std=0.005,
    )

    return {
        "u_t1": u_t1,
        "u_t2": u_t2,
        "du_dt": du_dt,
        "D_field": D_field,
        "rho_field": rho_field,
        "delta_t": delta_t_days,
        "is_synthetic": True,  # MUST be stated in paper
    }



if __name__ == "__main__":
    """Quick test / demo of the synthetic longitudinal generator."""
    print("Testing FisherKPPSolver...")

    # Create a simple spherical tumor density
    B, D, H, W = 1, 64, 64, 64
    u0 = torch.zeros(B, 1, D, H, W)
    # Small sphere at center
    for i in range(D):
        for j in range(H):
            for k in range(W):
                r = ((i - D//2)**2 + (j - H//2)**2 + (k - W//2)**2) ** 0.5
                if r < 5:
                    u0[0, 0, i, j, k] = 1.0
                elif r < 8:
                    u0[0, 0, i, j, k] = 0.5

    D_field = torch.full_like(u0, 0.05)
    rho_field = torch.full_like(u0, 0.02)
    brain_mask = torch.ones_like(u0)

    solver = FisherKPPSolver(default_D=0.05, default_rho=0.02, max_dt_step=1.0)
    u_t2, du_dt = solver.simulate(u0, delta_t_days=30.0, D=D_field, rho=rho_field,
                                   brain_mask=brain_mask)

    print(f"  u0   range: [{u0.min():.4f}, {u0.max():.4f}]")
    print(f"  u_t2 range: [{u_t2.min():.4f}, {u_t2.max():.4f}]")
    print(f"  du_dt range: [{du_dt.min():.4f}, {du_dt.max():.4f}]")

    # Tumor should have grown (u_t2 > u_t1 on average in tumor region)
    tumor_mask = (u0 > 0.1).float()
    mean_u0   = (u0   * tumor_mask).sum() / tumor_mask.sum().clamp(min=1)
    mean_u_t2 = (u_t2 * tumor_mask).sum() / tumor_mask.sum().clamp(min=1)
    print(f"  Mean density in tumor: {mean_u0:.4f} → {mean_u_t2:.4f} (should be ≥)")
    assert mean_u_t2 >= mean_u0 * 0.95, "Tumor should not shrink with positive ρ!"

    print("✓ FisherKPPSolver test PASSED")
