"""
Spatial differential operators for 3D volumes using finite differences.

Computes ∇u (gradient), ∇²u (Laplacian), and ∇·(D∇u) (anisotropic diffusion)
on regular 3D grids using convolution-based finite difference stencils.
All operations are differentiable for backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGradients3D(nn.Module):
    """
    Compute spatial derivatives on 3D volumes using finite differences
    implemented as fixed-weight 3D convolutions. This is efficient and
    fully differentiable.
    
    Supports:
        - First-order central differences: ∂u/∂x, ∂u/∂y, ∂u/∂z
        - Second-order central differences: ∂²u/∂x², ∂²u/∂y², ∂²u/∂z²
        - 3D Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
        - Divergence of flux: ∇·(D(x)∇u)
    """

    def __init__(self, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0):
        """
        Args:
            dx, dy, dz: Voxel spacing in mm along each axis.
        """
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dz = dz

        # --- First-order central difference kernels ---
        # ∂u/∂x: [-1, 0, 1] / (2*dx) along depth axis
        kernel_dx = torch.zeros(1, 1, 3, 3, 3)
        kernel_dx[0, 0, 0, 1, 1] = -1.0 / (2.0 * dx)
        kernel_dx[0, 0, 2, 1, 1] = 1.0 / (2.0 * dx)
        self.register_buffer('kernel_dx', kernel_dx)

        # ∂u/∂y: [-1, 0, 1] / (2*dy) along height axis
        kernel_dy = torch.zeros(1, 1, 3, 3, 3)
        kernel_dy[0, 0, 1, 0, 1] = -1.0 / (2.0 * dy)
        kernel_dy[0, 0, 1, 2, 1] = 1.0 / (2.0 * dy)
        self.register_buffer('kernel_dy', kernel_dy)

        # ∂u/∂z: [-1, 0, 1] / (2*dz) along width axis
        kernel_dz = torch.zeros(1, 1, 3, 3, 3)
        kernel_dz[0, 0, 1, 1, 0] = -1.0 / (2.0 * dz)
        kernel_dz[0, 0, 1, 1, 2] = 1.0 / (2.0 * dz)
        self.register_buffer('kernel_dz', kernel_dz)

        # --- Second-order central difference kernels ---
        # ∂²u/∂x²: [1, -2, 1] / dx² along depth axis
        kernel_dxx = torch.zeros(1, 1, 3, 3, 3)
        kernel_dxx[0, 0, 0, 1, 1] = 1.0 / (dx * dx)
        kernel_dxx[0, 0, 1, 1, 1] = -2.0 / (dx * dx)
        kernel_dxx[0, 0, 2, 1, 1] = 1.0 / (dx * dx)
        self.register_buffer('kernel_dxx', kernel_dxx)

        # ∂²u/∂y²: [1, -2, 1] / dy² along height axis
        kernel_dyy = torch.zeros(1, 1, 3, 3, 3)
        kernel_dyy[0, 0, 1, 0, 1] = 1.0 / (dy * dy)
        kernel_dyy[0, 0, 1, 1, 1] = -2.0 / (dy * dy)
        kernel_dyy[0, 0, 1, 2, 1] = 1.0 / (dy * dy)
        self.register_buffer('kernel_dyy', kernel_dyy)

        # ∂²u/∂z²: [1, -2, 1] / dz² along width axis
        kernel_dzz = torch.zeros(1, 1, 3, 3, 3)
        kernel_dzz[0, 0, 1, 1, 0] = 1.0 / (dz * dz)
        kernel_dzz[0, 0, 1, 1, 1] = -2.0 / (dz * dz)
        kernel_dzz[0, 0, 1, 1, 2] = 1.0 / (dz * dz)
        self.register_buffer('kernel_dzz', kernel_dzz)

        # --- 7-point 3D Laplacian stencil (isotropic case) ---
        kernel_lap = torch.zeros(1, 1, 3, 3, 3)
        kernel_lap[0, 0, 1, 1, 1] = -(2.0/(dx*dx) + 2.0/(dy*dy) + 2.0/(dz*dz))
        kernel_lap[0, 0, 0, 1, 1] = 1.0 / (dx * dx)
        kernel_lap[0, 0, 2, 1, 1] = 1.0 / (dx * dx)
        kernel_lap[0, 0, 1, 0, 1] = 1.0 / (dy * dy)
        kernel_lap[0, 0, 1, 2, 1] = 1.0 / (dy * dy)
        kernel_lap[0, 0, 1, 1, 0] = 1.0 / (dz * dz)
        kernel_lap[0, 0, 1, 1, 2] = 1.0 / (dz * dz)
        self.register_buffer('kernel_laplacian', kernel_lap)

    def gradient(self, u: torch.Tensor) -> tuple:
        """
        Compute first-order spatial gradient ∇u = (∂u/∂x, ∂u/∂y, ∂u/∂z).

        Args:
            u: Tensor of shape [B, 1, D, H, W] — tumor density field.

        Returns:
            (du_dx, du_dy, du_dz): Each [B, 1, D, H, W].
        """
        du_dx = F.conv3d(u, self.kernel_dx, padding=1)
        du_dy = F.conv3d(u, self.kernel_dy, padding=1)
        du_dz = F.conv3d(u, self.kernel_dz, padding=1)
        return du_dx, du_dy, du_dz

    def laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute the 3D Laplacian ∇²u using the 7-point stencil.

        Args:
            u: Tensor of shape [B, 1, D, H, W].

        Returns:
            Tensor of shape [B, 1, D, H, W] — the Laplacian.
        """
        return F.conv3d(u, self.kernel_laplacian, padding=1)

    def second_derivatives(self, u: torch.Tensor) -> tuple:
        """
        Compute second-order partial derivatives.

        Returns:
            (d²u/dx², d²u/dy², d²u/dz²): Each [B, 1, D, H, W].
        """
        uxx = F.conv3d(u, self.kernel_dxx, padding=1)
        uyy = F.conv3d(u, self.kernel_dyy, padding=1)
        uzz = F.conv3d(u, self.kernel_dzz, padding=1)
        return uxx, uyy, uzz

    def divergence_of_flux(self, u: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇·(D(x)∇u) — divergence of the diffusion flux.
        
        Uses the product rule: ∇·(D∇u) = D·∇²u + ∇D·∇u

        Args:
            u: Tumor density [B, 1, D, H, W].
            D: Spatially-varying diffusion coefficient [B, 1, D, H, W].

        Returns:
            Tensor [B, 1, D, H, W] — the divergence of the flux.
        """
        # Compute ∇u
        du_dx, du_dy, du_dz = self.gradient(u)

        # Compute ∇D
        dD_dx, dD_dy, dD_dz = self.gradient(D)

        # Compute ∇²u
        lap_u = self.laplacian(u)

        # ∇·(D∇u) = D·∇²u + ∇D·∇u
        divergence = D * lap_u + (dD_dx * du_dx + dD_dy * du_dy + dD_dz * du_dz)
        return divergence

    def gradient_magnitude(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute |∇u| = sqrt((∂u/∂x)² + (∂u/∂y)² + (∂u/∂z)²).

        Args:
            u: Tensor [B, 1, D, H, W].

        Returns:
            Tensor [B, 1, D, H, W] — gradient magnitude.
        """
        du_dx, du_dy, du_dz = self.gradient(u)
        return torch.sqrt(du_dx**2 + du_dy**2 + du_dz**2 + 1e-8)

    def normal_gradient_at_boundary(self, u: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute ∇u·n at the boundary of the domain.
        
        For a regular grid, the boundary is the edge voxels.
        The outward normal at edges is along the axis direction.

        Args:
            u: Tensor [B, 1, D, H, W].
            mask: Optional brain mask [B, 1, D, H, W]. If given,
                  boundary = edge of the mask.

        Returns:
            Tensor of boundary normal gradients [B, 1, D, H, W].
            Non-zero only at boundary voxels.
        """
        du_dx, du_dy, du_dz = self.gradient(u)

        if mask is not None:
            # Boundary = voxels in mask that neighbor voxels outside mask
            # Erode mask by 1 voxel
            erode_kernel = torch.ones(1, 1, 3, 3, 3, device=u.device) / 27.0
            eroded = F.conv3d(mask.float(), erode_kernel, padding=1)
            boundary = mask.float() * (1.0 - (eroded > 0.99).float())
        else:
            # Use volume edges as boundary
            boundary = torch.zeros_like(u)
            boundary[:, :, 0, :, :] = 1.0   # front face
            boundary[:, :, -1, :, :] = 1.0  # back face
            boundary[:, :, :, 0, :] = 1.0   # top face
            boundary[:, :, :, -1, :] = 1.0  # bottom face
            boundary[:, :, :, :, 0] = 1.0   # left face
            boundary[:, :, :, :, -1] = 1.0  # right face

        # At boundaries, approximate ∇u·n using the gradient magnitude
        grad_mag = torch.sqrt(du_dx**2 + du_dy**2 + du_dz**2 + 1e-8)
        return grad_mag * boundary


def compute_temporal_derivative(u_t1: torch.Tensor, u_t2: torch.Tensor, delta_t: float) -> torch.Tensor:
    """
    Approximate ∂u/∂t using forward difference between two time points.

    Args:
        u_t1: Tumor density at time t₁ [B, 1, D, H, W].
        u_t2: Tumor density at time t₂ [B, 1, D, H, W].
        delta_t: Time difference (t₂ - t₁) in days.

    Returns:
        Tensor [B, 1, D, H, W] — approximate ∂u/∂t.
    """
    return (u_t2 - u_t1) / (delta_t + 1e-8)
