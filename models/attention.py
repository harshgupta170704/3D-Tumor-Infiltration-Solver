"""
Attention mechanisms for 3D medical image analysis.

    - Squeeze-and-Excitation (SE) channel attention
    - Spatial Attention Gate (for skip connections)
    - CBAM (Convolutional Block Attention Module) adapted for 3D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation3D(nn.Module):
    """Channel attention via global pooling → FC → sigmoid reweighting."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.squeeze(x).view(b, c)
        w = self.excite(w).view(b, c, 1, 1, 1)
        return x * w


class AttentionGate3D(nn.Module):
    """
    Attention gate for skip connections (Oktay et al., 2018).

    Learns to suppress irrelevant regions in skip features using
    the gating signal from the decoder. Critical for focusing on
    tumor regions and ignoring healthy tissue.

        α = σ(W_ψ(ReLU(W_g·g + W_x·x + b)) + b_ψ)
        output = x * α
    """

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int = None):
        super().__init__()
        inter_ch = inter_ch or skip_ch // 2
        self.W_gate = nn.Conv3d(gate_ch, inter_ch, 1, bias=False)
        self.W_skip = nn.Conv3d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.InstanceNorm3d(inter_ch)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        # Upsample gate to match skip spatial size
        if g.shape[2:] != skip.shape[2:]:
            g = F.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.W_skip(skip)
        attn = self.relu(g + x)
        attn = self.norm(attn)
        attn = self.psi(attn)
        return skip * attn


class SpatialAttention3D(nn.Module):
    """Spatial attention using max+avg pooling across channels."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.max(dim=1, keepdim=True)[0]
        combined = torch.cat([avg_pool, max_pool], dim=1)
        attn = self.sigmoid(self.conv(combined))
        return x * attn


class CBAM3D(nn.Module):
    """Convolutional Block Attention Module: channel + spatial attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = SqueezeExcitation3D(channels, reduction)
        self.spatial_attn = SpatialAttention3D(kernel_size=7)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
