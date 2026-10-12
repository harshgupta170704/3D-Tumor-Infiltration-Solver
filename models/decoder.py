"""
Decoder modules for dense prediction from ResNet encoder features.

Includes:
    - TumorDensityDecoder: Predicts u(x,t) field [B, 1, D, H, W]
    - PhysicsParameterHead: Predicts D(x), ρ(x) spatially
    - SegmentationHead: Predicts segmentation classes (pretraining)
    
Uses transposed convolutions + skip connections for upsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionGate3D, CBAM3D


class ConvBlock3D(nn.Module):
    """3D convolution + InstanceNorm + LeakyReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 padding: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.residual = nn.Sequential()
        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True)
            )
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.residual(x))


class UpBlock3D(nn.Module):
    """Upsample + concatenate skip connection + ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0, use_attention: bool = True):
        """
        Args:
            in_ch: Input channels from previous decoder level.
            skip_ch: Channels from encoder skip connection.
            out_ch: Output channels.
            use_attention: Whether to use Spatial Attention Gate on skip connection.
        """
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        
        self.use_attention = use_attention
        if use_attention and skip_ch > 0:
            self.attn_gate = AttentionGate3D(gate_ch=in_ch, skip_ch=skip_ch)
        else:
            self.attn_gate = None
            
        self.conv = ConvBlock3D(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        
        if skip is not None:
            # Handle size mismatch due to odd dimensions
            if x.shape != skip.shape:
                diff_d = skip.shape[2] - x.shape[2]
                diff_h = skip.shape[3] - x.shape[3]
                diff_w = skip.shape[4] - x.shape[4]
                x = F.pad(x, [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2,
                    diff_d // 2, diff_d - diff_d // 2,
                ])
            
            if self.attn_gate is not None:
                skip = self.attn_gate(gate=x, skip=skip)
                
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class TumorDensityDecoder(nn.Module):
    """
    Decoder that produces a dense tumor density field u(x,t) ∈ [0, 1].
    
    Takes multi-scale features from the ResNet encoder and upsamples
    to full resolution with skip connections.
    
    Architecture (for ResNet50 encoder):
        layer4 [2048, 4,4,4] → up1 [256, 8,8,8]    + skip from layer3
        up1    [256, 8,8,8]  → up2 [128, 16,16,16]  + skip from layer2
        up2    [128, 16,16,16]→ up3 [64, 32,32,32]   + skip from layer1
        up3    [64, 32,32,32] → up4 [32, 64,64,64]   + skip from stem
        up4    [32, 64,64,64] → up5 [16, 128,128,128] (no skip)
        up5    → conv1x1 → [1, 128,128,128] → sigmoid
    """

    def __init__(self, encoder_channels: dict, decoder_channels=(256, 128, 64, 32, 16),
                 dropout: float = 0.1, deep_supervision: bool = True):
        """
        Args:
            encoder_channels: Dict with keys "stem", "layer1", ..., "layer4"
                              mapping to channel counts.
            decoder_channels: Tuple of channel counts for each decoder level.
            dropout: Dropout rate in decoder blocks.
        """
        super().__init__()

        enc = encoder_channels
        dec = decoder_channels

        # Bottleneck reduction
        self.bottleneck = ConvBlock3D(enc["layer4"], dec[0], dropout=dropout)

        # Decoder levels (from deepest to shallowest)
        self.up1 = UpBlock3D(dec[0], enc["layer3"], dec[0], dropout=dropout)
        self.up2 = UpBlock3D(dec[0], enc["layer2"], dec[1], dropout=dropout)
        self.up3 = UpBlock3D(dec[1], enc["layer1"], dec[2], dropout=dropout)
        self.up4 = UpBlock3D(dec[2], enc["stem"],   dec[3], dropout=dropout)
        self.up5 = UpBlock3D(dec[3], 0,             dec[4], dropout=0.0)

        # Final 1x1 conv → tumor density
        self.final_conv = nn.Conv3d(dec[4], 1, kernel_size=1)
        
        # Deep supervision heads
        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.ds2 = nn.Conv3d(dec[3], 1, kernel_size=1)
            self.ds3 = nn.Conv3d(dec[2], 1, kernel_size=1)

    def forward(self, features: dict, return_ds: bool = False):
        """
        Args:
            features: Dict of encoder features with keys:
                "stem", "layer1", "layer2", "layer3", "layer4"
            return_ds: Whether to return deep supervision list.

        Returns:
            Tumor density field u(x,t) [B, 1, D, H, W] ∈ [0, 1].
            Or if return_ds is True, a list of outputs at different resolutions.
        """
        x = self.bottleneck(features["layer4"])

        x = self.up1(x, features["layer3"])
        x = self.up2(x, features["layer2"])
        x = self.up3(x, features["layer1"])
        ds3 = x
        
        x = self.up4(x, features["stem"])
        ds2 = x
        
        x = self.up5(x)  # No skip connection — full resolution

        out = torch.sigmoid(self.final_conv(x))  # Ensure u ∈ [0, 1]
        
        if self.deep_supervision and return_ds:
            return [out, torch.sigmoid(self.ds2(ds2)), torch.sigmoid(self.ds3(ds3))]

        return out


class PhysicsParameterHead(nn.Module):
    """
    Predicts spatially-varying physics parameters from encoder features.
    
    Outputs:
        - D(x): Diffusion coefficient field [B, 1, D, H, W]
        - ρ(x): Proliferation rate field [B, 1, D, H, W]
    
    Both outputs are constrained to positive values via softplus activation,
    and clamped to physically meaningful ranges.
    """

    def __init__(self, encoder_channels: dict,
                 diffusion_range=(0.0001, 0.5),
                 proliferation_range=(0.001, 0.5)):
        """
        Args:
            encoder_channels: Dict of channel counts from encoder.
            diffusion_range: (min, max) for D(x) in mm²/day.
            proliferation_range: (min, max) for ρ(x) in 1/day.
        """
        super().__init__()
        self.d_min, self.d_max = diffusion_range
        self.rho_min, self.rho_max = proliferation_range

        # Light decoder from layer3 features (coarser is fine for smooth params)
        in_ch = encoder_channels["layer3"]
        
        self.shared = nn.Sequential(
            nn.Conv3d(in_ch, 128, 3, padding=1, bias=False),
            nn.InstanceNorm3d(128, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(128, 64, 3, padding=1, bias=False),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # Diffusion head
        self.diffusion_head = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(32, 1, 1),
        )

        # Proliferation head
        self.proliferation_head = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(32, 1, 1),
        )

    def forward(self, features: dict, target_size: tuple = None) -> tuple:
        """
        Args:
            features: Encoder features dict.
            target_size: Target spatial size for upsampling (D, H, W).

        Returns:
            (D_field, rho_field): Each [B, 1, D, H, W], positive-valued.
        """
        x = self.shared(features["layer3"])

        # Diffusion coefficient
        D = self.diffusion_head(x)
        D = torch.sigmoid(D) * (self.d_max - self.d_min) + self.d_min

        # Proliferation rate
        rho = self.proliferation_head(x)
        rho = torch.sigmoid(rho) * (self.rho_max - self.rho_min) + self.rho_min

        # Upsample to target resolution
        if target_size is not None:
            D = F.interpolate(D, size=target_size, mode="trilinear", align_corners=False)
            rho = F.interpolate(rho, size=target_size, mode="trilinear", align_corners=False)

        return D, rho


class SegmentationHead(nn.Module):
    """
    Segmentation output head for BraTS pretraining.
    
    Produces class probabilities for each voxel.
    Uses the same decoder structure as the density decoder.
    """

    def __init__(self, encoder_channels: dict, num_classes: int = 4,
                 decoder_channels=(256, 128, 64, 32, 16), dropout: float = 0.1, deep_supervision: bool = True):
        super().__init__()

        enc = encoder_channels
        dec = decoder_channels

        self.bottleneck = ConvBlock3D(enc["layer4"], dec[0], dropout=dropout)
        self.up1 = UpBlock3D(dec[0], enc["layer3"], dec[0], dropout=dropout)
        self.up2 = UpBlock3D(dec[0], enc["layer2"], dec[1], dropout=dropout)
        self.up3 = UpBlock3D(dec[1], enc["layer1"], dec[2], dropout=dropout)
        self.up4 = UpBlock3D(dec[2], enc["stem"],   dec[3], dropout=dropout)
        self.up5 = UpBlock3D(dec[3], 0,             dec[4], dropout=0.0)
        self.final_conv = nn.Conv3d(dec[4], num_classes, kernel_size=1)
        
        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.ds2 = nn.Conv3d(dec[3], num_classes, kernel_size=1)
            self.ds3 = nn.Conv3d(dec[2], num_classes, kernel_size=1)

    def forward(self, features: dict, return_ds: bool = False):
        """
        Returns:
            Segmentation logits [B, num_classes, D, H, W] or list if return_ds.
        """
        x = self.bottleneck(features["layer4"])
        x = self.up1(x, features["layer3"])
        x = self.up2(x, features["layer2"])
        x = self.up3(x, features["layer1"])
        ds3 = x
        x = self.up4(x, features["stem"])
        ds2 = x
        x = self.up5(x)
        
        out = self.final_conv(x)
        if self.deep_supervision and return_ds:
            return [out, self.ds2(ds2), self.ds3(ds3)]
            
        return out
