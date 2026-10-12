"""
MONAI 3D ResNet Encoder with multi-scale feature extraction.

Wraps MONAI's ResNet to expose intermediate features at multiple
spatial resolutions for the decoder (skip connections).

Architecture (ResNet50):
    Input: [B, 4, 128, 128, 128]
    
    stem  (conv1 + bn + relu + pool):  [B, 64,  64, 64, 64]
    layer1 (3 bottleneck blocks):      [B, 256, 32, 32, 32]
    layer2 (4 bottleneck blocks):      [B, 512, 16, 16, 16]
    layer3 (6 bottleneck blocks):      [B, 1024, 8,  8,  8]
    layer4 (3 bottleneck blocks):      [B, 2048, 4,  4,  4]
"""

import torch
import torch.nn as nn
from collections import OrderedDict

from monai.networks.nets import ResNet, resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200


# Channel dimensions for each ResNet variant at each stage
RESNET_CHANNELS = {
    "resnet10":  {"stem": 64, "layer1": 64,  "layer2": 128, "layer3": 256, "layer4": 512},
    "resnet18":  {"stem": 64, "layer1": 64,  "layer2": 128, "layer3": 256, "layer4": 512},
    "resnet34":  {"stem": 64, "layer1": 64,  "layer2": 128, "layer3": 256, "layer4": 512},
    "resnet50":  {"stem": 64, "layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
    "resnet101": {"stem": 64, "layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
    "resnet152": {"stem": 64, "layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
    "resnet200": {"stem": 64, "layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
}

# Block type for each variant
RESNET_BLOCKS = {
    "resnet10": "basic", "resnet18": "basic", "resnet34": "basic",
    "resnet50": "bottleneck", "resnet101": "bottleneck", "resnet152": "bottleneck",
    "resnet200": "bottleneck",
}

# Layer configs
RESNET_LAYERS = {
    "resnet10": [1, 1, 1, 1], "resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3],
    "resnet50": [3, 4, 6, 3], "resnet101": [3, 4, 23, 3], "resnet152": [3, 8, 36, 3],
    "resnet200": [3, 24, 36, 3],
}


class ResNetEncoder(nn.Module):
    """
    3D ResNet encoder that returns multi-scale feature maps.
    
    Wraps MONAI's ResNet and provides an interface to extract
    features at each resolution level for skip connections.
    """

    def __init__(self, variant: str = "resnet50", in_channels: int = 4,
                 pretrained_path: str = None, freeze_layers: int = 0):
        """
        Args:
            variant: ResNet variant name (e.g., "resnet50").
            in_channels: Number of input channels (4 for multi-modal MRI).
            pretrained_path: Path to pretrained weights (.pth file).
            freeze_layers: Number of initial layers to freeze (0-5).
                0 = train all, 1 = freeze stem, 2 = freeze stem+layer1, etc.
        """
        super().__init__()
        self.variant = variant
        self.channels = RESNET_CHANNELS[variant]

        # Build the ResNet backbone
        block_type = RESNET_BLOCKS[variant]
        layers = RESNET_LAYERS[variant]

        if block_type == "bottleneck":
            block_inplanes = [64, 128, 256, 512]
        else:
            block_inplanes = [64, 128, 256, 512]

        self.backbone = ResNet(
            block=block_type,
            layers=layers,
            block_inplanes=block_inplanes,
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=1,  # Placeholder — we won't use the FC layer
            conv1_t_stride=2,
        )

        # Load pretrained weights if provided
        if pretrained_path:
            self._load_pretrained(pretrained_path, in_channels)

        # Freeze layers
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def _load_pretrained(self, weights_path: str, in_channels: int):
        """Load pretrained weights with flexible key matching."""
        print(f"Loading pretrained weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu")

        # Handle DataParallel prefix
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model_dict = self.backbone.state_dict()

        # Filter compatible weights
        compatible = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    compatible[k] = v
                elif k == "conv1.weight" and v.shape[1] != in_channels:
                    # Handle channel mismatch in first conv
                    print(f"  Adapting conv1: {v.shape[1]}ch → {in_channels}ch")
                    if in_channels > v.shape[1]:
                        # Repeat channels
                        repeats = in_channels // v.shape[1] + 1
                        v = v.repeat(1, repeats, 1, 1, 1)[:, :in_channels]
                    else:
                        v = v[:, :in_channels]
                    compatible[k] = v
                else:
                    print(f"  Skipping {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")

        model_dict.update(compatible)
        self.backbone.load_state_dict(model_dict, strict=False)
        print(f"  Loaded {len(compatible)}/{len(model_dict)} parameters")

    def _freeze_layers(self, n_layers: int):
        """Freeze the first n_layers of the network."""
        layer_groups = [
            [self.backbone.conv1, self.backbone.bn1],
            [self.backbone.layer1],
            [self.backbone.layer2],
            [self.backbone.layer3],
            [self.backbone.layer4],
        ]
        for i in range(min(n_layers, len(layer_groups))):
            for module in layer_groups[i]:
                for param in module.parameters():
                    param.requires_grad = False
            print(f"  Froze layer group {i}")

    def forward(self, x: torch.Tensor) -> dict:
        """
        Extract multi-scale features from the input volume.

        Args:
            x: Input tensor [B, C, D, H, W] where C = num modalities.

        Returns:
            Dictionary with feature maps at each scale:
                "stem":   [B, 64,   D/2,  H/2,  W/2]
                "layer1": [B, 256,  D/4,  H/4,  W/4]   (ResNet50)
                "layer2": [B, 512,  D/8,  H/8,  W/8]
                "layer3": [B, 1024, D/16, H/16, W/16]
                "layer4": [B, 2048, D/32, H/32, W/32]
        """
        features = {}

        # Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act(x)
        features["stem"] = x  # [B, 64, D/2, H/2, W/2]

        x = self.backbone.maxpool(x)

        # Residual layers
        x = self.backbone.layer1(x)
        features["layer1"] = x  # [B, 256, D/4, H/4, W/4]

        x = self.backbone.layer2(x)
        features["layer2"] = x  # [B, 512, D/8, H/8, W/8]

        x = self.backbone.layer3(x)
        features["layer3"] = x  # [B, 1024, D/16, H/16, W/16]

        x = self.backbone.layer4(x)
        features["layer4"] = x  # [B, 2048, D/32, H/32, W/32]

        return features

    def get_channels(self) -> dict:
        """Return the channel dimensions at each feature level."""
        return self.channels.copy()
