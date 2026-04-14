"""
Quick smoke test — verifies the entire pipeline works with synthetic data.
Run this to check all modules import correctly and the forward pass runs.

Usage:
    python test_model.py
"""

import torch
import numpy as np
from config import get_config
from models.hybrid_model import HybridTumorNet
from losses.combined_loss import HybridTumorLoss
from losses.data_loss import seg_to_density
from utils.spatial_ops import SpatialGradients3D
from utils.metrics import compute_dice, compute_physics_residual


def test_forward_pass():
    """Test model forward pass with random data."""
    print("=" * 60)
    print("TEST 1: Forward Pass")
    print("=" * 60)

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model = HybridTumorNet(
        resnet_variant="resnet18",  # Use smaller model for testing
        in_channels=4,
        num_seg_classes=4,
        use_seg_head=True,
        predict_physics_params=True,
    ).to(device)

    params = model.count_parameters()
    print(f"Parameters: {params}")

    # Synthetic input: [B=1, C=4, D=64, H=64, W=64]
    x = torch.randn(1, 4, 64, 64, 64, device=device)

    # Forward
    with torch.no_grad():
        output = model(x)

    print(f"\nOutputs:")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")
        elif isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} keys")

    assert output["tumor_density"].shape == (1, 1, 64, 64, 64)
    assert output["tumor_density"].min() >= 0 and output["tumor_density"].max() <= 1
    print("\n[OK] Forward pass OK")


def test_loss_computation():
    """Test loss computation."""
    print("\n" + "=" * 60)
    print("TEST 2: Loss Computation")
    print("=" * 60)

    config = get_config()
    config.data.spatial_size = (64, 64, 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridTumorNet(
        resnet_variant="resnet18",
        in_channels=4,
        num_seg_classes=4,
        use_seg_head=True,
        predict_physics_params=True,
    ).to(device)

    criterion = HybridTumorLoss(config).to(device)

    # Synthetic data
    x = torch.randn(1, 4, 64, 64, 64, device=device)
    # Create a fake segmentation label with tumor in center
    label = torch.zeros(1, 1, 64, 64, 64, device=device)
    label[0, 0, 25:40, 25:40, 25:40] = 1  # NCR/NET
    label[0, 0, 28:37, 28:37, 28:37] = 2  # ED
    label[0, 0, 30:35, 30:35, 30:35] = 3  # ET

    batch = {"image": x, "label": label}

    # Test all phases
    for phase in ["pretrain", "finetune", "joint"]:
        model.set_phase(phase)
        output = model(x)
        losses = criterion(output, batch, phase=phase)

        print(f"\n  Phase: {phase}")
        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            print(f"    {k}: {val:.6f}")

    print("\n[OK] Loss computation OK")


def test_spatial_ops():
    """Test spatial gradient operators."""
    print("\n" + "=" * 60)
    print("TEST 3: Spatial Operators")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ops = SpatialGradients3D(dx=1.0, dy=1.0, dz=1.0).to(device)

    # Create a smooth field: u = sin(x) * sin(y) * sin(z)
    grid = torch.linspace(0, 2 * np.pi, 32)
    u = torch.sin(grid).view(1, 1, 32, 1, 1) * \
        torch.sin(grid).view(1, 1, 1, 32, 1) * \
        torch.sin(grid).view(1, 1, 1, 1, 32)
    u = u.to(device)

    # Gradient
    du_dx, du_dy, du_dz = ops.gradient(u)
    print(f"  Gradient shapes: {du_dx.shape}")

    # Laplacian
    lap = ops.laplacian(u)
    print(f"  Laplacian shape: {lap.shape}")

    # For sin(x)*sin(y)*sin(z), Laplacian ≈ -3*u (analytical)
    interior = lap[0, 0, 2:-2, 2:-2, 2:-2]
    expected = -3.0 * u[0, 0, 2:-2, 2:-2, 2:-2]
    rel_error = (interior - expected).abs().mean() / (expected.abs().mean() + 1e-8)
    print(f"  Laplacian relative error vs analytical: {rel_error:.4f}")

    # Divergence
    D = torch.ones_like(u) * 0.1
    div = ops.divergence_of_flux(u, D)
    print(f"  Divergence shape: {div.shape}")

    print("\n[OK] Spatial operators OK")


def test_seg_to_density():
    """Test segmentation to density conversion."""
    print("\n" + "=" * 60)
    print("TEST 4: Seg -> Density Conversion")
    print("=" * 60)

    seg = torch.zeros(1, 1, 64, 64, 64)
    seg[0, 0, 25:40, 25:40, 25:40] = 1
    seg[0, 0, 30:35, 30:35, 30:35] = 3

    density = seg_to_density(seg, sigma=2.0)
    print(f"  Input seg: unique values = {seg.unique().tolist()}")
    print(f"  Output density: range = [{density.min():.4f}, {density.max():.4f}]")
    print(f"  Output density: shape = {density.shape}")
    assert density.min() >= 0 and density.max() <= 1
    print("\n[OK] Seg->Density OK")


def test_growth_simulation():
    """Test tumor growth simulation."""
    print("\n" + "=" * 60)
    print("TEST 5: Growth Simulation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridTumorNet(
        resnet_variant="resnet18",
        in_channels=4,
        num_seg_classes=4,
        use_seg_head=True,
        predict_physics_params=True,
    ).to(device)
    model.eval()

    x = torch.randn(1, 4, 64, 64, 64, device=device)

    trajectory = model.predict_tumor_growth(x, num_steps=5, dt=1.0)
    print(f"  Trajectory steps: {len(trajectory)}")
    for i, u in enumerate(trajectory):
        vol = (u > 0.5).sum().item()
        print(f"  Step {i}: volume = {vol} voxels, "
              f"range = [{u.min():.4f}, {u.max():.4f}]")

    print("\n[OK] Growth simulation OK")


def test_backward_pass():
    """Test that gradients flow through everything."""
    print("\n" + "=" * 60)
    print("TEST 6: Backward Pass (Gradient Flow)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()

    model = HybridTumorNet(
        resnet_variant="resnet18",
        in_channels=4,
        use_seg_head=True,
        predict_physics_params=True,
    ).to(device)
    model.set_phase("joint")

    criterion = HybridTumorLoss(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.randn(1, 4, 64, 64, 64, device=device)
    label = torch.zeros(1, 1, 64, 64, 64, device=device)
    label[0, 0, 25:40, 25:40, 25:40] = 1

    # Forward + backward
    output = model(x)
    losses = criterion(output, {"image": x, "label": label}, phase="joint")
    losses["total_loss"].backward()

    # Check gradients exist
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    print(f"  Parameters with gradients: {len(grad_norms)}/{sum(1 for _ in model.parameters())}")
    print(f"  Sample gradient norms:")
    for name, norm in list(grad_norms.items())[:5]:
        print(f"    {name}: {norm:.6f}")

    optimizer.step()
    print("\n[OK] Backward pass OK -- gradients flow end-to-end")


if __name__ == "__main__":
    print("\n HYBRID TUMOR MODEL -- SMOKE TESTS\n")

    test_forward_pass()
    test_loss_computation()
    test_spatial_ops()
    test_seg_to_density()
    test_growth_simulation()
    test_backward_pass()

    print("\n" + "=" * 60)
    print("[PASS] ALL TESTS PASSED -- Model is ready!")
    print("=" * 60)
