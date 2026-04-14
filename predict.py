"""
Inference & Prediction Script for Hybrid Tumor Model.

Given an MRI scan, this script:
    1. Loads the trained model
    2. Predicts tumor density u(x,t)
    3. Predicts physics parameters D(x), ρ(x)
    4. Simulates tumor growth forward in time
    5. Generates segmentation
    6. Saves visualizations and NIfTI outputs

Usage:
    python predict.py --input_dir ./patient_scan --checkpoint checkpoints/finetune_best.pth
    python predict.py --input_dir ./patient_scan --checkpoint checkpoints/finetune_best.pth --simulate_days 180
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from monai.inferers import sliding_window_inference

from config import get_config
from models.hybrid_model import HybridTumorNet
from data.preprocessing import get_inference_transforms
from utils.spatial_ops import SpatialGradients3D


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint."""
    model = HybridTumorNet(
        resnet_variant=config.model.backbone,
        in_channels=config.data.num_channels,
        num_seg_classes=config.model.num_seg_classes,
        use_seg_head=config.model.use_seg_head,
        predict_physics_params=config.model.predict_diffusion,
        diffusion_range=config.model.diffusion_range,
        proliferation_range=config.model.proliferation_range,
        decoder_channels=config.model.decoder_channels,
    )
    if checkpoint_path == "random":
        print("Using randomly initialized weights for demonstration purposes.")
        model = model.to(device)
        model.eval()
        return model

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "ema_model_state_dict" in ckpt:
            model.load_state_dict(ckpt["ema_model_state_dict"], strict=False)
            print("Loaded EMA weights from checkpoint.")
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        epoch_str = ckpt.get('epoch', '?')
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using random weights.")
        epoch_str = 'unknown'
        
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path} (epoch {epoch_str})")
    return model


def load_patient_scan(input_dir, spatial_size=(128, 128, 128)):
    """Load and preprocess a patient's MRI scan."""
    import glob

    # Check if input is a direct 4-channel file (like MSD format)
    if os.path.isfile(input_dir) and input_dir.endswith(".nii.gz"):
        print(f"Loading single multi-channel scan: {os.path.basename(input_dir)}")
        files = input_dir
        ref_img = nib.load(input_dir)
        affine = ref_img.affine
    else:
        # BraTS format: Find modality files
        modalities = ["t1", "t1ce", "t2", "flair"]
        files = []
        for mod in modalities:
            matches = glob.glob(os.path.join(input_dir, f"*_{mod}.nii.gz"))
            if not matches:
                matches = glob.glob(os.path.join(input_dir, f"*{mod}*.nii.gz"))
            if matches:
                files.append(matches[0])
            else:
                raise FileNotFoundError(f"Could not find {mod} modality in {input_dir}")
        print(f"Found modalities: {[os.path.basename(f) for f in files]}")
        ref_img = nib.load(files[0])
        affine = ref_img.affine

    # Load with MONAI transforms
    transforms = get_inference_transforms(spatial_size)
    data = {"image": files}
    data = transforms(data)

    return data["image"].unsqueeze(0), affine  # Add batch dim


@torch.no_grad()
def predict(model, image, device, simulate_days=0, dt=1.0, spatial_size=(128,128,128), mc_samples=0):
    """
    Run full prediction pipeline.

    Args:
        model: Trained HybridTumorNet.
        image: Preprocessed MRI [1, 4, D, H, W].
        device: torch device.
        simulate_days: If > 0, simulate tumor growth forward.
        dt: Time step for simulation (days).
        spatial_size: Spatial size for sliding window inference.
        mc_samples: Number of Monte Carlo samples (0 for standard deterministic prediction).

    Returns:
        Dictionary with all predictions.
    """
    image = image.to(device)
    
    use_mc_dropout = mc_samples > 1
    num_samples = mc_samples if use_mc_dropout else 1

    if use_mc_dropout:
        model.train()  # Enable dropout
    else:
        model.eval()

    def forward_wrapper(x):
        out = model(x, return_features=False)
        td = out["tumor_density"]
        if isinstance(td, list): td = td[0]
        
        diff = out.get("diffusion", torch.zeros_like(td))
        prolif = out.get("proliferation", torch.zeros_like(td))
        
        seg = out.get("segmentation", torch.zeros_like(td).repeat(1, 4, 1, 1, 1))
        
        return torch.cat([td, diff, prolif, seg], dim=1)

    all_outputs = []
    print(f"Running inference (Sliding Window, ROI: {spatial_size})...")
    
    for idx in range(num_samples):
        if use_mc_dropout:
            print(f"  MC Dropout Sample {idx+1}/{num_samples}...")
        
        full_out = sliding_window_inference(
            inputs=image,
            roi_size=spatial_size,
            sw_batch_size=1,
            predictor=forward_wrapper,
            overlap=0.5
        )
        all_outputs.append(full_out)

    all_outputs = torch.stack(all_outputs, dim=0) # [samples, B, C, D, H, W]
    mean_out = all_outputs.mean(dim=0)
    std_out = all_outputs.std(dim=0) if use_mc_dropout else None

    results = {
        "tumor_density": mean_out[0, 0].cpu().numpy(),
        "diffusion": mean_out[0, 1].cpu().numpy(),
        "proliferation": mean_out[0, 2].cpu().numpy(),
    }
    
    if use_mc_dropout:
        results["uncertainty"] = std_out[0, 0].cpu().numpy()

    # Segmentation
    seg_logits = mean_out[:, 3:]
    if seg_logits.shape[1] > 0 and seg_logits.sum() != 0:
        seg = torch.argmax(seg_logits, dim=1)
        results["segmentation"] = seg.cpu().numpy()[0]

    # Tumor growth simulation
    if simulate_days > 0:
        print(f"Simulating tumor growth for {simulate_days} days...")
        num_steps = int(simulate_days / dt)
        trajectory = model.predict_tumor_growth(
            image, num_steps=num_steps, dt=dt
        )
        results["growth_trajectory"] = [
            t.cpu().numpy()[0, 0] for t in trajectory
        ]
        print(f"  Generated {len(trajectory)} time steps")

    # Compute tumor statistics
    u = results["tumor_density"]
    results["stats"] = {
        "tumor_volume_voxels": int((u > 0.5).sum()),
        "tumor_volume_ml": float((u > 0.5).sum() * 0.001),  # assuming 1mm³ voxels
        "max_density": float(u.max()),
        "mean_density_in_tumor": float(u[u > 0.1].mean()) if (u > 0.1).any() else 0,
        "mean_diffusion": float(results["diffusion"].mean()) if results["diffusion"].ndim > 1 else 0,
        "mean_proliferation": float(results["proliferation"].mean()) if results["proliferation"].ndim > 1 else 0,
    }

    return results


def save_nifti(data, affine, path):
    """Save a 3D numpy array as NIfTI file."""
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, path)
    print(f"  Saved: {path}")


def visualize_results(results, output_dir, num_slices=5):
    """Generate visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    u = results["tumor_density"]
    D, H, W = u.shape

    # Custom colormap: dark blue → cyan → yellow → red
    colors = ["#0d0887", "#46039f", "#7201a8", "#9c179e",
              "#bd3786", "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"]
    tumor_cmap = LinearSegmentedColormap.from_list("tumor", colors)

    # --- 1. Tumor Density Slices ---
    fig, axes = plt.subplots(1, num_slices, figsize=(4 * num_slices, 4))
    slice_indices = np.linspace(D * 0.3, D * 0.7, num_slices, dtype=int)

    for i, si in enumerate(slice_indices):
        ax = axes[i] if num_slices > 1 else axes
        im = ax.imshow(u[si], cmap=tumor_cmap, vmin=0, vmax=1)
        ax.set_title(f"Slice {si}", fontsize=12)
        ax.axis("off")

    plt.colorbar(im, ax=axes, shrink=0.8, label="Tumor Density")
    plt.suptitle("Predicted Tumor Density u(x,t)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tumor_density.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- 2. Segmentation (if available) ---
    if "segmentation" in results:
        seg = results["segmentation"]
        fig, axes = plt.subplots(1, num_slices, figsize=(4 * num_slices, 4))
        seg_cmap = plt.cm.get_cmap("Set1", 4)

        for i, si in enumerate(slice_indices):
            ax = axes[i] if num_slices > 1 else axes
            ax.imshow(seg[si], cmap=seg_cmap, vmin=0, vmax=3)
            ax.set_title(f"Slice {si}", fontsize=12)
            ax.axis("off")

        plt.suptitle("Segmentation (0:BG, 1:NCR, 2:ED, 3:ET)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "segmentation.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- 3. Physics Parameters ---
    if results["diffusion"].ndim > 1:
        fig, axes = plt.subplots(2, num_slices, figsize=(4 * num_slices, 8))
        mid = D // 2

        for i, si in enumerate(slice_indices):
            axes[0, i].imshow(results["diffusion"][si], cmap="hot")
            axes[0, i].set_title(f"D(x) slice {si}")
            axes[0, i].axis("off")

            axes[1, i].imshow(results["proliferation"][si], cmap="hot")
            axes[1, i].set_title(f"ρ(x) slice {si}")
            axes[1, i].axis("off")

        plt.suptitle("Learned Physics Parameters", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "physics_params.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- 4. Growth Trajectory (if simulated) ---
    if "growth_trajectory" in results:
        traj = results["growth_trajectory"]
        n_frames = min(8, len(traj))
        frame_indices = np.linspace(0, len(traj) - 1, n_frames, dtype=int)
        mid_slice = D // 2

        fig, axes = plt.subplots(1, n_frames, figsize=(3 * n_frames, 3))
        for i, fi in enumerate(frame_indices):
            ax = axes[i] if n_frames > 1 else axes
            ax.imshow(traj[fi][mid_slice], cmap=tumor_cmap, vmin=0, vmax=1)
            ax.set_title(f"Day {fi}", fontsize=10)
            ax.axis("off")

        plt.suptitle("Tumor Growth Simulation", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "growth_simulation.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- 5. Volume Growth Curve ---
    if "growth_trajectory" in results:
        volumes = [(t > 0.5).sum() for t in results["growth_trajectory"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(volumes, linewidth=2, color="#e74c3c")
        ax.fill_between(range(len(volumes)), volumes, alpha=0.2, color="#e74c3c")
        ax.set_xlabel("Time (days)", fontsize=12)
        ax.set_ylabel("Tumor Volume (voxels)", fontsize=12)
        ax.set_title("Predicted Tumor Volume Over Time", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "volume_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- 6. Uncertainty Map (if MC Dropout used) ---
    if "uncertainty" in results:
        unc = results["uncertainty"]
        fig, axes = plt.subplots(1, num_slices, figsize=(4 * num_slices, 4))
        for i, si in enumerate(slice_indices):
            ax = axes[i] if num_slices > 1 else axes
            im = ax.imshow(unc[si], cmap="magma")
            ax.set_title(f"Slice {si}", fontsize=12)
            ax.axis("off")
            
        plt.colorbar(im, ax=axes, shrink=0.8, label="Uncertainty (Std Dev)")
        plt.suptitle("Tumor Density Uncertainty (MC Dropout)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "uncertainty.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Prediction")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing patient MRI files")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./predictions",
                        help="Output directory for results")
    parser.add_argument("--simulate_days", type=int, default=0,
                        help="Days to simulate tumor growth (0 = no sim)")
    parser.add_argument("--dt", type=float, default=1.0,
                        help="Time step for simulation")
    parser.add_argument("--mc_samples", type=int, default=0,
                        help="Number of MC Dropout samples for uncertainty (0 to disable)")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = get_config()
    config.model.backbone = args.backbone
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Load patient scan
    print(f"Loading patient scan from {args.input_dir}...")
    image, affine = load_patient_scan(args.input_dir, config.data.spatial_size)

    # Predict
    print("Running prediction...")
    results = predict(model, image, device,
                      simulate_days=args.simulate_days, dt=1.0)

    # Print stats
    print("\n" + "=" * 50)
    print("TUMOR ANALYSIS RESULTS")
    print("=" * 50)
    for k, v in results["stats"].items():
        print(f"  {k}: {v}")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    save_nifti(results["tumor_density"], affine,
               os.path.join(args.output_dir, "tumor_density.nii.gz"))

    if "segmentation" in results:
        save_nifti(results["segmentation"].astype(np.float32), affine,
                   os.path.join(args.output_dir, "segmentation.nii.gz"))

    if results["diffusion"].ndim > 1:
        save_nifti(results["diffusion"], affine,
                   os.path.join(args.output_dir, "diffusion_D.nii.gz"))
        save_nifti(results["proliferation"], affine,
                   os.path.join(args.output_dir, "proliferation_rho.nii.gz"))

    # Visualize
    visualize_results(results, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
