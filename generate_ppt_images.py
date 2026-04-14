import os
import glob
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from config import get_config
from predict import load_model, predict, load_patient_scan

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config()
    
    # Load model from your newly trained checkpoint
    model = load_model("checkpoints/finetune_best.pth", config, device)
    
    # Get 5 valid patient brain MRI scans
    files = sorted(glob.glob("data/Task01_BrainTumour/imagesTr/BRATS_*.nii.gz"))
    files = [f for f in files if not os.path.basename(f).startswith("._")][:5]
    
    out_dir = "final_output/ppt_images"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Generating input-output pairs for {len(files)} patients...")
    
    for idx, fpath in enumerate(files):
        patient_name = os.path.basename(fpath).split(".")[0]
        print(f"\nProcessing {patient_name} ({idx+1}/5)...")
        
        # 1. Load Data
        image, affine = load_patient_scan(fpath, config.data.spatial_size)
        
        # 2. Predict (including 30-day growth simulation)
        results = predict(model, image, device, simulate_days=30, dt=1.0)
        
        # 3. Create a beautiful comparison plot
        mid = image.shape[-1] // 2  # Get the middle physical slice (Z-axis)
        
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle(f"Patient Inference: {patient_name}", fontsize=20, fontweight='bold', y=1.02)
        
        # --- Top Row: Inputs (MRI Modalities) ---
        img_np = image.cpu().numpy()[0, :, :, :, mid] # [4, H, W]
        mod_names = ["Input: T1 (Anatomy)", "Input: T1c (Blood Flow)", "Input: T2 (Water/Edema)", "Input: FLAIR (Swelling)"]
        for j in range(4):
            ax = axes[0, j]
            # Normalize display for MRI
            slice_data = img_np[j]
            if slice_data.max() > 0:
                ax.imshow(slice_data, cmap="gray", vmin=0, vmax=np.percentile(slice_data, 99))
            else:
                ax.imshow(slice_data, cmap="gray")
            ax.set_title(mod_names[j], fontsize=14)
            ax.axis("off")
            
        # --- Bottom Row: Model Outputs ---
        # 1. Segmentation
        ax = axes[1, 0]
        if "segmentation" in results:
            ax.imshow(results["segmentation"][:, :, mid], cmap="Set1")
            ax.set_title("1. Model Output: Segmentation", fontsize=14)
        ax.axis("off")
        
        # 2. Current Tumor Density
        ax = axes[1, 1]
        im = ax.imshow(results["tumor_density"][:, :, mid], cmap="hot", vmin=0, vmax=1)
        ax.set_title("2. Model Output: Cellular Density", fontsize=14)
        ax.axis("off")
        
        # 3. Physics Parameters
        ax = axes[1, 2]
        im2 = ax.imshow(results["diffusion"][:, :, mid]*1000, cmap="magma")
        ax.set_title("3. PINN Physics: Diffusion D(x)", fontsize=14)
        ax.axis("off")
        
        # 4. Simulation
        ax = axes[1, 3]
        if "growth_trajectory" in results:
            ax.imshow(results["growth_trajectory"][-1][:, :, mid], cmap="hot", vmin=0, vmax=1)
            ax.set_title("4. PDE Simulation: Density at 30 Days", fontsize=14)
        ax.axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{patient_name}_inference.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved highly detailed graphical summary to: {save_path}")

if __name__ == "__main__":
    main()
