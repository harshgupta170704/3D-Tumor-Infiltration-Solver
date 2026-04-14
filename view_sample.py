import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def show_sample(data_dir="./data/Task01_BrainTumour"):
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")
    
    # Wait until images are available
    img_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    lbl_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
    
    if not img_files or not lbl_files:
        print("Data is still downloading or hasn't been extracted yet.")
        print("Please run this script again once the download completes!")
        return
        
    img_path = img_files[0] # Pick the first patient
    lbl_path = lbl_files[0]
    
    print(f"Loading MRI scan: {os.path.basename(img_path)}")
    
    # Load NIfTI files
    img = nib.load(img_path).get_fdata() # Shape: (H, W, D, 4)
    lbl = nib.load(lbl_path).get_fdata() # Shape: (H, W, D)
    
    # Pick a random middle slice where the tumor usually is
    slice_idx = img.shape[2] // 2
    
    # Create the visualization
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    # Modalities correspond to [FLAIR, T1, T1c, T2] in this specific BraTS mapping
    titles = ["FLAIR", "T1", "T1ce (Contrast)", "T2", "Segmentation Mask"]
    
    for i in range(4):
        im = axes[i].imshow(img[:, :, slice_idx, i].T, cmap="gray", origin="lower")
        axes[i].set_title(titles[i], fontsize=14, fontweight="bold")
        axes[i].axis("off")
        
    # Show segmentation mask with a color map
    # 0: BG, 1: Edema, 2: Non-Enhancing Core, 3: Enhancing Core
    cmap = plt.cm.get_cmap("nipy_spectral", 4)
    axes[4].imshow(lbl[:, :, slice_idx].T, cmap=cmap, origin="lower", vmin=0, vmax=3)
    axes[4].set_title(titles[4], fontsize=14, fontweight="bold")
    axes[4].axis("off")
    
    plt.tight_layout()
    output_path = "mri_sample_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Successfully saved demonstration slice to: {output_path}")

if __name__ == "__main__":
    show_sample()
