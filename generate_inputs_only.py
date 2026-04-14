"""
Generate a single composite image showing ONLY the 5 input MRI scans.
One row per patient, 4 modalities per row.
"""
import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, Compose, EnsureTyped
)

def load_mri_for_display(filepath):
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0)),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])
    data = transforms({"image": filepath})
    return data["image"].numpy()  # [4, D, H, W]

def get_middle_slice(vol_4ch):
    """Return middle axial slice from 4-channel volume."""
    mid = vol_4ch.shape[-1] // 2
    return vol_4ch[:, :, :, mid]  # [4, H, W]

files = sorted(glob.glob(
    r"data\Task01_BrainTumour\imagesTr\BRATS_*.nii.gz"
))
files = [f for f in files if not os.path.basename(f).startswith("._")][:5]

MOD_NAMES = ["T1\n(Anatomy)", "T1c\n(Active Tumor / Blood Flow)",
             "T2\n(Edema / Water Content)", "FLAIR\n(Swelling)"]
PATIENT_LABELS = [os.path.basename(f).split(".")[0] for f in files]

fig = plt.figure(figsize=(20, 14), facecolor="black")
fig.suptitle(
    "MRI Input Data - 5 Patients x 4 Modalities\n"
    "Hybrid MONAI ResNet + PINN | Brain Tumor Decathlon Dataset",
    fontsize=18, fontweight="bold", color="white", y=0.98
)

gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.1, wspace=0.05)

for row, (fpath, label) in enumerate(zip(files, PATIENT_LABELS)):
    print(f"Loading {label} ({row+1}/5)...")
    vol = load_mri_for_display(fpath)          # [4, D, H, W]
    slices = get_middle_slice(vol)              # [4, H, W]

    for col in range(4):
        ax = fig.add_subplot(gs[row, col])
        img = slices[col]
        # Robust normalization for display
        if img.max() > 0:
            vmax = np.percentile(img[img > 0], 99.5)
            ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, aspect="auto")
        else:
            ax.imshow(img, cmap="gray", aspect="auto")
            
        ax.set_facecolor("black")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(0.5)

        # Column headers (top row only)
        if row == 0:
            ax.set_title(MOD_NAMES[col], fontsize=13, color="white", pad=12, fontweight="bold")

        # Patient label (left column only)
        if col == 0:
            ax.set_ylabel(label, fontsize=12, color="#00e5ff", rotation=0, labelpad=70, va="center", fontweight="bold")

out_path = r"final_output\ppt_images\ALL_5_INPUTS_ONLY.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="black", edgecolor="none")
plt.close()
print(f"Success! Image saved to: {out_path}")
