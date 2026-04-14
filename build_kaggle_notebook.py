"""Build the self-contained Kaggle notebook."""
import json, os

nb = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {"accelerator": "gpu", "dataSources": [], "isGpuEnabled": True, "isInternetEnabled": True}
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": []
}

def md(src):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src.split("\n")})

def code(src):
    nb["cells"].append({"cell_type": "code", "metadata": {"trusted": True}, "source": src.split("\n"), "outputs": [], "execution_count": None})

# Helper to read a file
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

BASE = r"c:\Users\Lenovo\OneDrive\Desktop\Fine tuning monia resnet modal with pinn modal"

# ========== CELL 0: Title ==========
md("""# 🧠 Hybrid MONAI ResNet + Physics-Informed Neural Network (PINN)
## Brain Tumor Segmentation & Growth Prediction — Full Training Pipeline

This notebook runs the **complete** two-phase training pipeline:
1. **Phase 1 (Pretrain):** Segmentation pretraining on BraTS MSD data (100 epochs)
2. **Phase 2 (Finetune):** Physics-informed fine-tuning with Fisher-KPP PDE constraints (100 epochs)
3. **Inference:** Predict tumor density, segmentation, physics parameters, and 30-day growth simulation

**GPU Required:** Enable GPU in Kaggle Settings (T4 x2 or P100)
""")

# ========== CELL 1: Install ==========
code("""# Install dependencies
!pip install -q "monai[nibabel,tqdm,einops]" nibabel SimpleITK nilearn medpy tensorboard scipy scikit-learn
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
""")

# ========== CELL 2: Project Structure ==========
code("""# Create project structure
import os
for d in ["models", "losses", "data", "utils", "checkpoints", "logs", "final_output"]:
    os.makedirs(d, exist_ok=True)
# Create __init__.py files
for d in ["models", "losses", "data", "utils"]:
    with open(os.path.join(d, "__init__.py"), "w") as f:
        f.write("")
print("Project structure created.")
""")

# ========== CELL 3: Write all source files ==========
files_to_embed = [
    ("utils/spatial_ops.py", os.path.join(BASE, "utils", "spatial_ops.py")),
    ("utils/metrics.py", os.path.join(BASE, "utils", "metrics.py")),
    ("utils/ema.py", os.path.join(BASE, "utils", "ema.py")),
    ("models/attention.py", os.path.join(BASE, "models", "attention.py")),
    ("models/resnet_backbone.py", os.path.join(BASE, "models", "resnet_backbone.py")),
    ("models/decoder.py", os.path.join(BASE, "models", "decoder.py")),
    ("models/hybrid_model.py", os.path.join(BASE, "models", "hybrid_model.py")),
    ("data/preprocessing.py", os.path.join(BASE, "data", "preprocessing.py")),
    ("data/synthetic_longitudinal.py", os.path.join(BASE, "data", "synthetic_longitudinal.py")),
    ("data/dataset.py", os.path.join(BASE, "data", "dataset.py")),
    ("losses/physics_loss.py", os.path.join(BASE, "losses", "physics_loss.py")),
    ("losses/data_loss.py", os.path.join(BASE, "losses", "data_loss.py")),
    ("losses/combined_loss.py", os.path.join(BASE, "losses", "combined_loss.py")),
    ("config.py", os.path.join(BASE, "config.py")),
    ("train.py", os.path.join(BASE, "train.py")),
    ("predict.py", os.path.join(BASE, "predict.py")),
]

for target_path, source_path in files_to_embed:
    content = read_file(source_path)
    # Escape for embedding
    escaped = content.replace("\\", "\\\\").replace('"""', '\\"\\"\\"').replace("'''", "\\'\\'\\'")
    
    cell_code = f'''# Write {target_path}
with open("{target_path}", "w") as f:
    f.write("""{escaped}""")
print("Written: {target_path}")'''
    
    code(cell_code)

# ========== CELL: Fix config for Kaggle ==========
md("## ⚙️ Configure for Kaggle GPU Training (100 epochs)")

code("""# Patch config.py for Kaggle: 100 epochs, batch_size=2, num_workers=2
import re

with open("config.py", "r") as f:
    cfg = f.read()

# Set epochs to 100
cfg = re.sub(r'num_epochs_pretrain: int = \\d+', 'num_epochs_pretrain: int = 100', cfg)
cfg = re.sub(r'num_epochs_finetune: int = \\d+', 'num_epochs_finetune: int = 100', cfg)
# Set batch_size=2 for GPU
cfg = re.sub(r'batch_size: int = \\d+', 'batch_size: int = 2', cfg)
# Set num_workers=2 for Kaggle
cfg = re.sub(r'num_workers: int = \\d+', 'num_workers: int = 2', cfg)
# Set device=cuda
cfg = re.sub(r'device: str = "\\w+"', 'device: str = "cuda"', cfg)

with open("config.py", "w") as f:
    f.write(cfg)

print("Config patched for Kaggle GPU training (100 epochs, batch=2, cuda)")
""")

# ========== CELL: Download Data ==========
md("## 📥 Download BraTS MSD Dataset (~4.6 GB)")

code("""# Download Medical Segmentation Decathlon - Brain Tumour
from monai.apps import download_and_extract
import glob, json, numpy as np

DATA_DIR = "./data"
MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
task_dir = os.path.join(DATA_DIR, "Task01_BrainTumour")

if not os.path.exists(task_dir) or len(glob.glob(os.path.join(task_dir, "imagesTr", "*.nii.gz"))) == 0:
    print("Downloading BraTS MSD dataset...")
    download_and_extract(
        url=MSD_URL,
        filepath=os.path.join(DATA_DIR, "Task01_BrainTumour.tar"),
        output_dir=DATA_DIR,
    )
    # Clean up tar
    tar_path = os.path.join(DATA_DIR, "Task01_BrainTumour.tar")
    if os.path.exists(tar_path):
        os.remove(tar_path)
        print("Removed .tar to save space.")
else:
    print(f"Dataset already exists at {task_dir}")

# Verify
n_images = len(glob.glob(os.path.join(task_dir, "imagesTr", "*.nii.gz")))
n_labels = len(glob.glob(os.path.join(task_dir, "labelsTr", "*.nii.gz")))
print(f"Training images: {n_images}, Labels: {n_labels}")
""")

# ========== CELL: Phase 1 ==========
md("## 🔬 Phase 1: Segmentation Pretraining (100 epochs)")

code("""# Phase 1: Pretrain segmentation
!python train.py --data_root data/Task01_BrainTumour --phase pretrain --device cuda --batch_size 2
""")

# ========== CELL: Phase 2 ==========
md("## ⚛️ Phase 2: Physics-Informed Fine-Tuning (100 epochs)")

code("""# Phase 2: Finetune with physics constraints
!python train.py --data_root data/Task01_BrainTumour --phase finetune --device cuda --batch_size 1 --resume checkpoints/pretrain_best.pth
""")

# ========== CELL: Visualize Input ==========
md("## 🖼️ Visualize Input MRI & Output Predictions")

code("""# Visualize a sample input MRI
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

sample_path = glob.glob("data/Task01_BrainTumour/imagesTr/BRATS_*.nii.gz")[0]
img = nib.load(sample_path)
data = img.get_fdata()

modality_names = ["FLAIR", "T1", "T1ce", "T2"]
mid = data.shape[2] // 2

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i in range(4):
    s = data[:, :, mid, i]
    s = (s - s.min()) / (s.max() + 1e-8)
    axes[i].imshow(np.rot90(s), cmap='gray')
    axes[i].set_title(modality_names[i], fontsize=15)
    axes[i].axis('off')
plt.suptitle(f"Input MRI: {os.path.basename(sample_path)} (Slice {mid})", fontsize=18)
plt.tight_layout()
plt.savefig("input_preview.png", dpi=150)
plt.show()
print("Input preview saved.")
""")

# ========== CELL: Run Prediction ==========
md("## 🔮 Run Inference & Generate Output Images")

code("""# Run prediction on a real patient scan
import glob
sample = glob.glob("data/Task01_BrainTumour/imagesTr/BRATS_*.nii.gz")[0]
!python predict.py --input_dir "{sample}" --checkpoint checkpoints/finetune_best.pth --simulate_days 30 --output_dir final_output --device cuda
""")

# ========== CELL: Display Results ==========
code("""# Display all output images
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

output_images = [
    ("Tumor Density", "final_output/tumor_density.png"),
    ("Segmentation", "final_output/segmentation.png"),
    ("Physics Parameters", "final_output/physics_params.png"),
    ("Growth Simulation", "final_output/growth_simulation.png"),
    ("Volume Curve", "final_output/volume_curve.png"),
]

for title, path in output_images:
    if os.path.exists(path):
        print(f"\\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        img = mpimg.imread(path)
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Missing: {path}")
""")

# ========== CELL: Save to Kaggle Output ==========
code("""# Copy results to Kaggle output directory
import shutil
kaggle_out = "/kaggle/working/results"
os.makedirs(kaggle_out, exist_ok=True)

# Copy checkpoints
for f in glob.glob("checkpoints/*.pth"):
    shutil.copy2(f, kaggle_out)
    print(f"Saved checkpoint: {os.path.basename(f)}")

# Copy output images
for f in glob.glob("final_output/*"):
    shutil.copy2(f, kaggle_out)
    print(f"Saved output: {os.path.basename(f)}")

# Copy logs
for f in glob.glob("logs/*"):
    shutil.copy2(f, kaggle_out)
    print(f"Saved log: {os.path.basename(f)}")

print(f"\\nAll results saved to {kaggle_out}")
print("Download from Kaggle Output tab!")
""")

# ========== Save notebook ==========
output_path = os.path.join(BASE, "Kaggle_PINN_Training.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Kaggle notebook saved to: {output_path}")
print(f"Total cells: {len(nb['cells'])}")
