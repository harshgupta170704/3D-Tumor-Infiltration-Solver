"""Build the self-contained Kaggle notebook for Phase 2."""
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

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

BASE = r"c:\Users\Lenovo\OneDrive\Desktop\Fine tuning monia resnet modal with pinn modal"

md("""# ⚛️ Phase 2: Physics-Informed Fine-Tuning (PINN)

**IMPORTANT: Before running this!**
1. Click **"Add Data"** in the top right of Kaggle.
2. Click **"Upload Data"** (the up arrow icon).
3. Upload your `pretrain_best.pth` file.
4. Once uploaded, copy the file path and paste it into the **Phase 2 Training Cell** below (replace `YOUR_UPLOADED_PATH_HERE`).
""")

code("""!pip install -q "monai[nibabel,tqdm,einops]" nibabel SimpleITK nilearn medpy tensorboard scipy scikit-learn
import torch, os
for d in ["models", "losses", "data", "utils", "checkpoints", "logs", "final_output"]:
    os.makedirs(d, exist_ok=True)
for d in ["models", "losses", "data", "utils"]:
    with open(os.path.join(d, "__init__.py"), "w") as f: f.write("")
""")

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
    escaped = content.replace("\\", "\\\\").replace('"""', '\\"\\"\\"').replace("'''", "\\'\\'\\'")
    code(f'''with open("{target_path}", "w") as f:\n    f.write("""{escaped}""")''')

code("""import re
with open("config.py", "r") as f: cfg = f.read()
cfg = re.sub(r'num_epochs_pretrain: int = \\d+', 'num_epochs_pretrain: int = 0', cfg)
cfg = re.sub(r'num_epochs_finetune: int = \\d+', 'num_epochs_finetune: int = 100', cfg)
cfg = re.sub(r'batch_size: int = \\d+', 'batch_size: int = 2', cfg)
cfg = re.sub(r'num_workers: int = \\d+', 'num_workers: int = 2', cfg)
cfg = re.sub(r'device: str = "\\w+"', 'device: str = "cuda"', cfg)
with open("config.py", "w") as f: f.write(cfg)
""")

md("## 📥 Download Dataset")
code("""from monai.apps import download_and_extract
import glob
DATA_DIR = "./data"
MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
task_dir = os.path.join(DATA_DIR, "Task01_BrainTumour")
if not os.path.exists(task_dir) or len(glob.glob(os.path.join(task_dir, "imagesTr", "*.nii.gz"))) == 0:
    download_and_extract(url=MSD_URL, filepath=os.path.join(DATA_DIR, "Task01_BrainTumour.tar"), output_dir=DATA_DIR)
""")

md("## ⚛️ Run Phase 2 (Physics Finetuning)")
code("""# IMPORTANT: Replace the path below with your uploaded checkpoint path!
CHECKPOINT_PATH = "/kaggle/input/YOUR_UPLOADED_PATH_HERE/pretrain_best.pth"

!python train.py --data_root data/Task01_BrainTumour --phase finetune --device cuda --batch_size 1 --resume "{CHECKPOINT_PATH}"
""")

md("## 🔮 Run Inference (Produce Final Images)")
code("""import glob
sample = glob.glob("data/Task01_BrainTumour/imagesTr/BRATS_*.nii.gz")[0]
!python predict.py --input_dir "{sample}" --checkpoint checkpoints/finetune_best.pth --simulate_days 30 --output_dir final_output --device cuda
""")

code("""import shutil, glob
kaggle_out = "/kaggle/working/results"
os.makedirs(kaggle_out, exist_ok=True)
for f in glob.glob("checkpoints/*.pth") + glob.glob("final_output/*") + glob.glob("logs/*"):
    shutil.copy2(f, kaggle_out)
print("All saved to", kaggle_out)
""")

output_path = os.path.join(BASE, "Kaggle_Phase2.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
