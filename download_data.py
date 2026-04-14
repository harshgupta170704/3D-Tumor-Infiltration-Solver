"""
Download brain tumor MRI data for training.

Uses MONAI's Medical Segmentation Decathlon (MSD) Task01_BrainTumour,
which contains 484 training cases with:
    - 4 MRI modalities (T1, T1ce, T2, FLAIR)
    - Expert segmentation masks
    - Same format as BraTS

No registration or API key needed — downloads directly.
"""

import os
import sys
import shutil
import tarfile
import zipfile
import json
import glob
import numpy as np

# Check for required packages
try:
    from monai.apps import download_and_extract, DecathlonDataset
    from monai.config import print_config
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MSD_TASK = "Task01_BrainTumour"

# MSD direct download URL (Google Drive hosted)
MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"


def download_msd_brain_tumor(data_dir=None):
    """
    Download Medical Segmentation Decathlon — Brain Tumour dataset.

    This dataset contains 484 training + 266 test multimodal brain MRI scans
    with expert segmentation annotations. Each scan has 4 channels:
        Channel 0: FLAIR
        Channel 1: T1w
        Channel 2: T1gd (T1ce)
        Channel 3: T2w

    Labels (NIfTI voxel values — already remapped from BraTS {0,1,2,4}):
        0: Background
        1: NCR/NET (Necrotic core & non-enhancing tumor)
        2: ED (Peritumoral edema)
        3: ET (Enhancing tumor — remapped from BraTS label 4)

    Total download size: ~4.6 GB
    """
    if data_dir is None:
        data_dir = DATA_DIR

    os.makedirs(data_dir, exist_ok=True)
    task_dir = os.path.join(data_dir, MSD_TASK)

    if os.path.exists(task_dir) and len(glob.glob(os.path.join(task_dir, "imagesTr", "*.nii.gz"))) > 0:
        n_files = len(glob.glob(os.path.join(task_dir, "imagesTr", "*.nii.gz")))
        print(f"Dataset already exists at {task_dir} with {n_files} training files.")
        print("Skipping download. Delete the folder to re-download.")
        return task_dir

    print("=" * 60)
    print("DOWNLOADING BRAIN TUMOR MRI DATASET")
    print("=" * 60)
    print(f"Source: Medical Segmentation Decathlon - {MSD_TASK}")
    print(f"URL: {MSD_URL}")
    print(f"Destination: {task_dir}")
    print(f"Size: ~4.6 GB (484 training cases, 4 modalities each)")
    print("=" * 60)
    print()

    if MONAI_AVAILABLE:
        print("Using MONAI download_and_extract...")
        try:
            download_and_extract(
                url=MSD_URL,
                filepath=os.path.join(data_dir, "Task01_BrainTumour.tar"),
                output_dir=data_dir,
            )
            print("Download and extraction complete!")
        except Exception as e:
            print(f"MONAI download failed: {e}")
            print("Trying fallback method...")
            _fallback_download(data_dir)
    else:
        print("MONAI not available, using fallback download...")
        _fallback_download(data_dir)

    # Verify
    if os.path.exists(task_dir):
        _verify_dataset(task_dir)
    else:
        print(f"ERROR: Expected directory {task_dir} not found after download.")
        sys.exit(1)

    return task_dir


def _fallback_download(data_dir):
    """Download using urllib if MONAI is not available."""
    import urllib.request

    tar_path = os.path.join(data_dir, "Task01_BrainTumour.tar")

    if not os.path.exists(tar_path):
        print(f"Downloading to {tar_path}...")
        print("This may take 15-30 minutes depending on your connection.")

        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                bar_len = 40
                filled = int(bar_len * pct / 100)
                bar = "=" * filled + "-" * (bar_len - filled)
                sys.stdout.write(f"\r  [{bar}] {pct:.1f}% ({mb_down:.0f}/{mb_total:.0f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(MSD_URL, tar_path, reporthook=progress_hook)
        print("\nDownload complete!")

    print("Extracting...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete!")

    # Clean up tar file to save space
    os.remove(tar_path)
    print(f"Removed archive to save disk space.")


def _verify_dataset(task_dir):
    """Verify the downloaded dataset is complete."""
    print("\n--- Dataset Verification ---")

    images_dir = os.path.join(task_dir, "imagesTr")
    labels_dir = os.path.join(task_dir, "labelsTr")

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

    print(f"  Training images: {len(image_files)}")
    print(f"  Training labels: {len(label_files)}")

    # Check dataset.json
    json_path = os.path.join(task_dir, "dataset.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        print(f"  Task: {meta.get('name', 'unknown')}")
        print(f"  Description: {meta.get('description', 'N/A')}")
        print(f"  Modalities: {meta.get('modality', {})}")
        print(f"  Labels: {meta.get('labels', {})}")
        print(f"  Training samples: {meta.get('numTraining', '?')}")
        print(f"  Test samples: {meta.get('numTest', '?')}")

    if len(image_files) > 0:
        # Quick check first file
        try:
            import nibabel as nib
            sample = nib.load(image_files[0])
            print(f"\n  Sample file: {os.path.basename(image_files[0])}")
            print(f"  Shape: {sample.shape}")
            print(f"  Voxel size: {sample.header.get_zooms()}")
            print(f"  Data type: {sample.get_data_dtype()}")

            if len(sample.shape) == 4 and sample.shape[3] == 4:
                print("  Channels: [FLAIR, T1, T1ce, T2] -- 4 modalities confirmed!")
        except Exception as e:
            print(f"  Could not inspect sample file: {e}")

    print("\n  Dataset is ready for training!")
    return True


def create_data_list(task_dir, val_split=0.2, seed=42):
    """
    Create train/val split and save as JSON for easy loading.
    
    The MSD format stores all 4 modalities in a single 4D NIfTI file,
    so we need to adapt our data loading pipeline.
    """
    images_dir = os.path.join(task_dir, "imagesTr")
    labels_dir = os.path.join(task_dir, "labelsTr")

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

    # Build data list
    data_list = []
    for img_path in image_files:
        basename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, basename)
        if os.path.exists(label_path):
            data_list.append({
                "image": img_path,
                "label": label_path,
            })

    # Split
    np.random.seed(seed)
    indices = np.random.permutation(len(data_list))
    n_val = int(len(indices) * val_split)

    val_list = [data_list[i] for i in indices[:n_val]]
    train_list = [data_list[i] for i in indices[n_val:]]

    # Save
    splits = {
        "train": train_list,
        "val": val_list,
    }
    split_path = os.path.join(task_dir, "splits.json")
    with open(split_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Created data split: {len(train_list)} train, {len(val_list)} val")
    print(f"Saved to: {split_path}")

    return train_list, val_list


if __name__ == "__main__":
    print()
    task_dir = download_msd_brain_tumor()
    print()
    create_data_list(task_dir)
    print()
    print("Next step: Run training with:")
    print(f"  python train.py --data_root {task_dir} --phase pretrain")
