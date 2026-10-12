"""
BraTS Dataset loaders.

Supports:
    - Standard BraTS 2021 layout (single time-point per patient)
    - Longitudinal pairs (for patients with multiple scans)
    - Automatic train/val split
    - Missing modality handling
"""

import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai.data import CacheDataset, SmartCacheDataset, list_data_collate

from .preprocessing import (
    get_train_transforms,
    get_val_transforms,
    get_longitudinal_transforms,
)


def discover_brats_data(data_root: str, modalities=("t1", "t1ce", "t2", "flair")):
    """
    Auto-discover BraTS dataset structure.

    Expected layout:
        data_root/
        ├── BraTS2021_00000/
        │   ├── BraTS2021_00000_t1.nii.gz
        │   ├── BraTS2021_00000_t1ce.nii.gz
        │   ├── BraTS2021_00000_t2.nii.gz
        │   ├── BraTS2021_00000_flair.nii.gz
        │   └── BraTS2021_00000_seg.nii.gz
        ├── BraTS2021_00001/
        ...

    Returns:
        List of dicts with keys "image" (list of 4 paths) and "label" (seg path).
    """
    data_dicts = []
    patient_dirs = sorted(glob.glob(os.path.join(data_root, "BraTS*")))

    if not patient_dirs:
        # Try alternative naming
        patient_dirs = sorted(glob.glob(os.path.join(data_root, "*")))
        patient_dirs = [d for d in patient_dirs if os.path.isdir(d)]

    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        
        # Find modality files
        mod_files = []
        found_all = True
        for mod in modalities:
            pattern = os.path.join(patient_dir, f"*_{mod}.nii.gz")
            matches = glob.glob(pattern)
            if not matches:
                # Try alternative naming
                pattern = os.path.join(patient_dir, f"*{mod}*.nii.gz")
                matches = glob.glob(pattern)
            
            if matches:
                mod_files.append(matches[0])
            else:
                found_all = False
                print(f"Warning: Missing {mod} for {patient_id}")
                break

        # Find segmentation
        seg_pattern = os.path.join(patient_dir, "*_seg.nii.gz")
        seg_matches = glob.glob(seg_pattern)

        if found_all and seg_matches:
            data_dicts.append({
                "image": mod_files,  # List of 4 modality paths
                "label": seg_matches[0],
                "patient_id": patient_id,
            })

    print(f"Found {len(data_dicts)} patients with all modalities")
    return data_dicts


def discover_msd_data(data_root: str):
    """
    Auto-discover Medical Segmentation Decathlon (MSD) dataset structure.

    Expected layout:
        data_root/
        ├── imagesTr/
        │   ├── BRATS_001.nii.gz
        │   ...
        ├── labelsTr/
        │   ├── BRATS_001.nii.gz
        │   ...

    Returns:
        List of dicts with keys "image" and "label" pointing to the paired files.
    """
    data_dicts = []
    images_dir = os.path.join(data_root, "imagesTr")
    labels_dir = os.path.join(data_root, "labelsTr")

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    
    for img_path in image_files:
        basename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, basename)
        
        if os.path.exists(label_path):
            data_dicts.append({
                "image": img_path,  # single 4D nifti
                "label": label_path,
                "patient_id": basename.replace(".nii.gz", ""),
            })

    print(f"Found {len(data_dicts)} MSD patients with paired labels")
    return data_dicts


class BraTSDataset:
    """
    BraTS dataset manager. Creates MONAI CacheDatasets with
    appropriate transforms for training and validation.
    """

    def __init__(self, data_root: str, spatial_size=(128, 128, 128),
                 val_split: float = 0.2, cache_rate: float = 0.5,
                 seed: int = 42):
        """
        Args:
            data_root: Path to BraTS data directory.
            spatial_size: Target volume dimensions.
            val_split: Fraction of data for validation.
            cache_rate: Fraction of data to cache in memory (0.0 to 1.0).
            seed: Random seed for reproducible splits.
        """
        self.data_root = data_root
        self.spatial_size = spatial_size
        self.val_split = val_split
        self.cache_rate = cache_rate

        # Discover data
        if os.path.exists(os.path.join(data_root, "imagesTr")) and os.path.exists(os.path.join(data_root, "labelsTr")):
            self.all_data = discover_msd_data(data_root)
        else:
            self.all_data = discover_brats_data(data_root)

        if len(self.all_data) == 0:
            raise RuntimeError(
                f"No BraTS data found in {data_root}. "
                "Check directory structure. Expected: data_root/BraTS2021_XXXXX/"
            )

        # Split
        np.random.seed(seed)
        indices = np.random.permutation(len(self.all_data))
        val_size = int(len(indices) * val_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        self.train_data = [self.all_data[i] for i in train_indices]
        self.val_data = [self.all_data[i] for i in val_indices]

        print(f"Train: {len(self.train_data)}, Val: {len(self.val_data)}")

    def get_train_dataset(self):
        """Create cached training dataset."""
        return CacheDataset(
            data=self.train_data,
            transform=get_train_transforms(self.spatial_size),
            cache_rate=self.cache_rate,
            num_workers=4,
        )

    def get_val_dataset(self):
        """Create cached validation dataset."""
        return CacheDataset(
            data=self.val_data,
            transform=get_val_transforms(self.spatial_size),
            cache_rate=1.0,  # Cache all validation data
            num_workers=4,
        )


class LongitudinalBraTSDataset(Dataset):
    """
    Dataset for longitudinal (paired) brain MRI scans.
    
    Each sample contains two time-point scans from the same patient,
    used for computing temporal derivatives ∂u/∂t for the physics loss.
    
    Expected layout:
        data_root/
        ├── patient_001/
        │   ├── timepoint_1/
        │   │   ├── t1.nii.gz, t1ce.nii.gz, t2.nii.gz, flair.nii.gz, seg.nii.gz
        │   └── timepoint_2/
        │       ├── t1.nii.gz, t1ce.nii.gz, t2.nii.gz, flair.nii.gz, seg.nii.gz
        ...
    
    Or provide a JSON manifest:
    [
        {
            "patient_id": "001",
            "timepoint_1": {"image": [...], "label": "...", "days": 0},
            "timepoint_2": {"image": [...], "label": "...", "days": 90}
        }
    ]
    """

    def __init__(self, manifest_path: str = None, data_root: str = None,
                 spatial_size=(128, 128, 128), transform=None):
        self.spatial_size = spatial_size
        self.pairs = []

        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.pairs = json.load(f)
        elif data_root:
            self.pairs = self._discover_longitudinal(data_root)

        self.transform = transform or get_longitudinal_transforms(spatial_size)
        print(f"Found {len(self.pairs)} longitudinal pairs")

    def _discover_longitudinal(self, data_root):
        """Auto-discover longitudinal data structure."""
        pairs = []
        patient_dirs = sorted(glob.glob(os.path.join(data_root, "*")))
        
        for pdir in patient_dirs:
            if not os.path.isdir(pdir):
                continue
            
            timepoints = sorted(glob.glob(os.path.join(pdir, "timepoint_*")))
            if len(timepoints) < 2:
                continue
            
            # Create pairs from consecutive timepoints
            for i in range(len(timepoints) - 1):
                tp1_mods = sorted(glob.glob(os.path.join(timepoints[i], "*.nii.gz")))
                tp2_mods = sorted(glob.glob(os.path.join(timepoints[i + 1], "*.nii.gz")))
                
                if len(tp1_mods) >= 5 and len(tp2_mods) >= 5:  # 4 modalities + seg
                    pairs.append({
                        "patient_id": os.path.basename(pdir),
                        "image_t1": [f for f in tp1_mods if "seg" not in f][:4],
                        "image_t2": [f for f in tp2_mods if "seg" not in f][:4],
                        "label_t1": [f for f in tp1_mods if "seg" in f][0],
                        "label_t2": [f for f in tp2_mods if "seg" in f][0],
                        "delta_t": 90.0,  # Default days between scans
                    })
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        sample = {
            "image_t1": pair["image_t1"],
            "image_t2": pair["image_t2"],
            "label_t1": pair["label_t1"],
            "label_t2": pair["label_t2"],
        }
        if self.transform:
            sample = self.transform(sample)
        sample["delta_t"] = torch.tensor(pair.get("delta_t", 90.0), dtype=torch.float32)
        sample["patient_id"] = pair["patient_id"]
        return sample


def get_train_val_dataloaders(data_root: str, batch_size: int = 2,
                               spatial_size=(128, 128, 128),
                               num_workers: int = 4,
                               cache_rate: float = 0.5,
                               val_split: float = 0.2,
                               pin_memory: bool = True):
    """
    Create training and validation DataLoaders.

    Returns:
        (train_loader, val_loader)
    """
    dataset = BraTSDataset(
        data_root=data_root,
        spatial_size=spatial_size,
        val_split=val_split,
        cache_rate=cache_rate,
    )

    train_loader = DataLoader(
        dataset.get_train_dataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=list_data_collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset.get_val_dataset(),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=list_data_collate,
    )

    return train_loader, val_loader
