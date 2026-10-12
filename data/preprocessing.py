"""
MRI preprocessing and augmentation transforms using MONAI.

Pipeline:
    1. Load NIfTI files (4 modalities + segmentation mask)
    2. Reorient to RAS+
    3. Resample to uniform spacing
    4. Crop to brain region / fixed ROI
    5. Intensity normalization (z-score per modality within brain mask)
    6. Random augmentations (train only)
"""

from monai import transforms as T


def get_train_transforms(spatial_size=(128, 128, 128), 
                          intensity_shift=0.1,
                          intensity_scale=0.1):
    """
    Get training transforms with data augmentation.

    Returns:
        MONAI Compose transform pipeline.
    """
    return T.Compose([
        # --- Load ---
        T.LoadImaged(keys=["image", "label"], ensure_channel_first=True),

        # --- Orientation & Spacing ---
        T.Orientationd(keys=["image", "label"], axcodes="RAS"),
        T.Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),

        # --- Intensity normalization (z-score per channel) ---
        T.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # --- Crop / Resize ---
        T.CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        T.SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        T.RandSpatialCropd(
            keys=["image", "label"],
            roi_size=spatial_size,
            random_center=True,
            random_size=False,
        ),

        # --- Augmentation ---
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        T.RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        T.RandScaleIntensityd(keys=["image"], factors=intensity_scale, prob=0.5),
        T.RandShiftIntensityd(keys=["image"], offsets=intensity_shift, prob=0.5),
        T.RandGaussianNoised(keys=["image"], std=0.05, prob=0.3),
        T.RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.15),
            sigma_y=(0.5, 1.15),
            sigma_z=(0.5, 1.15),
            prob=0.2,
        ),

        # --- Convert labels ---
        # BraTS labels: 0, 1, 2, 4 → remap 4→3
        T.MapLabelValued(keys=["label"], orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3]),

        # --- To Tensor ---
        T.EnsureTyped(keys=["image", "label"], dtype="float32"),
    ])


def get_val_transforms(spatial_size=(128, 128, 128)):
    """
    Get validation/test transforms (no augmentation).

    Returns:
        MONAI Compose transform pipeline.
    """
    return T.Compose([
        # --- Load ---
        T.LoadImaged(keys=["image", "label"], ensure_channel_first=True),

        # --- Orientation & Spacing ---
        T.Orientationd(keys=["image", "label"], axcodes="RAS"),
        T.Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),

        # --- Intensity normalization ---
        T.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # --- Crop / Resize (deterministic) ---
        T.CropForegroundd(keys=["image", "label"], source_key="image", margin=10),
        T.SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        T.CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size),

        # --- Convert labels ---
        T.MapLabelValued(keys=["label"], orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3]),

        # --- To Tensor ---
        T.EnsureTyped(keys=["image", "label"], dtype="float32"),
    ])


def get_inference_transforms(spatial_size=(128, 128, 128)):
    """
    Transforms for inference (no label loading).

    Returns:
        MONAI Compose transform pipeline.
    """
    return T.Compose([
        T.LoadImaged(keys=["image"], ensure_channel_first=True),
        T.Orientationd(keys=["image"], axcodes="RAS"),
        T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        T.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        T.CropForegroundd(keys=["image"], source_key="image", margin=10),
        T.SpatialPadd(keys=["image"], spatial_size=spatial_size),
        T.CenterSpatialCropd(keys=["image"], roi_size=spatial_size),
        T.EnsureTyped(keys=["image"], dtype="float32"),
    ])


def get_longitudinal_transforms(spatial_size=(128, 128, 128)):
    """
    Transforms for longitudinal (paired) MRI data.
    Loads two time-point images with the same spatial transforms.
    
    Keys: "image_t1", "image_t2", "label_t1", "label_t2"
    """
    return T.Compose([
        # --- Load all ---
        T.LoadImaged(
            keys=["image_t1", "image_t2", "label_t1", "label_t2"],
            ensure_channel_first=True,
        ),
        T.Orientationd(
            keys=["image_t1", "image_t2", "label_t1", "label_t2"],
            axcodes="RAS",
        ),
        T.Spacingd(
            keys=["image_t1", "image_t2", "label_t1", "label_t2"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "bilinear", "nearest", "nearest"),
        ),

        # --- Normalize ---
        T.NormalizeIntensityd(keys=["image_t1", "image_t2"], nonzero=True, channel_wise=True),

        # --- Crop & Pad (use t1 as reference) ---
        T.CropForegroundd(
            keys=["image_t1", "image_t2", "label_t1", "label_t2"],
            source_key="image_t1",
            margin=10,
        ),
        T.SpatialPadd(
            keys=["image_t1", "image_t2", "label_t1", "label_t2"],
            spatial_size=spatial_size,
        ),
        T.CenterSpatialCropd(
            keys=["image_t1", "image_t2", "label_t1", "label_t2"],
            roi_size=spatial_size,
        ),

        # --- Labels ---
        T.MapLabelValued(
            keys=["label_t1", "label_t2"],
            orig_labels=[0, 1, 2, 4],
            target_labels=[0, 1, 2, 3],
        ),

        T.EnsureTyped(
            keys=["image_t1", "image_t2", "label_t1", "label_t2"],
            dtype="float32",
        ),
    ])
