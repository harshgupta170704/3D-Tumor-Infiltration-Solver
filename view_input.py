import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_input_modalities(nifti_path, output_path):
    print(f"Loading input NIfTI: {nifti_path}")
    img = nib.load(nifti_path)
    data = img.get_fdata() # [H, W, D, 4] for MSD
    
    # MSD format is often [H, W, D, Channel]
    # Let's ensure we handle the shape correctly
    if len(data.shape) == 4:
        # Check which dimension is the channel (usually the last one in nibabel/MSD)
        if data.shape[-1] == 4:
            H, W, D, C = data.shape
        else:
            # Fallback if channels are first
            C, H, W, D = data.shape
            data = data.transpose(1, 2, 3, 0)
    else:
        print(f"Unexpected shape: {data.shape}")
        return

    # Titles for MSD modalities
    modality_names = ["FLAIR", "T1", "T1ce", "T2"]
    
    # Pick a slice with some tumor visibility (often middle of the axial plane)
    mid_slice = D // 2
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i in range(4):
        slice_data = data[:, :, mid_slice, i]
        # Normalize for visualization
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) + 1e-8)
        
        axes[i].imshow(np.rot90(slice_data), cmap='gray')
        axes[i].set_title(modality_names[i], fontsize=15)
        axes[i].axis('off')
        
    plt.suptitle(f"Input MRI Modalities: {os.path.basename(nifti_path)} (Slice {mid_slice})", fontsize=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Input visualization saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    patient_file = "data/Task01_BrainTumour/imagesTr/BRATS_127.nii.gz"
    output_img = "input_modalities_preview.png"
    
    if os.path.exists(patient_file):
        visualize_input_modalities(patient_file, output_img)
    else:
        print(f"Error: File {patient_file} not found.")
