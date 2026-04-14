import numpy as np
import nibabel as nib
import os
import torch

def generate_synthetic_scan(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    size = 128
    
    # 4 modalities (FLAIR, T1, T1ce, T2)
    center = size // 2
    y, x, z = np.ogrid[-center:size-center, -center:size-center, -center:size-center]
    brain_mask = x**2 + y**2 + z**2 <= (size // 2.5)**2
    
    # Tumor
    tumor_mask = (x - 10)**2 + (y + 5)**2 + (z - 5)**2 <= 15**2
    necrotic = (x - 10)**2 + (y + 5)**2 + (z - 5)**2 <= 6**2
    enhancing = tumor_mask & ~necrotic
    edema = (x - 10)**2 + (y + 5)**2 + (z - 5)**2 <= 22**2
    edema = edema & ~tumor_mask
    
    affine = np.eye(4)
    
    modalities = {
        "flair": 0,
        "t1": 1,
        "t1ce": 2,
        "t2": 3
    }
    
    for mod_name, mod_idx in modalities.items():
        img_data = np.zeros((size, size, size), dtype=np.float32)
        img_data[brain_mask] = 0.2 + np.random.randn(int(brain_mask.sum())) * 0.05
        
        if mod_name == "flair":
            img_data[edema] += 0.5
            img_data[enhancing] += 0.2
        elif mod_name == "t1ce":
            img_data[enhancing] += 0.8
            img_data[necrotic] -= 0.1
        elif mod_name == "t2":
            img_data[tumor_mask | edema] += 0.4
            
        nib.save(nib.Nifti1Image(img_data, affine), os.path.join(out_dir, f"patient001_{mod_name}.nii.gz"))

    # Save label
    label_data = np.zeros((size, size, size), dtype=np.uint8)
    label_data[edema] = 2
    label_data[necrotic] = 1
    label_data[enhancing] = 4 # BraTS format (4 = ET)
    nib.save(nib.Nifti1Image(label_data, affine), os.path.join(out_dir, "patient001_seg.nii.gz"))
    
    print(f"Generated synthetic scan at {out_dir}")

if __name__ == "__main__":
    generate_synthetic_scan("./demo_data")
