import os
import glob
import torch
import sys
import numpy as np

sys.path.append('C:/Users/harsh/OneDrive/Desktop/Cancer-Research/3D-Tumor-Infiltration-Solver')

from models.hybrid_model import HybridTumorNet
from monai import transforms as T
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric

def main():
    print('=== EVALUATING PRETRAINED MODEL ON 10 LUMIERE PATIENTS (ALIGNED ATLAS) ===')
    device = torch.device('cpu')
    
    # 1. Load Model
    model = HybridTumorNet(
        resnet_variant='resnet10', 
        decoder_channels=(128, 64, 32, 16, 8), 
        in_channels=4, 
        num_seg_classes=4
    ).to(device)
    
    checkpoint_path = 'C:/Users/harsh/OneDrive/Desktop/Cancer-Research/3D-Tumor-Infiltration-Solver/checkpoints/pretrain_best.pth'
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Select 10 patients
    lumiere_path = 'C:/Users/harsh/OneDrive/Desktop/Cancer-Research/datasets/LUMIERE_Subset/Imaging'
    patients = sorted([p for p in os.listdir(lumiere_path) if p.startswith('Patient')])[:10]
    
    data_dicts = []
    for p in patients:
        p_dir = os.path.join(lumiere_path, p)
        for tp in sorted(os.listdir(p_dir)):
            tp_dir = os.path.join(p_dir, tp)
            atlas_dir = os.path.join(tp_dir, 'DeepBraTumIA-segmentation', 'atlas')
            t1 = os.path.join(atlas_dir, 'skull_strip', 't1_skull_strip.nii.gz')
            ct1 = os.path.join(atlas_dir, 'skull_strip', 'ct1_skull_strip.nii.gz')
            t2 = os.path.join(atlas_dir, 'skull_strip', 't2_skull_strip.nii.gz')
            flair = os.path.join(atlas_dir, 'skull_strip', 'flair_skull_strip.nii.gz')
            mask = os.path.join(atlas_dir, 'segmentation', 'seg_mask.nii.gz')
            
            if os.path.exists(t1) and os.path.exists(ct1) and os.path.exists(t2) and os.path.exists(flair) and os.path.exists(mask):
                data_dicts.append({
                    'patient': f'{p} ({tp})',
                    't1': t1, 't1ce': ct1, 't2': t2, 'flair': flair,
                    'label': mask
                })
                break

    print(f'Found {len(data_dicts)} valid patient scans.')
    
    # 3. Exact matching transforms
    spatial_size = (128, 128, 128)
    eval_transforms = T.Compose([
        T.LoadImaged(keys=['t1', 't1ce', 't2', 'flair', 'label'], ensure_channel_first=True),
        T.ConcatItemsd(keys=['t1', 't1ce', 't2', 'flair'], name='image'),
        T.DeleteItemsd(keys=['t1', 't1ce', 't2', 'flair']),
        T.Orientationd(keys=['image', 'label'], axcodes='RAS'),
        T.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
        T.CropForegroundd(keys=['image', 'label'], source_key='image', margin=10),
        T.SpatialPadd(keys=['image', 'label'], spatial_size=spatial_size),
        T.CenterSpatialCropd(keys=['image', 'label'], roi_size=spatial_size),
        T.EnsureTyped(keys=['image', 'label'], dtype='float32'),
    ])

    eval_ds = Dataset(data=data_dicts, transform=eval_transforms)
    eval_loader = DataLoader(eval_ds, batch_size=1)
    
    results = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(eval_loader):
            pt_name = data_dicts[i]['patient']
            inputs = batch_data['image'].to(device)
            labels = batch_data['label'].to(device).long()
            
            outputs = model(inputs)
            logits = outputs['segmentation'] if 'segmentation' in outputs else outputs[0]
            preds = torch.argmax(logits, dim=1, keepdim=True)
            
            # Dice for WT (classes 1, 2, 3)
            pred_wt = (preds > 0).float()
            label_wt = (labels > 0).float()
            intersection_wt = 2.0 * (pred_wt * label_wt).sum()
            total_wt = pred_wt.sum() + label_wt.sum()
            dice_wt = (intersection_wt / (total_wt + 1e-6)).item()
            
            # Dice for TC (classes 1, 3)
            pred_tc = ((preds == 1) | (preds == 3)).float()
            label_tc = ((labels == 1) | (labels == 3)).float()
            intersection_tc = 2.0 * (pred_tc * label_tc).sum()
            total_tc = pred_tc.sum() + label_tc.sum()
            dice_tc = (intersection_tc / (total_tc + 1e-6)).item()
            
            # Dice for ET (class 3)
            pred_et = (preds == 3).float()
            label_et = (labels == 3).float()
            intersection_et = 2.0 * (pred_et * label_et).sum()
            total_et = pred_et.sum() + label_et.sum()
            dice_et = (intersection_et / (total_et + 1e-6)).item()
            
            mean_d = (dice_wt + dice_tc + dice_et) / 3.0
            results.append({
                'patient': pt_name,
                'wt': dice_wt,
                'tc': dice_tc,
                'et': dice_et,
                'mean': mean_d
            })
            print(f'[{i+1}/10] {pt_name:25s} | WT: {dice_wt*100:5.1f}% | TC: {dice_tc*100:5.1f}% | ET: {dice_et*100:5.1f}% | Mean: {mean_d*100:5.1f}%')

    print('\n' + '='*70)
    print('FINAL SUMMARY ACROSS 10 PATIENTS')
    print('='*70)
    avg_wt = np.mean([r['wt'] for r in results])
    avg_tc = np.mean([r['tc'] for r in results])
    avg_et = np.mean([r['et'] for r in results])
    avg_mean = np.mean([r['mean'] for r in results])
    
    med_wt = np.median([r['wt'] for r in results])
    med_tc = np.median([r['tc'] for r in results])
    med_et = np.median([r['et'] for r in results])
    med_mean = np.median([r['mean'] for r in results])
    
    print(f'Average Whole Tumor (WT):    {avg_wt*100:5.1f}% (Median: {med_wt*100:5.1f}%)')
    print(f'Average Tumor Core (TC):     {avg_tc*100:5.1f}% (Median: {med_tc*100:5.1f}%)')
    print(f'Average Enhancing Tumor (ET): {avg_et*100:5.1f}% (Median: {med_et*100:5.1f}%)')
    print(f'Overall Average Dice:        {avg_mean*100:5.1f}% (Median: {med_mean*100:5.1f}%)')
    print('='*70)

if __name__ == '__main__':
    main()
