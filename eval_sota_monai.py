import os
import torch
import sys
import numpy as np
from monai.networks.nets import SegResNet
from monai import transforms as T
from monai.data import Dataset, DataLoader

def main():
    print('=== EVALUATING SOTA MONAI BRA-TS MODEL (SegResNet) ON 10 LUMIERE PATIENTS ===')
    device = torch.device('cpu')
    
    # 1. Initialize SegResNet
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2
    ).to(device)
    
    # 2. Load official pretrained weights
    ckpt_path = 'C:/Users/harsh/OneDrive/Desktop/Cancer-Research/pretrained_models/brats_mri_segmentation/models/model.pt'
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print('Official SOTA SegResNet weights loaded successfully.')
    
    # 3. Select 10 patients from LUMIERE aligned atlas
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
                    # Note channel order for MONAI BraTS: T1c, T1, T2, FLAIR
                    't1ce': ct1, 't1': t1, 't2': t2, 'flair': flair,
                    'label': mask
                })
                break

    print(f'Testing on {len(data_dicts)} patients...')
    
    # 4. Standard validation transforms with sliding-window or center-crop
    spatial_size = (128, 128, 128)
    eval_transforms = T.Compose([
        T.LoadImaged(keys=['t1ce', 't1', 't2', 'flair', 'label'], ensure_channel_first=True),
        T.ConcatItemsd(keys=['t1ce', 't1', 't2', 'flair'], name='image'),
        T.DeleteItemsd(keys=['t1ce', 't1', 't2', 'flair']),
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
            
            # Predict
            logits = model(inputs) # Shape: [1, 3, D, H, W]
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # MONAI SegResNet channel definition:
            # Channel 0: TC (Tumor Core) -> Ground Truth is (labels == 1) | (labels == 3)
            # Channel 1: WT (Whole Tumor) -> Ground Truth is (labels > 0)
            # Channel 2: ET (Enhancing Tumor) -> Ground Truth is (labels == 3)
            
            pred_tc = preds[:, 0]
            pred_wt = preds[:, 1]
            pred_et = preds[:, 2]
            
            gt_tc = ((labels == 1) | (labels == 3)).float()
            gt_wt = (labels > 0).float()
            gt_et = (labels == 3).float()
            
            # Dice WT
            inter_wt = 2.0 * (pred_wt * gt_wt).sum()
            total_wt = pred_wt.sum() + gt_wt.sum()
            dice_wt = (inter_wt / (total_wt + 1e-6)).item()
            
            # Dice TC
            inter_tc = 2.0 * (pred_tc * gt_tc).sum()
            total_tc = pred_tc.sum() + gt_tc.sum()
            dice_tc = (inter_tc / (total_tc + 1e-6)).item()
            
            # Dice ET
            inter_et = 2.0 * (pred_et * gt_et).sum()
            total_et = pred_et.sum() + gt_et.sum()
            dice_et = (inter_et / (total_et + 1e-6)).item()
            
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
    print('SOTA MONAI (SegResNet) BENCHMARK ACROSS 10 PATIENTS')
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
