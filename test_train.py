import os
import torch
from config import get_config
from data.dataset import get_train_val_dataloaders
from models.hybrid_model import HybridTumorNet
from losses.combined_loss import HybridTumorLoss
import sys

def main():
    config = get_config()
    config.data.data_root = r"D:\BrainTumorData\Task01_BrainTumour"
    config.train.phase = "pretrain"
    config.train.batch_size = 2
    
    print("Loading data...")
    train_loader, val_loader = get_train_val_dataloaders(
        data_root=config.data.data_root,
        batch_size=1,
        spatial_size=config.data.spatial_size,
        num_workers=0,
        cache_rate=0.0,
    )
    
    model = HybridTumorNet(
        resnet_variant="resnet18",
        in_channels=4,
        num_seg_classes=4,
    )
    model.set_phase("pretrain")
    criterion = HybridTumorLoss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("Running a single batch...")
    for batch_idx, batch in enumerate(train_loader):
        images = batch["image"]
        labels = batch["label"]
        print(f"Image shape: {images.shape}, Label shape: {labels.shape}")
        
        batch_gpu = {"image": images, "label": labels}
        output = model(images)
        loss_dict = criterion(output, batch_gpu, phase="pretrain")
        
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        optimizer.step()
        print("Success! Loss:", total_loss.item())
        break

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()

