"""
Training script for Hybrid MONAI ResNet + Physics-Informed Tumor Model.

Two-phase training:
    Phase 1 (Pretrain): Train segmentation head on BraTS data
    Phase 2 (Finetune): Fine-tune with physics-informed losses
                        + synthetic longitudinal du/dt from FisherKPPSolver

Usage:
    python train.py --data_root ./data/Task01_BrainTumour --phase pretrain
    python train.py --data_root ./data/Task01_BrainTumour --phase finetune --resume checkpoints/pretrain_best.pth
    python train.py --data_root ./data/Task01_BrainTumour --phase joint
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config, get_config
from models.hybrid_model import HybridTumorNet
from losses.combined_loss import HybridTumorLoss
from losses.data_loss import seg_to_density
from data.dataset import get_train_val_dataloaders
from utils.metrics import compute_dice, compute_physics_residual
from utils.spatial_ops import SpatialGradients3D
from utils.ema import EMAModel


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_model(config):
    """Create the HybridTumorNet model."""
    mc = config.model
    model = HybridTumorNet(
        resnet_variant=mc.backbone,
        in_channels=config.data.num_channels,
        num_seg_classes=mc.num_seg_classes,
        pretrained_path=mc.pretrained_weights_path,
        freeze_encoder_layers=0,
        decoder_channels=tuple(mc.decoder_channels),
        dropout=mc.dropout_rate,
        use_seg_head=mc.use_seg_head,
        predict_physics_params=mc.predict_diffusion,
        diffusion_range=mc.diffusion_range,
        proliferation_range=mc.proliferation_range,
    )
    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,} total, "
          f"{params['trainable']:,} trainable")
    return model


def build_optimizer(model, config):
    """Create optimizer with differential learning rates."""
    tc = config.train
    param_groups = model.get_parameter_groups(
        lr_backbone=tc.lr_backbone,
        lr_head=tc.learning_rate,
    )
    if tc.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=tc.weight_decay)
    elif tc.optimizer == "adam":
        return torch.optim.Adam(param_groups, weight_decay=tc.weight_decay)
    elif tc.optimizer == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9,
                                weight_decay=tc.weight_decay)
    raise ValueError(f"Unknown optimizer: {tc.optimizer}")


def build_scheduler(optimizer, config, num_epochs):
    """Create learning rate scheduler."""
    tc = config.train
    if tc.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=tc.min_lr
        )
    elif tc.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
    elif tc.scheduler == "warmup_cosine":
        def lr_lambda(epoch):
            if epoch < tc.warmup_epochs:
                return (epoch + 1) / tc.warmup_epochs
            progress = (epoch - tc.warmup_epochs) / (num_epochs - tc.warmup_epochs)
            return max(tc.min_lr / tc.learning_rate,
                       0.5 * (1 + np.cos(np.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif tc.scheduler == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=tc.min_lr
        )
    return None


def generate_synthetic_du_dt_for_batch(batch, config, device):
    """
    Generate synthetic temporal derivatives for physics-informed training.

    Uses FisherKPPSolver to numerically integrate the Fisher-KPP PDE
    from the segmentation-derived density u(t₁), producing:
        - du/dt = (u_simulated(t₂) - u(t₁)) / Δt

    This provides the temporal derivative needed for the PDE residual loss
    when real longitudinal data is not available (cross-sectional datasets).

    Args:
        batch: Training batch dict with "label" key.
        config: Config object.
        device: torch device.

    Returns:
        Updated batch dict with "du_dt" key added.
    """
    from data.synthetic_longitudinal import FisherKPPSolver

    label = batch.get("label")
    if label is None:
        return batch

    pc = config.physics

    # Convert segmentation to density field (initial condition)
    u_t1 = seg_to_density(label, sigma=1.0, smooth=True).to(device)

    # Create brain mask from input
    brain_mask = (batch["image"].abs().sum(dim=1, keepdim=True) > 0).float().to(device)

    # Initialize solver with default biophysical parameters
    solver = FisherKPPSolver(
        default_D=pc.default_diffusion,
        default_rho=pc.default_proliferation,
        max_dt_step=1.0,
        device=str(device),
    )

    # Simulate tumor growth forward by synthetic_delta_t days
    u_t2, du_dt = solver.simulate(
        u0=u_t1,
        delta_t_days=pc.synthetic_delta_t,
        brain_mask=brain_mask,
        add_noise=True,
        noise_std=0.005,
    )

    batch["du_dt"] = du_dt
    batch["density_t2"] = u_t2
    batch["delta_t"] = torch.tensor(pc.synthetic_delta_t)

    return batch


def train_one_epoch(model, loader, criterion, optimizer, scaler,
                     device, phase, epoch, config, writer=None, ema_model=None):
    """Train for one epoch."""
    model.train()
    epoch_losses = {}
    num_batches = 0
    accumulation_steps = getattr(config.train, 'accumulation_steps', 1)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        batch_gpu = {"image": images, "label": labels}

        # For finetune/joint phases: generate synthetic temporal derivatives
        # This wires the FisherKPPSolver into the training loop
        if phase in ("finetune", "joint"):
            batch_gpu = generate_synthetic_du_dt_for_batch(batch_gpu, config, device)

        optimizer.zero_grad()

        # Forward pass
        with autocast(device_type=device.type, enabled=scaler is not None):
            output = model(images)
            loss_dict = criterion(output, batch_gpu, phase=phase)
            total_loss = loss_dict["total_loss"] / accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema_model is not None:
                    ema_model.update(model)
        else:
            total_loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if ema_model is not None:
                    ema_model.update(model)

        # Track losses
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            # If we divided total_loss by accumulation_steps above, we should 
            # log the actual non-divided value for clarity, but they are from loss_dict so it's fine.
            epoch_losses[k] = epoch_losses.get(k, 0) + v
        num_batches += 1

        # Progress bar
        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    # Tensorboard
    if writer:
        for k, v in epoch_losses.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

    return epoch_losses


@torch.no_grad()
def validate(model, loader, criterion, device, phase, epoch, config, writer=None):
    """Validate the model."""
    model.eval()
    epoch_losses = {}
    all_dice = []
    num_batches = 0

    for batch in tqdm(loader, desc=f"Val {epoch}"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        batch_gpu = {"image": images, "label": labels}

        # Generate synthetic du/dt for validation too (consistent evaluation)
        if phase in ("finetune", "joint"):
            batch_gpu = generate_synthetic_du_dt_for_batch(batch_gpu, config, device)

        output = model(images)
        loss_dict = criterion(output, batch_gpu, phase=phase)

        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            epoch_losses[k] = epoch_losses.get(k, 0) + v

        # Dice scores
        if "segmentation" in output:
            dice = compute_dice(output["segmentation"], labels)
            all_dice.append(dice)

        num_batches += 1

    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    # Average dice
    if all_dice:
        avg_dice = {}
        for k in all_dice[0]:
            avg_dice[k] = np.mean([d[k] for d in all_dice])
        epoch_losses.update(avg_dice)

    if writer:
        for k, v in epoch_losses.items():
            writer.add_scalar(f"val/{k}", v, epoch)

    return epoch_losses


def save_checkpoint(model, optimizer, scheduler, epoch, losses, path, ema_model=None):
    """Save model checkpoint."""
    checkpoint_state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "losses": losses,
    }
    if ema_model is not None:
        checkpoint_state["ema_model_state_dict"] = ema_model.state_dict()
    
    torch.save(checkpoint_state, path)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(model, optimizer, scheduler, path, device, ema_model=None):
    """Load model checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            print("  Warning: Could not load optimizer state")
    if scheduler and ckpt.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            pass
    if ema_model and "ema_model_state_dict" in ckpt:
        try:
            ema_model.load_state_dict(ckpt["ema_model_state_dict"], strict=False)
            print("  Loaded EMA model state")
        except Exception:
            print("  Warning: Could not load EMA model state")
            
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    return ckpt.get("epoch", 0)


def train(config: Config):
    """Main training function."""
    tc = config.train
    set_seed(tc.seed)

    device = torch.device(tc.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Loading data...")
    train_loader, val_loader = get_train_val_dataloaders(
        data_root=config.data.data_root,
        batch_size=tc.batch_size,
        spatial_size=config.data.spatial_size,
        num_workers=0,  # CRITICAL FOR WINDOWS: Avoids Multiprocessing OOM / Deadlocks
        cache_rate=0.0, # Do not cache into RAM on machines with low memory
    )

    # Model
    print("Building model...")
    model = build_model(config).to(device)
    model.set_phase(tc.phase)
    
    ema_model = EMAModel(model, device=device)

    # Loss
    criterion = HybridTumorLoss(config).to(device)

    # Optimizer & scheduler
    num_epochs = (tc.num_epochs_pretrain if tc.phase == "pretrain"
                  else tc.num_epochs_finetune)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, num_epochs)

    # AMP scaler
    scaler = GradScaler(device.type) if tc.use_amp and device.type == "cuda" else None

    # Resume
    start_epoch = 0
    if tc.resume_from and os.path.exists(tc.resume_from):
        start_epoch = load_checkpoint(model, optimizer, scheduler,
                                       tc.resume_from, device, ema_model)

    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(tc.log_dir, tc.phase))

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training Phase: {tc.phase.upper()}")
    print(f"Epochs: {start_epoch} -> {num_epochs}")
    print(f"Batch size: {tc.batch_size}")
    if tc.phase in ("finetune", "joint"):
        print(f"Synthetic Δt: {config.physics.synthetic_delta_t} days")
        print(f"PDE model: {config.physics.pde_model}")
    print(f"{'='*60}\n")

    # JSON metrics log
    metrics_log_path = os.path.join(tc.log_dir, f"{tc.phase}_metrics.jsonl")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        # Disable automatic zero_grad in loop start since we handle it in accumulation
        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, tc.phase, epoch, config, writer, ema_model
        )

        # Validate (using EMA model if available)
        val_losses = {}
        if (epoch + 1) % tc.val_every == 0:
            val_model = ema_model if ema_model is not None else model
            val_losses = validate(
                val_model, val_loader, criterion, device,
                tc.phase, epoch, config, writer
            )

        # Scheduler step
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses.get("total_loss", train_losses["total_loss"]))
            else:
                scheduler.step()

        # Logging
        elapsed = time.time() - t0
        
        # Log GPU memory
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_losses['total_loss']:.4f} | "
              f"Val Loss: {val_losses.get('total_loss', 'N/A')} | "
              f"Time: {elapsed:.1f}s | GPU Mem: {gpu_mem:.0f}MB")

        if val_losses:
            for k, v in val_losses.items():
                if "dice" in k:
                    print(f"  {k}: {v:.4f}")
                    
        # Log metrics to JSONL
        epoch_metrics = {
            "epoch": epoch,
            "phase": tc.phase,
            "train": train_losses,
            "val": val_losses,
            "time_sec": elapsed,
            "gpu_mem_mb": gpu_mem,
            "lr": optimizer.param_groups[0]["lr"]
        }
        with open(metrics_log_path, "a") as f:
            f.write(json.dumps(epoch_metrics) + "\n")

        # Save best
        val_loss = val_losses.get("total_loss", train_losses["total_loss"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_losses,
                os.path.join(tc.checkpoint_dir, f"{tc.phase}_best.pth"),
                ema_model=ema_model
            )
        else:
            patience_counter += 1

        # Periodic save
        if (epoch + 1) % tc.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_losses,
                os.path.join(tc.checkpoint_dir, f"{tc.phase}_epoch{epoch}.pth"),
                ema_model=ema_model
            )

        # Early stopping
        if patience_counter >= tc.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Tumor Model")
    parser.add_argument("--data_root", type=str, default="./data/Task01_BrainTumour")
    parser.add_argument("--phase", type=str, default="pretrain",
                        choices=["pretrain", "finetune", "joint"])
    parser.add_argument("--backbone", type=str, default="resnet10")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pde_model", type=str, default="fisher_kpp",
                        choices=["fisher_kpp", "gompertz"])
    args = parser.parse_args()

    config = get_config()
    config.data.data_root = args.data_root
    config.train.phase = args.phase
    config.model.backbone = args.backbone
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    config.train.device = args.device
    config.physics.pde_model = args.pde_model

    if args.resume:
        config.train.resume_from = args.resume
    if args.epochs:
        if args.phase == "pretrain":
            config.train.num_epochs_pretrain = args.epochs
        else:
            config.train.num_epochs_finetune = args.epochs

    train(config)


if __name__ == "__main__":
    main()
