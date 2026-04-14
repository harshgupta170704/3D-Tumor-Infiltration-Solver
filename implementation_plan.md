# Improvements to Hybrid MONAI ResNet + PINN Project

After a thorough code audit of every file, I've identified **8 high-impact improvements** that would meaningfully strengthen the project from both scientific rigor and engineering quality perspectives.

## Proposed Changes

### 1. Gradient Accumulation (Critical for Small GPUs)

#### [MODIFY] [train.py](file:///c:/Users/Lenovo/OneDrive/Desktop/Fine%20tuning%20monia%20resnet%20modal%20with%20pinn%20modal/train.py)

Currently batch_size=1 is the only option on small GPUs. Gradient accumulation allows simulating batch_size=4/8 without extra VRAM. This directly improves training stability and convergence.

- Add `accumulation_steps` config parameter
- Modify training loop to accumulate gradients over N mini-batches before stepping

#### [MODIFY] [config.py](file:///c:/Users/Lenovo/OneDrive/Desktop/Fine%20tuning%20monia%20resnet%20modal%20with%20pinn%20modal/config.py)

- Add `gradient_accumulation_steps: int = 4` to `TrainConfig`

---

### 2. Exponential Moving Average (EMA) Model

#### [NEW] [utils/ema.py](file:///c:/Users/Lenovo/OneDrive/Desktop/Fine%20tuning%20monia%20resnet%20modal%20with%20pinn%20modal/utils/ema.py)

EMA maintains a shadow copy of model weights as an exponentially decayed average. At inference, the EMA model consistently outperforms the raw training model (Polyak averaging). Standard practice in competitive medical imaging (nnU-Net).

- Implement `EMAModel` class with configurable decay (default 0.999)
- Save EMA weights alongside regular checkpoints
- Use EMA weights for validation and final inference

---

### 3. Deep Supervision in Decoders

#### [MODIFY] [decoder.py](file:///c:/Users/Lenovo/OneDrive/Desktop/Fine%20tuning%20monia%20resnet%20modal%20with%20pinn%20modal/models/decoder.py)

Deep supervision adds auxiliary loss outputs at intermediate decoder resolutions (proven in V-Net, nnU-Net). This provides stronger gradient signal to earlier layers and significantly improves convergence speed and final accuracy.

- Add intermediate output heads at decoder levels 2 and 3
- Return list of outputs at multiple scales during training
- Only return full-resolution output during inference

#### [MODIFY] [combined_loss.py](file:///c:/Users/Lenovo/OneDrive/Desktop/Fine%20tuning%20monia%20resnet%20modal%20with%20pinn%20modal/losses/combined_loss.py)

- Handle multi-scale deep supervision outputs with exponentially decaying weights

---

### 4. Sliding Window Inference for Full-Resolution Prediction

#### [MODIFY] [predict.py](file:///c:/Users/Lenovo/OneDrive/Desktop/Fine%20tuning%20monia%20resnet%20modal%20with%20pinn%20modal/predict.py)

Current inference crops to 64³/128³ and loses spatial context. MONAI's `sliding_window_inference` processes the full volume in overlapping patches and blends results — critical for actual clinical use.

- Use `monai.inferers.sliding_window_inference` with Gaussian-weighted overlap blending
- Configurable overlap ratio (default 0.5)

---

### 5. Monte Carlo Dropout Uncertainty Quantification

#### [MODIFY] [predict.py](file:///c:/Users/Lenovo/OneDrive/Desktop/Fine%20tuning%20monia%20resnet%20modal%20with%20pinn%20modal/predict.py)

The README lists uncertainty quantification as a "future direction." MC Dropout (Gal & Ghahramani 2016) is straightforward to implement and adds real scientific value: it produces voxel-wise uncertainty maps by running multiple stochastic forward passes with dropout enabled.

- Add `--mc_samples` CLI argument (default 0 = disabled, 10-20 for UQ)
- Compute mean prediction and predictive uncertainty (standard deviation across samples)
- Save uncertainty maps as NIfTI and visualization

---

### 6. Cosine Annealing with Warm Restarts

#### [MODIFY] [train.py](file:///c:/Users/Lenovo/OneDrive/Desktop/Fine%20tuning%20monia%20resnet%20modal%20with%20pinn%20modal/train.py)

Add `CosineAnnealingWarmRestarts` (Loshchilov & Hutter 2017) scheduler option. Periodic warm restarts help escape local minima and are especially effective for physics-informed training where the loss landscape is complex.

#### [MODIFY] [config.py](file:///c:/Users/Lenovo/OneDrive/Desktop/Fine%20tuning%20monia%20resnet%20modal%20with%20pinn%20modal/config.py)

- Add `"cosine_warm_restarts"` as scheduler option

---

### 7. Comprehensive Training Dashboard & Metrics Logging

#### [MODIFY] [train.py](file:///c:/Users/Lenovo/OneDrive/Desktop/Fine%20tuning%20monia%20resnet%20modal%20with%20pinn%20modal/train.py)

Currently, training logs are minimal. Add:
- Per-epoch JSON log files with all metrics for reproducibility
- Training summary at the end with best metrics across all epochs
- Log physics parameter statistics (D_mean, D_std, ρ_mean, ρ_std) every epoch
- GPU memory usage logging

---

### 8. Residual Connections in Decoder ConvBlocks

#### [MODIFY] [decoder.py](file:///c:/Users/Lenovo/OneDrive/Desktop/Fine%20tuning%20monia%20resnet%20modal%20with%20pinn%20modal/models/decoder.py)

The `ConvBlock3D` decoder blocks lack residual connections. Adding identity shortcuts (with 1×1 conv for channel mismatch) improves gradient flow through the decoder, which is particularly important for the deep physics-parameter head.

---

## Verification Plan

### Automated Tests
```bash
python validate_physics.py
```
All existing tests must still pass after changes. New features (EMA, deep supervision, MC Dropout) will be validated by the existing test infrastructure since they don't change the model's numerical contract.

### Manual Verification
- Run `python train.py --phase pretrain --epochs 2` to verify gradient accumulation and EMA work correctly
- Run `python predict.py` with `--mc_samples 5` to verify uncertainty maps generate correctly
