# Honest Audit: Was Our LUMIERE Evaluation Correct?

## 1. What Is HybridTumorNet Actually Designed To Do?

The model has **three heads**, each with a different purpose:

| Head | Output | Purpose |
|------|--------|---------|
| **TumorDensityDecoder** | `u(x,t)` ∈ [0,1] | Predicts a continuous tumor cell density field |
| **PhysicsParameterHead** | `D(x)`, `ρ(x)` > 0 | Predicts spatially-varying diffusion and proliferation parameters for the Fisher-KPP PDE |
| **SegmentationHead** | 4-class logits | Standard BraTS segmentation (BG, NCR, ED, ET) — used **only for pretraining** |

The **core purpose** of HybridTumorNet is **tumor growth prediction** via physics-informed learning. The segmentation head is explicitly documented as a pretraining auxiliary — it exists to give the encoder meaningful features before the physics-informed fine-tuning phase kicks in.

The full training pipeline is designed in **two phases**:
1. **Pretrain phase**: Train the segmentation head on BraTS data (Dice+CE loss)
2. **Finetune phase**: Freeze the seg head, train the density decoder and physics heads using PDE residual losses

## 2. What State Is the Checkpoint In?

The checkpoint `pretrain_best.pth` tells us:

- **Epoch**: 69
- **Training phase**: Pretraining (the filename literally says "pretrain")
- **Backbone**: ResNet-10 (the lightest variant, `decoder_channels=[128, 64, 32, 16, 8]`)
- **Training data**: BraTS Task01_BrainTumour (MSD), spatial size 64×64×64
- **Stored metrics at save time**:
  - Dice WT (Whole Tumor): **0.461**
  - Dice TC (Tumor Core): **0.408**
  - Dice ET (Enhancing Tumor): **0.317**
  - Mean Dice: **0.395**

> [!IMPORTANT]
> The model was saved during **pretraining on BraTS** with a mean Dice of only **0.395** — which is itself quite low even for BraTS. The physics-informed fine-tuning phase (Phase 2) was **never run** on this checkpoint.

## 3. What Did We Actually Do in the Evaluation?

We took the segmentation head output and compared it against LUMIERE ground truth masks using MONAI's `DiceMetric`. Here is what happened step by step:

### What went in:
- **Input MRI**: Raw LUMIERE T1, T1c, T2, FLAIR scans (native space, **640×640×24**, 0.36mm in-plane, **6mm slice thickness**)
- **Ground truth mask**: DeepBraTumIA atlas-space `seg_mask.nii.gz` (**182×218×182**, 1mm isotropic MNI space)

### What the script did:
1. Loaded all 4 modalities + mask separately
2. Resampled everything to 4mm isotropic via `Spacingd`
3. Resized everything to 128×128×128 via `Resized`
4. Concatenated the 4 modalities
5. Ran the model → took `outputs['segmentation']` → argmax → compared to mask

### Result: Dice ≈ 0.003 (0.3%)

## 4. What Was Wrong With This Evaluation

### Problem 1: Spatial Mismatch (CRITICAL)
The MRI scans (T1, T1c, etc.) are in **native acquisition space** (640×640×24, thick 6mm slices). The ground truth `seg_mask.nii.gz` from the `atlas/segmentation` folder is in **MNI atlas space** (182×218×182, 1mm isotropic). 

**These are literally in different coordinate systems.** When MONAI's `Spacingd` and `Resized` transforms operate on them independently, the tumor in the MRI ends up at a completely different spatial location than the tumor in the mask. The Dice score is near zero partly because the predictions and ground truth aren't even aligned.

### Problem 2: We Evaluated the Wrong Head
The segmentation head is the **pretraining auxiliary**. The core purpose of HybridTumorNet is the density decoder + physics parameters. If we wanted to evaluate the model's actual purpose, we should have:
- Looked at the `tumor_density` output (continuous u(x,t) field)
- Compared it against a density-converted ground truth using the project's own `seg_to_density()` function
- Or evaluated the physics parameters (D, ρ) for physical plausibility

We evaluated the segmentation head as if it were the main model — but the architecture documentation explicitly calls it "(pretrain)" and "(optional)".

### Problem 3: The Model Was Trained on 64³, We Tested on 128³
The config shows `spatial_size: (64, 64, 64)`. The model was trained on volumes resized to 64×64×64. We forced 128×128×128 inputs through it. While the model architecture can technically handle different input sizes (fully convolutional), the feature statistics the model learned are calibrated for 64³ volumes.

### Problem 4: BraTS Preprocessing vs. Raw LUMIERE
BraTS data is:
- Skull-stripped
- Co-registered (all modalities aligned to a common space)
- Resampled to 1mm isotropic
- Intensity-normalized

LUMIERE raw data has:
- Skulls present
- Each modality potentially in its own native space
- 6mm thick slices (very anisotropic)
- No standardized intensity normalization

This domain gap alone would tank performance, even if everything else were correct.

### Problem 5: Label Semantics May Differ
BraTS uses labels `{0, 1, 2, 4}` which get remapped to `{0, 1, 2, 3}`. LUMIERE's DeepBraTumIA segmentation already uses `{0, 1, 2, 3}` — but we never verified that label 1 in LUMIERE means the same thing as label 1 in BraTS (NCR/NET), or that label 3 means ET. If the label semantics differ, the Dice comparison is meaningless.

## 5. Was the Core Idea of Evaluating on LUMIERE Valid?

**Yes, but the execution had fundamental flaws.**

The idea was sound: take a model pretrained on BraTS and see how it generalizes to an independent dataset (LUMIERE). This is a legitimate zero-shot transfer experiment.

But for the results to mean anything, you need:
1. ✅ Same modalities (T1, T1c, T2, FLAIR) — LUMIERE has these
2. ❌ Same preprocessing pipeline — we didn't apply BraTS-equivalent preprocessing
3. ❌ Spatially aligned inputs and ground truth — we mixed atlas-space masks with native-space MRIs
4. ❌ Matching spatial resolution — trained on 64³, tested on 128³
5. ❌ Verified label semantics — never confirmed LUMIERE label definitions match BraTS

## 6. What Should We Have Done Instead

A correct zero-shot evaluation on LUMIERE would require:

1. **Use the native-space masks** (`DeepBraTumIA/native/segmentation/t1_seg_mask.nii.gz`) that are already in the same space as the T1 scan, OR register the atlas mask back to native space
2. **Apply BraTS-equivalent preprocessing**: skull-strip, co-register all modalities, resample to 1mm isotropic
3. **Resize to 64×64×64** (matching training resolution) or at minimum center-crop to the training size
4. **Verify label semantics** before computing Dice
5. **Evaluate the density decoder** (the actual model purpose) in addition to or instead of the segmentation head

## 7. Bottom Line

| Question | Answer |
|----------|--------|
| Was the evaluation concept valid? | Yes — zero-shot transfer to new data is a reasonable experiment |
| Was the execution correct? | **No** — critical spatial misalignment between MRI and ground truth mask |
| Is the 0.3% Dice score meaningful? | **No** — it reflects preprocessing errors more than model quality |
| What does the checkpoint actually tell us? | The model achieved ~0.40 mean Dice on BraTS itself during pretraining, which is moderate |
| Was the right output head evaluated? | Partially — the seg head is legitimate to evaluate, but it's the auxiliary head, not the core purpose |
