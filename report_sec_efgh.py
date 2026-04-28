import os
FIG_DIR = os.path.abspath('.')

SEC_EFGH = f"""
<div class="section-break">
<h1>4. PROPOSED METHODOLOGY AND WORK</h1>

<h2>4.1 Dataset: BraTS 2021</h2>
<p>The Brain Tumor Segmentation (BraTS) 2021 Challenge dataset comprises 1,251 pre-operative multi-parametric MRI scans of glioma patients sourced from 23 participating institutions. Each case includes four co-registered, skull-stripped volumetric sequences: native T1, contrast-enhanced T1 (T1ce), T2-weighted, and T2-FLAIR, accompanied by expert-annotated segmentation labels delineating three sub-tumor regions — necrotic/non-enhancing tumor core (NCR/NET, label 1), peritumoral edematous/invaded tissue (ED, label 2), and the enhancing tumor (ET, label 4, remapped to 3). The dataset is split 80/20 into training (1,000 cases) and validation (251 cases) sets using a stratified random split with seed 42 to ensure reproducibility. All volumes have isotropic 1mm³ voxel resolution and are pre-registered to the SRI24 brain atlas.</p>

<h2>4.2 Preprocessing Pipeline</h2>
<p><b>Step 1 — Loading and Channel Stacking:</b> The four modality NIfTI files per patient are loaded using nibabel and stacked along the channel dimension to produce a 4×240×240×155 input tensor. MONAI's LoadImaged and EnsureChannelFirstd transforms handle I/O and tensor formatting.</p>
<p><b>Step 2 — Intensity Normalization:</b> Each modality channel is independently z-score normalized (zero mean, unit variance) computed within the non-zero brain region to avoid bias from background voxels. This is performed by NormalizeIntensityd with nonzero=True.</p>
<p><b>Step 3 — Label Remapping:</b> The original BraTS label 4 (ET) is remapped to 3 using ConvertToMultiChannelBasedOnBratsClassesd to produce contiguous integer labels {0,1,2,3}.</p>
<p><b>Step 4 — Foreground Cropping:</b> CropForegroundd removes background-only slices along all axes based on the union of non-zero regions across all channels, reducing computational waste.</p>
<p><b>Step 5 — Spatial Resampling:</b> Volumes are randomly cropped to the target spatial size of 64³ (development) or 128³ (final experiments) using RandSpatialCropd. This crop size respects the VRAM budget of the target GPU.</p>
<p><b>Step 6 — Data Augmentation (Training Only):</b> Random horizontal/vertical/depth axis flips (p=0.5 per axis), random affine rotations (±15°, p=0.3), random Gaussian noise (σ=0.1, p=0.3), and random intensity scaling (±10%, p=0.3) are applied to improve generalization. All augmentations preserve label validity.</p>

<h2>4.3 Model Architecture: HybridTumorNet</h2>
<p><b>4.3.1 ResNet Encoder Backbone:</b> The encoder employs MONAI's 3D ResNet, configurable between ResNet-10 (1.7M parameters, basic blocks) and ResNet-50 (46M parameters, bottleneck blocks). The backbone extracts five levels of multi-scale spatial features from the 4-channel MRI input:</p>
<table>
<tr><th>Level</th><th>ResNet-50 Channels</th><th>Spatial Scale (for 128³ input)</th></tr>
<tr><td>Stem (conv1+bn+relu)</td><td>64</td><td>64³</td></tr>
<tr><td>Layer 1 (3 blocks)</td><td>256</td><td>32³</td></tr>
<tr><td>Layer 2 (4 blocks)</td><td>512</td><td>16³</td></tr>
<tr><td>Layer 3 (6 blocks)</td><td>1024</td><td>8³</td></tr>
<tr><td>Layer 4 (3 blocks)</td><td>2048</td><td>4³</td></tr>
</table>
<p class="caption">Table 1: ResNet-50 encoder feature dimensions and spatial scales for 128³ input volume.</p>

<p><b>4.3.2 Tumor Density Decoder:</b> A U-Net-style decoder progressively upsamples Layer 4 features back to the input resolution through 5 transposed-convolution + skip-connection stages. At each stage, encoder features at the corresponding resolution are concatenated (skip connections) and processed by two Conv3D-BN-ReLU blocks. The final output passes through a 1×1×1 convolution followed by Sigmoid activation to produce u(x,t) ∈ [0,1]. Deep supervision during training computes auxiliary losses at intermediate resolutions (weights: 1.0, 0.5, 0.25) to improve gradient flow to early layers.</p>

<p><b>4.3.3 Physics Parameter Head:</b> A lightweight branch applies global average pooling to Layer 4 features, processes through two fully-connected layers with ReLU, then branches into two output heads. Each head applies a Softplus activation followed by affine scaling to constrain outputs within biophysically valid ranges: D(x) ∈ [0.0001, 0.5] mm²/day and ρ(x) ∈ [0.001, 0.5] /day. The output is upsampled to the full volume resolution to produce spatially varying parameter maps.</p>

<p><b>4.3.4 Segmentation Head:</b> A separate U-Net decoder identical in structure to the density decoder but with 4-class softmax output for BraTS sub-region delineation. This head is active during Phase 1 (pre-training) and frozen during Phase 2 (physics fine-tuning), ensuring segmentation quality does not regress during PINN training.</p>

<div class="fig-center">
<img src="{FIG_DIR}/fig2_architecture.png" width="580"/>
</div>
<p class="caption">Figure 1: HybridTumorNet three-head architecture. The shared MONAI 3D ResNet encoder feeds three parallel decoder heads for tumor density, physics parameters, and segmentation.</p>

<h2>4.4 Physics-Informed Loss Function</h2>
<p>The total training loss combines six terms:</p>
<p style="text-align:center; font-family:monospace; background:#f5f5f5; padding:6pt;">
L_total = λ_data·L_data + λ_pde·L_PDE + λ_ic·L_IC + λ_bc·L_BC + λ_reg·L_reg + λ_seg·L_seg
</p>
<table>
<tr><th>Term</th><th>Formula</th><th>Weight</th><th>Purpose</th></tr>
<tr><td>L_data</td><td>||u_pred − u_target||²</td><td>1.0</td><td>Data fidelity to density map</td></tr>
<tr><td>L_PDE</td><td>||∂u/∂t − ∇·(D∇u) − ρu(1−u)||²</td><td>0.1</td><td>Fisher-KPP PDE compliance</td></tr>
<tr><td>L_IC</td><td>||u(x,t₀) − u₀(x)||²</td><td>0.05</td><td>Initial condition matching</td></tr>
<tr><td>L_BC</td><td>||∇u·n||² at ∂Ω</td><td>0.01</td><td>No-flux Neumann boundary</td></tr>
<tr><td>L_reg</td><td>ReLU(−u)² + ReLU(u−1)² + λ||∇u||²</td><td>0.01</td><td>Physical constraints</td></tr>
<tr><td>L_seg</td><td>DiceLoss + CrossEntropyLoss</td><td>1.0</td><td>BraTS segmentation</td></tr>
</table>
<p class="caption">Table 2: Loss function components, formulations, default weights, and purposes.</p>

<p><b>Adaptive Weight Balancing:</b> Weights are automatically adjusted via learnable log-variance parameters following Kendall et al. (2018): L_adaptive = Σ_i (exp(−log_var_i)·L_i + log_var_i), where log_var_i are trainable scalar parameters initialized to zero. This prevents any single loss term from dominating when losses have different magnitudes and units.</p>

<h2>4.5 Synthetic Longitudinal Data Generation</h2>
<p>BraTS 2021 is a cross-sectional dataset — each patient has only one imaging timepoint. The PDE residual loss requires the temporal derivative ∂u/∂t. To supply this without real longitudinal pairs, the FisherKPPSolver numerically integrates the Fisher-KPP equation forward by Δt = 90 days from the segmentation-derived initial density u₀, producing a synthetic future state u(t₂). The temporal derivative is then approximated as du/dt = (u_t2 − u_t1)/Δt.</p>
<p><b>Critical Non-Degeneracy Design:</b> The solver uses DEFAULT physics parameters (D₀ = 0.05 mm²/day, ρ₀ = 0.02 /day), NOT the network's predicted D_pred and ρ_pred. This creates a genuine training signal: the PDE residual R = (u_t2 − u_t1)/Δt − [∇·(D_pred∇u) + ρ_pred·u·(1−u)] is zero only when the network's predictions match the biophysical reality, not trivially. The solver uses 4th-order Runge-Kutta integration with a CFL-stable timestep Δt_step ≤ dx²/(6·D_max) ≈ 1.67 days, requiring ~54 integration steps per training sample.</p>

<h2>4.6 Tissue-Aware Diffusion</h2>
<p>Following Harpold et al. (2007), glioma cells invade white matter approximately 5× faster than grey matter. The effective diffusion is computed as: D_effective(x) = D_pred(x) × [D_wm in WM, D_gm in GM] where D_wm = 0.08 mm²/day, D_gm = 0.016 mm²/day, and white-matter probability is estimated from the MRI intensity. This tissue-aware modulation ensures that the predicted D(x) field encodes the anatomically realistic anisotropy of tumor invasion.</p>

<h2>4.7 Two-Phase Training Protocol</h2>
<p><b>Phase 1 — Supervised Pre-training (50 epochs):</b> The segmentation head is activated, physics head is frozen. Training minimizes L_seg + L_data using BraTS ground-truth labels. The backbone learns to extract tumor-relevant spatial features before physics constraints are imposed. Optimizer: AdamW, LR=1e-4 (heads) / 1e-5 (backbone), cosine annealing with warmup.</p>
<p><b>Phase 2 — Physics Fine-tuning (30 epochs):</b> Segmentation head is frozen. The density decoder and physics head are activated. Training minimizes the full L_total including all PDE terms. The FisherKPPSolver generates du/dt per mini-batch. EMA (decay=0.999) tracks a smoothed model for validation. Gradient clipping (max_norm=1.0) prevents exploding gradients from the PDE residual.</p>
</div>

<div class="section-break">
<h1>5. ALGORITHM</h1>
<h2>5.1 Training Algorithm (Two-Phase PINN)</h2>
<p>The following pseudocode describes the complete training procedure for HybridTumorNet.</p>
<pre style="background:#f5f5f5; padding:10pt; font-size:10pt; font-family:monospace; border-left:3px solid #1a237e; white-space:pre-wrap;">
ALGORITHM: HybridTumorNet Training

INPUT:
  - D_train: BraTS 2021 training set (1,000 patients)
  - Config: architecture, loss weights, physics parameters
  - N_pre: pre-training epochs (default 50)
  - N_fine: fine-tuning epochs (default 30)

INITIALIZE:
  model ← HybridTumorNet(backbone=ResNet-10, heads=[density, physics, seg])
  optimizer ← AdamW(backbone_lr=1e-5, head_lr=1e-4)
  scheduler ← CosineAnnealing(T_max=N_pre, eta_min=1e-7)
  solver ← FisherKPPSolver(D₀=0.05, ρ₀=0.02, dt_max=1.0 day)
  ema ← EMAModel(model, decay=0.999)

─── PHASE 1: SUPERVISED PRE-TRAINING ───────────────────────
model.set_phase("pretrain")
FOR epoch = 1 TO N_pre:
  FOR EACH batch (I, y_seg) IN D_train:
    I ← augment(I)
    outputs ← model.forward(I)
    L ← λ_seg · DiceCELoss(outputs.seg, y_seg)
       + λ_data · MSELoss(outputs.density, seg_to_density(y_seg))
    L.backward(); clip_grad(1.0); optimizer.step(); ema.update()
  scheduler.step()
  IF val_dice > best_dice: save_checkpoint("pretrain_best.pth")

─── PHASE 2: PHYSICS FINE-TUNING ────────────────────────────
model.load("pretrain_best.pth"); model.set_phase("finetune")
FOR epoch = 1 TO N_fine:
  FOR EACH batch (I, y_seg) IN D_train:
    u₀ ← seg_to_density(y_seg)          # Initial condition from segmentation
    brain_mask ← (I.sum(dim=1) > 0)     # Non-zero brain region
    u_t2, du_dt ← solver.simulate(u₀, Δt=90 days, brain_mask)  # Synthetic du/dt
    outputs ← model.forward(I)
    D, ρ ← outputs.diffusion, outputs.proliferation
    u ← outputs.tumor_density

    # Compute all loss terms
    L_pde ← PDEResidualLoss(u, D, ρ, du_dt, brain_mask)
    L_ic  ← ||u − u₀||²
    L_bc  ← BoundaryLoss(u, brain_mask)
    L_reg ← RegLoss(u, D, ρ, brain_mask)
    L_data← MSELoss(u, u₀)

    L_total ← AdaptiveWeighting([L_data, L_pde, L_ic, L_bc, L_reg])
    L_total.backward(); clip_grad(1.0); optimizer.step(); ema.update()
  IF val_pde < best_pde: save_checkpoint("finetune_best.pth")

OUTPUT: Trained model f_θ* with learned D(x), ρ(x), u(x,t), seg(x)
</pre>

<h2>5.2 Inference Algorithm</h2>
<pre style="background:#f5f5f5; padding:10pt; font-size:10pt; font-family:monospace; border-left:3px solid #1a237e; white-space:pre-wrap;">
ALGORITHM: Tumor Growth Prediction (Inference)

INPUT: Patient MRI I (T1, T1ce, T2, FLAIR), trained model f_θ*, simulate_days T

1. LOAD &amp; PREPROCESS: I ← normalize(I), crop, reformat to [1,4,D,H,W]
2. SLIDING WINDOW INFERENCE (overlap=0.5):
   outputs ← sliding_window_inference(I, roi=64³, predictor=f_θ*)
   u, D, ρ, seg ← outputs.split_channels()
3. IF simulate_days > 0:
   trajectory ← [u]
   FOR step = 1 TO T (using RK4, dt=1 day):
     k1 ← F(u)     where F(u) = ∇·(D∇u) + ρu(1−u)
     k2 ← F(u + 0.5·dt·k1)
     k3 ← F(u + 0.5·dt·k2)
     k4 ← F(u + dt·k3)
     u ← u + (dt/6)·(k1 + 2k2 + 2k3 + k4)
     u ← clamp(u, 0, 1)
     trajectory.append(u)
4. SAVE: NIfTI(u, D, ρ, seg), growth_curve.png, uncertainty.png (if MC Dropout)
OUTPUT: u(x, t+T), D(x), ρ(x), seg(x), volume growth curve
</pre>
</div>

<div class="section-break">
<h1>6. PROPOSED FLOWCHART / BLOCK DIAGRAM</h1>
<div class="fig-center">
<img src="{FIG_DIR}/fig1_pipeline.png" width="600"/>
</div>
<p class="caption">Figure 2: Complete pipeline flowchart of HybridTumorNet from multi-modal MRI input through two-phase training to growth simulation output. Color coding: blue = data/input; purple = model architecture; pink = loss function; light purple = PDE physics; indigo = simulation and output.</p>

<div class="fig-center" style="margin-top:20pt;">
<img src="{FIG_DIR}/fig2_architecture.png" width="580"/>
</div>
<p class="caption">Figure 3: Block diagram of the three-head HybridTumorNet architecture showing the shared ResNet encoder, parallel decoder branches, and output tensor shapes for ResNet-50 with 128³ input.</p>
</div>
"""
