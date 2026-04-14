# 🎓 PPT Presentation Order & Talking Points
### Hybrid MONAI ResNet + PINN — Brain Tumor Growth Prediction

---

## SLIDE 1 — Title Slide
**Visual:** Project title + your name + university logo

**Say:**
> "Today I will present a hybrid AI system that combines deep learning with mathematical physics to predict how brain tumors grow over time — from a single MRI scan, without needing any follow-up patient data."

---

## SLIDE 2 — Problem Statement
**Points to cover:**
1. Brain tumors (Gliomas) are extremely aggressive and grow unpredictably
2. Standard MRI only gives a photograph of today — not a prediction of tomorrow
3. Doctors cannot know when or where the tumor will spread next before the next scan
4. Standard AI segments the tumor — but segmentation alone has no predictive power

**Say:**
> "The core problem: MRI is a snapshot. Doctors plan treatment based on today's image, but the tumor is already moving by the time the treatment begins."

---

## SLIDE 3 — Motivation & Goal
**Points to cover:**
1. Goal: Build an AI that automatically extracts tumor growth physics from a single MRI
2. No need for follow-up scans (expensive and unavailable in most hospitals)
3. Output: Not just a tumor outline — but a 30-day growth simulation
4. Clinical value: Better surgical margins, radiation planning, recurrence prediction

**Say:**
> "Our motivation was to give a doctor not just a map of where the tumor is — but a forecast of where it will be in 30 days."

---

## SLIDE 4 — Background: What is a Brain Tumor?
**Points to cover:**
1. Gliomas grow by two mechanisms: **Proliferation** (cells dividing) and **Diffusion** (cells invading)
2. Tumor cells travel 5–10x faster along white matter tracts than grey matter
3. This anisotropic (direction-dependent) growth means standard circular tumor models fail
4. Mathematical models exist (Fisher-KPP) but take expert manual calibration — our model automates this

**Say:**
> "Biology tells us the math model to use. Our AI simply learns to estimate that math's parameters directly from the MRI."

---

## SLIDE 5 — Dataset
**Points to cover:**
1. Brain Tumor MRI Decathlon — Task 01 (BraTS dataset)
2. 484 patients, each with 4 MRI modality channels
3. Each volume is 240×240×155 voxels
4. Expert-annotated ground truth labels per patient
5. **Key challenge:** All 484 scans are single time-point (no follow-up) — solved by our physics simulator

**Say:**
> "We have no Day-30 MRIs. The physics solver generates synthetic temporal derivatives — teaching the model the rules of growth without needing real longitudinal data."

---

## SLIDE 6 — The 5 MRI Input Images (INSERT: `ALL_5_INPUTS_ONLY.png`)
**Points to cover:**
1. Every row = one patient, every column = one MRI modality
2. T1 shows anatomy — brain structure, white vs grey matter
3. T1c (contrast-enhanced) reveals the **actively growing tumor** — bright white = broken blood-brain barrier
4. T2 shows water content — big bright clouds = tumor bulk + swelling (edema)
5. FLAIR suppresses normal fluid — what remains bright is **abnormal infiltration**

**Say:**
> "These 4 grayscale images are what the AI sees. Each modality is a different lens. The AI must combine all 4 channels simultaneously to understand the full biological picture."

**Point out in image:**
- BRATS_002 T1c: Two enormous bright lobes — bilateral aggressive tumor
- BRATS_004 T1c: Large bright top region — matches the biggest 30-day growth prediction
- BRATS_005 T1c: Round, well-defined mass — a more localized but still large lesion

---

## SLIDE 7 — MRI Modalities Explained (keep brief)
**Quick table — 30 seconds per row:**

| Modality | Pulse Sequence | Highlights |
|---|---|---|
| T1 | Standard | Anatomy, fat |
| T1c | + Gadolinium contrast dye | Active tumor (enhancing region) |
| T2 | Long echo | Water, edema, tumor bulk |
| FLAIR | Fluid suppression | Abnormal infiltration, edema boundary |

---

## SLIDE 8 — Ground Truth Labels
**Points to cover:**
1. Expert-drawn 3-class segmentation per patient
2. Label 4 = Enhancing Tumor (ET) — dense, proliferating cells
3. Label 1 = Necrotic Core (NCR) — dead tissue center from insufficient blood
4. Label 2 = Edema (ED) — sparsely invaded surrounding tissue
5. These labels are mapped to continuous density values (0.0–1.0) for the physics equation

**Say:**
> "We convert the hard class labels into continuous biological cell density values — this is the critical bridge between digital imaging and mathematical physics."

---

## SLIDE 9 — Preprocessing Pipeline
**Points to cover:**
1. Raw MRI from different hospitals has inconsistent scales and orientations
2. Step 1: Orient all brains to RAS coordinate system
3. Step 2: Resample all voxels to 1mm³ resolution
4. Step 3: Normalize signal intensities within brain mask only
5. Step 4: Crop 128³ sub-volumes for GPU memory efficiency

**Say:**
> "Without preprocessing, the AI would learn scanner noise instead of tumor biology."

---

## SLIDE 10 — The Fisher-KPP Physics Equation
**Points to display:**
$$\frac{\partial u}{\partial t} = \nabla \cdot (D(x)\nabla u) + \rho(x) u(1-u)$$

1. $u(x,t)$ = tumor cell density at location $x$ and time $t$
2. Left side = rate of change over time
3. $D(x)$ term = spatial diffusion — how fast cells migrate
4. $\rho(x)$ term = logistic proliferation — how fast cells divide (with carrying capacity)
5. The AI's job: estimate $D(x)$ and $\rho(x)$ directly from the MRI

**Say:**
> "This equation is the gold standard in mathematical neuro-oncology for over 30 years. We do not change the math — we automate the calibration."

---

## SLIDE 11 — Hybrid Network Architecture
**Points to cover:**
1. 3D ResNet-50 as the feature extraction backbone (spatial understanding)
2. Decoder Branch 1: Predicts current tumor density $u(x)$
3. Decoder Branch 2: Predicts physics parameters $D(x)$, $\rho(x)$ spatially
4. Deep supervision: intermediate outputs at 3 scales for faster training
5. Skip connections + Attention Gates: focuses the model on tumor-relevant features

**Say:**
> "The ResNet sees the tumor. The physics decoder asks: given what I see, what must D and ρ be?"

---

## SLIDE 12 — Two-Phase Training Strategy
**Points to cover:**
1. **Phase 1 — Pretraining:** Focus on segmentation (Dice + Cross-Entropy loss)
2. Goal: Learn what the tumor looks like spatially (73 epochs completed on Kaggle GPU)
3. **Phase 2 — Physics Finetuning:** Activate the PDE loss, physics decoder learns $D$, $\rho$
4. The pre-trained backbone is used as initialization — no training from scratch
5. Result: Stable convergence without getting stuck in bad physics solutions

**Say:**
> "Teaching the AI anatomy before teaching it physics is the same strategy a medical student follows — learn the structure before learning the disease dynamics."

---

## SLIDE 13 — Combined Loss Function
**Show the equation:**
$$\mathcal{L}_{total} = \lambda_{data}\mathcal{L}_{data} + \lambda_{PDE}\mathcal{L}_{PDE}$$

**Points to cover:**
1. Data Loss = Dice + Cross-Entropy — ensures spatial accuracy
2. PDE Loss = MSE of Fisher-KPP residual — ensures physics accuracy
3. Synthetic temporal derivatives: an internal RK4 numerical solver generates $\partial u/\partial t$ as training target
4. This is how we train physics without longitudinal follow-up scans
5. $\lambda_{PDE}$ is tuned to balance both losses — physics should not overwhelm anatomy

**Say:**
> "The PDE loss is the key innovation: it penalizes the model whenever its predicted D and ρ would produce growth patterns that violate the Fisher-KPP equation."

---

## SLIDE 14 — Patient Inference Results: BRATS_001 (INSERT patient image)
**Points to cover:**
1. **Input:** Relatively small, localized bright spot in T1c — confirmed by T2 edema at bottom
2. **Cellular Density:** Maximum density at bottom matching T1c exactly ✓
3. **Diffusion D(x):** High at skull-adjacent left edge, zero in air outside brain ✓ (proves Neumann BC learned)
4. **30-Day Simulation:** Tumor expands upward and to the right along white matter path

---

## SLIDE 15 — Patient Inference Results: BRATS_002 (INSERT patient image)
**Points to cover:**
1. **Input:** Enormous bilateral tumor — both hemispheres lit up in T1c
2. **Cellular Density:** Two separate hot spots detected — primary + satellite ✓
3. **Diffusion D(x):** Sharp boundary at ventricles (no diffusion into liquid) ✓
4. **30-Day Simulation:** Highly localized concentrated growth — **proliferation-dominant** pattern

---

## SLIDE 16 — Patient Inference Results: BRATS_003 (INSERT patient image)
**Points to cover:**
1. **Input:** Left hemisphere more severely affected — T1c shows heterogeneous (mixed dead/alive) texture
2. **Cellular Density:** Correctly placed HIGHER density in left hemisphere ✓
3. **Diffusion D(x):** High diffusion along left white matter — where this tumor is most mobile
4. **30-Day Simulation:** Multi-focal spread — NEW secondary lesion appears in lower-right quadrant

---

## SLIDE 17 — Patient Inference Results: BRATS_004 (INSERT patient image)
**Points to cover:**
1. **Input:** Large bright T1c region at top, complex internal texture (cystic + solid)
2. **Cellular Density:** Two separate hot spots — primary and satellite lesion ✓
3. **Diffusion D(x):** Unusually hot (yellow) arc at bottom-left — high white matter mobility
4. **30-Day Simulation:** Most AGGRESSIVE expansion of all 5 patients — largest predicted growth

---

## SLIDE 18 — Patient Inference Results: BRATS_005 (INSERT patient image)
**Points to cover:**
1. **Input:** Single round, well-defined bright T1c mass
2. **Cellular Density:** Clear single peak at top-center, secondary glow in center ✓
3. **Diffusion D(x):** Diffusion concentrated at inferior-lateral corner — white matter pathway
4. **30-Day Simulation:** Distributed, even spread (not a sharp core) — **diffusion-dominant** pattern

---

## SLIDE 19 — Cross-Patient Comparison (Key Scientific Finding)
**Summary table to display:**

| Patient | Tumor Character | D(x) Pattern | Growth Type | 30-Day Outcome |
|---|---|---|---|---|
| BRATS_001 | Localized | Skull-edge high | Moderate | Upward migration |
| BRATS_002 | Bilateral | Ventricle-bounded | Proliferation-dominant | Stays concentrated |
| BRATS_003 | Left-dominant | Left WM high | Multi-focal | New secondary lesion |
| BRATS_004 | Complex/Cystic | High WM arc | Most aggressive | Largest expansion |
| BRATS_005 | Round/Defined | Inferior WM | Diffusion-dominant | Even distributed spread |

**Say:**
> "This is exactly what standard segmentation AI cannot provide. Every patient gets their unique biological fingerprint — their personal $D$ and $\rho$. That personalization is the scientific contribution."

---

## SLIDE 20 — Conclusion, Limitations & Future Work
**Advantages:**
- Automates 30-year-old biophysical modeling
- Works on single-time-point clinical MRI (no follow-up needed)
- Patient-specific growth parameters — not population averages
- MC Dropout provides uncertainty maps for clinical safety

**Limitations:**
- Segmentation head needs more pretraining epochs for clean outputs
- Validated on synthetic temporal derivatives — needs longitudinal dataset validation
- CPU prediction takes 3–5 mins per patient (GPU resolves this)

**Future Work:**
- Validate on LUMIERE / BraTS-GLI 2024 longitudinal datasets
- Add radiation treatment sink term to Fisher-KPP equation
- Deploy as a web-based clinical decision support tool

**Final say:**
> "We successfully combined 30 years of mathematical neuro-oncology with modern 3D deep learning. This system gives doctors something new: not just a map of where the tumor IS — but a physics-based forecast of where it WILL BE."

---

> [!IMPORTANT]
> **Files to attach to your PPT:**
> - `final_output/ppt_images/ALL_5_INPUTS_ONLY.png` → Use on Slide 6
> - `final_output/ppt_images/BRATS_001_inference.png` → Slide 14
> - `final_output/ppt_images/BRATS_002_inference.png` → Slide 15
> - `final_output/ppt_images/BRATS_003_inference.png` → Slide 16
> - `final_output/ppt_images/BRATS_004_inference.png` → Slide 17
> - `final_output/ppt_images/BRATS_005_inference.png` → Slide 18
