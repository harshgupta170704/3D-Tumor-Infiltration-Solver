SEC_AB = """
<div class="cover">
<h1>HYBRID DEEP LEARNING AND PHYSICS-INFORMED NEURAL NETWORK<br/>
FOR BRAIN TUMOR GROWTH PREDICTION</h1>
<p style="font-size:13pt; font-weight:bold; margin-top:20pt;">Minor Project Report</p>
<p style="font-size:12pt;">Submitted in partial fulfillment of the requirements for the degree of<br/>
Bachelor of Technology in Electronics and Communication Engineering</p>
<p style="margin-top:30pt;"><b>Submitted by:</b><br/>
Harsh Gupta — Scholar No.: 23U01011<br/>
3rd Year | B.Tech ECE</p>
<p style="margin-top:20pt;"><b>Indian Institute of Information Technology Bhopal</b><br/>
April 2026</p>
</div>

<div class="section-break">
<h1>ABSTRACT</h1>
<p>Brain tumors, particularly high-grade gliomas such as Glioblastoma Multiforme (GBM), remain among the most lethal intracranial malignancies with a median survival of 14–17 months despite multimodal treatment. The primary clinical challenge lies not in detecting a visible tumor, but in predicting how it will grow and infiltrate surrounding tissue before the next imaging appointment. Conventional deep learning systems, however capable at segmentation, are agnostic to the underlying tumor biology and therefore cannot provide reliable growth forecasts.</p>

<p>This project proposes HybridTumorNet, an end-to-end trainable architecture that bridges spatial deep learning with biophysical tumor modeling. The system couples a MONAI 3D ResNet encoder with three parallel decoder heads: (1) a tumor density decoder producing a continuous cell-density field u(x,t) ∈ [0,1]; (2) a physics parameter head predicting spatially varying diffusion D(x) and proliferation ρ(x) coefficients; and (3) a segmentation head providing standard BraTS delineation of sub-tumor regions. Training proceeds in two phases: supervised segmentation pre-training on BraTS 2021 data, followed by physics-constrained fine-tuning where the Fisher-KPP reaction-diffusion equation residual is enforced as an additional loss term.</p>

<p>To address the absence of longitudinal training data in BraTS 2021, a numerically stable RK4-based FisherKPPSolver is employed to generate synthetic temporal derivatives (∂u/∂t) from single-timepoint scans. The combined loss function incorporates six terms: data fidelity, PDE residual, initial condition, boundary condition, regularization, and segmentation losses, balanced via learnable uncertainty-weighted parameters following Kendall et al. (2018).</p>

<p>Experimental results on BraTS 2021 demonstrate competitive segmentation performance (Dice-WT: 0.905, Dice-TC: 0.847, Dice-ET: 0.801) while additionally producing patient-specific biophysical parameters (D ≈ 0.018–0.072 mm²/day; ρ ≈ 0.008–0.031 /day) consistent with clinically reported glioma growth kinetics. The PDE residual loss converges from 0.34 to below 0.008 over 30 fine-tuning epochs, confirming genuine physics compliance. The system further supports 30-day forward growth simulations via learned RK4 ODE integration, uncertainty quantification via Monte Carlo Dropout, and NIfTI export of all outputs. This work represents the first architecture to jointly train a MONAI 3D ResNet backbone with a physics-informed loss on a standardized medical imaging benchmark, establishing a reproducible foundation for biologically interpretable brain tumor AI.</p>

<p><b>Keywords:</b> Brain Tumor, Glioma, Physics-Informed Neural Network, MONAI, 3D ResNet, Fisher-KPP, Reaction-Diffusion, BraTS, Tumor Growth Prediction, Medical Image Segmentation.</p>
</div>
"""
