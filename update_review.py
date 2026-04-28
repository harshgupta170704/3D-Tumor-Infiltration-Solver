import re

def update_literature_review():
    source_file = r'c:\Users\Lenovo\OneDrive\Desktop\Fine tuning monia resnet modal with pinn modal\LITERATURE_REVIEW_EXPORTABLE.md'
    target_file = r'c:\Users\Lenovo\OneDrive\Desktop\Fine tuning monia resnet modal with pinn modal\LITERATURE_REVIEW.md'
    
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # INJECTION 1: Classification Methodology
    methodology_text = """

### 1.1 Classification Methodology: Why These 8 Categories?

The field of neuro-oncology AI is vast and rapidly evolving, making a simple chronological timeline insufficient to capture the depth of the research. To properly contextualize the evolution of brain tumor modeling, this review classifies the literature into eight distinct thematic domains. This taxonomy was deliberately chosen to highlight the functional gaps in current research:

*   **Segmentation vs. Prediction (Categories 1 & 2):** Highlights the critical difference between *retrospective* image processing (identifying what is already there) and *prospective* growth modeling (predicting what will happen).
*   **Data-Driven vs. Physics-Driven (Categories 3, 4 vs. 6):** Contrasts purely statistical Deep Learning approaches (like Radiomics and standard CNNs) against Physics-Informed models (PINNs/PDEs) that are constrained by actual biological and physical laws.
*   **Generative vs. Multimodal (Categories 5 & 7):** Distinguishes between synthetic data generation to overcome data scarcity (GANs) and the fusion of real-world multi-parametric sequences (Multimodal Analysis).

By structuring the review this way, we clearly map the historical progression from simple image segmentation toward the ultimate goal of the field: **Hybrid Physics-Informed Neural Networks** that combine the spatial power of modern CNNs with the temporal biophysics of tumor growth.

---
"""

    content = content.replace("## 2. Category 1: Tumor Segmentation", methodology_text + "\n## 2. Category 1: Tumor Segmentation")
    
    # INJECTION 2: Recent Advancements in ResNet + PINNs
    advancement_text = """
### 12.6 Hybrid Spatial-Temporal Frameworks (MONAI 3D ResNet + PINNs)
The most cutting-edge recent advancement (2023-2024) is the hybridization of powerful 3D feature extractors with biophysical mathematical constraints. While traditional models treated segmentation and growth prediction as separate pipelines, modern architectures are fusing them:
*   **MONAI 3D ResNet Backbones:** Researchers are increasingly utilizing Medical Open Network for AI (MONAI) 3D Residual Networks to process multimodal 3D MRI scans. The residual skip-connections allow these networks to extract incredibly deep spatial features without the vanishing gradient problem, producing highly accurate latent representations of the tumor core and surrounding edema.
*   **Biophysical PINN Constraints:** Instead of relying purely on mean-squared error against sparse longitudinal data, these architectures feed the 3D ResNet embeddings directly into a **Physics-Informed Neural Network (PINN)**. The PINN loss function acts as a "soft constraint," penalizing the model if its growth predictions violate the Fisher-KPP reaction-diffusion equations. 
*   **Clinical Impact:** This specific hybrid architecture (ResNet + PINN) represents the frontier of neuro-oncology AI. It bridges the gap between raw pixel data and biological reality, allowing for the extraction of patient-specific diffusion ($D$) and proliferation ($\\rho$) parameters directly from standard imaging, paving the way for biologically consistent, personalized growth simulations.
"""

    content = content.replace("## 13. Research Gaps and Future Directions", advancement_text + "\n## 13. Research Gaps and Future Directions")
    
    # Write the updated content to the main LITERATURE_REVIEW.md file (making it presentable and updated)
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Literature Review successfully updated with Classification Methodology and Recent Advancements!")

if __name__ == "__main__":
    update_literature_review()
