import fitz
import os

pdf_path = r'C:\Users\harsh\OneDrive\Desktop\Cancer-Research\papers\Martens_2022_Deep_Learning_Glioma_Growth_Modeling.pdf'
out_dir = r'C:\Users\harsh\OneDrive\Desktop\Cancer-Research\papers\martens_figures\clean_figures'
os.makedirs(out_dir, exist_ok=True)

doc = fitz.open(pdf_path)

# Precise coordinates for each figure based on text block layout:
# Each entry: (page_index_0_based, y0, y1, filename)
crops = {
    'figure_1': (4, 530, 725),   # Page 5: Processed MR data
    'figure_2': (6, 115, 450),   # Page 7: DTI data processing
    'figure_3': (7, 270, 725),   # Page 8: Parameter sampling distributions
    'figure_4': (8, 140, 500),   # Page 9: Synthetic tumor growth over time t1-t4
    'figure_5': (10, 45, 305),   # Page 11: 3D U-Net for Cell Density Estimation
    'figure_6': (11, 45, 345),   # Page 12: Conv-Encoder for Parameter Estimation
    'figure_7': (13, 45, 270),   # Page 14 top: MAE boxplots
    'figure_8': (13, 275, 580),  # Page 14 bottom: Hausdorff & ASSD boxplots
    'figure_9': (14, 45, 620),   # Page 15: True vs Pred density & error maps
    'figure_10': (15, 45, 290),  # Page 16 top: Relative error boxplots (lambda, v, D, rho)
    'figure_11': (15, 295, 595), # Page 16 bottom: Scatterplots true vs predicted
    'figure_12': (16, 45, 420),  # Page 17: Real GBM Patient scan (T1Gd, FLAIR, Cell density)
}

for fig_id, (p_idx, y0, y1) in crops.items():
    page = doc[p_idx]
    rect = fitz.Rect(35, y0, page.rect.width - 35, y1)
    pix = page.get_pixmap(clip=rect, dpi=250) # High quality 250 DPI
    dest = os.path.join(out_dir, f'{fig_id}.png')
    pix.save(dest)
    print(f'Rendered {fig_id}.png at {pix.width}x{pix.height} (Page {p_idx+1})')

print('All 12 primary figures cleanly rendered at 250 DPI!')
