from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image
import os

FIG = r"C:\Users\harsh\OneDrive\Desktop\Cancer-Research\papers\martens_figures\clean_figures"
OUT = r"C:\Users\harsh\OneDrive\Desktop\Cancer-Research\Martens_2022_Deep_Dive_Presentation.pptx"

BG = RGBColor(11, 18, 32)
PANEL = RGBColor(22, 32, 52)
WHITE = RGBColor(255, 255, 255)
GREY = RGBColor(170, 184, 204)
CYAN = RGBColor(34, 211, 238)
PINK = RGBColor(244, 114, 182)
GREEN = RGBColor(52, 211, 153)
AMBER = RGBColor(251, 191, 36)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
total = [0]


def rect(s, x, y, w, h, color, shape=MSO_SHAPE.RECTANGLE):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid()
    r.fill.fore_color.rgb = color
    r.line.fill.background()
    return r


def text(s, x, y, w, h, t, size=16, color=WHITE, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = t
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = "Calibri"
    return tb


def bullets(s, x, y, w, h, items, size=15, color=WHITE, accent=CYAN):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(9)
        r0 = p.add_run()
        r0.text = "\u25B8 "
        r0.font.color.rgb = accent
        r0.font.size = Pt(size)
        r = p.add_run()
        r.text = it
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.name = "Calibri"
    return tb


def base(title, tag, accent=CYAN):
    s = prs.slides.add_slide(BLANK)
    total[0] += 1
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    rect(s, 0, 0, 0.18, 7.5, accent)
    text(s, 0.5, 0.25, 11.5, 0.4, tag.upper(), 12, accent, True)
    text(s, 0.5, 0.55, 12.3, 0.8, title, 28, WHITE, True)
    text(s, 12.3, 7.05, 0.8, 0.3, str(total[0]), 11, GREY, align=PP_ALIGN.RIGHT)
    text(s, 0.5, 7.05, 9, 0.3, "Martens et al. (2022) | HybridTumorNet | IIIT Bhopal", 10, GREY)
    return s


def figure_slide(n, title, tag, points, caption, accent=CYAN):
    s = base(title, tag, accent)
    path = os.path.join(FIG, "figure_%d.png" % n)
    iw, ih = Image.open(path).size
    bx, by, bw, bh = 0.5, 1.5, 8.3, 5.4
    rect(s, bx, by, bw, bh, WHITE, MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.02
    pad = 0.15
    mw, mh = bw - 2 * pad, bh - 2 * pad
    sc = min(mw / iw, mh / ih)
    w, h = iw * sc, ih * sc
    s.shapes.add_picture(path, Inches(bx + (bw - w) / 2), Inches(by + (bh - h) / 2), Inches(w), Inches(h))
    rect(s, 9.05, 1.5, 3.8, 5.4, PANEL, MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.03
    text(s, 9.25, 1.6, 3.4, 0.4, "Figure %d \u2013 Samajh kya aaya" % n, 15, accent, True)
    bullets(s, 9.25, 2.05, 3.45, 3.9, points, 13, WHITE, accent)
    text(s, 9.25, 5.95, 3.4, 0.9, caption, 11, GREY)
    return s


def text_slide(title, tag, items, accent=CYAN, right=None):
    s = base(title, tag, accent)
    w = 7.6 if right else 12.2
    bullets(s, 0.6, 1.6, w, 5.2, items, 18, WHITE, accent)
    if right:
        rect(s, 8.5, 1.6, 4.35, 5.0, PANEL, MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.04
        text(s, 8.75, 1.75, 3.9, 0.4, right[0], 16, accent, True)
        bullets(s, 8.75, 2.25, 3.9, 4.2, right[1], 14, WHITE, accent)
    return s


# 1 Title
s = prs.slides.add_slide(BLANK)
total[0] += 1
s.background.fill.solid()
s.background.fill.fore_color.rgb = BG
rect(s, 0, 0, 13.333, 0.25, CYAN)
rect(s, 0.8, 2.0, 0.12, 2.2, PINK)
text(s, 1.2, 1.9, 11, 0.5, "PAPER DEEP DIVE  |  LITERATURE REVIEW", 14, CYAN, True)
text(s, 1.2, 2.4, 11.5, 1.6, "Deep Learning for Fast Glioma Growth Modeling", 42, WHITE, True)
text(s, 1.2, 3.7, 11, 0.6, "Martens et al. (2022) \u2013 Parameter estimation of the Fisher-KPP model using 3D U-Net & Conv-Encoder", 18, GREY)
text(s, 1.2, 4.9, 11, 0.4, "Harsh Gupta (23U01011)  &  Mayank Kumar Verma (23U01026)", 18, WHITE, True)
text(s, 1.2, 5.3, 11, 0.4, "Supervisor: Dr. Kamlesh Kumar Chauhan  |  Dept. of ECE, IIIT Bhopal", 15, GREY)
text(s, 1.2, 5.9, 11, 0.4, "Project: HybridTumorNet \u2013 Hybrid DL + PINN for Brain Tumor Growth Prediction", 15, PINK, True)

# 2 Agenda
text_slide("Aaj hum kya discuss karenge", "Roadmap", [
    "Problem: GBM ka growth model karna itna slow kyun hai?",
    "Math: Fisher-KPP equation \u2013 D aur \u03C1 kya batate hain",
    "Synthetic data: brain anatomy + DTI + sampling (Fig 1\u20134)",
    "Do neural networks: 3D U-Net (Task 1) aur Conv-Encoder (Task 2) (Fig 5\u20136)",
    "Results: MAE, ASSD, Hausdorff, CCC (Fig 7\u201311)",
    "Real GBM patient par test (Fig 12)",
    "Hamare HybridTumorNet se relation aur next steps",
])

# 3 Problem
text_slide("Clinical problem: GBM ko model karna slow hai", "Motivation", [
    "Glioblastoma (GBM) sabse aggressive brain tumor hai \u2013 median survival ~15 months.",
    "MRI me sirf dikhta hai tumor ka visible hissa; infiltrating cells (jo MRI me nahi dikhte) radiotherapy margin ke liye important hain.",
    "Fisher-KPP PDE se tumor growth simulate hota hai, par patient-specific D aur \u03C1 nikalna iterative optimization maangta hai \u2013 ghanton ka kaam.",
    "Martens ka idea: ek neural network train karo jo seconds me D aur \u03C1 de de.",
], PINK, ("Key numbers", [
    "Classical solver: hours per patient",
    "Neural network: < 1 second",
    "Training data: synthetic Fisher-KPP simulations",
    "Test: synthetic + 1 real GBM",
]))

# 4 Math
text_slide("Fisher-KPP model: D aur \u03C1", "Mathematics", [
    "\u2202u/\u2202t = \u2207\u00B7(D \u2207u) + \u03C1 u (1 \u2212 u)",
    "u(x,t): tumor cell density (0 se 1). D: diffusion (cells kitni tezi se failte hain). \u03C1: proliferation rate (kitni tezi se badhte hain).",
    "Grey matter me D chhota, white matter me D bada \u2013 isliye DTI tensor se anisotropic diffusion lete hain.",
    "Inverse problem: u ke snapshots se (D, \u03C1) wapas nikalna \u2013 yahi network ka kaam hai.",
    "Hamari HybridTumorNet me bhi yehi PDE loss ke roop me use hota hai (RK4 integration).",
], CYAN, ("Intuition", [
    "D \u2191  \u2192  tumor fail jata hai",
    "\u03C1 \u2191  \u2192  density jaldi saturate",
    "Dono alag-alag identify karna mushkil \u2013 ill-posed",
]))

# Figures
figure_slide(1, "Brain tissue segmentation: anatomy ka base", "Synthetic data pipeline", [
    "Real brain MRI (T1) se White Matter, Grey Matter, CSF alag kiye gaye.",
    "Tumor sirf brain tissue me grow kar sakta hai, CSF me nahi \u2013 isliye mask zaroori.",
    "Ye mask Fisher-KPP simulation ka domain banta hai.",
], "Paper Fig. 1: Tissue segmentation of the brain used for simulations.")
figure_slide(2, "DTI tensor processing: white-matter tracts", "Synthetic data pipeline", [
    "Diffusion Tensor Imaging se har voxel ka diffusion tensor milta hai.",
    "Tensor ko tumor-diffusion tensor me convert karte hain \u2013 cells fibre direction me jaldi fail hoti hain.",
    "Isse simulation realistic ho jata hai.",
], "Paper Fig. 2: Processing of DTI for anisotropic tumor diffusion.", PINK)
figure_slide(3, "Parameter sampling: D, \u03C1 aur tumor start point", "Synthetic data pipeline", [
    "Har simulation ke liye D, \u03C1 aur initial tumor location random sample hoti hai.",
    "Distributions biologically plausible ranges se li gayi hain.",
    "Isse network ko wide variety of tumors dekhne ko milti hai.",
], "Paper Fig. 3: Distributions used to sample simulation parameters.", GREEN)
figure_slide(4, "Synthetic tumor evolution: t1 se t4 tak", "Synthetic data pipeline", [
    "Ek hi parameter set se 4 timepoints ka tumor density evolve hota hai.",
    "Time ke saath tumor boundary diffuse hoti jati hai.",
    "Ye sequences network ke training input/target bante hain.",
], "Paper Fig. 4: Tumor evolution at four timepoints of a simulation.", AMBER)
figure_slide(5, "Task 1: 3D U-Net \u2013 tumor density predict karna", "Network architecture", [
    "Input: 2 time ke tumor snapshots + anatomy; output: future tumor density.",
    "Encoder\u2013decoder with skip connections, 3D convolutions.",
    "Task 1 me network seedha density map predict karta hai.",
    "Hamari segmentation backbone ka concept isse milta hai.",
], "Paper Fig. 5: 3D U-Net used for tumor density prediction.", CYAN)
figure_slide(6, "Task 2: Conv-Encoder \u2013 D aur \u03C1 regress karna", "Network architecture", [
    "Convolutional encoder volume ko compress karke 2 numbers deta hai: D aur \u03C1.",
    "Output head fully-connected layers se.",
    "Yehi part HybridTumorNet ke PINN parameter head jaisa hai.",
], "Paper Fig. 6: Conv-Encoder for estimating D and \u03C1.", PINK)
figure_slide(7, "Results: Mean Absolute Error boxplots", "Results", [
    "Voxel-wise MAE of predicted vs true density.",
    "Median error bahut kam \u2013 network accurate hai.",
    "Outliers kuch cases me, mostly low-signal tumors.",
], "Paper Fig. 7: MAE of predictions across test set.", GREEN)
figure_slide(8, "Results: ASSD aur Hausdorff distance", "Results", [
    "ASSD average surface distance \u2013 boundary kitni sahi hai.",
    "Hausdorff worst-case boundary error batata hai.",
    "Sub-voxel level ke errors \u2013 clinically useful precision.",
], "Paper Fig. 8: ASSD and Hausdorff distance.", AMBER)
figure_slide(9, "True vs predicted tumor density (forward simulation)", "Results", [
    "Row-wise: ground truth, prediction, difference.",
    "Predicted tumor ground truth se visually almost identical.",
    "Difference maps me error sirf boundary par thoda dikhta hai.",
], "Paper Fig. 9: Comparison of true and predicted tumor density.", CYAN)
figure_slide(10, "Relative error on D aur \u03C1", "Parameter accuracy", [
    "Network se nikle D aur \u03C1 ka relative error plot.",
    "Dono parameters alag-alag precisely identify nahi hote \u2013 par product-like combination stable hai.",
    "Ye ill-posedness ka asli evidence hai.",
], "Paper Fig. 10: Relative errors for estimated D and \u03C1.", PINK)
figure_slide(11, "Scatterplots aur CCC (agreement)", "Parameter accuracy", [
    "True vs predicted D aur \u03C1 ka scatter.",
    "Concordance Correlation Coefficient high \u2013 strong agreement.",
    "Points diagonal ke paas \u2013 bias kam hai.",
], "Paper Fig. 11: True vs. predicted parameters with CCC.", GREEN)
figure_slide(12, "Real GBM patient par test", "Clinical validation", [
    "Ek real patient ke MRI se tumor segment karke network ko diya gaya.",
    "Network ne seconds me D, \u03C1 nikale aur growth simulate kiya.",
    "Sirf ek patient hai \u2013 isliye paper khud bolta hai ki aur validation chahiye.",
], "Paper Fig. 12: Application to a real glioblastoma patient.", AMBER)

# Limitations
text_slide("Paper ki limitations (honest view)", "Critical analysis", [
    "Training poori tarah synthetic data par \u2013 real tumors Fisher-KPP se alag behave kar sakte hain.",
    "Real validation sirf ek patient \u2013 statistical claim nahi ban sakta.",
    "D aur \u03C1 identifiability weak \u2013 ek snapshot se dono alag nikalna ill-posed.",
    "Treatment effect (surgery, chemo) model nahi hota.",
    "Fisher-KPP simplified model hai \u2013 mass effect, necrosis ignore.",
], PINK)

# Relevance
text_slide("Hamare HybridTumorNet se kya relation hai", "Relevance to our project", [
    "Martens: network D, \u03C1 predict karta hai \u2192 HybridTumorNet bhi PDE parameters predict karta hai.",
    "Martens: synthetic Fisher-KPP data \u2192 hamare paas bhi RK4-based FisherKPPSolver hai.",
    "Difference: hum segmentation + PINN physics loss end-to-end combine karte hain.",
    "Difference: hamare paas real longitudinal LUMIERE data hai (91 GBM patients) \u2013 Martens ke paas sirf 1 real patient.",
    "Isliye hamara contribution: real longitudinal validation.",
], GREEN, ("Comparison", [
    "Martens: 3D U-Net + Conv-Encoder",
    "Ours: ResNet-10 + PINN loss",
    "Martens: 1 real patient",
    "Ours: LUMIERE 91 patients",
]))

# Our results
text_slide("Hamare abhi tak ke results", "Our evaluation on LUMIERE", [
    "MONAI SegResNet (SOTA) LUMIERE par: avg Whole-Tumor Dice 89.7%, median 93.1%, peak 96.1%.",
    "Hamara ResNet-10 HybridTumorNet pretrained: avg WT Dice ~49.2% (10 patients), peak 70%.",
    "Gap ka reason: chhota backbone + synthetic pretraining; fine-tuning on real data pending.",
    "Pipeline ne coordinate mismatch bug fix karke atlas-space data par evaluate kiya.",
], CYAN, ("Takeaway", [
    "SOTA benchmark ready",
    "Gap clearly measured",
    "Next: fine-tune on LUMIERE",
]))

# Next steps
text_slide("Next steps & conclusion", "Roadmap", [
    "Backbone upgrade: SegResNet / SwinUNETR ko HybridTumorNet me use karna.",
    "LUMIERE longitudinal pairs par growth-prediction fine-tuning (t_i \u2192 t_{i+1}).",
    "Metrics: Dice, ASSD, Hausdorff aur parameter CCC \u2013 Martens ke jaise.",
    "Uncertainty quantification (MC dropout / ensembles).",
    "Conclusion: Martens ne fast parameter estimation ka proof of concept diya; hum usse real longitudinal data par extend kar rahe hain.",
], GREEN)

prs.save(OUT)
print("saved", OUT, total[0], "slides")
