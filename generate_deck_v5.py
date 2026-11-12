from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image
import os

FIG = r"C:\Users\harsh\OneDrive\Desktop\Cancer-Research\papers\martens_figures\clean_figures"
OUT = r"C:\Users\harsh\OneDrive\Desktop\Cancer-Research\Martens_2022_Formal_Presentation.pptx"

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
cnt = [0]


def rect(s, x, y, w, h, color, shape=MSO_SHAPE.RECTANGLE, adj=None):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid()
    r.fill.fore_color.rgb = color
    r.line.fill.background()
    if adj is not None:
        r.adjustments[0] = adj
    return r


def text(s, x, y, w, h, t, size=16, color=WHITE, bold=False, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = t
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = "Calibri"
    return tb


def bullets(s, x, y, w, h, items, size=15, accent=CYAN, gap=7):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        if it.startswith("##"):
            r = p.add_run()
            r.text = it[2:].strip()
            r.font.bold = True
            r.font.size = Pt(size + 1)
            r.font.color.rgb = accent
            r.font.name = "Calibri"
            continue
        r0 = p.add_run()
        r0.text = "\u25B8 "
        r0.font.color.rgb = accent
        r0.font.size = Pt(size)
        r = p.add_run()
        r.text = it
        r.font.size = Pt(size)
        r.font.color.rgb = WHITE
        r.font.name = "Calibri"
    return tb


def base(title, tag, accent=CYAN):
    s = prs.slides.add_slide(BLANK)
    cnt[0] += 1
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    rect(s, 0, 0, 0.18, 7.5, accent)
    text(s, 0.5, 0.22, 11.5, 0.4, tag.upper(), 12, accent, True)
    text(s, 0.5, 0.5, 12.3, 0.8, title, 27, WHITE, True)
    text(s, 12.2, 7.07, 0.9, 0.3, str(cnt[0]), 11, GREY, align=PP_ALIGN.RIGHT)
    text(s, 0.5, 7.07, 9, 0.3, "Martens et al. (2022) | HybridTumorNet | IIIT Bhopal", 10, GREY)
    return s


def text_slide(title, tag, items, accent=CYAN, right=None, size=17):
    s = base(title, tag, accent)
    w = 7.7 if right else 12.2
    bullets(s, 0.6, 1.5, w, 5.4, items, size, accent)
    if right:
        rect(s, 8.55, 1.5, 4.3, 5.3, PANEL, MSO_SHAPE.ROUNDED_RECTANGLE, 0.03)
        if isinstance(right, tuple):
            right = ["##" + right[0]] + list(right[1])
        bullets(s, 8.75, 1.65, 3.95, 5.0, right, 14, accent, 6)
    return s


def table_slide(title, tag, header, rows, accent=CYAN, note=None, colw=None):
    s = base(title, tag, accent)
    nr, nc = len(rows) + 1, len(header)
    gt = s.shapes.add_table(nr, nc, Inches(0.6), Inches(1.5), Inches(12.1), Inches(0.5 * nr))
    t = gt.table
    for j, h in enumerate(header):
        c = t.cell(0, j)
        c.fill.solid()
        c.fill.fore_color.rgb = accent
        c.text = h
        for p in c.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(15)
                r.font.bold = True
                r.font.color.rgb = BG
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i + 1, j)
            c.fill.solid()
            c.fill.fore_color.rgb = PANEL if i % 2 == 0 else RGBColor(30, 42, 66)
            c.text = v
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(14)
                    r.font.color.rgb = WHITE
    if colw:
        for j, w in enumerate(colw):
            t.columns[j].width = Inches(w)
    if note:
        bullets(s, 0.6, 1.7 + 0.5 * nr + 0.2, 12.0, 2.0, note, 15, accent)
    return s


def figure_slide(n, title, tag, sections, caption, accent=CYAN):
    s = base(title, tag, accent)
    path = os.path.join(FIG, "figure_%d.png" % n)
    iw, ih = Image.open(path).size
    bx, by, bw, bh = 0.5, 1.4, 7.4, 5.55
    rect(s, bx, by, bw, bh, WHITE, MSO_SHAPE.ROUNDED_RECTANGLE, 0.02)
    pad = 0.12
    sc = min((bw - 2 * pad) / iw, (bh - 2 * pad) / ih)
    w, h = iw * sc, ih * sc
    s.shapes.add_picture(path, Inches(bx + (bw - w) / 2), Inches(by + (bh - h) / 2), Inches(w), Inches(h))
    rect(s, 8.1, 1.4, 4.8, 5.55, PANEL, MSO_SHAPE.ROUNDED_RECTANGLE, 0.03)
    items = []
    for head, its in sections:
        items.append("##" + head)
        items.extend(its)
    bullets(s, 8.25, 1.48, 4.5, 5.0, items, 12, accent, 4)
    text(s, 8.25, 6.55, 4.5, 0.4, caption, 9, GREY)
    return s


# ---------------- CONTENT (English + light Hinglish) ----------------
s = prs.slides.add_slide(BLANK)
cnt[0] += 1
s.background.fill.solid()
s.background.fill.fore_color.rgb = BG
rect(s, 0, 0, 13.333, 0.25, CYAN)
rect(s, 0.8, 2.0, 0.12, 2.3, PINK)
text(s, 1.2, 1.9, 11, 0.5, "PAPER DEEP DIVE  |  LITERATURE REVIEW", 14, CYAN, True)
text(s, 1.2, 2.4, 11.5, 1.6, "Deep Learning for Fast Glioma Growth Modeling", 42, WHITE, True)
text(s, 1.2, 3.75, 11, 0.7, "Martens et al. (2022): estimating tumor cell density and Fisher-KPP parameters (D, ρ) with 3D CNNs", 18, GREY)
text(s, 1.2, 4.9, 11, 0.4, "Harsh Gupta (23U01011)  &  Mayank Kumar Verma (23U01026)", 18, WHITE, True)
text(s, 1.2, 5.3, 11, 0.4, "Supervisor: Dr. Kamlesh Kumar Chauhan  |  Dept. of ECE, IIIT Bhopal", 15, GREY)
text(s, 1.2, 5.9, 11, 0.4, "Project: HybridTumorNet – Hybrid DL + PINN for Brain Tumor Growth Prediction", 15, PINK, True)

text_slide("Presentation Outline", "Roadmap", [
    "##Part A – Basics (slides 3–6): MRI, tumor, PDE, CNN. Recommended starting point for first-time readers.",
    "##Part B – The idea (slides 7–9): the problem, the solution, and the overall pipeline.",
    "##Part C – Data synthesis (slides 10–14): Figures 1–4 and Table 1.",
    "##Part D – Networks (slides 15–18): Figures 5–6 and training details.",
    "##Part E – Results (slides 19–25): Figures 7–12, with the actual numbers.",
    "##Part F – Critical analysis (slides 26–32): limitations, relation to our project, glossary, teacher Q&A.",
    "On every figure slide: the original figure on the left, a panel-by-panel explanation on the right.",
], CYAN, size=16)

text_slide("Basics 1: Brain tumor and MRI", "Part A | Basics", [
    "##What is a glioma / glioblastoma (GBM)?",
    "A tumor of the brain's glial cells. GBM is the most aggressive type. It does not form a compact mass; it infiltrates surrounding tissue along tentacle-like extensions.",
    "##What does MRI show?",
    "MRI shows only the region where tumor cell density is high. Regions with lower cell density appear normal on the scan, although tumor cells are present there.",
    "##Why is that a problem?",
    "If surgery or radiotherapy targets only the visible tumor, the hidden cells regrow it. So we need a map of the invisible cells.",
], PINK, ("Key terms", [
    "T1Gd: contrast MRI, shows the enhancing core",
    "T2 FLAIR: shows the oedema (swelling) region",
    "DTI: MRI that measures nerve-fibre direction",
    "Voxel: a 3D pixel (here 1 mm³)",
]))

text_slide("Basics 2: PDE and the Fisher-KPP model", "Part A | Basics", [
    "##A PDE is a rule that says how a quantity changes over space and time.",
    "##Fisher-KPP:  ∂u/∂t = ∇·(D∇u) + ρu(1−u)",
    "u = tumor cell density (0 to 1). First term = spreading (diffusion), second term = growth (proliferation).",
    "D = diffusion: how fast cells spread. ρ = proliferation: how fast cells divide.",
    "u(1−u): when u is small, growth is fast; when u→1 (voxel full), growth stops. This is called saturation.",
    "##Analogy: ink spreading in water (diffusion) while the ink also multiplies itself (reaction).",
], CYAN, ("Forward vs Inverse", [
    "Forward: D, ρ and the starting tumor are known → simulate the future (easy).",
    "Inverse: we only have the MRI → find D, ρ and the hidden cells (hard).",
    "This paper solves the inverse problem fast.",
]))

text_slide("Basics 3: Front width and speed (λ, v)", "Part A | Basics", [
    "A Fisher-KPP tumor moves like a travelling wave: the front moves at a constant speed and its edge decays exponentially.",
    "##Infiltration depth: λ = √(D/ρ) (mm). It describes how wide or diffuse the tumor front is.",
    "##Front velocity: v = 2√(Dρ) (mm/year). It describes how fast the tumor boundary advances.",
    "Inverting these: D = λv/2 and ρ = v/(2λ) (paper's Eq. 12 and 13).",
    "So the network predicts λ and v (easy to read from images), and then we compute D and ρ from them.",
], GREEN, ("How to measure", [
    "λ: from the distance between two contours",
    "v: from how far one contour shifts between two times",
]))

text_slide("Basics 4: CNN and U-Net", "Part A | Basics", [
    "##CNN: a neural network that learns patterns (edges, shapes) from images by itself. A 3D CNN works on a full 3D volume.",
    "##U-Net: encoder (shrink the image to get context) + decoder (grow it back to get the output) + skip connections (to preserve fine details).",
    "##Convolutional encoder: only the encoder part plus a final fully-connected layer. It turns an image into a few numbers (regression).",
    "##Loss: MAE (mean |error|) or MSE (mean error²). The Adam optimizer updates the weights.",
    "##Training data is synthetic, because nobody knows the true cell density of a real tumor.",
], PINK)

text_slide("The paper's problem and contribution", "Part B | Idea", [
    "##Problem: reaction-diffusion models have existed for decades, but they are not used on patients. Two reasons: (1) the initial condition (hidden cells) and (2) estimating D, ρ.",
    "##Contribution: two deep CNNs that solve both problems in seconds.",
    "Network 1 (3D U-Net): contours → full cell density map.",
    "Network 2 (conv-encoder): contours + time gap → λ, v → D, ρ.",
    "Then the model simulates the future tumor (90 and 180 days ahead).",
    "Training data: 1,200 synthetic tumors grown on the real brains of 6 healthy volunteers.",
], CYAN, ("In one line", [
    "Visible MRI contours → CNN → hidden-cell map + D, ρ → future growth",
]))

text_slide("Pipeline at a glance", "Part B | Idea", [
    "##Step 1 – Data: MRI + DTI of healthy volunteers → brain map and tumor diffusion tensor (Fig 1, 2).",
    "##Step 2 – Simulate: random λ, v, seed and time gaps → grow a tumor with Fisher-KPP (Fig 3, 4).",
    "##Step 3 – Contours: threshold the density at t₂ (c₁=0.80, c₂=0.16) and at t₁ (c₂).",
    "##Step 4 – Train: Network 1 and Network 2, separately (Fig 5, 6).",
    "##Step 5 – Validate: estimate density at t₂, simulate t₃ and t₄, compare with the truth (Fig 7–11).",
    "##Step 6 – Real data: apply Network 1 to one real GBM patient (Fig 12).",
], GREEN)

text_slide("Imaging contours and thresholds (c₁, c₂)", "Part B | Idea", [
    "MRI does not show the exact density. It shows a boundary: voxels with density above a threshold appear 'inside'.",
    "##c₁ = 0.80: the enhancing core on T1Gd (dense tumor).",
    "##c₂ = 0.16: the oedema outline on T2 FLAIR (outer boundary).",
    "These thresholds come from Swanson et al. (2008), the paper's ref [5].",
    "In the synthetic data, the same thresholds turn the density into binary masks, and those masks are the network input.",
    "Caveat: oedema is not truly an invasion zone, so the c₂ assumption is approximate (the paper says this in its discussion).",
], AMBER, ("Terms", [
    "Γ₁: contour at c=0.80, time t₂",
    "Γ₂: contour at c=0.16, time t₂",
    "Γ₃: contour at c=0.16, time t₁",
]))

figure_slide(1, "Figure 1: Splitting the brain into tissue types", "Part C | Data synthesis", [
    ("Panels", [
        "(a) T2 FLAIR slice with the red brain mask: the simulation runs only inside this region.",
        "(b) T1 BRAVO image of the same slice: clear anatomy, the skull is visible too.",
        "(c) Segmented map: white matter, grey matter and CSF get separate labels.",
    ]),
    ("How it was made", [
        "MICO algorithm (removes the brightness bias, then clusters) + manual correction.",
    ]),
    ("Why it matters", [
        "Tumor cannot spread into CSF, and WM and GM have different D.",
        "This map is used to build the diffusion tensor of Figure 2.",
    ]),
], "Paper Fig. 1: processed MR data.")

figure_slide(2, "Figure 2: From DTI to tumor diffusion tensor", "Part C | Data synthesis", [
    ("What is the grid?", [
        "The 3×3 grid shows the 9 tensor components d(i,j). It is symmetric, so only 6 are independent.",
    ]),
    ("(a) Water tensor", [
        "Real DTI scan, corrected for distortion, eddy currents and motion with FSL. Ventricles look bright.",
    ]),
    ("(b) Tumor tensor", [
        "Built from (a) + the brain map: anisotropy a=10 in WM, d_grey/d_white = 0.1, CSF removed.",
    ]),
    ("Difference", [
        "(a) = how water diffuses. (b) = how tumor cells diffuse. The simulation uses (b).",
    ]),
], "Paper Fig. 2: processed DTI data.", PINK)

table_slide("Table 1: Parameter ranges for synthetic tumors", "Part C | Data synthesis",
            ["Parameter", "Min", "Max", "Unit", "Meaning"], [
    ["λ (infiltration depth)", "0.5", "2.0", "mm", "how wide the front is"],
    ["v (front velocity)", "28.28", "48.99", "mm/year", "how fast the boundary moves"],
    ["Δt₁ (t₀ to t₁)", "90", "180", "days", "time until the first scan"],
    ["Δt₂ (t₁ to t₂)", "90", "180", "days", "gap between the two scans"],
], GREEN, [
    "All values are sampled uniformly at random. Δt₃ = Δt₄ = 90 days are fixed for validation.",
    "D = λv/2 and ρ = v/(2λ). Sampling D and ρ independently would give tumors that are too small or too big, which is why λ and v are sampled instead.",
    "Seed: a 3×3×3 voxel block in white matter with c=1. Total: 200 tumors × 6 volunteers = 1,200.",
], [3.9, 1.3, 1.3, 1.7, 3.9])

figure_slide(3, "Figure 3: How the parameters were sampled", "Part C | Data synthesis", [
    ("(a) (λ, v) scatter", [
        "Each blue dot is one tumor. λ in 0.5–2 mm, v in 28–49 mm/yr, uniform. Red box = boundary.",
    ]),
    ("(b) (d_white, ρ) scatter", [
        "The same points mapped to D, ρ with Eq. 12, 13. The rectangle becomes a curved shape (non-linear map).",
    ]),
    ("(c) d_white histogram", [
        "Blue = sampled, red = theory. d_white spans about 7–49 mm²/yr.",
    ]),
    ("(d) ρ histogram", [
        "Right-skewed: peak around 10–12 /yr with a long tail. Most tumors are slow-growing, while a few are very aggressive.",
    ]),
    ("Takeaway", ["The network sees slow to aggressive tumors alike."]),
], "Paper Fig. 3: sampling of model parameters.", GREEN)

figure_slide(4, "Figure 4: Synthetic tumor evolution (t₁ to t₄)", "Part C | Data synthesis", [
    ("What is shown", [
        "3 tumors (rows) at 4 time points t₁–t₄ (columns), axial slices. Colour = cell density 0.01 to 1.",
    ]),
    ("Table 2 values", [
        "T1: d=10.87, ρ=31.77, λ=1.71, v=37.16",
        "T2: d=15.07, ρ=13.95, λ=0.96, v=29.00",
        "T3: d=41.49, ρ=11.31, λ=0.52, v=43.33",
    ]),
    ("Observation", [
        "Small λ gives a sharp front, large λ a blurry one. The tumor grows with time.",
        "Solver: forward Euler, Δt=0.5 day, about 1 s per tumor on a GPU.",
    ]),
], "Paper Fig. 4: simulated cell density at t₁–t₄.", AMBER)

figure_slide(5, "Figure 5: Network 1 – 3D U-Net (cell density)", "Part D | Networks", [
    ("Input (8 channels)", [
        "2 binary contour masks (Γ₁, Γ₂) + 6 unit tensor components. Size 192×192×128.",
    ]),
    ("Output", ["A cell density map of the same size: u at every voxel."]),
    ("Structure", [
        "4 down blocks, 4 up blocks, 1 output block. Features 32→64→128 at the bottom.",
        "Grey boxes = skip connections (concatenation).",
        "Arrows: Conv3×3, Conv2×2 stride 2 (shrink), Deconv2×2 (grow), Conv1×1 (final).",
    ]),
    ("Goal", ["Turn visible contours into a map of invisible tumor cells."]),
], "Paper Fig. 5: 3D U-Net.", CYAN)

figure_slide(6, "Figure 6: Network 2 – Conv-Encoder (λ and v)", "Part D | Networks", [
    ("Input (9 channels + Δt₂)", [
        "3 contours (Γ₁, Γ₂ at t₂; Γ₃ at t₁) + 6 tensor components, plus the time gap Δt₂.",
    ]),
    ("Structure", [
        "6 down-sampling blocks → flatten (144 features) + Δt₂ → 1 fully connected layer.",
    ]),
    ("Output", ["Two numbers: λ and v. Then Eq. 12, 13 give d_white and ρ."]),
    ("Why 3 contours?", [
        "λ comes from the distance between two contours, v from the shift of one contour over two times.",
    ]),
], "Paper Fig. 6: convolutional regressor.", PINK)

table_slide("Training details of both networks", "Part D | Networks",
            ["", "Network 1 (U-Net)", "Network 2 (Conv-Encoder)"], [
    ["Task", "Cell density estimation", "λ, v estimation"],
    ["Input", "192×192×128×8", "192×192×128×9 + Δt₂"],
    ["Loss", "MAE", "MSE (targets standardised)"],
    ["Optimizer", "Adam, lr 1e-4, batch 1", "Adam, lr 1e-4"],
    ["Augmentation", "Random shifts ±15 voxels", "Random shifts ±15 voxels"],
    ["Best test loss", "MAE 1.24e-4 (epoch 876)", "MSE 6.75e-2 (epoch 628)"],
    ["Split", "5 volunteers train / 1 test (83% / 17%)", "same"],
], PINK, ["Framework: TensorFlow 2.5.0, GPU: RTX 3090. Early stopping after 100 epochs without improvement. No rotation augmentation, because the tensor components would also need rotating."],
            [2.4, 4.9, 4.8])

text_slide("Validation protocol and post-processing", "Part D | Networks", [
    "##Experiment: Network 1 gives the density at t₂, Network 2 gives d_white and ρ. Then Fisher-KPP simulates t₃ (t₂+90 d) and t₄ (t₂+180 d).",
    "##Compared with the truth (true density + true parameters): MAE only where c>0.01, plus Hausdorff distance and ASSD between contours.",
    "##Hausdorff = the worst boundary error (max). ASSD = the average boundary error.",
    "##Post-processing: (1) CSF/background density = 0. (2) clip density above 1 to 1. (3) remove small maxima outside the largest connected region.",
    "Reason: tiny network artefacts would otherwise grow into fake new tumor foci during simulation.",
], AMBER, ("Test set", [
    "200 tumors on one unseen volunteer brain.",
    "Thresholds: c₁=0.80, c₂=0.16",
]))

figure_slide(7, "Figure 7: Density error (MAE) boxplot", "Part E | Results", [
    ("Axes", [
        "Y: mean absolute error [10⁻¹], X: t₂, t₃ (+90 d), t₄ (+180 d).",
        "Box = middle 50%, line = median, whiskers = 1.5×IQR, dots = outliers.",
    ]),
    ("Medians", [
        "t₂: 9.58×10⁻³  |  t₃: 1.38×10⁻²  |  t₄: 2.20×10⁻²",
    ]),
    ("Takeaway", [
        "The further ahead we predict, the larger the error (errors accumulate).",
        "Still, the median error stays below 0.03.",
    ]),
], "Paper Fig. 7: MAE within c>0.01.", GREEN)

figure_slide(8, "Figure 8: Hausdorff distance and ASSD", "Part E | Results", [
    ("Panels", [
        "(a) Hausdorff, c₁=0.80   (b) Hausdorff, c₂=0.16",
        "(c) ASSD, c₁=0.80   (d) ASSD, c₂=0.16. Y axis in mm.",
    ]),
    ("What it tells us", [
        "How far the predicted contour is from the true one. Hausdorff = worst point, ASSD = average.",
    ]),
    ("Authors' claim", [
        "ASSD is ≤ 1 mm (the voxel size) in most cases.",
        "A few outliers have large Hausdorff but small ASSD: caused by artefact foci.",
    ]),
], "Paper Fig. 8: Hausdorff and ASSD.", AMBER)

figure_slide(9, "Figure 9: True vs predicted density (one example)", "Part E | Results", [
    ("Rows", [
        "1: target density, 2: predicted density, 3: absolute error (0–0.10).",
        "4 and 5: contours on T1 and T2 FLAIR. Blue = target, red = prediction, green = overlap.",
    ]),
    ("Columns", ["t₂, t₃, t₄ (axial slices)."]),
    ("Example tumor", ["d=43.47 mm²/yr, ρ=11.22 /yr, t₁=94 d, t₂=264 d."]),
    ("Note", [
        "Density at t₂ comes from Network 1; t₃, t₄ come from simulation with Network 2's parameters.",
        "Error grows with time, mostly along the boundary.",
    ]),
], "Paper Fig. 9: true vs estimated density.", CYAN)

figure_slide(10, "Figure 10: Relative error of the parameters", "Part E | Results", [
    ("Panels", [
        "(a) relative error of λ and v (Network 2's direct output).",
        "(b) relative error of d_white and ρ (derived with Eq. 12, 13).",
    ]),
    ("Median errors", [
        "λ: 3.41%   v: 3.30%",
        "d_white: 5.86%   ρ: 2.75%",
    ]),
    ("Takeaway", [
        "The parameters are estimated accurately. d_white has a bit more error because the errors of λ and v combine.",
    ]),
], "Paper Fig. 10: relative error of parameters.", PINK)

figure_slide(11, "Figure 11: True vs predicted scatter (CCC)", "Part E | Results", [
    ("Panels", [
        "(a) λ  (b) v  (c) d_white  (d) ρ. X = true, Y = predicted. Red line = perfect (y=x).",
    ]),
    ("CCC (Lin)", [
        "λ: 0.99, v: 0.95, d_white: 0.97, ρ: 0.99.",
        "CCC = 1 means perfect agreement. It checks correlation and bias together.",
    ]),
    ("Takeaway", [
        "Points lie near the diagonal, so predictions match the true values.",
        "These results are on synthetic data only.",
    ]),
], "Paper Fig. 11: true vs predicted parameters.", GREEN)

figure_slide(12, "Figure 12: A real GBM patient", "Part E | Results", [
    ("Layout", [
        "Rows: T1Gd, T2 FLAIR, estimated cell density. Columns: axial, sagittal, coronal.",
        "Red contour = segmented enhancing core (T1Gd) and oedema (T2 FLAIR).",
    ]),
    ("Patient", ["IDH-wildtype glioblastoma (only Network 1 was used)."]),
    ("Important", [
        "No biopsy and the patient had already been treated, so the result is not validated.",
        "The dense region (c~1) overlaps the necrotic area, which the paper calls counter-intuitive (necrosis is not in the model).",
    ]),
], "Paper Fig. 12: real GBM patient.", AMBER)

table_slide("Results summary", "Part E | Results",
            ["Metric", "Value", "Meaning"], [
    ["Density MAE at t₂ (median)", "9.58×10⁻³", "very accurate density"],
    ["Density MAE at t₃ / t₄ (median)", "1.38×10⁻² / 2.20×10⁻²", "error slowly accumulates"],
    ["Median rel. error λ, v", "3.41% / 3.30%", "direct outputs"],
    ["Median rel. error d_white, ρ", "5.86% / 2.75%", "derived parameters"],
    ["CCC λ, v, d_white, ρ", "0.99, 0.95, 0.97, 0.99", "strong agreement"],
    ["ASSD", "≤ 1 mm in most cases", "contour within voxel size"],
], GREEN, ["All numbers are on synthetic test tumors (200 tumors, one unseen volunteer brain)."], [4.6, 3.6, 3.9])

text_slide("Limitations (as stated by the authors)", "Part F | Critical analysis", [
    "##Synthetic validation: assumes Fisher-KPP describes real gliomas, which has not been extensively proven.",
    "##No real validation: one GBM patient, no biopsy, after treatment.",
    "##Mass effect and necrosis are not in the model.",
    "##Treatment effects (surgery, chemo, radiotherapy) are not modelled.",
    "##Fixed thresholds: if c₁, c₂ change, the networks must be retrained.",
    "##Constant D, ρ assumed over time in untreated patients.",
    "##Error accumulation: at longer times, artefact foci and errors grow.",
], PINK, size=16)

text_slide("Relation to our HybridTumorNet", "Part F | Critical analysis", [
    "##Same: both predict PDE parameters (D, ρ) and use Fisher-KPP.",
    "##Same: both train on synthetic Fisher-KPP simulations.",
    "##Solver: Martens uses forward Euler (Δt=0.5 d); our code uses RK4.",
    "##Loss: in Martens the PDE only generates the data; in ours the PDE is also a physics loss during training.",
    "##Data: Martens has 6 healthy brains + 1 GBM; we have LUMIERE with 91 real GBM patients.",
    "##Our contribution: validation on real longitudinal data, which Martens lists as future work.",
], GREEN, ("Comparison", [
    "Martens: U-Net + conv-encoder, TensorFlow",
    "Ours: ResNet-based HybridTumorNet, PyTorch/MONAI",
    "Martens: contours as input",
    "Ours: MRI modalities as input",
]), size=16)

text_slide("Our results so far (LUMIERE)", "Part F | Critical analysis", [
    "MONAI SegResNet (SOTA bundle) on LUMIERE: average Whole-Tumor Dice 89.7%, median 93.1%, peak 96.1%.",
    "Our pretrained ResNet-10 HybridTumorNet: average WT Dice about 49.2% (10 patients), peak about 70%.",
    "The gap: a smaller backbone and synthetic pretraining; fine-tuning on real data is still pending.",
    "Evaluation was done on atlas-space data (skull-stripped, 1 mm) after fixing a coordinate mismatch bug.",
    "Note: this is segmentation Dice, not a growth-prediction metric. For growth we need Martens-style ASSD/Hausdorff and parameter CCC.",
], CYAN, ("Takeaway", [
    "SOTA benchmark ready",
    "Gap measured",
    "Next: fine-tune on LUMIERE",
]), size=16)

text_slide("Next steps", "Part F | Critical analysis", [
    "Upgrade the backbone: use SegResNet / SwinUNETR inside HybridTumorNet.",
    "Fine-tune growth prediction on LUMIERE longitudinal pairs (tᵢ → tᵢ₊₁).",
    "Use Martens-style metrics: MAE, ASSD, Hausdorff, parameter CCC.",
    "Add uncertainty quantification (MC dropout / ensembles).",
    "Address Martens' limitations where data allows: real data, mass effect, treatment.",
], GREEN)

table_slide("Glossary", "Part F | Critical analysis",
            ["Term", "Simple meaning"], [
    ["GBM", "Glioblastoma, the most aggressive brain tumor"],
    ["Voxel", "3D pixel (1 mm³)"],
    ["DTI", "MRI that measures the direction of nerve fibres"],
    ["WM / GM / CSF", "White matter (fibres) / Grey matter (neurons) / Brain fluid"],
    ["MICO", "Algorithm that segments tissues and also corrects the MRI brightness bias"],
    ["MAE", "Mean absolute error"],
    ["Hausdorff / ASSD", "Boundary error: worst case / average"],
    ["CCC", "Agreement score (1 = perfect)"],
    ["Anisotropy a=10", "In WM, diffusion along the fibre is 10× larger"],
], AMBER, None, [3.2, 8.9])

text_slide("Anticipated Discussion Questions", "Part F | Critical analysis", [
    "##Q: Why not predict D, ρ directly? A: λ and v can be measured from images; D and ρ are just combinations of them.",
    "##Q: Why only synthetic data? A: Nobody knows the true cell density of a real tumor, so ground truth exists only in simulation.",
    "##Q: Can we trust the results? A: They are on synthetic data; the authors themselves say real validation is pending.",
    "##Q: How is our work different? A: We have real longitudinal LUMIERE data and a physics loss during training.",
    "##Q: Why does error grow with time? A: Errors in the initial density and parameters accumulate in the simulation.",
], CYAN, size=16)

text_slide("Conclusion", "Summary", [
    "Martens et al. showed that deep CNNs can estimate cell density and Fisher-KPP parameters from MRI contours in seconds (median relative error 2.75–5.86%, CCC ≥ 0.95).",
    "But the validation is entirely synthetic; validation on real gliomas is still pending.",
    "Our HybridTumorNet takes the same idea forward with real longitudinal LUMIERE data and physics-informed training.",
    "Thank you. Questions and discussion.",
], PINK, size=19)

prs.save(OUT)
print("saved", cnt[0], "slides")
