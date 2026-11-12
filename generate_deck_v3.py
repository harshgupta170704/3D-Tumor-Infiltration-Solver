from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image
import os

FIG = r"C:\Users\harsh\OneDrive\Desktop\Cancer-Research\papers\martens_figures\clean_figures"
OUT = r"C:\Users\harsh\OneDrive\Desktop\Cancer-Research\Martens_2022_Detailed_Presentation.pptx"

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


# ---------------- 1 Title
s = prs.slides.add_slide(BLANK)
cnt[0] += 1
s.background.fill.solid()
s.background.fill.fore_color.rgb = BG
rect(s, 0, 0, 13.333, 0.25, CYAN)
rect(s, 0.8, 2.0, 0.12, 2.3, PINK)
text(s, 1.2, 1.9, 11, 0.5, "PAPER DEEP DIVE  |  LITERATURE REVIEW", 14, CYAN, True)
text(s, 1.2, 2.4, 11.5, 1.6, "Deep Learning for Fast Glioma Growth Modeling", 42, WHITE, True)
text(s, 1.2, 3.75, 11, 0.7, "Martens et al. (2022): cell density aur Fisher-KPP parameters (D, \u03C1) ko 3D CNN se estimate karna", 18, GREY)
text(s, 1.2, 4.9, 11, 0.4, "Harsh Gupta (23U01011)  &  Mayank Kumar Verma (23U01026)", 18, WHITE, True)
text(s, 1.2, 5.3, 11, 0.4, "Supervisor: Dr. Kamlesh Kumar Chauhan  |  Dept. of ECE, IIIT Bhopal", 15, GREY)
text(s, 1.2, 5.9, 11, 0.4, "Project: HybridTumorNet \u2013 Hybrid DL + PINN for Brain Tumor Growth Prediction", 15, PINK, True)

text_slide("Is presentation ko kaise padhein", "Roadmap", [
    "##Part A: Basics (slide 3\u20136): MRI, tumor, PDE, CNN. Agar pehli baar padh rahe ho to yahan se shuru karo.",
    "##Part B: Paper ka idea (slide 7\u20139): kaunsa problem, kaunsa solution, pipeline.",
    "##Part C: Data banana (slide 10\u201314): Figure 1\u20134 aur Table 1\u20132.",
    "##Part D: Networks (slide 15\u201318): Figure 5\u20136, training details.",
    "##Part E: Results (slide 19\u201325): Figure 7\u201312, numbers ke saath.",
    "##Part F: Critical analysis (slide 26\u201332): limitations, humare project se relation, glossary, teacher Q&A.",
    "Har figure slide par: left me asli figure, right me panel-wise explanation + key numbers.",
], CYAN, size=16)

# ---------------- Part A basics
text_slide("Basics 1: Brain tumor aur MRI", "Part A | Basics", [
    "##Glioma / Glioblastoma (GBM) kya hai?",
    "Brain me glial cells se bana tumor. GBM sabse aggressive type hai; ye ek gend jaisa nahi, balki tentacles ki tarah aas-paas ke tissue me ghus jata hai (infiltration).",
    "##MRI me kya dikhta hai?",
    "MRI scan sirf wahi hissa dikhata hai jahan tumor cells bahut zyada hain. Jahan cells kam hain wahan scan normal dikhta hai, par cells waha hote hain.",
    "##Isse kya problem hoti hai?",
    "Surgery/radiotherapy sirf visible tumor par ho to bache hue invisible cells se tumor wapas aa jata hai. Isliye invisible cells ka map chahiye.",
], PINK, ("Chhota glossary", [
    "T1Gd: contrast wala MRI, tumor ka enhancing core dikhata hai",
    "T2 FLAIR: oedema (sujan) wala area dikhata hai",
    "DTI: nerve fibres ki direction batane wala MRI",
    "Voxel: 3D pixel (yahan 1mm\u00B3)",
]))

text_slide("Basics 2: PDE aur Fisher-KPP model", "Part A | Basics", [
    "##PDE kya hai? Ek rule jo batata hai ki koi quantity space aur time me kaise badalti hai.",
    "##Fisher-KPP: \u2202u/\u2202t = \u2207\u00B7(D\u2207u) + \u03C1u(1\u2212u)",
    "u = tumor cell density (0 se 1). Pehla part fail na (diffusion), dusra part badhna (proliferation).",
    "D = diffusion: cells kitni tezi se faile. \u03C1 = proliferation: kitni tezi se divide ho.",
    "u(1\u2212u): jab u chhota hai to growth tez; jab u\u21921 (voxel bhar gaya) to growth ruk jati hai.",
    "##Analogy: paani me syahi (diffusion) + syahi ka khud ko multiply karna (reaction).",
], CYAN, ("Forward vs Inverse", [
    "Forward: D, \u03C1, starting tumor pata hain \u2192 future tumor simulate karo (aasaan).",
    "Inverse: MRI dikh raha hai \u2192 D, \u03C1 aur invisible cells nikalo (mushkil).",
    "Paper ka goal inverse problem fast solve karna hai.",
]))

text_slide("Basics 3: Wave speed aur front width (\u03BB, v)", "Part A | Basics", [
    "Fisher-KPP ka tumor ek travelling wave ki tarah aage badhta hai: front ek constant speed se chalta hai aur edge exponentially girti hai.",
    "##Infiltration depth: \u03BB = \u221A(D/\u03C1) (mm). Matlab front kitni chaudi/dhundhli hai.",
    "##Front velocity: v = 2\u221A(D\u03C1) (mm/year). Matlab tumor ki boundary kitni tezi se badh rahi hai.",
    "Ulta karke: D = \u03BBv/2 aur \u03C1 = v/(2\u03BB) (paper ke Eq. 12, 13).",
    "Isliye network seedha D, \u03C1 predict nahi karta; wo \u03BB aur v predict karta hai, jo images se naapna aasaan hai, phir D, \u03C1 nikalte hain.",
], GREEN, ("Pehchaan", [
    "\u03BB: do contours ke beech ki doori se pata chalta hai",
    "v: ek hi contour ke do alag time par shift se pata chalta hai",
]))

text_slide("Basics 4: CNN aur U-Net kya hota hai", "Part A | Basics", [
    "##CNN: image me patterns (edges, shapes) khud seekhne wala neural network. 3D CNN poore 3D volume par kaam karta hai.",
    "##U-Net: encoder (image chhoti karke context samajhna) + decoder (wapas bada karke output banana) + skip connections (details na khona).",
    "##Convolutional Encoder: sirf encoder + last me fully-connected layer. Image se kuch numbers (regression) nikalta hai.",
    "##Loss: MAE (average |error|) ya MSE (average error\u00B2). Adam optimizer weights update karta hai.",
    "##Training data: synthetic tumors, kyunki asli tumor ki true cell density kisi ko nahi pata.",
], PINK)

# ---------------- Part B idea
text_slide("Paper ka problem aur contribution", "Part B | Idea", [
    "##Problem: Reaction-diffusion models decades se hain, par patient par use nahi hote kyunki (1) starting condition (invisible cells) aur (2) D, \u03C1 estimate karna mushkil hai.",
    "##Contribution: do deep CNNs jo dono problems ek saath solve karte hain, seconds me.",
    "Network 1 (3D U-Net): contours \u2192 poora cell density map.",
    "Network 2 (Conv-Encoder): contours + time gap \u2192 \u03BB, v \u2192 D, \u03C1.",
    "Phir model se future tumor simulate karte hain (90 aur 180 din aage).",
    "Training: 1,200 synthetic tumors, 6 healthy volunteers ke asli brains par.",
], CYAN, ("Ek line me", [
    "MRI ke visible contours \u2192 CNN \u2192 invisible cells ka map + D, \u03C1 \u2192 future growth",
]))

text_slide("Pipeline ek nazar me", "Part B | Idea", [
    "##Step 1 (Data): healthy volunteers ka MRI + DTI \u2192 brain map aur tumor-diffusion tensor (Fig 1, 2).",
    "##Step 2 (Simulate): random \u03BB, v, seed, time gaps \u2192 Fisher-KPP se tumor grow karo (Fig 3, 4).",
    "##Step 3 (Contours): t\u2082 par density ko threshold karke 2 contours (c\u2081=0.80, c\u2082=0.16) banao; t\u2081 par ek (c\u2082).",
    "##Step 4 (Train): Network 1 aur Network 2 alag-alag train (Fig 5, 6).",
    "##Step 5 (Validate): t\u2082 par density, phir simulation se t\u2083, t\u2084 predict karke true se compare (Fig 7\u201311).",
    "##Step 6 (Real): ek real GBM patient par Network 1 lagana (Fig 12).",
], GREEN)

text_slide("Imaging contours aur thresholds (c\u2081, c\u2082)", "Part B | Idea", [
    "MRI tumor ka exact density nahi dikhata; wo ek threshold par boundary dikhata hai: jahan density threshold se zyada, wo andar.",
    "##c\u2081 = 0.80: T1Gd ka enhancing core (dense tumor).",
    "##c\u2082 = 0.16: T2 FLAIR ka oedema outline (bahar ki boundary).",
    "Ye threshold values Swanson et al. (2008) se li gayi hain (paper ref [5]).",
    "Synthetic data me same thresholds par density ko binary mask me badalte hain; wahi network ka input hai.",
    "Caveat: oedema asli invasion zone nahi hota, isliye c\u2082 wali assumption approximate hai (paper ka discussion).",
], AMBER, ("Terms", [
    "\u0393\u2081: c=0.80 contour at t\u2082",
    "\u0393\u2082: c=0.16 contour at t\u2082",
    "\u0393\u2083: c=0.16 contour at t\u2081",
]))

# ---------------- Part C data
figure_slide(1, "Figure 1: Brain ko tissue types me baantna", "Part C | Data synthesis", [
    ("Panels", [
        "(a) T2 FLAIR slice + red brain mask: simulation sirf is region me chalegi.",
        "(b) Usi slice ki T1 BRAVO image: anatomy saaf, skull bhi dikhta hai.",
        "(c) Segmented map: white matter, grey matter, CSF alag labels.",
    ]),
    ("Kaise bana", [
        "MICO algorithm (bias field hata kar clustering) + manual correction.",
    ]),
    ("Kyun zaroori", [
        "Tumor CSF me nahi failta; WM aur GM me D alag hota hai.",
        "Ye map Figure 2 ke diffusion tensor ko banane me use hota hai.",
    ]),
], "Paper Fig. 1: processed MR data.")

figure_slide(2, "Figure 2: DTI se tumor diffusion tensor", "Part C | Data synthesis", [
    ("Grid kya hai", [
        "3\u00D73 grid = tensor ke 9 components d(i,j). Symmetric hai, isliye 6 independent.",
    ]),
    ("(a) Water tensor", [
        "Asli DTI scan, FSL se distortion, eddy current, motion correct. Ventricles bright.",
    ]),
    ("(b) Tumor tensor", [
        "(a) + brain map se: WM me anisotropy a=10, GM me d_grey/d_white=0.1, CSF hata diya.",
    ]),
    ("Fark", [
        "(a) paani ka diffusion, (b) tumor cells ka diffusion. Simulation (b) use karti hai.",
    ]),
], "Paper Fig. 2: processed DTI data.", PINK)

table_slide("Table 1: Synthetic tumors ke parameter ranges", "Part C | Data synthesis",
            ["Parameter", "Min", "Max", "Unit", "Matlab"], [
    ["\u03BB (infiltration depth)", "0.5", "2.0", "mm", "front kitni chaudi"],
    ["v (front velocity)", "28.28", "48.99", "mm/year", "boundary kitni tez badhti"],
    ["\u0394t\u2081 (t\u2080 se t\u2081 tak)", "90", "180", "days", "pehli scan tak ka time"],
    ["\u0394t\u2082 (t\u2081 se t\u2082)", "90", "180", "days", "do scans ke beech ka gap"],
], GREEN, [
    "Sab uniform distribution se random. \u0394t\u2083 = \u0394t\u2084 = 90 din validation ke liye fixed.",
    "D = \u03BBv/2 aur \u03C1 = v/(2\u03BB). D aur \u03C1 ko independent sample karne se bahut chhote ya bahut bade tumors ban jate, isliye \u03BB, v sample kiye.",
    "Seed: white matter me 3\u00D73\u00D73 voxel block, jahan c=1 set hota hai. Total 200 tumors per volunteer \u00D7 6 volunteers = 1,200.",
], [3.9, 1.3, 1.3, 1.7, 3.9])

figure_slide(3, "Figure 3: Parameters kaise sample hue", "Part C | Data synthesis", [
    ("(a) (\u03BB, v) scatter", [
        "Blue dot = ek tumor. \u03BB 0.5\u20132 mm, v 28\u201349 mm/yr me uniform. Red box = boundary.",
    ]),
    ("(b) (d_white, \u03C1) scatter", [
        "Wahi points Eq. 12, 13 se D, \u03C1 me. Rectangle curved shape ban gaya (non-linear map).",
    ]),
    ("(c) d_white histogram", [
        "Blue = sampled, red = theoretical. d_white lagbhag 7\u201349 mm\u00B2/yr.",
    ]),
    ("(d) \u03C1 histogram", [
        "Right-skewed: peak ~10\u201312 /yr, lambi tail. Zyada tumors slow, kuch bahut fast.",
    ]),
    ("Takeaway", ["Network ko slow se aggressive tak sab type ke tumors dikhte hain."]),
], "Paper Fig. 3: sampling of model parameters.", GREEN)

figure_slide(4, "Figure 4: Synthetic tumors ka evolution (t\u2081\u2192t\u2084)", "Part C | Data synthesis", [
    ("Kya hai", [
        "3 tumors (rows), 4 time points t\u2081\u2013t\u2084 (columns), axial slices. Colour = cell density 0.01\u21921.",
    ]),
    ("Table 2 values", [
        "T1: d=10.87, \u03C1=31.77, \u03BB=1.71, v=37.16",
        "T2: d=15.07, \u03C1=13.95, \u03BB=0.96, v=29.00",
        "T3: d=41.49, \u03C1=11.31, \u03BB=0.52, v=43.33",
    ]),
    ("Observation", [
        "Chhota \u03BB \u2192 sharp front, bada \u03BB \u2192 dhundhla front. Time ke saath tumor badhta hai.",
        "Solver: forward Euler, \u0394t=0.5 din, GPU par ~1 s per tumor.",
    ]),
], "Paper Fig. 4: simulated cell density at t\u2081\u2013t\u2084.", AMBER)

# ---------------- Part D networks
figure_slide(5, "Figure 5: Network 1 \u2013 3D U-Net (cell density)", "Part D | Networks", [
    ("Input (8 channels)", [
        "2 binary contour masks (\u0393\u2081, \u0393\u2082) + 6 unit tensor components. Size 192\u00D7192\u00D7128.",
    ]),
    ("Output", ["Cell density map (same size). Har voxel me u."]),
    ("Structure", [
        "4 down blocks, 4 up blocks, 1 output block. Features 32\u219264\u2192128 bottom par.",
        "Grey boxes = skip connections (concatenation).",
        "Arrows: Conv3\u00D73, Conv2\u00D72 stride 2 (shrink), Deconv2\u00D72 (grow), Conv1\u00D71 (final).",
    ]),
    ("Goal", ["Visible contours se invisible tumor cells ka map banana."]),
], "Paper Fig. 5: 3D U-Net.", CYAN)

figure_slide(6, "Figure 6: Network 2 \u2013 Conv-Encoder (\u03BB aur v)", "Part D | Networks", [
    ("Input (9 channels + \u0394t\u2082)", [
        "3 contours (\u0393\u2081, \u0393\u2082 at t\u2082; \u0393\u2083 at t\u2081) + 6 tensor components, plus time gap \u0394t\u2082.",
    ]),
    ("Structure", [
        "6 down-sampling blocks \u2192 flatten (144 features) + \u0394t\u2082 \u2192 1 fully connected layer.",
    ]),
    ("Output", ["2 numbers: \u03BB aur v. Phir Eq. 12, 13 se d_white aur \u03C1."]),
    ("Kyun 3 contours", [
        "\u03BB do contours ki doori se, v ek contour ke do time par shift se nikalta hai.",
    ]),
], "Paper Fig. 6: convolutional regressor.", PINK)

table_slide("Dono networks ki training details", "Part D | Networks",
            ["", "Network 1 (U-Net)", "Network 2 (Conv-Encoder)"], [
    ["Task", "Cell density estimation", "\u03BB, v estimation"],
    ["Input", "192\u00D7192\u00D7128\u00D78", "192\u00D7192\u00D7128\u00D79 + \u0394t\u2082"],
    ["Loss", "MAE", "MSE (targets standardised)"],
    ["Optimizer", "Adam, lr 1e-4, batch 1", "Adam, lr 1e-4"],
    ["Augmentation", "Random shifts \u00B115 voxels", "Random shifts \u00B115 voxels"],
    ["Best test loss", "MAE 1.24e-4 (epoch 876)", "MSE 6.75e-2 (epoch 628)"],
    ["Split", "5 volunteers train / 1 test (83%/17%)", "same"],
], PINK, ["Framework: TensorFlow 2.5.0, GPU: RTX 3090. Early stopping: 100 epochs bina improvement. Rotation augmentation nahi, kyunki tensor components bhi rotate karne padte."],
            [2.4, 4.9, 4.8])

text_slide("Validation protocol aur post-processing", "Part D | Networks", [
    "##Experiment: Network 1 se t\u2082 par density, Network 2 se d_white aur \u03C1. Phir Fisher-KPP se t\u2083 (t\u2082+90 din) aur t\u2084 (t\u2082+180 din) simulate karo.",
    "##Compare with truth (true density + true parameters): MAE sirf c>0.01 area me, Hausdorff distance aur ASSD contours ke beech.",
    "##Hausdorff = sabse bura boundary error (max). ASSD = average boundary error.",
    "##Post-processing: (1) CSF/background density = 0. (2) density > 1 ko 1 par clip. (3) sabse bade connected region ke bahar ke chhote maxima hata do.",
    "Reason: network ke chhote artefacts simulation me naye tumor foci bana dete hain.",
], AMBER, ("Test set", [
    "200 tumors, ek unseen volunteer ke brain par.",
    "Thresholds: c\u2081=0.80, c\u2082=0.16",
]))

# ---------------- Part E results
figure_slide(7, "Figure 7: Density error (MAE) boxplot", "Part E | Results", [
    ("Axes", [
        "Y: mean absolute error [10\u207B\u00B9], X: t\u2082, t\u2083 (+90 d), t\u2084 (+180 d).",
        "Box = middle 50%, line = median, whiskers = 1.5\u00D7IQR, dots = outliers.",
    ]),
    ("Medians", [
        "t\u2082: 9.58\u00D710\u207B\u00B3  |  t\u2083: 1.38\u00D710\u207B\u00B2  |  t\u2084: 2.20\u00D710\u207B\u00B2",
    ]),
    ("Samajh", [
        "Jitna aage predict karo error badhta hai (errors accumulate hote hain).",
        "Phir bhi median error 0.03 se kam.",
    ]),
], "Paper Fig. 7: MAE within c>0.01.", GREEN)

figure_slide(8, "Figure 8: Hausdorff aur ASSD", "Part E | Results", [
    ("Panels", [
        "(a) Hausdorff, c\u2081=0.80  (b) Hausdorff, c\u2082=0.16",
        "(c) ASSD, c\u2081=0.80  (d) ASSD, c\u2082=0.16. Y in mm.",
    ]),
    ("Kya batata hai", [
        "Predicted aur true contour kitni door hain. Hausdorff = worst point, ASSD = average.",
    ]),
    ("Paper ka claim", [
        "ASSD zyada tar cases me \u2264 1 mm (voxel size).",
        "Kuch outliers me Hausdorff bada par ASSD chhota: artefact foci ki wajah se.",
    ]),
], "Paper Fig. 8: Hausdorff and ASSD.", AMBER)

figure_slide(9, "Figure 9: True vs predicted density (ek example)", "Part E | Results", [
    ("Rows", [
        "1: target density, 2: predicted density, 3: absolute error (0\u20130.10).",
        "4 aur 5: contours on T1 aur T2 FLAIR. Blue=target, red=prediction, green=overlap.",
    ]),
    ("Columns", ["t\u2082, t\u2083, t\u2084 (axial)."]),
    ("Example", ["d=43.47 mm\u00B2/yr, \u03C1=11.22 /yr, t\u2081=94 d, t\u2082=264 d."]),
    ("Dhyan", [
        "t\u2082 par density Network 1 se; t\u2083, t\u2084 simulation se (Network 2 ke parameters).",
        "Error time ke saath badhta hai, mostly boundary par.",
    ]),
], "Paper Fig. 9: true vs estimated density.", CYAN)

figure_slide(10, "Figure 10: Parameters ka relative error", "Part E | Results", [
    ("Panels", [
        "(a) \u03BB aur v ka relative error (Network 2 ka direct output).",
        "(b) d_white aur \u03C1 ka relative error (Eq. 12, 13 se derived).",
    ]),
    ("Median errors", [
        "\u03BB: 3.41%   v: 3.30%",
        "d_white: 5.86%   \u03C1: 2.75%",
    ]),
    ("Samajh", [
        "Parameters bahut accurately nikle. d_white ka error thoda zyada, kyunki \u03BB aur v ke errors multiply hote hain.",
    ]),
], "Paper Fig. 10: relative error of parameters.", PINK)

figure_slide(11, "Figure 11: True vs predicted scatter (CCC)", "Part E | Results", [
    ("Panels", [
        "(a) \u03BB  (b) v  (c) d_white  (d) \u03C1. X = true, Y = predicted. Red line = perfect (y=x).",
    ]),
    ("CCC (Lin)", [
        "\u03BB: 0.99, v: 0.95, d_white: 0.97, \u03C1: 0.99.",
        "CCC 1 matlab perfect agreement; ye correlation + bias dono check karta hai.",
    ]),
    ("Samajh", [
        "Points diagonal ke paas \u2192 network ke predictions true values se match karte hain.",
        "Ye sirf synthetic data par hai.",
    ]),
], "Paper Fig. 11: true vs predicted parameters.", GREEN)

figure_slide(12, "Figure 12: Real GBM patient", "Part E | Results", [
    ("Layout", [
        "Rows: T1Gd, T2 FLAIR, estimated cell density. Columns: axial, sagittal, coronal.",
        "Red contour = segmented enhancing core (T1Gd) aur oedema (T2 FLAIR).",
    ]),
    ("Patient", ["IDH-wildtype glioblastoma (Network 1 hi use hua)."]),
    ("Important", [
        "Biopsy nahi hai aur patient ka treatment ho chuka hai, isliye result validate nahi hua.",
        "Dense region (c~1) necrotic area se match karta hai: paper khud kehta hai ye counter-intuitive hai (necrosis model me nahi).",
    ]),
], "Paper Fig. 12: real GBM patient.", AMBER)

table_slide("Results ka summary", "Part E | Results",
            ["Metric", "Value", "Meaning"], [
    ["Density MAE at t\u2082 (median)", "9.58\u00D710\u207B\u00B3", "very accurate density"],
    ["Density MAE t\u2083 / t\u2084 (median)", "1.38\u00D710\u207B\u00B2 / 2.20\u00D710\u207B\u00B2", "error slowly accumulates"],
    ["Median rel. error \u03BB, v", "3.41% / 3.30%", "direct outputs"],
    ["Median rel. error d_white, \u03C1", "5.86% / 2.75%", "derived parameters"],
    ["CCC \u03BB, v, d_white, \u03C1", "0.99, 0.95, 0.97, 0.99", "strong agreement"],
    ["ASSD", "\u2264 1 mm mostly", "contour within voxel size"],
], GREEN, ["Ye saare numbers synthetic test tumors par hain (200 tumors, 1 unseen volunteer brain)."], [4.6, 3.6, 3.9])

# ---------------- Part F
text_slide("Paper ki limitations (khud authors ke hisaab se)", "Part F | Critical analysis", [
    "##Synthetic validation: model ye maanta hai ki Fisher-KPP asli gliomas ko sahi describe karta hai; ye extensively proven nahi hai.",
    "##Real validation nahi: ek GBM patient, bina biopsy, treatment ke baad.",
    "##Mass effect aur necrosis model me nahi hain.",
    "##Treatment effects (surgery, chemo, radiotherapy) model me nahi.",
    "##Fixed thresholds: c\u2081, c\u2082 badle to networks retrain karne padenge.",
    "##Constant D, \u03C1 assumption: untreated patient me time ke saath constant maana.",
    "##Error accumulation: lambe time par artefact foci aur error badhta hai.",
], PINK, size=16)

text_slide("Hamare HybridTumorNet se relation", "Part F | Critical analysis", [
    "##Same: dono PDE parameters (D, \u03C1) predict karte hain aur Fisher-KPP use karte hain.",
    "##Same: dono synthetic Fisher-KPP simulations se train hote hain.",
    "##Difference (solver): Martens forward Euler (\u0394t=0.5 d), humare code me RK4.",
    "##Difference (loss): Martens me PDE sirf data banane me; hamare me PDE physics loss ke roop me training me.",
    "##Difference (data): Martens: 6 healthy brains + 1 GBM; hamare paas LUMIERE ke 91 real GBM patients.",
    "##Hamara contribution: real longitudinal data par validation, jo Martens ne future work bataya.",
], GREEN, ("Comparison", [
    "Martens: U-Net + Conv-Encoder, TensorFlow",
    "Ours: ResNet-based HybridTumorNet, PyTorch/MONAI",
    "Martens: contours input",
    "Ours: MRI modalities input",
]), size=16)

text_slide("Hamare abhi tak ke results (LUMIERE)", "Part F | Critical analysis", [
    "MONAI SegResNet (SOTA bundle) LUMIERE par: avg Whole-Tumor Dice 89.7%, median 93.1%, peak 96.1%.",
    "Hamara ResNet-10 HybridTumorNet pretrained: avg WT Dice ~49.2% (10 patients), peak ~70%.",
    "Gap: chhota backbone + synthetic pretraining; real-data fine-tuning abhi baaki hai.",
    "Pipeline: atlas-space (skull-stripped, 1 mm) data par evaluate kiya, coordinate mismatch bug fix karke.",
    "Note: ye segmentation Dice hai, growth prediction ka metric nahi. Growth ke liye Martens jaise ASSD/Hausdorff aur parameter CCC chahiye.",
], CYAN, ("Takeaway", [
    "SOTA benchmark ready",
    "Gap measured",
    "Next: fine-tune on LUMIERE",
]), size=16)

text_slide("Next steps", "Part F | Critical analysis", [
    "Backbone upgrade: SegResNet / SwinUNETR ko HybridTumorNet me lagana.",
    "LUMIERE longitudinal pairs par growth prediction (t\u1d62 \u2192 t\u1d62\u208a\u2081) fine-tuning.",
    "Metrics Martens ke jaise: MAE, ASSD, Hausdorff, parameter CCC.",
    "Uncertainty quantification (MC dropout / ensembles).",
    "Martens ki limitation address karna: real data, mass effect, treatment (agar data ho).",
], GREEN)

table_slide("Glossary: terms ek jagah", "Part F | Critical analysis",
            ["Term", "Simple matlab"], [
    ["GBM", "Glioblastoma, sabse aggressive brain tumor"],
    ["Voxel", "3D pixel (1 mm\u00B3)"],
    ["DTI", "MRI jo nerve fibre ki direction naapta hai"],
    ["WM / GM / CSF", "White matter (fibres) / Grey matter (neurons) / Brain fluid"],
    ["MICO", "MRI ko tissues me baantne wala algorithm jo brightness bias bhi theek karta hai"],
    ["MAE", "Average absolute error"],
    ["Hausdorff / ASSD", "Boundary error: worst / average"],
    ["CCC", "Agreement score (1 = perfect)"],
    ["Anisotropy a=10", "WM me fibre direction me diffusion 10 guna"],
], AMBER, None, [3.2, 8.9])

text_slide("Teacher ke liye possible questions", "Part F | Critical analysis", [
    "##Q: Network seedha D, \u03C1 kyun nahi deta? A: \u03BB, v images se directly naap sakte hain; D, \u03C1 sirf unke combination hain.",
    "##Q: Sirf synthetic data par kyun? A: Real tumor ki true cell density kisi ko pata nahi, isliye ground truth sirf simulation me milta hai.",
    "##Q: Result bharosemand hai? A: Synthetic par hai; real patient par validate nahi hua, authors khud maante hain.",
    "##Q: Humara kaam alag kaise? A: Hamare paas real longitudinal LUMIERE data hai aur physics loss training me hai.",
    "##Q: Error time ke saath kyun badhta hai? A: Initial density aur parameter errors simulation me accumulate hote hain.",
], CYAN, size=16)

text_slide("Conclusion", "Summary", [
    "Martens et al. ne dikhaya ki deep CNNs MRI contours se cell density aur Fisher-KPP parameters seconds me nikaal sakte hain (median rel. error 2.75\u20135.86%, CCC \u2265 0.95).",
    "Par poora validation synthetic data par hai; real glioma validation baaki hai.",
    "Hamara HybridTumorNet isi idea ko real longitudinal LUMIERE data aur physics-informed training se aage badhata hai.",
    "Thank you! Questions?",
], PINK, size=19)

prs.save(OUT)
print("saved", cnt[0], "slides")
