# -*- coding: utf-8 -*-
import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# Color Palette
BG_DARK = RGBColor(15, 23, 42)         # Deep Navy (#0F172A)
BG_LIGHT = RGBColor(248, 250, 252)     # Off-white (#F8FAFC)
CARD_BG = RGBColor(255, 255, 255)      # White
CARD_BORDER = RGBColor(226, 232, 240)  # Slate 200
NAVY_PRIMARY = RGBColor(30, 41, 59)    # Slate 800
NAVY_SECONDARY = RGBColor(51, 65, 85)  # Slate 700
ACCENT_BLUE = RGBColor(14, 116, 144)   # Cyan 700 (#0E7490)
ACCENT_CRIMSON = RGBColor(190, 24, 93) # Crimson (#BE185D)
ACCENT_EMERALD = RGBColor(16, 185, 129)# Emerald (#10B981)
ACCENT_PURPLE = RGBColor(124, 58, 237) # Purple (#7C3AED)
TEXT_MUTED = RGBColor(100, 116, 139)   # Slate 500

figures_dir = r'C:\Users\harsh\OneDrive\Desktop\Cancer-Research\papers\martens_figures\clean_figures'

def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_header(slide, title, category="RESEARCH PAPER DEEP DIVE • MARTENS ET AL. (2022)", dark_mode=False):
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(0.3))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category.upper()
    p_c.font.size = Pt(11)
    p_c.font.bold = True
    p_c.font.color.rgb = ACCENT_CRIMSON if dark_mode else ACCENT_BLUE

    t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.65))
    tf_t = t_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = RGBColor(255, 255, 255) if dark_mode else NAVY_PRIMARY

def add_card(slide, left, top, width, height, fill_color=CARD_BG, border_color=CARD_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.2)
    return shape

# ==============================================================
# SLIDE 0: TITLE SLIDE (Dark Mode)
# ==============================================================
slide0 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide0, BG_DARK)

add_card(slide0, Inches(1.0), Inches(1.0), Inches(11.333), Inches(5.5), fill_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))

tb = slide0.shapes.add_textbox(Inches(1.5), Inches(1.4), Inches(10.3), Inches(0.4))
p = tb.text_frame.paragraphs[0]
p.text = "COMPREHENSIVE RESEARCH ANALYSIS & ARCHITECTURAL FOUNDATION"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = ACCENT_EMERALD

tb = slide0.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(10.3), Inches(1.8))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Deep Learning for Reaction-Diffusion Glioma Growth Modelling"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(255, 255, 255)

p2 = tf.add_paragraph()
p2.text = "Martens et al. (2022) Paper Deep Dive & Direct Architectural Blueprint for HybridTumorNet (LUMIERE)"
p2.font.size = Pt(16)
p2.font.color.rgb = RGBColor(148, 163, 184)
p2.space_before = Pt(8)

tb = slide0.shapes.add_textbox(Inches(1.5), Inches(3.8), Inches(10.3), Inches(2.2))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Paper Reference: Corentin Martens, Antonin Rovai, Daniele Bonatto, et al. (Univ. Libre de Bruxelles) • arXiv:2111.13404"
p.font.size = Pt(12)
p.font.color.rgb = RGBColor(203, 213, 225)

p2 = tf.add_paragraph()
p2.text = "Project: Hybrid Deep Learning & Physics-Informed Neural Network for Brain Tumor Growth Prediction"
p2.font.size = Pt(13)
p2.font.bold = True
p2.font.color.rgb = RGBColor(255, 255, 255)
p2.space_before = Pt(8)

p3 = tf.add_paragraph()
p3.text = "Presented By: Harsh Gupta (23U01011) & Mayank Kumar Verma (23U01026) • IIIT Bhopal"
p3.font.size = Pt(13)
p3.font.color.rgb = ACCENT_BLUE
p3.space_before = Pt(6)

p4 = tf.add_paragraph()
p4.text = "Supervisor: Dr. Kamlesh Kumar Chauhan • Department of Electronics & Communication Engineering"
p4.font.size = Pt(12)
p4.font.color.rgb = TEXT_MUTED
p4.space_before = Pt(4)


# ==============================================================
# SLIDE 1: Asli Problem Kya Hai? (The Clinical Roadblock)
# ==============================================================
slide1 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide1, BG_LIGHT)
add_header(slide1, "Asli Problem Kya Hai? Why Was Glioma Growth Prediction Stuck?")

# Left Column: The 3 Core Roadblocks
add_card(slide1, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb = slide1.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.3), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Reaction-Diffusion Models 30 Saal Se Hain, Phir Bhi Hospital Mein Kyu Nahi Use Hote?"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_CRIMSON

p = tf.add_paragraph()
p.text = "Scientists Murray (1990) aur Swanson (2000) ne mathematical equation toh de di thi, par doctors isse patient ke liye personalize nahi kar paate the due to 3 big hurdles:"
p.font.size = Pt(12)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(6)

p = tf.add_paragraph()
p.text = "1. MRI Invisible Cells Ko Nahi Dikhata (The Iceberg Problem):"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(10)
p = tf.add_paragraph()
p.text = "MRI mein sirf 2 boundary dikhti hain (Enhancing core & Edema). Par lakho glioma cells in boundaries ke 2-3 cm aage healthy brain tissue mein penetrate kar chuke hote hain. Clinical MRI se full 3D continuous density field c(x) recover karna mathematically ill-posed tha."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "2. Har Patient Ka Growth Rate Alag Hota Hai (Unknown D & rho):"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "Kiska tumor diffuse hai (high diffusion D), aur kiska highly proliferative hai (high rho)? Ek scan se dono alag-alag calculate karna impossible tha."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "3. Traditional Inverse Solvers Boiled Down to Days of Compute:"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "Pehle ke mathematical PDE optimization methods ek single patient par ghanton ya din laga dete the, jo clinical emergency mein chal hi nahi sakta."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

# Right Column: The Breakthrough Solution
add_card(slide1, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb = slide1.shapes.add_textbox(Inches(7.0), Inches(1.55), Inches(5.3), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Martens et al. Ka Core Idea: Deep Learning as Inverse PDE Solvers"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE

p = tf.add_paragraph()
p.text = "Martens et al. ne socha: Agar traditional numerical solvers slow hain, toh kya hum Deep Convolutional Neural Networks (DCNNs) ko train kar sakte hain direct inverse mapping seekhne ke liye?"
p.font.size = Pt(12)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(8)

p = tf.add_paragraph()
p.text = "Unka 2-Step Master Plan:"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(12)

p = tf.add_paragraph()
p.text = "- Task 1 (Density Estimation via 3D U-Net):"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = ACCENT_PURPLE
p.space_before = Pt(6)
p = tf.add_paragraph()
p.text = "Sirf 2 MRI contours (T1Gd Enhancing + T2 FLAIR Edema) aur DTI water tensor ko feed karo -> Model poora continuous 3D invisible tumor density map c(x) predict kar deta hai!"
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "- Task 2 (Parameter Estimation via Conv-Encoder):"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = ACCENT_PURPLE
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "Two timepoints (t1 and t2) ke scans aur unka time-gap (Delta t) feed karo -> Model us individual patient ka diffusion D aur proliferation rho accurately calculate kar deta hai!"
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "Direct Relevance to Harsh's Project:"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_EMERALD
p.space_before = Pt(12)
p = tf.add_paragraph()
p.text = "Humara HybridTumorNet bhi exact yahi reaction-diffusion framework use karta hai! Martens ke paas real data nahi tha, isliye unhone synthetic banaya. Humare paas LUMIERE hai jisme real longitudinal scans hain!"
p.font.size = Pt(11)
p.font.color.rgb = NAVY_PRIMARY


# ==============================================================
# SLIDE 2: Mathematical Foundation (Fisher-KPP & Swanson's Law)
# ==============================================================
slide2 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide2, BG_LIGHT)
add_header(slide2, "Mathematical Foundation: Fisher-KPP Reaction-Diffusion PDE")

# Left Column: The Equations
add_card(slide2, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb = slide2.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.3), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "The Governing Physics: Fisher-KPP Equation"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE

p = tf.add_paragraph()
p.text = "del c / del t = div( D(r) grad c ) + rho * c * (1 - c)"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = ACCENT_CRIMSON
p.space_before = Pt(8)

p = tf.add_paragraph()
p.text = "Where c(r, t) is the normalized tumor cell density in range [0, 1] relative to carrying capacity c_max."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "Equation Ke Do Main Engines:"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(10)

p = tf.add_paragraph()
p.text = "1. Anisotropic Diffusion: div( D(r) grad c )"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = NAVY_SECONDARY
p = tf.add_paragraph()
p.text = "Tumor cells brain mein har direction mein barabar nahi failte! White matter fiber tracts ke along bohot tezi se diffuse karte hain. D(r) is a 3x3 symmetric tensor."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "2. Logistic Proliferation: rho * c * (1 - c)"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(6)
p = tf.add_paragraph()
p.text = "Tumor cells divide karte hain rate rho se. Par space aur oxygen limited hai, isliye jab c = 1 (carrying capacity) ho jaata hai, multiplication ruk jaata hai."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "Boundary Condition (No-Flux Neumann):"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "D(r) grad c . n = 0 at brain boundary (tumor cells skull ke bahar nahi nikal sakte)."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

# Right Column: Traveling Wave & Swanson's Hypothesis
add_card(slide2, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb = slide2.shapes.add_textbox(Inches(7.0), Inches(1.55), Inches(5.3), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Traveling Wave Theory & Swanson's MRI Thresholds"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_PURPLE

p = tf.add_paragraph()
p.text = "Fisher-KPP PDE ki ek bohot sundar mathematical property hoti hai: Asymptotically yeh ek Traveling Wave ban jaata hai:"
p.font.size = Pt(12)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(8)

p = tf.add_paragraph()
p.text = "- Front Propagation Velocity:  v = 2 * sqrt(d_white * rho)"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_CRIMSON
p.space_before = Pt(6)
p = tf.add_paragraph()
p.text = "Tumor boundary kitni speed se aage badh rahi hai (mm/year)."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "- Infiltration Depth:  lambda = sqrt(d_white / rho)"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE
p.space_before = Pt(6)
p = tf.add_paragraph()
p.text = "Tumor profile kitna steep ya gentle decay kar raha hai (mm). Agar lambda nikal liya aur v nikal liya, toh d_white aur rho seedha mil jaate hain:"
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED
p = tf.add_paragraph()
p.text = "d_white = (lambda * v) / 2      |      rho = v / (2 * lambda)"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY

p = tf.add_paragraph()
p.text = "Swanson's Clinical MRI Threshold Hypothesis:"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_EMERALD
p.space_before = Pt(10)
p = tf.add_paragraph()
p.text = "1. Enhancing Core (T1Gd):  c >= c1 = 0.80 (Macroscopic solid tumor)"
p.font.size = Pt(11)
p.font.color.rgb = NAVY_PRIMARY
p = tf.add_paragraph()
p.text = "2. Vasogenic Edema (T2-FLAIR):  c >= c2 = 0.16 (Visible infiltrating boundary)"
p.font.size = Pt(11)
p.font.color.rgb = NAVY_PRIMARY
p = tf.add_paragraph()
p.text = "Yeh dono thresholds hamare project ke liye baseline ground truth ka kaam karte hain!"
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED


# ==============================================================
# SLIDE 3: Brain Anatomy & DTI (Figure 1 & Figure 2)
# ==============================================================
slide3 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide3, BG_LIGHT)
add_header(slide3, "Brain Anatomy & DTI Tensor Field (Figure 1 & 2)")

# Left Column: Explanations in Hinglish
add_card(slide3, Inches(0.8), Inches(1.4), Inches(5.5), Inches(5.6))
tb = slide3.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.1), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Tumor Cells Highway vs Traffic: DTI Kyu Zaroori Hai?"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE

p = tf.add_paragraph()
p.text = "Brain koi uniform gel nahi hai. Ismein White Matter (nerve fibers) aur Grey Matter (cell bodies) hote hain:"
p.font.size = Pt(12)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(6)

p = tf.add_paragraph()
p.text = "- Figure 1: Brain Tissue Segmentation"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "T1/T2 FLAIR scans se White Matter, Grey Matter, aur CSF ko segment kiya gaya. Grey matter mein diffusion 10-times slower hota hai (d_grey / d_white = 0.1) kyunki waha fibers nahi hote."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "- Figure 2: DTI Water Diffusion Tensor Field"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "Diffusion Tensor Imaging (DTI) paani ke flow ki direction naapta hai. Jaha brain fibers jaa rahe hain, waha paani tezi se behta hai. Martens ne isi water tensor ko tumor diffusion tensor D(r) mein convert kiya with anisotropy factor a = 10 (cells fiber ke parallel 10x tezi se diffuse karte hain)."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "Connection to Our Project (LUMIERE):"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_EMERALD
p.space_before = Pt(10)
p = tf.add_paragraph()
p.text = "Hamare LUMIERE Atlas folder mein bhi exact yahi skull-stripped aur tissue-aligned 1mm isotropic grids hain, jo is biophysical model ke liye 100% compatible hain!"
p.font.size = Pt(11)
p.font.color.rgb = NAVY_PRIMARY

# Right Column: Images Figure 1 and Figure 2
add_card(slide3, Inches(6.5), Inches(1.4), Inches(6.0), Inches(5.6))
fig1_path = os.path.join(figures_dir, 'figure_1.png')
fig2_path = os.path.join(figures_dir, 'figure_2.png')

slide3.shapes.add_picture(fig1_path, Inches(6.7), Inches(1.5), width=Inches(5.6))
tb = slide3.shapes.add_textbox(Inches(6.7), Inches(3.6), Inches(5.6), Inches(0.4))
tb.text_frame.word_wrap = True
p = tb.text_frame.paragraphs[0]
p.text = "Figure 1: (a) T2 FLAIR with brain mask, (b) Tissue labels, (c) White/Grey/CSF map"
p.font.size = Pt(10)
p.font.color.rgb = TEXT_MUTED

slide3.shapes.add_picture(fig2_path, Inches(6.7), Inches(4.1), width=Inches(5.6))
tb = slide3.shapes.add_textbox(Inches(6.7), Inches(6.5), Inches(5.6), Inches(0.4))
tb.text_frame.word_wrap = True
p = tb.text_frame.paragraphs[0]
p.text = "Figure 2: (a) DTI water diffusion tensor, (b) Derived tumor diffusion tensor D(r)"
p.font.size = Pt(10)
p.font.color.rgb = TEXT_MUTED


# ==============================================================
# SLIDE 4: Dataset Synthesis (Figure 3 & Figure 4)
# ==============================================================
slide4 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide4, BG_LIGHT)
add_header(slide4, "Synthetic Dataset Generation: 1,200 Tumors (Figure 3 & 4)")

# Left Column: Images Figure 3 & Figure 4
add_card(slide4, Inches(0.8), Inches(1.4), Inches(5.8), Inches(5.6))
fig3_path = os.path.join(figures_dir, 'figure_3.png')
fig4_path = os.path.join(figures_dir, 'figure_4.png')

slide4.shapes.add_picture(fig3_path, Inches(0.9), Inches(1.5), width=Inches(5.6))
tb = slide4.shapes.add_textbox(Inches(0.9), Inches(3.9), Inches(5.6), Inches(0.4))
tb.text_frame.word_wrap = True
p = tb.text_frame.paragraphs[0]
p.text = "Figure 3: Joint distributions of sampled lambda, v, d_white, and rho parameters"
p.font.size = Pt(10)
p.font.color.rgb = TEXT_MUTED

slide4.shapes.add_picture(fig4_path, Inches(0.9), Inches(4.35), width=Inches(5.6))
tb = slide4.shapes.add_textbox(Inches(0.9), Inches(6.5), Inches(5.6), Inches(0.4))
tb.text_frame.word_wrap = True
p = tb.text_frame.paragraphs[0]
p.text = "Figure 4: Simulated tumor evolution at four time points (t1, t2, t3, t4) across 3 tumors"
p.font.size = Pt(10)
p.font.color.rgb = TEXT_MUTED

# Right Column: Hinglish explanation
add_card(slide4, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb = slide4.shapes.add_textbox(Inches(7.0), Inches(1.55), Inches(5.3), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Kyu Banaye 1,200 Synthetic Tumors? (Crucial Detail!)"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_CRIMSON

p = tf.add_paragraph()
p.text = "2021 mein real longitudinal MRI data available nahi tha. Isliye unhone 6 healthy volunteers ke real brain shapes par 1,200 fictitious tumors simulate kiye (200 per subject)."
p.font.size = Pt(12)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(6)

p = tf.add_paragraph()
p.text = "Parameter Sampling Strategy (Figure 3):"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "Unhone d_white aur rho ko directly sample nahi kiya (warna kuch tumor unrealistic ban jaate). Instead, unhone physically realistic infiltration depth lambda in [0.5, 2.0] mm aur front speed v in [28, 49] mm/yr sample kiya, aur usse d aur rho derive kiya!"
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "C++/CUDA Solver - TGSTK (Figure 4):"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "Unhone apna custom GPU-based finite-difference solver banaya (TGSTK). Har tumor ko t0 seed se grow kiya aur 4 timepoints (t1, t2, t3, t4) par snapshot save kiya (each ~90 to 180 days apart)."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "Our Big Advantage (Where We Beat Martens!):"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_EMERALD
p.space_before = Pt(10)
p = tf.add_paragraph()
p.text = "Reviewers Martens ke paper ko yahi bolte the ki 'tumhara data synthetic hai, real patient par fail ho sakta hai'. Hamare project mein LUMIERE dataset ke 91 REAL patients hain! Hum real biological growth par train karenge!"
p.font.size = Pt(11)
p.font.color.rgb = NAVY_PRIMARY


# ==============================================================
# SLIDE 5: Deep Learning Task 1 (Figure 5 - 3D U-Net)
# ==============================================================
slide5 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide5, BG_LIGHT)
add_header(slide5, "Deep Learning Task 1: Reconstructing Cell Density via 3D U-Net (Fig 5)")

# Top: Image Figure 5
add_card(slide5, Inches(0.8), Inches(1.4), Inches(11.733), Inches(2.9))
fig5_path = os.path.join(figures_dir, 'figure_5.png')
slide5.shapes.add_picture(fig5_path, Inches(1.0), Inches(1.5), width=Inches(11.3))

# Bottom Left: Architecture Details
add_card(slide5, Inches(0.8), Inches(4.5), Inches(5.7), Inches(2.6))
tb = slide5.shapes.add_textbox(Inches(1.0), Inches(4.6), Inches(5.3), Inches(2.4))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "3D U-Net Architecture Breakdown:"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE

p = tf.add_paragraph()
p.text = "- Input Dimensions: 192 x 192 x 128 with 8 channels"
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(4)
p = tf.add_paragraph()
p.text = "  - Channel 1: Binary mask of Gamma_1 (Enhancing Core, c >= 0.80)\n  - Channel 2: Binary mask of Gamma_2 (Edema boundary, c >= 0.16)\n  - Channels 3-8: 6 components of the unit DTI diffusion tensor field"
p.font.size = Pt(10)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "- Structure: 4 down-sampling blocks + 4 up-sampling blocks with skip connections (concatenation) and final 1x1 conv outputting c(x) in [0, 1]."
p.font.size = Pt(10)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(4)

# Bottom Right: Results and Relevance
add_card(slide5, Inches(6.8), Inches(4.5), Inches(5.7), Inches(2.6))
tb = slide5.shapes.add_textbox(Inches(7.0), Inches(4.6), Inches(5.3), Inches(2.4))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Results & Connection to Our SegResNet:"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = ACCENT_PURPLE

p = tf.add_paragraph()
p.text = "- Performance: Mean Absolute Error (MAE) was 1.24 x 10^-4 on unseen test subject! Model smooth cell concentration accurately recreate kar leta hai."
p.font.size = Pt(11)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(4)

p = tf.add_paragraph()
p.text = "- Why this proves our SegResNet strategy:"
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = ACCENT_EMERALD
p.space_before = Pt(4)
p = tf.add_paragraph()
p.text = "Martens ne mana tha ki unka network raw image nahi, binary contours leta hai. Humare SegResNet ne LUMIERE par 90-96% Dice diya hai! That means humara SegResNet Martens ke is model ko crystal-clear Gamma_1 and Gamma_2 feed kar sakta hai!"
p.font.size = Pt(10)
p.font.color.rgb = NAVY_PRIMARY


# ==============================================================
# SLIDE 6: Deep Learning Task 2 (Figure 6 - Parameter Estimation)
# ==============================================================
slide6 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide6, BG_LIGHT)
add_header(slide6, "Deep Learning Task 2: Parameter Estimation (D, rho) via Conv-Encoder (Fig 6)")

# Top: Image Figure 6
add_card(slide6, Inches(0.8), Inches(1.4), Inches(11.733), Inches(2.9))
fig6_path = os.path.join(figures_dir, 'figure_6.png')
slide6.shapes.add_picture(fig6_path, Inches(1.0), Inches(1.5), width=Inches(11.3))

# Bottom Left: Architecture
add_card(slide6, Inches(0.8), Inches(4.5), Inches(5.7), Inches(2.6))
tb = slide6.shapes.add_textbox(Inches(1.0), Inches(4.6), Inches(5.3), Inches(2.4))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Convolutional Encoder for (lambda, v):"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE

p = tf.add_paragraph()
p.text = "- Input: 9 channels (192 x 192 x 128)"
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(4)
p = tf.add_paragraph()
p.text = "  - Contour Gamma_2 at past time t1 (past edema)\n  - Contour Gamma_2 at current time t2 (current edema)\n  - Contour Gamma_1 at current time t2 (current enhancing core)\n  - 6 DTI tensor channels"
p.font.size = Pt(10)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "- Time Injection: Bottleneck par feature vector flatten karke scalar time-gap Delta t concatenate kiya, followed by dense layer outputting (lambda, v)."
p.font.size = Pt(10)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(4)

# Bottom Right: Relevance
add_card(slide6, Inches(6.8), Inches(4.5), Inches(5.7), Inches(2.6))
tb = slide6.shapes.add_textbox(Inches(7.0), Inches(4.6), Inches(5.3), Inches(2.4))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Why This Solves The Biggest Inverse Problem:"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = ACCENT_CRIMSON

p = tf.add_paragraph()
p.text = "- Speed of Tumor Front (v) = Distance between Edema(t1) and Edema(t2) divided by Delta t."
p.font.size = Pt(11)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(4)

p = tf.add_paragraph()
p.text = "- Steepness of Front (lambda) = Distance between Core(t2) and Edema(t2) at same timepoint."
p.font.size = Pt(11)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(4)

p = tf.add_paragraph()
p.text = "- D and rho derived instantly! In seconds, instead of days of numerical fitting."
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = ACCENT_EMERALD
p.space_before = Pt(4)

p = tf.add_paragraph()
p.text = "Hamare model mein: We have PhysicsParameterHead predicting D(x) and rho directly inside PyTorch. Martens ka logic validates our architecture!"
p.font.size = Pt(10)
p.font.color.rgb = NAVY_PRIMARY


# ==============================================================
# SLIDE 7: Forward Simulation & Future Prediction (Figure 9)
# ==============================================================
slide7 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide7, BG_LIGHT)
add_header(slide7, "Forward Simulation: Predicting Future Tumor Growth (Figure 9)")

# Left Column: Image Figure 9
add_card(slide7, Inches(0.8), Inches(1.4), Inches(5.8), Inches(5.6))
fig9_path = os.path.join(figures_dir, 'figure_9.png')
slide7.shapes.add_picture(fig9_path, Inches(0.9), Inches(1.5), width=Inches(5.6))

# Right Column: Hinglish Breakdown
add_card(slide7, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb = slide7.shapes.add_textbox(Inches(7.0), Inches(1.55), Inches(5.3), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Bhavishya Ki Prediction (t3 = +90 Days, t4 = +180 Days)"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE

p = tf.add_paragraph()
p.text = "Figure 9 dikhata hai ki model ne t2 par density reconstruct ki aur parameters nikaale, phir TGSTK PDE solver se 3 mahine aur 6 mahine aage ka scan simulate kiya:"
p.font.size = Pt(12)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(6)

p = tf.add_paragraph()
p.text = "Row-by-Row Analysis of Figure 9:"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(8)

p = tf.add_paragraph()
p.text = "- Row 1 (Target Ground Truth): Asli simulated tumor at t2, t3, t4."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED
p = tf.add_paragraph()
p.text = "- Row 2 (Model Prediction): Model ka forecasted future state."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED
p = tf.add_paragraph()
p.text = "- Row 3 (Absolute Error Map): Error almost 0 hai (dark black/blue)! Only slight error at active proliferating boundaries."
p.font.size = Pt(11)
p.font.color.rgb = ACCENT_CRIMSON
p = tf.add_paragraph()
p.text = "- Rows 4 & 5 (Contour Overlap): Blue = True, Red = Predicted, Green = Perfect Overlap! 90% se zyada boundary green hai, showing high fidelity!"
p.font.size = Pt(11)
p.font.color.rgb = ACCENT_EMERALD

p = tf.add_paragraph()
p.text = "How This Applies to LUMIERE:"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_PURPLE
p.space_before = Pt(10)
p = tf.add_paragraph()
p.text = "LUMIERE mein patient ka Week 0 scan t2 ka kaam karega, aur Week 44 scan t3/t4 ka kaam karega! Hum exactly aisi hi Target vs Prediction comparison slide banayenge for our paper!"
p.font.size = Pt(11)
p.font.color.rgb = NAVY_PRIMARY


# ==============================================================
# SLIDE 8: Quantitative Error Analysis (Figure 7 & 8)
# ==============================================================
slide8 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide8, BG_LIGHT)
add_header(slide8, "Quantitative Results: MAE, Hausdorff, & ASSD Across Time (Fig 7 & 8)")

# Left Column: Images Figure 7 and Figure 8
add_card(slide8, Inches(0.8), Inches(1.4), Inches(5.8), Inches(5.6))
fig7_path = os.path.join(figures_dir, 'figure_7.png')
fig8_path = os.path.join(figures_dir, 'figure_8.png')

slide8.shapes.add_picture(fig7_path, Inches(0.9), Inches(1.5), width=Inches(5.6))
tb = slide8.shapes.add_textbox(Inches(0.9), Inches(3.6), Inches(5.6), Inches(0.4))
tb.text_frame.word_wrap = True
p = tb.text_frame.paragraphs[0]
p.text = "Figure 7: Mean Absolute Error distribution over time (t2, t3, t4)"
p.font.size = Pt(10)
p.font.color.rgb = TEXT_MUTED

slide8.shapes.add_picture(fig8_path, Inches(0.9), Inches(4.0), width=Inches(5.6))
tb = slide8.shapes.add_textbox(Inches(0.9), Inches(6.5), Inches(5.6), Inches(0.4))
tb.text_frame.word_wrap = True
p = tb.text_frame.paragraphs[0]
p.text = "Figure 8: (a-b) Hausdorff distance, (c-d) Average Symmetric Surface Distance (ASSD)"
p.font.size = Pt(10)
p.font.color.rgb = TEXT_MUTED

# Right Column: Detailed Insights
add_card(slide8, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb = slide8.shapes.add_textbox(Inches(7.0), Inches(1.55), Inches(5.3), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Kitna Accurate Tha Model Over Time?"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_CRIMSON

p = tf.add_paragraph()
p.text = "Martens et al. ne 3 alag-alag metrics se prove kiya ki model future prediction mein reliable hai:"
p.font.size = Pt(12)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(6)

p = tf.add_paragraph()
p.text = "1. Mean Absolute Error (Figure 7):"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "t2 par MAE is below 0.01 (less than 1% cell error). Future mein (t3 = +90d, t4 = +180d) thoda badhta hai due to compounding error, par median remains around 0.03 (3% error). That is incredibly tight!"
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "2. ASSD - Sub-Voxel Accuracy (Figure 8c,d):"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_EMERALD
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "Average Symmetric Surface Distance (ASSD) 1.0 mm se chota tha for almost all cases! Matlab predicted boundary aur true boundary ke beech ka gap 1 voxel thickness se bhi kam hai!"
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "3. Hausdorff Distance & Outliers (Figure 8a,b):"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_PURPLE
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "Median Hausdorff distance was ~3 mm. Par kuch outliers the (asterisks up to 8-10 mm). Aisa kyu hua? Wo hum Slide 10 ke failure modes mein dekhenge!"
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED


# ==============================================================
# SLIDE 9: Parameter Concordance & Statistical Proof (Fig 10 & 11)
# ==============================================================
slide9 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide9, BG_LIGHT)
add_header(slide9, "Parameter Estimation Accuracy & Statistical Proof (Fig 10 & 11)")

# Left Column: Images Figure 10 & Figure 11
add_card(slide9, Inches(0.8), Inches(1.4), Inches(5.8), Inches(5.6))
fig10_path = os.path.join(figures_dir, 'figure_10.png')
fig11_path = os.path.join(figures_dir, 'figure_11.png')

slide9.shapes.add_picture(fig10_path, Inches(0.9), Inches(1.5), width=Inches(5.6))
tb = slide9.shapes.add_textbox(Inches(0.9), Inches(3.6), Inches(5.6), Inches(0.4))
tb.text_frame.word_wrap = True
p = tb.text_frame.paragraphs[0]
p.text = "Figure 10: Relative errors on lambda, v, d_white, and rho"
p.font.size = Pt(10)
p.font.color.rgb = TEXT_MUTED

slide9.shapes.add_picture(fig11_path, Inches(0.9), Inches(4.0), width=Inches(5.6))
tb = slide9.shapes.add_textbox(Inches(0.9), Inches(6.5), Inches(5.6), Inches(0.4))
tb.text_frame.word_wrap = True
p = tb.text_frame.paragraphs[0]
p.text = "Figure 11: True vs Predicted parameter scatterplots with Lin's Concordance (CCC)"
p.font.size = Pt(10)
p.font.color.rgb = TEXT_MUTED

# Right Column: Statistical Breakdown
add_card(slide9, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb = slide9.shapes.add_textbox(Inches(7.0), Inches(1.55), Inches(5.3), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Kya Neural Network Physics Parameters Sach Mein Seekha?"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE

p = tf.add_paragraph()
p.text = "Figure 10 aur 11 prove karte hain ki neural network ne koi random guess ya shortcut nahi maara, balki exact physics recover ki:"
p.font.size = Pt(12)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(6)

p = tf.add_paragraph()
p.text = "- Relative Error (Figure 10):"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "  - Proliferation rate rho error: Median only 2.75%!\n  - White matter diffusion d_white error: Median only 5.86%!\nYeh error medical physics ke standard acceptable range (<10%) se bohot chota hai."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "- Lin's Concordance Correlation (CCC) (Figure 11):"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_CRIMSON
p.space_before = Pt(8)
p = tf.add_paragraph()
p.text = "Scatterplots dekho: saare blue dots red identity line y = x ke upar chipke hue hain!\n  - CCC for lambda = 0.99 (Near Perfect!)\n  - CCC for rho = 0.99 (Near Perfect!)\n  - CCC for d_white = 0.97\n  - CCC for velocity v = 0.95"
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "Why This Gives Us Full Confidence:"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = ACCENT_EMERALD
p.space_before = Pt(10)
p = tf.add_paragraph()
p.text = "Is paper ne establish kar diya ki 2 imaging timepoints + time-gap se diffusion aur proliferation predict karna 100% mathematically feasible hai deep learning se!"
p.font.size = Pt(11)
p.font.color.rgb = NAVY_PRIMARY


# ==============================================================
# SLIDE 10: Real Patient Case Study & Failure Modes (Figure 12)
# ==============================================================
slide10 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide10, BG_LIGHT)
add_header(slide10, "Real Patient Case Study & Vulnerabilities (Figure 12)")

# Left Column: Image Figure 12
add_card(slide10, Inches(0.8), Inches(1.4), Inches(5.8), Inches(5.6))
fig12_path = os.path.join(figures_dir, 'figure_12.png')
slide10.shapes.add_picture(fig12_path, Inches(0.9), Inches(1.5), width=Inches(5.6))

# Right Column: Case study and Failure Modes
add_card(slide10, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
tb = slide10.shapes.add_textbox(Inches(7.0), Inches(1.55), Inches(5.3), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Real Glioblastoma Patient & Critical Failure Modes"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = ACCENT_CRIMSON

p = tf.add_paragraph()
p.text = "Figure 12 dikhata hai ek 69-year-old male IDH-wildtype GBM patient par application. Network ne realistic 3D cell density produce ki jo white matter tracts ke sath align ho rahi thi."
p.font.size = Pt(11)
p.font.color.rgb = NAVY_SECONDARY
p.space_before = Pt(4)

p = tf.add_paragraph()
p.text = "Paper Ne Admit Kiye 3 Big Pitfalls (Vulnerabilities):"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(8)

p = tf.add_paragraph()
p.text = "1. Phantom Tumor Foci (The Artifact Trap):"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = ACCENT_CRIMSON
p = tf.add_paragraph()
p.text = "Agar neural network normal tissue mein chota sa fake artifact (c = 0.03) bana deta hai, toh PDE ka exponential term rho*c*(1-c) use 180 days mein doosra poora tumor bana deta hai! Unhe threshold filter lagana pada to kill low-density noise."
p.font.size = Pt(10)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "2. Mass Effect Ignored:"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = ACCENT_PURPLE
p.space_before = Pt(4)
p = tf.add_paragraph()
p.text = "Tumor jab grow karta hai toh brain tissue ko physically push/squish karta hai. Reaction-diffusion PDE rigid grid manti hai, isliye physical brain shifting miss ho jati hai."
p.font.size = Pt(10)
p.font.color.rgb = TEXT_MUTED

p = tf.add_paragraph()
p.text = "3. Zero Real Longitudinal Validation:"
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE
p.space_before = Pt(4)
p = tf.add_paragraph()
p.text = "This is Martens' biggest weakness! Unhone sirf EK real patient par dikhaya (wo bhi single timepoint). Future growth prediction kabhi kisi real patient par validate hi nahi ki!"
p.font.size = Pt(10)
p.font.color.rgb = NAVY_PRIMARY


# ==============================================================
# SLIDE 11: Direct Comparison (Martens vs HybridTumorNet)
# ==============================================================
slide11 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide11, BG_LIGHT)
add_header(slide11, "Martens et al. (2022) vs. Our HybridTumorNet: Direct Comparison")

# Big Comparison Card
add_card(slide11, Inches(0.8), Inches(1.4), Inches(11.733), Inches(5.6))
tb = slide11.shapes.add_textbox(Inches(1.1), Inches(1.55), Inches(11.1), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Where Martens Succeeded vs. Where Our Project Takes It to the Next Level"
p.font.size = Pt(17)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY

headers = [
    ("FEATURE / DIMENSION", "MARTENS ET AL. (2022) PREPRINT", "OUR HYBRIDTUMORNET PROJECT"),
    ("Patient Dataset", "1,200 Synthetic Tumors on 6 Healthy Brains + 1 Real Patient", "LUMIERE Dataset: 91 Real GBM Patients across 600+ Longitudinal Scans"),
    ("Tumor Boundary Input", "Manual / Assumed perfect binary thresholds (c1=0.80, c2=0.16)", "SOTA SegResNet (~90%-96% Whole Tumor Dice on LUMIERE atlas)"),
    ("Model Architecture", "Two Disconnected Networks (3D U-Net + Separate Conv-Encoder)", "Unified Differentiable PyTorch PINN with Physics Parameter Head"),
    ("Forward Solver", "Offline C++/CUDA External Tool (TGSTK)", "In-Graph Differentiable RK4 / Euler PDE Solver directly in PyTorch"),
    ("Real Longitudinal Proof", "NO (Tested on synthetic; no real future scan validation)", "YES! Week 0 -> Week 44 real MRI scans to measure actual Dice score"),
    ("Research Impact", "Theoretical proof-of-concept on simulated data", "First full-scale clinical validation of Deep PDE modeling on LUMIERE")
]

for idx, (col1, col2, col3) in enumerate(headers):
    p = tf.add_paragraph()
    if idx == 0:
        p.text = f"{col1:<24} | {col2:<45} | {col3}"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = ACCENT_BLUE
        p.space_before = Pt(8)
    else:
        p.text = f"- {col1}:"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CRIMSON if idx == 1 or idx == 5 else NAVY_PRIMARY
        p.space_before = Pt(6)
        
        p_sub = tf.add_paragraph()
        p_sub.text = f"   Martens: {col2}\n   Ours:    {col3}"
        p_sub.font.size = Pt(10)
        p_sub.font.color.rgb = NAVY_SECONDARY if idx != 5 else ACCENT_EMERALD


# ==============================================================
# SLIDE 12: Our Action Plan (Upgrading the Blueprint)
# ==============================================================
slide12 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide12, BG_LIGHT)
add_header(slide12, "Our Game Plan: How We Upgrade Martens' Blueprint for LUMIERE")

# Step 1 Card
add_card(slide12, Inches(0.8), Inches(1.4), Inches(2.7), Inches(5.6))
tb = slide12.shapes.add_textbox(Inches(0.9), Inches(1.55), Inches(2.5), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "STEP 1"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = ACCENT_BLUE
p = tf.add_paragraph()
p.text = "Boundary Extraction via SegResNet"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(4)
p = tf.add_paragraph()
p.text = "Humare paas already pre-trained SegResNet hai (MONAI bundle) jo LUMIERE par 90-96% Whole Tumor Dice de raha hai.\n\nYeh model patient ke Week 0 scan se Enhancing Core (Gamma 1) aur Edema (Gamma 2) automatically extract kar dega."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED
p.space_before = Pt(10)

# Step 2 Card
add_card(slide12, Inches(3.8), Inches(1.4), Inches(2.7), Inches(5.6))
tb = slide12.shapes.add_textbox(Inches(3.9), Inches(1.55), Inches(2.5), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "STEP 2"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = ACCENT_PURPLE
p = tf.add_paragraph()
p.text = "Initial Cell Density Mapping"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(4)
p = tf.add_paragraph()
p.text = "Swanson aur Martens ke validated thresholds use karke:\n\n- Core voxels = 1.0 / 0.80\n- Edema voxels = 0.20 / 0.16\n- Background = 0.0\n\nGaussian smoothing apply karke initial 3D continuous density field u(x, t0) construct karenge."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED
p.space_before = Pt(10)

# Step 3 Card
add_card(slide12, Inches(6.8), Inches(1.4), Inches(2.7), Inches(5.6))
tb = slide12.shapes.add_textbox(Inches(6.9), Inches(1.55), Inches(2.5), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "STEP 3"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = ACCENT_CRIMSON
p = tf.add_paragraph()
p.text = "Physics Parameter Learning (D, rho)"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(4)
p = tf.add_paragraph()
p.text = "LUMIERE longitudinal scan pairs (Week 0 aur Week 44) feed karenge.\n\nHumara PhysicsParameterHead us patient ke specific diffusion D(x) aur proliferation rate rho ko estimate karega inside PyTorch using temporal PDE consistency loss."
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED
p.space_before = Pt(10)

# Step 4 Card
add_card(slide12, Inches(9.8), Inches(1.4), Inches(2.7), Inches(5.6))
tb = slide12.shapes.add_textbox(Inches(9.9), Inches(1.55), Inches(2.5), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "STEP 4"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = ACCENT_EMERALD
p = tf.add_paragraph()
p.text = "Real Future Validation & Publishing"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = NAVY_PRIMARY
p.space_before = Pt(4)
p = tf.add_paragraph()
p.text = "Predict_tumor_growth() se PDE solve karke Week 44 ka predicted tumor generate karenge.\n\nUsko real Week 44 follow-up MRI se compare karke Final Growth Prediction Dice Score nikalenge.\n\nYeh world-first real patient PINN validation hogi!"
p.font.size = Pt(11)
p.font.color.rgb = TEXT_MUTED
p.space_before = Pt(10)

# Save Final Presentation
out_pptx_path = r'C:\Users\harsh\OneDrive\Desktop\Cancer-Research\Martens_2022_Deep_Dive_Presentation.pptx'
prs.save(out_pptx_path)
print(f'Presentation saved successfully with {len(prs.slides)} slides at:\n{out_pptx_path}')
