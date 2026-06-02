import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import base64

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=" CIFAR-10 Vision System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --border:    #1e2330;
    --accent:    #00e5ff;
    --accent2:   #ff3cac;
    --accent3:   #b8ff3c;
    --text:      #e8eaf0;
    --muted:     #5a6070;
    --radius:    8px;
}

html, body, [class*="css"] { background: var(--bg); color: var(--text); }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* Typography */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }
p, li, span, div, label, code { font-family: 'Space Mono', monospace !important; font-size: 0.78rem; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #0a1628 50%, #0d1117 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at 80% 50%, rgba(0,229,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-family: 'Space Mono', monospace !important;
    color: var(--muted);
    font-size: 0.8rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.3);
    color: var(--accent);
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-right: 6px;
    margin-top: 12px;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-accent { border-top: 2px solid var(--accent); }
.card-accent2 { border-top: 2px solid var(--accent2); }
.card-accent3 { border-top: 2px solid var(--accent3); }

/* Section headers */
.section-label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
    display: block;
}
.section-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    margin-bottom: 0.3rem;
}

/* Prediction bar */
.pred-bar-wrap { margin: 6px 0; }
.pred-label {
    display: flex; justify-content: space-between;
    font-size: 0.7rem; color: var(--muted); margin-bottom: 2px;
}
.pred-bar-bg {
    background: var(--border);
    border-radius: 2px; height: 8px; overflow: hidden;
}
.pred-bar-fill {
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width 0.6s ease;
}

/* Info box */
.info-box {
    background: rgba(0,229,255,0.05);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.8rem 1rem;
    margin: 1rem 0;
    font-size: 0.75rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace !important;
}

/* Streamlit widget overrides */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #0099bb) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.stSelectbox > div > div,
.stSlider > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    color: var(--accent) !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--muted) !important;
    text-transform: uppercase; letter-spacing: 0.1em;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.05em;
    color: var(--muted) !important;
    padding: 0.7rem 1.4rem;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── imports (lazy so Streamlit can boot fast) ─────────────────────────────────
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">Computer Vision </div>
  <div class="hero-sub">CIFAR-10 · Neural Classification · Morphological Analysis · Image Fusion</div>
  <span class="badge">CIFAR-10 Dataset</span>
</div>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="section-label">Navigation</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Lab Modules</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-family: Space Mono, monospace; font-size:0.7rem; color:#5a6070; line-height:2">
    📌 Tab 1 — Image Classifier<br>
    📌 Tab 2 — Morphological Ops<br>
    📌 Tab 3 — Hit-or-Miss<br>
    📌 Tab 4 — Image Fusion<br>
    📌 Tab 5 — Model Info
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    Lab Instructor<br><b style="color:#e8eaf0">Sania Akhtar</b><br><br>
    sania.akhtar@students<br>.au.edu.pk
    </div>
    """, unsafe_allow_html=True)

# ── tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Classifier",
    "  Morphological Ops",
    "  Hit-or-Miss",
    "  Image Fusion",
    "  Model Info",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<span class="section-label">Module 01</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">CIFAR-10 Image Classifier</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    CNN trained on CIFAR-10 (60,000 images · 10 classes). Upload any image to classify it.
    Morphological preprocessing (erosion/dilation) is applied before inference.
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
        st.markdown("**Upload Image**")
        uploaded = st.file_uploader(
            "Choose an image file", type=["png", "jpg", "jpeg", "bmp"],
            key="classifier_upload"
        )

        apply_morph = st.checkbox("Apply morphological preprocessing", value=True)
        morph_op = st.selectbox("Preprocessing operation", ["Erosion", "Dilation", "Both (open)"])
        kernel_size = st.slider("Kernel size", 3, 9, 3, step=2)

        run_btn = st.button("▶  Run Classification", key="run_cls")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if uploaded and run_btn:
            with st.spinner("Loading model & classifying…"):
                try:
                    import tensorflow as tf
                    from model import load_or_train_model, CLASSES
                    model = load_or_train_model()

                    img = Image.open(uploaded).convert("RGB")
                    img_np = np.array(img)

                    # morphological preprocessing
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    preprocessed = img_np.copy()
                    if apply_morph:
                        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                        if morph_op == "Erosion":
                            proc_gray = cv2.erode(gray, kernel)
                        elif morph_op == "Dilation":
                            proc_gray = cv2.dilate(gray, kernel)
                        else:
                            proc_gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
                        preprocessed = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2RGB)

                    # resize → 32×32 for CIFAR-10
                    inp = cv2.resize(preprocessed, (32, 32)).astype("float32") / 255.0
                    inp = np.expand_dims(inp, 0)

                    preds = model.predict(inp, verbose=0)[0]
                    top_idx = int(np.argmax(preds))

                    st.markdown('<div class="card card-accent2">', unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(img, caption="Original", use_container_width=True)
                    with c2:
                        st.image(preprocessed, caption=f"After {morph_op}", use_container_width=True)

                    st.markdown(f"""
                    <div style="text-align:center; padding:1rem 0">
                      <div style="font-family:Space Mono,monospace;font-size:0.65rem;
                                  color:#5a6070;text-transform:uppercase;letter-spacing:.15em">
                        Prediction
                      </div>
                      <div style="font-family:Syne,sans-serif;font-size:2.2rem;
                                  font-weight:800;color:#00e5ff;letter-spacing:-.02em">
                        {CLASSES[top_idx].upper()}
                      </div>
                      <div style="font-family:Space Mono,monospace;font-size:0.75rem;color:#5a6070">
                        confidence: {preds[top_idx]*100:.1f}%
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**Top-5 Probabilities**")
                    top5 = np.argsort(preds)[::-1][:5]
                    for idx in top5:
                        pct = preds[idx] * 100
                        st.markdown(f"""
                        <div class="pred-bar-wrap">
                          <div class="pred-label"><span>{CLASSES[idx]}</span><span>{pct:.1f}%</span></div>
                          <div class="pred-bar-bg">
                            <div class="pred-bar-fill" style="width:{pct}%"></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {e}")
        elif uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", use_container_width=True)
            st.info("Click **▶ Run Classification** to classify.")
        else:
            st.markdown('<div class="card" style="text-align:center;padding:3rem 1rem">', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:Syne,sans-serif;font-size:3rem;margin-bottom:.5rem">🖼️</div>
            <div style="font-family:Space Mono,monospace;font-size:0.7rem;color:#5a6070">
            Upload an image to begin classification
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MORPHOLOGICAL OPS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<span class="section-label">Module 02 — Lab Tasks 5 & 6</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Morphological Operations</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Apply erosion, dilation, opening, closing, and morphological gradient on any image.
    The gradient formula: <b>Gradient = Dilation − Erosion</b> highlights object edges.
    </div>
    """, unsafe_allow_html=True)

    morph_upload = st.file_uploader("Upload image for morphological analysis",
                                     type=["png","jpg","jpeg","bmp"], key="morph_up")

    col_m1, col_m2 = st.columns([1,2], gap="large")
    with col_m1:
        st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
        ops = st.multiselect(
            "Select operations to apply",
            ["Original (Grayscale)", "Erosion", "Dilation",
             "Opening (Erode→Dilate)", "Closing (Dilate→Erode)", "Morphological Gradient"],
            default=["Original (Grayscale)", "Erosion", "Dilation", "Morphological Gradient"]
        )
        k = st.slider("Kernel size", 3, 15, 5, step=2, key="morph_k")
        k_shape = st.selectbox("Kernel shape", ["Rectangle", "Ellipse", "Cross"])
        iterations = st.slider("Iterations", 1, 5, 1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m2:
        if morph_upload:
            img_m = np.array(Image.open(morph_upload).convert("RGB"))
            gray_m = cv2.cvtColor(img_m, cv2.COLOR_RGB2GRAY)

            shapes = {"Rectangle": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Cross": cv2.MORPH_CROSS}
            kern = cv2.getStructuringElement(shapes[k_shape], (k, k))

            results = {}
            for op in ops:
                if op == "Original (Grayscale)": results[op] = gray_m
                elif op == "Erosion":            results[op] = cv2.erode(gray_m, kern, iterations=iterations)
                elif op == "Dilation":           results[op] = cv2.dilate(gray_m, kern, iterations=iterations)
                elif op == "Opening (Erode→Dilate)":  results[op] = cv2.morphologyEx(gray_m, cv2.MORPH_OPEN, kern)
                elif op == "Closing (Dilate→Erode)":  results[op] = cv2.morphologyEx(gray_m, cv2.MORPH_CLOSE, kern)
                elif op == "Morphological Gradient":
                    d = cv2.dilate(gray_m, kern)
                    e = cv2.erode(gray_m, kern)
                    results[op] = cv2.subtract(d, e)

            if results:
                cols = st.columns(min(len(results), 3))
                for i, (name, img_r) in enumerate(results.items()):
                    with cols[i % 3]:
                        st.image(img_r, caption=name, use_container_width=True, clamp=True)
        else:
            st.markdown('<div class="card" style="text-align:center;padding:3rem">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:2.5rem">🔬</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:Space Mono,monospace;font-size:.7rem;color:#5a6070">Upload an image to begin</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HIT-OR-MISS TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="section-label">Module 03 — Lab Task 1</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Hit-or-Miss Transform</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Detects specific structural patterns in binary images using two structuring elements:
    <b>B1 (Hit)</b> matches foreground pixels · <b>B2 (Miss)</b> matches background pixels.<br>
    Formula: <b>HMT(A) = Erode(A, B1) ∩ Erode(Aᶜ, B2)</b>
    </div>
    """, unsafe_allow_html=True)

    hom_upload = st.file_uploader("Upload binary/grayscale image",
                                   type=["png","jpg","jpeg","bmp"], key="hom_up")

    col_h1, col_h2 = st.columns([1, 2], gap="large")
    with col_h1:
        st.markdown('<div class="card card-accent2">', unsafe_allow_html=True)
        st.markdown("**B1 (Hit) Structuring Element**")
        st.markdown('<div style="font-size:.65rem;color:#5a6070;margin-bottom:.5rem">3×3 pattern — check cells that must be foreground</div>', unsafe_allow_html=True)

        b1_default = [[0,1,0],[1,1,1],[0,1,0]]
        b2_default = [[1,0,1],[0,0,0],[1,0,1]]

        b1 = []
        for r in range(3):
            row_vals = []
            c1,c2,c3 = st.columns(3)
            for ci, col_c in enumerate([c1,c2,c3]):
                v = col_c.checkbox("", value=bool(b1_default[r][ci]),
                                   key=f"b1_{r}_{ci}", label_visibility="collapsed")
                row_vals.append(int(v))
            b1.append(row_vals)

        st.markdown("**B2 (Miss) Structuring Element**")
        st.markdown('<div style="font-size:.65rem;color:#5a6070;margin-bottom:.5rem">3×3 pattern — check cells that must be background</div>', unsafe_allow_html=True)
        b2 = []
        for r in range(3):
            row_vals = []
            c1,c2,c3 = st.columns(3)
            for ci, col_c in enumerate([c1,c2,c3]):
                v = col_c.checkbox("", value=bool(b2_default[r][ci]),
                                   key=f"b2_{r}_{ci}", label_visibility="collapsed")
                row_vals.append(int(v))
            b2.append(row_vals)

        threshold = st.slider("Binarization threshold", 0, 255, 127)
        run_hom = st.button("▶  Apply HMT", key="run_hom")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_h2:
        if hom_upload and run_hom:
            img_h = np.array(Image.open(hom_upload).convert("L"))
            _, binary = cv2.threshold(img_h, threshold, 255, cv2.THRESH_BINARY)

            B1 = np.array(b1, dtype=np.uint8)
            B2 = np.array(b2, dtype=np.uint8)

            # HMT = erode(A, B1) ∩ erode(complement(A), B2)
            hit   = cv2.erode(binary, B1)
            compl = cv2.bitwise_not(binary)
            miss  = cv2.erode(compl, B2)
            hmt   = cv2.bitwise_and(hit, miss)

            c1,c2,c3,c4 = st.columns(4)
            with c1: st.image(binary,   caption="Binary Input",  use_container_width=True)
            with c2: st.image(hit,      caption="Hit (erode A·B1)",  use_container_width=True)
            with c3: st.image(miss,     caption="Miss (erode Aᶜ·B2)", use_container_width=True)
            with c4: st.image(hmt,      caption="HMT Result",  use_container_width=True)

            n_detected = int(np.sum(hmt > 0))
            st.markdown(f"""
            <div class="card" style="margin-top:1rem">
              <span style="font-family:Space Mono,monospace;font-size:.7rem;color:#5a6070">
              Pattern occurrences detected:
              </span>
              <span style="font-family:Syne,sans-serif;font-size:1.8rem;
                           font-weight:800;color:#00e5ff;margin-left:1rem">
              {n_detected}
              </span>
            </div>
            """, unsafe_allow_html=True)

        elif hom_upload:
            st.info("Configure structuring elements and click **▶ Apply HMT**")
        else:
            st.markdown('<div class="card" style="text-align:center;padding:3rem">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:2.5rem">🎯</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:Space Mono,monospace;font-size:.7rem;color:#5a6070">Upload a binary image to begin</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — IMAGE FUSION
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<span class="section-label">Module 04 — Lab Task 3</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Image Fusion</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Combines multiple images into a single informative image.
    Pixel-level fusion (weighted average · max · min) | Feature-level (Laplacian pyramid).
    </div>
    """, unsafe_allow_html=True)

    fc1, fc2 = st.columns(2, gap="large")
    with fc1:
        img_a = st.file_uploader("Upload Image A", type=["png","jpg","jpeg","bmp"], key="fuse_a")
    with fc2:
        img_b = st.file_uploader("Upload Image B", type=["png","jpg","jpeg","bmp"], key="fuse_b")

    col_f1, col_f2 = st.columns([1, 2], gap="large")
    with col_f1:
        st.markdown('<div class="card card-accent3">', unsafe_allow_html=True)
        fusion_type = st.selectbox("Fusion Method", [
            "Pixel-level · Weighted Average",
            "Pixel-level · Maximum",
            "Pixel-level · Minimum",
            "Feature-level · Laplacian Pyramid",
        ])
        alpha = st.slider("Weight α (Image A)", 0.0, 1.0, 0.5, 0.05,
                          disabled="Weighted" not in fusion_type)
        run_fuse = st.button("▶  Fuse Images", key="run_fuse")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_f2:
        if img_a and img_b and run_fuse:
            SIZE = (256, 256)
            A = cv2.resize(np.array(Image.open(img_a).convert("RGB")), SIZE).astype(np.float32)
            B = cv2.resize(np.array(Image.open(img_b).convert("RGB")), SIZE).astype(np.float32)

            if "Weighted" in fusion_type:
                fused = (alpha * A + (1 - alpha) * B).clip(0, 255).astype(np.uint8)
                label = f"Weighted Average (α={alpha:.2f})"

            elif "Maximum" in fusion_type:
                fused = np.maximum(A, B).astype(np.uint8)
                label = "Pixel Maximum Fusion"

            elif "Minimum" in fusion_type:
                fused = np.minimum(A, B).astype(np.uint8)
                label = "Pixel Minimum Fusion"

            else:
                # Laplacian pyramid fusion (3 levels)
                def laplacian_pyramid_fusion(imgA, imgB, levels=4):
                    gpA, gpB = [imgA.copy()], [imgB.copy()]
                    for _ in range(levels):
                        gpA.append(cv2.pyrDown(gpA[-1]))
                        gpB.append(cv2.pyrDown(gpB[-1]))
                    lpA = [gpA[levels]]
                    lpB = [gpB[levels]]
                    for i in range(levels, 0, -1):
                        laA = cv2.subtract(gpA[i-1], cv2.pyrUp(gpA[i], dstsize=(gpA[i-1].shape[1], gpA[i-1].shape[0])))
                        laB = cv2.subtract(gpB[i-1], cv2.pyrUp(gpB[i], dstsize=(gpB[i-1].shape[1], gpB[i-1].shape[0])))
                        lpA.append(laA); lpB.append(laB)
                    ls = [0.5 * la + 0.5 * lb for la, lb in zip(lpA, lpB)]
                    result = ls[0]
                    for i in range(1, levels+1):
                        result = cv2.add(cv2.pyrUp(result, dstsize=(ls[i].shape[1], ls[i].shape[0])), ls[i])
                    return result.clip(0, 255).astype(np.uint8)

                fused = laplacian_pyramid_fusion(A, B)
                label = "Laplacian Pyramid Fusion"

            r1,r2,r3 = st.columns(3)
            with r1: st.image(A.astype(np.uint8), caption="Image A", use_container_width=True)
            with r2: st.image(B.astype(np.uint8), caption="Image B", use_container_width=True)
            with r3: st.image(fused, caption=label, use_container_width=True)

            # metrics
            m1,m2,m3 = st.columns(3)
            diff = np.abs(A - B).mean()
            psnr_val = cv2.PSNR(A.astype(np.uint8), fused)
            fused_mean = fused.mean()
            with m1: st.metric("Mean Pixel Diff (A vs B)", f"{diff:.2f}")
            with m2: st.metric("PSNR (A vs Fused)", f"{psnr_val:.2f} dB")
            with m3: st.metric("Fused Mean Intensity", f"{fused_mean:.2f}")

        elif img_a or img_b:
            st.info("Upload both images, then click **▶ Fuse Images**")
        else:
            st.markdown('<div class="card" style="text-align:center;padding:3rem">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:2.5rem">🌀</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:Space Mono,monospace;font-size:.7rem;color:#5a6070">Upload two images to begin fusion</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<span class="section-label">Module 05</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Architecture & Training Info</div>', unsafe_allow_html=True)

    col_i1, col_i2 = st.columns(2, gap="large")

    with col_i1:
        st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
        st.markdown("**CNN Architecture**")
        arch = """
Block 1 · Conv2D(32, 3×3, ReLU) + BN
        · Conv2D(32, 3×3, ReLU) + BN
        · MaxPool(2×2) + Dropout(0.25)

Block 2 · Conv2D(64, 3×3, ReLU) + BN
        · Conv2D(64, 3×3, ReLU) + BN
        · MaxPool(2×2) + Dropout(0.25)

Block 3 · Conv2D(128, 3×3, ReLU) + BN
        · Conv2D(128, 3×3, ReLU) + BN
        · MaxPool(2×2) + Dropout(0.25)

Dense  · Flatten → Dense(512, ReLU) + BN
       · Dropout(0.5)
       · Dense(10, Softmax)
"""
        st.code(arch, language="text")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_i2:
        st.markdown('<div class="card card-accent2">', unsafe_allow_html=True)
        st.markdown("**Training Configuration**")
        cfg = {
            "Dataset": "CIFAR-10 (Kaggle)",
            "Train samples": "50,000",
            "Test samples": "10,000",
            "Image size": "32 × 32 × 3",
            "Classes": "10",
            "Optimizer": "Adam (lr=1e-3)",
            "Loss": "Categorical Crossentropy",
            "Epochs": "30 (EarlyStopping)",
            "Augmentation": "Flip, Rotate, Zoom",
            "Target Accuracy": "≥ 80%",
        }
        for k, v in cfg.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        border-bottom:1px solid #1e2330;padding:4px 0;
                        font-family:Space Mono,monospace;font-size:.72rem">
              <span style="color:#5a6070">{k}</span>
              <span style="color:#e8eaf0">{v}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:1rem">', unsafe_allow_html=True)
    st.markdown("**CIFAR-10 Classes**")
    classes = ["✈️ Airplane","🚗 Automobile","🐦 Bird","🐱 Cat","🦌 Deer",
               "🐶 Dog","🐸 Frog","🐴 Horse","🚢 Ship","🚚 Truck"]
    cols_cls = st.columns(5)
    for i, cls in enumerate(classes):
        with cols_cls[i % 5]:
            st.markdown(f"""
            <div style="background:#111318;border:1px solid #1e2330;border-radius:6px;
                        padding:.5rem;text-align:center;font-family:Space Mono,monospace;
                        font-size:.65rem;color:#e8eaf0;margin:.2rem 0">
              {cls}
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:1rem">', unsafe_allow_html=True)
    st.markdown("**Morphological Preprocessing Pipeline**")
    st.code("""
# Applied before every inference call
kernel = cv2.getStructuringElement(MORPH_RECT, (k, k))

# Erosion  → removes noise, shrinks bright regions
eroded = cv2.erode(image, kernel, iterations=1)

# Dilation → fills gaps, expands bright regions  
dilated = cv2.dilate(image, kernel, iterations=1)

# Opening (default) = Erosion → Dilation
preprocessed = cv2.morphologyEx(image, MORPH_OPEN, kernel)
    """, language="python")
    st.markdown('</div>', unsafe_allow_html=True)
