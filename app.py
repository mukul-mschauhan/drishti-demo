"""
DRISHTI-SENTINEL | Fire & Smoke Detection System
Dhanush AI Innovation Pvt Ltd — BEL Evaluation Demo
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import random
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# --- WHATSAPP/SOCIAL PREVIEW FIX ---
# Replace 'YOUR_IMAGE_URL' with the direct link to your logo on GitHub 
# Example: https://raw.githubusercontent.com/mukul-mschauhan/drishti-demo/main/logo.png
LOGO_URL = "https://raw.githubusercontent.com/mukul-mschauhan/drishti-demo/main/Logos.png"

st.markdown(f"""
    <head>
        <!-- Primary Meta Tags -->
        <title>DRISHTI-SENTINEL</title>
        <meta name="title" content="DRISHTI-SENTINEL | Evaluation Dashboard">
        <meta name="description" content="Sovereign AI Fire & Smoke Detection System for Armoured Fighting Vehicles.">

        <!-- Open Graph / Facebook / WhatsApp -->
        <meta property="og:type" content="website">
        <meta property="og:url" content="https://drishti-demo.streamlit.app/">
        <meta property="og:title" content="DRISHTI-SENTINEL | Evaluation Dashboard">
        <meta property="og:description" content="Sovereign AI Fire & Smoke Detection System for AFVs.">
        <meta property="og:image" content="{LOGO_URL}">

        <!-- Schema.org for Google+ / WhatsApp -->
        <meta itemprop="name" content="DRISHTI-SENTINEL">
        <meta itemprop="description" content="Sovereign AI Fire & Smoke Detection System for AFVs.">
        <meta itemprop="image" content="{LOGO_URL}">

        <!-- Twitter -->
        <meta property="twitter:card" content="summary_large_image">
        <meta property="twitter:title" content="DRISHTI-SENTINEL">
        <meta property="twitter:description" content="Sovereign AI Fire & Smoke Detection System for AFVs.">
        <meta property="twitter:image" content="{LOGO_URL}">
    </head>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="DRISHTI-SENTINEL",
    page_icon="Logos.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── FONTS: Inter & Roboto Mono for a Tactical UI ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@500;700&display=swap');

* {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    box-sizing: border-box;
}

:root {
    --bg:           #ffffff;
    --bg-subtle:    #f4f5f7;
    --bg-panel:     #f9fafb;
    --border:       #d0d5dd;
    --border-light: #e4e7ec;
    --navy:         #0b1219; 
    --navy-2:       #131e2a;
    --navy-3:       #1e2d3d;
    --text-1:       #0f1923;
    --text-2:       #3d4f61;
    --text-3:       #6b7c8d;
    --text-4:       #98a8b8;
    --red:          #b91c1c;
    --red-bg:       #fef2f2;
    --red-border:   #fecaca;
    --blue:         #1d4ed8;
    --blue-bg:      #eff6ff;
    --blue-border:  #bfdbfe;
    --green:        #166534;
    --green-bg:     #f0fdf4;
    --green-border: #bbf7d0;
    --line:         #e4e7ec;
}

html, body, .main, [class*="css"] {
    background: var(--bg) !important;
    color: var(--text-1);
    font-size: 13px;
    line-height: 1.5;
}

/* ── LAYOUT & HEADER OVERRIDES ── */
[data-testid="stHeader"] {
    display: none !important;
}

.block-container {
    padding: 0 1.75rem 2rem 1.75rem !important;
    max-width: 1440px !important;
    margin-top: 0 !important;
}

/* ── TOPBAR ── */
.topbar {
    background: var(--navy);
    margin: 0 -1.75rem 0 -1.75rem;
    padding: 0 1.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 52px;
    border-bottom: 2px solid var(--red); 
}
.topbar-left {
    display: flex;
    align-items: center;
    gap: 14px;
}
.topbar-logo {
    width: 28px; height: 28px;
    background: var(--red);
    border-radius: 3px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.topbar-name {
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}
.topbar-sep {
    width: 1px; height: 18px;
    background: var(--navy-3);
}
.topbar-sub {
    font-size: 11px;
    color: #8c9ba8;
    letter-spacing: 0.5px;
}
.topbar-right {
    display: flex;
    align-items: center;
    gap: 20px;
}
.topbar-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: #8c9ba8;
    letter-spacing: 0.5px;
}
.dot {
    width: 6px; height: 6px;
    border-radius: 50%;
}
.dot-green { background: #22c55e; box-shadow: 0 0 5px #22c55e; animation: blink 2.5s infinite; }
.dot-blue  { background: #60a5fa; }
@keyframes blink { 0%,100%{opacity:1} 60%{opacity:0.25} }
.topbar-time {
    font-size: 11px;
    color: #6b7c8d;
    font-family: 'Roboto Mono', monospace !important;
    letter-spacing: 0.5px;
}

/* ── NATIVE TABS OVERRIDE (Replaces Subnav) ── */
div[data-testid="stTabs"] {
    margin: 0 -1.75rem 20px -1.75rem;
}
div[data-baseweb="tab-list"] {
    background: var(--bg-subtle) !important;
    padding: 0 1.75rem !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
button[data-baseweb="tab"] {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: var(--text-3) !important;
    padding: 12px 16px !important;
    border-bottom: 2px solid transparent !important;
    letter-spacing: 0.2px !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border-top: none !important;
    border-left: none !important;
    border-right: none !important;
    margin-right: 4px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--navy) !important;
    border-bottom-color: var(--red) !important;
    font-weight: 700 !important;
}
div[data-baseweb="tab-highlight"] {
    display: none !important; /* Hides default active highlight */
}

/* ── METRICS STRIP ── */
.metrics-strip {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 18px;
}
.metric-cell {
    background: var(--bg);
    padding: 12px 14px;
    position: relative;
}
.metric-cell::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.mc-overall::before { background: #166534; }
.mc-fire::before    { background: var(--red); }
.mc-smoke::before   { background: var(--blue); }
.mc-speed::before   { background: #7c3aed; }
.metric-cell-val {
    font-size: 20px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 3px;
}
.mc-overall .metric-cell-val { color: #166534; }
.mc-fire    .metric-cell-val { color: var(--red); }
.mc-smoke   .metric-cell-val { color: var(--blue); }
.mc-speed   .metric-cell-val { color: #7c3aed; }
.metric-cell-lbl {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--text-3);
    line-height: 1.2;
}
.metric-cell-sub {
    font-size: 10px;
    color: var(--text-4);
    margin-top: 3px;
}

/* ── SECTION RULE ── */
.sec-rule {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-rule::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--line);
}

/* ── SURGICALLY CLEANED NATIVE UPLOADER ── */
[data-testid="stFileUploader"] {
    width: 100%;
}
[data-testid="stFileUploader"] section {
    background-color: var(--bg-panel) !important;
    border: 1px dashed var(--text-4) !important;
    border-radius: 4px !important;
    padding: 32px 24px !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
[data-testid="stFileUploader"] section button { display: none !important; }
[data-testid="stFileUploader"] small { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: transparent !important; }
[data-testid="stFileUploaderDropzoneInstructions"]::before {
    content: "SECURE DROPZONE: CLICK OR DRAG FILE HERE";
    color: var(--text-3) !important;
    font-family: 'Roboto Mono', monospace !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: block;
    text-align: center;
}

/* ── DETECTION FRAME ── */
.det-frame {
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    background: #000;
}
.det-frame-header {
    background: var(--navy);
    padding: 8px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.det-frame-title {
    font-size: 11px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.det-frame-status {
    font-size: 10px;
    color: #22c55e;
    font-family: 'Roboto Mono', monospace !important;
    letter-spacing: 0.5px;
}

/* ── CARDS & TAGS ── */
.det-card { border: 1px solid var(--border-light); border-radius: 4px; padding: 12px 14px; margin-bottom: 8px; }
.det-card-fire  { border-left: 3px solid var(--red);  background: var(--red-bg); }
.det-card-smoke { border-left: 3px solid var(--blue); background: var(--blue-bg); }
.det-card-cls { font-size: 12px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; }
.det-card-fire .det-card-cls { color: var(--red); }
.det-card-smoke .det-card-cls { color: var(--blue); }
.det-row { display: flex; align-items: center; gap: 8px; margin-top: 7px; }
.det-lbl { font-size: 10.5px; color: var(--text-3); min-width: 72px; font-weight: 600; }
.bar-track { flex: 1; height: 5px; background: rgba(0,0,0,0.06); border-radius: 3px; overflow: hidden; }
.bar-fill-fire  { height:100%; background:var(--red);  border-radius:3px; }
.bar-fill-smoke { height:100%; background:var(--blue); border-radius:3px; }
.det-pct { font-size: 11px; font-weight: 700; min-width: 34px; text-align: right; font-family: 'Roboto Mono', monospace !important; }
.det-card-fire .det-pct { color: var(--red); }
.det-card-smoke .det-pct { color: var(--blue); }
.det-status-tag { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 2px; letter-spacing: 0.5px; text-transform: uppercase; margin-top: 6px; display: inline-block; }
.tag-alert { background:#fef2f2; color:var(--red); border:1px solid var(--red-border); }
.tag-info  { background:#f0f9ff; color:#0369a1; border:1px solid #bae6fd; }

.alert-item { border-radius: 4px; padding: 10px 12px; margin-bottom: 8px; display: flex; gap: 10px; align-items: flex-start; }
.alert-fire  { background:var(--red-bg);   border:1px solid var(--red-border);   border-left:3px solid var(--red); }
.alert-smoke { background:var(--blue-bg);  border:1px solid var(--blue-border);  border-left:3px solid var(--blue); }
.alert-clear { background:var(--green-bg); border:1px solid var(--green-border); border-left:3px solid #16a34a; }
.alert-title { font-size: 12px; font-weight: 700; letter-spacing: 0.3px; text-transform: uppercase; }
.alert-fire .alert-title { color: var(--red); }
.alert-smoke .alert-title { color: var(--blue); }
.alert-clear .alert-title { color: var(--green); }
.alert-body { font-size: 10.5px; color: var(--text-3); margin-top: 2px; font-family: 'Roboto Mono', monospace !important; }

.stat-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-bottom: 14px; }
.stat-cell { background: var(--bg-subtle); border: 1px solid var(--border-light); border-radius: 4px; padding: 10px 12px; text-align: center; }
.stat-cell-val { font-size: 18px; font-weight: 700; line-height: 1; }
.stat-cell-lbl { font-size: 9.5px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: var(--text-3); margin-top: 3px; }

/* ── DATA TABLES ── */
.log-tbl, .info-tbl { width:100%; border-collapse:collapse; }
.log-tbl th { font-size: 9.5px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: var(--text-3); padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border); background: var(--bg-subtle); }
.log-tbl td, .info-tbl td { font-size: 11.5px; padding: 6px 8px; border-bottom: 1px solid var(--border-light); }
.log-tbl td { font-family: 'Roboto Mono', monospace !important; color: var(--text-1); }
.log-tbl tr:last-child td, .info-tbl tr:last-child td { border-bottom: none; }
.log-tbl tr:hover td { background: var(--bg-subtle); }

.cls-badge { font-size: 9.5px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; padding: 1px 6px; border-radius: 2px; }
.cls-fire  { background:var(--red-bg);  color:var(--red);  border:1px solid var(--red-border); }
.cls-smoke { background:var(--blue-bg); color:var(--blue); border:1px solid var(--blue-border); }

.info-tbl td { padding: 6px 0; vertical-align: top; }
.info-k { color:var(--text-3); font-weight:700; width:45%; padding-right:10px; text-transform:uppercase; font-size: 10px !important;}
.info-v { color:var(--text-1); font-family:'Roboto Mono', monospace !important; font-size:11px !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] { background: var(--navy) !important; border-right: 1px solid var(--navy-3) !important; width: 240px !important; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div { color: #8c9ba8 !important; font-size: 12px !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #ffffff !important; font-size: 13px !important; font-weight: 700 !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
[data-testid="stSidebar"] hr { border-color: var(--navy-3) !important; }
[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] { display:none; }

.stRadio > div > label { font-size:12px !important; font-weight: 600 !important; color: var(--text-2) !important;}
.stRadio > div { gap:12px !important; }

/* ── OVERRIDES ── */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none !important; }
.spacer-sm { height: 10px; }
.spacer-md { height: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Auto-Reset Callback ───────────────────────────────────────────────────────
def reset_session_state():
    """Wipes the log and counts when a new file is uploaded, resets speed baseline."""
    st.session_state.log.clear()
    st.session_state.total_fire  = 0
    st.session_state.total_smoke = 0
    st.session_state.frames      = 0
    st.session_state.inf_ms      = 1.8  # Default hardware target baseline
    st.session_state.inf_fps     = 555

# ── Session state ─────────────────────────────────────────────────────────────
if 'log'         not in st.session_state: st.session_state.log         = deque(maxlen=200)
if 'total_fire'  not in st.session_state: st.session_state.total_fire  = 0
if 'total_smoke' not in st.session_state: st.session_state.total_smoke = 0
if 'frames'      not in st.session_state: st.session_state.frames      = 0
if 'inf_ms'      not in st.session_state: st.session_state.inf_ms      = 1.8 
if 'inf_fps'     not in st.session_state: st.session_state.inf_fps     = 555 

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### DRISHTI-SENTINEL")
    st.markdown('<p style="font-size:10px;color:#8c9ba8;letter-spacing:1.5px;margin-top:-10px;text-transform:uppercase;">Control Panel</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:2px;color:#4a6078;text-transform:uppercase;margin-bottom:6px;">Inference</p>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.90, 0.15, 0.05)
    iou_threshold  = st.slider("IoU Threshold",        0.10, 0.90, 0.45, 0.05)

    st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:2px;color:#4a6078;text-transform:uppercase;margin:10px 0 6px 0;">Alerts</p>', unsafe_allow_html=True)
    fire_alert  = st.checkbox("Enable Fire Alerts",  value=True)
    smoke_alert = st.checkbox("Enable Smoke Alerts", value=True)
    alert_conf  = st.slider("Alert Min Confidence", 0.10, 0.90, 0.40, 0.05)

    st.markdown("---")
    st.markdown('<p style="font-size:10px;color:#4a6078;line-height:1.9;letter-spacing:0.3px;">Dhanush AI Innovation Pvt Ltd<br>DRISHTI-SENTINEL v4.0<br>Fire &amp; Smoke Detection<br>Armoured Fighting Vehicles<br><span style="color:#8c9ba8;font-weight:600;">Confidential — Evaluation Only</span></p>', unsafe_allow_html=True)

# ── Top bar ───────────────────────────────────────────────────────────────────
now = datetime.now().strftime("%d %b %Y  %H:%M:%S")
st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">🔥</div>
    <span class="topbar-name">DRISHTI-SENTINEL</span>
    <div class="topbar-sep"></div>
    <span class="topbar-sub">Fire &amp; Smoke Detection System — Armoured Fighting Vehicles</span>
  </div>
  <div class="topbar-right">
    <span class="topbar-status"><span class="dot dot-green"></span>System Active</span>
    <span class="topbar-status"><span class="dot dot-blue"></span>GPU Ready</span>
    <span class="topbar-time">{now}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Native Tabs ───────────────────────────────────────────────────────────────
tab_console, tab_perf, tab_log, tab_config = st.tabs([
    "DETECTION CONSOLE", 
    "PERFORMANCE", 
    "SESSION LOG", 
    "CONFIGURATION"
])

# ── GLOBAL Placeholder: Metrics Strip ─────────────────────────────────────────
# Defined here to sit under the tabs, but populated at the bottom of the script
# so it always grabs the absolute latest inference speeds.
metrics_strip_placeholder = st.empty()

@st.cache_resource
def load_model():
    return YOLO(r"weights/best.pt")

model = load_model()

# ==============================================================================
# TAB 1: DETECTION CONSOLE
# ==============================================================================
with tab_console:
    left_col, right_col = st.columns([2.4, 1], gap="large")

    with left_col:
        mode = st.radio("", ["📷  Image", "📁  Batch Inference"], horizontal=True, label_visibility="collapsed")
        st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

        if mode == "📷  Image":
            uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png","bmp"], label_visibility="collapsed", on_change=reset_session_state)
            if uploaded:
                img     = Image.open(uploaded).convert("RGB")
                img_arr = np.array(img)

                # — Inference —
                results = model.predict(img_arr, conf=conf_threshold, iou=iou_threshold, device="cpu")
                
                # Dynamic Speed Tracking
                total_time_ms = sum(results[0].speed.values())
                st.session_state.inf_ms = total_time_ms
                st.session_state.inf_fps = 1000 / total_time_ms if total_time_ms > 0 else 0
                
                annotated_img = Image.fromarray(results[0].plot())
                detections = [(model.names[int(b.cls)], float(b.conf)) for b in results[0].boxes]

                st.markdown("""<div class="det-frame">
                  <div class="det-frame-header">
                    <span class="det-frame-title">Detection Output</span>
                    <span class="det-frame-status">● ANALYSIS COMPLETE</span>
                  </div>""", unsafe_allow_html=True)
                st.image(annotated_img, width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)

                st.session_state.frames += 1
                for cls, cf in detections:
                    ts = datetime.now().strftime("%H:%M:%S")
                    st.session_state.log.appendleft({'Time':ts, 'Class':cls, 'Confidence':cf})
                    if cls == 'Fire': st.session_state.total_fire  += 1
                    else:             st.session_state.total_smoke += 1

                st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

                if detections:
                    st.markdown('<div class="sec-rule">Detections — This Frame</div>', unsafe_allow_html=True)
                    dcols = st.columns(len(detections))
                    for i,(cls,cf) in enumerate(detections):
                        card_cls = "det-card-fire" if cls=="Fire" else "det-card-smoke"
                        fill_cls = "bar-fill-fire" if cls=="Fire" else "bar-fill-smoke"
                        tag_cls  = "tag-alert" if cf>=alert_conf else "tag-info"
                        tag_txt  = "ALERT — Above Threshold" if cf>=alert_conf else "Detected — Below Threshold"
                        with dcols[i]:
                            st.markdown(f"""
                            <div class="det-card {card_cls}">
                              <div class="det-card-cls">{cls}</div>
                              <div class="det-row">
                                <span class="det-lbl">Confidence</span>
                                <div class="bar-track"><div class="{fill_cls}" style="width:{cf*100:.0f}%"></div></div>
                                <span class="det-pct">{cf:.1%}</span>
                              </div>
                              <div style="margin-top:6px;"><span class="det-status-tag {tag_cls}">{tag_txt}</span></div>
                            </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="alert-item alert-clear">
                      <div><div class="alert-title">No Threat Detected</div>
                      <div class="alert-body">Compartment clear — no fire or smoke above confidence threshold.</div></div>
                    </div>""", unsafe_allow_html=True)

        else: # Batch Inference Logic
            uploaded = st.file_uploader("Batch Image Upload", type=["jpg","jpeg","png"], accept_multiple_files=True, label_visibility="collapsed", on_change=reset_session_state)
            if uploaded:
                st.caption(f"Processing {len(uploaded)} images for batch inference...")
                
                # Define a 4-column grid layout
                num_cols = 4
                cols = st.columns(num_cols)
                
                total_batch_time_ms = 0
                
                # Loop through ALL uploaded images
                for i, f in enumerate(uploaded):
                    # 1. Load Image
                    img = Image.open(f).convert("RGB")
                    img_arr = np.array(img)
                    
                    # 2. Run Inference
                    results = model.predict(img_arr, conf=conf_threshold, iou=iou_threshold, device="cpu")
                    
                    # Accumulate time for average speed calculation
                    total_batch_time_ms += sum(results[0].speed.values())
                    
                    annotated_img = Image.fromarray(results[0].plot())
                    detections = [(model.names[int(b.cls)], float(b.conf)) for b in results[0].boxes]
                    
                    # 3. Update Session State (Telemetry)
                    st.session_state.frames += 1
                    for cls, cf in detections:
                        ts = datetime.now().strftime("%H:%M:%S")
                        st.session_state.log.appendleft({'Time':ts, 'Class':cls, 'Confidence':cf})
                        if cls == 'Fire': st.session_state.total_fire  += 1
                        else:             st.session_state.total_smoke += 1
                        
                    # 4. Display in the grid (modulo operator wraps to next row)
                    col_idx = i % num_cols
                    with cols[col_idx]: 
                        st.image(annotated_img, width="stretch")
                        if detections:
                            # Added margin-bottom to create spacing between rows
                            st.markdown(f'<p style="font-size:10px; color:var(--red); font-weight:700; margin-bottom: 24px;">THREAT DETECTED</p>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<p style="font-size:10px; color:var(--green); font-weight:700; margin-bottom: 24px;">CLEAR</p>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="sec-rule">Threat Status</div>', unsafe_allow_html=True)
        if st.session_state.total_fire > 0:
            st.markdown(f"""<div class="alert-item alert-fire"><div><div class="alert-title">Fire Detected</div><div class="alert-body">Events: {st.session_state.total_fire} &nbsp;·&nbsp; Frames: {st.session_state.frames}</div></div></div>""", unsafe_allow_html=True)
        if st.session_state.total_smoke > 0:
            st.markdown(f"""<div class="alert-item alert-smoke"><div><div class="alert-title">Smoke Detected</div><div class="alert-body">Events: {st.session_state.total_smoke} &nbsp;·&nbsp; Frames: {st.session_state.frames}</div></div></div>""", unsafe_allow_html=True)
        if st.session_state.total_fire == 0 and st.session_state.total_smoke == 0:
            st.markdown("""<div class="alert-item alert-clear"><div><div class="alert-title">All Clear</div><div class="alert-body">No threats detected in current session.</div></div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="spacer-sm"></div><div class="sec-rule">Session Statistics</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-cell"><div class="stat-cell-val" style="color:#b91c1c">{st.session_state.total_fire}</div><div class="stat-cell-lbl">Fire Events</div></div>
          <div class="stat-cell"><div class="stat-cell-val" style="color:#1d4ed8">{st.session_state.total_smoke}</div><div class="stat-cell-lbl">Smoke Events</div></div>
          <div class="stat-cell"><div class="stat-cell-val" style="color:#0f1923">{st.session_state.frames}</div><div class="stat-cell-lbl">Frames</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="spacer-sm"></div><div class="sec-rule">Recent Detections</div>', unsafe_allow_html=True)
        if st.session_state.log:
            rows = ""
            for e in list(st.session_state.log)[:8]:
                badge = f'<span class="cls-badge cls-{e["Class"].lower()}">{e["Class"]}</span>'
                rows += f"<tr><td>{e['Time']}</td><td>{badge}</td><td>{e['Confidence']:.3f}</td></tr>"
            st.markdown(f"""<table class="log-tbl"><thead><tr><th>Time</th><th>Class</th><th>Conf</th></tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)
        else:
            st.markdown('<p style="font-size:11px;color:#98a8b8;padding:8px 0;font-family:\'Roboto Mono\',monospace;">No detections recorded.</p>', unsafe_allow_html=True)

        st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
        if st.button("Clear Session Data", width="stretch", key="clear_btn_console"):
            reset_session_state()
            st.rerun()

        st.markdown('<div class="spacer-sm"></div><div class="sec-rule">System & Model Card</div>', unsafe_allow_html=True)
        info = [
            ("Architecture",  "YOLO11n (Ultralytics)"),
            ("Parameters",    "2.58M"),
            ("GFLOPs",        "6.3"),
            ("Training Data", "31,781 AFV internal/external images"),
            ("Classes",       "0: Fire, 1: Smoke"),
            ("Inference",     f"{st.session_state.inf_ms:.1f}ms / frame (Live)"),
            ("Model Version", "v4 Fine-Tuned Data Augmentation"),
        ]
        rows = "".join(f'<tr><td class="info-k" style="width:35%">{k}</td><td class="info-v">{v}</td></tr>' for k,v in info)
        st.markdown(f'<table class="info-tbl"><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

# ==============================================================================
# TAB 2: PERFORMANCE
# ==============================================================================
with tab_perf:
    st.markdown("#### Validation Metrics (v4 Fine-Tuned)")
    st.markdown('<p style="font-size:12px;color:var(--text-3);margin-top:-10px;">Metrics evaluated on 5,200 held-out AFV interior and exterior frames.</p>', unsafe_allow_html=True)
    
    pcol1, pcol2 = st.columns(2, gap="large")
    with pcol1:
        st.markdown('<div class="sec-rule">Fire Detection Curves</div>', unsafe_allow_html=True)
        perf_fire = [("AP@50", 73.6, "#b91c1c"), ("AP@50-95", 48.2, "#dc2626"), ("Recall", 68.5, "#ef4444"), ("Precision", 71.3, "#f87171")]
        for lbl,val,color in perf_fire:
            st.markdown(f"""
            <div class="perf-block">
              <div class="perf-head-row"><span class="perf-name">{lbl}</span><span class="perf-num" style="color:{color}">{val}%</span></div>
              <div class="perf-track"><div class="perf-fill" style="width:{val}%;background:{color}"></div></div>
            </div>""", unsafe_allow_html=True)
            
    with pcol2:
        st.markdown('<div class="sec-rule">Smoke Detection Curves</div>', unsafe_allow_html=True)
        perf_smoke = [("AP@50", 52.1, "#1d4ed8"), ("AP@50-95", 34.6, "#2563eb"), ("Recall", 47.4, "#3b82f6"), ("Precision", 61.8, "#60a5fa")]
        for lbl,val,color in perf_smoke:
            st.markdown(f"""
            <div class="perf-block">
              <div class="perf-head-row"><span class="perf-name">{lbl}</span><span class="perf-num" style="color:{color}">{val}%</span></div>
              <div class="perf-track"><div class="perf-fill" style="width:{val}%;background:{color}"></div></div>
            </div>""", unsafe_allow_html=True)

# ==============================================================================
# TAB 3: SESSION LOG
# ==============================================================================
with tab_log:
    st.markdown("#### Complete Session Telemetry")
    if not st.session_state.log:
        st.info("Awaiting telemetry. Upload an image in the Detection Console to begin logging.")
    else:
        df = pd.DataFrame(st.session_state.log)
        colA, colB = st.columns([4, 1])
        with colA:
            st.dataframe(df, width="stretch", height=400)
        with colB:
            st.markdown('<div class="sec-rule">Export Data</div>', unsafe_allow_html=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Log",
                data=csv,
                file_name=f"drishti_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", width="stretch")
            st.markdown('<p style="font-size:11px;color:var(--text-4);margin-top:10px;">Export raw telemetry data for post-mission analysis and threshold tuning.</p>', unsafe_allow_html=True)

# ==============================================================================
# TAB 4: CONFIGURATION
# ==============================================================================
with tab_config:
    st.markdown("#### Hardware & Deployment Settings")
    st.markdown('<p style="font-size:12px;color:var(--text-3);margin-top:-10px;">Select target architecture for TensorRT export and deployment.</p>', unsafe_allow_html=True)
    
    ccol1, ccol2, ccol3 = st.columns(3, gap="large")
    with ccol1:
        st.markdown('<div class="sec-rule">Hardware Target</div>', unsafe_allow_html=True)
        deploy_target = st.selectbox("Edge Device", ["Jetson Orin Nano Super", "Windows PC", "Jetson AGX Orin"], label_visibility="collapsed")
        st.caption(f"Currently targeting: **{deploy_target}**")
        
    with ccol2:
        st.markdown('<div class="sec-rule">Precision Engine</div>', unsafe_allow_html=True)
        precision_mode = st.selectbox("Quantization",  ["FP32 (PyTorch)", "FP16 (TensorRT AMP)", "INT8 (TensorRT Calibration)"], label_visibility="collapsed")
        st.caption(f"Export format: **{precision_mode}**")
        
    with ccol3:
        st.markdown('<div class="sec-rule">Video Streams</div>', unsafe_allow_html=True)
        st.text_input("RTSP Camera IP", value="rtsp://192.168.1.100:554/stream1", disabled=True)
        st.caption("Live streaming disabled in Demo Mode.")

    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sec-rule">Model Architecture Details</div>', unsafe_allow_html=True)
    info_conf = [
        ("Architecture",  "YOLO11n (Ultralytics)"),
        ("Parameters",    "2.58M"),
        ("GFLOPs",        "6.3"),
        ("Training Data", "31,781 AFV internal/external images"),
        ("Classes",       "0: Fire, 1: Smoke"),
        ("Inference",     f"{st.session_state.inf_ms:.1f}ms / frame (Live)"),
        ("Model Version", "v4 Fine-Tuned Data Augmentation"),
    ]
    rows_conf = "".join(f'<tr><td class="info-k" style="width:25%">{k}</td><td class="info-v">{v}</td></tr>' for k,v in info_conf)
    st.markdown(f'<table class="info-tbl"><tbody>{rows_conf}</tbody></table>', unsafe_allow_html=True)


# ── POPULATE GLOBAL METRICS STRIP ─────────────────────────────────────────────
# This runs after all inference is complete, injecting the absolute latest
# speed and FPS numbers back into the top placeholder.
metrics_strip_placeholder.markdown(f"""
<div class="metrics-strip">
  <div class="metric-cell mc-overall">
    <div class="metric-cell-val">62.9%</div>
    <div class="metric-cell-lbl">Overall mAP@50</div>
    <div class="metric-cell-sub">+13.1 pts vs baseline</div>
  </div>
  <div class="metric-cell mc-fire">
    <div class="metric-cell-val">73.6%</div>
    <div class="metric-cell-lbl">Fire AP@50</div>
    <div class="metric-cell-sub">Recall 68.5%</div>
  </div>
  <div class="metric-cell mc-fire">
    <div class="metric-cell-val">71.3%</div>
    <div class="metric-cell-lbl">Fire Precision</div>
    <div class="metric-cell-sub">F1 0.699</div>
  </div>
  <div class="metric-cell mc-smoke">
    <div class="metric-cell-val">52.1%</div>
    <div class="metric-cell-lbl">Smoke AP@50</div>
    <div class="metric-cell-sub">+38.7 pts vs baseline</div>
  </div>
  <div class="metric-cell mc-smoke">
    <div class="metric-cell-val">61.8%</div>
    <div class="metric-cell-lbl">Smoke Precision</div>
    <div class="metric-cell-sub">Recall 47.4%</div>
  </div>
  <div class="metric-cell mc-speed">
    <div class="metric-cell-val">{st.session_state.inf_ms:.1f}ms</div>
    <div class="metric-cell-lbl">Inference / Frame</div>
    <div class="metric-cell-sub">~{st.session_state.inf_fps:.0f} FPS (Live Streamlit)</div>
  </div>
  <div class="metric-cell mc-speed">
    <div class="metric-cell-val">31.8K</div>
    <div class="metric-cell-lbl">Training Images</div>
    <div class="metric-cell-sub">3 dataset sources</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style="background:var(--navy);border-radius:4px;padding:10px 18px;
            display:flex;justify-content:space-between;align-items:center;">
  <span style="font-size:10px;color:#4a6078;font-family:'Roboto Mono',monospace;letter-spacing:0.5px;">
    DRISHTI-SENTINEL © 2026 · Dhanush AI Innovation Pvt Ltd · All rights reserved
  </span>
  <span style="font-size:10px;color:#4a6078;font-family:'Roboto Mono',monospace;letter-spacing:0.5px;">
    Confidential — For authorised evaluation only
  </span>
</div>
""", unsafe_allow_html=True)
