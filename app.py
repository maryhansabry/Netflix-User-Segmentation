"""
app.py — Netflix User Segmentation · Entry point
"""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Segmentation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8e8; }

section[data-testid="stSidebar"] {
    background: #111118 !important;
    border-right: 1px solid #222230;
}

.metric-card {
    background: linear-gradient(135deg, #16161f 0%, #1c1c28 100%);
    border: 1px solid #2a2a3d;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    margin-bottom: 12px;
}
.metric-card:hover { transform: translateY(-2px); border-color: #e50914; }
.metric-card .label { font-size: 11px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #888; margin-bottom: 8px; }
.metric-card .value { font-family: 'DM Serif Display', serif; font-size: 32px; color: #fff; line-height: 1; }
.metric-card .sub   { font-size: 12px; color: #e50914; margin-top: 6px; font-weight: 500; }

.cluster-badge      { display: inline-block; padding: 6px 18px; border-radius: 100px; font-size: 13px; font-weight: 600; letter-spacing: 0.5px; }
.badge-active       { background: #0f3d1e; color: #4ade80; border: 1px solid #22c55e40; }
.badge-inactive     { background: #3d0f0f; color: #f87171; border: 1px solid #ef444440; }
.badge-casual       { background: #1e2a3d; color: #60a5fa; border: 1px solid #3b82f640; }

.result-box {
    background: linear-gradient(135deg, #16161f, #1c1c28);
    border: 1px solid #2a2a3d;
    border-left: 4px solid #e50914;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 20px 0;
}
.result-box h2 { font-family: 'DM Serif Display', serif; font-size: 28px; color: #fff; margin: 0 0 8px 0; }
.result-box p  { color: #aaa; font-size: 14px; margin: 0; line-height: 1.6; }

.reco-item {
    background: #16161f;
    border: 1px solid #2a2a3d;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 14px;
    color: #ccc;
}

.chat-user {
    background: #1c2033;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 18px;
    margin: 8px 0;
    margin-left: 15%;
    color: #e0e0e0;
    font-size: 14px;
    line-height: 1.6;
}
.chat-ai {
    background: linear-gradient(135deg, #1a1020, #1c1830);
    border: 1px solid #332244;
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    margin-right: 10%;
    color: #e8e0f0;
    font-size: 14px;
    line-height: 1.7;
}
.chat-label      { font-size: 10px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
.chat-label.user { color: #60a5fa; }
.chat-label.ai   { color: #a78bfa; }

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #fff;
    margin: 32px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2a3d;
}

.stButton > button {
    background: #e50914 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover { background: #c2070f !important; transform: translateY(-1px) !important; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load models (cached) ───────────────────────────────────────────
from features import load_artifacts
scaler, kmeans, encoder = load_artifacts()

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Netflix Segmentation")
    st.markdown("---")
    mode = st.radio("**Mode**", ["👤 Single User", "📁 Batch Analysis"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**🤖 Gemini AI**")
    if GEMINI_API_KEY:
        st.success("✓ Gemini connected", icon="🤖")
    else:
        st.warning("GEMINI_API_KEY not set", icon="⚠️")
        st.caption("Add `GEMINI_API_KEY=...` to your `.env` file")
    st.markdown("---")
    st.caption("Autoencoder + K-Means · Keras 3")

# ── Route to page ──────────────────────────────────────────────────
if mode == "👤 Single User":
    from ui_single import render as render_single
    render_single(scaler, kmeans, encoder, gemini_api_key=GEMINI_API_KEY)
else:
    from ui_batch import render as render_batch
    render_batch(scaler, kmeans, encoder)
