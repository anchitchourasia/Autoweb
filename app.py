import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.extractor import extract_page_data
from core.screenshot import capture_screenshot
from core.generator import generate_clone_html
from ui.sidebar import render_sidebar
from ui.preview import render_preview

st.set_page_config(
    page_title="AutoWeb – AI Website Clone Generator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero-title {
    font-size: 2.8rem; font-weight: 800; line-height: 1.2;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.hero-sub { font-size: 1.05rem; color: #666; margin-bottom: 28px; }
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important; border: none; border-radius: 12px;
    padding: 12px 32px; font-size: 1rem; font-weight: 600;
    width: 100%; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Persistent session state ───────────────────────────────────────────────
for key in ["clone_html", "data", "shot_path", "agent_used", "agent_logs", "chars"]:
    if key not in st.session_state:
        st.session_state[key] = None

render_sidebar()

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌐 AutoWeb</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Paste any public URL. Three AI agents collaborate: '
    '<b>Gemini</b> builds structure → <b>Groq</b> perfects styling → '
    '<b>Ollama</b> finalizes & polishes.</div>',
    unsafe_allow_html=True
)

# FIXED
col_input, col_btn = st.columns([4, 1])
with col_input:
    url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        label_visibility="collapsed",
        key="url_input"
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    generate = st.button("⚡ Generate")

if generate:
    if not url.strip():
        st.error("Please enter a valid URL.")
    elif not url.startswith("http"):
        st.error("URL must start with http:// or https://")
    else:
        progress = st.progress(0, text="🚀 Starting pipeline...")

        with st.spinner("📡 Fetching page data..."):
            data, fetch_error = extract_page_data(url)
            progress.progress(15, text="✅ Page data extracted")
        if fetch_error:
            st.warning(f"Fetch warning (continuing): {fetch_error}")

        with st.spinner("📸 Capturing screenshot..."):
            shot_path, shot_error = capture_screenshot(url)
            progress.progress(30, text="✅ Screenshot captured")
        if shot_error:
            st.warning(f"Screenshot warning: {shot_error}")

        with st.spinner("🧠 Stage 1/3 — Gemini building structure..."):
            progress.progress(40, text="🧠 Stage 1: Gemini — HTML Architect...")
            result = generate_clone_html(data, shot_path=shot_path or "")
            progress.progress(100, text="✅ All agents done!")

        progress.empty()

        # ── Save to session state — persists across ALL tab switches ──────
        st.session_state["clone_html"] = result["html"]
        st.session_state["data"]       = data
        st.session_state["shot_path"]  = shot_path
        st.session_state["agent_used"] = result["agent_used"]
        st.session_state["agent_logs"] = result["logs"]
        st.session_state["chars"]      = result["chars"]

        st.success(f"✅ Pipeline complete: **{result['agent_used']}**")

# ── Render from session state — never loses data on tab switch ─────────────
if st.session_state["clone_html"]:
    render_preview(
        clone_html = st.session_state["clone_html"],
        data       = st.session_state["data"],
        shot_path  = st.session_state["shot_path"],
        agent_used = st.session_state["agent_used"],
        agent_logs = st.session_state["agent_logs"],
        chars      = st.session_state["chars"]
    )