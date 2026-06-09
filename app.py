import streamlit as st
from core.extractor import extract_page_data
from core.screenshot import capture_screenshot
from core.generator import generate_clone_html
from ui.sidebar import render_sidebar
from ui.preview import render_preview
from dotenv import load_dotenv

load_dotenv()

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
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub { font-size: 1.15rem; color: #555; margin-bottom: 30px; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 12px;
        padding: 12px 32px; font-size: 1rem;
        font-weight: 600; width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

render_sidebar()

st.markdown('<div class="hero-title">🌐 AutoWeb</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Paste any public website URL. AutoWeb captures, analyses, and generates a clone using Gemini AI.</div>', unsafe_allow_html=True)

col_input, col_help = st.columns([3, 1])
with col_input:
    url = st.text_input("", placeholder="https://example.com", label_visibility="collapsed")
with col_help:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Works best on public landing pages")

if st.button("⚡ Generate Clone"):
    if not url.strip():
        st.error("Please enter a valid URL.")
    elif not url.startswith("http"):
        st.error("URL must start with http:// or https://")
    else:
        progress = st.progress(0, text="Starting...")

        with st.spinner("📡 Fetching page data..."):
            data, fetch_error = extract_page_data(url)
            progress.progress(33, text="Page data extracted...")

        if fetch_error:
            st.warning(f"Partial fetch: {fetch_error}")

        with st.spinner("📸 Capturing screenshot..."):
            shot_path, shot_error = capture_screenshot(url)
            progress.progress(66, text="Screenshot captured...")

        if shot_error:
            st.warning(f"Screenshot issue: {shot_error}")

        with st.spinner("🤖 Gemini is generating your clone..."):
            clone_html = generate_clone_html(data)
            progress.progress(100, text="Done!")

        progress.empty()

        st.session_state["data"] = data
        st.session_state["clone_html"] = clone_html
        st.session_state["shot_path"] = shot_path
        st.success("✅ Clone generated successfully!")

if "clone_html" in st.session_state:
    render_preview(
        st.session_state["clone_html"],
        st.session_state["data"],
        st.session_state.get("shot_path")
    )