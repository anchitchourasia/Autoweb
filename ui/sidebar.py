import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/web.png", width=64)
        st.markdown("## AutoWeb")
        st.markdown("AI-powered website clone generator using **Gemini 1.5 Flash**.")
        st.divider()

        st.markdown("### How it works")
        st.markdown("""
<div style='font-size:0.92rem;line-height:1.9'>
📡 <b>Step 1</b> – Fetches target URL<br>
📸 <b>Step 2</b> – Captures full-page screenshot<br>
🧠 <b>Step 3</b> – Extracts structure & content<br>
✨ <b>Step 4</b> – Gemini generates a clone
</div>
""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Tips")
        st.markdown("""
- Works best on public landing pages
- Avoid login-gated pages
- Simple marketing sites clone best
""")
        st.divider()
        st.caption("Built by Anchit Chourasia")
        st.caption("[GitHub](https://github.com/anchitchourasia) · [LinkedIn](https://linkedin.com/in/anchit-chourasia-65b603226)")