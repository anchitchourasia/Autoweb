import streamlit as st
from core.agent_chain import get_agent_status
from datetime import datetime


def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/web.png", width=60)
        st.markdown("## AutoWeb")
        st.markdown("AI-powered website clone generator using a **smart multi-agent chain**.")
        st.divider()

        st.markdown("### How it works")
        st.markdown("""
<div style='font-size:0.9rem;line-height:2.0'>
📡 <b>Step 1</b> — Fetches target URL<br>
📷 <b>Step 2</b> — Captures full-page screenshot<br>
🔍 <b>Step 3</b> — Extracts structure & content<br>
🤖 <b>Step 4</b> — A2A agent chain generates clone
</div>
""", unsafe_allow_html=True)
        st.divider()

        # Live agent health panel
        st.markdown("### 🏥 Agent Health")
        status = get_agent_status()
        for provider, s in status.items():
            avail = s["available"]
            cd    = s["cooldown_until"]
            color = "#e8f5e9" if avail else "#fff3e0"
            dot   = "🟢" if avail else "🟡"
            label = s["label"]
            detail = f"Calls: {s['calls']} | Errors: {s['errors']}"
            cooldown_txt = f"<br><small>⏳ recovers at {cd}</small>" if cd else ""
            st.markdown(
                f"<div style='background:{color};border-radius:10px;padding:10px 12px;"
                f"margin-bottom:6px;font-size:.85rem'>"
                f"{dot} <b>{label}</b><br>"
                f"<span style='color:#666'>{detail}{cooldown_txt}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.divider()
        st.markdown("### Tips")
        st.markdown("""
- Works best on public landing pages
- Avoid login-gated or JS-heavy pages
- Simple marketing sites clone best
""")
        st.divider()
        st.caption("Built by Anchit Chourasia")
        st.caption("[GitHub](https://github.com/anchitchourasia) · [LinkedIn](https://linkedin.com/in/anchit-chourasia-65b603226)")