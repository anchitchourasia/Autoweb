import streamlit as st
import os
import base64
from core.agent_chain import get_agent_status


def render_preview(clone_html, data, shot_path,
                   agent_used="", agent_logs=None, chars=0):

    if isinstance(clone_html, tuple):
        clone_html = clone_html

    if not clone_html or len(clone_html) < 100:
        st.error("⚠️ Clone HTML is empty. Check API keys and try again.")
        return

    st.markdown("---")
    st.caption(f"⚡ Pipeline: **{agent_used}** | Size: **{chars:,}** characters")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🖥️ Clone Preview",
        "📸 Original Screenshot",
        "📊 Extracted Data",
        "🤖 Agent Pipeline"
    ])

    # ── Tab 1: Clone Preview ──────────────────────────────────────────────
    with tab1:
        # FIXED
        col_dl, col_info = st.columns([2, 5])[3][4]
        with col_dl:
            st.download_button(
                label="⬇️ Download HTML",
                data=clone_html,
                file_name="clone.html",
                mime="text/html",
                use_container_width=False
            )
        with col_info:
            st.caption(f"📄 {chars:,} chars | {agent_used}")

        # base64 iframe — most reliable cross-platform render
        encoded = base64.b64encode(clone_html.encode("utf-8")).decode("utf-8")
        st.markdown(
            f'<iframe src="data:text/html;base64,{encoded}" '
            f'width="100%" height="950" '
            f'style="border:1px solid #e0e0e0;border-radius:12px;background:white;" '
            f'sandbox="allow-scripts allow-same-origin" loading="lazy">'
            f'</iframe>',
            unsafe_allow_html=True
        )

        with st.expander("🔍 View Raw HTML"):
            st.code(clone_html[:6000], language="html")
            if len(clone_html) > 6000:
                st.caption(f"... {len(clone_html)-6000:,} more chars")

    # ── Tab 2: Screenshot ─────────────────────────────────────────────────
    with tab2:
        if shot_path and os.path.exists(shot_path):
            st.image(shot_path, use_container_width=True)
        else:
            st.info("📸 Screenshot unavailable.")

    # ── Tab 3: Extracted Data ─────────────────────────────────────────────
    with tab3:
        st.markdown("### 📋 Page Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🏷️ Title**")
            st.code(data.get("title", "N/A"))
            st.markdown("**📝 Description**")
            st.code(data.get("meta_desc") or "Not found")

            st.markdown("**🎨 Colors**")
            colors = data.get("colors", [])
            if colors:
                cols = st.columns(min(len(colors), 6))
                for i, c in enumerate(colors[:6]):
                    with cols[i]:
                        st.markdown(
                            f"<div style='background:{c};height:36px;border-radius:6px;"
                            f"border:1px solid #ddd'></div>"
                            f"<p style='font-size:.7rem;text-align:center;margin-top:2px'>{c}</p>",
                            unsafe_allow_html=True
                        )

            st.markdown("**📏 Stats**")
            st.markdown(f"""
- HTML fetched: `{data.get('html_length',0):,}` bytes
- Clone size: `{chars:,}` chars
- Headings: `{len(data.get('headings',[]))}`
- Nav links: `{len(data.get('nav_links',[]))}`
- Buttons: `{len(data.get('buttons',[]))}`
""")

        with col2:
            st.markdown("**📌 Headings**")
            for h in data.get("headings", [])[:12]:
                tag = h["tag"].upper()
                c   = {"H1":"#667eea","H2":"#764ba2","H3":"#9e9e9e"}.get(tag,"#333")
                st.markdown(
                    f"<div style='margin-bottom:5px'>"
                    f"<span style='background:{c};color:white;padding:2px 7px;"
                    f"border-radius:4px;font-size:.7rem;font-weight:700'>{tag}</span>"
                    f" {h['text']}</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("**🔗 Nav Links**")
        link_data = [{"Text": l["text"], "Href": l["href"]}
                     for l in data.get("nav_links", [])[:15]]
        if link_data:
            st.dataframe(link_data, use_container_width=True, height=250)

    # ── Tab 4: A2A Pipeline Logs ──────────────────────────────────────────
    with tab4:
        st.markdown("### 🤖 A2A Collaborative Pipeline")
        st.markdown(
            "**All 3 agents work together** — each builds on the previous agent's output."
        )

        # Pipeline flow diagram
        st.markdown("""
<div style='background:#f8f9ff;border-radius:14px;padding:20px;
            text-align:center;margin-bottom:20px;font-size:.95rem'>
🧠 <b>Gemini</b> (HTML Architect)
&nbsp;→&nbsp;
⚡ <b>Groq</b> (CSS Stylist)
&nbsp;→&nbsp;
🦙 <b>Ollama</b> (Quality Finalizer)
</div>
""", unsafe_allow_html=True)

        if not agent_logs:
            st.info("No logs yet.")
            return

        stage_colors = {
            "✅ Success":          ("#e8f5e9", "#4caf50"),
            "❌ Failed/Skipped":   ("#ffebee", "#f44336"),
        }

        for stage_log in agent_logs:
            result  = stage_log.get("result", "")
            bg, bdr = stage_colors.get(result, ("#f5f5f5", "#9e9e9e"))
            chars_n = stage_log.get("chars", 0)

            st.markdown(
                f"<div style='background:{bg};border-left:5px solid {bdr};"
                f"border-radius:12px;padding:16px;margin-bottom:12px'>"
                f"<b>{stage_log.get('icon','')} {stage_log.get('stage','')}</b> "
                f"— {stage_log.get('agent','')} &nbsp; {result}<br>"
                f"<span style='font-size:.83rem;color:#555'>"
                f"{stage_log.get('role','')}</span>"
                + (f"<br><span style='font-size:.8rem;color:#888'>"
                   f"Output: {chars_n:,} chars</span>" if chars_n else "") +
                f"</div>",
                unsafe_allow_html=True
            )

            # Show sub-model attempts
            for attempt in stage_log.get("details", []):
                s  = attempt.get("status","")
                ic = {"success":"✅","failed":"❌","rate_limited":"⚠️",
                      "skipped":"⏭️"}.get(s,"❓")
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp; {ic} `{attempt.get('agent','')}` — "
                    f"**{s}** "
                    f"({attempt.get('duration',0)}s"
                    + (f", {attempt.get('chars',0):,} chars" if attempt.get('chars') else "") +
                    f") "
                    + (f"<span style='color:#888;font-size:.8rem'>{attempt.get('reason','')}</span>" if attempt.get('reason') else ""),
                    unsafe_allow_html=True
                )

        # Live health
        st.markdown("---")
        st.markdown("### 🏥 Live Agent Health")
        registry = get_agent_status()
        cols     = st.columns(3)
        for i, (provider, s) in enumerate(registry.items()):
            with cols[i]:
                avail = s["available"]
                cd    = s["cooldown_until"]
                st.markdown(
                    f"<div style='background:{'#e8f5e9' if avail else '#fff3e0'};"
                    f"border-radius:12px;padding:14px;text-align:center'>"
                    f"<b>{'🟢' if avail else '🟡'} {s['label']}</b><br>"
                    f"<small>Calls: {s['calls']} | Errors: {s['errors']}</small><br>"
                    f"<small>{'✅ Available' if avail else f'⏳ Recovers {cd}'}</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )