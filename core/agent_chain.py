import os
import re
import json
import time
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMIT TRACKER
# ─────────────────────────────────────────────────────────────────────────────

_lock = threading.Lock()

agent_registry = {
    "gemini": {
        "label":          "Google Gemini",
        "calls":          0,
        "errors":         0,
        "available":      True,
        "cooldown_until": None,
        "last_error":     "",
    },
    "groq": {
        "label":          "Groq Cloud",
        "calls":          0,
        "errors":         0,
        "available":      True,
        "cooldown_until": None,
        "last_error":     "",
    },
    "ollama": {
        "label":          "Ollama Cloud",
        "calls":          0,
        "errors":         0,
        "available":      True,
        "cooldown_until": None,
        "last_error":     "",
    },
}


def _is_available(provider: str) -> bool:
    with _lock:
        s = agent_registry[provider]
        if s["available"]:
            return True
        if s["cooldown_until"] and datetime.now() >= s["cooldown_until"]:
            s["available"] = True
            s["cooldown_until"] = None
            print(f"[A2A] 🔄 {provider} recovered from cooldown.")
            return True
        return False


def _mark_rate_limited(provider: str, retry_after: int = 60):
    with _lock:
        s = agent_registry[provider]
        s["available"]      = False
        s["cooldown_until"] = datetime.now() + timedelta(seconds=retry_after)
        s["errors"]        += 1
        print(f"[A2A] ⏳ {provider} cooldown {retry_after}s → "
              f"recovers at {s['cooldown_until'].strftime('%H:%M:%S')}")


def _mark_success(provider: str):
    with _lock:
        agent_registry[provider]["calls"]    += 1
        agent_registry[provider]["available"] = True


def _mark_error(provider: str, error: str):
    with _lock:
        agent_registry[provider]["errors"]     += 1
        agent_registry[provider]["last_error"]  = error[:300]


def get_agent_status() -> dict:
    return {
        k: {
            **v,
            "cooldown_until": v["cooldown_until"].strftime("%H:%M:%S")
                              if v["cooldown_until"] else None
        }
        for k, v in agent_registry.items()
    }


def _parse_retry_after(err: str) -> int:
    m = re.search(r"retry.{0,10}?(\d+)", err, re.IGNORECASE)
    return max(int(m.group(1)), 5) if m else 60


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES — each agent has a DIFFERENT specialized role
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_agent1_structure(data: dict) -> str:
    """
    AGENT 1 — GEMINI
    Role: Architect. Builds the full HTML skeleton with layout, sections,
    semantic structure. Focus on completeness and correct structure.
    """
    return f"""You are Agent 1 of 3 in an AI pipeline. Your role: HTML ARCHITECT.
Your job is to generate the complete HTML structure and layout.
Agents 2 and 3 will enhance your output — so focus on COMPLETE STRUCTURE.

=== WEBSITE TO CLONE ===
URL:         {data['url']}
TITLE:       {data['title']}
DESCRIPTION: {data['meta_desc']}
HEADINGS:    {json.dumps(data['headings'], ensure_ascii=False)}
NAV LINKS:   {json.dumps(data['nav_links'], ensure_ascii=False)}
CTA BUTTONS: {json.dumps(data['buttons'], ensure_ascii=False)}
COLORS:      {json.dumps(data['colors'])}
CONTENT:     {data['text'][:4000]}

=== YOUR TASK ===
Generate a complete single-file HTML clone with embedded CSS <style> tag.
Include ALL these sections:
1. Sticky navbar with logo + all nav links + CTA button
2. Hero section — big headline, subheading, 2 CTA buttons, gradient background
3. Features/cards section — use headings as card titles (min 3 cards)
4. Stats section — 3-4 impressive numbers
5. Testimonial or About section
6. CTA banner section
7. Footer — links, copyright, social icons

RULES:
- Return ONLY raw HTML. Start with <!doctype html>. Zero markdown. Zero explanation.
- Use detected colors: {json.dumps(data['colors'])}
- Google Fonts @import (Inter or Poppins)
- Placeholder images: https://placehold.co/600x400?text=Preview
- CSS variables: --primary, --secondary, --bg, --text
- Mobile responsive with @media (max-width: 768px)
"""


def _prompt_agent2_style(data: dict, html_from_agent1: str) -> str:
    """
    AGENT 2 — GROQ
    Role: CSS Stylist. Takes Agent 1's HTML and makes it visually stunning.
    Rewrites/enhances all CSS while keeping the structure.
    """
    return f"""You are Agent 2 of 3 in an AI pipeline. Your role: CSS STYLIST.
Agent 1 (Gemini) generated the HTML structure. Your job: make it VISUALLY STUNNING.

=== ORIGINAL SITE INFO ===
URL:    {data['url']}
TITLE:  {data['title']}
COLORS: {json.dumps(data['colors'])}

=== AGENT 1's OUTPUT (enhance this) ===
{html_from_agent1[:6000]}

=== YOUR TASK ===
Rewrite the entire <style> section to be world-class CSS. Keep all HTML structure intact.
Add/enhance:
1. CSS custom properties (--primary, --secondary, --accent, --bg, --text, --shadow)
2. Smooth animations: @keyframes fadeInUp, slideIn for hero elements
3. Hover effects: card lift (translateY(-8px)), button glow (box-shadow)
4. Card grid with glassmorphism or neumorphism effect
5. Gradient text for headings using -webkit-background-clip
6. Sticky navbar with blur backdrop-filter
7. Smooth scroll behavior
8. Professional typography scale

RULES:
- Return the COMPLETE HTML file (structure + your enhanced CSS)
- Start with <!doctype html>. No markdown. No explanation.
- Keep ALL existing HTML content and sections — only enhance CSS
- Use the detected color palette: {json.dumps(data['colors'])}
"""


def _prompt_agent3_refine(data: dict, html_from_agent2: str) -> str:
    """
    AGENT 3 — OLLAMA
    Role: Finalizer. Reviews both agents' work, fixes issues,
    adds final polish and makes it production-ready.
    """
    return f"""You are Agent 3 of 3 in an AI pipeline. Your role: QUALITY FINALIZER.
Agents 1 (structure) and 2 (styling) have already worked on this HTML.
Your job: final polish, fix any issues, make it PRODUCTION READY.

=== ORIGINAL SITE ===
URL:   {data['url']}
TITLE: {data['title']}

=== PREVIOUS AGENTS' OUTPUT (finalize this) ===
{html_from_agent2[:7000]}

=== YOUR TASK ===
Review and improve the HTML. Fix these common issues:
1. Any broken layout or missing closing tags
2. Add smooth scroll JavaScript at bottom
3. Add mobile hamburger menu toggle (JS)
4. Improve accessibility (aria-labels, alt texts)
5. Add loading animation for page (fade in body)
6. Make CTA buttons more prominent with gradient
7. Add subtle background pattern or texture to hero
8. Ensure all sections have proper padding/margin

RULES:
- Return the COMPLETE final HTML. Start with <!doctype html>.
- No markdown. No explanation. No code fences.
- Keep all existing content — only improve and fix
- This is the FINAL output so make it perfect
"""


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL AGENT CALLERS
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(model: str, prompt: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=8192
        )
    )
    return response.text.strip()


def _call_groq(model: str, prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a frontend developer in a multi-agent pipeline. "
                           "Return ONLY complete raw HTML with embedded CSS. "
                           "No markdown, no explanation, no code fences."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=8192
    )
    return response.choices[0].message.content.strip()


def _call_ollama(model: str, prompt: str) -> str:
    from ollama import Client
    api_key = os.getenv("OLLAMA_API_KEY")
    client  = Client(
        host    = "https://ollama.com" if api_key else "http://localhost:11434",
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    )
    response = client.chat(
        model    = model,
        messages = [
            {
                "role":    "system",
                "content": "You are a frontend developer finalizing HTML. "
                           "Return ONLY complete raw HTML. No markdown."
            },
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response["message"]["content"].strip()


def _clean_html(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$",        "", text)
    if "<!doctype" not in text.lower() and "<html" not in text.lower():
        return ""
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK CHAINS — if primary model fails, try next in same provider
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_MODELS  = ["gemini-2.5-flash", "gemini-1.5-flash"]
GROQ_MODELS    = ["llama-3.3-70b-versatile", "llama3-70b-8192"]
OLLAMA_MODELS  = ["gpt-oss:120b-cloud", "qwen3.5", "deepseek-v4-flash"]


def _try_provider(provider: str, models: list, prompt: str, log: list) -> str:
    """Try each model in a provider until one succeeds."""
    for model in models:
        if not _is_available(provider):
            log.append({
                "agent":    f"{provider} / {model}",
                "status":   "skipped",
                "reason":   "Provider on cooldown",
                "duration": 0, "chars": 0
            })
            return ""

        t = time.time()
        try:
            print(f"[A2A]    → trying {provider}/{model}...")
            if provider == "gemini":
                raw = _call_gemini(model, prompt)
            elif provider == "groq":
                raw = _call_groq(model, prompt)
            elif provider == "ollama":
                raw = _call_ollama(model, prompt)

            html     = _clean_html(raw)
            duration = round(time.time() - t, 1)

            if len(html) < 300:
                raise ValueError(f"Too short: {len(html)} chars")

            _mark_success(provider)
            log.append({
                "agent":    f"{provider} / {model}",
                "status":   "success",
                "reason":   "",
                "duration": duration,
                "chars":    len(html)
            })
            print(f"[A2A]    ✅ {provider}/{model} → {len(html):,} chars in {duration}s")
            return html

        except Exception as e:
            err      = str(e)
            duration = round(time.time() - t, 1)
            _mark_error(provider, err)
            is_rl = any(x in err.lower() for x in ["429","quota","rate limit","too many","rate_limit"])
            if is_rl:
                ra = _parse_retry_after(err)
                _mark_rate_limited(provider, ra)
                log.append({
                    "agent":    f"{provider} / {model}",
                    "status":   "rate_limited",
                    "reason":   f"Rate limited — cooldown {ra}s",
                    "duration": duration, "chars": 0
                })
                return ""   # whole provider done
            else:
                log.append({
                    "agent":    f"{provider} / {model}",
                    "status":   "failed",
                    "reason":   err[:120],
                    "duration": duration, "chars": 0
                })
                continue    # try next model in same provider

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN A2A PIPELINE — ALL 3 AGENTS COLLABORATE
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_chain(data: dict) -> dict:
    """
    TRUE A2A PIPELINE — all 3 agents work together:

    Stage 1 — GEMINI  : HTML Architect  → generates full structure
    Stage 2 — GROQ    : CSS Stylist     → enhances styling on Stage 1 output
    Stage 3 — OLLAMA  : Finalizer       → polishes & fixes Stage 2 output

    If any stage fails / rate-limits → that stage is skipped,
    next stage uses best available output so far.
    Final result is the most refined HTML possible.
    """

    pipeline_logs = []
    stage_outputs = {}   # stage_name → html

    # ── STAGE 1: GEMINI — Structure & Layout ─────────────────────────────
    print("\n[A2A] ═══════════════════════════════════════")
    print("[A2A] 🧠 STAGE 1: Gemini — HTML Architect")
    print("[A2A] ═══════════════════════════════════════")

    stage1_log  = []
    stage1_prompt = _prompt_agent1_structure(data)
    stage1_html   = _try_provider("gemini", GEMINI_MODELS, stage1_prompt, stage1_log)

    pipeline_logs.append({
        "stage":   "Stage 1 — HTML Architect",
        "agent":   "Gemini",
        "icon":    "🧠",
        "role":    "Generates full HTML structure, layout, all sections",
        "result":  "✅ Success" if stage1_html else "❌ Failed/Skipped",
        "chars":   len(stage1_html),
        "details": stage1_log
    })

    if stage1_html:
        stage_outputs["stage1"] = stage1_html
        print(f"[A2A] 🧠 Stage 1 complete — {len(stage1_html):,} chars")
    else:
        print("[A2A] ⚠️ Stage 1 failed — Stage 2 will generate from scratch")

    # ── STAGE 2: GROQ — CSS Stylist ──────────────────────────────────────
    print("\n[A2A] ═══════════════════════════════════════")
    print("[A2A] ⚡ STAGE 2: Groq — CSS Stylist")
    print("[A2A] ═══════════════════════════════════════")

    stage2_log    = []
    best_so_far   = stage_outputs.get("stage1", "")
    stage2_prompt = _prompt_agent2_style(data, best_so_far)
    stage2_html   = _try_provider("groq", GROQ_MODELS, stage2_prompt, stage2_log)

    pipeline_logs.append({
        "stage":   "Stage 2 — CSS Stylist",
        "agent":   "Groq",
        "icon":    "⚡",
        "role":    "Enhances CSS, adds animations, makes it visually stunning",
        "result":  "✅ Success" if stage2_html else "❌ Failed/Skipped",
        "chars":   len(stage2_html),
        "details": stage2_log
    })

    if stage2_html:
        stage_outputs["stage2"] = stage2_html
        print(f"[A2A] ⚡ Stage 2 complete — {len(stage2_html):,} chars")
    else:
        print("[A2A] ⚠️ Stage 2 failed — using Stage 1 output for Stage 3")

    # ── STAGE 3: OLLAMA — Finalizer ──────────────────────────────────────
    print("\n[A2A] ═══════════════════════════════════════")
    print("[A2A] 🦙 STAGE 3: Ollama — Quality Finalizer")
    print("[A2A] ═══════════════════════════════════════")

    stage3_log    = []
    best_so_far   = stage_outputs.get("stage2") or stage_outputs.get("stage1", "")
    stage3_prompt = _prompt_agent3_refine(data, best_so_far)
    stage3_html   = _try_provider("ollama", OLLAMA_MODELS, stage3_prompt, stage3_log)

    pipeline_logs.append({
        "stage":   "Stage 3 — Quality Finalizer",
        "agent":   "Ollama Cloud",
        "icon":    "🦙",
        "role":    "Polishes, fixes issues, adds JS interactions, production ready",
        "result":  "✅ Success" if stage3_html else "❌ Failed/Skipped",
        "chars":   len(stage3_html),
        "details": stage3_log
    })

    if stage3_html:
        stage_outputs["stage3"] = stage3_html
        print(f"[A2A] 🦙 Stage 3 complete — {len(stage3_html):,} chars")

    # ── Pick best available output ────────────────────────────────────────
    final_html = (
        stage_outputs.get("stage3") or
        stage_outputs.get("stage2") or
        stage_outputs.get("stage1") or
        _fallback_html(data)
    )

    completed = [s for s in ["stage3","stage2","stage1"] if s in stage_outputs]
    used_agents = {
        "stage3": "All 3 Agents (Gemini→Groq→Ollama) ✨",
        "stage2": "Gemini + Groq (Ollama skipped)",
        "stage1": "Gemini only (Groq + Ollama skipped)",
    }
    used_label = used_agents.get(completed if completed else "", "Static Fallback")

    print(f"\n[A2A] 🏁 Pipeline complete — {used_label}")
    print(f"[A2A] 📄 Final output: {len(final_html):,} chars\n")

    return {
        "html":       final_html,
        "agent_used": used_label,
        "logs":       pipeline_logs,
        "chars":      len(final_html),
        "stages":     stage_outputs
    }


def _fallback_html(data: dict) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{data['title']} - Clone</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#f4f6f8;padding:40px}}
.card{{max-width:860px;margin:auto;background:#fff;border-radius:20px;
       padding:48px;box-shadow:0 10px 40px rgba(0,0,0,.08)}}
h1{{font-size:2.2rem;color:#333;margin-bottom:12px}}
p{{color:#666;line-height:1.8;margin-bottom:20px}}
.warn{{background:#fff8e1;border-left:4px solid #ffc107;border-radius:8px;padding:16px;color:#856404}}
</style>
</head>
<body>
<div class="card">
  <h1>{data['title']}</h1>
  <p>{data['meta_desc'] or 'No description available.'}</p>
  <div class="warn">⚠️ All AI agents exhausted. Please wait and try again.</div>
</div>
</body>
</html>"""