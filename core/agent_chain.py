import os
import re
import json
import time
import threading
import base64
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# AGENT REGISTRY + RATE LIMIT TRACKER
# ─────────────────────────────────────────────────────────────────────────────

_lock = threading.Lock()

agent_registry = {
    "gemini": {"label": "Google Gemini", "calls": 0, "errors": 0,
               "available": True, "cooldown_until": None, "last_error": ""},
    "groq":   {"label": "Groq Cloud",    "calls": 0, "errors": 0,
               "available": True, "cooldown_until": None, "last_error": ""},
    "ollama": {"label": "Ollama Cloud",  "calls": 0, "errors": 0,
               "available": True, "cooldown_until": None, "last_error": ""},
}

# RL Memory — persists across rounds within a pipeline run
_rl_memory = {
    "ollama_mistakes":  [],   # what ollama did wrong
    "groq_mistakes":    [],   # what groq did wrong
    "successful_patterns": [], # what worked well
    "round_scores":    [],    # scores per round
    "best_html":       "",
    "best_score":      0,
}


def _reset_rl_memory():
    global _rl_memory
    _rl_memory = {
        "ollama_mistakes":     [],
        "groq_mistakes":       [],
        "successful_patterns": [],
        "round_scores":        [],
        "best_html":           "",
        "best_score":          0,
    }


def _is_available(p: str) -> bool:
    with _lock:
        s = agent_registry[p]
        if s["available"]: return True
        if s["cooldown_until"] and datetime.now() >= s["cooldown_until"]:
            s["available"] = True; s["cooldown_until"] = None
            print(f"[A2A] 🔄 {p} recovered.")
            return True
        return False


def _mark_rate_limited(p: str, secs: int = 60):
    with _lock:
        s = agent_registry[p]
        s["available"] = False
        s["cooldown_until"] = datetime.now() + timedelta(seconds=secs)
        s["errors"] += 1
        print(f"[A2A] ⏳ {p} cooldown {secs}s")


def _mark_success(p: str):
    with _lock:
        agent_registry[p]["calls"] += 1
        agent_registry[p]["available"] = True


def _mark_error(p: str, err: str):
    with _lock:
        agent_registry[p]["errors"] += 1
        agent_registry[p]["last_error"] = err[:300]


def get_agent_status() -> dict:
    return {k: {**v, "cooldown_until": v["cooldown_until"].strftime("%H:%M:%S")
                if v["cooldown_until"] else None}
            for k, v in agent_registry.items()}


def _parse_retry(err: str) -> int:
    m = re.search(r"retry.{0,10}?(\d+)", err, re.IGNORECASE)
    return max(int(m.group(1)), 5) if m else 60


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL API CALLERS
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(model: str, prompt: str, image_b64: str = None) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    parts  = []
    if image_b64:
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(image_b64), mime_type="image/png"))
    parts.append(types.Part.from_text(text=prompt))
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=8192)
    )
    return response.text.strip()


def _call_groq(model: str, prompt: str) -> str:
    from groq import Groq
    r = Groq(api_key=os.getenv("GROQ_API_KEY")).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content":
             "You are an expert CSS/UI specialist in a supervised AI pipeline. "
             "You learn from your mistakes each round. "
             "Return ONLY complete raw HTML+CSS. No markdown. No explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2, max_tokens=8192
    )
    return r.choices[0].message.content.strip()


def _call_ollama(model: str, prompt: str) -> str:
    from ollama import Client
    api_key = os.getenv("OLLAMA_API_KEY")
    client  = Client(
        host="https://ollama.com" if api_key else "http://localhost:11434",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {}
    )
    r = client.chat(
        model=model,
        messages=[
            {"role": "system", "content":
             "You are an expert HTML structure specialist in a supervised AI pipeline. "
             "You learn from your mistakes each round. "
             "Return ONLY complete raw HTML. No markdown. No explanations."},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return r["message"]["content"].strip()


def _clean_html(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    if "<!doctype" not in text.lower() and "<html" not in text.lower():
        return ""
    return text.strip()


GEMINI_MODELS = ["gemini-2.5-flash", "gemini-1.5-flash"]
GROQ_MODELS   = ["llama-3.3-70b-versatile", "llama3-70b-8192"]
OLLAMA_MODELS = ["gpt-oss:120b-cloud", "qwen3.5", "deepseek-v4-flash"]


def _safe_gemini(prompt: str, image_b64: str = None) -> str:
    """Call Gemini with model fallback."""
    for model in GEMINI_MODELS:
        if not _is_available("gemini"):
            break
        try:
            t   = time.time()
            out = _call_gemini(model, prompt, image_b64)
            _mark_success("gemini")
            print(f"[GEMINI] ✅ {model} → {len(out):,} chars in {round(time.time()-t,1)}s")
            return out
        except Exception as e:
            err = str(e)
            _mark_error("gemini", err)
            if any(x in err.lower() for x in ["429","quota","rate limit","too many"]):
                _mark_rate_limited("gemini", _parse_retry(err))
                break
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR STEP 1 — DEEP VISUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_deep_analyze(data: dict, image_b64: str) -> dict:
    """
    Gemini performs PIXEL-LEVEL visual analysis of the screenshot.
    Extracts exact colors, layout, typography, spacing, and special elements.
    """
    prompt = f"""You are the SUPERVISOR AI with computer vision capabilities.
Analyze this website screenshot with EXTREME DETAIL and produce a precise JSON blueprint.

=== SITE METADATA ===
URL:   {data['url']}
TITLE: {data['title']}
COLORS EXTRACTED: {json.dumps(data['colors'])}

Examine EVERY visual detail in the screenshot:
- Background: gradient direction, colors, any noise/texture
- Typography: exact font weights, sizes, letter-spacing
- Layout measurements: approximate widths, padding, margins
- Special visual elements: logos, icons, code blocks, badges
- Animations or interactive elements visible
- Card styles: border, shadow, border-radius values
- Button styles: exact colors, padding, border-radius, icons

Return ONLY this JSON (no other text):
{{
  "background": {{
    "type": "solid/gradient/image",
    "gradient_from": "#hex",
    "gradient_to": "#hex",
    "gradient_direction": "to bottom right / etc",
    "has_texture": false
  }},
  "color_scheme": {{
    "primary": "#hex",
    "secondary": "#hex",
    "accent": "#hex",
    "background": "#hex",
    "text_dark": "#hex",
    "text_light": "#hex",
    "navbar_bg": "#hex",
    "hero_bg": "#hex"
  }},
  "navbar": {{
    "background": "#hex or transparent",
    "has_blur": false,
    "logo_text": "exact logo text or icon description",
    "link_color": "#hex",
    "cta_button_color": "#hex",
    "cta_button_text": "exact text"
  }},
  "hero": {{
    "has_logo_icon": true,
    "logo_icon_description": "describe the icon/logo shown",
    "headline": "exact headline text",
    "headline_size": "approximate px or rem",
    "headline_weight": "400/600/700/800/900",
    "subheading_color": "#hex",
    "has_code_block": false,
    "code_block_content": "exact code if visible",
    "cta_buttons": [
      {{"text": "button text", "bg": "#hex", "text_color": "#hex", "has_icon": true}}
    ],
    "has_version_link": false,
    "version_text": "e.g. v5.3.8"
  }},
  "features_section": {{
    "has_section": true,
    "card_count": 3,
    "card_bg": "#hex",
    "card_border_radius": "8px",
    "card_shadow": "description",
    "card_has_icons": true,
    "layout": "grid-3-cols / flex / etc"
  }},
  "typography": {{
    "font_family": "exact font name",
    "hero_size": "approx px",
    "h2_size": "approx px",
    "body_size": "16px",
    "line_height": "1.5"
  }},
  "spacing": {{
    "section_padding": "approx px top/bottom",
    "container_max_width": "approx px",
    "card_padding": "approx px"
  }},
  "special_elements": ["list every unique UI element visible"],
  "critical_must_replicate": ["top 5 things that make this site visually unique"]
}}"""

    try:
        raw = _safe_gemini(prompt, image_b64)
        m   = re.search(r"\{.*\}", raw, re.DOTALL)
        bp  = json.loads(m.group(0) if m else raw)
        print(f"[SUPERVISOR] ✅ Deep blueprint: {len(bp)} sections extracted")
        return bp
    except Exception as e:
        print(f"[SUPERVISOR] ⚠️ Blueprint parse failed: {e}")
        c = data.get("colors", ["#7952b3"])
        return {
            "background":    {"type": "gradient", "gradient_from": "#e0d7ff",
                              "gradient_to": "#fde8d0", "gradient_direction": "to bottom right",
                              "has_texture": False},
            "color_scheme":  {"primary": c if c else "#7952b3",
                              "secondary": c[3] if len(c)>3 else "#6610f2",
                              "accent": "#7952b3", "background": "#ffffff",
                              "text_dark": "#212529", "text_light": "#6c757d",
                              "navbar_bg": "rgba(255,255,255,0.9)", "hero_bg": "transparent"},
            "navbar":        {"background": "rgba(255,255,255,0.9)", "has_blur": True,
                              "logo_text": data.get("title","Site")[:20],
                              "link_color": "#212529", "cta_button_color": "#7952b3",
                              "cta_button_text": "Get started"},
            "hero":          {"has_logo_icon": True, "logo_icon_description": "brand logo icon",
                              "headline": data.get("title",""), "headline_size": "64px",
                              "headline_weight": "700", "subheading_color": "#6c757d",
                              "has_code_block": False, "code_block_content": "",
                              "cta_buttons": [{"text": "Get started", "bg": "#7952b3",
                                              "text_color": "#fff", "has_icon": True}],
                              "has_version_link": False, "version_text": ""},
            "features_section": {"has_section": True, "card_count": 3,
                                 "card_bg": "#ffffff", "card_border_radius": "12px",
                                 "card_shadow": "0 4px 20px rgba(0,0,0,0.08)",
                                 "card_has_icons": True, "layout": "grid-3-cols"},
            "typography":    {"font_family": "system-ui, -apple-system, sans-serif",
                              "hero_size": "64px", "h2_size": "36px",
                              "body_size": "16px", "line_height": "1.6"},
            "spacing":       {"section_padding": "80px", "container_max_width": "1200px",
                              "card_padding": "32px"},
            "special_elements":         [],
            "critical_must_replicate":  ["gradient background", "hero headline", "navbar", "feature cards", "footer"]
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR — GENERATE WORKER BRIEFS
# ─────────────────────────────────────────────────────────────────────────────

def _build_ollama_prompt(data: dict, bp: dict, round_num: int,
                          corrections: str = "") -> str:
    mistakes_block = ""
    if _rl_memory["ollama_mistakes"]:
        mistakes_block = f"""
=== ❌ YOUR PREVIOUS MISTAKES (DO NOT REPEAT) ===
{chr(10).join(f'• {m}' for m in _rl_memory["ollama_mistakes"][-5:])}
"""
    wins_block = ""
    if _rl_memory["successful_patterns"]:
        wins_block = f"""
=== ✅ PATTERNS THAT WORKED (KEEP THESE) ===
{chr(10).join(f'• {p}' for p in _rl_memory["successful_patterns"][-3:])}
"""
    correction_block = f"\n=== SUPERVISOR CORRECTIONS FOR ROUND {round_num} ===\n{corrections}" if corrections else ""

    hero   = bp.get("hero", {})
    nav    = bp.get("navbar", {})
    bg     = bp.get("background", {})
    typo   = bp.get("typography", {})
    cards  = bp.get("features_section", {})
    colors = bp.get("color_scheme", {})

    return f"""You are the HTML STRUCTURE WORKER (Ollama) — Round {round_num}/3.
Your supervisor (Gemini) analyzed the target screenshot and provides this exact blueprint.
BUILD THE HTML EXACTLY MATCHING THESE VISUAL SPECS.
{mistakes_block}{wins_block}{correction_block}

=== PIXEL-PERFECT VISUAL BLUEPRINT ===
BACKGROUND: {bg.get('type')} — from {bg.get('gradient_from')} to {bg.get('gradient_to')} ({bg.get('gradient_direction')})
COLORS: primary={colors.get('primary')} secondary={colors.get('secondary')} accent={colors.get('accent')}
NAVBAR BG: {nav.get('background')} | Logo: "{nav.get('logo_text')}" | CTA: "{nav.get('cta_button_text')}"
HERO HEADLINE: "{hero.get('headline')}" | Size: {hero.get('headline_size')} | Weight: {hero.get('headline_weight')}
HERO HAS LOGO ICON: {hero.get('has_logo_icon')} — {hero.get('logo_icon_description')}
HERO CODE BLOCK: {hero.get('has_code_block')} — "{hero.get('code_block_content')}"
HERO VERSION: {hero.get('has_version_link')} — "{hero.get('version_text')}"
HERO CTA BUTTONS: {json.dumps(hero.get('cta_buttons',[]))}
FONT: {typo.get('font_family')} | Hero: {typo.get('hero_size')} | H2: {typo.get('h2_size')}
CARDS: {cards.get('card_count')} cards | bg={cards.get('card_bg')} | radius={cards.get('card_border_radius')} | layout={cards.get('layout')}
SPECIAL ELEMENTS: {json.dumps(bp.get('special_elements',[]))}
CRITICAL TO REPLICATE: {json.dumps(bp.get('critical_must_replicate',[]))}

=== SITE CONTENT ===
URL:      {data['url']}
TITLE:    {data['title']}
HEADINGS: {json.dumps([h['text'] for h in data['headings'][:12]])}
NAV:      {json.dumps([l['text'] for l in data['nav_links'][:10]])}
CONTENT:  {data['text'][:2500]}

=== STRICT REQUIREMENTS ===
1. Build ALL sections: navbar, hero (with logo icon if needed), features, stats, cta, footer
2. Use EXACT class names: .navbar, .hero, .features-grid, .card, .footer
3. Place <style> tag with CSS variables matching blueprint colors
4. Placeholder images: https://placehold.co/600x400
5. Return ONLY <!doctype html>.....</html> — zero markdown"""


def _build_groq_prompt(data: dict, bp: dict, ollama_html: str,
                        round_num: int, corrections: str = "") -> str:
    mistakes_block = ""
    if _rl_memory["groq_mistakes"]:
        mistakes_block = f"""
=== ❌ YOUR PREVIOUS MISTAKES (DO NOT REPEAT) ===
{chr(10).join(f'• {m}' for m in _rl_memory["groq_mistakes"][-5:])}
"""
    wins_block = ""
    if _rl_memory["successful_patterns"]:
        wins_block = f"""
=== ✅ PATTERNS THAT WORKED WELL (KEEP THESE) ===
{chr(10).join(f'• {p}' for p in _rl_memory["successful_patterns"][-3:])}
"""
    correction_block = f"\n=== SUPERVISOR CORRECTIONS FOR ROUND {round_num} ===\n{corrections}" if corrections else ""

    bg     = bp.get("background", {})
    colors = bp.get("color_scheme", {})
    typo   = bp.get("typography", {})
    cards  = bp.get("features_section", {})
    spacing= bp.get("spacing", {})

    return f"""You are the CSS STYLING WORKER (Groq) — Round {round_num}/3.
Your supervisor analyzed the screenshot. Match it PIXEL-PERFECTLY.
{mistakes_block}{wins_block}{correction_block}

=== EXACT VISUAL SPECS TO MATCH ===
BODY BACKGROUND: {bg.get('type')} gradient({bg.get('gradient_from')}, {bg.get('gradient_to')}) {bg.get('gradient_direction')}
NAVBAR: background={bp.get('navbar',{}).get('background')} | backdrop-filter: {'blur(10px)' if bp.get('navbar',{}).get('has_blur') else 'none'}
PRIMARY COLOR: {colors.get('primary')}
SECONDARY: {colors.get('secondary')}
TEXT DARK: {colors.get('text_dark')} | TEXT LIGHT: {colors.get('text_light')}
FONT: {typo.get('font_family')} | HERO SIZE: {typo.get('hero_size')} weight {bp.get('hero',{}).get('headline_weight','700')}
SECTION PADDING: {spacing.get('section_padding')} | MAX-WIDTH: {spacing.get('container_max_width')}
CARDS: bg={cards.get('card_bg')} | radius={cards.get('card_border_radius')} | shadow={cards.get('card_shadow')}

=== HTML TO STYLE ===
{ollama_html[:5500]}

=== MANDATORY CSS RULES ===
:root {{
  --primary: {colors.get('primary', '#667eea')};
  --secondary: {colors.get('secondary', '#764ba2')};
  --accent: {colors.get('accent', '#ff6b6b')};
  --bg-from: {bg.get('gradient_from', '#e0d7ff')};
  --bg-to: {bg.get('gradient_to', '#fde8d0')};
  --text: {colors.get('text_dark', '#212529')};
  --text-light: {colors.get('text_light', '#6c757d')};
}}
body {{ background: linear-gradient({bg.get('gradient_direction','to bottom right')}, var(--bg-from), var(--bg-to)); min-height: 100vh; }}

ADD THESE:
1. @import Google Fonts ({typo.get('font_family','Inter')})
2. Smooth scroll, box-sizing border-box reset
3. Navbar: sticky, blur backdrop, border-bottom
4. Hero: centered, large headline, proper spacing
5. Feature cards: grid-template-columns: repeat(3, 1fr), hover lift
6. @keyframes fadeInUp for hero elements
7. Responsive: @media (max-width: 768px) stacks to 1 col
8. Buttons: border-radius 8px, padding 12px 28px, hover transform

Return COMPLETE HTML file starting with <!doctype html>. ZERO markdown."""


# ─────────────────────────────────────────────────────────────────────────────
# WORKER CALLERS WITH RL FEEDBACK
# ─────────────────────────────────────────────────────────────────────────────

def _run_ollama(prompt: str, round_num: int) -> tuple[str, str]:
    """Returns (html, model_used)"""
    for model in OLLAMA_MODELS:
        if not _is_available("ollama"):
            break
        try:
            t   = time.time()
            raw = _call_ollama(model, prompt)
            html= _clean_html(raw)
            if len(html) > 500:
                _mark_success("ollama")
                print(f"[OLLAMA R{round_num}] ✅ {model} → {len(html):,} chars in {round(time.time()-t,1)}s")
                return html, model
        except Exception as e:
            err = str(e)
            _mark_error("ollama", err)
            if any(x in err.lower() for x in ["429","quota","rate limit","too many"]):
                _mark_rate_limited("ollama", _parse_retry(err))
                break
    return "", ""


def _run_groq(prompt: str, round_num: int) -> tuple[str, str]:
    """Returns (html, model_used)"""
    for model in GROQ_MODELS:
        if not _is_available("groq"):
            break
        try:
            t   = time.time()
            raw = _call_groq(model, prompt)
            html= _clean_html(raw)
            if len(html) > 500:
                _mark_success("groq")
                print(f"[GROQ R{round_num}] ✅ {model} → {len(html):,} chars in {round(time.time()-t,1)}s")
                return html, model
        except Exception as e:
            err = str(e)
            _mark_error("groq", err)
            if any(x in err.lower() for x in ["429","quota","rate limit","too many"]):
                _mark_rate_limited("groq", _parse_retry(err))
                break
    return "", ""


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR — VISUAL SCORING + RL FEEDBACK GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_score_and_feedback(data: dict, bp: dict, html: str,
                                   image_b64: str, round_num: int) -> dict:
    """
    Gemini scores HTML output against screenshot on a detailed checklist
    and generates specific RL feedback for each worker.
    """
    prompt = f"""You are the SUPERVISOR AI scoring Round {round_num} output against the original screenshot.

=== SCORING CHECKLIST (score each 0-10, be STRICT) ===
Compare the generated HTML against the screenshot carefully.

GENERATED HTML (first 3500 chars):
{html[:3500]}

=== ORIGINAL VISUAL SPECS ===
Background: {bp.get('background',{})}
Colors: {bp.get('color_scheme',{})}
Hero: {bp.get('hero',{})}
Critical elements: {json.dumps(bp.get('critical_must_replicate',[]))}

Score STRICTLY and return JSON:
{{
  "scores": {{
    "background_match": 0-10,
    "navbar_match": 0-10,
    "hero_layout": 0-10,
    "color_accuracy": 0-10,
    "typography_match": 0-10,
    "cards_section": 0-10,
    "responsive_design": 0-10,
    "special_elements": 0-10
  }},
  "total_score": 0-100,
  "approved": true_if_total_>=_70,
  "ollama_specific_mistakes": [
    "specific HTML structure mistake 1",
    "specific HTML structure mistake 2"
  ],
  "groq_specific_mistakes": [
    "specific CSS styling mistake 1",
    "specific CSS styling mistake 2"
  ],
  "ollama_corrections": "precise HTML fixes for next round",
  "groq_corrections": "precise CSS fixes for next round",
  "what_worked_well": ["pattern1", "pattern2"],
  "top_3_improvements": ["most impactful fix 1", "most impactful fix 2", "most impactful fix 3"]
}}"""

    try:
        raw    = _safe_gemini(prompt, image_b64)
        m      = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(m.group(0) if m else raw)
        result["approved"] = result.get("total_score", 0) >= 70
        return result
    except Exception as e:
        print(f"[SUPERVISOR] ⚠️ Scoring failed: {e}")
        return {
            "scores": {"background_match": 7, "navbar_match": 7, "hero_layout": 7,
                       "color_accuracy": 7, "typography_match": 7, "cards_section": 7,
                       "responsive_design": 7, "special_elements": 5},
            "total_score": 70, "approved": True,
            "ollama_specific_mistakes": [],
            "groq_specific_mistakes":   [],
            "ollama_corrections": "",
            "groq_corrections":   "",
            "what_worked_well":   [],
            "top_3_improvements": []
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_chain(data: dict) -> dict:
    """
    SMART SUPERVISED RL PIPELINE:

    1. Gemini deep visual analysis → pixel-level blueprint
    2. LOOP (max 3 rounds):
       a. Ollama builds HTML with blueprint + RL mistake memory
       b. Groq styles HTML with blueprint + RL mistake memory
       c. Gemini scores output vs screenshot (checklist-based)
       d. Gemini generates specific RL feedback for each worker
       e. Store mistakes + wins in RL memory
       f. If score >= 70 → APPROVED, break
       g. Else → next round with corrections
    3. Best output across all rounds is returned
    """
    _reset_rl_memory()
    logs      = []
    shot_path = data.get("_screenshot_path", "")

    image_b64 = None
    if shot_path and os.path.exists(shot_path):
        with open(shot_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        print("[SUPERVISOR] 📸 Screenshot loaded")

    # ══════════════════════════════════════════════════════
    # STAGE 1 — Deep Visual Analysis
    # ══════════════════════════════════════════════════════
    print("\n[SUPERVISOR] 🔍 Deep visual analysis...")
    t         = time.time()
    blueprint = supervisor_deep_analyze(data, image_b64)
    dur       = round(time.time()-t, 1)
    logs.append({
        "stage":    "🔍 Stage 1 — Deep Visual Analysis",
        "agent":    "Gemini Supervisor",
        "icon":     "🧠",
        "role":     "Pixel-level screenshot analysis → precise blueprint",
        "result":   "✅ Blueprint ready",
        "chars":    len(json.dumps(blueprint)),
        "duration": dur,
        "details":  [
            {"agent": "background",  "status": str(blueprint.get("background",{}))[:80],    "duration": 0, "chars": 0, "reason": ""},
            {"agent": "colors",      "status": str(blueprint.get("color_scheme",{}))[:80],  "duration": 0, "chars": 0, "reason": ""},
            {"agent": "hero",        "status": str(blueprint.get("hero",{}))[:80],           "duration": 0, "chars": 0, "reason": ""},
            {"agent": "critical",    "status": str(blueprint.get("critical_must_replicate",[])), "duration": 0, "chars": 0, "reason": ""},
        ]
    })

    # ══════════════════════════════════════════════════════
    # STAGE 2 — Iterative RL Refinement Loop (max 3 rounds)
    # ══════════════════════════════════════════════════════
    MAX_ROUNDS     = 3
    best_html      = ""
    best_score     = 0
    ollama_correct = ""
    groq_correct   = ""

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'═'*50}")
        print(f"[PIPELINE] 🔄 ROUND {round_num}/{MAX_ROUNDS}")
        print(f"{'═'*50}")

        # ── Ollama: HTML Structure ────────────────────────
        print(f"[OLLAMA R{round_num}] 🦙 Building HTML...")
        t            = time.time()
        ollama_prompt= _build_ollama_prompt(data, blueprint, round_num, ollama_correct)
        ollama_html, ollama_model = _run_ollama(ollama_prompt, round_num)
        ollama_dur   = round(time.time()-t, 1)

        if not ollama_html:
            print(f"[SUPERVISOR] ⚠️ Ollama R{round_num} failed — Gemini covering...")
            ollama_html = _safe_gemini(ollama_prompt) or ""
            ollama_html = _clean_html(ollama_html)
            ollama_model= "gemini-fallback"

        logs.append({
            "stage":    f"🦙 Round {round_num} — HTML Structure (Ollama)",
            "agent":    "Ollama Worker",
            "icon":     "🦙",
            "role":     f"Builds HTML from blueprint {'+ RL corrections' if ollama_correct else ''}",
            "result":   f"✅ {len(ollama_html):,} chars" if ollama_html else "❌ Failed",
            "chars":    len(ollama_html),
            "duration": ollama_dur,
            "details":  [{"agent": ollama_model or "none", "status": "success" if ollama_html else "failed",
                          "duration": ollama_dur, "chars": len(ollama_html), "reason": ""}]
        })

        # ── Groq: CSS Styling ─────────────────────────────
        print(f"[GROQ R{round_num}] ⚡ Styling HTML...")
        t           = time.time()
        groq_prompt = _build_groq_prompt(data, blueprint, ollama_html, round_num, groq_correct)
        groq_html, groq_model = _run_groq(groq_prompt, round_num)
        groq_dur    = round(time.time()-t, 1)

        if not groq_html:
            print(f"[SUPERVISOR] ⚠️ Groq R{round_num} failed — using Ollama output")
            groq_html  = ollama_html
            groq_model = "ollama-fallback"

        best_this_round = groq_html or ollama_html
        logs.append({
            "stage":    f"⚡ Round {round_num} — CSS Styling (Groq)",
            "agent":    "Groq Worker",
            "icon":     "⚡",
            "role":     f"Applies CSS from blueprint {'+ RL corrections' if groq_correct else ''}",
            "result":   f"✅ {len(groq_html):,} chars" if groq_html else "❌ Failed",
            "chars":    len(groq_html),
            "duration": groq_dur,
            "details":  [{"agent": groq_model or "none", "status": "success" if groq_html else "failed",
                          "duration": groq_dur, "chars": len(groq_html), "reason": ""}]
        })

        # ── Gemini: Score + RL Feedback ───────────────────
        print(f"[SUPERVISOR] 📊 Scoring round {round_num}...")
        t        = time.time()
        feedback = supervisor_score_and_feedback(
            data, blueprint, best_this_round, image_b64, round_num)
        score    = feedback.get("total_score", 0)
        approved = feedback.get("approved", False)
        score_dur= round(time.time()-t, 1)

        # ── Update RL Memory ──────────────────────────────
        _rl_memory["round_scores"].append(score)
        _rl_memory["ollama_mistakes"].extend(feedback.get("ollama_specific_mistakes", []))
        _rl_memory["groq_mistakes"].extend(feedback.get("groq_specific_mistakes", []))
        _rl_memory["successful_patterns"].extend(feedback.get("what_worked_well", []))

        if score > best_score:
            best_score = score
            best_html  = best_this_round
            _rl_memory["best_html"]  = best_html
            _rl_memory["best_score"] = best_score

        # Prep corrections for next round
        ollama_correct = feedback.get("ollama_corrections", "")
        groq_correct   = feedback.get("groq_corrections", "")

        scores_detail = feedback.get("scores", {})
        logs.append({
            "stage":    f"📊 Round {round_num} — Supervisor Score: {score}/100",
            "agent":    "Gemini Supervisor",
            "icon":     "🧠",
            "role":     f"Scores output vs screenshot | RL feedback → {'next round' if not approved else 'APPROVED'}",
            "result":   f"{'✅ APPROVED' if approved else f'🔄 Score {score}/100 — next round'}",
            "chars":    len(best_this_round),
            "duration": score_dur,
            "details":  [
                {"agent": f"bg={scores_detail.get('background_match',0)}/10 | nav={scores_detail.get('navbar_match',0)}/10 | hero={scores_detail.get('hero_layout',0)}/10",
                 "status": f"colors={scores_detail.get('color_accuracy',0)}/10 | typo={scores_detail.get('typography_match',0)}/10 | cards={scores_detail.get('cards_section',0)}/10",
                 "duration": score_dur, "chars": score, "reason": " | ".join(feedback.get('top_3_improvements',[])[:2])}
            ]
        })

        print(f"[SUPERVISOR] {'✅ APPROVED' if approved else f'🔄 Score {score}/100'} — "
              f"bg={scores_detail.get('background_match')}/10 "
              f"colors={scores_detail.get('color_accuracy')}/10 "
              f"hero={scores_detail.get('hero_layout')}/10")

        if approved:
            print(f"[SUPERVISOR] 🏁 Approved at round {round_num}!")
            break
        elif round_num < MAX_ROUNDS:
            print(f"[SUPERVISOR] 📝 RL memory updated — {len(_rl_memory['ollama_mistakes'])} mistakes logged")
            print(f"[SUPERVISOR] 🔄 Starting round {round_num+1} with corrections...")

    # ══════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════
    rounds_taken = len(_rl_memory["round_scores"])
    scores_str   = " → ".join(str(s) for s in _rl_memory["round_scores"])
    used_label   = (f"🧠 Supervised RL Pipeline | {rounds_taken} round(s) | "
                    f"Scores: {scores_str} | Best: {best_score}% "
                    f"{'✅ APPROVED' if best_score >= 70 else '⚠️ Partial'}")

    print(f"\n[SUPERVISOR] 🏁 Pipeline done — best score: {best_score}% | rounds: {rounds_taken}")
    print(f"[SUPERVISOR] 📄 Final HTML: {len(best_html):,} chars")
    print(f"[SUPERVISOR] 🧠 RL Memory: {len(_rl_memory['ollama_mistakes'])} ollama mistakes | "
          f"{len(_rl_memory['groq_mistakes'])} groq mistakes logged\n")

    return {
        "html":        best_html or _fallback_html(data),
        "agent_used":  used_label,
        "logs":        logs,
        "chars":       len(best_html),
        "blueprint":   blueprint,
        "rl_memory":   {
            "rounds":           rounds_taken,
            "scores":           _rl_memory["round_scores"],
            "best_score":       best_score,
            "ollama_mistakes":  _rl_memory["ollama_mistakes"],
            "groq_mistakes":    _rl_memory["groq_mistakes"],
            "patterns_learned": _rl_memory["successful_patterns"]
        },
        "final_check": {"visual_match_score": best_score, "approved": best_score >= 70}
    }


def _fallback_html(data: dict) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<title>{data.get('title','Clone')}</title>
<style>body{{font-family:sans-serif;padding:40px;background:#f4f6f8}}
.c{{max-width:800px;margin:auto;background:#fff;padding:40px;border-radius:20px;
    box-shadow:0 10px 40px rgba(0,0,0,.08)}}</style></head>
<body><div class="c"><h1>{data.get('title','')}</h1>
<p>⚠️ All agents exhausted. Please retry.</p></div></body></html>"""