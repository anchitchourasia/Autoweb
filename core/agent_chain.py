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
# PIPELINE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

APPROVAL_THRESHOLD = 90   # all 3 agents keep working until score >= 90%
MAX_ROUNDS         = 5    # max refinement rounds before best-effort return

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

# ─────────────────────────────────────────────────────────────────────────────
# RL MEMORY — persists across all rounds; agents READ it each round
# ─────────────────────────────────────────────────────────────────────────────

_rl_memory: dict = {}


def _reset_rl_memory():
    global _rl_memory
    _rl_memory = {
        "ollama_mistakes":     [],   # HTML structure problems Ollama made
        "groq_mistakes":       [],   # CSS problems Groq made
        "gemini_actions":      [],   # direct fixes supervisor applied
        "successful_patterns": [],   # what worked well (keep these)
        "failed_approaches":   [],   # approaches that didn't help
        "round_scores":        [],   # total score per round
        "score_breakdown":     [],   # per-round dict of dimension scores
        "handoffs":            [],   # when a worker failed and Gemini covered
        "best_html":           "",
        "best_score":          0,
        "best_round":          0,
    }


def _is_available(p: str) -> bool:
    with _lock:
        s = agent_registry[p]
        if s["available"]:
            return True
        if s["cooldown_until"] and datetime.now() >= s["cooldown_until"]:
            s["available"]     = True
            s["cooldown_until"] = None
            print(f"[A2A] 🔄 {p} recovered.")
            return True
        return False


def _mark_rate_limited(p: str, secs: int = 60):
    with _lock:
        s = agent_registry[p]
        s["available"]     = False
        s["cooldown_until"] = datetime.now() + timedelta(seconds=secs)
        s["errors"]        += 1
        print(f"[A2A] ⏳ {p} rate-limited — cooldown {secs}s")


def _mark_success(p: str):
    with _lock:
        agent_registry[p]["calls"]    += 1
        agent_registry[p]["available"] = True


def _mark_error(p: str, err: str):
    with _lock:
        agent_registry[p]["errors"]     += 1
        agent_registry[p]["last_error"]  = err[:300]


def get_agent_status() -> dict:
    return {
        k: {**v, "cooldown_until": v["cooldown_until"].strftime("%H:%M:%S")
            if v["cooldown_until"] else None}
        for k, v in agent_registry.items()
    }


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
        config=types.GenerateContentConfig(temperature=0.15, max_output_tokens=8192)
    )
    return response.text.strip()


def _call_groq(model: str, prompt: str) -> str:
    from groq import Groq
    r = Groq(api_key=os.getenv("GROQ_API_KEY")).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content":
             "You are an elite CSS/UI specialist in a supervised AI pipeline. "
             f"You MUST learn from every mistake in your context — NEVER repeat them. "
             f"Target: produce HTML+CSS that reaches {APPROVAL_THRESHOLD}% visual match vs the screenshot. "
             "Return ONLY raw HTML starting with <!doctype html>. Absolutely no markdown."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.15, max_tokens=8192
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
             "You are an elite HTML structure specialist in a supervised AI pipeline. "
             f"You MUST learn from every mistake in your context — NEVER repeat them. "
             f"Target: produce HTML structure that reaches {APPROVAL_THRESHOLD}% visual match vs the screenshot. "
             "Return ONLY raw HTML starting with <!doctype html>. Absolutely no markdown."},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return r["message"]["content"].strip()


def _clean_html(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$",        "", text)
    if "<!doctype" not in text.lower() and "<html" not in text.lower():
        return ""
    return text.strip()


GEMINI_MODELS = ["gemini-2.5-flash", "gemini-1.5-flash"]
GROQ_MODELS   = ["llama-3.3-70b-versatile", "llama3-70b-8192"]
OLLAMA_MODELS = ["gpt-oss:120b-cloud", "qwen3.5", "deepseek-v4-flash"]


def _safe_gemini(prompt: str, image_b64: str = None) -> str:
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
            if any(x in err.lower() for x in ["429", "quota", "rate limit", "too many"]):
                _mark_rate_limited("gemini", _parse_retry(err))
                break
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR — STEP 1: DEEP VISUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_deep_analyze(data: dict, image_b64: str) -> dict:
    """
    Gemini visually dissects the screenshot at pixel level.
    Produces a precise 12-field replication blueprint for workers.
    """
    prompt = f"""You are GEMINI SUPERVISOR with advanced computer vision.
Analyze this website screenshot with EXTREME precision. Every pixel matters.
Workers must reach {APPROVAL_THRESHOLD}% visual match — be hyper-specific.

=== SITE METADATA ===
URL:    {data['url']}
TITLE:  {data['title']}
COLORS: {json.dumps(data['colors'])}

Analyze the screenshot and return ONLY this JSON — no other text:
{{
  "background": {{
    "type": "solid|gradient|image",
    "css": "exact CSS background value e.g. linear-gradient(135deg, #e0d7ff 0%, #fde8d0 100%)",
    "gradient_from": "#hex",
    "gradient_to":   "#hex",
    "gradient_direction": "135deg"
  }},
  "color_scheme": {{
    "primary":    "#hex",
    "secondary":  "#hex",
    "accent":     "#hex",
    "background": "#hex",
    "text_dark":  "#hex",
    "text_light": "#hex",
    "navbar_bg":  "exact bg e.g. rgba(255,255,255,0.9)",
    "card_bg":    "#hex",
    "footer_bg":  "#hex"
  }},
  "navbar": {{
    "background":        "exact CSS value",
    "has_backdrop_blur": true,
    "position":          "sticky|fixed|relative",
    "height":            "approx px",
    "logo_text":         "exact logo text",
    "logo_has_icon":     true,
    "nav_links":         ["link1","link2"],
    "cta_text":          "exact CTA text",
    "cta_bg":            "#hex",
    "cta_text_color":    "#hex"
  }},
  "hero": {{
    "layout":             "center|left|split",
    "has_logo_icon":      true,
    "logo_icon_shape":    "describe shape+color e.g. purple rounded square with white B",
    "logo_icon_size":     "approx px",
    "headline":           "exact headline text",
    "headline_font_size": "approx px",
    "headline_weight":    "700|800|900",
    "headline_color":     "#hex",
    "subtext":            "first 80 chars of subtext",
    "subtext_color":      "#hex",
    "has_code_snippet":   false,
    "code_snippet_text":  "exact code text e.g. $ npm i bootstrap@5.3.8",
    "code_snippet_style": "describe appearance",
    "cta_buttons": [
      {{"text":"btn text","bg":"#hex","text_color":"#hex","border":"none","icon":"description"}}
    ],
    "version_badge": "e.g. Currently v5.3.8 · Download · All releases",
    "padding_top":   "approx px"
  }},
  "sections_order":  ["navbar","hero","section2",...],
  "features": {{
    "heading":         "section heading text",
    "card_count":      3,
    "card_layout":     "grid-3|grid-2|flex",
    "card_bg":         "#hex",
    "card_radius":     "px",
    "card_shadow":     "exact box-shadow CSS",
    "card_padding":    "px",
    "card_has_icon":   true,
    "card_icon_color": "#hex"
  }},
  "typography": {{
    "font_import":    "Google Fonts @import URL",
    "font_family":    "font-family CSS value",
    "hero_size":      "px",
    "h2_size":        "px",
    "body_size":      "px",
    "line_height":    "1.x",
    "letter_spacing": "em or px"
  }},
  "spacing": {{
    "section_v_padding": "px",
    "container_max_w":   "px",
    "navbar_height":     "px"
  }},
  "footer": {{
    "bg":       "#hex",
    "color":    "#hex",
    "has_links": true
  }},
  "special_elements":        ["describe each unique UI element visible"],
  "critical_must_replicate": ["top 7 elements that make this site instantly recognisable"]
}}"""

    try:
        raw = _safe_gemini(prompt, image_b64)
        m   = re.search(r"\{.*\}", raw, re.DOTALL)
        bp  = json.loads(m.group(0) if m else raw)
        print(f"[SUPERVISOR] ✅ Blueprint: {len(bp)} fields extracted")
        return bp
    except Exception as e:
        print(f"[SUPERVISOR] ⚠️ Blueprint parse failed: {e} — using text fallback")
        c = data.get("colors", [])
        return {
            "background": {
                "type": "gradient",
                "css":  "linear-gradient(135deg, #e0d7ff 0%, #fde8d0 100%)",
                "gradient_from":      c if len(c) > 0 else "#e0d7ff",
                "gradient_to":        c[1] if len(c) > 1 else "#fde8d0",
                "gradient_direction": "135deg"
            },
            "color_scheme": {
                "primary":    c if len(c) > 0 else "#7952b3",
                "secondary":  c if len(c) > 1 else "#6610f2",
                "accent":     c if len(c) > 2 else "#7952b3",
                "background": "#ffffff",
                "text_dark":  "#212529",
                "text_light": "#6c757d",
                "navbar_bg":  "rgba(255,255,255,0.9)",
                "card_bg":    "#ffffff",
                "footer_bg":  "#212529"
            },
            "navbar": {
                "background": "rgba(255,255,255,0.9)", "has_backdrop_blur": True,
                "position": "sticky", "height": "60px",
                "logo_text": data.get("title", "Site")[:20], "logo_has_icon": True,
                "nav_links": [l["text"] for l in data.get("nav_links", [])[:6]],
                "cta_text": "Get started",
                "cta_bg": c if len(c) > 0 else "#7952b3",
                "cta_text_color": "#fff"
            },
            "hero": {
                "layout": "center", "has_logo_icon": True,
                "logo_icon_shape": "rounded square brand icon",
                "logo_icon_size": "80px",
                "headline": data.get("title", ""),
                "headline_font_size": "64px", "headline_weight": "700",
                "headline_color": "#212529",
                "subtext": data.get("meta_desc", "")[:120],
                "subtext_color": "#6c757d",
                "has_code_snippet": False, "code_snippet_text": "",
                "code_snippet_style": "",
                "cta_buttons": [{"text": "Get started",
                                  "bg": c if len(c) > 0 else "#7952b3",
                                  "text_color": "#fff", "border": "none", "icon": ""}],
                "version_badge": "", "padding_top": "80px"
            },
            "sections_order": ["navbar", "hero", "features", "stats", "cta", "footer"],
            "features": {
                "heading": "Features", "card_count": 3, "card_layout": "grid-3",
                "card_bg": "#ffffff", "card_radius": "12px",
                "card_shadow": "0 4px 20px rgba(0,0,0,0.08)",
                "card_padding": "32px", "card_has_icon": True,
                "card_icon_color": c if len(c) > 0 else "#7952b3"
            },
            "typography": {
                "font_import":    "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap",
                "font_family":    "'Inter', system-ui, sans-serif",
                "hero_size":      "64px", "h2_size": "36px",
                "body_size":      "16px", "line_height": "1.6",
                "letter_spacing": "normal"
            },
            "spacing": {
                "section_v_padding": "80px",
                "container_max_w":   "1200px",
                "navbar_height":     "60px"
            },
            "footer": {"bg": "#212529", "color": "#adb5bd", "has_links": True},
            "special_elements":        [],
            "critical_must_replicate": [
                "gradient background", "hero headline", "logo icon",
                "code snippet", "navbar", "feature cards", "footer"
            ]
        }


# ─────────────────────────────────────────────────────────────────────────────
# WORKER PROMPT BUILDERS — inject full RL memory every round
# ─────────────────────────────────────────────────────────────────────────────

def _build_ollama_prompt(data: dict, bp: dict, round_num: int,
                          corrections: str = "") -> str:
    nl = "\n"

    mistakes_block = (
        f"\n=== ❌ YOUR PAST HTML MISTAKES — DO NOT REPEAT ===\n"
        + nl.join(f"  [{i+1}] {m}" for i, m in
                  enumerate(_rl_memory.get("ollama_mistakes", [])[-8:]))
        + "\n"
    ) if _rl_memory.get("ollama_mistakes") else ""

    wins_block = (
        f"\n=== ✅ HTML PATTERNS THAT WORKED — KEEP THESE ===\n"
        + nl.join(f"  • {p}" for p in _rl_memory.get("successful_patterns", [])[-4:])
        + "\n"
    ) if _rl_memory.get("successful_patterns") else ""

    handoff_block = (
        f"\n=== 🤝 SUPERVISOR HANDOFF NOTES ===\n"
        + nl.join(f"  • {h}" for h in _rl_memory.get("handoffs", [])[-3:])
        + "\n"
    ) if _rl_memory.get("handoffs") else ""

    score_hist     = " → ".join(str(s) for s in _rl_memory.get("round_scores", []))
    correction_blk = (
        f"\n=== 📌 SUPERVISOR CORRECTIONS (Round {round_num}) ===\n{corrections}\n"
    ) if corrections else ""

    hero   = bp.get("hero",        {})
    nav    = bp.get("navbar",      {})
    bg     = bp.get("background",  {})
    typo   = bp.get("typography",  {})
    feats  = bp.get("features",    {})
    colors = bp.get("color_scheme",{})
    space  = bp.get("spacing",     {})
    footer = bp.get("footer",      {})

    return f"""You are the HTML STRUCTURE WORKER (Ollama) — Round {round_num}/{MAX_ROUNDS}.
TARGET: Reach {APPROVAL_THRESHOLD}% visual match vs original screenshot.
Score history: {score_hist or 'Round 1 — no history yet'}
{mistakes_block}{wins_block}{handoff_block}{correction_blk}

══ PIXEL-PERFECT VISUAL BLUEPRINT ══

BACKGROUND CSS:  {bg.get('css', f"linear-gradient({bg.get('gradient_direction','135deg')}, {bg.get('gradient_from','#e0d7ff')}, {bg.get('gradient_to','#fde8d0')})")}

COLORS:
  primary:    {colors.get('primary')}
  secondary:  {colors.get('secondary')}
  text dark:  {colors.get('text_dark')}
  text light: {colors.get('text_light')}
  card bg:    {colors.get('card_bg')}
  footer bg:  {colors.get('footer_bg')}

NAVBAR:
  bg:         {nav.get('background')}
  blur:       {nav.get('has_backdrop_blur')}
  height:     {nav.get('height')}
  logo:       "{nav.get('logo_text')}"  icon={nav.get('logo_has_icon')}
  links:      {json.dumps(nav.get('nav_links', []))}
  cta:        "{nav.get('cta_text')}" bg={nav.get('cta_bg')}

HERO:
  layout:        {hero.get('layout')}
  logo icon:     {hero.get('has_logo_icon')} — {hero.get('logo_icon_shape')} size={hero.get('logo_icon_size')}
  headline:      "{hero.get('headline')}"
  headline:      {hero.get('headline_font_size')} / weight={hero.get('headline_weight')} / color={hero.get('headline_color')}
  subtext:       "{hero.get('subtext')}"
  code snippet:  {hero.get('has_code_snippet')} — "{hero.get('code_snippet_text')}" style="{hero.get('code_snippet_style')}"
  cta buttons:   {json.dumps(hero.get('cta_buttons', []))}
  version badge: "{hero.get('version_badge')}"

FEATURES:
  heading:      "{feats.get('heading')}"
  cards:        {feats.get('card_count')} in {feats.get('card_layout')} layout
  card:         bg={feats.get('card_bg')} radius={feats.get('card_radius')} shadow="{feats.get('card_shadow')}" padding={feats.get('card_padding')}
  icon color:   {feats.get('card_icon_color')}

TYPOGRAPHY:
  font:   {typo.get('font_family')}
  hero:   {typo.get('hero_size')}
  h2:     {typo.get('h2_size')}
  body:   {typo.get('body_size')}

SPACING:  section padding={space.get('section_v_padding')} | max-width={space.get('container_max_w')}
FOOTER:   bg={footer.get('bg')} color={footer.get('color')}
SECTIONS: {json.dumps(bp.get('sections_order', []))}
SPECIAL:  {json.dumps(bp.get('special_elements', []))}
CRITICAL: {json.dumps(bp.get('critical_must_replicate', []))}

══ SITE CONTENT ══
URL:      {data['url']}
TITLE:    {data['title']}
HEADINGS: {json.dumps([h['text'] for h in data['headings'][:12]])}
NAV:      {json.dumps([l['text'] for l in data['nav_links'][:10]])}
CONTENT:  {data['text'][:2500]}

══ STRICT RULES ══
1. Build ALL sections IN ORDER: {json.dumps(bp.get('sections_order', ['navbar','hero','features','footer']))}
2. CSS class names: .navbar .hero .hero-logo .hero-code .features-grid .card .footer
3. Embed full <style> block with :root CSS variables
4. If code snippet visible → build exact styled <pre><code> block
5. If logo icon → build a CSS-only styled div (no external images for icons)
6. Placeholder images: https://placehold.co/600x400
7. Return ONLY <!doctype html>...</html> — ZERO markdown, ZERO explanations"""


def _build_groq_prompt(data: dict, bp: dict, ollama_html: str,
                        round_num: int, corrections: str = "") -> str:
    nl = "\n"

    mistakes_block = (
        f"\n=== ❌ YOUR PAST CSS MISTAKES — DO NOT REPEAT ===\n"
        + nl.join(f"  [{i+1}] {m}" for i, m in
                  enumerate(_rl_memory.get("groq_mistakes", [])[-8:]))
        + "\n"
    ) if _rl_memory.get("groq_mistakes") else ""

    wins_block = (
        f"\n=== ✅ CSS PATTERNS THAT WORKED — KEEP ===\n"
        + nl.join(f"  • {p}" for p in _rl_memory.get("successful_patterns", [])[-4:])
        + "\n"
    ) if _rl_memory.get("successful_patterns") else ""

    score_hist     = " → ".join(str(s) for s in _rl_memory.get("round_scores", []))
    correction_blk = (
        f"\n=== 📌 SUPERVISOR CSS CORRECTIONS (Round {round_num}) ===\n{corrections}\n"
    ) if corrections else ""

    bg     = bp.get("background",  {})
    colors = bp.get("color_scheme",{})
    typo   = bp.get("typography",  {})
    feats  = bp.get("features",    {})
    space  = bp.get("spacing",     {})
    hero   = bp.get("hero",        {})
    footer = bp.get("footer",      {})
    nav    = bp.get("navbar",      {})

    return f"""You are the CSS STYLING WORKER (Groq) — Round {round_num}/{MAX_ROUNDS}.
TARGET: Reach {APPROVAL_THRESHOLD}% visual match vs original screenshot.
Score history: {score_hist or 'Round 1 — no history yet'}
{mistakes_block}{wins_block}{correction_blk}

══ EXACT SPECS TO MATCH ══

MANDATORY :root variables:
  --primary:    {colors.get('primary',   '#7952b3')};
  --secondary:  {colors.get('secondary', '#6610f2')};
  --accent:     {colors.get('accent',    '#7952b3')};
  --text:       {colors.get('text_dark', '#212529')};
  --text-muted: {colors.get('text_light','#6c757d')};
  --card-bg:    {colors.get('card_bg',   '#ffffff')};
  --footer-bg:  {colors.get('footer_bg', '#212529')};
  --bg-from:    {bg.get('gradient_from', '#e0d7ff')};
  --bg-to:      {bg.get('gradient_to',   '#fde8d0')};

BODY BACKGROUND: {bg.get('css', f"linear-gradient({bg.get('gradient_direction','135deg')}, {bg.get('gradient_from','#e0d7ff')}, {bg.get('gradient_to','#fde8d0')})")}
FONT IMPORT:     {typo.get('font_import', 'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap')}
FONT FAMILY:     {typo.get('font_family', "'Inter', system-ui, sans-serif")}
HERO FONT:       {typo.get('hero_size', '64px')} / weight {hero.get('headline_weight', '700')} / color {hero.get('headline_color','#212529')}
SECTION PADDING: {space.get('section_v_padding', '80px')} top/bottom
CONTAINER MAX-W: {space.get('container_max_w', '1200px')}
NAVBAR:          bg={nav.get('background','rgba(255,255,255,0.9)')} blur={nav.get('has_backdrop_blur',True)} height={nav.get('height','60px')}
CARD:            bg={feats.get('card_bg','#fff')} radius={feats.get('card_radius','12px')} shadow="{feats.get('card_shadow','0 4px 20px rgba(0,0,0,0.08)')}"
FOOTER:          bg={footer.get('bg','#212529')} color={footer.get('color','#adb5bd')}

══ OLLAMA'S HTML (apply complete CSS to this) ══
{ollama_html[:5500]}

══ MANDATORY CSS ADDITIONS ══
1. REPLACE the entire <style> block with a comprehensive one
2. @import Google Font at the very top of <style>
3. *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
4. html {{ scroll-behavior: smooth; }}
5. body: gradient background, min-height 100vh, font-family set
6. .navbar: position {nav.get('position','sticky')} top-0, z-index 1000, {'backdrop-filter: blur(12px);' if nav.get('has_backdrop_blur') else ''}
7. .hero: padding-top {hero.get('padding_top','80px')}, text-align {hero.get('layout','center')}
8. .hero-headline: font-size {typo.get('hero_size','64px')}, font-weight {hero.get('headline_weight','700')}
9. .hero-code: styled code block if site has one (dark bg #1e1e2e, monospace, border-radius 8px)
10. .features-grid: display grid, grid-template-columns repeat({feats.get('card_count',3)},1fr), gap 24px
11. .card {{ transition: transform 0.2s, box-shadow 0.2s; }}
    .card:hover {{ transform: translateY(-6px); box-shadow: 0 16px 48px rgba(0,0,0,0.18); }}
12. @keyframes fadeInUp — apply staggered to .hero > * children
13. @media (max-width:768px): grids → 1 col, hero font 36px, navbar links hidden
14. Buttons: border-radius 8px, padding 12px 28px, cursor pointer, transition

Return the COMPLETE HTML with your new <style> replacing old one. Start <!doctype html>. ZERO markdown."""


# ─────────────────────────────────────────────────────────────────────────────
# WORKER RUNNERS WITH SMART HANDOFF TO GEMINI
# ─────────────────────────────────────────────────────────────────────────────

def _run_ollama_with_handoff(prompt: str, round_num: int) -> tuple:
    """
    Runs Ollama for HTML. If unavailable, Gemini supervisor covers the task.
    Returns (html, model_used, agent_name, duration_s)
    """
    for model in OLLAMA_MODELS:
        if not _is_available("ollama"):
            break
        try:
            t    = time.time()
            raw  = _call_ollama(model, prompt)
            html = _clean_html(raw)
            if len(html) > 500:
                _mark_success("ollama")
                dur = round(time.time() - t, 1)
                print(f"[OLLAMA R{round_num}] ✅ {model} → {len(html):,} chars in {dur}s")
                return html, model, "ollama", dur
        except Exception as e:
            err = str(e)
            _mark_error("ollama", err)
            if any(x in err.lower() for x in ["429", "quota", "rate limit", "too many"]):
                _mark_rate_limited("ollama", _parse_retry(err))
                break

    # ── HANDOFF: Gemini covers Ollama's HTML task ──
    print(f"[SUPERVISOR] 🤝 Ollama unavailable R{round_num} — Gemini taking over HTML task")
    _rl_memory["handoffs"].append(f"R{round_num}: Gemini covered Ollama HTML task")
    t    = time.time()
    raw  = _safe_gemini(prompt)
    html = _clean_html(raw)
    dur  = round(time.time() - t, 1)
    print(f"[GEMINI COVER] HTML task done → {len(html):,} chars in {dur}s")
    return html, "gemini-html-cover", "gemini", dur


def _run_groq_with_handoff(prompt: str, ollama_html: str,
                             round_num: int, bp: dict) -> tuple:
    """
    Runs Groq for CSS. If unavailable, Gemini supervisor covers the styling task.
    Returns (html, model_used, agent_name, duration_s)
    """
    for model in GROQ_MODELS:
        if not _is_available("groq"):
            break
        try:
            t    = time.time()
            raw  = _call_groq(model, prompt)
            html = _clean_html(raw)
            if len(html) > 500:
                _mark_success("groq")
                dur = round(time.time() - t, 1)
                print(f"[GROQ R{round_num}] ✅ {model} → {len(html):,} chars in {dur}s")
                return html, model, "groq", dur
        except Exception as e:
            err = str(e)
            _mark_error("groq", err)
            if any(x in err.lower() for x in ["429", "quota", "rate limit", "too many"]):
                _mark_rate_limited("groq", _parse_retry(err))
                break

    # ── HANDOFF: Gemini covers Groq's CSS task ──
    print(f"[SUPERVISOR] 🤝 Groq unavailable R{round_num} — Gemini taking over CSS task")
    _rl_memory["handoffs"].append(f"R{round_num}: Gemini covered Groq CSS styling task")
    bg     = bp.get("background",  {})
    colors = bp.get("color_scheme",{})
    typo   = bp.get("typography",  {})
    css_prompt = f"""Add complete professional CSS to this HTML to reach {APPROVAL_THRESHOLD}% visual match.
Body background: {bg.get('css')}
Primary color: {colors.get('primary')}
Font: {typo.get('font_family')} | Hero size: {typo.get('hero_size')}
Critical: {json.dumps(bp.get('critical_must_replicate',[]))}
HTML:
{ollama_html[:5000]}
Return full HTML. No markdown."""
    t    = time.time()
    raw  = _safe_gemini(css_prompt)
    html = _clean_html(raw) or ollama_html
    dur  = round(time.time() - t, 1)
    print(f"[GEMINI COVER] CSS task done → {len(html):,} chars in {dur}s")
    return html, "gemini-css-cover", "gemini", dur


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR — STRICT 10-DIMENSION VISUAL SCORING + DEEP RL FEEDBACK
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_strict_score(data: dict, bp: dict, html: str,
                              image_b64: str, round_num: int) -> dict:
    """
    Gemini does STRICT visual comparison against the original screenshot.
    Scores 10 dimensions (10 pts each = 100 total).
    Approved ONLY if total >= APPROVAL_THRESHOLD (90).
    Produces per-worker actionable RL corrections.
    Can flag if supervisor should directly patch something.
    """
    prev_scores = _rl_memory.get("round_scores", [])
    score_trend = " → ".join(str(s) for s in prev_scores) if prev_scores else "First round"

    prompt = f"""You are GEMINI SUPERVISOR doing STRICT quality verification — Round {round_num}.
You have the ORIGINAL screenshot AND the HTML clone generated by workers.
Score with NO mercy. Threshold to approve: {APPROVAL_THRESHOLD}/100.

Score trend: {score_trend}
Ollama known mistakes: {json.dumps(_rl_memory.get('ollama_mistakes', [])[-4:])}
Groq known mistakes:   {json.dumps(_rl_memory.get('groq_mistakes',   [])[-4:])}
Handoffs so far:       {json.dumps(_rl_memory.get('handoffs', []))}

=== ORIGINAL VISUAL REQUIREMENTS ===
Background CSS:    {bp.get('background', {}).get('css')}
Primary color:     {bp.get('color_scheme', {}).get('primary')}
Hero headline:     "{bp.get('hero', {}).get('headline')}"
Hero font size:    {bp.get('typography', {}).get('hero_size')} / weight {bp.get('hero', {}).get('headline_weight')}
Hero has code:     {bp.get('hero', {}).get('has_code_snippet')} — "{bp.get('hero', {}).get('code_snippet_text')}"
Hero logo icon:    {bp.get('hero', {}).get('has_logo_icon')} — {bp.get('hero', {}).get('logo_icon_shape')}
Version badge:     "{bp.get('hero', {}).get('version_badge')}"
Font family:       {bp.get('typography', {}).get('font_family')}
Critical elements: {json.dumps(bp.get('critical_must_replicate', []))}

=== GENERATED HTML (first 4000 chars) ===
{html[:4000]}

Compare the screenshot image with the HTML above carefully.
Deduct points for EVERY mismatch.

Return ONLY this JSON — no other text:
{{
  "scores": {{
    "background_gradient":  0-10,
    "color_accuracy":       0-10,
    "navbar":               0-10,
    "hero_layout":          0-10,
    "hero_logo_icon":       0-10,
    "hero_code_snippet":    0-10,
    "typography_sizes":     0-10,
    "feature_cards":        0-10,
    "footer":               0-10,
    "responsive_css":       0-10
  }},
  "total_score": "sum of all 10 scores above",
  "approved": "true only if total_score >= {APPROVAL_THRESHOLD}",
  "what_is_still_wrong": [
    "very specific issue 1 — name the element and the exact fix needed",
    "very specific issue 2",
    "very specific issue 3",
    "very specific issue 4"
  ],
  "ollama_mistakes_this_round": [
    "HTML structure problem Ollama made — be specific",
    "another HTML problem"
  ],
  "groq_mistakes_this_round": [
    "CSS problem Groq made — be specific with property and value",
    "another CSS problem"
  ],
  "ollama_next_round_fix": "precise HTML instruction — name exact tags, classes, structure changes needed",
  "groq_next_round_fix":   "precise CSS instruction — name exact properties, values, selectors",
  "what_worked_well": ["specific element or pattern that looks correct"],
  "supervisor_direct_fix_needed": false,
  "supervisor_fix_instruction":   "if true: describe exactly what HTML/CSS to patch and where"
}}"""

    try:
        raw    = _safe_gemini(prompt, image_b64)
        m      = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(m.group(0) if m else raw)
        # Enforce strict threshold — never accept AI's own approved=true below threshold
        total           = int(result.get("total_score", 0))
        result["total_score"] = total
        result["approved"]    = total >= APPROVAL_THRESHOLD
        sd = result.get("scores", {})
        print(
            f"[SUPERVISOR] 📊 R{round_num} → "
            f"bg={sd.get('background_gradient',0)} clr={sd.get('color_accuracy',0)} "
            f"nav={sd.get('navbar',0)} hero={sd.get('hero_layout',0)} "
            f"icon={sd.get('hero_logo_icon',0)} code={sd.get('hero_code_snippet',0)} "
            f"typo={sd.get('typography_sizes',0)} cards={sd.get('feature_cards',0)} "
            f"footer={sd.get('footer',0)} resp={sd.get('responsive_css',0)} "
            f"= {total}/100 {'✅ APPROVED' if result['approved'] else f'🔄 ({APPROVAL_THRESHOLD-total} to go)'}"
        )
        return result
    except Exception as e:
        print(f"[SUPERVISOR] ⚠️ Scoring parse error: {e} — conservative fallback (no auto-approve)")
        return {
            "scores": {k: 6 for k in [
                "background_gradient", "color_accuracy", "navbar",
                "hero_layout", "hero_logo_icon", "hero_code_snippet",
                "typography_sizes", "feature_cards", "footer", "responsive_css"
            ]},
            "total_score": 60, "approved": False,
            "what_is_still_wrong":        ["Scoring unavailable — retry next round"],
            "ollama_mistakes_this_round":  [],
            "groq_mistakes_this_round":    [],
            "ollama_next_round_fix":       "Rebuild all sections more precisely per blueprint",
            "groq_next_round_fix":         "Apply exact gradient background, colors, and font sizes from blueprint",
            "what_worked_well":            [],
            "supervisor_direct_fix_needed": False,
            "supervisor_fix_instruction":  ""
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR — DIRECT HTML PATCH (when workers keep missing a specific issue)
# ─────────────────────────────────────────────────────────────────────────────

def _supervisor_direct_fix(html: str, bp: dict, instruction: str) -> str:
    """
    Gemini directly patches the HTML for issues workers keep failing to fix.
    Logs the action into RL memory so workers know supervisor intervened.
    """
    print(f"[SUPERVISOR] 🔧 Direct fix: {instruction[:80]}...")
    _rl_memory["gemini_actions"].append(f"Direct fix: {instruction[:80]}")
    prompt = f"""You are GEMINI SUPERVISOR directly patching a specific issue in the HTML clone.

ISSUE TO FIX: {instruction}

BLUEPRINT REFERENCE:
  Background: {bp.get('background',{}).get('css')}
  Primary:    {bp.get('color_scheme',{}).get('primary')}
  Hero font:  {bp.get('typography',{}).get('hero_size')} / {bp.get('hero',{}).get('headline_weight')}
  Critical:   {json.dumps(bp.get('critical_must_replicate',[]))}

CURRENT HTML:
{html[:6000]}

Fix ONLY the described issue. Keep everything else intact.
Return the COMPLETE fixed HTML. No markdown."""
    raw   = _safe_gemini(prompt)
    fixed = _clean_html(raw)
    if len(fixed) > 500:
        print(f"[SUPERVISOR] ✅ Direct fix applied ({len(fixed):,} chars)")
        return fixed
    print("[SUPERVISOR] ⚠️ Direct fix returned bad HTML — keeping previous")
    return html


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_chain(data: dict) -> dict:
    """
    SUPERVISED RL PIPELINE — 90% approval threshold:

    Stage 1: Gemini deep visual analysis → 12-field pixel-level blueprint

    Loop up to MAX_ROUNDS (5):
      a) Ollama builds HTML with blueprint + full RL mistake memory injected
         └─ if Ollama fails → Gemini covers HTML task (handoff logged)
      b) Groq styles HTML with blueprint + full RL mistake memory injected
         └─ if Groq fails → Gemini covers CSS task (handoff logged)
      c) Gemini strict-scores 10 dimensions vs original screenshot
      d) RL memory updated: mistakes, wins, handoffs, score breakdown
      e) If supervisor flags direct-fix needed → Gemini patches HTML
      f) If total_score >= 90 → APPROVED, stop
      g) Else → next round with targeted per-worker corrections

    Returns best HTML across all rounds.
    """
    _reset_rl_memory()
    logs      = []
    shot_path = data.get("_screenshot_path", "")

    image_b64 = None
    if shot_path and os.path.exists(shot_path):
        with open(shot_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        print("[SUPERVISOR] 📸 Screenshot loaded for visual analysis")
    else:
        print("[SUPERVISOR] ⚠️ No screenshot found — text-only mode")

    # ══════════════════════════════════════════════════════
    # Stage 1 — Deep Visual Analysis
    # ══════════════════════════════════════════════════════
    print("\n[SUPERVISOR] 🔍 Stage 1: Deep visual analysis...")
    t0        = time.time()
    blueprint = supervisor_deep_analyze(data, image_b64)
    dur0      = round(time.time() - t0, 1)

    logs.append({
        "stage":    "🔍 Stage 1 — Deep Visual Analysis",
        "agent":    "Gemini Supervisor",
        "icon":     "🧠",
        "role":     "Pixel-level screenshot analysis → 12-field precise blueprint",
        "result":   "✅ Blueprint ready",
        "chars":    len(json.dumps(blueprint)),
        "duration": dur0,
        "details":  [
            {"agent": "Background", "status": str(blueprint.get("background",{}).get("css",""))[:80],      "duration": 0, "chars": 0, "reason": ""},
            {"agent": "Colors",     "status": str(blueprint.get("color_scheme",{}))[:80],                  "duration": 0, "chars": 0, "reason": ""},
            {"agent": "Hero",       "status": f"headline='{blueprint.get('hero',{}).get('headline','')}'", "duration": 0, "chars": 0, "reason": ""},
            {"agent": "Critical",   "status": str(blueprint.get("critical_must_replicate",[]))[:100],      "duration": 0, "chars": 0, "reason": ""},
        ]
    })

    # ══════════════════════════════════════════════════════
    # Stage 2 — RL Refinement Loop (up to MAX_ROUNDS)
    # ══════════════════════════════════════════════════════
    best_html      = ""
    best_score     = 0
    ollama_correct = ""
    groq_correct   = ""

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'═'*58}")
        print(f"[PIPELINE] 🔄 ROUND {round_num}/{MAX_ROUNDS} — target: {APPROVAL_THRESHOLD}%")
        print(f"{'═'*58}")

        # ── Ollama: HTML Structure ───────────────────────────────
        ollama_prompt = _build_ollama_prompt(data, blueprint, round_num, ollama_correct)
        ollama_html, ollama_model, ollama_agent, ollama_dur = \
            _run_ollama_with_handoff(ollama_prompt, round_num)
        o_note = " 🤝 Gemini covered" if ollama_agent == "gemini" else ""

        logs.append({
            "stage":    f"🦙 Round {round_num} — HTML Structure (Ollama{o_note})",
            "agent":    f"Ollama Worker{o_note}",
            "icon":     "🦙",
            "role":     f"HTML from blueprint {'+ RL corrections applied' if ollama_correct else '(first draft)'}",
            "result":   f"✅ {len(ollama_html):,} chars" if ollama_html else "❌ Failed",
            "chars":    len(ollama_html),
            "duration": ollama_dur,
            "details":  [{"agent": ollama_model, "status": "success" if ollama_html else "failed",
                          "duration": ollama_dur, "chars": len(ollama_html), "reason": o_note.strip()}]
        })

        # ── Groq: CSS Styling ────────────────────────────────────
        groq_prompt = _build_groq_prompt(data, blueprint, ollama_html, round_num, groq_correct)
        groq_html, groq_model, groq_agent, groq_dur = \
            _run_groq_with_handoff(groq_prompt, ollama_html, round_num, blueprint)
        g_note = " 🤝 Gemini covered" if groq_agent == "gemini" else ""

        best_this_round = groq_html or ollama_html
        logs.append({
            "stage":    f"⚡ Round {round_num} — CSS Styling (Groq{g_note})",
            "agent":    f"Groq Worker{g_note}",
            "icon":     "⚡",
            "role":     f"CSS styling {'+ RL corrections applied' if groq_correct else '(first style pass)'}",
            "result":   f"✅ {len(groq_html):,} chars" if groq_html else "❌ Failed",
            "chars":    len(groq_html),
            "duration": groq_dur,
            "details":  [{"agent": groq_model, "status": "success" if groq_html else "failed",
                          "duration": groq_dur, "chars": len(groq_html), "reason": g_note.strip()}]
        })

        # ── Gemini: Strict Score ─────────────────────────────────
        t_score  = time.time()
        feedback = supervisor_strict_score(data, blueprint, best_this_round, image_b64, round_num)
        score    = feedback.get("total_score", 0)
        approved = feedback.get("approved", False)
        score_dur= round(time.time() - t_score, 1)
        sd       = feedback.get("scores", {})

        # Update RL memory
        _rl_memory["round_scores"].append(score)
        _rl_memory["score_breakdown"].append(sd)
        _rl_memory["ollama_mistakes"].extend(feedback.get("ollama_mistakes_this_round", []))
        _rl_memory["groq_mistakes"].extend(feedback.get("groq_mistakes_this_round", []))
        _rl_memory["successful_patterns"].extend(feedback.get("what_worked_well", []))

        if score > best_score:
            best_score               = score
            best_html                = best_this_round
            _rl_memory["best_score"] = best_score
            _rl_memory["best_round"] = round_num

        ollama_correct = feedback.get("ollama_next_round_fix", "")
        groq_correct   = feedback.get("groq_next_round_fix",   "")

        # Supervisor direct fix if workers keep missing something
        if feedback.get("supervisor_direct_fix_needed") and feedback.get("supervisor_fix_instruction"):
            best_html = _supervisor_direct_fix(
                best_html, blueprint, feedback["supervisor_fix_instruction"])

        logs.append({
            "stage":    f"📊 Round {round_num} — Supervisor Score: {score}/100",
            "agent":    "Gemini Supervisor",
            "icon":     "🧠",
            "role":     (f"10-dimension strict scoring | RL corrections → "
                         f"{'✅ APPROVED' if approved else f'next round ({APPROVAL_THRESHOLD-score} pts to threshold)'}"),
            "result":   f"{'✅ APPROVED' if approved else f'🔄 {score}/100 — continuing'}",
            "chars":    len(best_this_round),
            "duration": score_dur,
            "details":  [{
                "agent":  (f"bg={sd.get('background_gradient',0)} clr={sd.get('color_accuracy',0)} "
                           f"nav={sd.get('navbar',0)} hero={sd.get('hero_layout',0)} "
                           f"icon={sd.get('hero_logo_icon',0)}"),
                "status": (f"code={sd.get('hero_code_snippet',0)} typo={sd.get('typography_sizes',0)} "
                           f"cards={sd.get('feature_cards',0)} footer={sd.get('footer',0)} "
                           f"resp={sd.get('responsive_css',0)}"),
                "duration": score_dur,
                "chars":    score,
                "reason":   " | ".join(feedback.get("what_is_still_wrong", [])[:2])
            }]
        })

        print(
            f"[PIPELINE] {'✅ APPROVED' if approved else f'🔄 {score}/100 ({APPROVAL_THRESHOLD-score} to go)'} — "
            f"best so far: {best_score} (R{_rl_memory['best_round']})"
        )
        print(
            f"[RL] ollama_mistakes={len(_rl_memory['ollama_mistakes'])} | "
            f"groq_mistakes={len(_rl_memory['groq_mistakes'])} | "
            f"wins={len(_rl_memory['successful_patterns'])} | "
            f"handoffs={len(_rl_memory['handoffs'])}"
        )

        if approved:
            print(f"[SUPERVISOR] 🏁 Target {APPROVAL_THRESHOLD}% reached at round {round_num}!")
            break
        elif round_num < MAX_ROUNDS:
            print(f"[SUPERVISOR] 🔄 Round {round_num+1}: corrections injected into all workers...")
        else:
            print(f"[SUPERVISOR] ⏱️ Max rounds ({MAX_ROUNDS}) reached. Returning best ({best_score}%)")

    # ══════════════════════════════════════════════════════
    # Done
    # ══════════════════════════════════════════════════════
    rounds_taken = len(_rl_memory["round_scores"])
    scores_str   = " → ".join(str(s) for s in _rl_memory["round_scores"])
    status       = "✅ APPROVED" if best_score >= APPROVAL_THRESHOLD else "⚠️ Best-effort"
    used_label   = (
        f"🧠 Supervised RL | {rounds_taken} round(s) | "
        f"Scores: {scores_str} | Peak: {best_score}% {status}"
    )

    print(f"\n[SUPERVISOR] 🏁 Final: {best_score}% | {rounds_taken} round(s) | {status}")
    print(f"[SUPERVISOR] 📄 HTML: {len(best_html):,} chars")
    print(
        f"[RL SUMMARY] ollama_mistakes={len(_rl_memory['ollama_mistakes'])} | "
        f"groq_mistakes={len(_rl_memory['groq_mistakes'])} | "
        f"wins={len(_rl_memory['successful_patterns'])} | "
        f"handoffs={len(_rl_memory['handoffs'])} | "
        f"supervisor_fixes={len(_rl_memory['gemini_actions'])}\n"
    )

    return {
        "html":       best_html or _fallback_html(data),
        "agent_used": used_label,
        "logs":       logs,
        "chars":      len(best_html),
        "blueprint":  blueprint,
        "rl_memory":  {
            "rounds":           rounds_taken,
            "scores":           _rl_memory["round_scores"],
            "score_breakdown":  _rl_memory["score_breakdown"],
            "best_score":       best_score,
            "best_round":       _rl_memory["best_round"],
            "ollama_mistakes":  _rl_memory["ollama_mistakes"],
            "groq_mistakes":    _rl_memory["groq_mistakes"],
            "patterns_learned": _rl_memory["successful_patterns"],
            "handoffs":         _rl_memory["handoffs"],
            "gemini_actions":   _rl_memory["gemini_actions"],
        },
        "final_check": {
            "visual_match_score": best_score,
            "approved":           best_score >= APPROVAL_THRESHOLD
        }
    }


def _fallback_html(data: dict) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"/>'
        f'<title>{data.get("title", "Clone")}</title>'
        '<style>body{font-family:sans-serif;padding:40px;background:#f4f6f8}'
        '.c{max-width:800px;margin:auto;background:#fff;padding:40px;border-radius:20px;'
        'box-shadow:0 10px 40px rgba(0,0,0,.08)}</style></head>'
        f'<body><div class="c"><h1>{data.get("title","")}</h1>'
        '<p>⚠️ All agents exhausted. Please retry.</p></div></body></html>'
    )