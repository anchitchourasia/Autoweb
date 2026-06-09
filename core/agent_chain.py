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


def _mark_rate_limited(provider: str, retry_after_seconds: int = 60):
    with _lock:
        s = agent_registry[provider]
        s["available"] = False
        s["cooldown_until"] = datetime.now() + timedelta(seconds=retry_after_seconds)
        s["errors"] += 1
        print(f"[A2A] ⏳ {provider} cooldown {retry_after_seconds}s "
              f"→ recovers at {s['cooldown_until'].strftime('%H:%M:%S')}")


def _mark_success(provider: str):
    with _lock:
        agent_registry[provider]["calls"] += 1
        agent_registry[provider]["available"] = True


def _mark_error(provider: str, error: str):
    with _lock:
        agent_registry[provider]["errors"] += 1
        agent_registry[provider]["last_error"] = error[:300]


def get_agent_status() -> dict:
    return {
        k: {
            **v,
            "cooldown_until": v["cooldown_until"].strftime("%H:%M:%S")
                              if v["cooldown_until"] else None
        }
        for k, v in agent_registry.items()
    }


def _parse_retry_after(error_str: str) -> int:
    match = re.search(r"retry.{0,10}?(\d+)", error_str, re.IGNORECASE)
    if match:
        return max(int(match.group(1)), 5)
    return 60


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_structure(data: dict, partial_html: str = "") -> str:
    return f"""You are a senior frontend developer and UI/UX expert.
Analyze the website data below and generate a COMPLETE, BEAUTIFUL, PIXEL-PERFECT
single-file HTML clone with all CSS inside a <style> tag.

=== WEBSITE DATA ===
URL:         {data['url']}
TITLE:       {data['title']}
DESCRIPTION: {data['meta_desc']}
COLORS:      {json.dumps(data['colors'])}
HEADINGS:    {json.dumps(data['headings'], ensure_ascii=False)}
NAV LINKS:   {json.dumps(data['nav_links'], ensure_ascii=False)}
CTA BUTTONS: {json.dumps(data['buttons'], ensure_ascii=False)}
CONTENT:     {data['text'][:4500]}

=== REQUIREMENTS ===
1. Return ONLY raw HTML. Zero markdown. Zero explanation. Start with <!doctype html>
2. Sections: sticky navbar → hero → features/cards → stats → CTA banner → footer
3. Use the detected color palette authentically
4. Google Fonts @import allowed (prefer Inter or Poppins)
5. Placeholder images: https://placehold.co/600x400?text=Preview
6. CSS Grid + Flexbox, mobile responsive @media at 768px and 480px
7. Smooth hover transitions (0.3s ease) on all buttons and cards
8. Card shadows and rounded corners (border-radius: 12px+)
9. Hero section with gradient background using detected colors
10. Quality must genuinely impress a senior hiring manager
"""


def _prompt_style(data: dict, partial_html: str = "") -> str:
    base = f"""You are a CSS specialist and expert frontend developer.
Generate a modern, visually stunning, fully responsive HTML clone.

=== SITE INFO ===
URL:      {data['url']}
TITLE:    {data['title']}
COLORS:   {json.dumps(data['colors'])}
HEADINGS: {json.dumps(data['headings'][:15], ensure_ascii=False)}
NAV:      {json.dumps(data['nav_links'][:12], ensure_ascii=False)}
BUTTONS:  {json.dumps(data['buttons'], ensure_ascii=False)}
CONTENT:  {data['text'][:3500]}

=== FOCUS AREAS ===
- Modern typography using Google Fonts (Inter preferred)
- CSS custom properties (--primary, --secondary, --bg, --text)
- Smooth animations: fade-in (use @keyframes), hover effects
- Card grid layout with box-shadow and border-radius
- Mobile-first responsive with @media (max-width: 768px)
- Gradient hero section, professional footer

RULES: Return ONLY raw HTML starting with <!doctype html>. No markdown, no backticks.
Placeholder images: https://placehold.co/600x400?text=Image
"""
    if partial_html and len(partial_html) > 200:
        base += f"\n\nPrevious agent generated partial output. Complete and improve:\n{partial_html[-1500:]}"
    return base


def _prompt_content(data: dict, partial_html: str = "") -> str:
    return f"""You are a frontend developer. Generate a complete HTML webpage clone.

Site:        {data['url']}
Title:       {data['title']}
Description: {data['meta_desc']}
Headings:    {json.dumps([h['text'] for h in data['headings'][:10]])}
Navigation:  {json.dumps([l['text'] for l in data['nav_links'][:10]])}
Buttons:     {json.dumps(data['buttons'][:6])}
Colors:      {json.dumps(data['colors'][:6])}
Content:     {data['text'][:3000]}

Create a single HTML file with embedded CSS that includes:
1. Navigation bar with logo and all nav links
2. Hero section with main heading and CTA button
3. Features section using headings as card titles
4. Simple footer with links

CRITICAL: Return ONLY HTML code. Start immediately with <!doctype html>
No explanations, no markdown, no code blocks.
Use the detected colors. Make it clean and professional.
Placeholder images: https://placehold.co/600x400?text=Image
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
                "content": "You are a frontend developer. Return ONLY raw HTML "
                           "with all CSS embedded. No markdown, no explanation, no code fences."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=8192
    )
    # Groq returns Pydantic object — use dot notation
    return response.choices[0].message.content.strip()


def _call_ollama(model: str, prompt: str) -> str:
    from ollama import Client
    api_key = os.getenv("OLLAMA_API_KEY")
    if api_key:
        client = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"}
        )
    else:
        client = Client(host="http://localhost:11434")

    messages = [
        {
            "role": "system",
            "content": "You are a frontend developer. Return ONLY raw HTML with embedded CSS. No markdown."
        },
        {"role": "user", "content": prompt}
    ]
    response = client.chat(model=model, messages=messages, stream=False)
    return response["message"]["content"].strip()


def _clean_html(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    if "<!doctype" not in text.lower() and "<html" not in text.lower():
        return ""
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# A2A AGENT CHAIN
# ─────────────────────────────────────────────────────────────────────────────

AGENT_CHAIN = [
    {
        "id":        "gemini-2.5-flash",
        "name":      "Gemini 2.5 Flash",
        "provider":  "gemini",
        "model":     "gemini-2.5-flash",
        "prompt_fn": _prompt_structure,
        "role":      "Primary Structure & Layout Agent",
        "icon":      "🧠"
    },
    {
        "id":        "gemini-1.5-flash",
        "name":      "Gemini 1.5 Flash",
        "provider":  "gemini",
        "model":     "gemini-1.5-flash",
        "prompt_fn": _prompt_structure,
        "role":      "Structure Agent — Gemini Fallback",
        "icon":      "🧠"
    },
    {
        "id":        "groq-llama33",
        "name":      "Groq LLaMA 3.3 70B",
        "provider":  "groq",
        "model":     "llama-3.3-70b-versatile",
        "prompt_fn": _prompt_style,
        "role":      "Style & CSS Specialist Agent",
        "icon":      "⚡"
    },
    {
        "id":        "groq-llama3",
        "name":      "Groq LLaMA 3 70B",
        "provider":  "groq",
        "model":     "llama3-70b-8192",
        "prompt_fn": _prompt_style,
        "role":      "Style Agent — Groq Fallback",
        "icon":      "⚡"
    },
    {
        "id":        "ollama-gpt-oss",
        "name":      "Ollama Cloud — GPT-OSS 120B",
        "provider":  "ollama",
        "model":     "gpt-oss:120b-cloud",
        "prompt_fn": _prompt_content,
        "role":      "Content & Structure Agent — Ollama Cloud",
        "icon":      "🦙"
    },
    {
        "id":        "ollama-qwen",
        "name":      "Ollama Cloud — Qwen3.5",
        "provider":  "ollama",
        "model":     "qwen3.5",
        "prompt_fn": _prompt_content,
        "role":      "Content Agent — Ollama Fallback 1",
        "icon":      "🦙"
    },
    {
        "id":        "ollama-deepseek",
        "name":      "Ollama Cloud — DeepSeek V4 Flash",
        "provider":  "ollama",
        "model":     "deepseek-v4-flash",
        "prompt_fn": _prompt_content,
        "role":      "Content Agent — Ollama Fallback 2",
        "icon":      "🦙"
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_chain(data: dict) -> dict:
    """
    A2A smart chain:
    - Tries each agent in priority order
    - On rate limit → extracts retry-after → puts provider on cooldown
      → immediately hands off to next available agent
    - Provider auto-recovers after cooldown expires
    - Passes partial HTML context to next agent if available
    - Returns full result dict with execution logs for the UI
    """
    logs = []
    partial_html = ""
    final_html = ""
    used_agent = "none"

    for agent in AGENT_CHAIN:
        provider = agent["provider"]
        name     = agent["name"]
        role     = agent["role"]
        icon     = agent["icon"]

        # Check if provider is available (with auto-recovery)
        if not _is_available(provider):
            cd = agent_registry[provider]["cooldown_until"]
            reason = f"On cooldown until {cd.strftime('%H:%M:%S') if cd else 'unknown'}"
            logs.append({
                "agent":    name,
                "role":     role,
                "icon":     icon,
                "status":   "skipped",
                "reason":   reason,
                "duration": 0,
                "chars":    0
            })
            print(f"[A2A] ⏭️  Skipping {name} — {reason}")
            continue

        log_entry = {
            "agent":    name,
            "role":     role,
            "icon":     icon,
            "status":   "trying",
            "reason":   "",
            "duration": 0,
            "chars":    0
        }
        logs.append(log_entry)
        print(f"[A2A] {icon} Trying {name} ({role})...")
        t_start = time.time()

        try:
            # Build specialized prompt — pass partial to style agents
            if provider in ("groq", "ollama") and partial_html:
                prompt = agent["prompt_fn"](data, partial_html)
            else:
                prompt = agent["prompt_fn"](data)

            # Call the agent
            if provider == "gemini":
                raw = _call_gemini(agent["model"], prompt)
            elif provider == "groq":
                raw = _call_groq(agent["model"], prompt)
            elif provider == "ollama":
                raw = _call_ollama(agent["model"], prompt)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            html     = _clean_html(raw)
            duration = round(time.time() - t_start, 1)

            if len(html) < 300:
                raise ValueError(f"Output too short ({len(html)} chars)")

            # ✅ SUCCESS
            _mark_success(provider)
            log_entry.update({
                "status":   "success",
                "chars":    len(html),
                "duration": duration
            })
            final_html = html
            used_agent = name
            print(f"[A2A] ✅ {name} succeeded — {len(html):,} chars in {duration}s")
            break

        except Exception as e:
            err      = str(e)
            duration = round(time.time() - t_start, 1)
            _mark_error(provider, err)

            is_rate_limit = any(
                x in err.lower()
                for x in ["429", "quota", "rate limit", "too many", "rate_limit"]
            )

            if is_rate_limit:
                retry_after = _parse_retry_after(err)
                _mark_rate_limited(provider, retry_after)
                log_entry.update({
                    "status":   "rate_limited",
                    "reason":   f"Rate limited — cooldown {retry_after}s. Next agent taking over.",
                    "duration": duration
                })
                print(f"[A2A] ⚠️  {name} rate limited → handing off to next agent")
            else:
                log_entry.update({
                    "status":   "failed",
                    "reason":   err[:150],
                    "duration": duration
                })
                print(f"[A2A] ❌ {name} failed: {err[:80]}")

            continue

    if not final_html:
        final_html = _fallback_html(data)
        used_agent = "Static Fallback"
        logs.append({
            "agent":    "Static Fallback",
            "role":     "Emergency fallback — all agents exhausted",
            "icon":     "🆘",
            "status":   "fallback",
            "reason":   "All agents failed or rate limited",
            "duration": 0,
            "chars":    len(final_html)
        })

    return {
        "html":       final_html,
        "agent_used": used_agent,
        "logs":       logs,
        "chars":      len(final_html)
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
.warn{{background:#fff8e1;border-left:4px solid #ffc107;
       border-radius:8px;padding:16px;color:#856404}}
</style>
</head>
<body>
<div class="card">
  <h1>{data['title']}</h1>
  <p>{data['meta_desc'] or 'No description available.'}</p>
  <div class="warn">
    ⚠️ All AI agents exhausted. Please wait a few minutes and try again,
    or check your API keys in the .env file.
  </div>
</div>
</body>
</html>"""