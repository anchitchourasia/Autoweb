import os
import re
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def _clean_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def generate_clone_html(data: dict) -> str:
    prompt = f"""
You are an expert frontend developer specializing in pixel-accurate UI recreation.

Using the website data below, generate a COMPLETE, BEAUTIFUL, RESPONSIVE single HTML file
with all CSS embedded inside a <style> tag.

--- WEBSITE DATA ---
URL: {data['url']}
TITLE: {data['title']}
META DESCRIPTION: {data['meta_desc']}
COLOR PALETTE: {json.dumps(data['colors'])}
HEADINGS: {json.dumps(data['headings'], ensure_ascii=False)}
NAV LINKS: {json.dumps(data['nav_links'], ensure_ascii=False)}
CTA BUTTONS: {json.dumps(data['buttons'], ensure_ascii=False)}
VISIBLE TEXT: {data['text'][:5000]}

--- REQUIREMENTS ---
1. Return ONLY raw HTML. No markdown, no explanation, no ```html fences.
2. Use the color palette for authenticity.
3. Structure: sticky navbar → hero → features → about/stats → CTA banner → footer.
4. Use CSS Grid or Flexbox. Fully mobile responsive with media queries.
5. Typography: clean, modern. Google Fonts via @import allowed.
6. Images: use https://placehold.co/600x400?text=Image as placeholder.
7. Buttons: styled, rounded, with hover effects.
8. Footer with site title and nav links.
9. Overall quality must impress a hiring manager.
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4,
                max_output_tokens=8192
            )
        )
        return _clean_response(response.text)
    except Exception as e:
        return _fallback_html(data, str(e))


def _fallback_html(data: dict, error: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{data['title']} – Clone</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #f4f6f8; padding: 40px; }}
  .card {{ max-width: 860px; margin: auto; background: #fff; border-radius: 20px;
           padding: 40px; box-shadow: 0 10px 40px rgba(0,0,0,.08); }}
  h1 {{ font-size: 2.4rem; }}
  .error {{ background:#fff0f0; border:1px solid #fcc; border-radius:10px;
            padding:14px; color:#c00; }}
</style>
</head>
<body>
<div class="card">
  <h1>{data['title']}</h1>
  <p>{data['meta_desc']}</p>
  <div class="error">⚠️ Gemini error: {error}</div>
</div>
</body>
</html>"""