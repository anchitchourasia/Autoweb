import requests
from bs4 import BeautifulSoup
import re


def normalize_text(text: str, max_chars: int = 8000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def extract_page_data(url: str):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    error = None
    html = ""

    try:
        r = requests.get(url, headers=headers, timeout=25)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        error = str(e)

    soup = BeautifulSoup(html or "<html></html>", "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    meta_desc = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        meta_desc = meta["content"].strip()

    headings = []
    for tag in ["h1", "h2", "h3"]:
        for el in soup.find_all(tag):
            t = el.get_text(" ", strip=True)
            if t:
                headings.append({"tag": tag, "text": t})
    headings = headings[:25]

    nav_links = []
    for a in soup.find_all("a", href=True):
        txt = a.get_text(" ", strip=True)
        if txt:
            nav_links.append({"text": txt[:80], "href": a["href"]})
        if len(nav_links) >= 25:
            break

    buttons = []
    for btn in soup.find_all(["button", "a"]):
        cls = " ".join(btn.get("class", []))
        if any(k in cls.lower() for k in ["btn", "button", "cta"]):
            t = btn.get_text(" ", strip=True)
            if t:
                buttons.append(t[:60])
        if len(buttons) >= 10:
            break

    colors = []
    for s in soup.find_all("style"):
        found = re.findall(r"#[0-9a-fA-F]{3,6}|rgb\([^)]+\)", s.string or "")
        colors.extend(found)
    colors = list(set(colors))[:12]

    visible_text = normalize_text(soup.get_text(" ", strip=True))

    return {
        "url": url,
        "title": title or "Untitled Page",
        "meta_desc": meta_desc,
        "headings": headings,
        "nav_links": nav_links,
        "buttons": buttons,
        "colors": colors,
        "text": visible_text,
        "html_length": len(html)
    }, error