from playwright.sync_api import sync_playwright


def capture_screenshot(url: str, path: str = "screenshot.png") -> tuple:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1440, "height": 900},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(2000)
            page.screenshot(path=path, full_page=True)
            browser.close()
        return path, None
    except Exception as e:
        return None, str(e)