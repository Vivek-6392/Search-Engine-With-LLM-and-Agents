import asyncio
import urllib.request
from html.parser import HTMLParser
from playwright.async_api import async_playwright


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ["script", "style", "noscript", "svg", "head"]:
            self.skip = True

    def handle_endtag(self, tag):
        if tag.lower() in ["script", "style", "noscript", "svg", "head"]:
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            text = data.strip()
            if text:
                self.result.append(text)

    def get_text(self):
        return " ".join(self.result)


def _fast_http_fetch(url: str, max_chars: int = 4000) -> str:
    """Sub-second HTTP fetch avoiding heavy browser spin-up."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=3.5) as response:
            html = response.read().decode("utf-8", errors="ignore")
            parser = HTMLTextExtractor()
            parser.feed(html)
            text = parser.get_text()
            if len(text) > 80:
                return text[:max_chars]
    except Exception:
        pass
    return ""


async def _browse(url: str, max_chars: int = 4000) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(
            viewport={"width": 1280, "height": 720}
        )

        try:
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=6000,
            )

            content = await page.locator("body").inner_text()
            content = " ".join(content.split())
            return content[:max_chars]

        except Exception as e:
            return f"Browser error while opening {url}: {str(e)}"

        finally:
            await browser.close()


def browse_webpage(url: str) -> str:
    """
    Ultra-fast web browsing:
    1. Tries sub-second direct HTTP parse first (~200ms).
    2. Falls back to headless Chromium with strict 6s timeout only if needed.
    """
    # 1. Direct fast HTTP fetch
    fast_result = _fast_http_fetch(url)
    if fast_result:
        return fast_result

    # 2. Playwright fallback
    try:
        return asyncio.run(_browse(url))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_browse(url))
        finally:
            loop.close()