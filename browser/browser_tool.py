import asyncio
import ipaddress
import socket
import urllib.parse
from html.parser import HTMLParser
import requests


def is_safe_url(url: str) -> bool:
    """
    Validate that the URL has an allowed scheme (http/https) and does not
    resolve to private, loopback, link-local, or reserved IP ranges (SSRF protection).
    """
    if not isinstance(url, str) or not url.strip():
        return False

    try:
        parsed = urllib.parse.urlparse(url.strip())
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Resolve hostname to check IP addresses
        addr_info = socket.getaddrinfo(hostname, None)
        if not addr_info:
            return False

        for item in addr_info:
            ip_str = item[4][0]
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return False

        return True
    except Exception:
        return False


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
    """Sub-second HTTP fetch with standard browser headers."""
    if not is_safe_url(url):
        return ""

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=4, allow_redirects=True)
        if resp.status_code == 200 and resp.text:
            parser = HTMLTextExtractor()
            parser.feed(resp.text)
            text = parser.get_text()
            if len(text) > 80:
                return text[:max_chars]
    except Exception:
        pass
    return ""


async def _browse_playwright(url: str, max_chars: int = 4000) -> str:
    """Headless Chromium browser extraction with Playwright."""
    if not is_safe_url(url):
        return ""

    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=6000)
                content = await page.locator("body").inner_text()
                content = " ".join(content.split())
                if len(content) > 80:
                    return content[:max_chars]
            finally:
                await browser.close()
    except Exception:
        pass
    return ""


def _search_fallback_for_url(url: str) -> str:
    """Fallback to search engine snippet extraction if target website is blocked/timing out."""
    clean_target = url.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
    queries = [url, f"{clean_target} information"]

    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for q in queries:
                results = list(ddgs.text(q, max_results=3))
                if results:
                    snippets = [f"• {r.get('title')}: {r.get('body')}" for r in results]
                    return f"Web content for {url}:\n\n" + "\n\n".join(snippets)
    except Exception:
        pass
    return f"Unable to retrieve content from {url}."


def browse_webpage(url: str) -> str:
    """
    Multi-stage resilient webpage extraction:
    1. Validate URL to prevent SSRF against private/loopback/cloud metadata ranges.
    2. Sub-second direct HTTP parse (~200ms).
    3. Headless Chromium fallback (Playwright).
    4. Intelligent search snippet fallback if the domain is unsafe, times out, or blocks requests.
    """
    if not is_safe_url(url):
        return _search_fallback_for_url(url)

    # 1. Fast HTTP
    fast_result = _fast_http_fetch(url)
    if fast_result:
        return fast_result

    # 2. Playwright fallback
    try:
        pw_result = asyncio.run(_browse_playwright(url))
        if pw_result and "browser error" not in pw_result.lower():
            return pw_result
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            pw_result = loop.run_until_complete(_browse_playwright(url))
            if pw_result and "browser error" not in pw_result.lower():
                return pw_result
        finally:
            loop.close()
    except Exception:
        pass

    # 3. Resilient search fallback
    return _search_fallback_for_url(url)