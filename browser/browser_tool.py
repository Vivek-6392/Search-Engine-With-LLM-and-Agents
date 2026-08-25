import asyncio
import atexit
from html.parser import HTMLParser
import ipaddress
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.parse
import requests


# --------------------------------------------------
# 1. SSRF & Security Validation
# --------------------------------------------------

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


# --------------------------------------------------
# 2. HTML Text Extractor
# --------------------------------------------------

class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ["script", "style", "noscript", "svg", "head", "nav", "footer", "aside"]:
            self.skip = True

    def handle_endtag(self, tag):
        if tag.lower() in ["script", "style", "noscript", "svg", "head", "nav", "footer", "aside"]:
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            text = data.strip()
            if text:
                self.result.append(text)

    def get_text(self):
        return " ".join(self.result)


# --------------------------------------------------
# 3. Request-Scoped URL Caching
# --------------------------------------------------

class BrowserURLCache:
    """Thread-safe TTL URL cache preventing duplicate browser or HTTP fetches."""

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 128):
        self._cache: Dict[str, Tuple[float, str]] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._max_size = max_size

    def _make_key(self, url: str, request_id: Optional[str] = None) -> str:
        clean = url.strip().lower()
        req = (request_id or "").strip()
        return f"{req}::{clean}" if req else clean

    def get(self, url: str, request_id: Optional[str] = None) -> Optional[str]:
        key = self._make_key(url, request_id)
        with self._lock:
            if key in self._cache:
                ts, val = self._cache[key]
                if (time.perf_counter() - ts) <= self._ttl:
                    return val
                del self._cache[key]
        return None

    def set(self, url: str, content: str, request_id: Optional[str] = None):
        key = self._make_key(url, request_id)
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_k = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest_k]
            self._cache[key] = (time.perf_counter(), content)

    def clear(self):
        with self._lock:
            self._cache.clear()


_GLOBAL_URL_CACHE = BrowserURLCache()


# --------------------------------------------------
# 4. Reusable BrowserManager
# --------------------------------------------------

class BrowserManager:
    """
    Thread-safe BrowserManager with:
    - Reusable Playwright lifecycle and Chromium browser instance
    - Concurrency throttle (max 2 parallel pages)
    - Navigation & total page timeouts
    - Resource blocking (skips images, media, stylesheets for speed)
    - Safe lifecycle shutdown
    """

    def __init__(
        self,
        max_concurrent: int = 2,
        nav_timeout_ms: int = 5000,
        page_timeout_ms: int = 8000,
    ):
        self.max_concurrent = max_concurrent
        self.nav_timeout_ms = nav_timeout_ms
        self.page_timeout_ms = page_timeout_ms
        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.RLock()
        self._playwright = None
        self._browser = None
        self._is_closed = False

    def _ensure_browser(self):
        """Lazily start reusable Playwright and Chromium instance if not running."""
        with self._lock:
            if self._is_closed:
                return None
            if self._browser is None:
                try:
                    from playwright.sync_api import sync_playwright
                    self._playwright = sync_playwright().start()
                    self._browser = self._playwright.chromium.launch(
                        headless=True,
                        args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
                    )
                except Exception:
                    self._browser = None
            return self._browser

    def extract_with_playwright(self, url: str, max_chars: int = 3500) -> str:
        """Extract dynamic JavaScript content via managed Chromium instance."""
        if not is_safe_url(url):
            return ""

        collector = None
        try:
            from metrics.collector import get_current_metrics
            collector = get_current_metrics()
        except Exception:
            pass

        acquired = self._semaphore.acquire(timeout=self.page_timeout_ms / 1000.0)
        if not acquired:
            if collector:
                collector.record_browser_call(fetch_type="playwright", success=False)
            return ""

        browser = self._ensure_browser()
        if not browser:
            self._semaphore.release()
            return ""

        page = None
        try:
            context = browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )

            # Block heavy assets to save CPU/bandwidth
            context.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in ["image", "media", "font", "stylesheet"]
                else route.continue_(),
            )

            page = context.new_page()
            page.set_default_navigation_timeout(self.nav_timeout_ms)
            page.set_default_timeout(self.page_timeout_ms)

            page.goto(url, wait_until="domcontentloaded", timeout=self.nav_timeout_ms)

            # Extract body text
            content = page.locator("body").inner_text(timeout=3000)
            content = " ".join(content.split())

            if collector:
                collector.record_browser_call(fetch_type="playwright", success=True)

            if len(content) > 80:
                return content[:max_chars]
            return ""

        except Exception:
            if collector:
                collector.record_browser_call(fetch_type="playwright", success=False)
            return ""
        finally:
            if page:
                try:
                    page.context.close()
                except Exception:
                    pass
            self._semaphore.release()

    def close(self):
        """Clean shutdown of browser and Playwright."""
        with self._lock:
            self._is_closed = True
            if self._browser:
                try:
                    self._browser.close()
                except Exception:
                    pass
                self._browser = None
            if self._playwright:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None


_GLOBAL_BROWSER_MANAGER = BrowserManager()


def get_browser_manager() -> BrowserManager:
    return _GLOBAL_BROWSER_MANAGER


# Register safe process cleanup
atexit.register(lambda: _GLOBAL_BROWSER_MANAGER.close())


# --------------------------------------------------
# 5. HTTP-First Fetch Strategy
# --------------------------------------------------

def _fast_http_fetch(url: str, max_chars: int = 3500) -> str:
    """Sub-second direct HTTP fetch with standard browser headers."""
    if not is_safe_url(url):
        return ""

    collector = None
    try:
        from metrics.collector import get_current_metrics
        collector = get_current_metrics()
    except Exception:
        pass

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=3.5, allow_redirects=True)
        if collector:
            collector.record_browser_call(fetch_type="http", success=(resp.status_code == 200))

        if resp.status_code == 200 and resp.text:
            parser = HTMLTextExtractor()
            parser.feed(resp.text)
            text = parser.get_text()

            # Check if page is dynamic JS-only wall
            js_wall_indicators = [
                "javascript is required",
                "enable javascript to continue",
                "please turn on javascript",
                "you need to enable javascript to run this app",
            ]
            text_lower = text.lower()
            if any(ind in text_lower for ind in js_wall_indicators):
                return ""  # Trigger Playwright fallback

            if len(text) > 100:
                return " ".join(text.split())[:max_chars]
    except Exception:
        if collector:
            collector.record_browser_call(fetch_type="http", success=False)
    return ""


def _search_fallback_for_url(url: str) -> str:
    """Fallback to search engine snippet extraction if website is blocked or times out."""
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


# --------------------------------------------------
# 6. Main browse_webpage Pipeline
# --------------------------------------------------

def browse_webpage(
    url: str,
    request_id: Optional[str] = None,
    max_chars: int = 3500,
    manager: Optional[BrowserManager] = None,
    cache: Optional[BrowserURLCache] = None,
) -> str:
    """
    Multi-stage resilient webpage extraction:
    1. SSRF Safety check against private/loopback/cloud metadata IP ranges.
    2. Request-scoped URL cache hit (0 latency).
    3. QueryBudget check.
    4. HTTP-First fetch (~150ms).
    5. Playwright fallback only for dynamic JS-heavy pages or HTTP failure.
    6. Search snippet fallback if both fail.
    7. Aggressive truncation to max_chars.
    """
    if not is_safe_url(url):
        return _search_fallback_for_url(url)

    c = cache or _GLOBAL_URL_CACHE
    cached_content = c.get(url, request_id=request_id)
    if cached_content:
        return cached_content

    bg = None
    try:
        from budget import get_current_budget
        bg = get_current_budget()
    except Exception:
        pass

    if bg:
        if not bg.reserve("browser"):
            return _search_fallback_for_url(url)
        bg.record_usage(browser_calls=1)

    # 1. Fast HTTP First
    content = _fast_http_fetch(url, max_chars=max_chars)
    if content:
        c.set(url, content, request_id=request_id)
        return content

    # 2. Playwright Fallback
    bm = manager or _GLOBAL_BROWSER_MANAGER
    pw_content = bm.extract_with_playwright(url, max_chars=max_chars)
    if pw_content:
        c.set(url, pw_content, request_id=request_id)
        return pw_content

    # 3. Resilient Search Fallback
    fallback = _search_fallback_for_url(url)
    c.set(url, fallback, request_id=request_id)
    return fallback