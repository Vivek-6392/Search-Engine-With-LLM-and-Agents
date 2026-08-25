import unittest.mock as mock
import pytest

from browser.browser_tool import (
    BrowserManager,
    BrowserURLCache,
    browse_webpage,
    is_safe_url,
    _fast_http_fetch,
)


def test_ssrf_protections():
    # Loopback addresses
    assert is_safe_url("http://127.0.0.1:8000") is False
    assert is_safe_url("http://localhost:3000") is False
    assert is_safe_url("http://[::1]/") is False

    # Private network ranges (RFC 1918)
    assert is_safe_url("http://192.168.1.1/admin") is False
    assert is_safe_url("http://10.0.0.1/") is False
    assert is_safe_url("http://172.16.0.1/") is False

    # AWS/GCP Cloud Metadata endpoint
    assert is_safe_url("http://169.254.169.254/latest/meta-data/") is False

    # Prohibited schemes
    assert is_safe_url("file:///etc/passwd") is False
    assert is_safe_url("gopher://example.com") is False
    assert is_safe_url("ftp://example.com") is False

    # Valid external public URLs
    assert is_safe_url("https://www.google.com") is True
    assert is_safe_url("https://arxiv.org/abs/2301.00000") is True
    assert is_safe_url("https://github.com") is True


def test_http_success_path_bypasses_playwright():
    mock_manager = mock.MagicMock(spec=BrowserManager)
    mock_cache = BrowserURLCache()

    # Mock successful HTTP GET with rich HTML
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = (
        "<html><body><h1>DeepSearchAI Engine Overview</h1>"
        "<p>DeepSearchAI is an autonomous multi-agent research assistant built with LangChain and directed acyclic graphs.</p>"
        "</body></html>"
    )

    with mock.patch("requests.get", return_value=mock_resp):
        content = browse_webpage(
            "https://example.com/deepsearch",
            manager=mock_manager,
            cache=mock_cache,
        )

        assert "DeepSearchAI Engine Overview" in content
        assert "autonomous multi-agent" in content
        # Playwright extraction must NOT have been called!
        assert mock_manager.extract_with_playwright.call_count == 0


def test_http_failure_fallback_to_playwright():
    mock_manager = mock.MagicMock(spec=BrowserManager)
    mock_manager.extract_with_playwright.return_value = "Extracted JS rendered dynamic text from React application."
    mock_cache = BrowserURLCache()

    # Mock HTTP failure (e.g. 403 Forbidden or JS wall)
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "<html><body>Please enable JavaScript to view this application.</body></html>"

    with mock.patch("requests.get", return_value=mock_resp):
        content = browse_webpage(
            "https://example.com/react-spa",
            manager=mock_manager,
            cache=mock_cache,
        )

        assert content == "Extracted JS rendered dynamic text from React application."
        assert mock_manager.extract_with_playwright.call_count == 1


def test_duplicate_url_caching():
    mock_manager = mock.MagicMock(spec=BrowserManager)
    mock_cache = BrowserURLCache(ttl_seconds=60.0)

    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><body><p>Article Body Content</p></body></html>"

    with mock.patch("requests.get", return_value=mock_resp) as mock_get:
        # First call
        res1 = browse_webpage("https://example.com/article", manager=mock_manager, cache=mock_cache)
        assert mock_get.call_count == 1

        # Second call to same URL in request
        res2 = browse_webpage("https://example.com/article", manager=mock_manager, cache=mock_cache)
        # requests.get should NOT be called again (cache hit!)
        assert mock_get.call_count == 1
        assert res1 == res2


def test_browser_instance_reuse():
    manager = BrowserManager(max_concurrent=2)
    mock_browser_obj = mock.MagicMock()
    mock_playwright_obj = mock.MagicMock()
    mock_playwright_obj.chromium.launch.return_value = mock_browser_obj

    with mock.patch("playwright.sync_api.sync_playwright") as mock_pw_factory:
        mock_pw_factory.return_value.start.return_value = mock_playwright_obj

        b1 = manager._ensure_browser()
        b2 = manager._ensure_browser()

        # Chromium launch must only occur once!
        assert b1 == mock_browser_obj
        assert b2 == mock_browser_obj
        assert mock_playwright_obj.chromium.launch.call_count == 1


def test_oversized_content_truncation():
    mock_manager = mock.MagicMock(spec=BrowserManager)
    mock_cache = BrowserURLCache()

    huge_text = "Word " * 2000  # 10,000 chars
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = f"<html><body><p>{huge_text}</p></body></html>"

    with mock.patch("requests.get", return_value=mock_resp):
        content = browse_webpage(
            "https://huge-page.example.com",
            max_chars=1200,
            manager=mock_manager,
            cache=mock_cache,
        )

        assert len(content) <= 1200


def test_concurrency_limit_semaphore():
    manager = BrowserManager(max_concurrent=2)
    # Available slots initially 2
    assert manager._semaphore._value == 2

    # Acquire 2 slots
    assert manager._semaphore.acquire(blocking=False) is True
    assert manager._semaphore.acquire(blocking=False) is True

    # 3rd slot must be blocked
    assert manager._semaphore.acquire(blocking=False) is False

    # Release slots
    manager._semaphore.release()
    manager._semaphore.release()
    assert manager._semaphore._value == 2
