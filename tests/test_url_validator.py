import unittest.mock as mock
from browser.browser_tool import is_safe_url, browse_webpage


def test_valid_schemes():
    assert is_safe_url("https://example.com") is True
    assert is_safe_url("http://example.com/articles/123") is True


def test_prohibited_schemes():
    assert is_safe_url("file:///etc/passwd") is False
    assert is_safe_url("ftp://example.com/file.txt") is False
    assert is_safe_url("javascript:alert(1)") is False
    assert is_safe_url("data:text/html,hello") is False
    assert is_safe_url("") is False
    assert is_safe_url(None) is False


def test_loopback_and_local_ip_rejection():
    assert is_safe_url("http://127.0.0.1:8000") is False
    assert is_safe_url("http://127.0.0.1/admin") is False
    assert is_safe_url("http://localhost:5000") is False
    assert is_safe_url("http://0.0.0.0:80") is False


def test_private_and_link_local_ip_rejection():
    # Private RFC 1918
    assert is_safe_url("http://10.0.0.1/status") is False
    assert is_safe_url("http://192.168.1.1/router") is False
    assert is_safe_url("http://172.16.0.1/dashboard") is False

    # Cloud metadata endpoint (169.254.169.254)
    assert is_safe_url("http://169.254.169.254/latest/meta-data") is False


def test_browse_webpage_fallback_on_unsafe_url():
    with mock.patch("browser.browser_tool._search_fallback_for_url") as mock_fallback:
        mock_fallback.return_value = "Fallback content from search"
        result = browse_webpage("file:///etc/passwd")
        mock_fallback.assert_called_once_with("file:///etc/passwd")
        assert result == "Fallback content from search"


def test_browse_webpage_fallback_on_loopback():
    with mock.patch("browser.browser_tool._search_fallback_for_url") as mock_fallback:
        mock_fallback.return_value = "Fallback content from search"
        result = browse_webpage("http://127.0.0.1:8080")
        mock_fallback.assert_called_once_with("http://127.0.0.1:8080")
        assert result == "Fallback content from search"
