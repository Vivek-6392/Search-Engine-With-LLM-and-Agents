import pytest
from unittest.mock import MagicMock, patch

from utils.tools import extract_finance_tickers, search_finance, search_weather, convert_forex


def test_extract_finance_tickers_multi_entity():
    q1 = "What is the current stock price and market cap of Apple (AAPL) and Microsoft (MSFT)?"
    assert extract_finance_tickers(q1) == ["AAPL", "MSFT"]

    q2 = "Apple (AAPL)"
    assert extract_finance_tickers(q2) == ["AAPL"]

    q3 = "Microsoft MSFT stock price and market cap"
    assert extract_finance_tickers(q3) == ["MSFT"]

    q4 = "Compare Tesla and Nvidia market cap"
    assert extract_finance_tickers(q4) == ["TSLA", "NVDA"]

    q5 = "$GOOGL and $AMZN"
    assert extract_finance_tickers(q5) == ["GOOGL", "AMZN"]

    q6 = "Bitcoin and Ethereum prices"
    assert extract_finance_tickers(q6) == ["BTC-USD", "ETH-USD"]


def test_search_finance_mocked():
    mock_info = {
        "shortName": "Apple Inc.",
        "currentPrice": 315.0,
        "currency": "USD",
        "dayHigh": 316.0,
        "dayLow": 309.0,
        "marketCap": 4500000000000,
        "longBusinessSummary": "Apple designs hardware and software.",
    }

    with patch("yfinance.Ticker") as mock_ticker_cls:
        instance = MagicMock()
        instance.info = mock_info
        mock_ticker_cls.return_value = instance

        res = search_finance("Apple (AAPL)")
        assert "Apple Inc. (AAPL)" in res
        assert "315.0 USD" in res
        assert "4500.00B USD" in res


def test_search_weather_cleaning():
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"current_condition": [{"temp_C": "22", "temp_F": "72", "weatherDesc": [{"value": "Sunny"}], "humidity": "45", "windspeedKmph": "10"}]}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        res = search_weather("What is the current weather in Tokyo?")
        assert "Tokyo" in res
        assert "22\u00b0C" in res
        assert "Sunny" in res


def test_convert_forex_cleaning():
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"rates": {"USD": 1.0, "EUR": 0.92, "GBP": 0.78, "INR": 83.5}}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        res = convert_forex("What is the forex exchange rate for EUR to USD?")
        assert "Base: EUR" in res
