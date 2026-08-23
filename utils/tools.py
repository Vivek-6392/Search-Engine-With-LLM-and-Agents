import json
import math
import re
import urllib.parse
import urllib.request
import yfinance as yf


# --------------------------------------------------
# 1. GitHub Search Tool
# --------------------------------------------------

def search_github(query: str) -> str:
    """
    Search GitHub for top repositories, code libraries, stars, and descriptions.
    """
    clean_query = query.strip().replace('"', '')
    url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(clean_query)}&sort=stars&order=desc&per_page=4"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "SearchEngine-Agent/1.0",
                "Accept": "application/vnd.github.v3+json",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            items = data.get("items", [])
            if not items:
                return f"No GitHub repositories found matching '{query}'."

            results = []
            for item in items[:4]:
                name = item.get("full_name", "")
                desc = item.get("description") or "No description provided."
                stars = item.get("stargazers_count", 0)
                lang = item.get("language") or "General"
                repo_url = item.get("html_url", "")
                results.append(
                    f"• [{name}]({repo_url}) (Stars: {stars:,} | {lang})\n  {desc}"
                )
            return "GitHub Top Repositories:\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"GitHub search error: {str(e)}"


# --------------------------------------------------
# 2. Yahoo Finance / Crypto Tool
# --------------------------------------------------

def search_finance(query: str) -> str:
    """
    Fetch live market data, stock prices, crypto prices, and company statistics.
    Input should be a ticker symbol (e.g., NVDA, AAPL, BTC-USD, TSLA, MSFT, ETH-USD) or company name.
    """
    # Extract candidate ticker
    ticker_candidate = query.strip().upper().split()[-1].replace("$", "")
    # Check common company mappings
    mapping = {
        "BITCOIN": "BTC-USD",
        "BTC": "BTC-USD",
        "ETHEREUM": "ETH-USD",
        "ETH": "ETH-USD",
        "APPLE": "AAPL",
        "NVIDIA": "NVDA",
        "MICROSOFT": "MSFT",
        "TESLA": "TSLA",
        "GOOGLE": "GOOGL",
        "ALPHABET": "GOOGL",
        "AMAZON": "AMZN",
        "META": "META",
    }
    ticker_str = mapping.get(ticker_candidate, ticker_candidate)

    try:
        ticker = yf.Ticker(ticker_str)
        info = ticker.info
        if not info or "currentPrice" not in info and "regularMarketPrice" not in info and "previousClose" not in info:
            # Try searching ticker
            return f"No financial quote found for ticker/asset '{ticker_str}'."

        name = info.get("shortName") or info.get("longName") or ticker_str
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        currency = info.get("currency", "USD")
        day_high = info.get("dayHigh")
        day_low = info.get("dayLow")
        market_cap = info.get("marketCap")
        cap_str = f"{market_cap / 1e9:.2f}B" if market_cap else "N/A"
        summary = info.get("longBusinessSummary", "")
        summary_short = summary[:300] + "..." if len(summary) > 300 else summary

        return f"""
Asset: {name} ({ticker_str})
Current Price: {price} {currency}
Day Range: {day_low} - {day_high} {currency}
Market Cap: {cap_str} {currency}
Overview: {summary_short}
""".strip()
    except Exception as e:
        return f"Finance lookup error for '{ticker_str}': {str(e)}"


# --------------------------------------------------
# 3. PubMed Biomedical Research Tool
# --------------------------------------------------

def search_pubmed(query: str) -> str:
    """
    Search PubMed for clinical trials, medical discoveries, and biomedical literature.
    """
    clean_query = urllib.parse.quote(query.strip())
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={clean_query}&retmode=json&retmax=3"

    try:
        req = urllib.request.Request(
            search_url,
            headers={"User-Agent": "SearchEngine-Agent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return f"No PubMed medical papers found for query: '{query}'."

        ids_str = ",".join(id_list)
        summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids_str}&retmode=json"
        
        req_sum = urllib.request.Request(
            summary_url,
            headers={"User-Agent": "SearchEngine-Agent/1.0"}
        )
        with urllib.request.urlopen(req_sum, timeout=5) as resp_sum:
            sum_data = json.loads(resp_sum.read().decode())
            results_dict = sum_data.get("result", {})

            papers = []
            for p_id in id_list:
                doc = results_dict.get(p_id, {})
                title = doc.get("title", "Untitled")
                source = doc.get("source", "PubMed")
                pubdate = doc.get("pubdate", "")
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{p_id}/"
                papers.append(
                    f"• [{title}]({pubmed_url})\n  *Journal:* {source} ({pubdate})"
                )
            return "PubMed Biomedical Papers:\n\n" + "\n\n".join(papers)
    except Exception as e:
        return f"PubMed search error: {str(e)}"


# --------------------------------------------------
# 4. Safe Python Calculator / Math Tool
# --------------------------------------------------

def python_calculator(expression: str) -> str:
    """
    Evaluate mathematical expressions, formulas, unit conversions, and statistical computations.
    Examples: '2 ** 32', 'math.sqrt(144) * 12', '1500 * (1 + 0.08)**5'
    """
    clean_expr = expression.strip().replace("`", "")
    # Allow safe built-ins and math functions
    safe_dict = {
        "math": math,
        "sqrt": math.sqrt,
        "pow": math.pow,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }

    try:
        result = eval(clean_expr, {"__builtins__": {}}, safe_dict)
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Calculation Error: {str(e)}"
