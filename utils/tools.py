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
    """Search GitHub for top repositories, stars, and code descriptions."""
    clean_query = query.strip().replace('"', '')
    url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(clean_query)}&sort=stars&order=desc&per_page=3"

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
            for item in items[:3]:
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
    """Fetch live market data, stock prices, crypto prices, and company statistics."""
    ticker_candidate = query.strip().upper().split()[-1].replace("$", "")
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
        if not info or ("currentPrice" not in info and "regularMarketPrice" not in info and "previousClose" not in info):
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
    """Search PubMed for clinical trials, medical discoveries, and biomedical literature."""
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
                    f"• [{title}]({pubmed_url})\n  Journal: {source} ({pubdate})"
                )
            return "PubMed Biomedical Papers:\n\n" + "\n\n".join(papers)
    except Exception as e:
        return f"PubMed search error: {str(e)}"


# --------------------------------------------------
# 4. Safe Python Calculator / Math Tool
# --------------------------------------------------

def python_calculator(expression: str) -> str:
    """Evaluate mathematical expressions, formulas, and statistical computations."""
    clean_expr = expression.strip().replace("`", "")
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


# --------------------------------------------------
# 5. Hugging Face Hub Tool
# --------------------------------------------------

def search_huggingface(query: str) -> str:
    """Search Hugging Face Hub for top open-source AI models, weights, and datasets."""
    clean_query = urllib.parse.quote(query.strip())
    url = f"https://huggingface.co/api/models?search={clean_query}&limit=3&sort=downloads&direction=-1"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SearchEngine-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            models = json.loads(resp.read().decode())
            if not models:
                return f"No Hugging Face models found matching '{query}'."

            results = []
            for m in models[:3]:
                m_id = m.get("id", "")
                downloads = m.get("downloads", 0)
                likes = m.get("likes", 0)
                pipeline = m.get("pipeline_tag", "General")
                hf_url = f"https://huggingface.co/{m_id}"
                results.append(
                    f"• [{m_id}]({hf_url}) (Task: {pipeline} | Downloads: {downloads:,} | Likes: {likes:,})"
                )
            return "Hugging Face Top Models:\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Hugging Face lookup error: {str(e)}"


# --------------------------------------------------
# 6. Hacker News Search Tool
# --------------------------------------------------

def search_hackernews(query: str) -> str:
    """Search Y Combinator Hacker News for tech debates, discussions, and story comments."""
    clean_query = urllib.parse.quote(query.strip())
    url = f"https://hn.algolia.com/api/v1/search?query={clean_query}&tags=story&hitsPerPage=3"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SearchEngine-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            hits = data.get("hits", [])
            if not hits:
                return f"No Hacker News stories found for '{query}'."

            stories = []
            for h in hits[:3]:
                title = h.get("title", "Untitled")
                points = h.get("points", 0)
                comments = h.get("num_comments", 0)
                story_id = h.get("objectID")
                hn_url = f"https://news.ycombinator.com/item?id={story_id}"
                link = h.get("url") or hn_url
                stories.append(
                    f"• [{title}]({link}) (Points: {points} | Comments: {comments})\n  HN Thread: {hn_url}"
                )
            return "Hacker News Discussions:\n\n" + "\n\n".join(stories)
    except Exception as e:
        return f"Hacker News search error: {str(e)}"


# --------------------------------------------------
# 7. Stack Overflow Search Tool
# --------------------------------------------------

def search_stackoverflow(query: str) -> str:
    """Search Stack Overflow for programming errors, bug fixes, and accepted code solutions."""
    clean_query = urllib.parse.quote(query.strip())
    url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q={clean_query}&site=stackoverflow&pagesize=3"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SearchEngine-Agent/1.0", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            items = data.get("items", [])
            if not items:
                return f"No Stack Overflow questions found for '{query}'."

            questions = []
            for item in items[:3]:
                title = item.get("title", "Untitled")
                link = item.get("link", "")
                is_answered = item.get("is_answered", False)
                score = item.get("score", 0)
                tags = ", ".join(item.get("tags", [])[:3])
                status_str = "Solved" if is_answered else "Open"
                questions.append(
                    f"• [{title}]({link}) ({status_str} | Score: {score} | Tags: {tags})"
                )
            return "Stack Overflow Solutions:\n\n" + "\n\n".join(questions)
    except Exception as e:
        return f"Stack Overflow search error: {str(e)}"


# --------------------------------------------------
# 8. OpenAlex Academic Literature Tool (250M+ Papers)
# --------------------------------------------------

def search_academic_papers(query: str) -> str:
    """Search OpenAlex global academic repository for peer-reviewed research papers, citations, and open access literature."""
    clean_query = urllib.parse.quote(query.strip())
    url = f"https://api.openalex.org/works?search={clean_query}&per-page=3"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SearchEngine-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            results = data.get("results", [])
            if not results:
                return f"No academic papers found for query: '{query}'."

            papers = []
            for w in results[:3]:
                title = w.get("title", "Untitled")
                year = w.get("publication_year", "N/A")
                cited_by = w.get("cited_by_count", 0)
                oa_url = w.get("open_access", {}).get("oa_url") or w.get("doi") or f"https://openalex.org/{w.get('id')}"
                topic = w.get("primary_topic", {}).get("display_name", "Academic Research")
                papers.append(
                    f"• [{title}]({oa_url}) (Year: {year} | Cited by: {cited_by:,} | Topic: {topic})"
                )
            return "Academic Research Papers (OpenAlex):\n\n" + "\n\n".join(papers)
    except Exception as e:
        return f"Academic search error: {str(e)}"


# --------------------------------------------------
# 9. Real-Time Weather Tool
# --------------------------------------------------

def search_weather(location: str) -> str:
    """Fetch current meteorological conditions and forecast for a given city or location."""
    clean_loc = urllib.parse.quote(location.strip().replace("weather", "").replace("in", "").strip())
    url = f"https://wttr.in/{clean_loc}?format=j1"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            current = data.get("current_condition", [{}])[0]
            temp_c = current.get("temp_C", "N/A")
            temp_f = current.get("temp_F", "N/A")
            desc = current.get("weatherDesc", [{}])[0].get("value", "Clear")
            humidity = current.get("humidity", "N/A")
            wind = current.get("windspeedKmph", "N/A")

            return f"""
Weather for {location.strip().title()}:
Condition: {desc}
Temperature: {temp_c}°C ({temp_f}°F)
Humidity: {humidity}%
Wind Speed: {wind} km/h
""".strip()
    except Exception as e:
        return f"Weather lookup error: {str(e)}"


# --------------------------------------------------
# 10. PyPI & NPM Package Registry Tool
# --------------------------------------------------

def lookup_package(package_name: str) -> str:
    """Lookup latest versions, descriptions, and licenses for Python (PyPI) and Node (NPM) packages."""
    pkg = package_name.strip().lower().replace("package", "").strip()
    pypi_url = f"https://pypi.org/pypi/{pkg}/json"

    try:
        req = urllib.request.Request(pypi_url, headers={"User-Agent": "SearchEngine-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            info = data.get("info", {})
            version = info.get("version", "N/A")
            summary = info.get("summary", "No summary provided.")
            license_str = info.get("license", "Unknown")
            home_page = info.get("project_url") or info.get("home_page") or f"https://pypi.org/project/{pkg}/"

            return f"""
Package: {pkg} (PyPI)
Latest Version: {version}
License: {license_str}
URL: {home_page}
Summary: {summary}
""".strip()
    except Exception:
        try:
            npm_url = f"https://registry.npmjs.org/{pkg}/latest"
            req_npm = urllib.request.Request(npm_url, headers={"User-Agent": "SearchEngine-Agent/1.0"})
            with urllib.request.urlopen(req_npm, timeout=5) as resp_npm:
                npm_data = json.loads(resp_npm.read().decode())
                ver = npm_data.get("version", "N/A")
                desc = npm_data.get("description", "No description.")
                lic = npm_data.get("license", "Unknown")
                return f"""
Package: {pkg} (NPM)
Latest Version: {ver}
License: {lic}
URL: https://www.npmjs.com/package/{pkg}
Description: {desc}
""".strip()
        except Exception as npm_err:
            return f"Package '{pkg}' not found on PyPI or NPM ({str(npm_err)})."


# --------------------------------------------------
# 11. Currency & Forex Converter Tool
# --------------------------------------------------

def convert_forex(query: str) -> str:
    """Fetch real-time foreign exchange rates. Input example: 'USD', 'EUR', 'INR', 'JPY', 'GBP'."""
    clean = query.strip().upper().split()[-1]
    base = clean if len(clean) == 3 else "USD"
    url = f"https://open.er-api.com/v6/latest/{base}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SearchEngine-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            rates = data.get("rates", {})
            if not rates:
                return f"Exchange rates not available for {base}."

            targets = ["USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF", "CNY"]
            rate_lines = [f"• 1 {base} = {rates[t]:.4f} {t}" for t in targets if t in rates and t != base]

            return f"Live Foreign Exchange Rates (Base: {base}):\n" + "\n".join(rate_lines)
    except Exception as e:
        return f"Forex rate lookup error: {str(e)}"
