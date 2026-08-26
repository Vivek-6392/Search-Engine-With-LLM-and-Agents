import ast
import concurrent.futures
import json
import math
import operator
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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

KNOWN_FINANCIAL_ASSETS: Dict[str, str] = {
    "BITCOIN": "BTC-USD",
    "BTC": "BTC-USD",
    "ETHEREUM": "ETH-USD",
    "ETH": "ETH-USD",
    "SOLANA": "SOL-USD",
    "SOL": "SOL-USD",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "META": "META",
    "FACEBOOK": "META",
    "TESLA": "TSLA",
    "NETFLIX": "NFLX",
    "AMD": "AMD",
    "INTEL": "INTC",
    "TAIWAN SEMICONDUCTOR": "TSM",
    "TSMC": "TSM",
    "BERKSHIRE": "BRK-B",
    "BROADCOM": "AVGO",
    "QUALCOMM": "QCOM",
    "ORACLE": "ORCL",
    "ADOBE": "ADBE",
    "SALESFORCE": "CRM",
    "COINBASE": "COIN",
    "PALANTIR": "PLTR",
    "UBER": "UBER",
}

FINANCE_STOP_WORDS: Set[str] = {
    "WHAT", "IS", "THE", "CURRENT", "STOCK", "PRICE", "MARKET", "CAP", "AND",
    "OF", "FOR", "IN", "TO", "A", "AN", "ON", "AT", "BY", "WITH", "FROM",
    "COMPARE", "VERSUS", "VS", "TODAY", "LATEST", "SHARE", "VALUATION",
    "WORTH", "COST", "VALUE", "DOLLARS", "USD", "RATE", "HIGH", "LOW",
    "REAL", "TIME", "QUOTES", "DATA", "FINANCIAL", "INFO", "STATISTICS", "HOW", "MUCH"
}


def extract_finance_tickers(query: str) -> List[str]:
    """Extract one or more stock/crypto tickers from natural language questions or entity strings in order of appearance."""
    found: List[Tuple[int, str]] = []

    # 1. Look for ticker in parentheses, e.g. (AAPL)
    for m in re.finditer(r"\(([A-Za-z0-9\.\-\=]+)\)", query):
        cand = m.group(1).upper().replace("$", "").strip()
        if cand and cand not in FINANCE_STOP_WORDS:
            found.append((m.start(), KNOWN_FINANCIAL_ASSETS.get(cand, cand)))

    # 2. Look for $TICKER
    for m in re.finditer(r"\$([A-Za-z0-9\.\-]+)", query):
        cand = m.group(1).upper().strip()
        if cand and cand not in FINANCE_STOP_WORDS:
            found.append((m.start(), KNOWN_FINANCIAL_ASSETS.get(cand, cand)))

    # 3. Look for known company/crypto names
    q_upper = query.upper()
    for name, ticker in KNOWN_FINANCIAL_ASSETS.items():
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", q_upper):
            found.append((m.start(), ticker))

    # 4. Look for uppercase candidate words of 1-5 letters
    for m in re.finditer(r"\b[A-Za-z0-9\.\-]{1,6}\b", query):
        cand = m.group(0).upper().strip()
        if cand.isupper() and cand not in FINANCE_STOP_WORDS and len(cand) >= 2:
            if cand in KNOWN_FINANCIAL_ASSETS:
                found.append((m.start(), KNOWN_FINANCIAL_ASSETS[cand]))
            elif cand.isalpha() and len(cand) <= 5:
                found.append((m.start(), cand))

    # Sort by appearance position in query
    found.sort(key=lambda x: x[0])

    # Deduplicate preserving appearance order
    seen: Set[str] = set()
    deduped: List[str] = []
    for _, t in found:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def fetch_single_financial_quote(ticker_str: str) -> str:
    """Fetch live financial quote for a single ticker symbol using yfinance with fast_info fallback."""
    try:
        ticker = yf.Ticker(ticker_str)
        info = ticker.info
        if not info or ("currentPrice" not in info and "regularMarketPrice" not in info and "previousClose" not in info):
            try:
                fast = getattr(ticker, "fast_info", None)
                if fast and hasattr(fast, "last_price") and fast.last_price is not None:
                    p = fast.last_price
                    mc = getattr(fast, "market_cap", None)
                    mc_str = f"{mc / 1e9:.2f}B" if mc else "N/A"
                    curr = getattr(fast, "currency", "USD")
                    return f"Asset: {ticker_str}\nCurrent Price: {p:.2f} {curr}\nMarket Cap: {mc_str} {curr}"
            except Exception:
                pass
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


def search_finance(query: str) -> str:
    """Fetch live market data, stock prices, crypto prices, and company statistics for single or multiple entities."""
    tickers = extract_finance_tickers(query)
    if not tickers:
        fallback_cand = query.strip().upper().split()[-1].replace("$", "").strip("()?,.")
        if fallback_cand:
            tickers = [KNOWN_FINANCIAL_ASSETS.get(fallback_cand, fallback_cand)]
        else:
            return f"No ticker symbol or recognized asset found in query: '{query}'."

    quotes = [fetch_single_financial_quote(t) for t in tickers]
    return "\n\n---\n\n".join(quotes)


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

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_MATH_FUNCS = {
    "sqrt": math.sqrt,
    "pow": math.pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
}

_SAFE_MATH_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast_node(node):
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    # For Python < 3.8 backwards compatibility
    if sys.version_info < (3, 8) and isinstance(node, getattr(ast, "Num", ())):
        return node.n

    if isinstance(node, ast.Name):
        if node.id in _SAFE_MATH_CONSTANTS:
            return _SAFE_MATH_CONSTANTS[node.id]
        if node.id in _SAFE_MATH_FUNCS:
            return _SAFE_MATH_FUNCS[node.id]
        raise ValueError(f"Undefined or unauthorized name: '{node.id}'")

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand_val = _eval_ast_node(node.operand)
        return _UNARY_OPS[op_type](operand_val)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left_val = _eval_ast_node(node.left)
        right_val = _eval_ast_node(node.right)

        if op_type is ast.Pow:
            # Prevent excessive exponentiation that could freeze execution or overflow
            if isinstance(right_val, (int, float)):
                if right_val > 10000 or right_val < -10000:
                    raise ValueError("Exponent exceeds safe limits (|exp| <= 10000)")
                if isinstance(left_val, (int, float)) and abs(left_val) > 1000 and right_val > 100:
                    raise ValueError("Base and exponent combination exceeds safe limits")
        return _BIN_OPS[op_type](left_val, right_val)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Direct function calls only (attribute access is prohibited)")
        func_name = node.func.id
        if func_name not in _SAFE_MATH_FUNCS:
            raise ValueError(f"Unauthorized function call: '{func_name}'")
        func = _SAFE_MATH_FUNCS[func_name]
        args = [_eval_ast_node(arg) for arg in node.args]
        return func(*args)

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_ast_node(elt) for elt in node.elts]

    raise ValueError(f"Syntax not permitted in safe calculator: {type(node).__name__}")


def normalize_math_expression(expression: str) -> str:
    """Pre-process natural language math phrases into clean, valid Python mathematical expressions."""
    text = expression.strip().replace("`", "")
    if not text:
        return ""

    # Strip conversational prefixes and articles
    text = re.sub(
        r"^(what\s+is|what's|calculate|compute|evaluate|find|solve\s+for|solve|tell\s+me|show\s+me)\s+(the\s+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^the\s+", "", text, flags=re.IGNORECASE)
    text = text.rstrip("?. \t\n")

    # Convert natural language operations to Python syntax
    replacements = [
        (r"\bsquare\s+root\s+of\s+([0-9\.\(\)]+)", r"sqrt(\1)"),
        (r"\bsqrt\s+of\s+([0-9\.\(\)]+)", r"sqrt(\1)"),
        (r"\bcube\s+root\s+of\s+([0-9\.\(\)]+)", r"(\1)**(1/3)"),
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\bmultiplied\s+by\b", "*"),
        (r"\btimes\b", "*"),
        (r"\bdivided\s+by\b", "/"),
        (r"\bover\b", "/"),
        (r"\bmodulo\b|\bmod\b", "%"),
        (r"\bto\s+the\s+power\s+of\b", "**"),
        (r"\braised\s+to\b", "**"),
        (r"\^", "**"),
        (r"\bx\b", "*"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Handle "square root of (...)" or "square root 144"
    text = re.sub(r"\bsquare\s+root\s*\(([^)]+)\)", r"sqrt(\1)", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsquare\s+root\s+([a-zA-Z0-9_\.]+)", r"sqrt(\1)", text, flags=re.IGNORECASE)

    return text.strip()


def evaluate_math_expression(expression: str, timeout_seconds: float = 2.0):
    """Parse and evaluate a math expression using safe AST traversal with timeout."""
    clean_expr = normalize_math_expression(expression)
    if not clean_expr:
        raise ValueError("Empty mathematical expression")

    tree = ast.parse(clean_expr, mode="eval")

    def _run():
        return _eval_ast_node(tree)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        return future.result(timeout=timeout_seconds)


def python_calculator(expression: str) -> str:
    """Evaluate mathematical expressions, formulas, and statistical computations safely."""
    try:
        result = evaluate_math_expression(expression, timeout_seconds=2.0)
        return f"Calculation Result: {result}"
    except concurrent.futures.TimeoutError:
        return "Calculation Error: Evaluation timed out"
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
# 8b. arXiv Preprints Tool
# --------------------------------------------------

def search_arxiv(query: str) -> str:
    """Search arXiv for physics, mathematics, and computer science preprint papers."""
    try:
        clean_q = urllib.parse.quote(query.strip().replace('"', ''))
        url = f"https://export.arxiv.org/api/query?search_query=all:{clean_q}&start=0&max_results=3"
        req = urllib.request.Request(url, headers={"User-Agent": "SearchEngine-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
            root = ET.fromstring(content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            if not entries:
                return search_academic_papers(query)

            papers = []
            for entry in entries[:3]:
                title_elem = entry.find("atom:title", ns)
                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else "Untitled"
                id_elem = entry.find("atom:id", ns)
                paper_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
                summary_elem = entry.find("atom:summary", ns)
                summary = summary_elem.text.strip().replace("\n", " ")[:200] + "..." if summary_elem is not None and summary_elem.text else ""
                published_elem = entry.find("atom:published", ns)
                pub_year = published_elem.text[:4] if published_elem is not None and published_elem.text else ""
                papers.append(f"• [{title}]({paper_url}) ({pub_year})\n  {summary}")

            if papers:
                return "arXiv Preprints:\n\n" + "\n\n".join(papers)
    except Exception:
        pass
    return search_academic_papers(query)


# --------------------------------------------------
# 9. Real-Time Weather Tool
# --------------------------------------------------

def search_weather(location: str) -> str:
    """Fetch current meteorological conditions and forecast for a given city or location."""
    cleaned = location.strip()
    cleaned = re.sub(r"(?i)\b(what|is|the|weather|forecast|temperature|in|for|today|current|at|how|like|conditions)\b", " ", cleaned)
    cleaned = re.sub(r"[^\w\s\-]", "", cleaned).strip()
    target_loc = cleaned if cleaned else location.strip()
    clean_loc = urllib.parse.quote(target_loc)
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
Weather for {target_loc.title()}:
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
    KNOWN_CURRENCIES = {"USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF", "CNY", "HKD", "SGD"}
    tokens = re.findall(r"\b[A-Za-z]{3}\b", query.upper())
    base = "USD"
    for tok in tokens:
        if tok in KNOWN_CURRENCIES:
            base = tok
            break
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


# --------------------------------------------------
# 12. Wikipedia Encyclopedia Tool
# --------------------------------------------------

def search_wikipedia(query: str) -> str:
    """Search Wikipedia with automatic DuckDuckGo fallback for maximum resilience."""
    clean_query = query.strip()

    # 1. Try DuckDuckGo site search first (bypasses Wikipedia API rate limits)
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(f"site:wikipedia.org {clean_query}", max_results=3))
            if results:
                snippets = [f"• [{r.get('title', 'Wikipedia')}]({r.get('href', '')}):\n  {r.get('body', '')}" for r in results]
                return "Wikipedia Knowledge:\n\n" + "\n\n".join(snippets)
    except Exception:
        pass

    # 2. Try MediaWiki API
    try:
        enc_query = urllib.parse.quote(clean_query)
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={enc_query}&utf8=&format=json&srlimit=3"
        req = urllib.request.Request(
            search_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("query", {}).get("search", [])
            if items:
                snippets = []
                for item in items[:3]:
                    title = item.get("title", "")
                    raw_snippet = item.get("snippet", "")
                    clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet)
                    page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
                    snippets.append(f"• [{title}]({page_url}):\n  {clean_snippet}")
                return "Wikipedia Search Results:\n\n" + "\n\n".join(snippets)
    except Exception:
        pass

    return f"No Wikipedia reference found for '{query}'."
