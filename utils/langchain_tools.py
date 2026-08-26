from typing import List
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

from browser import browse_webpage
from utils.search_pipeline import search_and_extract_evidence
from utils.tools import (
    search_github,
    search_finance,
    search_pubmed,
    python_calculator,
    search_huggingface,
    search_hackernews,
    search_stackoverflow,
    search_academic_papers,
    search_arxiv,
    search_weather,
    lookup_package,
    convert_forex,
    search_wikipedia,
)

_wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
_arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())


def web_search(query: str) -> str:
    """Optimized web search with deduplication, ranking, and compact evidence extraction."""
    evidence_list = search_and_extract_evidence(query=query, max_results=3)
    if not evidence_list:
        return f"No verified web results found for query: '{query.strip()}'"
    return "\n\n".join([ev.to_markdown() for ev in evidence_list])


@tool
def web_search_tool(query: str) -> str:
    """Search the live web for current information, live facts, news, and real-time updates."""
    return web_search(query)


@tool
def wikipedia_tool(query: str) -> str:
    """Search Wikipedia for encyclopedic, historical, scientific, and general background knowledge."""
    try:
        res = search_wikipedia(query)
        if res and not res.startswith("No Wikipedia reference found"):
            return res
        return str(_wikipedia.run(query))
    except Exception:
        return search_wikipedia(query)


@tool
def arxiv_tool(query: str) -> str:
    """Search arXiv for physics, mathematics, and AI/ML academic preprint papers."""
    try:
        res = str(_arxiv.run(query))
        if "No good Arxiv Result was found" in res or "error" in res.lower():
            return search_arxiv(query)
        return res
    except Exception:
        return search_arxiv(query)


@tool
def academic_papers_tool(query: str) -> str:
    """Search OpenAlex global academic repository (250M+ papers) across science, medicine, engineering, and humanities."""
    return search_academic_papers(query)


@tool
def pubmed_tool(query: str) -> str:
    """Search PubMed for clinical trials, biomedical discoveries, healthcare papers, and medical treatments."""
    return search_pubmed(query)


@tool
def github_search_tool(query: str) -> str:
    """Search GitHub for top repositories, code libraries, star counts, descriptions, and open-source tools."""
    return search_github(query)


@tool
def huggingface_tool(query: str) -> str:
    """Search Hugging Face Hub for top open-source AI models, LLM weights (GGUF, safetensors), and datasets."""
    return search_huggingface(query)


@tool
def stackoverflow_tool(query: str) -> str:
    """Search Stack Overflow for programming errors, bug fixes, and accepted code solutions."""
    return search_stackoverflow(query)


@tool
def hackernews_tool(query: str) -> str:
    """Search Y Combinator Hacker News for tech debates, startup discussions, and community opinions."""
    return search_hackernews(query)


@tool
def package_lookup_tool(package_name: str) -> str:
    """Lookup package details, latest versions, and licenses for Python (PyPI) and Node.js (NPM) libraries."""
    return lookup_package(package_name)


@tool
def finance_tool(query: str) -> str:
    """Fetch real-time stock prices, crypto prices, market caps, currency valuations, and financial summaries."""
    return search_finance(query)


@tool
def forex_tool(query: str) -> str:
    """Fetch live foreign exchange rates between fiat currencies (e.g. USD, EUR, INR, GBP, JPY)."""
    return convert_forex(query)


@tool
def weather_tool(location: str) -> str:
    """Fetch current real-time weather conditions, temperature, humidity, and forecasts for any city."""
    return search_weather(location)


@tool
def calculator_tool(expression: str) -> str:
    """Evaluate exact mathematical calculations, formula computations, and unit conversions (e.g., '2+2', 'sqrt(144) * 12')."""
    return python_calculator(expression)


@tool
def web_browser_tool(url: str) -> str:
    """Open a webpage URL and extract its visible text when search results do not contain enough detail."""
    return browse_webpage(url)


ALL_RESEARCH_TOOLS = [
    web_search_tool,
    wikipedia_tool,
    arxiv_tool,
    academic_papers_tool,
    pubmed_tool,
    github_search_tool,
    huggingface_tool,
    stackoverflow_tool,
    hackernews_tool,
    package_lookup_tool,
    finance_tool,
    forex_tool,
    weather_tool,
    calculator_tool,
    web_browser_tool,
]
