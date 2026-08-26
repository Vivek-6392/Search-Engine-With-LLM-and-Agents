import unittest.mock as mock
import pytest

from utils.tool_router import (
    TOOL_GROUPS,
    MASTER_TOOL_ALLOWLIST,
    classify_query_tool_group,
    select_tools_for_task,
    measure_tool_context_reduction,
)


def _create_mock_tool(name: str, description: str = "A mock tool description with parameters and usage."):
    tool = mock.MagicMock()
    tool.name = name
    tool.description = description
    tool.args = {"query": {"title": "Query", "type": "string"}}
    return tool


ALL_15_MOCK_TOOLS = [
    _create_mock_tool("web_search_tool", "Search the live web for current information."),
    _create_mock_tool("wikipedia_tool", "Search Wikipedia for general background knowledge."),
    _create_mock_tool("arxiv_tool", "Search arXiv for physics, math, and AI papers."),
    _create_mock_tool("academic_papers_tool", "Search academic paper databases."),
    _create_mock_tool("pubmed_tool", "Search PubMed for biomedical and clinical studies."),
    _create_mock_tool("github_search_tool", "Search GitHub repositories and code."),
    _create_mock_tool("huggingface_tool", "Search HuggingFace for AI models and datasets."),
    _create_mock_tool("stackoverflow_tool", "Search StackOverflow for programming fixes."),
    _create_mock_tool("hackernews_tool", "Search Hacker News for tech news."),
    _create_mock_tool("package_lookup_tool", "Lookup Python/npm package releases."),
    _create_mock_tool("finance_tool", "Search stock prices, market cap, and crypto."),
    _create_mock_tool("forex_tool", "Lookup foreign exchange rates."),
    _create_mock_tool("weather_tool", "Search weather forecasts and temperature."),
    _create_mock_tool("calculator_tool", "Calculate mathematical formulas securely."),
    _create_mock_tool("web_browser_tool", "Browse and extract full webpage text."),
]


def test_classify_query_tool_groups():
    # Math & Utility
    assert classify_query_tool_group("Calculate the square root of 144") == "UTILITY"
    assert classify_query_tool_group("What is the square root of 144 plus 25 multiplied by 4?") == "UTILITY"
    assert classify_query_tool_group("Evaluate math expression 50 * 12") == "UTILITY"
    assert classify_query_tool_group("100 + 200 * 5") == "UTILITY"

    # Real-Time (Finance, Forex, Weather)
    assert classify_query_tool_group("What is Apple AAPL stock price and market cap?") == "REAL_TIME"
    assert classify_query_tool_group("Current weather and temperature in Tokyo") == "REAL_TIME"
    assert classify_query_tool_group("Convert USD to EUR exchange rate") == "REAL_TIME"

    # Code & Repos
    assert classify_query_tool_group("Search github for LangChain agent repositories") == "CODE"
    assert classify_query_tool_group("Stackoverflow fix for AttributeError in Python") == "CODE"

    # Academic & Science
    assert classify_query_tool_group("Find ArXiv papers on Diffusion Transformer architectures") == "ACADEMIC"
    assert classify_query_tool_group("PubMed clinical trials on mRNA vaccine efficacy") == "ACADEMIC"

    # News
    assert classify_query_tool_group("What is trending on Hacker News today?") == "NEWS"

    # Web / URLs
    assert classify_query_tool_group("Browse https://news.ycombinator.com and extract text") == "WEB"

    # General
    assert classify_query_tool_group("History of the Roman Empire and Julius Caesar") == "GENERAL"


def test_select_tools_for_task_representative_domains():
    # 1. Math query -> calculator only
    routed = select_tools_for_task(query="Calculate sqrt(256)")
    assert routed == ["calculator_tool"]

    # 2. Finance query -> real-time tool set
    routed = select_tools_for_task(query="Stock price and market cap for Nvidia NVDA")
    assert "finance_tool" in routed
    assert len(routed) <= 3

    # 3. Science paper query -> academic tool set
    routed = select_tools_for_task(query="ArXiv preprints on Quantum error correction")
    assert "arxiv_tool" in routed
    assert len(routed) <= 3

    # 4. Code query -> code tool set
    routed = select_tools_for_task(query="Search github repos for fast API starter kits")
    assert "github_search_tool" in routed
    assert len(routed) <= 3


def test_explicit_planner_selected_tools():
    # Explicit tool name override
    calc_tools = select_tools_for_task(tool_name="calculator", query="Any general text")
    assert calc_tools == ["calculator_tool"]

    arxiv_tools = select_tools_for_task(tool_name="arxiv", query="Any general text")
    assert arxiv_tools == ["arxiv_tool"]

    pubmed_tools = select_tools_for_task(tool_name="pubmed", query="Any general text")
    assert pubmed_tools == ["pubmed_tool"]


def test_select_tools_with_tool_objects():
    # Passing real or mock BaseTool objects
    routed_objs = select_tools_for_task(
        query="Calculate 99 * 45",
        all_tools=ALL_15_MOCK_TOOLS,
    )
    assert len(routed_objs) == 1
    assert routed_objs[0].name == "calculator_tool"

    routed_academic = select_tools_for_task(
        query="ArXiv papers on LLM reasoning",
        all_tools=ALL_15_MOCK_TOOLS,
    )
    assert len(routed_academic) <= 3
    names = [t.name for t in routed_academic]
    assert "arxiv_tool" in names


def test_never_returns_full_suite_and_enforces_allowlist():
    # Test random, noisy, or edge-case queries
    test_queries = [
        "Hello",
        "Explain photosynthesis",
        "What is the capital of France?",
        "12345",
        "random gibberish $%#@!",
    ]
    for q in test_queries:
        routed = select_tools_for_task(query=q, all_tools=ALL_15_MOCK_TOOLS)
        # MUST NEVER return all 15 tools
        assert 1 <= len(routed) <= 3
        # MUST all be in master allowlist
        for t in routed:
            assert t.name in MASTER_TOOL_ALLOWLIST


def test_measure_tool_context_reduction():
    # Measure reduction between full 15 tools and a 1-tool set (calculator)
    routed_calc = [t for t in ALL_15_MOCK_TOOLS if t.name == "calculator_tool"]
    reduction_metrics = measure_tool_context_reduction(
        all_tools=ALL_15_MOCK_TOOLS,
        routed_tools=routed_calc,
    )

    assert reduction_metrics["full_tool_count"] == 15
    assert reduction_metrics["routed_tool_count"] == 1
    # Token and char reduction must be > 70%
    assert reduction_metrics["reduction_percentage"] > 70.0
    assert reduction_metrics["saved_tokens"] > 0

    # Measure reduction for 3-tool academic set
    routed_academic = [t for t in ALL_15_MOCK_TOOLS if t.name in ("arxiv_tool", "academic_papers_tool", "pubmed_tool")]
    academic_reduction = measure_tool_context_reduction(
        all_tools=ALL_15_MOCK_TOOLS,
        routed_tools=routed_academic,
    )
    assert academic_reduction["routed_tool_count"] == 3
    assert academic_reduction["reduction_percentage"] > 60.0
