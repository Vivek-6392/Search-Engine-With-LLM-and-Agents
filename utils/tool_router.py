import re
from typing import Any, Dict, List, Optional, Set


# Canonical tool group definitions
TOOL_GROUPS = {
    "WEB": ["web_search_tool", "web_browser_tool"],
    "ACADEMIC": ["arxiv_tool", "academic_papers_tool", "pubmed_tool"],
    "CODE": ["github_search_tool", "stackoverflow_tool", "package_lookup_tool", "huggingface_tool"],
    "GENERAL": ["wikipedia_tool", "web_search_tool"],
    "REAL_TIME": ["finance_tool", "forex_tool", "weather_tool"],
    "UTILITY": ["calculator_tool"],
    "NEWS": ["hackernews_tool", "web_search_tool"],
}

# Explicit single tool mapping
EXPLICIT_TOOL_MAPPING = {
    "calculator": ["calculator_tool"],
    "calculator_tool": ["calculator_tool"],
    "math": ["calculator_tool"],
    "finance": ["finance_tool"],
    "finance_tool": ["finance_tool"],
    "stock": ["finance_tool"],
    "stocks": ["finance_tool"],
    "crypto": ["finance_tool"],
    "forex": ["forex_tool"],
    "forex_tool": ["forex_tool"],
    "weather": ["weather_tool"],
    "weather_tool": ["weather_tool"],
    "arxiv": ["arxiv_tool"],
    "arxiv_tool": ["arxiv_tool"],
    "pubmed": ["pubmed_tool"],
    "pubmed_tool": ["pubmed_tool"],
    "academic_papers": ["academic_papers_tool"],
    "academic_papers_tool": ["academic_papers_tool"],
    "github": ["github_search_tool"],
    "github_search": ["github_search_tool"],
    "github_search_tool": ["github_search_tool"],
    "stackoverflow": ["stackoverflow_tool"],
    "stackoverflow_tool": ["stackoverflow_tool"],
    "huggingface": ["huggingface_tool"],
    "huggingface_tool": ["huggingface_tool"],
    "package": ["package_lookup_tool"],
    "package_lookup": ["package_lookup_tool"],
    "package_lookup_tool": ["package_lookup_tool"],
    "wikipedia": ["wikipedia_tool"],
    "wikipedia_tool": ["wikipedia_tool"],
    "hackernews": ["hackernews_tool"],
    "hackernews_tool": ["hackernews_tool"],
    "browser": ["web_browser_tool", "web_search_tool"],
    "web_browser": ["web_browser_tool", "web_search_tool"],
    "web_browser_tool": ["web_browser_tool", "web_search_tool"],
    "web_search": ["web_search_tool"],
    "web_search_tool": ["web_search_tool"],
}

# Master Allowlist of valid tool names
MASTER_TOOL_ALLOWLIST = {
    "web_search_tool",
    "wikipedia_tool",
    "arxiv_tool",
    "academic_papers_tool",
    "pubmed_tool",
    "github_search_tool",
    "huggingface_tool",
    "stackoverflow_tool",
    "hackernews_tool",
    "package_lookup_tool",
    "finance_tool",
    "forex_tool",
    "weather_tool",
    "calculator_tool",
    "web_browser_tool",
}


def _get_tool_name(tool_obj: Any) -> str:
    """Extract tool identifier name from a LangChain BaseTool, callable, or string."""
    if hasattr(tool_obj, "name"):
        return str(tool_obj.name)
    if hasattr(tool_obj, "__name__"):
        return str(tool_obj.__name__)
    return str(tool_obj)


def classify_query_tool_group(query: str) -> str:
    """
    Classify a research query into the most relevant ToolGroup.
    """
    q = query.lower()

    # 1. Utility (Math / Arithmetic / Formula / Calculation)
    math_patterns = [
        "calculate", "calculator", "math", "sqrt", "square root", "cube root",
        "equation", "arithmetic", "formula", "multiplied by", "divided by",
        "plus", "minus", "times", "percentage of", "sum of", "product of",
        "evaluate", "solve for", "power of", "modulo", "squared", "cubed",
    ]
    if any(k in q for k in math_patterns) or re.search(r"\b\d+\s*[\+\-\*\/\^]\s*\d+\b", q):
        return "UTILITY"

    # 2. Real-time (Finance / Crypto / Forex / Weather)
    if any(k in q for k in ["stock", "market cap", "ticker", "nasdaq", "nyse", "share price", "crypto", "bitcoin", "ethereum", "btc", "eth", "forex", "exchange rate", "currency conversion", "weather", "temperature", "forecast", "humidity"]):
        return "REAL_TIME"

    # 3. Code & Developer tools
    if any(k in q for k in ["github", "repository", "repo", "stackoverflow", "stack overflow", "error traceback", "exception fix", "pypi", "npm package", "pip install", "huggingface", "transformer model", "lora weights"]):
        return "CODE"

    # 4. Academic & Scientific Literature
    if any(k in q for k in ["arxiv", "pubmed", "clinical trial", "biomedical", "scientific paper", "literature review", "peer-reviewed", "preprint", "journal article", "theorem proof", "drug efficacy"]):
        return "ACADEMIC"

    # 5. News & Tech Discussions
    if any(k in q for k in ["hackernews", "hacker news", "tech news", "y combinator", "trending on hn", "breaking news"]):
        return "NEWS"

    # 6. Web & Browsing (Direct URLs)
    if any(k in q for k in ["http://", "https://", "browse", "scrape", "visit website", "extract from url", "full webpage"]):
        return "WEB"

    # 7. General Knowledge
    return "GENERAL"


def select_tools_for_task(
    task_type: Optional[str] = None,
    query: str = "",
    tool_name: Optional[str] = None,
    all_tools: Optional[List[Any]] = None,
) -> List[Any]:
    """
    Select the minimum sufficient, task-specific subset of tools (1-3 tools).
    Never returns the full 15-tool suite to any worker node.

    Args:
        task_type: Task category or group (e.g. "tool", "llm", "ACADEMIC", "CODE", "UTILITY").
        query: The node's specific query / task string.
        tool_name: Explicit planner-selected tool name if provided.
        all_tools: List of available LangChain BaseTool objects or functions.

    Returns:
        List of selected tool objects (or string names if all_tools is None), validated against allowlist.
    """
    selected_names: List[str] = []

    # 1. Explicit Planner-Selected Tool Override
    if tool_name:
        clean_name = str(tool_name).strip().lower()
        if clean_name in EXPLICIT_TOOL_MAPPING:
            selected_names = list(EXPLICIT_TOOL_MAPPING[clean_name])
        elif f"{clean_name}_tool" in MASTER_TOOL_ALLOWLIST:
            selected_names = [f"{clean_name}_tool"]

    # 2. Task Group Override via task_type
    if not selected_names and task_type:
        group_key = str(task_type).upper().strip()
        if group_key in TOOL_GROUPS:
            selected_names = list(TOOL_GROUPS[group_key])

    # 3. Query Intent Classification
    if not selected_names:
        group_key = classify_query_tool_group(query)
        selected_names = list(TOOL_GROUPS.get(group_key, TOOL_GROUPS["GENERAL"]))

    # 4. Allowlist Validation & Deduplication
    valid_names = []
    seen = set()
    for name in selected_names:
        if name in MASTER_TOOL_ALLOWLIST and name not in seen:
            valid_names.append(name)
            seen.add(name)

    if not valid_names:
        valid_names = ["web_search_tool"]

    # Cap to at most 3 tools per node (strictly enforce smallest relevant tool set)
    valid_names = valid_names[:3]

    # 5. Map back to tool objects if provided
    if all_tools:
        tool_dict = {_get_tool_name(t): t for t in all_tools}
        matched_tools = [tool_dict[name] for name in valid_names if name in tool_dict]
        if matched_tools:
            return matched_tools
        # Fallback to whatever first matching tool exists
        return [all_tools[0]]

    return valid_names


def measure_tool_context_reduction(
    all_tools: List[Any],
    routed_tools: List[Any],
) -> Dict[str, Any]:
    """
    Measure prompt character and estimated token reduction between the full tool suite
    and the routed subset.
    """
    def _tool_repr_len(tool_list: List[Any]) -> int:
        total = 0
        for t in tool_list:
            name = _get_tool_name(t)
            desc = getattr(t, "description", "")
            args_schema = str(getattr(t, "args", {}))
            total += len(name) + len(desc) + len(args_schema) + 40  # formatting overhead
        return total

    full_chars = _tool_repr_len(all_tools)
    routed_chars = _tool_repr_len(routed_tools)

    full_tokens_est = max(1, full_chars // 4)
    routed_tokens_est = max(1, routed_chars // 4)

    saved_tokens = max(0, full_tokens_est - routed_tokens_est)
    pct_reduction = round((saved_tokens / full_tokens_est) * 100, 2) if full_tokens_est else 0.0

    return {
        "full_tool_count": len(all_tools),
        "routed_tool_count": len(routed_tools),
        "full_chars": full_chars,
        "routed_chars": routed_chars,
        "full_tokens_est": full_tokens_est,
        "routed_tokens_est": routed_tokens_est,
        "saved_tokens": saved_tokens,
        "reduction_percentage": pct_reduction,
    }
