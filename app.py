import json
import os
import urllib.request
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.tools import Tool

from langchain_community.tools import (
    WikipediaQueryRun,
    ArxivQueryRun,
)

from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
)

from groq import Groq
from langchain_groq import ChatGroq

from tavily import TavilyClient
from ddgs import DDGS

from dag import (
    DAGPlanner,
    DAGExecutor,
    ChatOllama,
    get_ollama_models,
    render_dag_graph,
    render_synthesis_skeleton,
)

from browser import browse_webpage


# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv(override=True)


def get_secret(key: str, default: str = "") -> str:
    """Safely fetch a secret from Streamlit secrets or environment variables."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            val = str(st.secrets[key]).strip()
            if val:
                return val
    except Exception:
        pass
    val = os.getenv(key, default)
    return str(val).strip() if val else default


@st.cache_data(ttl=3600, show_spinner=False)
def get_groq_models(api_key: str) -> list[str]:
    """Fetch active chat models dynamically from Groq API using Groq SDK."""
    fallback_models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "allam-2-7b",
    ]
    if not api_key:
        return fallback_models

    try:
        client = Groq(api_key=api_key)
        model_list = client.models.list()
        exclude_keywords = [
            "whisper",
            "embed",
            "tts",
            "guard",
            "orpheus",
            "compound",  # compound models do not support langchain tool calling
        ]
        active_models = [
            m.id
            for m in model_list.data
            if m.active and not any(k in m.id.lower() for k in exclude_keywords)
        ]

        if active_models:
            def model_priority(m_id: str) -> int:
                m_lower = m_id.lower()
                if "20b" in m_lower and "safeguard" not in m_lower:
                    return 0  # #1 Default: openai/gpt-oss-20b
                if "120b" in m_lower:
                    return 1
                if "qwen" in m_lower:
                    return 2
                if "allam" in m_lower:
                    return 3
                return 4

            sorted_models = sorted(active_models, key=lambda x: (model_priority(x), x))
            return sorted_models
    except Exception:
        pass

    return fallback_models


# --------------------------------------------------
# Page Configuration & Notion-Style Clean CSS
# --------------------------------------------------

st.set_page_config(
    page_title="LLM Research Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Notion-like Callout Box */
    .notion-callout {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 12px 16px;
        border-radius: 8px;
        background: rgba(140, 140, 140, 0.06);
        border: 1px solid rgba(140, 140, 140, 0.16);
        margin-bottom: 16px;
    }
    .notion-callout-icon {
        font-size: 1.15rem;
        line-height: 1.2;
    }
    .notion-callout-text {
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Notion Card */
    .notion-card {
        border-radius: 8px;
        padding: 14px 16px;
        background: rgba(140, 140, 140, 0.04);
        border: 1px solid rgba(140, 140, 140, 0.16);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        margin-bottom: 10px;
    }

    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .badge-pending {
        background: rgba(140, 140, 140, 0.15);
        color: #888;
    }
    .badge-running {
        background: rgba(245, 158, 11, 0.15);
        color: #D97706;
    }
    .badge-completed {
        background: rgba(16, 185, 129, 0.15);
        color: #059669;
    }
    .badge-failed {
        background: rgba(239, 68, 68, 0.15);
        color: #DC2626;
    }

    /* Header */
    .page-header {
        padding-bottom: 14px;
        margin-bottom: 18px;
        border-bottom: 1px solid rgba(140, 140, 140, 0.15);
    }
    .page-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .page-subtitle {
        font-size: 0.9rem;
        color: #718096;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar Settings
# --------------------------------------------------

with st.sidebar:

    st.markdown("### ⚙️ Settings")

    provider = st.radio(
        "LLM Provider",
        ["Groq (Cloud)", "Ollama (Local)"],
        horizontal=True,
    )

    if provider == "Groq (Cloud)":

        groq_env = get_secret("GROQ_API_KEY")
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=groq_env,
            placeholder="gsk_...",
            help="Automatically loaded from .env if present.",
        )
        if groq_api_key.strip():
            st.caption("🟢 Groq Key loaded")
        else:
            st.caption("🔴 Groq Key required")

        available_models = get_groq_models(groq_api_key)

        model_name = st.selectbox(
            "Groq Model",
            available_models,
            index=0,
        )

    else:

        ollama_base_url = st.text_input(
            "Ollama Base URL",
            value=get_secret("OLLAMA_BASE_URL", "http://localhost:11434"),
            placeholder="http://localhost:11434",
        )

        available_models = get_ollama_models(ollama_base_url)

        model_name = st.selectbox(
            "Ollama Model",
            available_models,
            index=0,
        )

    tavily_env = get_secret("TAVILY_API_KEY")
    tavily_api_key = st.text_input(
        "Tavily API Key",
        type="password",
        value=tavily_env,
        placeholder="tvly-...",
        help="Automatically loaded from .env if present.",
    )
    if tavily_api_key.strip():
        st.caption("🟢 Tavily Key loaded")
    else:
        st.caption("🟡 No Tavily Key (will use DDGS fallback)")

    max_dag_workers = st.slider(
        "Parallel DAG Tasks",
        min_value=1,
        max_value=6,
        value=5,
    )

    st.divider()

    st.markdown(
        """
**DAG Execution Pipeline**
1. 🎯 **Planner**: Breaks query into directed sub-tasks.
2. ⚡ **Parallel Nodes**: Concurrently runs research agents.
3. 🌐 **Tools**: Tavily Search, DDGS fallback, Web Browser.
4. 📄 **Synthesis**: Merges findings into a cohesive report.
"""
    )


# --------------------------------------------------
# LLM Initialization
# --------------------------------------------------

if provider == "Groq (Cloud)":

    if not groq_api_key:
        st.info("💡 Enter your Groq API key in the sidebar to start.")
        st.stop()

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,
    )

else:

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_base_url,
        temperature=0,
    )


# --------------------------------------------------
# Tavily Search
# --------------------------------------------------

def tavily_search(query: str) -> str:
    """
    Primary web search using Tavily.
    """

    if not tavily_api_key:
        raise ValueError(
            "Tavily API key is not configured."
        )

    client = TavilyClient(
        api_key=tavily_api_key
    )

    response = client.search(
        query=query,
        search_depth="basic",
        max_results=5,
        include_answer=False,
    )

    results = response.get(
        "results",
        []
    )

    if not results:
        raise ValueError(
            "Tavily returned no results."
        )

    formatted_results = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        title = result.get(
            "title",
            "Untitled",
        )

        url = result.get(
            "url",
            "",
        )

        content = result.get(
            "content",
            "",
        )[:500]

        formatted_results.append(
            f"""
Result {index}

Title: {title}

URL: {url}

Content:
{content}
"""
        )

    return "\n".join(
        formatted_results
    )


# --------------------------------------------------
# DDGS Search
# --------------------------------------------------

def ddgs_search(query: str) -> str:
    """
    Fallback search using DDGS.
    """

    results = []

    with DDGS() as ddgs:

        search_results = ddgs.text(
            query,
            max_results=4,
        )

        for result in search_results:

            results.append(
                {
                    "title": result.get(
                        "title",
                        "Untitled",
                    ),
                    "url": result.get(
                        "href",
                        result.get(
                            "url",
                            "",
                        ),
                    ),
                    "content": result.get(
                        "body",
                        result.get(
                            "content",
                            "",
                        ),
                    )[:500],
                }
            )

    if not results:

        raise ValueError(
            "DDGS returned no results."
        )

    formatted_results = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        formatted_results.append(
            f"""
Result {index}

Title: {result["title"]}

URL: {result["url"]}

Content:
{result["content"]}
"""
        )

    return "\n".join(
        formatted_results
    )


# --------------------------------------------------
# Primary Search With DDGS Fallback
# --------------------------------------------------

def web_search(query: str) -> str:
    """
    Search strategy:

    1. Try Tavily.
    2. If Tavily fails, use DDGS.
    """

    try:

        result = tavily_search(
            query
        )

        return (
            "SEARCH PROVIDER: TAVILY\n\n"
            + result
        )

    except Exception as tavily_error:

        try:

            fallback_result = ddgs_search(
                query
            )

            return (
                "SEARCH PROVIDER: DDGS FALLBACK\n\n"
                f"Tavily error: {str(tavily_error)}\n\n"
                + fallback_result
            )

        except Exception as ddgs_error:

            return (
                "SEARCH FAILED\n\n"
                f"Tavily error: {str(tavily_error)}\n\n"
                f"DDGS error: {str(ddgs_error)}"
            )


from utils.tools import (
    search_github,
    search_finance,
    search_pubmed,
    python_calculator,
    search_huggingface,
    search_hackernews,
    search_stackoverflow,
    search_academic_papers,
    search_weather,
    lookup_package,
    convert_forex,
)

from langchain_core.tools import tool

# --------------------------------------------------
# Wikipedia & Arxiv APIs
# --------------------------------------------------

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper()
)


# --------------------------------------------------
# Full Specialized Multi-Domain Tool Suite
# --------------------------------------------------

@tool
def web_search_tool(query: str) -> str:
    """Search the live web for current information, live facts, news, and real-time updates."""
    return web_search(query)


@tool
def wikipedia_tool(query: str) -> str:
    """Search Wikipedia for encyclopedic, historical, biographical, and general background knowledge."""
    try:
        return str(wikipedia.run(query))
    except Exception as e:
        return f"Wikipedia search error: {str(e)}"


@tool
def arxiv_tool(query: str) -> str:
    """Search arXiv for physics, mathematics, and AI/ML academic preprint papers."""
    try:
        return str(arxiv.run(query))
    except Exception as e:
        return f"ArXiv search error: {str(e)}"


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


tools = [
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


# --------------------------------------------------
# Research Agent & DAG Components
# --------------------------------------------------

planner = DAGPlanner(
    llm=llm
)

executor = DAGExecutor(
    llm=llm,
    tools=tools,
    max_workers=max_dag_workers,
)


# --------------------------------------------------
# Notion Clean Slate Main Header & Controls
# --------------------------------------------------

st.markdown(
    """
    <div class="page-header">
        <h1 class="page-title">🔍 LLM Research Engine</h1>
        <div class="page-subtitle">Autonomous multi-agent research with directed acyclic graphs & live web synthesis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_mode, col_space = st.columns([2.5, 1])

with col_mode:
    search_mode = st.segmented_control(
        "Research Mode",
        ["⚡ Fast / Overview", "🔬 Deep Research", "🎓 Academic Literature"],
        default="🔬 Deep Research",
        label_visibility="collapsed",
    )
    # segmented_control is deselectable by default (returns None) unless
    # your Streamlit version supports required=True - fall back explicitly
    # so an accidental un-click doesn't silently break mode_key below.
    if search_mode is None:
        search_mode = "🔬 Deep Research"

mode_key = "deep"
if "Fast" in search_mode:
    mode_key = "fast"
elif "Academic" in search_mode:
    mode_key = "academic"

query = st.text_input(
    "Search Query",
    placeholder="Ask anything (e.g. Compare recent breakthrough advances in quantum computing)...",
    label_visibility="collapsed",
)

col_btn, col_info = st.columns([1.2, 5])
with col_btn:
    run_search = st.button("🚀 Start Research", type="primary", use_container_width=True)
with col_info:
    st.caption(f"Active Provider: **{provider}** | Model: `{model_name}` | Mode: `{search_mode.split('/')[0].strip()}`")


# --------------------------------------------------
# Research Execution Flow
# --------------------------------------------------

if run_search:

    if not query.strip():
        st.warning("Please enter a search query.")
        st.stop()

    # Layout into Dual-Column Canvas
    col_dag_canvas, col_report_canvas = st.columns([1, 1.35], gap="large")

    with col_dag_canvas:
        st.markdown("### 🧠 Research DAG Flow")
        progress_container = st.empty()
        dag_graph_container = st.empty()

    with col_report_canvas:
        report_status_container = st.empty()

    # 1. Create DAG
    with col_dag_canvas:
        with st.spinner("Planning research graph..."):
            dag = planner.create_dag(query, mode=mode_key)

    # 2. Render initial DAG graph + progress bar
    node_statuses = {node_id: "PENDING" for node_id in dag.nodes}

    def render_progress():
        total = len(node_statuses)
        done = sum(1 for s in node_statuses.values() if s in ("COMPLETED", "FAILED"))
        with progress_container:
            st.caption(f"{done} of {total} nodes complete")
            st.progress(done / total if total else 0.0)

    def render_dag():
        html_code = render_dag_graph(dag, node_statuses)
        if hasattr(dag_graph_container, "html"):
            dag_graph_container.html(html_code)
        else:
            dag_graph_container.markdown(html_code, unsafe_allow_html=True)

    render_progress()
    render_dag()

    # Show a shimmer skeleton in the synthesis panel while nodes run
    with report_status_container:
        skeleton_html = render_synthesis_skeleton()
        if hasattr(st, "html"):
            st.html(skeleton_html)
        else:
            st.markdown(skeleton_html, unsafe_allow_html=True)

    # Progress updater
    def update_progress(node, status):
        node_statuses[node.id] = status
        render_progress()
        render_dag()

    # 3. Execute DAG
    with st.spinner(f"Executing DAG with {max_dag_workers} parallel workers..."):
        dag = executor.execute(dag, progress_callback=update_progress)

    # 4. Collect results & Generate final synthesis
    results = dag.get_results()
    combined_results = "\n\n".join(
        [f"### {node_id}\n\n{result}" for node_id, result in results.items()]
    )

    # Dynamic length guidance based on research mode & query intent
    if mode_key == "fast":
        length_guideline = """
- **Concise & Direct**: Keep your answer crisp, clear, and to the point (1-3 focused paragraphs or succinct key points). Answer the user's specific question directly without unnecessary filler.
"""
    elif mode_key == "academic":
        length_guideline = """
- **Academic Depth**: Provide a detailed literature synthesis with research findings, methodology context, and paper citations.
"""
    else:
        length_guideline = """
- **Adaptive Length**: Match the depth to the question. For simple or definition questions (e.g. "what is X", "who won Y"), provide a clear, concise 2-3 paragraph answer. For complex or multi-dimensional questions, provide a thorough, structured breakdown.
"""

    final_prompt = f"""
You are an intelligent AI research assistant synthesizing findings for a user query.

User Query:
"{query}"

Research Mode: {search_mode}

Research Findings from DAG nodes:
{combined_results}

Formatting & Presentation Guidelines:
1. **Direct Answer First**: Begin immediately with the direct, definitive answer in the opening paragraph.
2. **Markdown Tables**: For comparisons or multi-entity data (prices, market caps, weather, versions), ALWAYS format them in clean GitHub-flavored Markdown tables with column headers and delimiter rows:
| Company | Ticker | Current Price (USD) | Market Cap (USD) |
| :--- | :--- | :--- | :--- |
| Apple | AAPL | $309.35 | $4,514.71 B |
| Microsoft | MSFT | $483.24 | $3,588.32 B |
3. **Clean Calculations**: Present calculations in clean, readable bullet points:
- **Total Market Cap**: $18,954.02 B
- **Average Market Cap**: $3,790.80 B
(Do NOT output raw addition formulas or duplicated equation terms).
4. {length_guideline.strip()}
5. **Adaptive Proportionality**: Make the response length natural and appropriate to the question asked.
6. **Natural Sources & Hyperlinks**: Seamlessly embed markdown links (e.g., [Yahoo Finance AAPL](https://finance.yahoo.com/quote/AAPL)) when citing sources or facts.
7. **Clean Bold Headers**: Use clean section headings (e.g. `### Current Stock Prices and Market Capitalization`) or bold text (e.g. `**Current Stock Prices and Market Capitalization**`) without leading or trailing spaces inside the asterisks.
8. **No Meta-Talk**: Never mention "DAG nodes", "internal tasks", or execution logs.
"""

    synth_system_prompt = (
        "You are an expert research synthesizer. All data gathering, web research, calculations, "
        "and fact extraction have already been completed by upstream agent workers.\n"
        "Your ONLY task is to write a final, comprehensive, well-structured Markdown answer strictly based on the provided findings.\n"
        "CRITICAL CONSTRAINTS:\n"
        "1. Do NOT call, invoke, or output any tool calls, function calls, or JSON actions (such as `browser.search`, `web_search`, etc.).\n"
        "2. Output ONLY direct, fluent Markdown text with clean Markdown tables (`| ... |`)."
    )

    with report_status_container:
        with st.spinner("Synthesizing final research report..."):
            try:
                final_response = llm.invoke(
                    [
                        SystemMessage(content=synth_system_prompt),
                        HumanMessage(content=final_prompt),
                    ]
                )
            except Exception as synth_err:
                err_str = str(synth_err)
                if any(x in err_str for x in ["tool_use_failed", "Tool choice is none", "failed_generation"]):
                    # Fallback to plain prompt format without tool triggers
                    fallback_text = (
                        f"Please write a comprehensive final summary answering this question: '{query}' based on these gathered findings:\n\n"
                        f"{combined_results}\n\n"
                        f"Do not call tools. Respond only in clean Markdown tables and text."
                    )
                    final_response = llm.invoke(fallback_text)
                else:
                    raise synth_err

    if hasattr(final_response, "content"):
        answer = str(final_response.content)
    else:
        answer = str(final_response)

    # 1. Clean Unicode spacing artifacts and non-breaking hyphens
    answer = answer.replace("\u2217", "*").replace("∗", "*")
    answer = answer.replace("\u202f", " ").replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    answer = answer.replace("\u2011", "-").replace("‑", "-")

    # 2. Clean garbled inline arithmetic equations from open-source tokenizers
    answer = re.sub(r"=\s*[*]+\s*[\d,\.]+\s*[A-Za-z]*\s*=\s*[*]+", " = **", answer)
    answer = re.sub(r"÷\s*\d+\s*=\s*[*]+[\d,\.]+\s*[A-Za-z]*\s*÷\s*\d+\s*=\s*[*]+", " ÷ 5 = **", answer)

    # 3. Auto-convert unpiped text tables into standard GitHub-flavored Markdown tables
    lines = [l.strip() for l in answer.split("\n")]
    formatted_lines = []
    idx = 0
    while idx < len(lines):
        # Detect table headers (e.g. Company / Ticker / Price / Market Cap)
        if idx + 3 < len(lines) and any(lines[idx].lower().startswith(h) for h in ["company", "asset", "entity", "city", "item", "library", "model"]) and any(lines[idx+1].lower().startswith(h) for h in ["ticker", "symbol", "code", "version", "temp", "metric", "task"]):
            headers = [lines[idx], lines[idx+1], lines[idx+2], lines[idx+3]]
            table_md = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join([":---"] * len(headers)) + " |",
            ]
            idx += 4
            while idx + 3 < len(lines) and lines[idx] and not lines[idx].startswith("**") and not lines[idx].startswith("#") and not lines[idx].startswith("Source") and not lines[idx].startswith("These"):
                row = [lines[idx], lines[idx+1], lines[idx+2], lines[idx+3]]
                table_md.append("| " + " | ".join(row) + " |")
                idx += 4
            formatted_lines.extend(table_md)
        else:
            formatted_lines.append(lines[idx])
            idx += 1

    answer = "\n".join(formatted_lines)

    # 4. Fix spaces inside bold markers e.g. ** $309.35 ** -> **$309.35**
    answer = re.sub(r"\*\*\s+([^\*\n]+?)\s+\*\*", r"**\1**", answer)
    answer = re.sub(r"\*\*\s+([^\*\n]+?)\*\*", r"**\1**", answer)
    answer = re.sub(r"\*\*([^\*\n]+?)\s+\*\*", r"**\1**", answer)
    # Fix glued bold asterisks e.g. trillion**and -> trillion** and
    answer = re.sub(r"(\w)\*\*", r"\1 **", answer)
    answer = re.sub(r"\*\*(\w)", r"** \1", answer)
    # Escape dollar signs preceding numbers so Streamlit KaTeX doesn't parse currency as LaTeX formulas
    answer = re.sub(r"(?<!\\)\$([0-9])", r"\\$\1", answer)

    # Clear status container and render final report
    report_status_container.empty()

    with col_report_canvas:
        st.markdown(answer)

    # Left Column: Add Expandable Detailed Inspection
    with col_dag_canvas:
        with st.expander("🔍 Inspect Node Findings", expanded=False):
            for node in dag.nodes.values():
                st.markdown(f"**🔵 {node.id}: {node.task}**")
                st.caption(f"Status: `{node.status}`")
                if node.result:
                    st.text_area(f"Output ({node.id})", value=node.result, height=140, disabled=True)
                if node.error:
                    st.error(f"Error: {node.error}")
                st.divider()