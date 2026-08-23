import json
import os
import urllib.request

import streamlit as st
from dotenv import load_dotenv

from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
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
    """Fetch active chat models dynamically from Groq API."""
    fallback_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-safeguard-20b",
    ]
    if not api_key:
        return fallback_models

    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Streamlit-SearchEngine",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = [
                m["id"]
                for m in data.get("data", [])
                if m.get("active", True)
                and not any(x in m["id"] for x in ["whisper", "embed", "tts"])
            ]
            if models:
                # Prefer 120b at the top if available
                models = sorted(
                    models,
                    key=lambda x: (0 if "120b" in x else 1, x),
                )
                return models
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
        max_value=4,
        value=3,
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
)

from langchain_core.tools import tool

# --------------------------------------------------
# Wikipedia Tool
# --------------------------------------------------

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)


# --------------------------------------------------
# Arxiv Tool
# --------------------------------------------------

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper()
)


# --------------------------------------------------
# 8-Tool Specialized Suite with Explicit Typed Schemas
# --------------------------------------------------

@tool
def web_search_tool(query: str) -> str:
    """Search the web for current information, live facts, scores, weather, and breaking news."""
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
    """Search arXiv for scientific papers, AI/ML research, physics, mathematics, and academic computer science."""
    try:
        return str(arxiv.run(query))
    except Exception as e:
        return f"ArXiv search error: {str(e)}"


@tool
def github_search_tool(query: str) -> str:
    """Search GitHub for top repositories, code libraries, star counts, descriptions, and open-source tools (e.g. vLLM, PyTorch, LangChain)."""
    return search_github(query)


@tool
def finance_tool(query: str) -> str:
    """Fetch real-time stock prices, crypto prices, market caps, currency valuations, and financial summaries. Input should be a ticker symbol (NVDA, AAPL, BTC-USD, TSLA) or company name."""
    return search_finance(query)


@tool
def pubmed_tool(query: str) -> str:
    """Search PubMed for clinical trials, biomedical discoveries, healthcare papers, and medical treatments."""
    return search_pubmed(query)


@tool
def calculator_tool(expression: str) -> str:
    """Evaluate exact mathematical calculations, formula computations, and unit conversions. Examples: '2+2', 'sqrt(144) * 12', '1500 * (1+0.08)**5'."""
    return python_calculator(expression)


@tool
def web_browser_tool(url: str) -> str:
    """Open a webpage URL and extract its visible text when search results do not contain enough detail."""
    return browse_webpage(url)


tools = [
    web_search_tool,
    wikipedia_tool,
    arxiv_tool,
    github_search_tool,
    finance_tool,
    pubmed_tool,
    calculator_tool,
    web_browser_tool,
]


# --------------------------------------------------
# Research Agent
# --------------------------------------------------

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert AI research agent equipped with 8 specialized tools: "
            "(web_search_tool, wikipedia_tool, arxiv_tool, github_search_tool, finance_tool, pubmed_tool, calculator_tool, web_browser_tool). "
            "Select the most appropriate tool for the task: "
            "- For arithmetic, math formulas, equations, or numbers (e.g. 2+2): use `calculator_tool`. "
            "- For open-source code/repos: use `github_search_tool`. "
            "- For stock/crypto prices: use `finance_tool`. "
            "- For medical/health topics: use `pubmed_tool`. "
            "- For academic papers: use `arxiv_tool`. "
            "- For general facts/news/weather: use `web_search_tool`. "
            "- For encyclopedic overviews: use `wikipedia_tool`. "
            "Always include concrete facts, numbers, and source URLs when available.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent_runnable = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt,
)

agent = AgentExecutor(
    agent=agent_runnable,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=2,
)


# --------------------------------------------------
# DAG Components
# --------------------------------------------------

planner = DAGPlanner(
    llm=llm
)

executor = DAGExecutor(
    agent=agent,
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
        st.markdown("### 📄 Synthesis Canvas")
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

Guidelines:
1. **Direct Answer First**: Begin immediately with the direct, definitive answer in the opening paragraph.
2. {length_guideline.strip()}
3. **Adaptive Proportionality**: Do NOT write an overly long mega-report for short, simple, or definition questions. Make the response length natural and appropriate to the question asked.
4. **Natural Sources & Hyperlinks**: Seamlessly embed markdown links (e.g., [vLLM GitHub](https://github.com/vllm-project/vllm)) when citing sources or facts.
5. **No Meta-Talk**: Never mention "DAG nodes", "internal tasks", or execution logs.
"""

    with report_status_container:
        with st.spinner("Synthesizing final research report..."):
            final_response = llm.invoke(final_prompt)

    if hasattr(final_response, "content"):
        answer = final_response.content
    else:
        answer = str(final_response)

    # Clear status container and render final report
    report_status_container.empty()

    with col_report_canvas:
        # Notion-style query callout
        st.markdown(
            f"""
            <div class="notion-callout">
                <div class="notion-callout-icon">📌</div>
                <div class="notion-callout-text">
                    <strong>Research Query:</strong> {query}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(answer)

        st.divider()

        # 1-Click Download Report Button
        report_export_text = f"""# Research Report: {query}

**Mode:** {search_mode}
**Model:** {model_name} ({provider})

---

{answer}

---

## Appendix: DAG Node Outputs

{combined_results}
"""
        st.download_button(
            label="📥 Download Report (.md)",
            data=report_export_text,
            file_name=f"research_report_{query[:20].strip().replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

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