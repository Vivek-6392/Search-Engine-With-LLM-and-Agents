import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
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
from langchain_core.tools import Tool, tool
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
    build_agent_executor,
    ChatOllama,
    get_ollama_models,
    render_dag_graph,
    render_synthesis_skeleton,
)
from browser import browse_webpage, is_safe_url
import chat_store
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
    search_wikipedia,
)


# --------------------------------------------------
# Load Environment Variables & Initialize Database
# --------------------------------------------------

load_dotenv(override=True)
chat_store.init_db()


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
            "compound",
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
                    return 0
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

    /* Suggestion cards for empty state */
    .suggestion-card {
        background: rgba(140, 140, 140, 0.05);
        border: 1px solid rgba(140, 140, 140, 0.15);
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------

if "current_chat_id" not in st.session_state:
    all_chats = chat_store.list_chats()
    if all_chats:
        st.session_state.current_chat_id = all_chats[0]["id"]
    else:
        st.session_state.current_chat_id = None


# --------------------------------------------------
# Sidebar Settings & Multi-Chat Sessions
# --------------------------------------------------

with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_chat_id = None
        st.rerun()

    st.markdown("### 💬 Chat History")
    all_chats = chat_store.list_chats()

    if not all_chats:
        st.caption("No previous chats yet. Start a research session!")
    else:
        today_date = datetime.now(timezone.utc).date()
        today_chats = []
        earlier_chats = []

        for c in all_chats:
            try:
                c_date = datetime.fromisoformat(c["updated_at"]).date()
            except Exception:
                c_date = today_date
            if c_date == today_date:
                today_chats.append(c)
            else:
                earlier_chats.append(c)

        def render_chat_item(c):
            is_active = (c["id"] == st.session_state.get("current_chat_id"))
            title_display = c["title"]
            if len(title_display) > 24:
                title_display = title_display[:22] + "..."

            col_select, col_opt = st.columns([3.8, 1.2])
            with col_select:
                btn_label = f"🟢 {title_display}" if is_active else title_display
                if st.button(btn_label, key=f"chat_btn_{c['id']}", use_container_width=True, help=c["title"]):
                    st.session_state.current_chat_id = c["id"]
                    st.rerun()

            with col_opt:
                with st.popover("⚙️", help="Chat options"):
                    st.markdown(f"**Manage:** *{c['title']}*")
                    new_title_input = st.text_input("Rename", value=c["title"], key=f"rename_input_{c['id']}")
                    if st.button("Save", key=f"save_title_{c['id']}"):
                        if new_title_input.strip():
                            chat_store.rename_chat(c["id"], new_title_input.strip())
                            st.rerun()
                    st.divider()
                    if st.button("🗑️ Delete", key=f"del_confirm_{c['id']}", type="secondary"):
                        chat_store.delete_chat(c["id"])
                        if st.session_state.get("current_chat_id") == c["id"]:
                            st.session_state.current_chat_id = None
                        st.rerun()

        if today_chats:
            st.caption("📅 Today")
            for c in today_chats:
                render_chat_item(c)

        if earlier_chats:
            st.caption("🕒 Earlier")
            for c in earlier_chats:
                render_chat_item(c)

    st.divider()

    with st.expander("⚙️ Pipeline Settings", expanded=False):
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


# --------------------------------------------------
# LLM Initialization
# --------------------------------------------------

if provider == "Groq (Cloud)":
    if not groq_api_key:
        st.info("💡 Enter your Groq API key in the sidebar settings to start.")
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
        timeout=300,
    )


# --------------------------------------------------
# Search Providers & Tool Suite
# --------------------------------------------------

def tavily_search(query: str) -> str:
    """Primary web search using Tavily."""
    if not tavily_api_key:
        raise ValueError("Tavily API key is not configured.")

    client = TavilyClient(api_key=tavily_api_key)
    response = client.search(
        query=query,
        search_depth="basic",
        max_results=5,
        include_answer=False,
    )
    results = response.get("results", [])
    if not results:
        raise ValueError("Tavily returned no results.")

    formatted_results = []
    for index, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")[:500]
        formatted_results.append(f"Result {index}\n\nTitle: {title}\n\nURL: {url}\n\nContent:\n{content}\n")
    return "\n".join(formatted_results)


def ddgs_search(query: str) -> str:
    """Fallback search using DDGS."""
    results = []
    with DDGS() as ddgs:
        search_results = ddgs.text(query, max_results=4)
        for result in search_results:
            results.append(
                {
                    "title": result.get("title", "Untitled"),
                    "url": result.get("href", result.get("url", "")),
                    "content": result.get("body", result.get("content", ""))[:500],
                }
            )

    if not results:
        raise ValueError("DDGS returned no results.")

    formatted_results = []
    for index, result in enumerate(results, start=1):
        formatted_results.append(f"Result {index}\n\nTitle: {result['title']}\n\nURL: {result['url']}\n\nContent:\n{result['content']}\n")
    return "\n".join(formatted_results)


def web_search(query: str) -> str:
    """Search strategy: Try Tavily first, fallback to DDGS."""
    try:
        result = tavily_search(query)
        return "SEARCH PROVIDER: TAVILY\n\n" + result
    except Exception as tavily_error:
        try:
            fallback_result = ddgs_search(query)
            return f"SEARCH PROVIDER: DDGS FALLBACK\n\nTavily error: {str(tavily_error)}\n\n" + fallback_result
        except Exception as ddgs_error:
            return f"SEARCH FAILED\n\nTavily error: {str(tavily_error)}\n\nDDGS error: {str(ddgs_error)}"


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())


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
        return str(wikipedia.run(query))
    except Exception:
        return search_wikipedia(query)


@tool
def arxiv_tool(query: str) -> str:
    """Search arXiv for physics, mathematics, and AI/ML academic preprint papers."""
    try:
        res = str(arxiv.run(query))
        if "No good Arxiv Result was found" in res or "error" in res.lower():
            return search_academic_papers(query)
        return res
    except Exception:
        return search_academic_papers(query)


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
# DAG Components
# --------------------------------------------------

planner = DAGPlanner(llm=llm)
executor = DAGExecutor(llm=llm, tools=tools, max_workers=max_dag_workers)


# --------------------------------------------------
# Multi-Chat Helpers (Titling, Rolling Memory, Formatting)
# --------------------------------------------------

def generate_chat_title(llm_instance, query_text: str) -> str:
    """Generate a concise 4-6 word title summarizing the user query."""
    fallback_title = query_text.strip()[:40].strip()
    if len(query_text.strip()) > 40:
        fallback_title += "..."

    prompt = (
        f"Generate a concise 4 to 6 word title summarizing this research query. "
        f"Return ONLY the title text without quotes, markdown, or punctuation.\n\n"
        f"Query: {query_text}"
    )
    try:
        resp = llm_instance.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        title = content.strip().replace('"', "").replace("'", "").replace("\n", " ").strip()
        words = title.split()
        if len(words) > 8:
            title = " ".join(words[:6])
        return title if len(title) > 2 else fallback_title
    except Exception:
        return fallback_title


def build_conversation_context(llm_instance, chat_id: str, db_path: Optional[str] = None) -> str:
    """Build compact conversation memory for follow-up turns, maintaining rolling summary for > 6 turns."""
    if not chat_id:
        return ""
    messages = chat_store.get_messages(chat_id, db_path=db_path)
    if not messages:
        return ""

    if len(messages) <= 6:
        lines = []
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"[{role}]: {m['content']}")
        return "\n\n".join(lines)

    # Chat exceeds 6 messages: collapse older turns into rolling summary
    recent_messages = messages[-4:]  # Last 2 turns (user + assistant)
    older_messages = messages[:-4]

    cached_summary = chat_store.get_chat_summary(chat_id, db_path=db_path)

    older_text_list = []
    for m in older_messages:
        role = "User" if m["role"] == "user" else "Assistant"
        c_text = m["content"]
        if len(c_text) > 800:
            c_text = c_text[:800] + "..."
        older_text_list.append(f"[{role}]: {c_text}")
    older_text = "\n\n".join(older_text_list)

    if not cached_summary:
        summary_prompt = (
            "Summarize the key facts, research findings, and entities discussed in this prior research context into a concise summary:\n\n"
            f"{older_text}\n\n"
            "Summary:"
        )
        try:
            resp = llm_instance.invoke(summary_prompt)
            cached_summary = resp.content if hasattr(resp, "content") else str(resp)
            chat_store.update_chat_summary(chat_id, cached_summary, db_path=db_path)
        except Exception:
            cached_summary = older_text[:1000]

    recent_text_list = []
    for m in recent_messages:
        role = "User" if m["role"] == "user" else "Assistant"
        recent_text_list.append(f"[{role}]: {m['content']}")
    recent_text = "\n\n".join(recent_text_list)

    return f"Rolling Summary of Earlier Research:\n{cached_summary}\n\nRecent Turns:\n{recent_text}"


def format_and_clean_answer(raw_answer: str) -> str:
    """Clean tokenization artifacts, bold markers, and format tables."""
    answer = raw_answer.replace("\u2217", "*").replace("∗", "*")
    answer = answer.replace("\u202f", " ").replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    answer = answer.replace("\u2011", "-").replace("‑", "-")

    answer = re.sub(r"=\s*[*]+\s*[\d,\.]+\s*[A-Za-z]*\s*=\s*[*]+", " = **", answer)
    answer = re.sub(r"÷\s*\d+\s*=\s*[*]+[\d,\.]+\s*[A-Za-z]*\s*÷\s*\d+\s*=\s*[*]+", " ÷ 5 = **", answer)

    lines = [l.strip() for l in answer.split("\n")]
    formatted_lines = []
    idx = 0
    while idx < len(lines):
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
    answer = re.sub(r"\*\*\s+([^\*\n]+?)\s+\*\*", r"**\1**", answer)
    answer = re.sub(r"\*\*\s+([^\*\n]+?)\*\*", r"**\1**", answer)
    answer = re.sub(r"\*\*([^\*\n]+?)\s+\*\*", r"**\1**", answer)
    answer = re.sub(r'(?<=\w)(\*\*[^*\n]+?\*\*)', r' \1', answer)   # space before span if glued to prior word
    answer = re.sub(r'(\*\*[^*\n]+?\*\*)(?=\w)', r'\1 ', answer)     # space after span if glued to next word
    answer = re.sub(r"(?<!\\)\$([0-9])", r"\\$\1", answer)
    return answer


# --------------------------------------------------
# Main Header & Controls
# --------------------------------------------------

st.markdown(
    """
    <div class="page-header">
        <h1 class="page-title">🔍 LLM Research Engine</h1>
        <div class="page-subtitle">Autonomous multi-agent research with directed acyclic graphs & multi-turn memory</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_mode, col_info = st.columns([2.5, 1])

with col_mode:
    search_mode = st.segmented_control(
        "Research Mode",
        ["⚡ Fast / Overview", "🔬 Deep Research", "🎓 Academic Literature"],
        default="🔬 Deep Research",
        label_visibility="collapsed",
    )
    if search_mode is None:
        search_mode = "🔬 Deep Research"

mode_key = "deep"
if "Fast" in search_mode:
    mode_key = "fast"
elif "Academic" in search_mode:
    mode_key = "academic"

with col_info:
    st.caption(f"Provider: **{provider}** | Model: `{model_name}`")


# --------------------------------------------------
# Transcript Display
# --------------------------------------------------

current_chat_id = st.session_state.get("current_chat_id")
messages = chat_store.get_messages(current_chat_id) if current_chat_id else []

if not messages:
    st.markdown(
        """
        <div class="notion-callout">
            <div class="notion-callout-icon">💡</div>
            <div class="notion-callout-text">
                <b>Welcome to the Autonomous Research Assistant!</b><br>
                Ask any complex, multi-entity, or academic question. The agent will construct a Directed Acyclic Graph (DAG), execute parallel tool calls across 15 specialized search domains, and synthesize a verified report.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("dag_snapshot"):
                try:
                    snapshot_data = json.loads(msg["dag_snapshot"])
                    nodes_list = snapshot_data.get("nodes", [])
                    if nodes_list:
                        with st.expander("🔍 Inspect Node Findings", expanded=False):
                            for node in nodes_list:
                                st.markdown(f"**🔵 {node.get('id')}: {node.get('task')}**")
                                st.caption(f"Status: `{node.get('status')}` | Tool: `{node.get('tool_used', 'direct answer')}`")
                                if node.get("result"):
                                    st.text_area(
                                        f"Output ({node.get('id')})",
                                        value=node["result"],
                                        height=130,
                                        disabled=True,
                                        key=f"hist_res_{msg['id']}_{node.get('id')}",
                                    )
                                if node.get("error"):
                                    st.error(f"Error: {node['error']}")
                                st.divider()
                except Exception:
                    pass


# --------------------------------------------------
# Chat Input & Execution Pipeline
# --------------------------------------------------

user_query = st.chat_input("Ask anything or follow up on your research...")

if user_query:
    user_text = user_query.strip()
    if not user_text:
        st.stop()

    is_first_turn = False
    if not current_chat_id:
        current_chat_id = chat_store.create_chat(title="New Research")
        st.session_state.current_chat_id = current_chat_id
        is_first_turn = True
    else:
        existing_msgs = chat_store.get_messages(current_chat_id)
        if len(existing_msgs) == 0:
            is_first_turn = True

    # 1. Store and display user message
    chat_store.append_message(current_chat_id, "user", user_text)
    with st.chat_message("user"):
        st.markdown(user_text)

    # 2. Build prior conversation memory context
    chat_context = build_conversation_context(llm, current_chat_id)

    # 3. Execute Research inside Assistant message
    with st.chat_message("assistant"):
        col_dag_canvas, col_report_canvas = st.columns([1, 1.35], gap="large")

        with col_dag_canvas:
            st.markdown("### 🧠 Research DAG Flow")
            progress_container = st.empty()
            dag_graph_container = st.empty()

        with col_report_canvas:
            report_status_container = st.empty()

        # Step A: Create DAG with prior context
        with col_dag_canvas:
            with st.spinner("Planning research graph..."):
                dag = planner.create_dag(user_text, mode=mode_key, context=chat_context)

        # Step B: Render initial DAG graph
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

        with report_status_container:
            skeleton_html = render_synthesis_skeleton()
            if hasattr(st, "html"):
                st.html(skeleton_html)
            else:
                st.markdown(skeleton_html, unsafe_allow_html=True)

        def update_progress(node, status):
            node_statuses[node.id] = status
            render_progress()
            render_dag()

        # Step C: Execute DAG
        with st.spinner(f"Executing DAG with {max_dag_workers} parallel workers..."):
            dag = executor.execute(dag, progress_callback=update_progress)

        # Step D: Synthesis
        results = dag.get_results()
        combined_results = "\n\n".join(
            [f"### {node_id}\n\n{result}" for node_id, result in results.items()]
        )

        if mode_key == "fast":
            length_guideline = "- **Concise & Direct**: Keep your answer crisp, clear, and to the point (1-3 focused paragraphs or succinct key points)."
        elif mode_key == "academic":
            length_guideline = "- **Academic Depth**: Provide a detailed literature synthesis with research findings, methodology context, and paper citations."
        else:
            length_guideline = "- **Adaptive Length**: Match the depth to the question. Provide a clear, comprehensive breakdown."

        context_block = f"\nPrior Conversation Context:\n{chat_context}\n" if chat_context else ""

        final_prompt = f"""
You are an intelligent AI research assistant synthesizing findings for a user query.
{context_block}
User Query:
"{user_text}"

Research Mode: {search_mode}

Research Findings from DAG nodes:
{combined_results}

Formatting & Presentation Guidelines:
1. **Direct Answer First**: Begin immediately with the direct, definitive answer in the opening paragraph. If prior conversation context is relevant, build seamlessly on it.
2. **Markdown Tables**: For comparisons or multi-entity data (prices, market caps, weather, versions), ALWAYS format them in clean GitHub-flavored Markdown tables with column headers and delimiter rows:
| Company | Ticker | Current Price (USD) | Market Cap (USD) |
| :--- | :--- | :--- | :--- |
| Apple | AAPL | $309.35 | $4,514.71 B |
| Microsoft | MSFT | $483.24 | $3,588.32 B |
3. **Clean Calculations**: Present calculations in clean, readable bullet points:
- **Total Market Cap**: $18,954.02 B
- **Average Market Cap**: $3,790.80 B
4. {length_guideline.strip()}
5. **Natural Sources & Hyperlinks**: Seamlessly embed markdown links (e.g., [Yahoo Finance AAPL](https://finance.yahoo.com/quote/AAPL)) when citing sources or facts.
6. **Clean Bold Headers**: Use clean section headings or bold text without leading or trailing spaces inside the asterisks.
7. **No Meta-Talk**: Never mention "DAG nodes", "internal tasks", or execution logs.
"""

        synth_system_prompt = (
            "You are an expert research synthesizer. All data gathering, web research, calculations, "
            "and fact extraction have already been completed by upstream agent workers.\n"
            "Your ONLY task is to write a final, comprehensive, well-structured Markdown answer strictly based on the provided findings.\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. Do NOT call, invoke, or output any tool calls, function calls, or JSON actions.\n"
            "2. Output ONLY direct, fluent Markdown text with clean Markdown tables (`| ... |`)."
        )

        with report_status_container:
            with st.spinner("Synthesizing final research report..."):
                final_response = None
                try:
                    final_response = llm.invoke(
                        [
                            SystemMessage(content=synth_system_prompt),
                            HumanMessage(content=final_prompt),
                        ]
                    )
                except Exception as synth_err:
                    err_str = str(synth_err)
                    try:
                        fallback_text = (
                            f"Please write a comprehensive final summary answering this question: '{user_text}' based on these gathered findings:\n\n"
                            f"{combined_results}\n\n"
                            f"Do not call tools. Respond only in clean Markdown tables and text."
                        )
                        final_response = llm.invoke(fallback_text)
                    except Exception as fb_err:
                        final_response = (
                            f"### Research Findings\n\n{combined_results}\n\n"
                            f"> ⚠️ *Note: Final LLM synthesis timed out or encountered an error ({str(fb_err)[:120]}). Displaying raw verified findings above.*"
                        )

        if hasattr(final_response, "content"):
            answer = str(final_response.content)
        else:
            answer = str(final_response)

        answer = format_and_clean_answer(answer)
        report_status_container.empty()

        with col_report_canvas:
            st.markdown(answer)

        # Snapshot DAG execution
        dag_snapshot_data = {
            "nodes": [
                {
                    "id": node.id,
                    "task": node.task,
                    "status": node.status,
                    "tool_used": node.tool_used,
                    "result": node.result,
                    "error": node.error,
                    "dependencies": node.dependencies,
                }
                for node in dag.nodes.values()
            ]
        }
        dag_snapshot_json = json.dumps(dag_snapshot_data)

        # Append assistant message
        chat_store.append_message(
            current_chat_id,
            "assistant",
            answer,
            dag_snapshot=dag_snapshot_json,
        )

        # Auto-title on first turn
        if is_first_turn:
            new_title = generate_chat_title(llm, user_text)
            chat_store.rename_chat(current_chat_id, new_title)

        with col_dag_canvas:
            with st.expander("🔍 Inspect Node Findings", expanded=False):
                for node in dag.nodes.values():
                    st.markdown(f"**🔵 {node.id}: {node.task}**")
                    st.caption(f"Status: `{node.status}` | Tool: `{node.tool_used or 'direct answer'}`")
                    if node.result:
                        st.text_area(
                            f"Output ({node.id})",
                            value=node.result,
                            height=140,
                            disabled=True,
                            key=f"live_res_{node.id}",
                        )
                    if node.error:
                        st.error(f"Error: {node.error}")
                    st.divider()

    st.rerun()