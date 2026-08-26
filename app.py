import json
import os
import re
import time
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
from metrics import (
    MetricsCollector,
    get_current_metrics,
    set_current_metrics,
    log_event,
    MetricsCallbackHandler,
)
from budget import QueryBudget, load_budget_config, get_current_budget
from evidence import EvidenceStore, build_research_packet
from utils.tools import (
    search_github,
    search_finance,
    search_pubmed,
    python_calculator,
    normalize_math_expression,
    evaluate_math_expression,
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
from utils.langchain_tools import (
    ALL_RESEARCH_TOOLS,
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

        use_llm_titling = st.checkbox(
            "Use LLM for Chat Titling",
            value=False,
            help="Default is False (uses deterministic 0-call titling to save tokens & latency).",
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


from utils.search_pipeline import search_and_extract_evidence


tools = ALL_RESEARCH_TOOLS


# --------------------------------------------------
# DAG Components
# --------------------------------------------------

planner = DAGPlanner(llm=llm)
executor = DAGExecutor(llm=llm, tools=tools, max_workers=max_dag_workers)


# --------------------------------------------------
# Multi-Chat Helpers (Titling, Rolling Memory, Formatting)
# --------------------------------------------------

def generate_deterministic_title(query_text: str, max_words: int = 10) -> str:
    """
    Generate a clean 6-10 word title deterministically with ZERO LLM calls:
    - Extracts first meaningful sentence/clause
    - Strips noisy prefixes ('what is', 'can you tell me about', 'compare', etc.)
    - Removes punctuation and markdown
    - Truncates to max_words
    """
    if not query_text or not query_text.strip():
        return "New Research"

    text = query_text.strip()
    text = re.sub(r"[`\"'#\*\-_]", "", text)
    sentences = re.split(r"[.!?\n]", text)
    first_sent = sentences[0].strip() if sentences else text

    filler_patterns = [
        r"^(?:can\s+you\s+|could\s+you\s+|please\s+)*(?:tell\s+me\s+about|explain|what\s+is|what\s+are|how\s+does|how\s+to|search\s+for|look\s+up|find\s+out\s+about|give\s+me|compare)\s*",
    ]
    cleaned = first_sent
    for pat in filler_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE).strip()

    if not cleaned:
        cleaned = first_sent

    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])

    if cleaned.islower():
        cleaned = cleaned.title()
    elif len(cleaned) > 0:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned[:50].strip() or "New Research"


def generate_chat_title(llm_instance, query_text: str, use_llm: bool = False) -> str:
    """Generate a clean title. Uses deterministic titling (0 LLM calls) by default."""
    if not use_llm or llm_instance is None:
        return generate_deterministic_title(query_text)

    try:
        prompt = (
            f"Generate a concise 4 to 6 word title summarizing this research query. "
            f"Return ONLY the title text without quotes, markdown, or punctuation.\n\n"
            f"Query: {query_text}"
        )
        resp = llm_instance.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        title = content.strip().replace('"', "").replace("'", "").replace("\n", " ").strip()
        words = title.split()
        if len(words) > 8:
            title = " ".join(words[:6])
        return title if len(title) > 2 else generate_deterministic_title(query_text)
    except Exception:
        return generate_deterministic_title(query_text)


def build_conversation_context(
    llm_instance,
    chat_id: str,
    db_path: Optional[str] = None,
    use_llm_summary: bool = False,
    budget: Optional[QueryBudget] = None,
) -> str:
    """
    Build compact conversation memory for follow-up turns:
    - Zero LLM calls for <= 8 messages (deterministic recent turns).
    - Uses cached summary if already generated.
    - Only triggers LLM summarization if messages > 10 turns AND use_llm_summary is True AND budget permits.
    """
    if not chat_id:
        return ""
    messages = chat_store.get_messages(chat_id, db_path=db_path)
    if not messages:
        return ""

    if len(messages) <= 8 or not use_llm_summary:
        # Deterministic recent context (last 3 turns / 6 messages)
        recent = messages[-6:]
        lines = []
        for m in recent:
            role = "User" if m["role"] == "user" else "Assistant"
            text = m["content"].strip()
            if len(text) > 400:
                text = text[:400] + "..."
            lines.append(f"[{role}]: {text}")
        return "\n\n".join(lines)

    # Chat exceeds 8 messages: check cached summary
    recent_messages = messages[-4:]
    older_messages = messages[:-4]

    cached_summary = chat_store.get_chat_summary(chat_id, db_path=db_path)

    if not cached_summary and len(messages) > 10 and llm_instance is not None:
        bg = budget or get_current_budget()
        can_call = (bg.can_call_llm() and not bg.should_skip_optional_work()) if bg else True
        if can_call:
            older_text_list = []
            for m in older_messages:
                role = "User" if m["role"] == "user" else "Assistant"
                c_text = m["content"][:400]
                older_text_list.append(f"[{role}]: {c_text}")
            older_text = "\n\n".join(older_text_list)

            summary_prompt = (
                "Summarize the key facts, research findings, and entities discussed in this prior research context into a concise summary:\n\n"
                f"{older_text}\n\n"
                "Summary:"
            )
            try:
                if bg:
                    bg.reserve("llm")
                resp = llm_instance.invoke(summary_prompt)
                cached_summary = resp.content if hasattr(resp, "content") else str(resp)
                chat_store.update_chat_summary(chat_id, cached_summary, db_path=db_path)
                if bg:
                    bg.record_usage(llm_calls=1, input_tokens=max(1, len(summary_prompt) // 4), output_tokens=max(1, len(cached_summary) // 4))
            except Exception:
                cached_summary = ""

    recent_text_list = []
    for m in recent_messages:
        role = "User" if m["role"] == "user" else "Assistant"
        recent_text_list.append(f"[{role}]: {m['content'][:400]}")
    recent_text = "\n\n".join(recent_text_list)

    if cached_summary:
        return f"Rolling Summary of Earlier Research:\n{cached_summary}\n\nRecent Turns:\n{recent_text}"
    return recent_text


def render_live_budget_meter_hud(budget_instance, collector_instance, container):
    """Render a live, interactive execution telemetry HUD with progress meters."""
    if not budget_instance:
        return

    cfg = budget_instance.config
    col_m = collector_instance.metrics if collector_instance else None

    tok_used = getattr(budget_instance, "total_tokens_used", 0)
    tok_max = cfg.max_total_tokens
    tok_pct = min(1.0, tok_used / tok_max) if tok_max else 0.0

    llm_used = getattr(budget_instance, "llm_calls_used", 0)
    llm_max = cfg.max_llm_calls

    tool_used = getattr(budget_instance, "tool_calls_used", 0)
    tool_max = cfg.max_tool_calls

    start_t = getattr(budget_instance, "_start_time", None)
    elapsed = round(time.perf_counter() - start_t, 1) if start_t else 0.0
    rem_time = round(budget_instance.remaining_time(), 1)

    browser_used = getattr(budget_instance, "browser_calls_used", 0)
    retries = col_m.retries if col_m else 0
    skipped = col_m.skipped_nodes if col_m else 0

    with container:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🛡️ Token Budget", f"{tok_used:,} / {tok_max:,}", f"{tok_pct:.0%}")
        with col2:
            st.metric("🤖 LLM Calls", f"{llm_used} / {llm_max}")
        with col3:
            st.metric("🔧 Tools", f"{tool_used} / {tool_max}")
        with col4:
            st.metric("⏱️ Elapsed", f"{elapsed:.1f}s", f"Left: {rem_time:.0f}s")

        if retries or skipped or browser_used:
            st.caption(f"⚡ Browser: `{browser_used}` | ⏱️ Retries: `{retries}` | ⇥ Skipped: `{skipped}`")


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


def render_developer_debug_section(metrics_data: dict, key_suffix: str = ""):
    """Render a comprehensive developer/debug panel in Streamlit for request tracing & performance."""
    if not metrics_data:
        return

    with st.expander("🛠️ Developer / Debug Performance & Tracing Metrics", expanded=False):
        # 1. Summary Cards
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            tot_lat = metrics_data.get("total_latency", 0.0)
            st.metric("⏱️ Total Latency", f"{tot_lat:.2f}s", help="End-to-end request duration")
        with col2:
            llm_calls = metrics_data.get("llm_call_count", 0)
            st.metric("🤖 LLM Calls", f"{llm_calls}", help="Total LLM invocations across planner, nodes, and synthesis")
        with col3:
            tot_tok = metrics_data.get("total_tokens", 0)
            in_tok = metrics_data.get("input_tokens", 0)
            out_tok = metrics_data.get("output_tokens", 0)
            st.metric("🧮 Total Tokens", f"{tot_tok:,}", f"In: {in_tok:,} | Out: {out_tok:,}", help="Prompt and completion tokens")
        with col4:
            tool_calls = metrics_data.get("tool_call_count", 0)
            st.metric("🛠️ Tool Calls", f"{tool_calls}", help="Total tool executions across DAG nodes")
        with col5:
            b_calls = metrics_data.get("browser_calls", 0)
            pw_calls = metrics_data.get("playwright_calls", 0)
            http_calls = metrics_data.get("http_fetch_calls", 0)
            st.metric("🌐 Browser Calls", f"{b_calls}", f"HTTP: {http_calls} | PW: {pw_calls}", help="Webpage extractions (Fast HTTP vs Playwright)")
        with col6:
            retries = metrics_data.get("retries", 0)
            rate_limits = metrics_data.get("rate_limit_errors", 0)
            st.metric("🔁 Retries / 429s", f"{retries} / {rate_limits}", help="Retried attempts and 429 Rate-limit errors encountered")

        st.divider()

        # QueryBudget Status Banner
        budget_st = metrics_data.get("budget_status", {})
        if budget_st:
            is_ex = metrics_data.get("budget_exhausted", False)
            badge = "⚠️ EXHAUSTED" if is_ex else "✅ OK"
            reasons = ", ".join(budget_st.get("exhaustion_reasons", [])) or "None"
            st.info(
                f"🛡️ **Query Budget:** `{badge}` | **LLM Calls:** `{budget_st.get('llm_calls')}` | "
                f"**Tokens:** `{budget_st.get('total_tokens')}` | "
                f"**Time Remaining:** `{budget_st.get('remaining_time_seconds')}s` | "
                f"**Exhaustion Reason:** `{reasons}`"
            )

        # 2. Latency & Node Breakdown
        col_l1, col_l2, col_l3 = st.columns(3)
        with col_l1:
            st.caption(f"🧠 **Planner Latency:** `{metrics_data.get('planner_latency', 0.0):.3f}s`")
        with col_l2:
            st.caption(f"✍️ **Synthesis Latency:** `{metrics_data.get('final_synthesis_latency', 0.0):.3f}s`")
        with col_l3:
            dag_count = metrics_data.get("dag_node_count", 0)
            done_nodes = metrics_data.get("completed_nodes", 0)
            failed_nodes = metrics_data.get("failed_nodes", 0)
            st.caption(f"📊 **DAG Nodes:** `{done_nodes}/{dag_count} completed` (Failed: `{failed_nodes}`)")

        # Tool Names
        tool_names = metrics_data.get("tool_names", [])
        if tool_names:
            st.caption(f"**Tools Used:** `{', '.join(tool_names)}`")

        # 3. DAG Node Stats Table
        per_node = metrics_data.get("per_node_latency", {})
        if per_node:
            st.markdown("##### 🧩 DAG Node Stats")
            node_table_rows = []
            for nid, ndata in per_node.items():
                node_table_rows.append({
                    "Node": nid,
                    "Status": ndata.get("status", "N/A"),
                    "Latency (s)": f"{ndata.get('latency_seconds', 0.0):.3f}",
                    "Tool Used": ndata.get("tool_used", "direct"),
                    "Error": ndata.get("error") or "None",
                })
            st.dataframe(node_table_rows, use_container_width=True)

        # 4. Context Size & LLM Calls table
        ctx_list = metrics_data.get("context_size_per_llm_call", [])
        if ctx_list:
            st.markdown("##### 🔍 LLM Call Context & Token Breakdown")
            ctx_table_rows = []
            for ctx in ctx_list:
                ctx_table_rows.append({
                    "Call #": ctx.get("call_index"),
                    "Stage": ctx.get("stage"),
                    "Prompt Chars": f"{ctx.get('chars', 0):,}",
                    "Est/Actual In Tokens": f"{ctx.get('input_tokens', 0):,}",
                    "Out Tokens": f"{ctx.get('output_tokens', 0):,}",
                    "Latency (s)": f"{ctx.get('latency_seconds', 0.0):.3f}",
                    "Model": ctx.get("model", ""),
                })
            st.dataframe(ctx_table_rows, use_container_width=True)

        # 5. Retries & Rate Limits breakdown
        retry_list = metrics_data.get("retry_details", [])
        if retry_list:
            st.markdown("##### ⏱️ Retries & Rate Limits Details")
            retry_rows = []
            for r in retry_list:
                retry_rows.append({
                    "Stage": r.get("stage"),
                    "Error Type": r.get("error_type", "rate_limit"),
                    "Wait (s)": f"{r.get('wait_duration_seconds', 0.0):.2f}",
                    "Reason / Details": r.get("reason", "")[:120],
                })
            st.dataframe(retry_rows, use_container_width=True)


# --------------------------------------------------
# Main Header & Controls
# --------------------------------------------------

col_head, col_info = st.columns([3.2, 1.2])

with col_head:
    st.markdown(
        """
        <div class="page-header">
            <h1 class="page-title">🔍 LLM Research Engine</h1>
            <div class="page-subtitle">Autonomous multi-agent research with directed acyclic graphs & multi-turn memory</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_info:
    st.caption(f"Provider: **{provider}** | Model: `{model_name}`")

mode_key = "deep"


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

                    if snapshot_data.get("metrics"):
                        render_developer_debug_section(snapshot_data["metrics"], key_suffix=f"hist_{msg['id']}")
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

    # Initialize request metrics collector & query budget
    collector = MetricsCollector(
        research_mode=mode_key,
        model_provider=f"{provider} / {model_name}",
    )
    collector.start()
    budget = QueryBudget(mode=mode_key)
    budget.start()

    log_event(
        "request_started",
        request_id=collector.metrics.request_id,
        query=user_text,
        mode=mode_key,
        provider=provider,
        model=model_name,
    )

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
            hud_container = st.container()
            progress_container = st.empty()
            dag_graph_container = st.empty()

        with col_report_canvas:
            report_status_container = st.empty()

        # Step A: Create DAG with prior context
        with col_dag_canvas:
            with st.spinner("Planning research graph..."):
                dag = planner.create_dag(
                    user_text,
                    mode=mode_key,
                    context=chat_context,
                    metrics=collector,
                    budget=budget,
                )

        # Step B: Render initial DAG graph
        node_statuses = {node_id: "PENDING" for node_id in dag.nodes}

        def render_progress():
            total = len(node_statuses)
            done = sum(1 for s in node_statuses.values() if s in ("COMPLETED", "FAILED", "SKIPPED"))
            render_live_budget_meter_hud(budget, collector, hud_container)
            with progress_container:
                st.caption(f"{done} of {total} nodes complete")
                st.progress(done / total if total else 0.0)

        def render_dag():
            if not dag.nodes:
                with dag_graph_container:
                    st.info("⚡ **Direct Fast Mode Execution:** DAG bypassed for minimum latency.")
                return
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

        # Step C: Execute DAG (if nodes exist)
        if dag.nodes:
            with st.spinner(f"Executing DAG with {max_dag_workers} parallel workers..."):
                dag = executor.execute(
                    dag,
                    progress_callback=update_progress,
                    metrics=collector,
                    budget=budget,
                )

        # Step D: Structured EvidenceStore & ResearchPacket Compilation
        results = dag.get_results()
        if results:
            evidence_store = EvidenceStore()
            for node_id, result in results.items():
                evidence_store.add_from_raw(result, node_id=node_id)
            research_packet = build_research_packet(question=user_text, store=evidence_store)
            combined_results = research_packet.to_markdown()
        else:
            combined_results = "Direct answer query without intermediate DAG nodes."

        length_guideline = "- **Adaptive Depth**: Match the depth to the question. Provide a clear, comprehensive breakdown with verified findings and citations."

        context_block = f"\nPrior Conversation Context:\n{chat_context}\n" if chat_context else ""

        final_prompt = f"""
You are an intelligent AI research assistant synthesizing findings for a user query.
{context_block}
User Query:
"{user_text}"

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

        synth_prompt_chars = len(synth_system_prompt) + len(final_prompt)
        with report_status_container:
            with st.spinner("Synthesizing final research report..."):
                final_response = None
                synth_start = time.perf_counter()
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
                synth_latency = time.perf_counter() - synth_start

        synth_in_tokens = 0
        synth_out_tokens = 0
        if hasattr(final_response, "usage_metadata") and isinstance(final_response.usage_metadata, dict):
            synth_in_tokens = final_response.usage_metadata.get("input_tokens", 0)
            synth_out_tokens = final_response.usage_metadata.get("output_tokens", 0)
        elif hasattr(final_response, "response_metadata") and isinstance(final_response.response_metadata, dict):
            tu = final_response.response_metadata.get("token_usage", {})
            if isinstance(tu, dict):
                synth_in_tokens = tu.get("prompt_tokens", 0)
                synth_out_tokens = tu.get("completion_tokens", 0)

        collector.record_synthesis(
            latency=synth_latency,
            context_chars=synth_prompt_chars,
            input_tokens=synth_in_tokens,
            output_tokens=synth_out_tokens,
            model=model_name,
        )

        if hasattr(final_response, "content"):
            answer = str(final_response.content)
        else:
            answer = str(final_response)

        answer = format_and_clean_answer(answer)
        report_status_container.empty()

        with col_report_canvas:
            st.markdown(answer)

        # Record and finalize budget
        if budget:
            budget.record_usage(
                llm_calls=1,
                input_tokens=synth_in_tokens or max(1, synth_prompt_chars // 4),
                output_tokens=synth_out_tokens or max(1, len(answer) // 4),
            )
            collector.record_budget(budget.get_status(), is_exhausted=budget.exhausted())
            budget.finish()

        # Finalize request metrics
        collector.finish()
        metrics_dict = collector.to_dict()
        log_event(
            "request_completed",
            request_id=collector.metrics.request_id,
            total_latency=collector.metrics.total_latency,
            total_tokens=collector.metrics.total_tokens,
            tool_calls=collector.metrics.tool_call_count,
            llm_calls=collector.metrics.llm_call_count,
        )

        # Snapshot DAG execution & metrics
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
            ],
            "metrics": metrics_dict,
        }
        dag_snapshot_json = json.dumps(dag_snapshot_data)

        # Append assistant message
        chat_store.append_message(
            current_chat_id,
            "assistant",
            answer,
            dag_snapshot=dag_snapshot_json,
        )

        # Auto-title on first turn (deterministic 0-call by default)
        if is_first_turn:
            new_title = generate_chat_title(llm, user_text, use_llm=use_llm_titling)
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

            render_developer_debug_section(metrics_dict, key_suffix="live")

    st.rerun()