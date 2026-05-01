import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_groq import ChatGroq

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LangChain Search Chat", page_icon="🔎")
st.title("🔎 LangChain – Chat with Search")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=st.secrets.get("GROQ_API_KEY", ""),
    placeholder="gsk_...",
)

# ── Tools (cached — created once per session) ──────────────────────────────────
@st.cache_resource
def get_tools():
    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
    )
    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
    )
    search = DuckDuckGoSearchRun(name="Search")
    return [search, arxiv, wiki]


tools = get_tools()

# ── Chat history ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm a chatbot that can search the web, Wikipedia, and Arxiv. How can I help you?",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
prompt = st.chat_input(placeholder="What is machine learning?")

if prompt:
    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to continue.")
        st.stop()

    # Record user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Build LLM (lightweight object — recreated only when key changes)
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        streaming=True,
    )

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,  # fixed typo: was "handling_"
        verbose=False,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            # Pass only the latest prompt — full history isn't the right format here
            response = agent.run(prompt, callbacks=[st_cb])
        except Exception as e:
            response = f"Sorry, something went wrong: {e}"
            st.error(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
