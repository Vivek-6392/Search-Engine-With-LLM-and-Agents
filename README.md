# 🔍 DeepSearchAI - Autonomous Multi-Agent Research Engine

An advanced, multi-agent AI research engine powered by **Directed Acyclic Graph (DAG) task decomposition**, **parallel agent execution**, and a suite of **15 specialized domain tools** for live web synthesis, academic literature review, code search, and market intelligence.

---

## 🌟 Key Highlights

- **🧠 Mode-Aware DAG Planning**: Automatically breaks down complex queries into structured, multi-tier dependency graphs with parallel execution.
- **⚡ 3 Specialized Research Modes**:
  - **`⚡ Fast / Overview`**: High-speed, focused research with direct answers and zero latency overhead.
  - **`🔬 Deep Research`**: Multi-perspective deep dive analyzing comparisons, nuances, and technical details.
  - **`🎓 Academic Literature`**: Comprehensive scientific literature search across arXiv and OpenAlex (250M+ papers).
- **🛠️ 15-Tool Multi-Domain Suite**: Autonomous routing across live web, browser extraction, academic papers, open-source repos, AI models, market quotes, and math computation.
- **🌈 Live Glowing Neon DAG Flow**: Real-time interactive UI displaying task status, glowing gradient connections, and live tool badges.
- **🔌 Multi-Provider Support**: Seamlessly switch between ultra-fast cloud LPUs (**Groq**), local offline LLMs (**Ollama**), and **OpenAI**.

---

## 🛠️ 15 Specialized Multi-Domain Tools

All tools are configured with strongly typed argument schemas and autonomous agent routing:

| Domain | Tool | Description & Engine |
|---|---|---|
| **🌐 Live Web** | `web_search_tool` | Real-time web facts, news, and scores (Tavily AI + DuckDuckGo fallback). |
| **🌐 Browser** | `web_browser_tool` | Sub-second HTTP fetch + Headless Playwright Chromium + Search fallback. |
| **📚 Knowledge** | `wikipedia_tool` | Entity definitions, history, biographies, and foundational overviews. |
| **🎓 AI / Physics** | `arxiv_tool` | Preprints in computer science, physics, mathematics, and machine learning. |
| **🔬 Global Science** | `academic_papers_tool` | 250M+ peer-reviewed papers, citations, and DOIs (OpenAlex Graph). |
| **🧬 Medicine** | `pubmed_tool` | Clinical trials, biomedical research, oncology, and pharmacology (NCBI). |
| **💻 Code / Repos** | `github_search_tool` | Top repositories, star counts, maintainers, and README snippets. |
| **🤗 AI Models** | `huggingface_tool` | Open-source models, GGUF weights, and datasets on Hugging Face Hub. |
| **🐛 Debugging** | `stackoverflow_tool` | Programming questions, code solutions, and accepted answers. |
| **💬 Tech Sentiment**| `hackernews_tool` | Y Combinator Hacker News discussions, startup launches, and tech debates. |
| **📦 Packages** | `package_lookup_tool` | Version inspection and licenses for Python (PyPI) and Node.js (NPM). |
| **📈 Markets** | `finance_tool` | Real-time stock prices (NVDA, AAPL), crypto quotes (BTC, ETH), and market cap. |
| **💱 Currency** | `forex_tool` | Live foreign exchange rates across USD, EUR, INR, GBP, JPY, CAD. |
| **🌦️ Weather** | `weather_tool` | Live meteorological conditions, temperature, humidity, and forecasts. |
| **🧮 Exact Math** | `calculator_tool` | Safe Python mathematical and statistical evaluation (eliminates math hallucinations). |

---

## 🏗️ Architecture

```
User Query ──► 🎯 DAG Planner (LLM) ──► Tiered Research Graph
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
             Node 1 (Agent Worker)                           Node 2 (Agent Worker)
             [Tools: Web, ArXiv, GitHub...]                  [Tools: Finance, Browser...]
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                                   Lead Synthesizer (LLM)
                                              │
                                              ▼
                                 Direct Factual Synthesis
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Vivek-6392/Search-Engine-With-LLM-and-Agents.git
cd Search-Engine-With-LLM-and-Agents
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
# LLM Providers (Choose one or both)
GROQ_API_KEY="your_groq_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"

# Local Ollama (Optional)
OLLAMA_BASE_URL="http://localhost:11434"

# Search Provider
TAVILY_API_KEY="your_tavily_api_key_here"
```

### 5. Launch the Application
```bash
streamlit run app.py
```

---

## ⚙️ Configuration & Options

- **Provider Switching**: Toggle between **Groq**, **Ollama (Local)**, and **OpenAI** in the sidebar.
- **Model Selection**: Switch between `qwen/qwen3.6-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, etc.
- **Parallel Workers**: Adjust parallel DAG concurrency (1 to 5 workers) based on your system hardware.

---

## 💬 Multi-Chat Sessions & Memory Architecture

The application includes multi-session chat persistence powered by SQLite (`data/chats.db`):
- **Named Chat Threads**: Create new threads, browse previous research runs, rename chats inline, and delete old conversations.
- **Auto-Titling**: Automatic 4–6 word title generation summarizing the first research turn using the active LLM.
- **Rolling Memory**: Automatically maintains rolling context summaries for extended research discussions beyond 6 turns to keep prompt token overhead bounded.
- **DAG Snapshots**: Every assistant response stores its full DAG node graph and tool findings, enabling retrospective inspection for historical turns.

---

## ☁️ Deploying to Streamlit Community Cloud

You can deploy this research engine for free on [Streamlit Community Cloud](https://share.streamlit.io):

1. **Push to GitHub**:
   Ensure your code is pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Configure Streamlit deployment"
   git push origin main
   ```

2. **Connect Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **"New app"** (or **"Create app"**).
   - Select your repository: `Vivek-6392/Search-Engine-With-LLM-and-Agents`
   - Set **Branch**: `main` (or your active branch)
   - Set **Main file path**: `app.py`

3. **Configure Secrets**:
   - Before clicking Deploy, click **"Advanced settings..."**
   - In the **Secrets** section, enter your API keys (TOML format):
     ```toml
     GROQ_API_KEY = "gsk_your_groq_api_key"
     TAVILY_API_KEY = "tvly-your_tavily_api_key"
     OPENAI_API_KEY = "sk-your_openai_api_key"
     ```
   - Click **Save**.

4. **Deploy**:
   - Click **Deploy!**
   - Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch the app with live updates.

---

## 📄 License
Distributed under the MIT License. See [LICENSE](https://github.com/Vivek-6392/Search-Engine-With-LLM-and-Agents/blob/feature/dag-web-browser/LICENSE) for more information.