import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from dag.mode_strategies import (
    execute_fast_mode,
    execute_normal_mode,
    execute_deep_mode,
    execute_academic_mode,
)
from dag.planner import DAGPlanner
from dag.executor import DAGExecutor
from browser import browse_webpage
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
from app import (
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
from metrics.collector import MetricsCollector
from budget import QueryBudget, load_budget_config

load_dotenv(override=True)

groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
model_name = "openai/gpt-oss-20b"

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=model_name,
    temperature=0,
)

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

planner = DAGPlanner(llm=llm)
executor = DAGExecutor(llm=llm, tools=tools, max_workers=4)


def run_single_benchmark(query: str, mode: str, label: str):
    print(f"\n=======================================================")
    print(f"Running Benchmark: {label} (Mode: {mode.upper()})")
    print(f"Query: {query}")
    print(f"=======================================================")

    collector = MetricsCollector(
        research_mode=mode,
        model_provider=f"Groq / {model_name}",
    )
    collector.start()

    budget = QueryBudget(mode=mode, config=load_budget_config(mode))
    budget.start()

    ans = ""
    dag = None
    start_time = time.perf_counter()

    if mode == "fast":
        ans, dag = execute_fast_mode(
            query=query,
            llm=llm,
            tools=tools,
            collector=collector,
            budget=budget,
            chat_context="",
        )
    elif mode == "normal":
        ans, dag = execute_normal_mode(
            query=query,
            llm=llm,
            tools=tools,
            planner=planner,
            executor=executor,
            collector=collector,
            budget=budget,
            chat_context="",
        )
    elif mode == "deep":
        ans, dag = execute_deep_mode(
            query=query,
            llm=llm,
            tools=tools,
            planner=planner,
            executor=executor,
            collector=collector,
            budget=budget,
            chat_context="",
        )
    elif mode == "academic":
        ans, dag = execute_academic_mode(
            query=query,
            llm=llm,
            tools=tools,
            planner=planner,
            executor=executor,
            collector=collector,
            budget=budget,
            chat_context="",
        )

    collector.finish()
    elapsed = time.perf_counter() - start_time

    m = collector.get_metrics()
    sources_count = len(dag.nodes) if dag else 0

    print("\n--- RESULTS ---")
    print(f"Label: {label}")
    print(f"Total Latency: {m.total_latency:.3f}s (Wall Clock: {elapsed:.3f}s)")
    print(f"Planner Latency: {m.planner_latency:.3f}s")
    print(f"Synthesis Latency: {m.final_synthesis_latency:.3f}s")
    print(f"LLM Calls: {m.llm_call_count}")
    print(f"Tokens: Total={m.total_tokens} (Input={m.input_tokens}, Output={m.output_tokens})")
    print(f"Tool Calls: {m.tool_call_count} ({m.tool_names})")
    print(f"Browser Calls: {m.browser_calls} (HTTP: {m.http_fetch_calls}, Playwright: {m.playwright_calls})")
    print(f"Retries: {m.retries} (Rate Limits: {m.rate_limit_errors})")
    print(f"DAG Nodes: {m.dag_node_count} (Completed: {m.completed_nodes}, Skipped: {m.skipped_nodes})")
    print(f"Sources/Citations Count: {sources_count}")
    print(f"Answer Length: {len(ans)} chars")

    return {
        "label": label,
        "query": query,
        "mode": mode,
        "metrics": m.to_dict(),
        "sources_count": sources_count,
        "answer_chars": len(ans),
        "answer_preview": ans[:300] + "..." if len(ans) > 300 else ans,
    }


def main():
    benchmarks = [
        ("What is the square root of 144 plus 25 multiplied by 4?", "fast", "1. Simple Factual Question"),
        ("What is the current stock price and market cap of Apple (AAPL) and Microsoft (MSFT)?", "normal", "2. Current Factual Question"),
        ("Compare the architecture, parameters, context window, and benchmark performance of Llama 3, Mistral Large, and Claude 3.5 Sonnet", "normal", "3. Comparison Question"),
        ("Analyze the mechanisms and optimizations of FlashAttention-2 vs FlashAttention-3 GPU kernels and memory access patterns", "deep", "4. Deep Technical Research Question"),
        ("Survey recent arXiv and PubMed advances in diffusion models for biomedical image segmentation", "academic", "5. Academic Research Question"),
    ]

    all_results = []
    for query, mode, label in benchmarks:
        try:
            res = run_single_benchmark(query, mode, label)
            all_results.append(res)
        except Exception as e:
            print(f"ERROR executing benchmark {label}: {e}")
            import traceback
            traceback.print_exc()
        time.sleep(3)  # brief pause to avoid bursting free tier

    with open("benchmark_results_optimized.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\nAll 5 benchmarks completed! Results saved to benchmark_results_optimized.json")


if __name__ == "__main__":
    main()
