import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

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
    format_and_clean_answer,
)
from metrics.collector import MetricsCollector
from langchain_core.messages import SystemMessage, HumanMessage

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
executor = DAGExecutor(llm=llm, tools=tools, max_workers=5)


def run_benchmark_query(query: str, mode: str, label: str):
    print(f"\n=======================================================")
    print(f"Running Benchmark: {label} (Mode: {mode})")
    print(f"Query: {query}")
    print(f"=======================================================")

    collector = MetricsCollector(
        research_mode=mode,
        model_provider=f"Groq / {model_name}",
    )
    collector.start()

    # Step A: Planner
    dag = planner.create_dag(query, mode=mode, context="", metrics=collector)

    # Step B: Executor
    dag = executor.execute(dag, metrics=collector)

    # Step C: Synthesis
    results = dag.get_results()
    combined_results = "\n\n".join(
        [f"### {node_id}\n\n{result}" for node_id, result in results.items()]
    )

    synth_system_prompt = (
        "You are an expert research synthesizer. All data gathering, web research, calculations, "
        "and fact extraction have already been completed by upstream agent workers.\n"
        "Your ONLY task is to write a final, comprehensive, well-structured Markdown answer strictly based on the provided findings.\n"
        "CRITICAL CONSTRAINTS:\n"
        "1. Do NOT call, invoke, or output any tool calls, function calls, or JSON actions.\n"
        "2. Output ONLY direct, fluent Markdown text with clean Markdown tables (`| ... |`)."
    )

    final_prompt = f"""
You are an intelligent AI research assistant synthesizing findings for a user query.

User Query:
"{query}"

Research Mode: {mode}

Research Findings from DAG nodes:
{combined_results}
"""

    synth_start = time.perf_counter()
    synth_chars = len(synth_system_prompt) + len(final_prompt)
    synth_resp = llm.invoke(
        [
            SystemMessage(content=synth_system_prompt),
            HumanMessage(content=final_prompt),
        ]
    )
    synth_latency = time.perf_counter() - synth_start

    in_tokens = 0
    out_tokens = 0
    if hasattr(synth_resp, "usage_metadata") and isinstance(synth_resp.usage_metadata, dict):
        in_tokens = synth_resp.usage_metadata.get("input_tokens", 0)
        out_tokens = synth_resp.usage_metadata.get("output_tokens", 0)
    elif hasattr(synth_resp, "response_metadata") and isinstance(synth_resp.response_metadata, dict):
        tu = synth_resp.response_metadata.get("token_usage", {})
        if isinstance(tu, dict):
            in_tokens = tu.get("prompt_tokens", 0)
            out_tokens = tu.get("completion_tokens", 0)

    collector.record_synthesis(
        latency=synth_latency,
        context_chars=synth_chars,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        model=model_name,
    )

    collector.finish()

    metrics = collector.get_metrics()
    print("\n--- RESULTS ---")
    print(f"Request ID: {metrics.request_id}")
    print(f"Total Latency: {metrics.total_latency:.3f}s")
    print(f"Planner Latency: {metrics.planner_latency:.3f}s")
    print(f"Synthesis Latency: {metrics.final_synthesis_latency:.3f}s")
    print(f"DAG Nodes: {metrics.dag_node_count} (Completed: {metrics.completed_nodes}, Failed: {metrics.failed_nodes})")
    print(f"Per-node Latencies: {metrics.per_node_latency}")
    print(f"Tool Calls: {metrics.tool_call_count} ({metrics.tool_names})")
    print(f"Browser Calls: {metrics.browser_calls} (HTTP: {metrics.http_fetch_calls}, Playwright: {metrics.playwright_calls})")
    print(f"LLM Calls: {metrics.llm_call_count}")
    print(f"Tokens: Total={metrics.total_tokens}, Input={metrics.input_tokens}, Output={metrics.output_tokens}")
    print(f"Retries: {metrics.retries}, Rate Limit Errors: {metrics.rate_limit_errors}")

    return {
        "label": label,
        "query": query,
        "mode": mode,
        "metrics": metrics.to_dict(),
    }


def main():
    benchmarks = [
        ("What is the square root of 144 plus 25 multiplied by 4?", "fast", "1. Simple Query"),
        ("What is the current stock price and market cap of Apple (AAPL) and Microsoft (MSFT)?", "deep", "2. Normal Research Query"),
        ("Compare the architecture, parameters, context window, and benchmark performance of Llama 3, Mistral Large, and Claude 3.5 Sonnet", "deep", "3. Deep Research Query"),
    ]

    all_results = []
    for query, mode, label in benchmarks:
        res = run_benchmark_query(query, mode, label)
        all_results.append(res)
        time.sleep(2)  # pause between benchmarks

    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\nAll benchmarks completed and saved to benchmark_results.json")


if __name__ == "__main__":
    main()
