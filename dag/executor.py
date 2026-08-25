from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from browser import browse_webpage
from budget import get_current_budget
from dag.models import DAGNode, ResearchDAG
from evidence import EvidenceStore
from metrics import get_current_metrics, log_event, MetricsCallbackHandler
from utils.rate_limit import (
    RateLimitType,
    classify_llm_error,
    calculate_backoff_delay,
    compress_prompt_for_context_limit,
)
from utils.tool_router import select_tools_for_task
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


DIRECT_TOOL_MAP = {
    "calculator": python_calculator,
    "python_calculator": python_calculator,
    "calculator_tool": python_calculator,
    "finance": search_finance,
    "search_finance": search_finance,
    "finance_tool": search_finance,
    "stock": search_finance,
    "stocks": search_finance,
    "crypto": search_finance,
    "github": search_github,
    "github_search": search_github,
    "github_search_tool": search_github,
    "pubmed": search_pubmed,
    "pubmed_tool": search_pubmed,
    "search_pubmed": search_pubmed,
    "wikipedia": search_wikipedia,
    "wikipedia_tool": search_wikipedia,
    "search_wikipedia": search_wikipedia,
    "weather": search_weather,
    "weather_tool": search_weather,
    "search_weather": search_weather,
    "academic_papers": search_academic_papers,
    "academic_papers_tool": search_academic_papers,
    "arxiv": search_academic_papers,
    "arxiv_tool": search_academic_papers,
    "hackernews": search_hackernews,
    "hackernews_tool": search_hackernews,
    "stackoverflow": search_stackoverflow,
    "stackoverflow_tool": search_stackoverflow,
    "huggingface": search_huggingface,
    "huggingface_tool": search_huggingface,
    "package": lookup_package,
    "package_lookup": lookup_package,
    "forex": convert_forex,
    "browser": browse_webpage,
    "web_browser": browse_webpage,
    "web_browser_tool": browse_webpage,
}


from utils.search_pipeline import search_and_extract_evidence


def _run_web_search(query: str, max_results: int = 3, request_id: Optional[str] = None) -> str:
    """Optimized web search with deduplication, ranking, and compact evidence extraction."""
    evidence_list = search_and_extract_evidence(query=query, max_results=max_results, request_id=request_id)
    if not evidence_list:
        return f"No results found for query: '{query.strip()}'"
    return "\n\n".join([ev.to_markdown() for ev in evidence_list])


def build_agent_executor(
    llm,
    tools,
    max_iterations: int = 1,
    callbacks: list = None,
) -> AgentExecutor:
    """Factory to construct a bounded AgentExecutor (max_iterations=1 by default)."""
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert AI research agent equipped with specialized domain tools: "
                "(web_search_tool, wikipedia_tool, arxiv_tool, academic_papers_tool, pubmed_tool, "
                "github_search_tool, huggingface_tool, stackoverflow_tool, hackernews_tool, "
                "package_lookup_tool, finance_tool, forex_tool, weather_tool, calculator_tool, web_browser_tool).\n\n"
                "RULES:\n"
                "- If you need fresh external data, make exactly 1 focused tool call.\n"
                "- If the question or dependency context already contains the answer, answer directly without tools.\n"
                "- Always return accurate, factual findings.",
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

    return AgentExecutor(
        agent=agent_runnable,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        return_intermediate_steps=True,
        callbacks=callbacks,
    )


class DAGExecutor:

    def __init__(
        self,
        llm=None,
        tools=None,
        agent=None,
        max_workers: int = 6,
        metrics=None,
        budget=None,
        use_legacy_agent: bool = False,
    ):
        self.llm = llm
        self.tools = tools or []
        self.agent = agent
        self.max_workers = max_workers
        self.metrics = metrics
        self.budget = budget
        self.use_legacy_agent = use_legacy_agent

    def _get_agent(self, collector=None, max_iterations: int = 1, tools: Optional[List[Any]] = None):
        callbacks = [MetricsCallbackHandler(collector, stage="node_agent")] if collector else None
        active_tools = tools if tools is not None else self.tools
        if self.llm is not None:
            return build_agent_executor(self.llm, active_tools, max_iterations=max_iterations, callbacks=callbacks)
        if self.agent is not None:
            return self.agent
        raise ValueError("DAGExecutor requires (llm, tools) or an agent instance")

    def execute(
        self,
        dag: ResearchDAG,
        progress_callback=None,
        metrics=None,
        budget=None,
    ):
        collector = metrics or self.metrics or get_current_metrics()
        bg = budget or self.budget or get_current_budget()
        req_id = collector.metrics.request_id if collector else ""

        while not dag.is_complete():
            # Check if budget is exhausted or critical headroom is depleted
            if bg and (bg.exhausted() or bg.should_skip_optional_work()):
                for node in dag.nodes.values():
                    if node.status in ("PENDING", "READY"):
                        node.status = "SKIPPED"
                        node.error = "Skipped due to query budget constraints"
                        if collector:
                            collector.record_node_end(
                                node_id=node.id,
                                status="SKIPPED",
                                error="Budget constraints",
                            )
                        log_event(
                            "node_skipped",
                            request_id=req_id,
                            node_id=node.id,
                            reason="budget_exhausted",
                        )
                        if progress_callback:
                            progress_callback(node, "SKIPPED")
                break

            ready_nodes = dag.get_ready_nodes()

            if not ready_nodes:
                for node in dag.nodes.values():
                    if node.status == "PENDING":
                        node.status = "FAILED"
                        node.error = "No executable dependency path found"
                break

            with ThreadPoolExecutor(
                max_workers=min(
                    self.max_workers,
                    len(ready_nodes)
                )
            ) as executor:

                futures = {}

                for node in ready_nodes:
                    node.status = "RUNNING"
                    if collector:
                        collector.record_node_start(node.id, task=node.task)
                    log_event(
                        "node_started",
                        request_id=req_id,
                        node_id=node.id,
                        task=node.task,
                    )

                    if progress_callback:
                        progress_callback(
                            node,
                            "RUNNING"
                        )

                    future = executor.submit(
                        self._execute_node,
                        node,
                        dag,
                        collector=collector,
                        budget=bg,
                    )

                    futures[future] = node

                for future in as_completed(futures):
                    node = futures[future]

                    try:
                        result, tools_used = future.result()

                        node.result = result
                        tool_str = ", ".join(tools_used) if tools_used else "direct answer"
                        node.tool_used = tool_str
                        node.status = "COMPLETED"

                        if collector:
                            collector.record_node_end(
                                node_id=node.id,
                                status="COMPLETED",
                                tool_used=tool_str,
                            )
                        log_event(
                            "node_completed",
                            request_id=req_id,
                            node_id=node.id,
                            tool_used=tool_str,
                        )

                        if progress_callback:
                            progress_callback(
                                node,
                                "COMPLETED"
                            )

                    except Exception as e:
                        node.status = "FAILED"
                        node.error = str(e)

                        if collector:
                            collector.record_node_end(
                                node_id=node.id,
                                status="FAILED",
                                error=str(e),
                            )
                        log_event(
                            "node_failed",
                            request_id=req_id,
                            node_id=node.id,
                            error=str(e),
                        )

                        if progress_callback:
                            progress_callback(
                                node,
                                "FAILED"
                            )

        return dag

    def _execute_node(
        self,
        node: DAGNode,
        dag: ResearchDAG,
        collector=None,
        budget=None,
    ) -> Tuple[str, List[str]]:
        col = collector or self.metrics or get_current_metrics()
        bg = budget or self.budget or get_current_budget()

        # 1. Deterministic Tool Task (Zero LLM calls)
        if getattr(node, "task_type", "llm") == "tool":
            return self._execute_deterministic_tool_task(node, collector=col, budget=bg)

        # 2. Bounded LLM Research Task
        return self._execute_llm_research_task(node, dag, collector=col, budget=bg)

    def _execute_deterministic_tool_task(
        self,
        node: DAGNode,
        collector=None,
        budget=None,
    ) -> Tuple[str, List[str]]:
        """
        Execute tool directly in Python with zero extra LLM calls.
        """
        tool_name = (getattr(node, "tool_name", "") or "").lower().strip()
        params = dict(getattr(node, "tool_params", {}) or {})
        query = str(params.get("query", "") or node.task).strip()

        if not tool_name:
            # Heuristic tool detection from task string
            q_lower = query.lower()
            if any(w in q_lower for w in ["calculate", "sqrt", "math", "+", "*", "/", "square root"]):
                tool_name = "calculator"
            elif any(w in q_lower for w in ["stock", "market cap", "ticker", "share price", "finance"]):
                tool_name = "finance"
            elif any(w in q_lower for w in ["weather", "temperature", "forecast"]):
                tool_name = "weather"
            elif any(w in q_lower for w in ["pubmed", "clinical", "disease", "medicine"]):
                tool_name = "pubmed"
            elif any(w in q_lower for w in ["arxiv", "academic paper", "scientific paper"]):
                tool_name = "arxiv"
            elif any(w in q_lower for w in ["github", "repository", "repo"]):
                tool_name = "github"
            elif any(w in q_lower for w in ["browse", "http://", "https://"]):
                tool_name = "browser"
            else:
                tool_name = "web_search"

        if budget:
            if tool_name in ("browser", "web_browser", "web_browser_tool"):
                if not budget.reserve("browser"):
                    tool_name = "web_search"
            if not budget.reserve("tool"):
                return f"Tool budget exhausted for {tool_name}.", [tool_name]

        tool_func = DIRECT_TOOL_MAP.get(tool_name)
        tool_used_name = tool_name
        try:
            if tool_name in ("web_search", "search", "google"):
                max_res = int(params.get("max_results", 5))
                result = _run_web_search(query, max_results=max_res)
                tool_used_name = "web_search"
            elif tool_func:
                result = tool_func(query)
            else:
                result = _run_web_search(query)
                tool_used_name = "web_search"
        except Exception as e:
            result = f"Tool '{tool_name}' observation: {str(e)}"

        if collector:
            collector.record_tool_call(tool_name=tool_used_name)
        if budget:
            budget.record_usage(tool_calls=1)

        return str(result), [tool_used_name]

    def _execute_llm_research_task(
        self,
        node: DAGNode,
        dag: ResearchDAG,
        collector=None,
        budget=None,
    ) -> Tuple[str, List[str]]:
        """
        Bounded research task: Max 1 tool call + Max 1 LLM decision + optional 1 synthesis.
        """
        # Extract compact evidence claims from completed dependency nodes (no raw data dumps)
        parent_store = EvidenceStore()
        for dependency_id in node.dependencies:
            dependency = dag.nodes.get(dependency_id)
            if dependency and dependency.result:
                parent_store.add_from_raw(dependency.result, node_id=dependency.id)

        context = parent_store.to_markdown(max_items=8)
        if not parent_store:
            context = ""

        task_prompt = f"""
You are executing one node of a research DAG.

Your task:
{node.task}

Context from completed dependency nodes:
{context if context else "No dependency context."}

Instructions:
- If the dependency context already provides the needed facts, formulate your concise findings directly.
- Otherwise, make at most 1 focused tool call.
"""

        max_retries = 2
        last_exception = None
        col = collector or self.metrics or get_current_metrics()
        bg = budget or self.budget or get_current_budget()
        req_id = col.metrics.request_id if col else ""

        # Reserve initial LLM call
        if bg and not bg.reserve("llm"):
            fallback_text = context if context else f"Completed with existing evidence for: {node.task}"
            return fallback_text, ["direct answer"]

        # Task-specific tool routing: select 1-3 relevant tools instead of the full 15-tool suite
        routed_tools = select_tools_for_task(
            task_type=getattr(node, "task_type", None),
            query=node.task,
            tool_name=getattr(node, "tool_name", None),
            all_tools=self.tools if self.tools else None,
        )

        agent_instance = self._get_agent(
            collector=col,
            max_iterations=1,
            tools=routed_tools if (self.llm is not None and self.tools) else None,
        )

        for attempt in range(max_retries):
            try:
                response = agent_instance.invoke({"input": task_prompt})

                output_text = ""
                tools_used = []
                steps = []

                if isinstance(response, dict):
                    output_text = response.get("output", str(response))
                    steps = response.get("intermediate_steps", [])
                    for step in steps:
                        if isinstance(step, (list, tuple)) and len(step) > 0:
                            action = step[0]
                            t_name = getattr(action, "tool", "")
                            if t_name:
                                clean_name = t_name.replace("_tool", "").replace("_", " ")
                                if clean_name not in tools_used:
                                    tools_used.append(clean_name)
                                if col:
                                    col.record_tool_call(tool_name=t_name)

                    if ("agent stopped" in output_text.lower() or not output_text.strip()) and steps:
                        obs_texts = [str(step[1]) for step in steps if isinstance(step, (list, tuple)) and len(step) > 1]
                        if obs_texts:
                            output_text = "\n\n".join(obs_texts)
                else:
                    output_text = str(response)

                in_est = max(1, len(task_prompt) // 4)
                out_est = max(1, len(output_text) // 4)

                if col:
                    col.record_llm_call(
                        stage=f"node_{node.id}",
                        prompt_chars=len(task_prompt),
                        input_tokens=in_est,
                        output_tokens=out_est,
                    )

                if bg:
                    bg.record_usage(
                        llm_calls=1,
                        input_tokens=in_est,
                        output_tokens=out_est,
                        tool_calls=len(tools_used),
                    )

                # Optional Step 2: If a tool was executed and budget allows, synthesize the raw observation
                if steps and self.llm and bg and (bg.can_call_llm() and not bg.should_skip_optional_work()):
                    try:
                        bg.reserve("llm")
                        synth_prompt = (
                            f"Summarize these extracted tool observations into key factual points for '{node.task}':\n\n"
                            f"{output_text[:2500]}"
                        )
                        synth_resp = self.llm.invoke(synth_prompt)
                        s_text = getattr(synth_resp, "content", str(synth_resp))
                        if s_text and len(s_text.strip()) > 20:
                            output_text = str(s_text).strip()
                            s_in = max(1, len(synth_prompt) // 4)
                            s_out = max(1, len(output_text) // 4)
                            if col:
                                col.record_llm_call(
                                    stage=f"node_{node.id}_synth",
                                    prompt_chars=len(synth_prompt),
                                    input_tokens=s_in,
                                    output_tokens=s_out,
                                )
                            bg.record_usage(llm_calls=1, input_tokens=s_in, output_tokens=s_out)
                    except Exception:
                        pass  # Keep raw output if optional second call encounters an issue

                return output_text, tools_used

            except Exception as e:
                last_exception = e
                err_info = classify_llm_error(e)
                is_rate_limit = (err_info.error_type in (RateLimitType.RATE_LIMIT_429, RateLimitType.CONTEXT_LENGTH_413))

                if is_rate_limit and col:
                    col.record_rate_limit(stage=f"node_{node.id}", error_msg=str(e))

                # Handle context length exceeded (413): compress prompt context for retry
                if err_info.error_type == RateLimitType.CONTEXT_LENGTH_413:
                    task_prompt = compress_prompt_for_context_limit(task_prompt)

                backoff_delay = calculate_backoff_delay(
                    attempt=attempt,
                    retry_after=err_info.retry_after,
                    base_delay=0.6,
                    max_delay=6.0,
                )

                if col:
                    col.record_retry(
                        stage=f"node_{node.id}",
                        reason=str(e),
                        error_type=err_info.error_type.value,
                        wait_duration_seconds=backoff_delay,
                    )

                log_event(
                    "retry_attempted",
                    request_id=req_id,
                    node_id=node.id,
                    attempt=attempt + 1,
                    error_type=err_info.error_type.value,
                    retry_after=err_info.retry_after,
                    backoff_delay=backoff_delay,
                    error=str(e),
                )

                # Budget-aware retry gating
                if bg:
                    # Check if retry reservation is granted and sufficient time/tokens remain
                    if (
                        not bg.reserve("retry")
                        or not bg.can_call_llm()
                        or bg.exhausted()
                        or (bg.remaining_time() < backoff_delay)
                    ):
                        log_event(
                            "retry_blocked_by_budget",
                            request_id=req_id,
                            node_id=node.id,
                            attempt=attempt + 1,
                            remaining_time=bg.remaining_time() if bg else 0.0,
                        )
                        # Graceful partial research fallback instead of hanging
                        fallback_text = context if context else f"Completed with partial evidence for: {node.task}"
                        return fallback_text, ["partial fallback"]
                    bg.record_usage(retries=1)

                if attempt < max_retries - 1 and err_info.is_retryable:
                    time.sleep(backoff_delay)
                    continue

                # If non-retryable or retries exhausted, return partial evidence rather than crashing
                if context:
                    return context, ["partial fallback"]
                raise last_exception