from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from budget import QueryBudget
from dag.models import DAGNode, ResearchDAG
from dag.planner import DAGPlanner
from dag.executor import DAGExecutor
from evidence import EvidenceStore, build_research_packet
from metrics import MetricsCollector, log_event
from utils.tool_router import classify_query_tool_group, select_tools_for_task


# --------------------------------------------------
# 1. FAST MODE STRATEGY
# --------------------------------------------------

def execute_fast_mode(
    query: str,
    llm: Any,
    tools: Optional[List[Any]] = None,
    collector: Optional[MetricsCollector] = None,
    budget: Optional[QueryBudget] = None,
    chat_context: str = "",
) -> Tuple[str, ResearchDAG]:
    """
    FAST MODE PIPELINE:
    query -> direct tool (max 1-2 calls, 0 LLM) -> single synthesis call.
    Target: 1 LLM call maximum, <= 2 tool calls, sub-second to 2s execution.
    """
    col = collector
    bg = budget
    req_id = col.metrics.request_id if col else ""
    dag = ResearchDAG()  # Empty DAG (0 nodes)

    # 1. Direct tool execution in Python (0 LLM calls)
    group = classify_query_tool_group(query)
    direct_tools = select_tools_for_task(task_type="tool", query=query, all_tools=tools)

    tool_outputs = []
    tools_used = []

    # Execute at most 1 primary tool directly
    if direct_tools:
        primary_tool = direct_tools[0]
        tool_name = getattr(primary_tool, "name", str(primary_tool))
        try:
            if bg and bg.reserve("tool"):
                bg.record_usage(tool_calls=1)
                if col:
                    col.record_tool_call(tool_name=tool_name)
                # Invoke tool
                if hasattr(primary_tool, "invoke"):
                    raw_out = primary_tool.invoke({"query": query} if "query" in getattr(primary_tool, "args", {}) else query)
                elif callable(primary_tool):
                    raw_out = primary_tool(query)
                else:
                    raw_out = ""
                tools_used.append(tool_name)
                tool_outputs.append(str(raw_out))
        except Exception as e:
            log_event("fast_mode_tool_error", request_id=req_id, error=str(e))

    # 2. Build compact evidence
    evidence_store = EvidenceStore()
    for out in tool_outputs:
        evidence_store.add_from_raw(out, node_id="fast_direct")

    packet = build_research_packet(question=query, store=evidence_store, max_sources=3, max_findings=6)
    evidence_text = packet.to_markdown()

    # 3. Single LLM Synthesis Call (1 LLM call)
    context_block = f"\nPrior Conversation Context:\n{chat_context}\n" if chat_context else ""
    synthesis_prompt = f"""
You are an intelligent, ultra-fast research assistant.
Provide a crisp, clear, direct answer to the user's question.
{context_block}
User Query:
"{query}"

Extracted Research Findings:
{evidence_text}

Instructions:
- Be concise and direct (1-3 focused paragraphs or bullet points).
- Answer the query immediately without introductory filler.
"""

    answer = ""
    if llm:
        try:
            if bg:
                bg.reserve("llm")
            synth_start = time.perf_counter()
            resp = llm.invoke(synthesis_prompt)
            synth_latency = time.perf_counter() - synth_start
            answer = getattr(resp, "content", str(resp)).strip()

            in_tokens = max(1, len(synthesis_prompt) // 4)
            out_tokens = max(1, len(answer) // 4)

            if col:
                col.record_synthesis(
                    latency=synth_latency,
                    context_chars=len(synthesis_prompt),
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )
            if bg:
                bg.record_usage(llm_calls=1, input_tokens=in_tokens, output_tokens=out_tokens)
        except Exception as e:
            answer = f"Fast synthesis encountered an error: {e}"

    return answer, dag


# --------------------------------------------------
# 2. NORMAL MODE STRATEGY
# --------------------------------------------------

def execute_normal_mode(
    query: str,
    llm: Any,
    tools: Optional[List[Any]] = None,
    planner: Optional[DAGPlanner] = None,
    executor: Optional[DAGExecutor] = None,
    collector: Optional[MetricsCollector] = None,
    budget: Optional[QueryBudget] = None,
    chat_context: str = "",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[str, ResearchDAG]:
    """
    NORMAL MODE PIPELINE:
    query -> lightweight planner (1 LLM) -> 1-3 parallel deterministic tools (0 LLM)
          -> evidence compaction -> 1 final synthesis call.
    Target: 2-3 LLM calls total.
    """
    col = collector
    bg = budget
    p = planner or DAGPlanner(llm=llm)
    e = executor or DAGExecutor(llm=llm, tools=tools)

    if progress_callback:
        progress_callback("Planning research DAG...", 0.1)

    # 1. Planner (1 LLM call)
    dag = p.create_dag(query, mode="normal", context=chat_context, metrics=col, budget=bg)

    if progress_callback:
        progress_callback(f"Executing {len(dag.nodes)} research tasks...", 0.3)

    # 2. Execute parallel deterministic tasks (0 LLM calls for tool tasks)
    dag = e.execute(dag, progress_callback=progress_callback, metrics=col, budget=bg)

    if progress_callback:
        progress_callback("Compacting evidence & synthesizing findings...", 0.8)

    # 3. Evidence Compaction
    evidence_store = EvidenceStore()
    for nid, node in dag.nodes.items():
        if node.result:
            evidence_store.add_from_raw(node.result, node_id=nid)

    packet = build_research_packet(question=query, store=evidence_store, max_sources=6, max_findings=10)
    evidence_text = packet.to_markdown()
    if not evidence_text or len(evidence_text.strip()) < 40:
        fallback_nodes = "\n\n".join([f"### Source Findings ({nid})\n{n.result}" for nid, n in dag.nodes.items() if n.result])
        evidence_text = fallback_nodes or "No evidence retrieved."

    # 4. Final Synthesis Call (1 LLM call)
    context_block = f"\nPrior Conversation Context:\n{chat_context}\n" if chat_context else ""
    synthesis_prompt = f"""
You are an intelligent AI research assistant synthesizing research findings.
{context_block}
User Query:
"{query}"

Compiled Research Evidence:
{evidence_text}

Instructions:
- Provide a clear, comprehensive breakdown addressing the query.
- Use clean markdown formatting with citations.
"""

    answer = ""
    if llm:
        try:
            if bg:
                bg.reserve("llm")
            synth_start = time.perf_counter()
            resp = llm.invoke(synthesis_prompt)
            synth_latency = time.perf_counter() - synth_start

            if hasattr(resp, "content"):
                if isinstance(resp.content, list):
                    answer = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in resp.content]).strip()
                else:
                    answer = str(resp.content).strip()
            else:
                answer = str(resp).strip()

            in_tokens = max(1, len(synthesis_prompt) // 4)
            out_tokens = max(1, len(answer) // 4)

            if col:
                col.record_synthesis(
                    latency=synth_latency,
                    context_chars=len(synthesis_prompt),
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )
            if bg:
                bg.record_usage(llm_calls=1, input_tokens=in_tokens, output_tokens=out_tokens)
        except Exception as e:
            answer = f"Synthesis encountered an error: {e}"

    if progress_callback:
        progress_callback("Completed!", 1.0)

    return answer, dag


# --------------------------------------------------
# 3. DEEP MODE STRATEGY
# --------------------------------------------------

def execute_deep_mode(
    query: str,
    llm: Any,
    tools: Optional[List[Any]] = None,
    planner: Optional[DAGPlanner] = None,
    executor: Optional[DAGExecutor] = None,
    collector: Optional[MetricsCollector] = None,
    budget: Optional[QueryBudget] = None,
    chat_context: str = "",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[str, ResearchDAG]:
    """
    DEEP MODE PIPELINE:
    query -> planner (1 LLM) -> 2-4 research branches (0-1 LLM each)
          -> evidence merge -> coverage check -> optional focused follow-up (0-1 LLM)
          -> final comprehensive synthesis (1 LLM).
    Target: 3-4 LLM calls total.
    """
    col = collector
    bg = budget
    p = planner or DAGPlanner(llm=llm)
    e = executor or DAGExecutor(llm=llm, tools=tools)

    if progress_callback:
        progress_callback("Analyzing & planning multi-branch deep research DAG...", 0.1)

    # 1. Multi-branch planner (1 LLM call)
    dag = p.create_dag(query, mode="deep", context=chat_context, metrics=col, budget=bg)

    if progress_callback:
        progress_callback(f"Executing {len(dag.nodes)} deep research branches...", 0.3)

    # 2. Parallel research branch execution
    dag = e.execute(dag, progress_callback=progress_callback, metrics=col, budget=bg)

    # 3. Evidence Merge & Coverage Check
    evidence_store = EvidenceStore()
    for nid, node in dag.nodes.items():
        if node.result:
            evidence_store.add_from_raw(node.result, node_id=nid)

    # 4. Coverage Check & Optional Follow-up (Max 1 focused follow-up if budget permits)
    if bg and bg.can_call_llm() and not bg.should_skip_optional_work() and len(evidence_store) < 4:
        try:
            if progress_callback:
                progress_callback("Running focused coverage follow-up...", 0.6)
            followup_tool = select_tools_for_task(task_type="tool", query=query, all_tools=tools)
            if followup_tool and bg.reserve("tool"):
                t = followup_tool[0]
                t_name = getattr(t, "name", str(t))
                if hasattr(t, "invoke"):
                    follow_out = t.invoke(query)
                elif callable(t):
                    follow_out = t(query)
                else:
                    follow_out = ""
                if col:
                    col.record_tool_call(tool_name=t_name)
                bg.record_usage(tool_calls=1)
                evidence_store.add_from_raw(str(follow_out), node_id="followup_coverage")
        except Exception:
            pass

    if progress_callback:
        progress_callback("Synthesizing comprehensive deep report...", 0.85)

    # 5. Build Deep ResearchPacket
    packet = build_research_packet(question=query, store=evidence_store, max_sources=8, max_findings=12)
    evidence_text = packet.to_markdown()

    # 6. Comprehensive Synthesis Call (1 LLM call)
    context_block = f"\nPrior Conversation Context:\n{chat_context}\n" if chat_context else ""
    synthesis_prompt = f"""
You are an expert deep research analyst. Synthesize the findings into an exhaustive, structured report.
{context_block}
User Query:
"{query}"

Deep Research Evidence:
{evidence_text}

Instructions:
- Provide an in-depth, structured report with executive summary, key insights, comparative analysis, and citations.
- Clearly present data points, tables if applicable, and reference links.
"""

    answer = ""
    if llm:
        try:
            if bg:
                bg.reserve("llm")
            synth_start = time.perf_counter()
            resp = llm.invoke(synthesis_prompt)
            synth_latency = time.perf_counter() - synth_start
            answer = getattr(resp, "content", str(resp)).strip()

            in_tokens = max(1, len(synthesis_prompt) // 4)
            out_tokens = max(1, len(answer) // 4)

            if col:
                col.record_synthesis(
                    latency=synth_latency,
                    context_chars=len(synthesis_prompt),
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )
            if bg:
                bg.record_usage(llm_calls=1, input_tokens=in_tokens, output_tokens=out_tokens)
        except Exception as e:
            answer = f"Deep synthesis encountered an error: {e}"

    if progress_callback:
        progress_callback("Completed!", 1.0)

    return answer, dag


# --------------------------------------------------
# 4. ACADEMIC MODE STRATEGY
# --------------------------------------------------

def execute_academic_mode(
    query: str,
    llm: Any,
    tools: Optional[List[Any]] = None,
    planner: Optional[DAGPlanner] = None,
    executor: Optional[DAGExecutor] = None,
    collector: Optional[MetricsCollector] = None,
    budget: Optional[QueryBudget] = None,
    chat_context: str = "",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[str, ResearchDAG]:
    """
    ACADEMIC MODE PIPELINE:
    query -> academic planner (1 LLM) -> academic sources (arXiv, PubMed, academic search)
          -> metadata-first retrieval (DOIs, abstracts) -> deduplication -> academic synthesis.
    Target: 2-4 LLM calls.
    """
    col = collector
    bg = budget
    p = planner or DAGPlanner(llm=llm)
    e = executor or DAGExecutor(llm=llm, tools=tools)

    if progress_callback:
        progress_callback("Planning academic literature review DAG...", 0.1)

    # 1. Academic Planner
    dag = p.create_dag(query, mode="academic", context=chat_context, metrics=col, budget=bg)

    if progress_callback:
        progress_callback(f"Retrieving academic literature from {len(dag.nodes)} sources...", 0.3)

    # 2. Execute academic research nodes
    dag = e.execute(dag, progress_callback=progress_callback, metrics=col, budget=bg)

    # 3. Metadata-first evidence collection & deduplication
    evidence_store = EvidenceStore()
    for nid, node in dag.nodes.items():
        if node.result:
            evidence_store.add_from_raw(node.result, node_id=nid, source_type="academic")

    packet = build_research_packet(question=query, store=evidence_store, max_sources=8, max_findings=12)
    evidence_text = packet.to_markdown()
    if not evidence_text or len(evidence_text.strip()) < 40:
        fallback_nodes = "\n\n".join([f"### Academic Source ({nid})\n{n.result}" for nid, n in dag.nodes.items() if n.result])
        evidence_text = fallback_nodes or "No academic evidence retrieved."

    # 4. Academic Literature Synthesis Call (1 LLM call)
    context_block = f"\nPrior Conversation Context:\n{chat_context}\n" if chat_context else ""
    synthesis_prompt = f"""
You are a senior academic researcher and peer reviewer.
Provide a rigorous literature review and synthesis of current academic papers.
{context_block}
Research Topic:
"{query}"

Academic Literature & Paper Findings:
{evidence_text}

Instructions:
- Provide academic depth: summarize core methodology, empirical results, and paper findings.
- Explicitly cite referenced papers and preprints.
- Highlight consensus, open questions, and methodological limitations.
"""

    answer = ""
    if llm:
        try:
            if bg:
                bg.reserve("llm")
            synth_start = time.perf_counter()
            resp = llm.invoke(synthesis_prompt)
            synth_latency = time.perf_counter() - synth_start

            if hasattr(resp, "content"):
                if isinstance(resp.content, list):
                    answer = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in resp.content]).strip()
                else:
                    answer = str(resp.content).strip()
            else:
                answer = str(resp).strip()

            in_tokens = max(1, len(synthesis_prompt) // 4)
            out_tokens = max(1, len(answer) // 4)

            if col:
                col.record_synthesis(
                    latency=synth_latency,
                    context_chars=len(synthesis_prompt),
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )
            if bg:
                bg.record_usage(llm_calls=1, input_tokens=in_tokens, output_tokens=out_tokens)
        except Exception as e:
            answer = f"Academic synthesis encountered an error: {e}"

    if progress_callback:
        progress_callback("Completed!", 1.0)

    return answer, dag
