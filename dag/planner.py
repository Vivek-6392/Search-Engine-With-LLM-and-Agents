import json
import re
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from budget import get_current_budget
from dag.models import DAGNode, ResearchDAG
from dag.validator import validate_and_sanitize_dag
from metrics import get_current_metrics, log_event


class DAGPlanner:

    def __init__(self, llm):
        self.llm = llm

    def create_dag(
        self,
        query: str,
        mode: str = "deep",
        context: str = "",
        metrics=None,
        budget=None,
    ) -> ResearchDAG:
        start_time = time.perf_counter()
        collector = metrics or get_current_metrics()
        req_id = collector.metrics.request_id if collector else ""
        bg = budget or get_current_budget()

        # FAST mode: Bypass DAG planning completely (0ms planning overhead)
        if mode == "fast" or (bg and not bg.can_call_llm(estimated_tokens=500)):
            dag = ResearchDAG()
            elapsed = time.perf_counter() - start_time
            if collector:
                collector.record_planner(latency=elapsed, node_count=0)
            log_event(
                "planner_finished",
                request_id=req_id,
                mode=mode,
                node_count=0,
                latency_seconds=round(elapsed, 4),
            )
            return dag

        if mode == "normal":
            max_nodes = 3
            mode_instructions = """
Mode: NORMAL RESEARCH (Minimum Sufficient Decomposition)
1. Decompose ONLY into 1 to 3 genuinely independent research tasks (e.g. 1 task per entity or question).
2. All independent tasks MUST have `dependencies: []` to run simultaneously.
3. NEVER create separate synthesis, summarization, or formatting nodes.
"""
        elif mode == "academic":
            max_nodes = 4
            mode_instructions = """
Mode: ACADEMIC & SCIENTIFIC (Max 4 Nodes)
1. Decompose into 2 to 4 focused scientific queries covering literature, papers, and key concepts.
2. Maximize parallelism: All independent paper searches MUST have `dependencies: []`.
3. NEVER create separate synthesis or formatting-only nodes.
"""
        else:  # deep
            max_nodes = 4
            mode_instructions = """
Mode: DEEP RESEARCH (Max 4 Nodes)
1. PARALLEL DECOMPOSITION: If query compares multiple entities/metrics, create 1 independent node (`dependencies: []`) per entity (e.g. Node 1 for Company A, Node 2 for Company B).
2. Max 4 nodes total.
3. NEVER create separate formatting, summarization, or final synthesis nodes (the global synthesis engine automatically combines all findings).
"""

        context_section = f"\nPrior Conversation Context:\n{context.strip()}\n" if context and context.strip() else ""

        prompt = f"""
You are a DAG research task planner adhering strictly to MINIMUM SUFFICIENT DECOMPOSITION.

Break the following user query into a minimal Directed Acyclic Graph (DAG) of research tasks.
{context_section}
{mode_instructions}

Graph Construction Rules:
1. Create up to {max_nodes} research nodes total.
2. Select `task_type`:
   - Use `"task_type": "tool"` (with `"tool": "web_search"|"calculator"|"finance"|"wikipedia"|"arxiv"|"pubmed"|"weather"|"github"` and `"query": "..."`) for direct factual retrieval/search tasks.
   - Use `"task_type": "llm"` only when dynamic reasoning or multi-aspect analysis is needed.
3. MAXIMIZE CONCURRENCY: All independent data-fetching tasks MUST have `"dependencies": []`.
4. Do NOT create duplicate tasks.
5. Do NOT create formatting-only, table-formatting, or trivial summarization nodes.
6. Do NOT create a final synthesis node (global synthesis executes automatically afterwards).
7. Maximum depth is 3. Maximum parent fan-out is 4.
8. Return ONLY valid JSON matching this schema:
{{
    "nodes": [
        {{
            "id": "node_1",
            "task_type": "tool",
            "tool": "web_search",
            "query": "Apple AAPL stock price and market cap",
            "dependencies": []
        }},
        {{
            "id": "node_2",
            "task_type": "tool",
            "tool": "web_search",
            "query": "Microsoft MSFT stock price and market cap",
            "dependencies": []
        }}
    ]
}}

User query:
{query}
"""

        system_prompt = (
            "You are a research task planner following minimum sufficient decomposition. "
            "Output ONLY a valid JSON object defining the research DAG. "
            "Do NOT call tools, search functions, or output markdown explanation."
        )

        prompt_total_chars = len(system_prompt) + len(prompt)
        try:
            llm_call_start = time.perf_counter()
            try:
                response = self.llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=prompt),
                    ]
                )
            except Exception:
                response = self.llm.invoke(prompt)

            llm_latency = time.perf_counter() - llm_call_start

            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            in_tokens = 0
            out_tokens = 0
            if hasattr(response, "usage_metadata") and isinstance(response.usage_metadata, dict):
                in_tokens = response.usage_metadata.get("input_tokens", 0)
                out_tokens = response.usage_metadata.get("output_tokens", 0)
            elif hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
                tu = response.response_metadata.get("token_usage", {})
                if isinstance(tu, dict):
                    in_tokens = tu.get("prompt_tokens", 0)
                    out_tokens = tu.get("completion_tokens", 0)

            if collector:
                collector.record_llm_call(
                    stage="planner",
                    prompt_chars=prompt_total_chars,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    latency_seconds=llm_latency,
                )

            if bg:
                bg.record_usage(
                    llm_calls=1,
                    input_tokens=in_tokens or max(1, prompt_total_chars // 4),
                    output_tokens=out_tokens or max(1, len(content) // 4),
                )

            data = self._extract_json(content)
            dag = self._build_dag(data, max_nodes=max_nodes)
            elapsed = time.perf_counter() - start_time
            if collector:
                collector.record_planner(latency=elapsed, node_count=len(dag.nodes))

            log_event(
                "planner_finished",
                request_id=req_id,
                mode=mode,
                node_count=len(dag.nodes),
                latency_seconds=round(elapsed, 4),
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
            return dag

        except Exception as e:
            # Safe fallback: one-node DAG
            dag = ResearchDAG()
            dag.add_node(
                DAGNode(
                    id="node_1",
                    task=query,
                    dependencies=[]
                )
            )

            elapsed = time.perf_counter() - start_time
            if collector:
                collector.record_planner(latency=elapsed, node_count=len(dag.nodes))

            log_event(
                "planner_fallback",
                request_id=req_id,
                mode=mode,
                error=str(e),
                node_count=len(dag.nodes),
                latency_seconds=round(elapsed, 4),
            )

            return dag

    def _extract_json(self, content: str) -> dict:
        content = content.strip()

        # Remove markdown code fences if present
        content = re.sub(
            r"^```(?:json)?",
            "",
            content,
            flags=re.IGNORECASE
        )

        content = re.sub(
            r"```$",
            "",
            content
        )

        content = content.strip()

        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON found in LLM response")

        json_string = content[start:end + 1]

        return json.loads(json_string)

    def _build_dag(self, data: dict, max_nodes: int = 4) -> ResearchDAG:
        dag = ResearchDAG()
        nodes = data.get("nodes", [])

        if not nodes:
            raise ValueError("No nodes generated")

        for node_data in nodes:
            node_id = str(node_data.get("id", "")).strip()
            task = str(node_data.get("task") or node_data.get("query") or "").strip()
            task_type = str(node_data.get("task_type") or ("tool" if node_data.get("tool") else "llm")).strip()
            tool_name = str(node_data.get("tool") or node_data.get("tool_name") or "").strip()
            query_param = str(node_data.get("query") or task).strip()
            tool_params = dict(node_data.get("tool_params") or {})
            if "query" not in tool_params and query_param:
                tool_params["query"] = query_param
            deps = node_data.get("dependencies", [])
            if not isinstance(deps, list):
                deps = []
            if node_id and task:
                dag.add_node(
                    DAGNode(
                        id=node_id,
                        task=task,
                        dependencies=deps,
                        task_type=task_type,
                        tool_name=tool_name,
                        tool_params=tool_params,
                    )
                )

        # Apply comprehensive post-planning validation & sanitization
        validated_dag = validate_and_sanitize_dag(
            dag,
            max_nodes=max_nodes,
            max_depth=3,
            max_fan_out=4,
        )

        if not validated_dag.nodes:
            raise ValueError("Could not create valid DAG")

        return validated_dag