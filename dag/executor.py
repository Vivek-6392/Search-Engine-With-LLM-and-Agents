import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dag.models import ResearchDAG


class DAGExecutor:

    def __init__(self, agent, max_workers: int = 4):
        self.agent = agent
        self.max_workers = max_workers

    def execute(
        self,
        dag: ResearchDAG,
        progress_callback=None
    ):

        while not dag.is_complete():

            ready_nodes = dag.get_ready_nodes()

            if not ready_nodes:

                # Prevent an infinite loop if
                # the generated DAG is invalid.
                for node in dag.nodes.values():

                    if node.status == "PENDING":
                        node.status = "FAILED"
                        node.error = (
                            "No executable dependency path found"
                        )

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

                    if progress_callback:
                        progress_callback(
                            node,
                            "RUNNING"
                        )

                    future = executor.submit(
                        self._execute_node,
                        node,
                        dag
                    )

                    futures[future] = node

                for future in as_completed(futures):

                    node = futures[future]

                    try:

                        result, tools_used = future.result()

                        node.result = result
                        node.tool_used = ", ".join(tools_used) if tools_used else "direct answer"
                        node.status = "COMPLETED"

                        if progress_callback:
                            progress_callback(
                                node,
                                "COMPLETED"
                            )

                    except Exception as e:

                        node.status = "FAILED"
                        node.error = str(e)

                        if progress_callback:
                            progress_callback(
                                node,
                                "FAILED"
                            )

        return dag

    def _execute_node(
        self,
        node,
        dag: ResearchDAG
    ) -> str:

        dependency_context = []

        for dependency_id in node.dependencies:

            dependency = dag.nodes.get(
                dependency_id
            )

            if (
                dependency
                and dependency.result
            ):

                dep_text = dependency.result.strip()
                # Truncate large dependency outputs to stay within LLM token limits
                if len(dep_text) > 1800:
                    dep_text = dep_text[:1800] + "\n...[truncated for length]"

                dependency_context.append(
                    f"""
Result from {dependency.id}:

{dep_text}
"""
                )

        context = "\n".join(
            dependency_context
        )

        if len(context) > 3500:
            context = context[:3500] + "\n...[context truncated for token limits]"

        task_prompt = f"""
You are executing one node of a research DAG.

Your task:
{node.task}

Context from completed dependency nodes:
{context if context else "No dependency context."}

Instructions:
- If the dependency context above already provides the information needed, synthesize and formulate your findings directly without extra search.
- If you need fresh data or computation, make 1 focused tool call (e.g. calculator_tool, web_search_tool, github_search_tool, finance_tool, pubmed_tool, arxiv_tool, wikipedia_tool).
- After receiving tool observations, immediately formulate and return your comprehensive findings, key facts, and URLs.
"""

        max_retries = 3
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self.agent.invoke(
                    {
                        "input": task_prompt
                    }
                )

                output_text = ""
                tools_used = []

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
                else:
                    output_text = str(response)

                return output_text, tools_used

            except Exception as e:
                last_exception = e
                err_msg = str(e).lower()
                # If rate limit or token limit exceeded, wait and retry
                if any(x in err_msg for x in ["rate_limit", "429", "413", "tokens", "tpm"]) and attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise last_exception