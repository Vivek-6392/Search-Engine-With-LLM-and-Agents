import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from dag.models import DAGNode, ResearchDAG


class DAGPlanner:

    def __init__(self, llm):
        self.llm = llm

    def create_dag(self, query: str, mode: str = "deep", context: str = "") -> ResearchDAG:

        if mode == "fast":
            # Connected 2-node direct research DAG (0ms planning overhead)
            dag = ResearchDAG()
            dag.add_node(
                DAGNode(
                    id="node_1",
                    task=f"Retrieve primary facts / calculate: {query}",
                    dependencies=[],
                )
            )
            dag.add_node(
                DAGNode(
                    id="node_2",
                    task=f"Verify and formulate concise result for: {query}",
                    dependencies=["node_1"],
                )
            )
            return dag

        if mode == "academic":
            mode_instructions = """
Mode: ACADEMIC & SCIENTIFIC
1. Create 3 to 6 focused nodes covering scientific literature, ArXiv papers, academic journals, and foundational concepts.
2. Maximize parallelism: Split research across different concepts/papers into independent parallel nodes (dependencies: []) so they run simultaneously.
3. Use a final dependent node to compare and synthesize findings.
"""
            max_nodes = 6
        else:
            mode_instructions = """
Mode: DEEP RESEARCH & PARALLEL EXECUTION
1. PARALLEL DECOMPOSITION (CRITICAL): If the query asks about multiple entities, companies, cities, questions, metrics, or comparisons, create a separate INDEPENDENT research node with `dependencies: []` for EACH entity/subject (e.g. Node 1 for Company A, Node 2 for Company B, Node 3 for Company C).
2. DO NOT bundle multiple subjects into a single node.
3. DO NOT create artificial sequential chains (e.g. node_1 -> node_2 -> node_3 for "validate", "format table", "analyze").
4. FINAL SYNTHESIS/CALCULATION NODE: Create a single dependent node at the end that lists all the parallel research nodes in its `dependencies` to aggregate, compare, calculate, or synthesize the final result.
"""
            max_nodes = 6

        context_section = f"\nPrior Conversation Context:\n{context.strip()}\n" if context and context.strip() else ""

        prompt = f"""
You are a DAG research task planner designed for high-concurrency parallel execution.

Break the following user query into a Directed Acyclic Graph (DAG) of research tasks.
{context_section}
{mode_instructions}

Graph Construction Rules:
1. Create up to {max_nodes} nodes.
2. MAXIMIZE CONCURRENCY: All independent data-fetching / research tasks MUST have `"dependencies": []` so they run simultaneously in parallel.
3. ONLY aggregation, comparison, calculation, or synthesis tasks should have dependencies (listing the IDs of the parallel nodes they depend on).
4. If prior conversation context is provided above, incorporate it to resolve entity references or follow-up comparisons.
5. Return ONLY valid JSON matching this schema:
{{
    "nodes": [
        {{
            "id": "node_1",
            "task": "Retrieve stock price and market cap for Apple (AAPL)",
            "dependencies": []
        }},
        {{
            "id": "node_2",
            "task": "Retrieve stock price and market cap for Microsoft (MSFT)",
            "dependencies": []
        }},
        {{
            "id": "node_3",
            "task": "Retrieve stock price and market cap for Nvidia (NVDA)",
            "dependencies": []
        }},
        {{
            "id": "node_4",
            "task": "Calculate combined average valuation and compare findings",
            "dependencies": ["node_1", "node_2", "node_3"]
        }}
    ]
}}

User query:
{query}
"""

        system_prompt = (
            "You are a research task planner. Output ONLY a valid JSON object defining the research DAG. "
            "Do NOT call tools, search functions, or output tool call JSON."
        )

        try:
            try:
                response = self.llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=prompt),
                    ]
                )
            except Exception:
                response = self.llm.invoke(prompt)

            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            data = self._extract_json(content)
            return self._build_dag(data, max_nodes=max_nodes)

        except Exception:
            # Safe fallback: one-node DAG
            dag = ResearchDAG()

            dag.add_node(
                DAGNode(
                    id="node_1",
                    task=query,
                    dependencies=[]
                )
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

    def _build_dag(self, data: dict, max_nodes: int = 6) -> ResearchDAG:

        dag = ResearchDAG()

        nodes = data.get("nodes", [])

        if not nodes:
            raise ValueError("No nodes generated")

        # Limit the DAG to max_nodes
        nodes = nodes[:max_nodes]

        valid_ids = {
            node.get("id")
            for node in nodes
            if node.get("id")
        }

        for node_data in nodes:

            node_id = node_data.get("id")

            if not node_id:
                continue

            dependencies = [
                dependency
                for dependency in node_data.get(
                    "dependencies",
                    []
                )
                if dependency in valid_ids
                and dependency != node_id
            ]

            node = DAGNode(
                id=node_id,
                task=node_data.get("task", ""),
                dependencies=dependencies
            )

            dag.add_node(node)

        if not dag.nodes:
            raise ValueError("Could not create DAG")

        return dag