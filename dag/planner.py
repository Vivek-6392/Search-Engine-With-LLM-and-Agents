import json
import re

from dag.models import DAGNode, ResearchDAG


class DAGPlanner:

    def __init__(self, llm):
        self.llm = llm

    def create_dag(self, query: str, mode: str = "deep") -> ResearchDAG:

        if mode == "fast":
            mode_instructions = """
Mode: FAST & CONCISE
1. Create exactly 1 or 2 high-level research tasks for a rapid overview.
2. Focus strictly on the primary direct answer.
"""
            max_nodes = 2
        elif mode == "academic":
            mode_instructions = """
Mode: ACADEMIC & SCIENTIFIC
1. Create 2 to 4 nodes focusing on scientific literature, ArXiv papers, academic journals, and foundational concepts.
2. Structure nodes to find seminal papers, empirical findings, and academic consensus.
"""
            max_nodes = 4
        else:
            mode_instructions = """
Mode: DEEP RESEARCH
1. Create between 2 and 4 comprehensive nodes covering overview, specific dimensions, comparisons, and synthesis.
2. Ensure dependent nodes build logically on previous findings.
"""
            max_nodes = 4

        prompt = f"""
You are a research task planner.

Break the following user query into a small directed acyclic graph (DAG).

Each node represents a research task.

{mode_instructions}

Rules:
1. Create at most {max_nodes} nodes.
2. Independent tasks should have an empty dependencies list [].
3. A task that needs results from another task should list that node ID inside dependencies.
4. Do not create circular dependencies.
5. Keep tasks concise and searchable.
6. Return ONLY valid JSON.

User query:
{query}

Return exactly this format:
{{
    "nodes": [
        {{
            "id": "node_1",
            "task": "research task",
            "dependencies": []
        }},
        {{
            "id": "node_2",
            "task": "another research task",
            "dependencies": ["node_1"]
        }}
    ]
}}
"""

        response = self.llm.invoke(prompt)

        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        try:
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

    def _build_dag(self, data: dict, max_nodes: int = 4) -> ResearchDAG:

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