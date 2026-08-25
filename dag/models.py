from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class DAGNode:
    id: str
    task: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "PENDING"
    result: Optional[str] = None
    error: Optional[str] = None
    tool_used: Optional[str] = None
    task_type: str = "llm"  # "tool" for deterministic Python tool execution, "llm" for bounded LLM research
    tool_name: Optional[str] = None
    tool_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchDAG:
    nodes: Dict[str, DAGNode] = field(default_factory=dict)

    def add_node(self, node: DAGNode):
        self.nodes[node.id] = node

    def get_ready_nodes(self) -> List[DAGNode]:
        """
        A node is ready when:
        1. It is still pending
        2. All its dependencies are completed
        """
        ready_nodes = []
        for node in self.nodes.values():
            if node.status != "PENDING":
                continue

            dependencies_completed = all(
                self.nodes[dependency].status == "COMPLETED"
                for dependency in node.dependencies
                if dependency in self.nodes
            )

            if dependencies_completed:
                ready_nodes.append(node)

        return ready_nodes

    def ready_nodes(self) -> List[DAGNode]:
        """Alias for get_ready_nodes() for clean DAG query interface."""
        return self.get_ready_nodes()

    def is_complete(self) -> bool:
        return all(
            node.status in ["COMPLETED", "FAILED", "SKIPPED"]
            for node in self.nodes.values()
        )

    def get_results(self) -> Dict[str, str]:
        return {
            node.id: node.result
            for node in self.nodes.values()
            if node.result
        }

    def topological_layers(self) -> List[List[DAGNode]]:
        """
        Partition nodes into topological dependency layers:
        Layer 0: Nodes with no dependencies.
        Layer k: Nodes whose dependencies are all in Layers < k.
        """
        if not self.nodes:
            return []

        in_degree = {}
        children = defaultdict(list)
        for node_id, node in self.nodes.items():
            valid_deps = [d for d in node.dependencies if d in self.nodes and d != node_id]
            in_degree[node_id] = len(valid_deps)
            for dep in valid_deps:
                children[dep].append(node_id)

        layers: List[List[DAGNode]] = []
        current_layer = [self.nodes[nid] for nid, deg in in_degree.items() if deg == 0]

        visited = set()
        while current_layer:
            layers.append(current_layer)
            next_layer_candidates = set()
            for node in current_layer:
                visited.add(node.id)
                for child_id in children[node.id]:
                    in_degree[child_id] -= 1
                    if in_degree[child_id] == 0:
                        next_layer_candidates.add(child_id)

            current_layer = [self.nodes[nid] for nid in next_layer_candidates if nid not in visited]

        # If any nodes were left unvisited due to cycles, place them in a final layer
        unvisited = [node for nid, node in self.nodes.items() if nid not in visited]
        if unvisited:
            layers.append(unvisited)

        return layers

    def max_depth(self) -> int:
        """Maximum dependency depth of the DAG (number of topological layers)."""
        return len(self.topological_layers())

    def max_fan_out(self) -> int:
        """Maximum number of immediate child nodes dependent on any single parent node."""
        if not self.nodes:
            return 0
        parent_counts = defaultdict(int)
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep in self.nodes:
                    parent_counts[dep] += 1
        return max(parent_counts.values()) if parent_counts else 0

    def critical_path(self) -> List[str]:
        """
        Compute the longest path through the DAG (list of node IDs on the critical execution path).
        """
        if not self.nodes:
            return []

        # Find memoized longest path ending at each node
        memo = {}

        def get_longest_path_to(node_id: str, visited_in_path: Set[str]) -> List[str]:
            if node_id in memo:
                return memo[node_id]
            if node_id in visited_in_path:
                return [node_id]  # Cycle guard

            node = self.nodes.get(node_id)
            if not node:
                return []

            valid_deps = [d for d in node.dependencies if d in self.nodes and d != node_id]
            if not valid_deps:
                path = [node_id]
            else:
                best_dep_path = []
                for dep in valid_deps:
                    sub_path = get_longest_path_to(dep, visited_in_path | {node_id})
                    if len(sub_path) > len(best_dep_path):
                        best_dep_path = sub_path
                path = best_dep_path + [node_id]

            memo[node_id] = path
            return path

        longest_overall = []
        for nid in self.nodes:
            p = get_longest_path_to(nid, set())
            if len(p) > len(longest_overall):
                longest_overall = p

        return longest_overall

    def estimated_token_cost(
        self,
        tokens_per_node: int = 1500,
        synthesis_tokens: int = 1500,
    ) -> int:
        """
        Estimate the total token cost required to execute this planned DAG and final synthesis.
        """
        node_count = len(self.nodes)
        return (node_count * tokens_per_node) + synthesis_tokens