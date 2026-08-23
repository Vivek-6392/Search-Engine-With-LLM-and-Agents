from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DAGNode:
    id: str
    task: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "PENDING"
    result: Optional[str] = None
    error: Optional[str] = None
    tool_used: Optional[str] = None


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

    def is_complete(self) -> bool:
        return all(
            node.status in ["COMPLETED", "FAILED"]
            for node in self.nodes.values()
        )

    def get_results(self) -> Dict[str, str]:
        return {
            node.id: node.result
            for node in self.nodes.values()
            if node.result
        }