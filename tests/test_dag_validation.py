import pytest

from dag.models import DAGNode, ResearchDAG
from dag.validator import validate_and_sanitize_dag
from dag.planner import DAGPlanner


def test_empty_task_rejection():
    dag = ResearchDAG()
    dag.add_node(DAGNode(id="node_1", task="Valid research task", dependencies=[]))
    dag.add_node(DAGNode(id="node_2", task="", dependencies=[]))
    dag.add_node(DAGNode(id="node_3", task="   ", dependencies=[]))

    sanitized = validate_and_sanitize_dag(dag)
    assert len(sanitized.nodes) == 1
    assert "node_1" in sanitized.nodes
    assert "node_2" not in sanitized.nodes
    assert "node_3" not in sanitized.nodes


def test_unique_node_ids_and_self_dependency():
    raw_nodes = [
        DAGNode(id="node_1", task="Research Apple", dependencies=["node_1"]),
        DAGNode(id="node_1", task="Research Microsoft", dependencies=["node_1"]),
    ]

    sanitized = validate_and_sanitize_dag(raw_nodes)
    assert len(sanitized.nodes) == 2
    # Self-dependency should be removed
    for node in sanitized.nodes.values():
        assert node.id not in node.dependencies


def test_invalid_dependency_filtering():
    dag = ResearchDAG()
    dag.add_node(DAGNode(id="node_1", task="Task 1", dependencies=["ghost_node_999", "imaginary_node"]))
    dag.add_node(DAGNode(id="node_2", task="Task 2", dependencies=["node_1", "invalid_node_42"]))

    sanitized = validate_and_sanitize_dag(dag)
    assert sanitized.nodes["node_1"].dependencies == []
    assert sanitized.nodes["node_2"].dependencies == ["node_1"]


def test_duplicate_task_detection_and_merging():
    dag = ResearchDAG()
    dag.add_node(DAGNode(id="node_1", task="Retrieve stock price of Apple AAPL", dependencies=[]))
    dag.add_node(DAGNode(id="node_2", task="retrieve stock price of apple aapl!", dependencies=[]))
    dag.add_node(DAGNode(id="node_3", task="Analyze results", dependencies=["node_2"]))

    sanitized = validate_and_sanitize_dag(dag)
    # Duplicate node_2 should be pruned
    assert len(sanitized.nodes) == 2
    assert "node_1" in sanitized.nodes
    assert "node_2" not in sanitized.nodes
    # Node 3's dependency on node_2 should be re-routed to canonical node_1
    assert sanitized.nodes["node_3"].dependencies == ["node_1"]


def test_cycle_detection_and_resolution():
    dag = ResearchDAG()
    # Cycle: node_1 -> node_2 -> node_3 -> node_1
    dag.add_node(DAGNode(id="node_1", task="Task 1", dependencies=["node_3"]))
    dag.add_node(DAGNode(id="node_2", task="Task 2", dependencies=["node_1"]))
    dag.add_node(DAGNode(id="node_3", task="Task 3", dependencies=["node_2"]))

    sanitized = validate_and_sanitize_dag(dag)
    # Cycle must be broken and at least 1 node must be ready with no dependencies
    assert len(sanitized.get_ready_nodes()) > 0
    # Topological layers should resolve cleanly without hanging
    layers = sanitized.topological_layers()
    assert len(layers) >= 1


def test_redundant_synthesis_node_pruning():
    dag = ResearchDAG()
    dag.add_node(DAGNode(id="node_1", task="Find Apple revenue", dependencies=[]))
    dag.add_node(DAGNode(id="node_2", task="Find Microsoft revenue", dependencies=[]))
    dag.add_node(
        DAGNode(
            id="node_3",
            task="Synthesize and format all findings into a final summary table",
            dependencies=["node_1", "node_2"],
        )
    )

    sanitized = validate_and_sanitize_dag(dag)
    # Redundant synthesis node 3 is pruned because global synthesis runs after DAG
    assert len(sanitized.nodes) == 2
    assert "node_1" in sanitized.nodes
    assert "node_2" in sanitized.nodes
    assert "node_3" not in sanitized.nodes


def test_excessive_node_capping():
    dag = ResearchDAG()
    for i in range(1, 10):
        dag.add_node(DAGNode(id=f"node_{i}", task=f"Research topic {i}", dependencies=[]))

    # NORMAL mode limit: 3 nodes
    normal_sanitized = validate_and_sanitize_dag(dag, max_nodes=3)
    assert len(normal_sanitized.nodes) == 3

    # DEEP mode limit: 4 nodes
    deep_sanitized = validate_and_sanitize_dag(dag, max_nodes=4)
    assert len(deep_sanitized.nodes) == 4


def test_max_depth_and_fan_out_pruning():
    dag = ResearchDAG()
    # Create chain of depth 5: node_1 -> node_2 -> node_3 -> node_4 -> node_5
    dag.add_node(DAGNode(id="node_1", task="Task 1", dependencies=[]))
    dag.add_node(DAGNode(id="node_2", task="Task 2", dependencies=["node_1"]))
    dag.add_node(DAGNode(id="node_3", task="Task 3", dependencies=["node_2"]))
    dag.add_node(DAGNode(id="node_4", task="Task 4", dependencies=["node_3"]))
    dag.add_node(DAGNode(id="node_5", task="Task 5", dependencies=["node_4"]))

    sanitized = validate_and_sanitize_dag(dag, max_nodes=5, max_depth=3, max_fan_out=4)
    # Max depth must be <= 3
    assert sanitized.max_depth() <= 3


def test_topological_layers_and_ready_nodes():
    dag = ResearchDAG()
    dag.add_node(DAGNode(id="node_1", task="Root 1", dependencies=[]))
    dag.add_node(DAGNode(id="node_2", task="Root 2", dependencies=[]))
    dag.add_node(DAGNode(id="node_3", task="Child of 1 and 2", dependencies=["node_1", "node_2"]))
    dag.add_node(DAGNode(id="node_4", task="Child of 3", dependencies=["node_3"]))

    layers = dag.topological_layers()
    assert len(layers) == 3
    assert {n.id for n in layers[0]} == {"node_1", "node_2"}
    assert {n.id for n in layers[1]} == {"node_3"}
    assert {n.id for n in layers[2]} == {"node_4"}

    # Initially node_1 and node_2 are ready
    ready = dag.ready_nodes()
    assert {n.id for n in ready} == {"node_1", "node_2"}


def test_critical_path_and_token_cost():
    dag = ResearchDAG()
    dag.add_node(DAGNode(id="node_1", task="Root 1", dependencies=[]))
    dag.add_node(DAGNode(id="node_2", task="Root 2", dependencies=[]))
    dag.add_node(DAGNode(id="node_3", task="Child of 1", dependencies=["node_1"]))
    dag.add_node(DAGNode(id="node_4", task="Grandchild of 1 via 3", dependencies=["node_3"]))

    crit_path = dag.critical_path()
    assert crit_path == ["node_1", "node_3", "node_4"]

    cost = dag.estimated_token_cost(tokens_per_node=1000, synthesis_tokens=1500)
    assert cost == (4 * 1000) + 1500  # 5500


def test_fast_mode_dag_bypass():
    class DummyLLM:
        def invoke(self, p):
            return "Should not be called"

    planner = DAGPlanner(llm=DummyLLM())
    dag = planner.create_dag("What is 10 + 20?", mode="fast")
    assert len(dag.nodes) == 0
    assert dag.is_complete() is True
