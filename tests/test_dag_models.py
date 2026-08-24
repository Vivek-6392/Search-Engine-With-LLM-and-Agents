import pytest
from dag.models import DAGNode, ResearchDAG
from dag.executor import DAGExecutor


def test_dag_node_defaults():
    node = DAGNode(id="node_1", task="Research quantum computing")
    assert node.id == "node_1"
    assert node.task == "Research quantum computing"
    assert node.dependencies == []
    assert node.status == "PENDING"
    assert node.result is None
    assert node.error is None
    assert node.tool_used is None


def test_dag_ready_nodes_resolution():
    dag = ResearchDAG()
    n1 = DAGNode(id="node_1", task="Task 1", dependencies=[])
    n2 = DAGNode(id="node_2", task="Task 2", dependencies=[])
    n3 = DAGNode(id="node_3", task="Task 3", dependencies=["node_1", "node_2"])

    dag.add_node(n1)
    dag.add_node(n2)
    dag.add_node(n3)

    # Initially, only n1 and n2 are ready
    ready = dag.get_ready_nodes()
    assert set(n.id for n in ready) == {"node_1", "node_2"}

    # Complete n1; n3 still not ready because n2 is pending
    n1.status = "COMPLETED"
    ready = dag.get_ready_nodes()
    assert set(n.id for n in ready) == {"node_2"}

    # Complete n2; now n3 becomes ready
    n2.status = "COMPLETED"
    ready = dag.get_ready_nodes()
    assert set(n.id for n in ready) == {"node_3"}

    # Complete n3; no nodes ready
    n3.status = "COMPLETED"
    assert dag.get_ready_nodes() == []
    assert dag.is_complete() is True


def test_dag_is_complete_and_get_results():
    dag = ResearchDAG()
    n1 = DAGNode(id="node_1", task="Task 1", status="COMPLETED", result="Result 1")
    n2 = DAGNode(id="node_2", task="Task 2", status="FAILED", error="Error 2")
    dag.add_node(n1)
    dag.add_node(n2)

    assert dag.is_complete() is True
    results = dag.get_results()
    assert results == {"node_1": "Result 1"}


def test_deadlock_cycle_fallback_in_executor():
    # Construct a cycle: n1 -> n2 -> n1
    dag = ResearchDAG()
    n1 = DAGNode(id="node_1", task="Task 1", dependencies=["node_2"])
    n2 = DAGNode(id="node_2", task="Task 2", dependencies=["node_1"])
    dag.add_node(n1)
    dag.add_node(n2)

    # Empty agent/mock since nodes should never execute
    executor = DAGExecutor(agent=None)
    result_dag = executor.execute(dag)

    assert result_dag.is_complete() is True
    assert n1.status == "FAILED"
    assert n2.status == "FAILED"
    assert "No executable dependency path found" in n1.error
    assert "No executable dependency path found" in n2.error
