import json
import unittest.mock as mock
import pytest
from dag.models import ResearchDAG, DAGNode
from dag.planner import DAGPlanner


def test_extract_json_clean():
    planner = DAGPlanner(llm=None)
    data = planner._extract_json('{"nodes": [{"id": "node_1", "task": "test", "dependencies": []}]}')
    assert "nodes" in data
    assert len(data["nodes"]) == 1


def test_extract_json_markdown_fences():
    planner = DAGPlanner(llm=None)
    raw = """```json
    {
        "nodes": [
            {"id": "node_1", "task": "task 1", "dependencies": []}
        ]
    }
    ```"""
    data = planner._extract_json(raw)
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["id"] == "node_1"


def test_extract_json_surrounding_text():
    planner = DAGPlanner(llm=None)
    raw = """Here is your research DAG plan:
    {
        "nodes": [
            {"id": "node_1", "task": "task 1", "dependencies": []}
        ]
    }
    Hope this helps!"""
    data = planner._extract_json(raw)
    assert len(data["nodes"]) == 1


def test_extract_json_invalid():
    planner = DAGPlanner(llm=None)
    with pytest.raises(ValueError):
        planner._extract_json("There is no json in this response.")


def test_build_dag_valid_and_sanitization():
    planner = DAGPlanner(llm=None)
    data = {
        "nodes": [
            {"id": "node_1", "task": "Task 1", "dependencies": ["node_1", "non_existent"]},
            {"id": "node_2", "task": "Task 2", "dependencies": ["node_1"]},
        ]
    }
    dag = planner._build_dag(data, max_nodes=6)
    assert len(dag.nodes) == 2
    # Self-dependency and non_existent should be filtered out from node_1
    assert dag.nodes["node_1"].dependencies == []
    # node_2 should keep valid dependency node_1
    assert dag.nodes["node_2"].dependencies == ["node_1"]


def test_build_dag_max_nodes_capping():
    planner = DAGPlanner(llm=None)
    data = {
        "nodes": [
            {"id": f"node_{i}", "task": f"Task {i}", "dependencies": []}
            for i in range(10)
        ]
    }
    dag = planner._build_dag(data, max_nodes=4)
    assert len(dag.nodes) == 4


def test_build_dag_empty_raises():
    planner = DAGPlanner(llm=None)
    with pytest.raises(ValueError):
        planner._build_dag({"nodes": []})


def test_create_dag_fast_mode():
    planner = DAGPlanner(llm=None)
    dag = planner.create_dag("Fast query test", mode="fast")
    assert len(dag.nodes) == 0
    assert dag.is_complete() is True


def test_create_dag_fallback_on_llm_error():
    mock_llm = mock.MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM connection failed")

    planner = DAGPlanner(llm=mock_llm)
    dag = planner.create_dag("Fallback test query", mode="deep")

    # Safe fallback should produce a 1-node DAG
    assert len(dag.nodes) == 1
    assert dag.nodes["node_1"].task == "Fallback test query"
    assert dag.nodes["node_1"].dependencies == []


def test_create_dag_with_context_invokes_llm_with_context():
    mock_llm = mock.MagicMock()
    mock_response = mock.MagicMock()
    mock_response.content = json.dumps({
        "nodes": [
            {"id": "node_1", "task": "Compare Apple with previously discussed Microsoft", "dependencies": []}
        ]
    })
    mock_llm.invoke.return_value = mock_response

    planner = DAGPlanner(llm=mock_llm)
    context_text = "Turn 1: Discussed Microsoft market cap of $3.5T."
    dag = planner.create_dag("Compare that to Apple", mode="deep", context=context_text)

    assert len(dag.nodes) == 1
    assert "node_1" in dag.nodes
    # Check that LLM received context in prompt
    call_args = mock_llm.invoke.call_args[0][0]
    human_msg = [m for m in call_args if hasattr(m, "content") and "Prior Conversation Context:" in m.content]
    assert len(human_msg) == 1
    assert context_text in human_msg[0].content

