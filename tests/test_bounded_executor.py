import unittest.mock as mock
import pytest

from budget import QueryBudget, BudgetConfig
from dag.models import DAGNode, ResearchDAG
from dag.executor import DAGExecutor
from metrics import MetricsCollector


def test_deterministic_tool_task_zero_llm_calls():
    mock_llm = mock.MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM should not be called for deterministic tool tasks!")

    collector = MetricsCollector(research_mode="normal")
    collector.start()

    budget = QueryBudget(mode="normal")
    budget.start()

    dag = ResearchDAG()
    dag.add_node(
        DAGNode(
            id="node_calc",
            task="Calculate square root of 144",
            task_type="tool",
            tool_name="calculator",
            tool_params={"query": "sqrt(144)"},
            dependencies=[],
        )
    )

    executor = DAGExecutor(llm=mock_llm, budget=budget, metrics=collector)
    res_dag = executor.execute(dag, metrics=collector, budget=budget)

    # 0 LLM calls were made!
    assert mock_llm.invoke.call_count == 0
    assert collector.metrics.llm_call_count == 0
    assert budget.llm_calls_used == 0

    # Tool was executed directly in Python
    assert res_dag.nodes["node_calc"].status == "COMPLETED"
    assert "12" in res_dag.nodes["node_calc"].result
    assert collector.metrics.tool_call_count == 1
    assert budget.tool_calls_used == 1

    budget.finish()
    collector.finish()


def test_llm_research_task_bounded_calls():
    mock_llm = mock.MagicMock()
    # Step 1 response: tool calling action
    mock_action = mock.MagicMock(tool="calculator_tool")
    mock_llm_response = {
        "output": "Calculated value is 42",
        "intermediate_steps": [(mock_action, "42.0")],
    }

    mock_agent = mock.MagicMock()
    mock_agent.invoke.return_value = mock_llm_response

    collector = MetricsCollector(research_mode="deep")
    collector.start()

    budget = QueryBudget(mode="deep")
    budget.start()

    dag = ResearchDAG()
    dag.add_node(
        DAGNode(
            id="node_research",
            task="Analyze data",
            task_type="llm",
            dependencies=[],
        )
    )

    executor = DAGExecutor(agent=mock_agent, budget=budget, metrics=collector)
    res_dag = executor.execute(dag, metrics=collector, budget=budget)

    # Only 1 invocation to agent
    assert mock_agent.invoke.call_count == 1
    assert res_dag.nodes["node_research"].status == "COMPLETED"

    budget.finish()
    collector.finish()


def test_budget_prevents_extra_iterations():
    # Budget allowing 2 LLM calls total (1 for first node, remaining 1 reserved for synthesis)
    cfg = BudgetConfig(max_llm_calls=2, max_total_tokens=5000)
    budget = QueryBudget(config=cfg)
    budget.start()

    collector = MetricsCollector(research_mode="normal")
    collector.start()

    mock_agent = mock.MagicMock()
    mock_agent.invoke.return_value = {
        "output": "Raw observation 100",
        "intermediate_steps": [(mock.MagicMock(tool="web_search_tool"), "Found data")],
    }

    dag = ResearchDAG()
    dag.add_node(
        DAGNode(
            id="node_1",
            task="First search",
            task_type="llm",
            dependencies=[],
        )
    )
    dag.add_node(
        DAGNode(
            id="node_2",
            task="Second search",
            task_type="llm",
            dependencies=["node_1"],
        )
    )

    executor = DAGExecutor(agent=mock_agent, budget=budget, metrics=collector)
    res_dag = executor.execute(dag, metrics=collector, budget=budget)

    # First node completed, using 1 LLM call
    assert res_dag.nodes["node_1"].status == "COMPLETED"
    assert budget.llm_calls_used == 1
    # Second node was skipped to reserve the final call for report synthesis
    assert res_dag.nodes["node_2"].status == "SKIPPED"

    budget.finish()
    collector.finish()


def test_tool_errors_do_not_cause_infinite_loops():
    # Mock a tool function that raises an exception
    dag = ResearchDAG()
    dag.add_node(
        DAGNode(
            id="node_failing_tool",
            task="Perform failing operation",
            task_type="tool",
            tool_name="calculator",
            tool_params={"query": "invalid syntax +++"},
            dependencies=[],
        )
    )

    collector = MetricsCollector(research_mode="normal")
    collector.start()

    budget = QueryBudget(mode="normal")
    budget.start()

    executor = DAGExecutor(budget=budget, metrics=collector)
    # Execution must finish immediately without looping or crashing
    res_dag = executor.execute(dag, metrics=collector, budget=budget)

    node = res_dag.nodes["node_failing_tool"]
    assert node.status == "COMPLETED"
    # Captured error observation cleanly
    assert "error" in node.result.lower() or "invalid" in node.result.lower()

    budget.finish()
    collector.finish()
