import json
import unittest.mock as mock
import pytest

from budget import QueryBudget
from dag.models import DAGNode, ResearchDAG
from dag.planner import DAGPlanner
from dag.executor import DAGExecutor
from dag.mode_strategies import (
    execute_fast_mode,
    execute_normal_mode,
    execute_deep_mode,
    execute_academic_mode,
)
from metrics import MetricsCollector


def test_fast_mode_execution_strategy_call_limits():
    collector = MetricsCollector(research_mode="fast")
    collector.start()
    budget = QueryBudget(mode="fast")
    budget.start()

    # Mock tool
    mock_calc = mock.MagicMock()
    mock_calc.name = "python_calculator"
    mock_calc.invoke.return_value = "Result: 42"
    mock_calc.args = {"query": "6 * 7"}

    # Mock LLM (should only be called ONCE for final answer)
    mock_llm = mock.MagicMock()
    mock_llm.invoke.return_value = mock.MagicMock(content="The answer to 6 * 7 is 42.")

    answer, dag = execute_fast_mode(
        query="What is 6 * 7?",
        llm=mock_llm,
        tools=[mock_calc],
        collector=collector,
        budget=budget,
    )

    assert "42" in answer
    assert len(dag.nodes) == 0  # DAG bypassed in fast mode
    # Exactly 1 LLM call
    assert mock_llm.invoke.call_count == 1
    assert collector.metrics.llm_call_count == 1
    assert collector.metrics.tool_call_count <= 2


def test_normal_mode_execution_strategy_call_limits():
    collector = MetricsCollector(research_mode="normal")
    collector.start()
    budget = QueryBudget(mode="normal")
    budget.start()

    # Planner response (1st LLM call)
    planner_json = json.dumps({
        "nodes": [
            {"id": "node_1", "task_type": "tool", "tool": "web_search", "query": "AAPL market cap", "dependencies": []},
            {"id": "node_2", "task_type": "tool", "tool": "web_search", "query": "MSFT market cap", "dependencies": []},
        ]
    })

    mock_llm = mock.MagicMock()
    # 1st call: planner; 2nd call: final synthesis
    mock_llm.invoke.side_effect = [
        mock.MagicMock(content=planner_json),
        mock.MagicMock(content="Apple market cap is $3.5T and Microsoft is $3.1T."),
    ]

    with mock.patch("utils.search_pipeline.search_and_extract_evidence") as mock_search:
        mock_search.return_value = [
            mock.MagicMock(to_markdown=lambda: "• Apple AAPL: $3.5T", to_dict=lambda: {"claim": "Apple AAPL: $3.5T", "title": "Apple", "url": "https://apple.com", "source": "Apple", "confidence": 0.95})
        ]

        answer, dag = execute_normal_mode(
            query="Compare Apple and Microsoft market cap",
            llm=mock_llm,
            tools=[],
            collector=collector,
            budget=budget,
        )

        assert "Apple" in answer
        assert len(dag.nodes) == 2
        # Exactly 2 LLM calls (1 planner + 0 for deterministic tools + 1 synthesis)
        assert mock_llm.invoke.call_count == 2
        assert collector.metrics.llm_call_count == 2


def test_deep_mode_execution_strategy_and_coverage_check():
    collector = MetricsCollector(research_mode="deep")
    collector.start()
    budget = QueryBudget(mode="deep")
    budget.start()

    planner_json = json.dumps({
        "nodes": [
            {"id": "node_1", "task_type": "tool", "tool": "web_search", "query": "Nvidia Blackwell GPU specs", "dependencies": []},
            {"id": "node_2", "task_type": "tool", "tool": "web_search", "query": "AMD MI300X GPU specs", "dependencies": []},
            {"id": "node_3", "task_type": "tool", "tool": "web_search", "query": "Google TPU v5p specs", "dependencies": []},
        ]
    })

    mock_llm = mock.MagicMock()
    mock_llm.invoke.side_effect = [
        mock.MagicMock(content=planner_json),
        mock.MagicMock(content="Comprehensive analysis of Nvidia Blackwell, AMD MI300X, and Google TPU v5p AI accelerators."),
    ]

    with mock.patch("utils.search_pipeline.search_and_extract_evidence") as mock_search:
        mock_search.return_value = [
            mock.MagicMock(to_markdown=lambda: "• AI accelerator benchmark specs", to_dict=lambda: {"claim": "Blackwell 208B transistors", "title": "Specs", "url": "https://nvidia.com", "source": "Nvidia", "confidence": 0.95})
        ]

        answer, dag = execute_deep_mode(
            query="Comprehensive comparison of AI accelerators 2025",
            llm=mock_llm,
            tools=[],
            collector=collector,
            budget=budget,
        )

        assert "accelerator" in answer.lower()
        assert len(dag.nodes) == 3
        # Target: <= 4 LLM calls
        assert mock_llm.invoke.call_count <= 4


def test_academic_mode_execution_strategy_and_metadata_retrieval():
    collector = MetricsCollector(research_mode="academic")
    collector.start()
    budget = QueryBudget(mode="academic")
    budget.start()

    planner_json = json.dumps({
        "nodes": [
            {"id": "node_1", "task_type": "tool", "tool": "arxiv", "query": "Transformer attention optimization", "dependencies": []},
            {"id": "node_2", "task_type": "tool", "tool": "academic_search", "query": "FlashAttention 3 kernel benchmarks", "dependencies": []},
        ]
    })

    mock_llm = mock.MagicMock()
    mock_llm.invoke.side_effect = [
        mock.MagicMock(content=planner_json),
        mock.MagicMock(content="Academic literature review on FlashAttention and linear complexity attention mechanisms."),
    ]

    mock_arxiv = mock.MagicMock()
    mock_arxiv.name = "arxiv_tool"
    mock_arxiv.invoke.return_value = (
        "• [Dao et al., 2024](https://arxiv.org/abs/2407.08608): FlashAttention-3: Fast and Accurate Attention with Asynchrony."
    )

    answer, dag = execute_academic_mode(
        query="FlashAttention advancements in transformer architectures",
        llm=mock_llm,
        tools=[mock_arxiv],
        collector=collector,
        budget=budget,
    )

    assert "FlashAttention" in answer
    assert len(dag.nodes) == 2
    assert mock_llm.invoke.call_count == 2
