import unittest.mock as mock
import pytest

from budget.budget import QueryBudget
from dag.models import DAGNode, ResearchDAG
from dag.executor import DAGExecutor
from metrics.collector import MetricsCollector
from utils.rate_limit import (
    RateLimitType,
    RateLimitInfo,
    classify_llm_error,
    parse_retry_after,
    calculate_backoff_delay,
    compress_prompt_for_context_limit,
)


def test_429_rate_limit_detection_and_classification():
    # 429 HTTP error
    e1 = Exception("Error code 429 - {'error': {'message': 'Rate limit reached for model llama-3.3-70b: TPM limit exceeded'}}")
    info1 = classify_llm_error(e1)
    assert info1.error_type == RateLimitType.RATE_LIMIT_429
    assert info1.is_retryable is True

    # 503 Server error
    e2 = Exception("503 Service Unavailable: Groq backend is currently overloaded")
    info2 = classify_llm_error(e2)
    assert info2.error_type == RateLimitType.SERVICE_UNAVAILABLE_503
    assert info2.is_retryable is True

    # Timeout
    e3 = Exception("HTTPSConnectionPool: Read timed out after 10.0 seconds")
    info3 = classify_llm_error(e3)
    assert info3.error_type == RateLimitType.TIMEOUT
    assert info3.is_retryable is True

    # 401 Auth error (non-retryable)
    e4 = Exception("401 Invalid API Key provided")
    info4 = classify_llm_error(e4)
    assert info4.error_type == RateLimitType.UNKNOWN
    assert info4.is_retryable is False


def test_retry_after_header_and_string_parsing():
    # String parsing
    assert parse_retry_after("Rate limit exceeded. Please try again in 2.34s") == 2.34
    assert parse_retry_after("Rate limit reached. Please try again in 450ms") == 0.45
    assert parse_retry_after("Error 429: Retry-After: 4.5") == 4.5
    assert parse_retry_after("Please retry after 3 seconds") == 3.0

    # Header parsing
    mock_err = mock.MagicMock()
    mock_err.headers = {"Retry-After": "1.8"}
    assert parse_retry_after(mock_err) == 1.8


def test_context_too_large_detection_and_prompt_compression():
    e = Exception("Error code 413: context_length_exceeded. Maximum context length is 8192 tokens.")
    info = classify_llm_error(e)
    assert info.error_type == RateLimitType.CONTEXT_LENGTH_413
    assert info.is_retryable is True

    huge_prompt = (
        "Instructions: Synthesize data\n"
        + "\n".join([f"Detailed observation line {i}: huge verbose research dump content" for i in range(50)])
        + "\nFinal Goal: Provide answer"
    )

    compressed = compress_prompt_for_context_limit(huge_prompt)
    assert len(compressed) < len(huge_prompt)
    assert "intermediate context trimmed" in compressed


def test_exponential_backoff_with_jitter():
    # Test attempt 0, 1, 2 scaling
    d0 = calculate_backoff_delay(attempt=0, base_delay=0.5, max_delay=6.0)
    d1 = calculate_backoff_delay(attempt=1, base_delay=0.5, max_delay=6.0)
    d2 = calculate_backoff_delay(attempt=2, base_delay=0.5, max_delay=6.0)

    # d0 should be ~0.55-0.85s, d1 ~1.05-1.35s, d2 ~2.05-2.35s
    assert 0.5 <= d0 <= 1.0
    assert 1.0 <= d1 <= 1.5
    assert 2.0 <= d2 <= 2.5

    # Test server-provided Retry-After
    d_server = calculate_backoff_delay(attempt=0, retry_after=1.5, max_delay=6.0)
    assert 1.5 <= d_server <= 2.0


def test_budget_aware_retry_blocking():
    budget = QueryBudget(mode="fast")
    budget.start()

    # Fast mode starts with max_retries = 0 or 1
    # Exhaust all retries in budget
    budget.record_usage(retries=budget.config.max_retries)

    # Attempting to reserve another retry should be blocked
    assert budget.reserve("retry") is False


def test_retry_success_flow():
    collector = MetricsCollector(research_mode="normal")
    collector.start()
    budget = QueryBudget(mode="normal")
    budget.start()

    executor = DAGExecutor(llm=None, tools=[])

    dag = ResearchDAG()
    node = DAGNode(
        id="node_1",
        task="Analyze Nvidia quarterly revenue",
        task_type="llm",
    )
    dag.add_node(node)

    # Mock agent invoking: fails on attempt 0 with 429, succeeds on attempt 1
    mock_agent = mock.MagicMock()
    mock_agent.invoke.side_effect = [
        Exception("Error 429: Rate limit exceeded. Try again in 0.05s"),
        {"output": "Nvidia reported record Q3 revenue of $35.1 billion."},
    ]

    with mock.patch.object(executor, "_get_agent", return_value=mock_agent):
        with mock.patch("time.sleep", return_value=None):
            result_dag = executor.execute(dag, metrics=collector, budget=budget)

            assert result_dag.nodes["node_1"].status == "COMPLETED"
            assert "35.1 billion" in result_dag.nodes["node_1"].result
            assert collector.metrics.retries == 1
            assert collector.metrics.rate_limit_errors == 1
            assert len(collector.metrics.retry_details) == 1
            assert collector.metrics.retry_details[0]["error_type"] == "rate_limit_429"


def test_retry_failure_and_partial_research_fallback():
    collector = MetricsCollector(research_mode="normal")
    collector.start()
    budget = QueryBudget(mode="normal")
    budget.start()

    executor = DAGExecutor(llm=None, tools=[])

    dag = ResearchDAG()
    # Parent node provides partial evidence
    parent = DAGNode(id="node_parent", task="Parent task", result="Parent factual evidence.", status="COMPLETED")
    child = DAGNode(
        id="node_child",
        task="Child task requiring synthesis",
        task_type="llm",
        dependencies=["node_parent"],
    )
    dag.add_node(parent)
    dag.add_node(child)

    # Mock agent invoking: persistently fails with 503
    mock_agent = mock.MagicMock()
    mock_agent.invoke.side_effect = Exception("503 Service Unavailable: persistent server overload")

    with mock.patch.object(executor, "_get_agent", return_value=mock_agent):
        with mock.patch("time.sleep", return_value=None):
            result_dag = executor.execute(dag, metrics=collector, budget=budget)

            # Child should complete with partial fallback evidence rather than crashing
            assert result_dag.nodes["node_child"].status == "COMPLETED"
            assert "Parent factual evidence" in result_dag.nodes["node_child"].result
            assert collector.metrics.retries >= 1
