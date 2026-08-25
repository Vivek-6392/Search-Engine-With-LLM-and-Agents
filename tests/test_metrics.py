import concurrent.futures
import json
import logging
import time
import unittest.mock as mock
import pytest

from dag.models import DAGNode, ResearchDAG
from dag.planner import DAGPlanner
from dag.executor import DAGExecutor
from metrics.collector import (
    MetricsCollector,
    RequestMetrics,
    get_current_metrics,
    set_current_metrics,
)
from metrics.callbacks import MetricsCallbackHandler
from metrics.logger import log_event, StructuredJSONFormatter


def test_request_metrics_fields_and_serialization():
    metrics = RequestMetrics(
        request_id="req-test-123",
        research_mode="deep",
        selected_model_provider="Groq / openai/gpt-oss-20b",
        total_latency=5.4321,
        planner_latency=0.85,
        dag_node_count=3,
        completed_nodes=2,
        failed_nodes=1,
        skipped_nodes=0,
        per_node_latency={
            "node_1": {"latency_seconds": 1.2, "status": "COMPLETED", "tool_used": "web_search_tool"},
            "node_2": {"latency_seconds": 1.5, "status": "COMPLETED", "tool_used": "calculator_tool"},
            "node_3": {"latency_seconds": 0.5, "status": "FAILED", "tool_used": None, "error": "timeout"},
        },
        tool_call_count=2,
        tool_names=["web_search_tool", "calculator_tool"],
        browser_calls=1,
        playwright_calls=0,
        http_fetch_calls=1,
        llm_call_count=4,
        input_tokens=1200,
        output_tokens=350,
        total_tokens=1550,
        retries=1,
        rate_limit_errors=1,
        final_synthesis_latency=1.1,
        context_size_per_llm_call=[
            {"call_index": 1, "stage": "planner", "chars": 800, "estimated_tokens": 200},
            {"call_index": 2, "stage": "node_agent", "chars": 1500, "estimated_tokens": 375},
        ],
    )

    data = metrics.to_dict()
    assert data["request_id"] == "req-test-123"
    assert data["research_mode"] == "deep"
    assert data["selected_model_provider"] == "Groq / openai/gpt-oss-20b"
    assert data["total_latency"] == 5.4321
    assert data["planner_latency"] == 0.85
    assert data["dag_node_count"] == 3
    assert data["completed_nodes"] == 2
    assert data["failed_nodes"] == 1
    assert data["tool_call_count"] == 2
    assert "web_search_tool" in data["tool_names"]
    assert data["browser_calls"] == 1
    assert data["http_fetch_calls"] == 1
    assert data["playwright_calls"] == 0
    assert data["total_tokens"] == 1550
    assert data["retries"] == 1
    assert data["rate_limit_errors"] == 1
    assert data["final_synthesis_latency"] == 1.1

    json_str = metrics.to_json()
    parsed = json.loads(json_str)
    assert parsed["request_id"] == "req-test-123"


def test_metrics_collector_lifecycle_and_contextvar():
    collector = MetricsCollector(
        request_id="custom-id-99",
        research_mode="fast",
        model_provider="Groq / openai/gpt-oss-20b",
    )

    assert get_current_metrics() is None

    with collector as col:
        assert get_current_metrics() is col
        assert col.metrics.request_id == "custom-id-99"
        time.sleep(0.01)

    assert get_current_metrics() is None
    assert collector.metrics.total_latency > 0.0


def test_metrics_collector_node_tracking():
    collector = MetricsCollector(research_mode="deep")
    collector.start()

    collector.record_node_start("node_1", task="Search Apple stock")
    collector.record_node_start("node_2", task="Search Microsoft stock")
    time.sleep(0.02)

    collector.record_node_end("node_1", status="COMPLETED", tool_used="finance_tool")
    collector.record_node_end("node_2", status="FAILED", error="API timeout")

    m = collector.get_metrics()
    assert m.completed_nodes == 1
    assert m.failed_nodes == 1
    assert m.dag_node_count == 2
    assert "node_1" in m.per_node_latency
    assert m.per_node_latency["node_1"]["status"] == "COMPLETED"
    assert m.per_node_latency["node_1"]["tool_used"] == "finance_tool"
    assert m.per_node_latency["node_1"]["latency_seconds"] > 0.0

    assert "node_2" in m.per_node_latency
    assert m.per_node_latency["node_2"]["status"] == "FAILED"
    assert m.per_node_latency["node_2"]["error"] == "API timeout"

    collector.finish()


def test_metrics_collector_tools_and_browser():
    collector = MetricsCollector()
    collector.record_tool_call("web_search_tool", 0.35)
    collector.record_tool_call("calculator_tool", 0.05)

    collector.record_browser_call(fetch_type="http", success=True)
    collector.record_browser_call(fetch_type="playwright", success=True)
    collector.record_browser_call(fetch_type="http", success=False)

    m = collector.get_metrics()
    assert m.tool_call_count == 2
    assert m.tool_names == ["web_search_tool", "calculator_tool"]
    assert m.browser_calls == 3
    assert m.http_fetch_calls == 2
    assert m.playwright_calls == 1


def test_metrics_collector_llm_and_synthesis():
    collector = MetricsCollector()
    collector.record_llm_call(
        stage="planner",
        prompt_chars=1200,
        input_tokens=300,
        output_tokens=80,
        model="gpt-oss-20b",
        latency_seconds=0.45,
    )
    collector.record_synthesis(
        latency=1.25,
        context_chars=2400,
        input_tokens=600,
        output_tokens=250,
        model="gpt-oss-20b",
    )
    collector.record_retry(stage="node_1", reason="timeout")
    collector.record_rate_limit(stage="node_1", error_msg="429 Too Many Requests")

    m = collector.get_metrics()
    assert m.llm_call_count == 2
    assert m.input_tokens == 900
    assert m.output_tokens == 330
    assert m.total_tokens == 1230
    assert m.final_synthesis_latency == 1.25
    assert m.retries == 1
    assert m.rate_limit_errors == 1
    assert len(m.context_size_per_llm_call) == 2
    assert m.context_size_per_llm_call[0]["stage"] == "planner"
    assert m.context_size_per_llm_call[1]["stage"] == "final_synthesis"


def test_metrics_collector_concurrency_thread_safety():
    collector = MetricsCollector()
    collector.start()

    def worker(idx):
        nid = f"node_{idx}"
        collector.record_node_start(nid, task=f"Task {idx}")
        collector.record_tool_call("web_search_tool", 0.05)
        collector.record_llm_call("node_agent", prompt_chars=500, input_tokens=120, output_tokens=40)
        time.sleep(0.01)
        collector.record_node_end(nid, status="COMPLETED", tool_used="web_search_tool")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(20)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    collector.finish()
    m = collector.get_metrics()
    assert m.completed_nodes == 20
    assert m.tool_call_count == 20
    assert m.llm_call_count == 20
    assert m.input_tokens == 2400
    assert m.output_tokens == 800
    assert len(m.per_node_latency) == 20


def test_metrics_callback_handler():
    collector = MetricsCollector()
    handler = MetricsCallbackHandler(collector=collector, stage="test_stage")

    # Mock LLM start
    handler.on_llm_start(serialized={}, prompts=["Hello world test prompt"], run_id="run_1")
    time.sleep(0.01)

    # Mock LLM end with token usage metadata
    mock_response = mock.MagicMock()
    mock_response.llm_output = {
        "token_usage": {"prompt_tokens": 15, "completion_tokens": 8},
        "model_name": "test-model",
    }
    mock_response.generations = []
    handler.on_llm_end(mock_response, run_id="run_1")

    # Mock tool start and end
    handler.on_tool_start(serialized={}, input_str="AAPL", run_id="tool_1")
    handler.on_tool_end(output="result", run_id="tool_1", name="finance_tool")

    # Mock rate limit error
    handler.on_llm_error(Exception("429 rate_limit exceeded"), run_id="err_1")

    m = collector.get_metrics()
    assert m.llm_call_count == 1
    assert m.input_tokens == 15
    assert m.output_tokens == 8
    assert m.tool_call_count == 1
    assert "finance_tool" in m.tool_names
    assert m.rate_limit_errors == 1


def test_structured_json_formatter():
    formatter = StructuredJSONFormatter()
    record = logging.LogRecord(
        name="deepsearchai",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Request completed",
        args=(),
        exc_info=None,
    )
    record.event_data = {"event": "request_completed", "request_id": "test-req-1", "latency": 2.5}
    formatted = formatter.format(record)

    data = json.loads(formatted)
    assert data["level"] == "INFO"
    assert data["event"] == "request_completed"
    assert data["request_id"] == "test-req-1"
    assert data["latency"] == 2.5


def test_planner_and_executor_with_metrics():
    collector = MetricsCollector(research_mode="deep")
    collector.start()

    mock_planner_llm = mock.MagicMock()
    mock_planner_llm.invoke.return_value = (
        '{"nodes": [{"id": "node_1", "task": "Task 1", "dependencies": []}, '
        '{"id": "node_2", "task": "Task 2", "dependencies": ["node_1"]}]}'
    )

    planner = DAGPlanner(llm=mock_planner_llm)
    dag = planner.create_dag("Research AAPL and MSFT", mode="deep", metrics=collector)

    assert len(dag.nodes) == 2
    assert collector.metrics.dag_node_count == 2
    assert collector.metrics.planner_latency >= 0.0

    # Test executing mock DAG with executor
    mock_agent = mock.MagicMock()
    mock_agent.invoke.return_value = {
        "output": "Mock findings",
        "intermediate_steps": [
            (mock.MagicMock(tool="web_search_tool"), "Observed content"),
        ],
    }

    executor = DAGExecutor(agent=mock_agent, max_workers=2, metrics=collector)
    res_dag = executor.execute(dag, metrics=collector)

    collector.finish()
    m = collector.get_metrics()
    assert res_dag.is_complete() is True
    assert m.completed_nodes == 2
    assert m.total_latency > 0.0
