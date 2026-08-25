import concurrent.futures
import threading
import time
import pytest

from budget import QueryBudget, BudgetConfig, load_budget_config
from dag.models import DAGNode, ResearchDAG
from dag.executor import DAGExecutor


def test_budget_config_defaults_and_env_overrides(monkeypatch):
    # Test default loading
    fast_cfg = load_budget_config("fast")
    assert fast_cfg.max_llm_calls == 1
    assert fast_cfg.max_total_tokens == 2500
    assert fast_cfg.max_tool_calls == 2

    normal_cfg = load_budget_config("normal")
    assert normal_cfg.max_llm_calls == 3
    assert normal_cfg.max_total_tokens == 5000
    assert normal_cfg.max_tool_calls == 5
    assert normal_cfg.max_browser_calls == 2

    deep_cfg = load_budget_config("deep")
    assert deep_cfg.max_llm_calls == 4
    assert deep_cfg.max_total_tokens == 6500
    assert deep_cfg.max_tool_calls == 8
    assert deep_cfg.max_browser_calls == 3

    # Test environment variable override
    monkeypatch.setenv("BUDGET_MAX_LLM_CALLS", "99")
    monkeypatch.setenv("BUDGET_FAST_MAX_TOTAL_TOKENS", "3333")
    overridden_fast = load_budget_config("fast")
    assert overridden_fast.max_llm_calls == 99
    assert overridden_fast.max_total_tokens == 3333


def test_exact_budget_tracking():
    cfg = BudgetConfig(
        max_llm_calls=2,
        max_total_tokens=1000,
        max_tool_calls=3,
        max_browser_calls=1,
        max_retries=1,
        timeout_seconds=10.0,
    )
    budget = QueryBudget(config=cfg)
    budget.start()

    assert budget.can_call_llm() is True
    assert budget.remaining_tokens() == 1000
    assert budget.remaining_time() > 0.0
    assert budget.exhausted() is False

    # Exact reservation and usage
    assert budget.reserve("llm", count=1, tokens=500) is True
    assert budget.remaining_tokens() == 500

    budget.record_usage(llm_calls=1, input_tokens=300, output_tokens=200, tool_calls=2)
    assert budget.llm_calls_used == 1
    assert budget.total_tokens_used == 500
    assert budget.tool_calls_used == 2
    assert budget.remaining_tokens() == 500
    assert budget.exhausted() is False

    # Second call uses remaining exact budget
    assert budget.reserve("llm", count=1, tokens=500) is True
    budget.record_usage(llm_calls=1, input_tokens=300, output_tokens=200, tool_calls=1)
    assert budget.llm_calls_used == 2
    assert budget.total_tokens_used == 1000
    assert budget.tool_calls_used == 3
    assert budget.remaining_tokens() == 0
    assert budget.exhausted() is True

    budget.finish()


def test_budget_exceeded():
    cfg = BudgetConfig(
        max_llm_calls=1,
        max_total_tokens=500,
        max_tool_calls=1,
        max_browser_calls=1,
        max_retries=0,
    )
    budget = QueryBudget(config=cfg)
    budget.start()

    # Reserve up to limit
    assert budget.reserve("llm", count=1, tokens=500) is True

    # Exceed LLM calls
    assert budget.can_call_llm() is False
    assert budget.reserve("llm", count=1) is False

    # Exceed browser calls
    assert budget.reserve("browser", count=1) is True
    assert budget.reserve("browser", count=1) is False

    # Exceed tool calls
    assert budget.reserve("tool", count=1) is True
    assert budget.reserve("tool", count=1) is False

    # Exceed retries
    assert budget.reserve("retry", count=1) is False

    # Record usage that blows tokens
    budget.record_usage(input_tokens=600, output_tokens=100)
    assert budget.exhausted() is True
    status = budget.get_status()
    assert status["exhausted"] is True
    assert "max_total_tokens" in status["exhaustion_reasons"]

    budget.finish()


def test_concurrent_reservations():
    cfg = BudgetConfig(
        max_llm_calls=10,
        max_total_tokens=10000,
        max_tool_calls=20,
    )
    budget = QueryBudget(config=cfg)
    budget.start()

    successful_llm_reservations = []
    lock = threading.Lock()

    def worker_reserve():
        res = budget.reserve("llm", count=1, tokens=500)
        with lock:
            if res:
                successful_llm_reservations.append(True)
                budget.record_usage(llm_calls=1, input_tokens=250, output_tokens=250)

    # Launch 20 concurrent threads attempting 10 max slots
    threads = [threading.Thread(target=worker_reserve) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(successful_llm_reservations) == 10
    assert budget.llm_calls_used == 10
    assert budget.total_tokens_used == 5000
    assert budget.can_call_llm() is False

    budget.finish()


def test_retry_reservation_and_blocking():
    cfg = BudgetConfig(
        max_llm_calls=3,
        max_total_tokens=5000,
        max_retries=1,
    )
    budget = QueryBudget(config=cfg)
    budget.start()

    # First retry allowed
    assert budget.reserve("retry", count=1) is True
    budget.record_usage(retries=1)

    # Second retry rejected
    assert budget.reserve("retry", count=1) is False
    assert budget.exhausted() is True
    assert "max_retries" in budget.get_status()["exhaustion_reasons"]

    budget.finish()


def test_partial_completion_and_skipping_in_executor():
    # Setup a budget that only allows 1 LLM call before running low
    cfg = BudgetConfig(
        max_llm_calls=2,  # 1 for node, 1 reserved for synthesis
        max_total_tokens=5000,
    )
    budget = QueryBudget(config=cfg)
    budget.start()

    dag = ResearchDAG()
    dag.add_node(DAGNode(id="node_1", task="Task 1", dependencies=[]))
    dag.add_node(DAGNode(id="node_2", task="Task 2", dependencies=["node_1"]))
    dag.add_node(DAGNode(id="node_3", task="Task 3", dependencies=["node_2"]))

    class MockAgent:
        def invoke(self, inp):
            return {"output": "Mock output", "intermediate_steps": []}

    executor = DAGExecutor(agent=MockAgent(), budget=budget)
    executed_dag = executor.execute(dag)

    # Node 1 executes
    assert executed_dag.nodes["node_1"].status == "COMPLETED"
    # Once node 1 used 1 LLM call, remaining LLM calls <= 1 (should_skip_optional_work() triggers)
    # Remaining nodes are marked SKIPPED rather than hanging or crashing
    assert executed_dag.nodes["node_2"].status == "SKIPPED"
    assert executed_dag.nodes["node_3"].status == "SKIPPED"
    assert "budget constraints" in executed_dag.nodes["node_2"].error.lower()

    budget.finish()
