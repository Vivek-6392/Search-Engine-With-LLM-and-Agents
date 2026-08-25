import unittest.mock as mock
import pytest

from app import (
    generate_deterministic_title,
    generate_chat_title,
    build_conversation_context,
    render_live_budget_meter_hud,
)
from budget import QueryBudget
from metrics import MetricsCollector
import chat_store


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_overhead.db")
    chat_store.init_db(db_file)
    return db_file


def test_deterministic_title_generation_zero_llm_calls():
    # Various query formats
    q1 = "What is the current stock price and market capitalization of Apple AAPL?"
    t1 = generate_deterministic_title(q1)
    assert "current stock price" in t1.lower()
    assert len(t1.split()) <= 10

    q2 = "Explain the difference between FlashAttention-2 and FlashAttention-3 kernels."
    t2 = generate_deterministic_title(q2)
    assert "difference between flashattention" in t2.lower()
    assert len(t2.split()) <= 10

    q3 = "Can you please tell me about latest discoveries on James Webb Space Telescope?"
    t3 = generate_deterministic_title(q3)
    assert "latest discoveries on james webb" in t3.lower()


def test_deterministic_title_sanitization_and_word_limits():
    # Noisy quotes, markdown headers, and question marks
    raw_query = "### **What is** `quantum computing` *breakthroughs* in 2025??? [Source: arXiv]"
    title = generate_deterministic_title(raw_query, max_words=6)
    assert "`" not in title
    assert "*" not in title
    assert "#" not in title
    assert "?" not in title
    assert len(title.split()) <= 6


def test_optional_llm_title_behind_flag():
    mock_llm = mock.MagicMock()
    mock_llm.invoke.return_value = mock.MagicMock(content="LLM Generated Summary Title")

    # Default: use_llm=False -> LLM is NOT called
    title_default = generate_chat_title(mock_llm, "What is photosynthesis?", use_llm=False)
    assert title_default == "Photosynthesis"
    mock_llm.invoke.assert_not_called()

    # Explicit: use_llm=True -> LLM IS called
    title_llm = generate_chat_title(mock_llm, "What is photosynthesis?", use_llm=True)
    assert title_llm == "LLM Generated Summary Title"
    assert mock_llm.invoke.call_count == 1


def test_chat_memory_deterministic_truncation_without_llm(temp_db):
    chat_id = chat_store.create_chat(title="Test Overhead Chat", db_path=temp_db)

    # Insert 6 messages
    for i in range(1, 4):
        chat_store.append_message(chat_id, "user", f"User question {i}", db_path=temp_db)
        chat_store.append_message(chat_id, "assistant", f"Assistant answer {i}", db_path=temp_db)

    mock_llm = mock.MagicMock()
    # build context with use_llm_summary=False (default)
    context = build_conversation_context(mock_llm, chat_id, db_path=temp_db, use_llm_summary=False)

    assert "[User]: User question 1" in context
    assert "[Assistant]: Assistant answer 3" in context
    # Zero LLM calls
    mock_llm.invoke.assert_not_called()


def test_budget_telemetry_hud_metrics():
    budget = QueryBudget(mode="deep")
    budget.start()
    budget.record_usage(llm_calls=3, input_tokens=3500, output_tokens=1350, tool_calls=5, browser_calls=2)

    collector = MetricsCollector(research_mode="deep")
    collector.start()
    collector.record_retry(stage="node_1", reason="429 rate limit")
    collector.record_node_end("node_2", status="SKIPPED")

    mock_container = mock.MagicMock()
    # Call render HUD
    render_live_budget_meter_hud(budget, collector, mock_container)

    # Verify container was used
    assert mock_container.__enter__.call_count >= 1
