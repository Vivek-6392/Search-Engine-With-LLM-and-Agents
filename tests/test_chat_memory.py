import unittest.mock as mock
import pytest
import chat_store
from app import generate_chat_title, build_conversation_context


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_memory.db")
    chat_store.init_db(db_file)
    return db_file


def test_generate_chat_title_success():
    mock_llm = mock.MagicMock()
    mock_resp = mock.MagicMock()
    mock_resp.content = "Quantum Computing Advancements Overview"
    mock_llm.invoke.return_value = mock_resp

    title = generate_chat_title(mock_llm, "What are the latest breakthroughs in quantum computing?")
    assert title == "Quantum Computing Advancements Overview"


def test_generate_chat_title_fallback_on_error():
    mock_llm = mock.MagicMock()
    mock_llm.invoke.side_effect = Exception("API error")

    long_query = "Compare Apple, Microsoft, and Nvidia in terms of current market capitalization and earnings growth"
    title = generate_chat_title(mock_llm, long_query)
    assert title.startswith("Compare Apple, Microsoft, and Nvidia in")
    assert title.endswith("...")


def test_build_conversation_context_short_history(temp_db):
    chat_id = chat_store.create_chat(title="Short Chat", db_path=temp_db)
    chat_store.append_message(chat_id, "user", "What is Rust?", db_path=temp_db)
    chat_store.append_message(chat_id, "assistant", "Rust is a systems language.", db_path=temp_db)

    mock_llm = mock.MagicMock()
    context = build_conversation_context(mock_llm, chat_id, db_path=temp_db)
    assert "[User]: What is Rust?" in context
    assert "[Assistant]: Rust is a systems language." in context
    # Should not invoke LLM summarization for short history
    mock_llm.invoke.assert_not_called()


def test_build_conversation_context_rolling_summary(temp_db):
    chat_id = chat_store.create_chat(title="Long Chat", db_path=temp_db)

    # Insert 8 messages (4 turns)
    for i in range(1, 5):
        chat_store.append_message(chat_id, "user", f"Question {i}", db_path=temp_db)
        chat_store.append_message(chat_id, "assistant", f"Answer {i}", db_path=temp_db)

    mock_llm = mock.MagicMock()
    mock_resp = mock.MagicMock()
    mock_resp.content = "Summary of Questions 1 and 2 and Answers 1 and 2."
    mock_llm.invoke.return_value = mock_resp

    context = build_conversation_context(mock_llm, chat_id, db_path=temp_db)
    assert "Rolling Summary of Earlier Research:" in context
    assert "Summary of Questions 1 and 2 and Answers 1 and 2." in context
    assert "Question 3" in context
    assert "Question 4" in context
    assert "Question 1" not in context  # Question 1 was collapsed into summary

    # Verify summary was cached in database
    saved_summary = chat_store.get_chat_summary(chat_id, db_path=temp_db)
    assert saved_summary == "Summary of Questions 1 and 2 and Answers 1 and 2."

