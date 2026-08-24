import os
import tempfile
import pytest
import chat_store


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_chats.db")
    chat_store.init_db(db_file)
    return db_file


def test_create_and_list_chats(temp_db):
    c1 = chat_store.create_chat(title="Chat One", db_path=temp_db)
    c2 = chat_store.create_chat(title="Chat Two", db_path=temp_db)

    chats = chat_store.list_chats(db_path=temp_db)
    assert len(chats) == 2
    chat_ids = [c["id"] for c in chats]
    assert c1 in chat_ids
    assert c2 in chat_ids


def test_rename_and_get_chat(temp_db):
    chat_id = chat_store.create_chat(title="Initial Title", db_path=temp_db)
    chat = chat_store.get_chat(chat_id, db_path=temp_db)
    assert chat["title"] == "Initial Title"

    chat_store.rename_chat(chat_id, "Updated Title", db_path=temp_db)
    updated_chat = chat_store.get_chat(chat_id, db_path=temp_db)
    assert updated_chat["title"] == "Updated Title"


def test_delete_chat_and_cascade_messages(temp_db):
    chat_id = chat_store.create_chat(title="To Delete", db_path=temp_db)
    chat_store.append_message(chat_id, "user", "Hello", db_path=temp_db)
    chat_store.append_message(chat_id, "assistant", "Hi there", db_path=temp_db)

    assert len(chat_store.get_messages(chat_id, db_path=temp_db)) == 2
    chat_store.delete_chat(chat_id, db_path=temp_db)

    assert chat_store.get_chat(chat_id, db_path=temp_db) is None
    assert len(chat_store.get_messages(chat_id, db_path=temp_db)) == 0


def test_messages_ordering_and_snapshot(temp_db):
    chat_id = chat_store.create_chat(title="Message Test", db_path=temp_db)
    m1 = chat_store.append_message(chat_id, "user", "What is quantum computing?", db_path=temp_db)
    snapshot = '{"nodes": [{"id": "node_1", "task": "Search"}]}'
    m2 = chat_store.append_message(
        chat_id, "assistant", "Quantum computing is...", dag_snapshot=snapshot, db_path=temp_db
    )

    messages = chat_store.get_messages(chat_id, db_path=temp_db)
    assert len(messages) == 2
    assert messages[0]["id"] == m1
    assert messages[0]["role"] == "user"
    assert messages[0]["dag_snapshot"] is None
    assert messages[1]["id"] == m2
    assert messages[1]["role"] == "assistant"
    assert messages[1]["dag_snapshot"] == snapshot


def test_summary_storage(temp_db):
    chat_id = chat_store.create_chat(title="Summary Test", db_path=temp_db)
    assert chat_store.get_chat_summary(chat_id, db_path=temp_db) is None

    chat_store.update_chat_summary(chat_id, "Conversation summary here...", db_path=temp_db)
    assert chat_store.get_chat_summary(chat_id, db_path=temp_db) == "Conversation summary here..."
