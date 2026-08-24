import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "chats.db")


def _now_iso() -> str:
    """Return timezone-aware ISO format UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _resolve_db_path(db_path: Optional[str] = None) -> str:
    """Resolve database path, falling back to current DEFAULT_DB_PATH."""
    return db_path if db_path is not None else DEFAULT_DB_PATH


def _get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Ensure database directory exists and return a connection with foreign keys enabled."""
    resolved_path = _resolve_db_path(db_path)
    db_dir = os.path.dirname(resolved_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(resolved_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize SQLite database tables for chats and messages."""
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                dag_snapshot TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);"
        )
        conn.commit()


def create_chat(title: str = "New Research", db_path: Optional[str] = None) -> str:
    """Create a new chat session and return its ID."""
    init_db(db_path)
    chat_id = uuid.uuid4().hex
    now = _now_iso()
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chats (id, title, summary, created_at, updated_at)
            VALUES (?, ?, NULL, ?, ?);
            """,
            (chat_id, title, now, now),
        )
        conn.commit()
    return chat_id


def list_chats(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all chat sessions ordered by newest first."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, title, summary, created_at, updated_at
            FROM chats
            ORDER BY updated_at DESC, created_at DESC;
            """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_chat(chat_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch details of a single chat session."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, title, summary, created_at, updated_at
            FROM chats
            WHERE id = ?;
            """,
            (chat_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def rename_chat(chat_id: str, new_title: str, db_path: Optional[str] = None) -> None:
    """Update the title of a chat session."""
    init_db(db_path)
    now = _now_iso()
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE chats
            SET title = ?, updated_at = ?
            WHERE id = ?;
            """,
            (new_title.strip() or "Untitled Research", now, chat_id),
        )
        conn.commit()


def delete_chat(chat_id: str, db_path: Optional[str] = None) -> None:
    """Delete a chat session and all associated messages."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute(
            "DELETE FROM chats WHERE id = ?;",
            (chat_id,),
        )
        conn.commit()


def get_messages(chat_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all messages for a given chat session in chronological order."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, chat_id, role, content, dag_snapshot, created_at
            FROM messages
            WHERE chat_id = ?
            ORDER BY rowid ASC;
            """,
            (chat_id,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def append_message(
    chat_id: str,
    role: str,
    content: str,
    dag_snapshot: Optional[str] = None,
    db_path: Optional[str] = None,
) -> str:
    """Append a user or assistant message to a chat session and update chat timestamp."""
    init_db(db_path)
    message_id = uuid.uuid4().hex
    now = _now_iso()
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO messages (id, chat_id, role, content, dag_snapshot, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (message_id, chat_id, role, content, dag_snapshot, now),
        )
        conn.execute(
            """
            UPDATE chats
            SET updated_at = ?
            WHERE id = ?;
            """,
            (now, chat_id),
        )
        conn.commit()
    return message_id


def update_chat_summary(
    chat_id: str,
    summary: str,
    db_path: Optional[str] = None,
) -> None:
    """Store or update the rolling memory summary for a chat session."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE chats
            SET summary = ?
            WHERE id = ?;
            """,
            (summary, chat_id),
        )
        conn.commit()


def get_chat_summary(
    chat_id: str,
    db_path: Optional[str] = None,
) -> Optional[str]:
    """Retrieve the stored rolling summary for a chat session."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT summary FROM chats WHERE id = ?;",
            (chat_id,),
        )
        row = cursor.fetchone()
        return row["summary"] if row and row["summary"] else None
