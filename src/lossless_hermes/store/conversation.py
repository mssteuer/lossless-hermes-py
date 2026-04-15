"""Conversation and message CRUD operations for LCM.

Handles storage and retrieval of conversations, messages, and message parts.
Based on the TypeScript conversation-store.ts implementation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from ..db.connection import LcmDatabase
from .identity import build_message_identity_hash

logger = logging.getLogger(__name__)


@dataclass
class ConversationRecord:
    conversation_id: int
    session_id: str
    session_key: str | None
    active: bool
    archived_at: datetime | None
    title: str | None
    bootstrapped_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass
class MessageRecord:
    message_id: int
    conversation_id: int
    seq: int
    role: str
    content: str
    token_count: int
    identity_hash: str
    created_at: datetime


@dataclass
class MessagePartRecord:
    part_id: str
    message_id: int
    session_id: str
    part_type: str
    ordinal: int
    text_content: str | None
    tool_call_id: str | None
    tool_name: str | None
    tool_input: str | None
    tool_output: str | None
    metadata: str | None


@dataclass
class CreateMessageInput:
    conversation_id: int
    seq: int
    role: str
    content: str
    token_count: int
    identity_hash: str | None = None


@dataclass
class CreateMessagePartInput:
    session_id: str
    part_type: str
    ordinal: int
    text_content: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_input: str | None = None
    tool_output: str | None = None
    metadata: str | None = None


@dataclass
class MessageSearchInput:
    conversation_id: int | None = None
    query: str = ""
    mode: str = "full_text"  # "regex" or "full_text"
    since: datetime | None = None
    before: datetime | None = None
    limit: int = 50
    sort: str = "relevance"  # "relevance", "newest", "oldest"


@dataclass
class MessageSearchResult:
    message_id: int
    conversation_id: int
    role: str
    snippet: str
    created_at: datetime
    rank: float | None = None


def parse_utc_timestamp(timestamp_str: str) -> datetime:
    """Parse UTC timestamp string to datetime."""
    if not timestamp_str:
        return datetime.utcnow()

    try:
        # Handle various timestamp formats
        if "T" in timestamp_str:
            if timestamp_str.endswith("Z"):
                return datetime.fromisoformat(timestamp_str[:-1])
            elif "+" in timestamp_str or timestamp_str.count(":") > 2:
                # Has timezone info, parse and convert to UTC
                return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                return datetime.fromisoformat(timestamp_str)
        else:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return datetime.utcnow()


def format_utc_timestamp(dt: datetime) -> str:
    """Format datetime as UTC timestamp string."""
    return dt.isoformat() + "Z"


class ConversationStore:
    """Store for managing conversations and messages."""

    def __init__(self, db: LcmDatabase):
        self.db = db

    def create_conversation(
        self,
        session_id: str,
        session_key: str | None = None,
        title: str | None = None,
        active: bool = True,
        archived_at: datetime | None = None,
    ) -> ConversationRecord:
        """Create a new conversation."""
        now = datetime.utcnow()

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversations (
                    session_id, session_key, active, archived_at, title, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    session_key,
                    1 if active else 0,
                    format_utc_timestamp(archived_at) if archived_at else None,
                    title,
                    format_utc_timestamp(now),
                    format_utc_timestamp(now),
                ),
            )

            conversation_id = cursor.lastrowid

            return ConversationRecord(
                conversation_id=conversation_id,
                session_id=session_id,
                session_key=session_key,
                active=active,
                archived_at=archived_at,
                title=title,
                bootstrapped_at=None,
                created_at=now,
                updated_at=now,
            )

    def get_conversation_by_session(self, session_id: str, session_key: str | None = None) -> ConversationRecord | None:
        """Get conversation by session ID and optional session key."""
        if session_key is not None:
            cursor = self.db.execute(
                """
                SELECT conversation_id, session_id, session_key, active, archived_at,
                       title, bootstrapped_at, created_at, updated_at
                FROM conversations
                WHERE session_id = ? AND session_key = ?
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (session_id, session_key),
            )
        else:
            cursor = self.db.execute(
                """
                SELECT conversation_id, session_id, session_key, active, archived_at,
                       title, bootstrapped_at, created_at, updated_at
                FROM conversations
                WHERE session_id = ? AND session_key IS NULL
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (session_id,),
            )

        row = cursor.fetchone()
        if not row:
            return None

        return ConversationRecord(
            conversation_id=row[0],
            session_id=row[1],
            session_key=row[2],
            active=bool(row[3]),
            archived_at=parse_utc_timestamp(row[4]) if row[4] else None,
            title=row[5],
            bootstrapped_at=parse_utc_timestamp(row[6]) if row[6] else None,
            created_at=parse_utc_timestamp(row[7]),
            updated_at=parse_utc_timestamp(row[8]),
        )

    def update_conversation_bootstrapped(self, conversation_id: int):
        """Mark conversation as bootstrapped."""
        now = datetime.utcnow()
        self.db.execute(
            """
            UPDATE conversations
            SET bootstrapped_at = ?, updated_at = ?
            WHERE conversation_id = ?
        """,
            (format_utc_timestamp(now), format_utc_timestamp(now), conversation_id),
        )

    def create_message(self, input_data: CreateMessageInput) -> MessageRecord:
        """Create a new message."""
        now = datetime.utcnow()

        # Generate identity hash if not provided
        identity_hash = input_data.identity_hash
        if not identity_hash:
            identity_hash = build_message_identity_hash(input_data.role, input_data.content)

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (
                    conversation_id, seq, role, content, token_count, identity_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    input_data.conversation_id,
                    input_data.seq,
                    input_data.role,
                    input_data.content,
                    input_data.token_count,
                    identity_hash,
                    format_utc_timestamp(now),
                ),
            )

            message_id = cursor.lastrowid

            # Update FTS table
            conn.execute(
                """
                INSERT INTO messages_fts (message_id, conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    message_id,
                    input_data.conversation_id,
                    input_data.role,
                    input_data.content,
                    format_utc_timestamp(now),
                ),
            )

        return MessageRecord(
            message_id=message_id,
            conversation_id=input_data.conversation_id,
            seq=input_data.seq,
            role=input_data.role,
            content=input_data.content,
            token_count=input_data.token_count,
            identity_hash=identity_hash,
            created_at=now,
        )

    def get_messages_by_conversation(
        self, conversation_id: int, since_seq: int | None = None, limit: int | None = None
    ) -> list[MessageRecord]:
        """Get messages for a conversation."""
        sql = """
            SELECT message_id, conversation_id, seq, role, content, token_count, identity_hash, created_at
            FROM messages
            WHERE conversation_id = ?
        """
        params = [conversation_id]

        if since_seq is not None:
            sql += " AND seq >= ?"
            params.append(since_seq)

        sql += " ORDER BY seq"

        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        cursor = self.db.execute(sql, params)

        messages = []
        for row in cursor.fetchall():
            messages.append(
                MessageRecord(
                    message_id=row[0],
                    conversation_id=row[1],
                    seq=row[2],
                    role=row[3],
                    content=row[4],
                    token_count=row[5],
                    identity_hash=row[6],
                    created_at=parse_utc_timestamp(row[7]),
                )
            )

        return messages

    def get_latest_message_seq(self, conversation_id: int) -> int:
        """Get the latest message sequence number for a conversation."""
        cursor = self.db.execute(
            """
            SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id = ?
        """,
            (conversation_id,),
        )

        result = cursor.fetchone()
        return result[0] if result else 0

    def search_messages(self, search_input: MessageSearchInput) -> list[MessageSearchResult]:
        """Search messages using FTS or regex."""
        if not search_input.query.strip():
            return []

        if search_input.mode == "full_text":
            return self._search_messages_fts(search_input)
        else:
            return self._search_messages_regex(search_input)

    def _search_messages_fts(self, search_input: MessageSearchInput) -> list[MessageSearchResult]:
        """Search messages using FTS5."""
        # Sanitize FTS query (basic implementation)
        query = search_input.query.replace('"', '""')

        sql = """
            SELECT m.message_id, m.conversation_id, m.role,
                   snippet(messages_fts, 3, '<mark>', '</mark>', '...', 32) as snippet,
                   m.created_at, rank
            FROM messages_fts
            JOIN messages m ON messages_fts.message_id = m.message_id
            WHERE messages_fts MATCH ?
        """
        params = [query]

        if search_input.conversation_id is not None:
            sql += " AND m.conversation_id = ?"
            params.append(search_input.conversation_id)

        if search_input.since:
            sql += " AND m.created_at >= ?"
            params.append(format_utc_timestamp(search_input.since))

        if search_input.before:
            sql += " AND m.created_at <= ?"
            params.append(format_utc_timestamp(search_input.before))

        # Sort order
        if search_input.sort == "newest":
            sql += " ORDER BY m.created_at DESC"
        elif search_input.sort == "oldest":
            sql += " ORDER BY m.created_at ASC"
        else:  # relevance
            sql += " ORDER BY rank"

        sql += f" LIMIT {search_input.limit}"

        cursor = self.db.execute(sql, params)

        results = []
        for row in cursor.fetchall():
            results.append(
                MessageSearchResult(
                    message_id=row[0],
                    conversation_id=row[1],
                    role=row[2],
                    snippet=row[3],
                    created_at=parse_utc_timestamp(row[4]),
                    rank=row[5] if len(row) > 5 else None,
                )
            )

        return results

    def _search_messages_regex(self, search_input: MessageSearchInput) -> list[MessageSearchResult]:
        """Search messages using LIKE (regex fallback)."""
        # Simple LIKE-based search as fallback
        pattern = f"%{search_input.query}%"

        sql = """
            SELECT message_id, conversation_id, role, content, created_at
            FROM messages
            WHERE content LIKE ?
        """
        params = [pattern]

        if search_input.conversation_id is not None:
            sql += " AND conversation_id = ?"
            params.append(search_input.conversation_id)

        if search_input.since:
            sql += " AND created_at >= ?"
            params.append(format_utc_timestamp(search_input.since))

        if search_input.before:
            sql += " AND created_at <= ?"
            params.append(format_utc_timestamp(search_input.before))

        sql += " ORDER BY created_at DESC"
        sql += f" LIMIT {search_input.limit}"

        cursor = self.db.execute(sql, params)

        results = []
        for row in cursor.fetchall():
            # Create simple snippet
            content = row[3]
            query_lower = search_input.query.lower()
            content_lower = content.lower()

            start = content_lower.find(query_lower)
            if start != -1:
                snippet_start = max(0, start - 50)
                snippet_end = min(len(content), start + len(search_input.query) + 50)
                snippet = content[snippet_start:snippet_end]
                if snippet_start > 0:
                    snippet = "..." + snippet
                if snippet_end < len(content):
                    snippet = snippet + "..."
            else:
                snippet = content[:100] + ("..." if len(content) > 100 else "")

            results.append(
                MessageSearchResult(
                    message_id=row[0],
                    conversation_id=row[1],
                    role=row[2],
                    snippet=snippet,
                    created_at=parse_utc_timestamp(row[4]),
                )
            )

        return results

    def create_message_part(self, message_id: int, input_data: CreateMessagePartInput) -> MessagePartRecord:
        """Create a new message part."""
        part_id = str(uuid4())

        self.db.execute(
            """
            INSERT INTO message_parts (
                part_id, message_id, session_id, part_type, ordinal,
                text_content, tool_call_id, tool_name, tool_input, tool_output, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                part_id,
                message_id,
                input_data.session_id,
                input_data.part_type,
                input_data.ordinal,
                input_data.text_content,
                input_data.tool_call_id,
                input_data.tool_name,
                input_data.tool_input,
                input_data.tool_output,
                input_data.metadata,
            ),
        )

        return MessagePartRecord(
            part_id=part_id,
            message_id=message_id,
            session_id=input_data.session_id,
            part_type=input_data.part_type,
            ordinal=input_data.ordinal,
            text_content=input_data.text_content,
            tool_call_id=input_data.tool_call_id,
            tool_name=input_data.tool_name,
            tool_input=input_data.tool_input,
            tool_output=input_data.tool_output,
            metadata=input_data.metadata,
        )

    def get_message_parts(self, message_id: int) -> list[MessagePartRecord]:
        """Get all parts for a message."""
        cursor = self.db.execute(
            """
            SELECT part_id, message_id, session_id, part_type, ordinal,
                   text_content, tool_call_id, tool_name, tool_input, tool_output, metadata
            FROM message_parts
            WHERE message_id = ?
            ORDER BY ordinal
        """,
            (message_id,),
        )

        parts = []
        for row in cursor.fetchall():
            parts.append(
                MessagePartRecord(
                    part_id=row[0],
                    message_id=row[1],
                    session_id=row[2],
                    part_type=row[3],
                    ordinal=row[4],
                    text_content=row[5],
                    tool_call_id=row[6],
                    tool_name=row[7],
                    tool_input=row[8],
                    tool_output=row[9],
                    metadata=row[10],
                )
            )

        return parts
