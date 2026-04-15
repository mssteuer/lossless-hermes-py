"""Summary and DAG management for LCM.

Handles storage and retrieval of summaries, DAG relationships,
and context assembly state.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from ..db.connection import LcmDatabase
from .conversation import format_utc_timestamp, parse_utc_timestamp

logger = logging.getLogger(__name__)


@dataclass
class SummaryRecord:
    summary_id: str
    conversation_id: int
    kind: str  # 'leaf' or 'condensed'
    depth: int
    content: str
    token_count: int
    earliest_at: datetime | None
    latest_at: datetime | None
    descendant_count: int
    descendant_token_count: int
    source_message_token_count: int
    file_ids: list[str]
    model: str
    created_at: datetime


@dataclass
class CreateSummaryInput:
    conversation_id: int
    kind: str
    depth: int
    content: str
    token_count: int
    earliest_at: datetime | None = None
    latest_at: datetime | None = None
    descendant_count: int = 0
    descendant_token_count: int = 0
    source_message_token_count: int = 0
    file_ids: list[str] | None = None
    model: str = "unknown"


@dataclass
class ContextItemRecord:
    id: int
    conversation_id: int
    position: int
    item_type: str  # 'message' or 'summary'
    message_id: int | None
    summary_id: str | None


class SummaryStore:
    """Store for managing summaries and DAG relationships."""

    def __init__(self, db: LcmDatabase):
        self.db = db

    def create_summary(self, input_data: CreateSummaryInput) -> SummaryRecord:
        """Create a new summary."""
        summary_id = str(uuid4())
        now = datetime.utcnow()
        file_ids = input_data.file_ids or []

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO summaries (
                    summary_id, conversation_id, kind, depth, content, token_count,
                    earliest_at, latest_at, descendant_count, descendant_token_count,
                    source_message_token_count, file_ids, model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    summary_id,
                    input_data.conversation_id,
                    input_data.kind,
                    input_data.depth,
                    input_data.content,
                    input_data.token_count,
                    format_utc_timestamp(input_data.earliest_at) if input_data.earliest_at else None,
                    format_utc_timestamp(input_data.latest_at) if input_data.latest_at else None,
                    input_data.descendant_count,
                    input_data.descendant_token_count,
                    input_data.source_message_token_count,
                    json.dumps(file_ids),
                    input_data.model,
                    format_utc_timestamp(now),
                ),
            )

            # Update FTS table
            conn.execute(
                """
                INSERT INTO summaries_fts (summary_id, conversation_id, kind, content, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    summary_id,
                    input_data.conversation_id,
                    input_data.kind,
                    input_data.content,
                    format_utc_timestamp(now),
                ),
            )

        return SummaryRecord(
            summary_id=summary_id,
            conversation_id=input_data.conversation_id,
            kind=input_data.kind,
            depth=input_data.depth,
            content=input_data.content,
            token_count=input_data.token_count,
            earliest_at=input_data.earliest_at,
            latest_at=input_data.latest_at,
            descendant_count=input_data.descendant_count,
            descendant_token_count=input_data.descendant_token_count,
            source_message_token_count=input_data.source_message_token_count,
            file_ids=file_ids,
            model=input_data.model,
            created_at=now,
        )

    def get_summary(self, summary_id: str) -> SummaryRecord | None:
        """Get a summary by ID."""
        cursor = self.db.execute(
            """
            SELECT summary_id, conversation_id, kind, depth, content, token_count,
                   earliest_at, latest_at, descendant_count, descendant_token_count,
                   source_message_token_count, file_ids, model, created_at
            FROM summaries
            WHERE summary_id = ?
        """,
            (summary_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        file_ids = []
        try:
            if row[11]:  # file_ids column
                file_ids = json.loads(row[11])
        except (json.JSONDecodeError, TypeError):
            pass

        return SummaryRecord(
            summary_id=row[0],
            conversation_id=row[1],
            kind=row[2],
            depth=row[3],
            content=row[4],
            token_count=row[5],
            earliest_at=parse_utc_timestamp(row[6]) if row[6] else None,
            latest_at=parse_utc_timestamp(row[7]) if row[7] else None,
            descendant_count=row[8],
            descendant_token_count=row[9],
            source_message_token_count=row[10],
            file_ids=file_ids,
            model=row[12],
            created_at=parse_utc_timestamp(row[13]),
        )

    def get_summaries_by_conversation(
        self, conversation_id: int, kind: str | None = None, depth: int | None = None
    ) -> list[SummaryRecord]:
        """Get summaries for a conversation."""
        sql = """
            SELECT summary_id, conversation_id, kind, depth, content, token_count,
                   earliest_at, latest_at, descendant_count, descendant_token_count,
                   source_message_token_count, file_ids, model, created_at
            FROM summaries
            WHERE conversation_id = ?
        """
        params = [conversation_id]

        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)

        if depth is not None:
            sql += " AND depth = ?"
            params.append(depth)

        sql += " ORDER BY created_at"

        cursor = self.db.execute(sql, params)

        summaries = []
        for row in cursor.fetchall():
            file_ids = []
            try:
                if row[11]:
                    file_ids = json.loads(row[11])
            except (json.JSONDecodeError, TypeError):
                pass

            summaries.append(
                SummaryRecord(
                    summary_id=row[0],
                    conversation_id=row[1],
                    kind=row[2],
                    depth=row[3],
                    content=row[4],
                    token_count=row[5],
                    earliest_at=parse_utc_timestamp(row[6]) if row[6] else None,
                    latest_at=parse_utc_timestamp(row[7]) if row[7] else None,
                    descendant_count=row[8],
                    descendant_token_count=row[9],
                    source_message_token_count=row[10],
                    file_ids=file_ids,
                    model=row[12],
                    created_at=parse_utc_timestamp(row[13]),
                )
            )

        return summaries

    def add_summary_message(self, summary_id: str, message_id: int):
        """Link a message to a summary."""
        self.db.execute(
            """
            INSERT OR IGNORE INTO summary_messages (summary_id, message_id)
            VALUES (?, ?)
        """,
            (summary_id, message_id),
        )

    def add_summary_messages(self, summary_id: str, message_ids: list[int]):
        """Link multiple messages to a summary."""
        if not message_ids:
            return

        with self.db.transaction() as conn:
            for message_id in message_ids:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO summary_messages (summary_id, message_id)
                    VALUES (?, ?)
                """,
                    (summary_id, message_id),
                )

    def get_summary_messages(self, summary_id: str) -> list[int]:
        """Get message IDs linked to a summary."""
        cursor = self.db.execute(
            """
            SELECT message_id FROM summary_messages
            WHERE summary_id = ?
            ORDER BY message_id
        """,
            (summary_id,),
        )

        return [row[0] for row in cursor.fetchall()]

    def add_summary_parent(self, parent_id: str, child_id: str):
        """Add a parent-child relationship between summaries."""
        self.db.execute(
            """
            INSERT OR IGNORE INTO summary_parents (parent_id, child_id)
            VALUES (?, ?)
        """,
            (parent_id, child_id),
        )

    def get_summary_parents(self, summary_id: str) -> list[str]:
        """Get parent summary IDs for a summary."""
        cursor = self.db.execute(
            """
            SELECT parent_id FROM summary_parents
            WHERE child_id = ?
        """,
            (summary_id,),
        )

        return [row[0] for row in cursor.fetchall()]

    def get_summary_children(self, summary_id: str) -> list[str]:
        """Get child summary IDs for a summary."""
        cursor = self.db.execute(
            """
            SELECT child_id FROM summary_parents
            WHERE parent_id = ?
        """,
            (summary_id,),
        )

        return [row[0] for row in cursor.fetchall()]

    def get_dag_roots(self, conversation_id: int, depth: int = 0) -> list[str]:
        """Get root summaries (no parents) at a given depth."""
        cursor = self.db.execute(
            """
            SELECT s.summary_id
            FROM summaries s
            LEFT JOIN summary_parents sp ON s.summary_id = sp.child_id
            WHERE s.conversation_id = ? AND s.depth = ? AND sp.child_id IS NULL
            ORDER BY s.created_at
        """,
            (conversation_id, depth),
        )

        return [row[0] for row in cursor.fetchall()]

    def get_dag_leaves(self, conversation_id: int, depth: int) -> list[str]:
        """Get leaf summaries (no children) at a given depth."""
        cursor = self.db.execute(
            """
            SELECT s.summary_id
            FROM summaries s
            LEFT JOIN summary_parents sp ON s.summary_id = sp.parent_id
            WHERE s.conversation_id = ? AND s.depth = ? AND sp.parent_id IS NULL
            ORDER BY s.created_at
        """,
            (conversation_id, depth),
        )

        return [row[0] for row in cursor.fetchall()]

    def update_summary_metadata(
        self,
        summary_id: str,
        descendant_count: int | None = None,
        descendant_token_count: int | None = None,
        earliest_at: datetime | None = None,
        latest_at: datetime | None = None,
    ):
        """Update summary metadata."""
        updates = []
        params = []

        if descendant_count is not None:
            updates.append("descendant_count = ?")
            params.append(descendant_count)

        if descendant_token_count is not None:
            updates.append("descendant_token_count = ?")
            params.append(descendant_token_count)

        if earliest_at is not None:
            updates.append("earliest_at = ?")
            params.append(format_utc_timestamp(earliest_at))

        if latest_at is not None:
            updates.append("latest_at = ?")
            params.append(format_utc_timestamp(latest_at))

        if not updates:
            return

        params.append(summary_id)
        sql = f"UPDATE summaries SET {', '.join(updates)} WHERE summary_id = ?"

        self.db.execute(sql, params)

    def clear_context_items(self, conversation_id: int):
        """Clear all context items for a conversation."""
        self.db.execute(
            """
            DELETE FROM context_items WHERE conversation_id = ?
        """,
            (conversation_id,),
        )

    def add_context_item(
        self,
        conversation_id: int,
        position: int,
        item_type: str,
        message_id: int | None = None,
        summary_id: str | None = None,
    ) -> ContextItemRecord:
        """Add a context item."""
        cursor = self.db.execute(
            """
            INSERT INTO context_items (conversation_id, position, item_type, message_id, summary_id)
            VALUES (?, ?, ?, ?, ?)
        """,
            (conversation_id, position, item_type, message_id, summary_id),
        )

        return ContextItemRecord(
            id=cursor.lastrowid,
            conversation_id=conversation_id,
            position=position,
            item_type=item_type,
            message_id=message_id,
            summary_id=summary_id,
        )

    def get_context_items(self, conversation_id: int) -> list[ContextItemRecord]:
        """Get context items for a conversation in order."""
        cursor = self.db.execute(
            """
            SELECT id, conversation_id, position, item_type, message_id, summary_id
            FROM context_items
            WHERE conversation_id = ?
            ORDER BY position
        """,
            (conversation_id,),
        )

        items = []
        for row in cursor.fetchall():
            items.append(
                ContextItemRecord(
                    id=row[0],
                    conversation_id=row[1],
                    position=row[2],
                    item_type=row[3],
                    message_id=row[4],
                    summary_id=row[5],
                )
            )

        return items

    def get_summary_depth_stats(self, conversation_id: int) -> dict[int, dict[str, int]]:
        """Get summary statistics by depth."""
        cursor = self.db.execute(
            """
            SELECT depth, kind, COUNT(*), SUM(token_count)
            FROM summaries
            WHERE conversation_id = ?
            GROUP BY depth, kind
            ORDER BY depth, kind
        """,
            (conversation_id,),
        )

        stats = {}
        for row in cursor.fetchall():
            depth = row[0]
            kind = row[1]
            count = row[2]
            tokens = row[3]

            if depth not in stats:
                stats[depth] = {}

            stats[depth][kind] = {"count": count, "tokens": tokens}

        return stats

    def search_summaries(self, conversation_id: int | None, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search summaries using FTS."""
        if not query.strip():
            return []

        # Sanitize query
        query = query.replace('"', '""')

        sql = """
            SELECT s.summary_id, s.conversation_id, s.kind, s.depth,
                   snippet(summaries_fts, 3, '<mark>', '</mark>', '...', 32) as snippet,
                   s.created_at
            FROM summaries_fts
            JOIN summaries s ON summaries_fts.summary_id = s.summary_id
            WHERE summaries_fts MATCH ?
        """
        params = [query]

        if conversation_id is not None:
            sql += " AND s.conversation_id = ?"
            params.append(conversation_id)

        sql += " ORDER BY rank"
        sql += f" LIMIT {limit}"

        cursor = self.db.execute(sql, params)

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "summary_id": row[0],
                    "conversation_id": row[1],
                    "kind": row[2],
                    "depth": row[3],
                    "snippet": row[4],
                    "created_at": parse_utc_timestamp(row[5]),
                }
            )

        return results

    def delete_summary(self, summary_id: str):
        """Delete a summary and its relationships."""
        with self.db.transaction() as conn:
            # Delete from FTS
            conn.execute("DELETE FROM summaries_fts WHERE summary_id = ?", (summary_id,))

            # Delete relationships
            conn.execute("DELETE FROM summary_messages WHERE summary_id = ?", (summary_id,))
            conn.execute(
                "DELETE FROM summary_parents WHERE parent_id = ? OR child_id = ?",
                (summary_id, summary_id),
            )

            # Delete context items
            conn.execute("DELETE FROM context_items WHERE summary_id = ?", (summary_id,))

            # Delete summary
            conn.execute("DELETE FROM summaries WHERE summary_id = ?", (summary_id,))
