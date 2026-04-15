"""SQLite database schema migrations for LCM.

Handles schema creation and upgrades for the LCM database.
Based on the TypeScript migration.ts implementation.
"""

import sqlite3
import logging
from typing import List, Dict, Any
from .connection import LcmDatabase


logger = logging.getLogger(__name__)


def get_table_columns(db: LcmDatabase, table_name: str) -> List[str]:
    """Get the column names for a table."""
    cursor = db.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]  # row[1] is the column name


def table_exists(db: LcmDatabase, table_name: str) -> bool:
    """Check if a table exists."""
    cursor = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


def create_core_tables(db: LcmDatabase):
    """Create the core LCM tables."""
    
    # Conversations table
    db.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            session_key TEXT,
            active INTEGER DEFAULT 1,
            archived_at TEXT,
            title TEXT,
            bootstrapped_at TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(session_id, session_key)
        )
    """)
    
    # Messages table  
    db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id),
            seq INTEGER NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
            content TEXT NOT NULL,
            token_count INTEGER DEFAULT 0,
            identity_hash TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(conversation_id, seq)
        )
    """)
    
    # Message parts table (for detailed message structure)
    db.execute("""
        CREATE TABLE IF NOT EXISTS message_parts (
            part_id TEXT PRIMARY KEY,
            message_id INTEGER NOT NULL REFERENCES messages(message_id),
            session_id TEXT NOT NULL,
            part_type TEXT NOT NULL CHECK (part_type IN (
                'text', 'reasoning', 'tool', 'patch', 'file', 'subtask',
                'compaction', 'step_start', 'step_finish', 'snapshot', 'agent', 'retry'
            )),
            ordinal INTEGER NOT NULL,
            text_content TEXT,
            tool_call_id TEXT,
            tool_name TEXT,
            tool_input TEXT,
            tool_output TEXT,
            metadata TEXT,
            UNIQUE(message_id, ordinal)
        )
    """)
    
    # Summaries table (DAG nodes)
    db.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            summary_id TEXT PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id),
            kind TEXT NOT NULL CHECK (kind IN ('leaf', 'condensed')),
            depth INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            token_count INTEGER DEFAULT 0,
            earliest_at TEXT,
            latest_at TEXT,
            descendant_count INTEGER DEFAULT 0,
            descendant_token_count INTEGER DEFAULT 0,
            source_message_token_count INTEGER DEFAULT 0,
            file_ids TEXT,  -- JSON array
            model TEXT NOT NULL DEFAULT 'unknown',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    
    # Summary to message relationships (many-to-many)
    db.execute("""
        CREATE TABLE IF NOT EXISTS summary_messages (
            summary_id TEXT NOT NULL REFERENCES summaries(summary_id),
            message_id INTEGER NOT NULL REFERENCES messages(message_id),
            PRIMARY KEY (summary_id, message_id)
        )
    """)
    
    # Summary DAG structure (parent-child relationships)
    db.execute("""
        CREATE TABLE IF NOT EXISTS summary_parents (
            parent_id TEXT NOT NULL REFERENCES summaries(summary_id),
            child_id TEXT NOT NULL REFERENCES summaries(summary_id),
            PRIMARY KEY (parent_id, child_id)
        )
    """)
    
    # Context items (the assembled context window)
    db.execute("""
        CREATE TABLE IF NOT EXISTS context_items (
            id INTEGER PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id),
            position INTEGER NOT NULL,
            item_type TEXT NOT NULL CHECK (item_type IN ('message', 'summary')),
            message_id INTEGER REFERENCES messages(message_id),
            summary_id TEXT REFERENCES summaries(summary_id),
            UNIQUE(conversation_id, position),
            CHECK (
                (item_type = 'message' AND message_id IS NOT NULL AND summary_id IS NULL) OR
                (item_type = 'summary' AND summary_id IS NOT NULL AND message_id IS NULL)
            )
        )
    """)
    
    # Compaction telemetry for cache-aware compaction
    db.execute("""
        CREATE TABLE IF NOT EXISTS conversation_compaction_telemetry (
            conversation_id INTEGER PRIMARY KEY REFERENCES conversations(conversation_id),
            last_prompt_cache_read INTEGER DEFAULT 0,
            last_prompt_cache_write INTEGER DEFAULT 0,
            cache_state TEXT NOT NULL DEFAULT 'unknown' CHECK (cache_state IN ('unknown', 'cold', 'warm', 'hot')),
            retention TEXT,
            saw_explicit_break INTEGER DEFAULT 0,
            consecutive_cold_observations INTEGER DEFAULT 0,
            last_leaf_compaction_at TEXT,
            turns_since_leaf_compaction INTEGER DEFAULT 0,
            tokens_accumulated_since_leaf_compaction INTEGER DEFAULT 0,
            last_activity_band TEXT NOT NULL DEFAULT 'low' CHECK (last_activity_band IN ('low', 'medium', 'high')),
            last_api_call_at TEXT,
            last_cache_touch_at TEXT,
            provider TEXT,
            model TEXT
        )
    """)
    
    # Compaction maintenance tracking
    db.execute("""
        CREATE TABLE IF NOT EXISTS compaction_maintenance (
            conversation_id INTEGER PRIMARY KEY REFERENCES conversations(conversation_id),
            last_maintenance_at TEXT,
            maintenance_version INTEGER DEFAULT 1,
            maintenance_checksum TEXT
        )
    """)


def create_fts_tables(db: LcmDatabase):
    """Create FTS5 full-text search tables."""
    
    # FTS5 table for message content search
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            message_id UNINDEXED,
            conversation_id UNINDEXED,
            role UNINDEXED,
            content,
            created_at UNINDEXED
        )
    """)
    
    # FTS5 table for summary content search  
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
            summary_id UNINDEXED,
            conversation_id UNINDEXED,
            kind UNINDEXED,
            content,
            created_at UNINDEXED
        )
    """)


def create_indices(db: LcmDatabase):
    """Create database indices for performance."""
    
    # Conversation indices
    db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_active ON conversations(active)")
    
    # Message indices  
    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_seq ON messages(conversation_id, seq)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_identity_hash ON messages(identity_hash)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)")
    
    # Message parts indices
    db.execute("CREATE INDEX IF NOT EXISTS idx_message_parts_message_id ON message_parts(message_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_message_parts_session_id ON message_parts(session_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_message_parts_tool_call_id ON message_parts(tool_call_id)")
    
    # Summary indices
    db.execute("CREATE INDEX IF NOT EXISTS idx_summaries_conversation_id ON summaries(conversation_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_summaries_kind ON summaries(kind)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_summaries_depth ON summaries(depth)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_summaries_created_at ON summaries(created_at)")
    
    # Summary relationship indices
    db.execute("CREATE INDEX IF NOT EXISTS idx_summary_messages_message_id ON summary_messages(message_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_summary_parents_child_id ON summary_parents(child_id)")
    
    # Context items indices
    db.execute("CREATE INDEX IF NOT EXISTS idx_context_items_conversation ON context_items(conversation_id, position)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_context_items_message_id ON context_items(message_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_context_items_summary_id ON context_items(summary_id)")


def ensure_columns_exist(db: LcmDatabase):
    """Add any missing columns to existing tables."""
    
    # Check and add missing columns to summaries
    summary_columns = get_table_columns(db, "summaries")
    
    if "depth" not in summary_columns:
        db.execute("ALTER TABLE summaries ADD COLUMN depth INTEGER NOT NULL DEFAULT 0")
    
    if "earliest_at" not in summary_columns:
        db.execute("ALTER TABLE summaries ADD COLUMN earliest_at TEXT")
        
    if "latest_at" not in summary_columns:
        db.execute("ALTER TABLE summaries ADD COLUMN latest_at TEXT")
        
    if "descendant_count" not in summary_columns:
        db.execute("ALTER TABLE summaries ADD COLUMN descendant_count INTEGER NOT NULL DEFAULT 0")
        
    if "descendant_token_count" not in summary_columns:
        db.execute("ALTER TABLE summaries ADD COLUMN descendant_token_count INTEGER NOT NULL DEFAULT 0")
        
    if "source_message_token_count" not in summary_columns:
        db.execute("ALTER TABLE summaries ADD COLUMN source_message_token_count INTEGER NOT NULL DEFAULT 0")
        
    if "model" not in summary_columns:
        db.execute("ALTER TABLE summaries ADD COLUMN model TEXT NOT NULL DEFAULT 'unknown'")
    
    # Check and add missing columns to messages
    message_columns = get_table_columns(db, "messages")
    
    if "identity_hash" not in message_columns:
        db.execute("ALTER TABLE messages ADD COLUMN identity_hash TEXT")
    
    # Check and add missing telemetry columns
    if table_exists(db, "conversation_compaction_telemetry"):
        telemetry_columns = get_table_columns(db, "conversation_compaction_telemetry")
        
        missing_cols = [
            ("consecutive_cold_observations", "INTEGER NOT NULL DEFAULT 0"),
            ("last_leaf_compaction_at", "TEXT"),
            ("turns_since_leaf_compaction", "INTEGER NOT NULL DEFAULT 0"),
            ("tokens_accumulated_since_leaf_compaction", "INTEGER NOT NULL DEFAULT 0"),
            ("last_activity_band", "TEXT NOT NULL DEFAULT 'low'"),
            ("last_api_call_at", "TEXT"),
            ("last_cache_touch_at", "TEXT"),
            ("provider", "TEXT"),
            ("model", "TEXT"),
        ]
        
        for col_name, col_def in missing_cols:
            if col_name not in telemetry_columns:
                db.execute(f"ALTER TABLE conversation_compaction_telemetry ADD COLUMN {col_name} {col_def}")


def seed_fts_tables(db: LcmDatabase):
    """Populate FTS tables with existing data."""
    
    # Only seed if FTS tables are empty
    cursor = db.execute("SELECT COUNT(*) FROM messages_fts")
    if cursor.fetchone()[0] == 0:
        # Populate messages FTS
        db.execute("""
            INSERT INTO messages_fts (message_id, conversation_id, role, content, created_at)
            SELECT message_id, conversation_id, role, content, created_at
            FROM messages
        """)
    
    cursor = db.execute("SELECT COUNT(*) FROM summaries_fts")
    if cursor.fetchone()[0] == 0:
        # Populate summaries FTS
        db.execute("""
            INSERT INTO summaries_fts (summary_id, conversation_id, kind, content, created_at)
            SELECT summary_id, conversation_id, kind, content, created_at
            FROM summaries
        """)


def run_lcm_migrations(db: LcmDatabase):
    """Run all LCM database migrations.
    
    This creates the full schema and ensures all tables, indices, 
    and FTS tables are properly set up.
    """
    logger.info("Running LCM database migrations")
    
    try:
        with db.transaction() as conn:
            # Create core tables
            create_core_tables(db)
            
            # Ensure any missing columns exist (for upgrades)
            ensure_columns_exist(db)
            
            # Create indices
            create_indices(db)
            
            # Create FTS tables
            create_fts_tables(db)
            
            # Seed FTS tables with existing data
            seed_fts_tables(db)
            
            logger.info("LCM database migrations completed successfully")
            
    except Exception as e:
        logger.error(f"LCM database migration failed: {e}")
        raise