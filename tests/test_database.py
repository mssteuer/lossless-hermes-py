"""Tests for lossless_claw.db.connection and migration modules."""

import sqlite3
import pytest

from lossless_claw.db.connection import LcmDatabase
from lossless_claw.db.migration import (
    run_lcm_migrations, table_exists, get_table_columns,
    create_core_tables, create_fts_tables, create_indices,
)
from tests.conftest import make_default_config


class TestLcmDatabase:
    def test_create_connection(self, tmp_path):
        config = make_default_config(str(tmp_path / "test.db"))
        db = LcmDatabase(config)
        conn = db.connection
        assert conn is not None
        db.close()

    def test_memory_db(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        conn = db.connection
        assert conn is not None
        db.close()

    def test_wal_mode(self, tmp_path):
        config = make_default_config(str(tmp_path / "test.db"))
        db = LcmDatabase(config)
        cursor = db.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"
        db.close()

    def test_foreign_keys_on(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        cursor = db.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1
        db.close()

    def test_transaction_commit(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        db.execute("CREATE TABLE t (x INTEGER)")
        with db.transaction() as conn:
            conn.execute("INSERT INTO t VALUES (1)")
        cursor = db.execute("SELECT x FROM t")
        assert cursor.fetchone()[0] == 1
        db.close()

    def test_transaction_rollback(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        db.execute("CREATE TABLE t (x INTEGER)")
        with pytest.raises(ValueError):
            with db.transaction() as conn:
                conn.execute("INSERT INTO t VALUES (1)")
                raise ValueError("test error")
        cursor = db.execute("SELECT COUNT(*) FROM t")
        assert cursor.fetchone()[0] == 0
        db.close()

    def test_execute_with_params(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        db.execute("CREATE TABLE t (x INTEGER)")
        db.execute("INSERT INTO t VALUES (?)", (42,))
        cursor = db.execute("SELECT x FROM t WHERE x = ?", (42,))
        assert cursor.fetchone()[0] == 42
        db.close()

    def test_executemany(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        db.execute("CREATE TABLE t (x INTEGER)")
        db.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
        cursor = db.execute("SELECT COUNT(*) FROM t")
        assert cursor.fetchone()[0] == 3
        db.close()

    def test_close_idempotent(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        _ = db.connection
        db.close()
        db.close()  # Should not raise

    def test_connection_singleton(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        c1 = db.connection
        c2 = db.connection
        assert c1 is c2
        db.close()


class TestMigrations:
    def test_run_migrations(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        run_lcm_migrations(db)

        assert table_exists(db, "conversations")
        assert table_exists(db, "messages")
        assert table_exists(db, "summaries")
        assert table_exists(db, "summary_messages")
        assert table_exists(db, "summary_parents")
        assert table_exists(db, "context_items")
        assert table_exists(db, "message_parts")
        assert table_exists(db, "conversation_compaction_telemetry")
        assert table_exists(db, "compaction_maintenance")
        db.close()

    def test_fts_tables_created(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        run_lcm_migrations(db)
        assert table_exists(db, "messages_fts")
        assert table_exists(db, "summaries_fts")
        db.close()

    def test_idempotent_migrations(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        run_lcm_migrations(db)
        run_lcm_migrations(db)  # Should not raise
        assert table_exists(db, "conversations")
        db.close()

    def test_indices_created(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        run_lcm_migrations(db)
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        indices = [row[0] for row in cursor.fetchall()]
        assert "idx_messages_conversation_id" in indices
        assert "idx_summaries_conversation_id" in indices
        db.close()

    def test_columns_exist(self):
        config = make_default_config(":memory:")
        db = LcmDatabase(config)
        run_lcm_migrations(db)
        cols = get_table_columns(db, "summaries")
        assert "depth" in cols
        assert "model" in cols
        assert "descendant_count" in cols
        db.close()
