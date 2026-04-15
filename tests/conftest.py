"""Shared fixtures for LCM tests."""

import os
import pytest
from datetime import datetime, timedelta

from lossless_hermes.db.config import LcmConfig, CacheAwareCompactionConfig, DynamicLeafChunkTokensConfig, resolve_lcm_config
from lossless_hermes.db.connection import LcmDatabase
from lossless_hermes.db.migration import run_lcm_migrations
from lossless_hermes.store.conversation import ConversationStore, CreateMessageInput
from lossless_hermes.store.summary import SummaryStore, CreateSummaryInput
from lossless_hermes.compaction import CompactionEngine, CompactionConfig
from lossless_hermes.summarizer import SummaryOptions


def make_default_config(db_path=":memory:"):
    """Create a default LcmConfig for testing."""
    return LcmConfig(
        enabled=True,
        database_path=db_path,
        large_files_dir="/tmp/lcm-files",
        ignore_session_patterns=[],
        stateless_session_patterns=[],
        skip_stateless_sessions=True,
        context_threshold=0.75,
        fresh_tail_count=4,
        fresh_tail_max_tokens=None,
        new_session_retain_depth=2,
        leaf_min_fanout=2,
        condensed_min_fanout=2,
        condensed_min_fanout_hard=2,
        incremental_max_depth=1,
        leaf_chunk_tokens=5000,
        bootstrap_max_tokens=6000,
        leaf_target_tokens=2400,
        condensed_target_tokens=2000,
        max_expand_tokens=4000,
        large_file_token_threshold=25000,
        summary_provider="",
        summary_model="",
        large_file_summary_provider="",
        large_file_summary_model="",
        expansion_provider="",
        expansion_model="",
        delegation_timeout_ms=120000,
        summary_timeout_ms=60000,
        timezone="UTC",
        prune_heartbeat_ok=False,
        transcript_gc_enabled=False,
        proactive_threshold_compaction_mode="deferred",
        max_assembly_token_budget=None,
        summary_max_overage_factor=3,
        custom_instructions="",
        circuit_breaker_threshold=5,
        circuit_breaker_cooldown_ms=1800000,
        fallback_providers=[],
        cache_aware_compaction=CacheAwareCompactionConfig(
            enabled=False,
            cache_ttl_seconds=300,
            max_cold_cache_catchup_passes=2,
            hot_cache_pressure_factor=4,
            hot_cache_budget_headroom_ratio=0.2,
            cold_cache_observation_threshold=3,
        ),
        dynamic_leaf_chunk_tokens=DynamicLeafChunkTokensConfig(
            enabled=True,
            max=40000,
        ),
    )


@pytest.fixture
def lcm_config(tmp_path):
    """LcmConfig pointing at a temp db file."""
    return make_default_config(str(tmp_path / "test.db"))


@pytest.fixture
def memory_config():
    """LcmConfig with in-memory db."""
    return make_default_config(":memory:")


@pytest.fixture
def db(memory_config):
    """Initialized LcmDatabase with migrations run."""
    database = LcmDatabase(memory_config)
    run_lcm_migrations(database)
    yield database
    database.close()


@pytest.fixture
def conversation_store(db):
    return ConversationStore(db)


@pytest.fixture
def summary_store(db):
    return SummaryStore(db)


@pytest.fixture
def sample_conversation(conversation_store):
    """Create a sample conversation with messages."""
    conv = conversation_store.create_conversation("test-session")
    messages = []
    for i, (role, content) in enumerate([
        ("user", "Hello, can you help me with Python?"),
        ("assistant", "Of course! What do you need help with?"),
        ("user", "I need to parse JSON files and extract specific fields."),
        ("assistant", "You can use the json module. Here's an example..."),
        ("user", "What about handling nested objects?"),
        ("assistant", "For nested objects, you can use recursive access or jsonpath."),
        ("user", "Can you show me error handling for malformed JSON?"),
        ("assistant", "Sure, wrap json.loads in a try/except for JSONDecodeError."),
        ("user", "Thanks! Now how about writing JSON back to a file?"),
        ("assistant", "Use json.dump() with an open file handle and indent=2 for readability."),
    ]):
        msg = conversation_store.create_message(CreateMessageInput(
            conversation_id=conv.conversation_id,
            seq=i + 1,
            role=role,
            content=content,
            token_count=len(content) // 4,
        ))
        messages.append(msg)
    return conv, messages


class MockSyncSummarizer:
    """Deterministic mock summarizer for tests."""
    model = "mock-model"

    def summarize(self, text, aggressive=False, options=None):
        options = options or SummaryOptions()
        prefix = "CONDENSED" if options.is_condensed else "LEAF"
        depth = options.depth
        word_count = len(text.split())
        return f"[{prefix} d={depth}] Summary of {word_count} words. Key points preserved."
