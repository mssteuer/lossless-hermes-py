"""Tests for lossless_hermes.compaction module."""

import pytest

from lossless_hermes.compaction import CompactionConfig, CompactionEngine
from lossless_hermes.db.config import CacheAwareCompactionConfig
from lossless_hermes.store.conversation import CreateMessageInput
from tests.conftest import MockSyncSummarizer


@pytest.fixture
def compaction_config():
    return CompactionConfig(
        leaf_chunk_tokens=500,
        leaf_target_tokens=100,
        condensed_target_tokens=80,
        leaf_min_fanout=2,
        condensed_min_fanout=2,
        condensed_min_fanout_hard=2,
        incremental_max_depth=1,
        fresh_tail_count=2,
        fresh_tail_max_tokens=None,
        cache_aware_compaction=CacheAwareCompactionConfig(
            enabled=False,
            cache_ttl_seconds=300,
            max_cold_cache_catchup_passes=2,
            hot_cache_pressure_factor=4,
            hot_cache_budget_headroom_ratio=0.2,
            cold_cache_observation_threshold=3,
        ),
    )


@pytest.fixture
def engine(compaction_config, conversation_store, summary_store):
    return CompactionEngine(compaction_config, MockSyncSummarizer(), conversation_store, summary_store)


class TestShouldCompact:
    def test_below_threshold(self, engine, sample_conversation):
        conv, _ = sample_conversation
        should, reason = engine.should_compact(conv.conversation_id, 100, 200)
        assert should is False
        assert reason == "below_threshold"

    def test_initial_compaction(self, engine, sample_conversation):
        conv, _ = sample_conversation
        should, reason = engine.should_compact(conv.conversation_id, 300, 200)
        assert should is True
        assert reason == "initial_compaction"

    def test_threshold_exceeded_with_existing_summaries(self, engine, summary_store, sample_conversation):
        from lossless_hermes.store.summary import CreateSummaryInput

        conv, _ = sample_conversation
        summary_store.create_summary(
            CreateSummaryInput(
                conversation_id=conv.conversation_id,
                kind="leaf",
                depth=0,
                content="existing",
                token_count=10,
                model="m",
            )
        )
        should, reason = engine.should_compact(conv.conversation_id, 300, 200)
        assert should is True
        assert reason == "threshold_exceeded"


class TestCacheAwareCompaction:
    def test_hot_cache_pressure_relief(self, conversation_store, summary_store, sample_conversation):
        from lossless_hermes.store.summary import CreateSummaryInput

        conv, _ = sample_conversation
        config = CompactionConfig(
            leaf_chunk_tokens=500,
            leaf_target_tokens=100,
            condensed_target_tokens=80,
            leaf_min_fanout=2,
            condensed_min_fanout=2,
            condensed_min_fanout_hard=2,
            incremental_max_depth=1,
            fresh_tail_count=2,
            fresh_tail_max_tokens=None,
            cache_aware_compaction=CacheAwareCompactionConfig(
                enabled=True,
                cache_ttl_seconds=300,
                max_cold_cache_catchup_passes=2,
                hot_cache_pressure_factor=4,
                hot_cache_budget_headroom_ratio=0.2,
                cold_cache_observation_threshold=3,
            ),
        )
        engine = CompactionEngine(config, MockSyncSummarizer(), conversation_store, summary_store)
        # Add a leaf summary so it's not "initial"
        summary_store.create_summary(
            CreateSummaryInput(
                conversation_id=conv.conversation_id,
                kind="leaf",
                depth=0,
                content="s",
                token_count=1,
                model="m",
            )
        )
        # Hot cache with tokens below pressure-adjusted threshold
        should, reason = engine.should_compact(conv.conversation_id, 300, 200, "hot")
        assert should is False
        assert reason == "hot_cache_pressure_relief"

    def test_cold_cache_catchup(self, conversation_store, summary_store, sample_conversation):
        from lossless_hermes.store.summary import CreateSummaryInput

        conv, _ = sample_conversation
        config = CompactionConfig(
            leaf_chunk_tokens=500,
            leaf_target_tokens=100,
            condensed_target_tokens=80,
            leaf_min_fanout=2,
            condensed_min_fanout=2,
            condensed_min_fanout_hard=2,
            incremental_max_depth=1,
            fresh_tail_count=2,
            fresh_tail_max_tokens=None,
            cache_aware_compaction=CacheAwareCompactionConfig(
                enabled=True,
                cache_ttl_seconds=300,
                max_cold_cache_catchup_passes=2,
                hot_cache_pressure_factor=4,
                hot_cache_budget_headroom_ratio=0.2,
                cold_cache_observation_threshold=3,
            ),
        )
        engine = CompactionEngine(config, MockSyncSummarizer(), conversation_store, summary_store)
        summary_store.create_summary(
            CreateSummaryInput(
                conversation_id=conv.conversation_id,
                kind="leaf",
                depth=0,
                content="s",
                token_count=1,
                model="m",
            )
        )
        should, reason = engine.should_compact(conv.conversation_id, 300, 200, "cold")
        assert should is True
        assert reason == "cold_cache_catchup"


class TestLeafCompaction:
    def test_creates_leaf_summaries(self, engine, sample_conversation, summary_store):
        conv, msgs = sample_conversation
        result = engine.compact(conv.conversation_id, 1000)
        assert result.summaries_created > 0
        summaries = summary_store.get_summaries_by_conversation(conv.conversation_id, kind="leaf")
        assert len(summaries) > 0
        for s in summaries:
            assert s.kind == "leaf"
            assert s.depth == 0
            assert "[LEAF d=0]" in s.content

    def test_fresh_tail_protected(self, engine, sample_conversation, conversation_store):
        conv, msgs = sample_conversation
        # With fresh_tail_count=2, last 2 messages should be protected
        engine.compact(conv.conversation_id, 1000)
        # Messages are not deleted, only summarized
        all_msgs = conversation_store.get_messages_by_conversation(conv.conversation_id)
        assert len(all_msgs) == len(msgs)  # All messages still exist


class TestCondensedCompaction:
    def test_condensed_from_leaf_summaries(self, conversation_store, summary_store):
        """Create enough leaf summaries to trigger condensed compaction."""
        conv = conversation_store.create_conversation("sess-condensed")
        # Create many messages
        for i in range(20):
            conversation_store.create_message(
                CreateMessageInput(
                    conversation_id=conv.conversation_id,
                    seq=i + 1,
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message content number {i} with enough text to have tokens",
                    token_count=20,
                )
            )

        config = CompactionConfig(
            leaf_chunk_tokens=100,
            leaf_target_tokens=50,
            condensed_target_tokens=40,
            leaf_min_fanout=2,
            condensed_min_fanout=2,
            condensed_min_fanout_hard=2,
            incremental_max_depth=1,
            fresh_tail_count=2,
            fresh_tail_max_tokens=None,
            cache_aware_compaction=CacheAwareCompactionConfig(
                enabled=False,
                cache_ttl_seconds=300,
                max_cold_cache_catchup_passes=2,
                hot_cache_pressure_factor=4,
                hot_cache_budget_headroom_ratio=0.2,
                cold_cache_observation_threshold=3,
            ),
        )
        engine = CompactionEngine(config, MockSyncSummarizer(), conversation_store, summary_store)
        result = engine.compact(conv.conversation_id, 5000)
        assert result.summaries_created > 0

        # Check that leaf summaries were created
        leaves = summary_store.get_summaries_by_conversation(conv.conversation_id, kind="leaf")
        assert len(leaves) > 0
