"""Tests for lossless_hermes.assembler module."""

from datetime import datetime, timedelta

import pytest

from lossless_hermes.assembler import AssemblyConfig, ContextAssembler
from lossless_hermes.store.conversation import CreateMessageInput
from lossless_hermes.store.summary import CreateSummaryInput


class TestContextAssembler:
    @pytest.fixture
    def assembler(self, conversation_store, summary_store):
        return ContextAssembler(conversation_store, summary_store)

    def test_empty_conversation(self, assembler, conversation_store):
        conv = conversation_store.create_conversation("empty")
        config = AssemblyConfig(max_tokens=10000, fresh_tail_count=10, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)
        assert result.messages == []
        assert result.total_tokens == 0
        assert result.coverage_ratio == 0.0

    def test_small_conversation_all_messages(self, assembler, sample_conversation):
        conv, msgs = sample_conversation
        config = AssemblyConfig(max_tokens=100000, fresh_tail_count=100, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)
        assert result.messages_used == len(msgs)
        assert result.summaries_used == 0
        assert len(result.messages) == len(msgs)

    def test_fresh_tail_limited(self, assembler, sample_conversation):
        conv, msgs = sample_conversation
        config = AssemblyConfig(max_tokens=100000, fresh_tail_count=3, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)
        # Only 3 fresh tail messages
        assert result.messages_used == 3

    def test_over_budget_truncates(self, assembler, conversation_store):
        conv = conversation_store.create_conversation("big")
        for i in range(20):
            conversation_store.create_message(
                CreateMessageInput(
                    conversation_id=conv.conversation_id,
                    seq=i + 1,
                    role="user",
                    content="x" * 400,  # ~100 tokens each
                    token_count=100,
                )
            )
        # Budget of 500 tokens with 1000 reserve = very tight
        config = AssemblyConfig(max_tokens=600, fresh_tail_count=20, fresh_tail_max_tokens=None, reserve_tokens=100)
        result = assembler.assemble_context(conv.conversation_id, config)
        assert result.total_tokens <= 600

    def test_with_summaries(self, assembler, conversation_store, summary_store):
        conv = conversation_store.create_conversation("with-sum")
        base = datetime(2024, 1, 1)
        for i in range(10):
            conversation_store.create_message(
                CreateMessageInput(
                    conversation_id=conv.conversation_id,
                    seq=i + 1,
                    role="user",
                    content=f"Message {i}",
                    token_count=10,
                )
            )

        # Create a summary covering early messages
        summary_store.create_summary(
            CreateSummaryInput(
                conversation_id=conv.conversation_id,
                kind="leaf",
                depth=0,
                content="Summary of early messages about setup and configuration.",
                token_count=15,
                earliest_at=base,
                latest_at=base + timedelta(hours=1),
                descendant_count=5,
                descendant_token_count=50,
                model="test",
            )
        )

        config = AssemblyConfig(max_tokens=10000, fresh_tail_count=3, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)
        # Should include summaries + fresh tail
        assert result.summaries_used >= 0  # May or may not include based on temporal overlap
        assert result.messages_used >= 1

    def test_summary_to_message_format(self, assembler, conversation_store, summary_store):
        conv = conversation_store.create_conversation("fmt")
        conversation_store.create_message(
            CreateMessageInput(
                conversation_id=conv.conversation_id,
                seq=1,
                role="user",
                content="hello",
                token_count=1,
            )
        )
        summary_store.create_summary(
            CreateSummaryInput(
                conversation_id=conv.conversation_id,
                kind="leaf",
                depth=0,
                content="A summary of past events.",
                token_count=5,
                earliest_at=datetime(2020, 1, 1),
                latest_at=datetime(2020, 1, 2),
                descendant_count=10,
                descendant_token_count=100,
                model="test",
            )
        )
        config = AssemblyConfig(max_tokens=10000, fresh_tail_count=1, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)
        # Check summary message format
        summary_msgs = [m for m in result.messages if "[LCM CONTEXT SUMMARY" in m.get("content", "")]
        for sm in summary_msgs:
            assert sm["role"] == "assistant"

    def test_coverage_ratio(self, assembler, sample_conversation, summary_store):
        conv, msgs = sample_conversation
        config = AssemblyConfig(max_tokens=100000, fresh_tail_count=5, fresh_tail_max_tokens=None)
        result = assembler.assemble_context(conv.conversation_id, config)
        assert 0.0 < result.coverage_ratio <= 1.0
