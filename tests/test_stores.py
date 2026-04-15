"""Tests for ConversationStore and SummaryStore."""

import pytest
from datetime import datetime

from lossless_hermes.store.conversation import (
    ConversationStore, CreateMessageInput, CreateMessagePartInput,
    MessageSearchInput,
)
from lossless_hermes.store.summary import SummaryStore, CreateSummaryInput


class TestConversationStore:
    def test_create_conversation(self, conversation_store):
        conv = conversation_store.create_conversation("sess-1")
        assert conv.session_id == "sess-1"
        assert conv.active is True
        assert conv.conversation_id > 0

    def test_get_conversation_by_session(self, conversation_store):
        conversation_store.create_conversation("sess-2")
        found = conversation_store.get_conversation_by_session("sess-2")
        assert found is not None
        assert found.session_id == "sess-2"

    def test_get_nonexistent_conversation(self, conversation_store):
        assert conversation_store.get_conversation_by_session("nope") is None

    def test_session_with_key(self, conversation_store):
        conversation_store.create_conversation("sess-3", session_key="key-a")
        found = conversation_store.get_conversation_by_session("sess-3", "key-a")
        assert found is not None
        not_found = conversation_store.get_conversation_by_session("sess-3", "key-b")
        assert not_found is None

    def test_create_message(self, conversation_store):
        conv = conversation_store.create_conversation("sess-msg")
        msg = conversation_store.create_message(CreateMessageInput(
            conversation_id=conv.conversation_id,
            seq=1,
            role="user",
            content="Hello!",
            token_count=2,
        ))
        assert msg.message_id > 0
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.identity_hash  # Auto-generated

    def test_get_messages_by_conversation(self, conversation_store):
        conv = conversation_store.create_conversation("sess-msgs")
        for i in range(3):
            conversation_store.create_message(CreateMessageInput(
                conversation_id=conv.conversation_id,
                seq=i + 1,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                token_count=5,
            ))
        msgs = conversation_store.get_messages_by_conversation(conv.conversation_id)
        assert len(msgs) == 3
        assert msgs[0].seq == 1
        assert msgs[2].seq == 3

    def test_get_messages_since_seq(self, conversation_store):
        conv = conversation_store.create_conversation("sess-since")
        for i in range(5):
            conversation_store.create_message(CreateMessageInput(
                conversation_id=conv.conversation_id, seq=i + 1,
                role="user", content=f"M{i}", token_count=1,
            ))
        msgs = conversation_store.get_messages_by_conversation(conv.conversation_id, since_seq=3)
        assert len(msgs) == 3

    def test_get_messages_with_limit(self, conversation_store):
        conv = conversation_store.create_conversation("sess-limit")
        for i in range(5):
            conversation_store.create_message(CreateMessageInput(
                conversation_id=conv.conversation_id, seq=i + 1,
                role="user", content=f"M{i}", token_count=1,
            ))
        msgs = conversation_store.get_messages_by_conversation(conv.conversation_id, limit=2)
        assert len(msgs) == 2

    def test_get_latest_message_seq(self, conversation_store):
        conv = conversation_store.create_conversation("sess-seq")
        assert conversation_store.get_latest_message_seq(conv.conversation_id) == 0
        conversation_store.create_message(CreateMessageInput(
            conversation_id=conv.conversation_id, seq=1,
            role="user", content="hi", token_count=1,
        ))
        assert conversation_store.get_latest_message_seq(conv.conversation_id) == 1

    def test_update_bootstrapped(self, conversation_store):
        conv = conversation_store.create_conversation("sess-boot")
        assert conv.bootstrapped_at is None
        conversation_store.update_conversation_bootstrapped(conv.conversation_id)
        updated = conversation_store.get_conversation_by_session("sess-boot")
        assert updated.bootstrapped_at is not None

    def test_search_messages_fts(self, conversation_store):
        conv = conversation_store.create_conversation("sess-fts")
        conversation_store.create_message(CreateMessageInput(
            conversation_id=conv.conversation_id, seq=1,
            role="user", content="Python is a great programming language", token_count=10,
        ))
        conversation_store.create_message(CreateMessageInput(
            conversation_id=conv.conversation_id, seq=2,
            role="assistant", content="JavaScript is also popular", token_count=8,
        ))
        results = conversation_store.search_messages(MessageSearchInput(
            conversation_id=conv.conversation_id,
            query="Python",
            mode="full_text",
        ))
        assert len(results) >= 1
        assert any("Python" in r.snippet for r in results)

    def test_search_messages_empty_query(self, conversation_store):
        results = conversation_store.search_messages(MessageSearchInput(query=""))
        assert results == []

    def test_create_message_part(self, conversation_store):
        conv = conversation_store.create_conversation("sess-parts")
        msg = conversation_store.create_message(CreateMessageInput(
            conversation_id=conv.conversation_id, seq=1,
            role="assistant", content="result", token_count=1,
        ))
        part = conversation_store.create_message_part(
            msg.message_id,
            CreateMessagePartInput(
                session_id="sess-parts",
                part_type="text",
                ordinal=0,
                text_content="result",
            )
        )
        assert part.part_id
        assert part.part_type == "text"

    def test_get_message_parts(self, conversation_store):
        conv = conversation_store.create_conversation("sess-getparts")
        msg = conversation_store.create_message(CreateMessageInput(
            conversation_id=conv.conversation_id, seq=1,
            role="assistant", content="x", token_count=1,
        ))
        conversation_store.create_message_part(
            msg.message_id,
            CreateMessagePartInput(session_id="s", part_type="text", ordinal=0, text_content="a")
        )
        conversation_store.create_message_part(
            msg.message_id,
            CreateMessagePartInput(session_id="s", part_type="text", ordinal=1, text_content="b")
        )
        parts = conversation_store.get_message_parts(msg.message_id)
        assert len(parts) == 2
        assert parts[0].ordinal == 0
        assert parts[1].ordinal == 1


class TestSummaryStore:
    def test_create_summary(self, summary_store, sample_conversation):
        conv, _ = sample_conversation
        summary = summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="leaf",
            depth=0,
            content="Summary of conversation about Python JSON.",
            token_count=10,
            model="test-model",
        ))
        assert summary.summary_id
        assert summary.kind == "leaf"
        assert summary.depth == 0

    def test_get_summary(self, summary_store, sample_conversation):
        conv, _ = sample_conversation
        created = summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="leaf", depth=0,
            content="Test summary", token_count=5, model="m",
        ))
        fetched = summary_store.get_summary(created.summary_id)
        assert fetched is not None
        assert fetched.content == "Test summary"

    def test_get_summary_not_found(self, summary_store):
        assert summary_store.get_summary("nonexistent-id") is None

    def test_get_summaries_by_conversation(self, summary_store, sample_conversation):
        conv, _ = sample_conversation
        for i in range(3):
            summary_store.create_summary(CreateSummaryInput(
                conversation_id=conv.conversation_id,
                kind="leaf", depth=0,
                content=f"Summary {i}", token_count=5, model="m",
            ))
        summaries = summary_store.get_summaries_by_conversation(conv.conversation_id)
        assert len(summaries) == 3

    def test_filter_by_kind(self, summary_store, sample_conversation):
        conv, _ = sample_conversation
        summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="leaf", depth=0, content="leaf", token_count=5, model="m",
        ))
        summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="condensed", depth=1, content="condensed", token_count=5, model="m",
        ))
        leaves = summary_store.get_summaries_by_conversation(conv.conversation_id, kind="leaf")
        assert len(leaves) == 1
        assert leaves[0].kind == "leaf"

    def test_summary_messages_link(self, summary_store, sample_conversation):
        conv, msgs = sample_conversation
        s = summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="leaf", depth=0, content="s", token_count=1, model="m",
        ))
        msg_ids = [msgs[0].message_id, msgs[1].message_id]
        summary_store.add_summary_messages(s.summary_id, msg_ids)
        linked = summary_store.get_summary_messages(s.summary_id)
        assert set(linked) == set(msg_ids)

    def test_dag_parent_child(self, summary_store, sample_conversation):
        conv, _ = sample_conversation
        child = summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="leaf", depth=0, content="child", token_count=1, model="m",
        ))
        parent = summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="condensed", depth=1, content="parent", token_count=1, model="m",
        ))
        summary_store.add_summary_parent(parent.summary_id, child.summary_id)
        parents = summary_store.get_summary_parents(child.summary_id)
        assert parent.summary_id in parents
        children = summary_store.get_summary_children(parent.summary_id)
        assert child.summary_id in children

    def test_dag_roots(self, summary_store, sample_conversation):
        conv, _ = sample_conversation
        s1 = summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="leaf", depth=0, content="root1", token_count=1, model="m",
        ))
        s2 = summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="leaf", depth=0, content="root2", token_count=1, model="m",
        ))
        roots = summary_store.get_dag_roots(conv.conversation_id, depth=0)
        assert s1.summary_id in roots
        assert s2.summary_id in roots

    def test_context_items(self, summary_store, sample_conversation):
        conv, msgs = sample_conversation
        summary_store.add_context_item(
            conv.conversation_id, 0, "message", message_id=msgs[0].message_id
        )
        items = summary_store.get_context_items(conv.conversation_id)
        assert len(items) == 1
        assert items[0].item_type == "message"

    def test_clear_context_items(self, summary_store, sample_conversation):
        conv, msgs = sample_conversation
        summary_store.add_context_item(
            conv.conversation_id, 0, "message", message_id=msgs[0].message_id
        )
        summary_store.clear_context_items(conv.conversation_id)
        assert summary_store.get_context_items(conv.conversation_id) == []

    def test_delete_summary(self, summary_store, sample_conversation):
        conv, msgs = sample_conversation
        s = summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="leaf", depth=0, content="del me", token_count=1, model="m",
        ))
        summary_store.add_summary_messages(s.summary_id, [msgs[0].message_id])
        summary_store.delete_summary(s.summary_id)
        assert summary_store.get_summary(s.summary_id) is None
        assert summary_store.get_summary_messages(s.summary_id) == []

    def test_search_summaries(self, summary_store, sample_conversation):
        conv, _ = sample_conversation
        summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="leaf", depth=0,
            content="Python JSON parsing techniques and error handling",
            token_count=10, model="m",
        ))
        results = summary_store.search_summaries(conv.conversation_id, "Python")
        assert len(results) >= 1

    def test_depth_stats(self, summary_store, sample_conversation):
        conv, _ = sample_conversation
        summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="leaf", depth=0, content="a", token_count=10, model="m",
        ))
        summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="leaf", depth=0, content="b", token_count=20, model="m",
        ))
        stats = summary_store.get_summary_depth_stats(conv.conversation_id)
        assert 0 in stats
        assert stats[0]["leaf"]["count"] == 2
        assert stats[0]["leaf"]["tokens"] == 30

    def test_update_summary_metadata(self, summary_store, sample_conversation):
        conv, _ = sample_conversation
        s = summary_store.create_summary(CreateSummaryInput(
            conversation_id=conv.conversation_id,
            kind="leaf", depth=0, content="x", token_count=1, model="m",
        ))
        summary_store.update_summary_metadata(
            s.summary_id, descendant_count=42, descendant_token_count=1000
        )
        updated = summary_store.get_summary(s.summary_id)
        assert updated.descendant_count == 42
        assert updated.descendant_token_count == 1000
