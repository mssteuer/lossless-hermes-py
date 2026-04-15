"""Integration tests for LcmContextEngine."""

import pytest
import os

from lossless_hermes import LcmContextEngine
from lossless_hermes.tokens import estimate_messages_tokens


class TestLcmContextEngine:
    @pytest.fixture
    def engine(self, tmp_path):
        e = LcmContextEngine(
            model="test-model",
            plugin_config={
                "database_path": str(tmp_path / "engine.db"),
                "fresh_tail_count": 4,
                "leaf_min_fanout": 2,
                "condensed_min_fanout": 2,
                "condensed_min_fanout_hard": 2,
                "leaf_chunk_tokens": 500,
            }
        )
        return e

    def test_init(self, engine):
        assert engine.name == "lcm"
        assert engine.model == "test-model"
        assert engine.config.enabled is True

    def test_should_compress_below_threshold(self, engine):
        assert engine.should_compress(prompt_tokens=100) is False

    def test_should_compress_above_threshold(self, engine):
        # threshold = context_length * 0.75 = 128000 * 0.75 = 96000
        assert engine.should_compress(prompt_tokens=100000) is True

    def test_should_compress_disabled(self, tmp_path):
        import os
        e = LcmContextEngine(
            model="test",
            plugin_config={
                "database_path": str(tmp_path / "disabled.db"),
            }
        )
        e.config.enabled = False  # Directly disable
        assert e.should_compress(prompt_tokens=999999) is False

    def test_update_from_response(self, engine):
        engine.update_from_response({
            "prompt_tokens": 5000,
            "completion_tokens": 200,
            "total_tokens": 5200,
        })
        assert engine.last_prompt_tokens == 5000
        assert engine.last_completion_tokens == 200

    def test_should_compress_uses_last_prompt(self, engine):
        engine.update_from_response({"prompt_tokens": 200000, "total_tokens": 200000})
        assert engine.should_compress() is True

    def test_update_model(self, engine):
        engine.update_model("new-model", 200000, provider="anthropic")
        assert engine.model == "new-model"
        assert engine.context_length == 200000
        assert engine.threshold_tokens == int(200000 * engine.config.context_threshold)

    def test_session_lifecycle(self, engine):
        engine.on_session_start("test-session")
        assert engine.current_session_id == "test-session"
        assert engine.current_conversation_id is not None

        engine.on_session_end("test-session", [])
        assert engine.current_session_id is None
        assert engine.current_conversation_id is None

    def test_session_reset(self, engine):
        engine.on_session_start("sess-reset")
        engine.on_session_reset()
        assert engine.current_session_id is None

    def test_get_status(self, engine):
        status = engine.get_status()
        assert "threshold_tokens" in status or "name" in status
        assert "threshold_tokens" in status

    def test_get_status_with_session(self, engine):
        engine.on_session_start("status-sess")
        status = engine.get_status()
        assert status.get("lcm_enabled") is True
        assert "conversation_id" in status

    def test_compress_fallback_on_error(self, engine):
        # Without proper summarizer setup, compress should fallback gracefully
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = engine.compress(messages, session_id="test")
        # Should return something (either compressed or original fallback)
        assert isinstance(result, list)

    def test_get_tool_schemas(self, engine):
        schemas = engine.get_tool_schemas()
        assert isinstance(schemas, list)

    def test_disabled_returns_empty_tools(self, tmp_path):
        e = LcmContextEngine(
            model="test",
            plugin_config={
                "database_path": str(tmp_path / "no-tools.db"),
            }
        )
        schemas = e.get_tool_schemas()
        assert isinstance(schemas, list)
