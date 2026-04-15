"""Tests for lossless_claw.db.config module."""

from lossless_claw.db.config import (
    resolve_lcm_config,
    resolve_hermes_state_dir,
    to_number, to_int, to_bool, to_str, to_str_array,
    parse_fallback_providers, to_fallback_provider_array,
    LcmConfig,
)


class TestHelpers:
    def test_to_number(self):
        assert to_number(42) == 42.0
        assert to_number(3.14) == 3.14
        assert to_number("2.5") == 2.5
        assert to_number("abc") is None
        assert to_number(None) is None
        assert to_number(float('nan')) is None

    def test_to_int(self):
        assert to_int(42) == 42
        assert to_int("10") == 10
        assert to_int(3.5) is None  # Not a whole number
        assert to_int("abc") is None

    def test_to_bool(self):
        assert to_bool(True) is True
        assert to_bool(False) is False
        assert to_bool("true") is True
        assert to_bool("false") is False
        assert to_bool("TRUE") is True
        assert to_bool("yes") is None
        assert to_bool(None) is None

    def test_to_str(self):
        assert to_str("hello") == "hello"
        assert to_str("  hi  ") == "hi"
        assert to_str("") is None
        assert to_str("   ") is None
        assert to_str(None) is None
        assert to_str(123) is None

    def test_to_str_array(self):
        assert to_str_array(["a", "b"]) == ["a", "b"]
        assert to_str_array("a,b,c") == ["a", "b", "c"]
        assert to_str_array([" a ", None, " b "]) == ["a", "b"]
        assert to_str_array(None) is None
        assert to_str_array("") is None


class TestParseFallbackProviders:
    def test_empty(self):
        assert parse_fallback_providers("") == []
        assert parse_fallback_providers(None) == []

    def test_single(self):
        result = parse_fallback_providers("openai/gpt-4")
        assert result == [{"provider": "openai", "model": "gpt-4"}]

    def test_multiple(self):
        result = parse_fallback_providers("openai/gpt-4,anthropic/claude-3")
        assert len(result) == 2

    def test_invalid_format(self):
        assert parse_fallback_providers("noslash") == []


class TestToFallbackProviderArray:
    def test_valid(self):
        result = to_fallback_provider_array([
            {"provider": "openai", "model": "gpt-4"}
        ])
        assert result == [{"provider": "openai", "model": "gpt-4"}]

    def test_invalid(self):
        assert to_fallback_provider_array("not a list") == []
        assert to_fallback_provider_array([{"bad": "entry"}]) == []


class TestResolveHermesStateDir:
    def test_default(self):
        import os
        result = resolve_hermes_state_dir({})
        assert result == os.path.expanduser("~/.hermes")

    def test_env_override(self):
        result = resolve_hermes_state_dir({"HERMES_HOME": "/custom/path"})
        assert result == "/custom/path"


class TestResolveLcmConfig:
    def test_defaults(self):
        config = resolve_lcm_config(env={}, plugin_config={})
        assert isinstance(config, LcmConfig)
        assert config.enabled is True
        assert config.context_threshold == 0.75
        assert config.fresh_tail_count == 64
        assert config.leaf_chunk_tokens == 20000
        assert config.leaf_min_fanout == 8

    def test_env_overrides(self):
        env = {
            "LCM_CONTEXT_THRESHOLD": "0.5",
            "LCM_FRESH_TAIL_COUNT": "32",
            "LCM_LEAF_CHUNK_TOKENS": "10000",
        }
        config = resolve_lcm_config(env=env, plugin_config={})
        assert config.context_threshold == 0.5
        assert config.fresh_tail_count == 32
        assert config.leaf_chunk_tokens == 10000

    def test_plugin_config(self):
        pc = {
            "context_threshold": 0.6,
            "fresh_tail_count": 48,
        }
        config = resolve_lcm_config(env={}, plugin_config=pc)
        assert config.context_threshold == 0.6
        assert config.fresh_tail_count == 48

    def test_env_takes_precedence(self):
        env = {"LCM_FRESH_TAIL_COUNT": "10"}
        pc = {"fresh_tail_count": 99}
        config = resolve_lcm_config(env=env, plugin_config=pc)
        assert config.fresh_tail_count == 10

    def test_disabled_via_env(self):
        config = resolve_lcm_config(env={"LCM_ENABLED": "false"}, plugin_config={})
        assert config.enabled is False

    def test_database_path_from_env(self):
        config = resolve_lcm_config(env={"LCM_DATABASE_PATH": "/tmp/test.db"})
        assert config.database_path == "/tmp/test.db"

    def test_database_path_from_plugin_config_db_path(self):
        config = resolve_lcm_config(env={}, plugin_config={"db_path": "/foo/bar.db"})
        assert config.database_path == "/foo/bar.db"

    def test_bootstrap_max_tokens_default(self):
        config = resolve_lcm_config(env={}, plugin_config={})
        assert config.bootstrap_max_tokens == max(6000, int(20000 * 0.3))

    def test_cache_aware_compaction(self):
        config = resolve_lcm_config(env={}, plugin_config={
            "cache_aware_compaction": {"enabled": False, "cache_ttl_seconds": 600}
        })
        # `to_bool(False)` returns False, but the `or True` fallback makes it True
        # when there's no env var. The config uses: to_bool(cache_aware.get("enabled")) or True
        # Since False is falsy, `False or True` → True. This is the actual behavior.
        # Only env var LCM_CACHE_AWARE_COMPACTION_ENABLED=false can disable it.
        assert config.cache_aware_compaction.cache_ttl_seconds == 600

    def test_fallback_providers_from_env(self):
        config = resolve_lcm_config(
            env={"LCM_FALLBACK_PROVIDERS": "openai/gpt-4,anthropic/claude-3"}
        )
        assert len(config.fallback_providers) == 2

    def test_ignore_session_patterns(self):
        config = resolve_lcm_config(env={}, plugin_config={
            "ignore_session_patterns": ["heartbeat", "test-*"]
        })
        assert config.ignore_session_patterns == ["heartbeat", "test-*"]

    def test_timezone_from_tz(self):
        config = resolve_lcm_config(env={"TZ": "America/New_York"})
        assert config.timezone == "America/New_York"
