"""LCM configuration resolution and management.

Handles three-tier precedence:
1. Environment variables (highest - backward compat)
2. Plugin config object (from plugin.yaml) 
3. Hardcoded defaults (lowest)
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class CacheAwareCompactionConfig:
    enabled: bool
    cache_ttl_seconds: int
    max_cold_cache_catchup_passes: int
    hot_cache_pressure_factor: float
    hot_cache_budget_headroom_ratio: float
    cold_cache_observation_threshold: int


@dataclass
class DynamicLeafChunkTokensConfig:
    enabled: bool
    max: int


@dataclass 
class LcmConfig:
    enabled: bool
    database_path: str
    large_files_dir: str
    ignore_session_patterns: List[str]
    stateless_session_patterns: List[str]
    skip_stateless_sessions: bool
    context_threshold: float
    fresh_tail_count: int
    fresh_tail_max_tokens: Optional[int]
    new_session_retain_depth: int
    leaf_min_fanout: int
    condensed_min_fanout: int
    condensed_min_fanout_hard: int
    incremental_max_depth: int
    leaf_chunk_tokens: int
    bootstrap_max_tokens: int
    leaf_target_tokens: int
    condensed_target_tokens: int
    max_expand_tokens: int
    large_file_token_threshold: int
    summary_provider: str
    summary_model: str
    large_file_summary_provider: str
    large_file_summary_model: str
    expansion_provider: str
    expansion_model: str
    delegation_timeout_ms: int
    summary_timeout_ms: int
    timezone: str
    prune_heartbeat_ok: bool
    transcript_gc_enabled: bool
    proactive_threshold_compaction_mode: str
    max_assembly_token_budget: Optional[int]
    summary_max_overage_factor: float
    custom_instructions: str
    circuit_breaker_threshold: int
    circuit_breaker_cooldown_ms: int
    fallback_providers: List[Dict[str, str]]
    cache_aware_compaction: CacheAwareCompactionConfig
    dynamic_leaf_chunk_tokens: DynamicLeafChunkTokensConfig


def resolve_hermes_state_dir(env: Dict[str, str] = None) -> str:
    """Resolve the active Hermes state directory.
    
    Precedence:
    1. HERMES_HOME environment variable 
    2. ~/.hermes (default)
    """
    if env is None:
        env = os.environ
    
    explicit = env.get("HERMES_HOME", "").strip()
    if explicit:
        return explicit
    
    return os.path.expanduser("~/.hermes")


def to_number(value: Any) -> Optional[float]:
    """Safely coerce an unknown value to a finite number, or return None."""
    if isinstance(value, (int, float)) and value == value:  # NaN check
        return float(value)
    if isinstance(value, str):
        try:
            n = float(value)
            return n if n == n else None  # NaN check
        except (ValueError, TypeError):
            pass
    return None


def to_int(value: Any) -> Optional[int]:
    """Safely parse a finite integer, or return None."""
    n = to_number(value)
    return int(n) if n is not None and n == int(n) else None


def to_bool(value: Any) -> Optional[bool]:
    """Safely coerce an unknown value to a boolean, or return None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return None


def to_str(value: Any) -> Optional[str]:
    """Safely coerce an unknown value to a trimmed non-empty string, or return None."""
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else None
    return None


def to_str_array(value: Any) -> Optional[List[str]]:
    """Coerce a plugin config value into a trimmed string array when possible."""
    if isinstance(value, list):
        normalized = [to_str(entry) for entry in value]
        return [entry for entry in normalized if entry is not None]
    
    single = to_str(value)
    if not single:
        return None
    
    return [entry.strip() for entry in single.split(",") if entry.strip()]


def parse_fallback_providers(value: Optional[str]) -> List[Dict[str, str]]:
    """Parse fallback providers from env string (format: 'provider/model,provider/model')."""
    if not value or not value.strip():
        return []
    
    entries = []
    for part in value.split(","):
        trimmed = part.strip()
        if not trimmed:
            continue
            
        slash_idx = trimmed.find("/")
        if slash_idx > 0 and slash_idx < len(trimmed) - 1:
            provider = trimmed[:slash_idx].strip()
            model = trimmed[slash_idx + 1:].strip()
            if provider and model:
                entries.append({"provider": provider, "model": model})
    
    return entries


def to_fallback_provider_array(value: Any) -> List[Dict[str, str]]:
    """Parse fallback providers from plugin config array (object items only)."""
    if not isinstance(value, list):
        return []
    
    entries = []
    for item in value:
        if isinstance(item, dict):
            provider = to_str(item.get("provider"))
            model = to_str(item.get("model"))
            if provider and model:
                entries.append({"provider": provider, "model": model})
    
    return entries


def resolve_lcm_config(
    env: Dict[str, str] = None,
    plugin_config: Dict[str, Any] = None,
) -> LcmConfig:
    """Resolve LCM configuration with three-tier precedence.
    
    Args:
        env: Environment variables (defaults to os.environ)
        plugin_config: Plugin config dict from plugin.yaml
        
    Returns:
        Resolved LCM configuration
    """
    if env is None:
        env = os.environ
    if plugin_config is None:
        plugin_config = {}
    
    pc = plugin_config
    state_dir = resolve_hermes_state_dir(env)
    
    # Handle nested config objects
    cache_aware = pc.get("cache_aware_compaction", {})
    if not isinstance(cache_aware, dict):
        cache_aware = {}
    
    dynamic_leaf = pc.get("dynamic_leaf_chunk_tokens", {})
    if not isinstance(dynamic_leaf, dict):
        dynamic_leaf = {}
    
    # Base calculations
    resolved_leaf_chunk_tokens = (
        to_int(env.get("LCM_LEAF_CHUNK_TOKENS")) or
        to_int(pc.get("leaf_chunk_tokens")) or
        20000
    )
    
    resolved_bootstrap_max_tokens = (
        to_int(env.get("LCM_BOOTSTRAP_MAX_TOKENS")) or
        to_int(pc.get("bootstrap_max_tokens")) or
        max(6000, int(resolved_leaf_chunk_tokens * 0.3))
    )
    
    resolved_dynamic_max = max(
        resolved_leaf_chunk_tokens,
        to_int(env.get("LCM_DYNAMIC_LEAF_CHUNK_TOKENS_MAX")) or
        to_int(dynamic_leaf.get("max")) or
        int(resolved_leaf_chunk_tokens * 2)
    )
    
    resolved_hot_cache_pressure = max(
        1,
        to_number(env.get("LCM_HOT_CACHE_PRESSURE_FACTOR")) or
        to_number(cache_aware.get("hot_cache_pressure_factor")) or
        4
    )
    
    resolved_hot_cache_headroom = min(
        0.95,
        max(
            0,
            to_number(env.get("LCM_HOT_CACHE_BUDGET_HEADROOM_RATIO")) or
            to_number(cache_aware.get("hot_cache_budget_headroom_ratio")) or
            0.2
        )
    )
    
    resolved_cold_cache_threshold = max(
        1,
        int(
            to_number(env.get("LCM_COLD_CACHE_OBSERVATION_THRESHOLD")) or
            to_number(cache_aware.get("cold_cache_observation_threshold")) or
            3
        )
    )
    
    # Parse pattern arrays
    ignore_patterns = (
        to_str_array(env.get("LCM_IGNORE_SESSION_PATTERNS")) or
        to_str_array(pc.get("ignore_session_patterns")) or
        []
    )
    
    stateless_patterns = (
        to_str_array(env.get("LCM_STATELESS_SESSION_PATTERNS")) or
        to_str_array(pc.get("stateless_session_patterns")) or
        []
    )
    
    # Fallback providers
    fallback_providers = (
        parse_fallback_providers(env.get("LCM_FALLBACK_PROVIDERS")) or
        to_fallback_provider_array(pc.get("fallback_providers")) or
        []
    )
    
    # Build config
    return LcmConfig(
        enabled=(
            env.get("LCM_ENABLED", "").lower() != "false" if "LCM_ENABLED" in env else
            to_bool(pc.get("enabled")) or True
        ),
        database_path=(
            env.get("LCM_DATABASE_PATH") or
            to_str(pc.get("db_path")) or
            to_str(pc.get("database_path")) or
            os.path.join(state_dir, "lcm.db")
        ),
        large_files_dir=(
            env.get("LCM_LARGE_FILES_DIR") or
            to_str(pc.get("large_files_dir")) or
            os.path.join(state_dir, "lcm-files")
        ),
        ignore_session_patterns=ignore_patterns,
        stateless_session_patterns=stateless_patterns,
        skip_stateless_sessions=(
            env.get("LCM_SKIP_STATELESS_SESSIONS", "").lower() == "true" if "LCM_SKIP_STATELESS_SESSIONS" in env else
            to_bool(pc.get("skip_stateless_sessions")) or True
        ),
        context_threshold=(
            to_number(env.get("LCM_CONTEXT_THRESHOLD")) or
            to_number(pc.get("context_threshold")) or
            0.75
        ),
        fresh_tail_count=(
            to_int(env.get("LCM_FRESH_TAIL_COUNT")) or
            to_int(pc.get("fresh_tail_count")) or
            64
        ),
        fresh_tail_max_tokens=(
            to_int(env.get("LCM_FRESH_TAIL_MAX_TOKENS")) or
            to_int(pc.get("fresh_tail_max_tokens"))
        ),
        new_session_retain_depth=(
            to_int(env.get("LCM_NEW_SESSION_RETAIN_DEPTH")) or
            to_int(pc.get("new_session_retain_depth")) or
            2
        ),
        leaf_min_fanout=(
            to_int(env.get("LCM_LEAF_MIN_FANOUT")) or
            to_int(pc.get("leaf_min_fanout")) or
            8
        ),
        condensed_min_fanout=(
            to_int(env.get("LCM_CONDENSED_MIN_FANOUT")) or
            to_int(pc.get("condensed_min_fanout")) or
            4
        ),
        condensed_min_fanout_hard=(
            to_int(env.get("LCM_CONDENSED_MIN_FANOUT_HARD")) or
            to_int(pc.get("condensed_min_fanout_hard")) or
            2
        ),
        incremental_max_depth=(
            to_int(env.get("LCM_INCREMENTAL_MAX_DEPTH")) or
            to_int(pc.get("incremental_max_depth")) or
            1
        ),
        leaf_chunk_tokens=resolved_leaf_chunk_tokens,
        bootstrap_max_tokens=resolved_bootstrap_max_tokens,
        leaf_target_tokens=(
            to_int(env.get("LCM_LEAF_TARGET_TOKENS")) or
            to_int(pc.get("leaf_target_tokens")) or
            2400
        ),
        condensed_target_tokens=(
            to_int(env.get("LCM_CONDENSED_TARGET_TOKENS")) or
            to_int(pc.get("condensed_target_tokens")) or
            2000
        ),
        max_expand_tokens=(
            to_int(env.get("LCM_MAX_EXPAND_TOKENS")) or
            to_int(pc.get("max_expand_tokens")) or
            4000
        ),
        large_file_token_threshold=(
            to_int(env.get("LCM_LARGE_FILE_TOKEN_THRESHOLD")) or
            to_int(pc.get("large_file_threshold_tokens")) or
            to_int(pc.get("large_file_token_threshold")) or
            25000
        ),
        summary_provider=(
            to_str(env.get("LCM_SUMMARY_PROVIDER")) or
            to_str(pc.get("summary_provider")) or
            "openai"  # Default: use OpenAI-compatible API (e.g. litellm proxy)
        ),
        summary_model=(
            to_str(env.get("LCM_SUMMARY_MODEL")) or
            to_str(pc.get("summary_model")) or
            "gemini-2.5-flash"  # Default: Gemini 2.5 Flash for cost-effective summarization
        ),
        large_file_summary_provider=(
            to_str(env.get("LCM_LARGE_FILE_SUMMARY_PROVIDER")) or
            to_str(pc.get("large_file_summary_provider")) or
            ""
        ),
        large_file_summary_model=(
            to_str(env.get("LCM_LARGE_FILE_SUMMARY_MODEL")) or
            to_str(pc.get("large_file_summary_model")) or
            ""
        ),
        expansion_provider=(
            to_str(env.get("LCM_EXPANSION_PROVIDER")) or
            to_str(pc.get("expansion_provider")) or
            ""
        ),
        expansion_model=(
            to_str(env.get("LCM_EXPANSION_MODEL")) or
            to_str(pc.get("expansion_model")) or
            ""
        ),
        delegation_timeout_ms=(
            to_int(env.get("LCM_DELEGATION_TIMEOUT_MS")) or
            to_int(pc.get("delegation_timeout_ms")) or
            120000
        ),
        summary_timeout_ms=(
            to_int(env.get("LCM_SUMMARY_TIMEOUT_MS")) or
            to_int(pc.get("summary_timeout_ms")) or
            60000
        ),
        timezone=(
            env.get("TZ") or
            to_str(pc.get("timezone")) or
            "UTC"
        ),
        prune_heartbeat_ok=(
            env.get("LCM_PRUNE_HEARTBEAT_OK", "").lower() == "true" if "LCM_PRUNE_HEARTBEAT_OK" in env else
            to_bool(pc.get("prune_heartbeat_ok")) or False
        ),
        transcript_gc_enabled=(
            env.get("LCM_TRANSCRIPT_GC_ENABLED", "").lower() == "true" if "LCM_TRANSCRIPT_GC_ENABLED" in env else
            to_bool(pc.get("transcript_gc_enabled")) or False
        ),
        proactive_threshold_compaction_mode=(
            to_str(env.get("LCM_PROACTIVE_THRESHOLD_COMPACTION_MODE")) or
            to_str(pc.get("proactive_threshold_compaction_mode")) or
            "deferred"
        ),
        max_assembly_token_budget=(
            to_int(env.get("LCM_MAX_ASSEMBLY_TOKEN_BUDGET")) or
            to_int(pc.get("max_assembly_token_budget"))
        ),
        summary_max_overage_factor=(
            to_number(env.get("LCM_SUMMARY_MAX_OVERAGE_FACTOR")) or
            to_number(pc.get("summary_max_overage_factor")) or
            3
        ),
        custom_instructions=(
            to_str(env.get("LCM_CUSTOM_INSTRUCTIONS")) or
            to_str(pc.get("custom_instructions")) or
            ""
        ),
        circuit_breaker_threshold=(
            to_int(env.get("LCM_CIRCUIT_BREAKER_THRESHOLD")) or
            to_int(pc.get("circuit_breaker_threshold")) or
            5
        ),
        circuit_breaker_cooldown_ms=(
            to_int(env.get("LCM_CIRCUIT_BREAKER_COOLDOWN_MS")) or
            to_int(pc.get("circuit_breaker_cooldown_ms")) or
            1800000
        ),
        fallback_providers=fallback_providers,
        cache_aware_compaction=CacheAwareCompactionConfig(
            enabled=(
                env.get("LCM_CACHE_AWARE_COMPACTION_ENABLED", "").lower() != "false" if "LCM_CACHE_AWARE_COMPACTION_ENABLED" in env else
                to_bool(cache_aware.get("enabled")) or True
            ),
            cache_ttl_seconds=(
                to_int(env.get("LCM_CACHE_TTL_SECONDS")) or
                to_int(cache_aware.get("cache_ttl_seconds")) or
                300
            ),
            max_cold_cache_catchup_passes=(
                to_int(env.get("LCM_MAX_COLD_CACHE_CATCHUP_PASSES")) or
                to_int(cache_aware.get("max_cold_cache_catchup_passes")) or
                2
            ),
            hot_cache_pressure_factor=resolved_hot_cache_pressure,
            hot_cache_budget_headroom_ratio=resolved_hot_cache_headroom,
            cold_cache_observation_threshold=resolved_cold_cache_threshold,
        ),
        dynamic_leaf_chunk_tokens=DynamicLeafChunkTokensConfig(
            enabled=(
                env.get("LCM_DYNAMIC_LEAF_CHUNK_TOKENS_ENABLED", "").lower() == "true" if "LCM_DYNAMIC_LEAF_CHUNK_TOKENS_ENABLED" in env else
                to_bool(dynamic_leaf.get("enabled")) or True
            ),
            max=resolved_dynamic_max,
        ),
    )