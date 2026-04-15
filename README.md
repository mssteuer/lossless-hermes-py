# lossless-hermes-py

[![PyPI version](https://img.shields.io/pypi/v/lossless-hermes-py.svg)](https://pypi.org/project/lossless-hermes-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**DAG-based lossless context management for LLM conversations. Never lose a message — summarize them into a directed acyclic graph.**

A Python port of [lossless-claw](https://github.com/Martian-Engineering/lossless-claw) for use with [Hermes Agent](https://hermes.nousresearch.com) or as a standalone library.

---

## What It Does

Long conversations with LLMs hit context window limits. The typical solution is to truncate or naively summarize, losing information permanently. Lossless Context Management (LCM) takes a different approach: it builds a **directed acyclic graph of summaries** that compresses older messages while preserving the ability to drill back into any detail.

Your conversation never loses information. It just gets more compact.

See the animated visualization at [losslesscontext.ai](https://losslesscontext.ai).

## How It Works

LCM operates in two compaction passes that build a summary DAG:

**Leaf pass** — When the context window fills up, raw messages (excluding a protected "fresh tail" of recent messages) are chunked and summarized into **leaf nodes** at depth 0. Each leaf summary covers a group of messages and links back to the originals.

**Condensed pass** — When enough leaf summaries accumulate, they are themselves summarized into **condensed nodes** at depth 1. This process repeats upward: depth-1 summaries get condensed into depth-2, and so on. The result is a tree-like DAG where the root captures the entire conversation at high compression, and any branch can be expanded to recover full detail.

**Context assembly** reconstructs the optimal prompt by combining:
- The highest-level summaries (covering the full history compactly)
- The fresh tail (recent messages kept verbatim for continuity)

**Cache-aware compaction** adapts to prompt caching behavior. When the cache is hot (high hit rate), compaction backs off to avoid invalidating cached prefixes. When the cache is cold, compaction runs more aggressively.

**Agent tools** (`lcm_grep`, `lcm_expand`, `lcm_describe`) let the LLM search and drill into compacted history on demand, recovering detail without keeping everything in context.

## Installation

### Hermes Agent Plugin (recommended)

```bash
hermes plugins install mssteuer/lossless-hermes-py
```

Then enable the engine in `~/.hermes/config.yaml` (top level, not under `agent:`):

```yaml
context:
  engine: lcm
```

**(Optional)** Configure a dedicated summarization model via environment variables in `~/.hermes/.env`:

```bash
LCM_SUMMARY_MODEL=gpt-4o-mini
LCM_SUMMARY_PROVIDER=openai
```

Or edit the plugin's `plugin.yaml` directly:

```bash
nano ~/.hermes/plugins/lossless-hermes/plugin.yaml
```

Restart the gateway:

```bash
hermes gateway restart
```

The agent will now have `lcm_grep`, `lcm_describe`, and `lcm_expand` tools available.

### Via pip

```bash
pip install lossless-hermes-py
```

This is useful for standalone library usage (see below) or if you prefer managing Python packages separately. For use with Hermes, the `hermes plugins install` method above is simpler.

### Standalone Usage

LCM also works as a standalone library without Hermes:

```python
from lossless_hermes import LcmContextEngine

engine = LcmContextEngine(
    model="gpt-4o-mini",
    provider="openai",
    config_context_length=128000,
)

# Start a session
engine.on_session_start("my-session")

# Check if compaction is needed
if engine.should_compress(prompt_tokens=100000):
    compressed = engine.compress(messages, current_tokens=100000)
```

## Configuration

Settings are resolved with three-tier precedence: **environment variables > plugin.yaml > defaults**.

### Key Settings

| Setting | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable/disable LCM |
| `context_threshold` | `0.75` | Fraction of context window that triggers compaction |
| `fresh_tail_count` | `64` | Number of recent messages kept verbatim |
| `fresh_tail_max_tokens` | `null` | Optional token budget cap for fresh tail |
| `leaf_chunk_tokens` | `20000` | Max tokens per leaf chunk |
| `leaf_target_tokens` | `2400` | Target summary size for leaf nodes |
| `condensed_target_tokens` | `2000` | Target summary size for condensed nodes |
| `leaf_min_fanout` | `8` | Minimum messages per leaf chunk |
| `condensed_min_fanout` | `4` | Minimum summaries per condensed chunk |
| `condensed_min_fanout_hard` | `2` | Hard minimum for condensed chunks |
| `incremental_max_depth` | `1` | Max depth levels to compact per pass |
| `summary_provider` | `""` | LLM provider for summarization (falls back to host agent's provider) |
| `summary_model` | `""` | Model for summarization (falls back to host agent's model) |
| `summary_timeout_ms` | `60000` | Timeout for summarization calls |
| `circuit_breaker_threshold` | `5` | Consecutive failures before circuit opens |
| `circuit_breaker_cooldown_ms` | `1800000` | Cooldown before retrying after circuit break (30 min) |

### Cache-Aware Compaction

```yaml
config:
  cache_aware_compaction:
    enabled: true
    cache_ttl_seconds: 300
    max_cold_cache_catchup_passes: 2
    hot_cache_pressure_factor: 4
    hot_cache_budget_headroom_ratio: 0.2
    cold_cache_observation_threshold: 3
```

### Dynamic Leaf Chunk Sizing

```yaml
config:
  dynamic_leaf_chunk_tokens:
    enabled: true
    max: 40000
```

### Environment Variables

All config keys can be set via environment variables with the `LCM_` prefix:

```bash
export LCM_SUMMARY_MODEL="gemini-2.5-flash"
export LCM_FRESH_TAIL_COUNT=32
export LCM_CONTEXT_THRESHOLD=0.8
```

## Tools

LCM exposes three tools that the agent can call to interact with compacted history:

### `lcm_grep`

Search conversation history and summaries using FTS5 full-text search or regex.

```json
{
  "query": "database migration strategy",
  "mode": "full_text",
  "include_messages": true,
  "include_summaries": true,
  "limit": 20
}
```

### `lcm_describe`

Get the current LCM state: summary statistics, DAG depth, message counts, recent compaction activity.

```json
{
  "include_stats": true,
  "include_recent": true
}
```

### `lcm_expand`

Drill into specific content. Expand a message to see its full text, a summary to see its children and linked messages, or search for related content across conversations.

```json
{
  "target_type": "summary",
  "target_id": "abc123"
}
```

```json
{
  "target_type": "related",
  "query": "authentication flow",
  "limit": 10
}
```

## Architecture

```
src/lossless_hermes/
    __init__.py          # LcmContextEngine — main plugin class, Hermes integration
    compaction.py        # CompactionEngine — leaf/condensed passes, cache-aware policy
    assembler.py         # ContextAssembler — reconstructs optimal context from DAG
    summarizer.py        # LLM summarization with circuit breaker pattern
    retrieval.py         # RetrievalEngine — FTS5 search, related content discovery
    tokens.py            # Unicode-aware token estimation (CJK, emoji)
    tools.py             # lcm_grep, lcm_describe, lcm_expand tool implementations
    db/
        config.py        # Three-tier config resolution (env > yaml > defaults)
        connection.py    # SQLite connection management (WAL mode)
        migration.py     # Schema migrations
    store/
        conversation.py  # ConversationStore — messages, parts, sequences
        summary.py       # SummaryStore — DAG nodes, edges, depth stats
        identity.py      # Content identity hashing for deduplication
```

### Storage

SQLite with WAL mode for concurrent reads. FTS5 virtual tables for full-text search across messages and summaries. All data is local — no external services beyond the LLM provider.

## Differences from the TypeScript Version

This is a Python port of the original [lossless-claw](https://github.com/Martian-Engineering/lossless-claw) TypeScript implementation. Key adaptations:

- **Target platform**: [Hermes Agent](https://hermes.nousresearch.com) (NousResearch) Python plugin system instead of OpenClaw
- **Plugin interface**: Implements `ContextEngine` base class from `agent.context_engine` with `register()` entry point
- **LLM calls**: Provider-agnostic via configurable summarizer (supports litellm, direct API calls) rather than being tied to a specific SDK
- **Default summarization model**: Gemini 2.5 Flash (configurable to any model)
- **Async model**: Synchronous by default (matching the Hermes plugin interface) rather than async-first
- **Config resolution**: Three-tier env/yaml/defaults pattern adapted for Python conventions
- **Token estimation**: Custom Unicode-aware estimator (`tokens.py`) with CJK and emoji handling
- **Database**: Same SQLite/FTS5 approach, using Python's built-in `sqlite3` module
- **Standalone support**: Works both as a Hermes plugin and as a standalone library — the `ContextEngine` import is optional

The core algorithm — DAG-based compaction with leaf and condensed passes, fresh-tail protection, cache-aware compaction policies — is a faithful port of the original.

## Credits

- **[lossless-claw](https://github.com/Martian-Engineering/lossless-claw)** by Josh Lehman / [Martian Engineering](https://github.com/Martian-Engineering) (MIT License) — the original TypeScript implementation this project is ported from
- **[The LCM Paper](https://papers.voltropy.com/LCM)** by Voltropy — the academic foundation for lossless context management
- **[Hermes Agent](https://github.com/NousResearch/hermes)** by NousResearch — the target platform whose context engine plugin system this integrates with

## License

MIT
