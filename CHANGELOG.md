# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-15

### Added
- Initial Python port of lossless-claw v0.9.1 (Martian Engineering)
- DAG-based lossless context management for Claude Code (NousResearch fork)
- SQLite storage with FTS5 full-text search and WAL mode
- Leaf and condensed compaction passes with depth-aware summarization
- Cache-aware compaction with hot/cold/warm cache state tracking
- Circuit breaker pattern for LLM provider resilience
- Context assembly with fresh-tail protection
- Agent tools: `lcm_grep`, `lcm_describe`, `lcm_expand`
- Unicode-aware token estimation (CJK, emoji support)
- Three-tier configuration (env vars > plugin.yaml > defaults)
- Dynamic leaf chunk sizing
- Message identity hashing for deduplication
- Comprehensive test suite
- GitHub Actions CI/CD pipeline
- PyPI packaging support
