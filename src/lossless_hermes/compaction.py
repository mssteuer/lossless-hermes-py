"""Compaction engine for LCM.

Handles the DAG-based compaction algorithm with leaf and condensed passes,
cache-aware compaction policies, and fresh-tail protection.
"""

import logging
from dataclasses import dataclass

from .db.config import CacheAwareCompactionConfig
from .store.conversation import ConversationStore, MessageRecord
from .store.summary import CreateSummaryInput, SummaryRecord, SummaryStore
from .summarizer import SummaryOptions, SyncLcmSummarizer
from .tokens import estimate_tokens

logger = logging.getLogger(__name__)


@dataclass
class CompactionConfig:
    """Configuration for compaction behavior."""

    leaf_chunk_tokens: int
    leaf_target_tokens: int
    condensed_target_tokens: int
    leaf_min_fanout: int
    condensed_min_fanout: int
    condensed_min_fanout_hard: int
    incremental_max_depth: int
    fresh_tail_count: int
    fresh_tail_max_tokens: int | None
    cache_aware_compaction: CacheAwareCompactionConfig


@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    summaries_created: int
    messages_compacted: int
    tokens_saved: int
    depth_reached: int
    cache_state: str = "unknown"


class CompactionEngine:
    """LCM DAG-based compaction engine."""

    def __init__(
        self,
        config: CompactionConfig,
        summarizer: SyncLcmSummarizer,
        conversation_store: ConversationStore,
        summary_store: SummaryStore,
    ):
        self.config = config
        self.summarizer = summarizer
        self.conversation_store = conversation_store
        self.summary_store = summary_store

    def should_compact(
        self,
        conversation_id: int,
        current_tokens: int,
        threshold_tokens: int,
        cache_state: str = "unknown",
    ) -> tuple[bool, str]:
        """Determine if compaction should run based on current state."""

        # Basic threshold check
        if current_tokens < threshold_tokens:
            return False, "below_threshold"

        # Get existing summaries to understand current state
        summaries = self.summary_store.get_summaries_by_conversation(conversation_id)
        leaf_summaries = [s for s in summaries if s.kind == "leaf"]

        # If no summaries exist, definitely compact
        if not leaf_summaries:
            return True, "initial_compaction"

        # Cache-aware compaction logic
        if self.config.cache_aware_compaction.enabled:
            return self._should_compact_cache_aware(conversation_id, current_tokens, threshold_tokens, cache_state)

        return True, "threshold_exceeded"

    def _should_compact_cache_aware(
        self, conversation_id: int, current_tokens: int, threshold_tokens: int, cache_state: str
    ) -> tuple[bool, str]:
        """Cache-aware compaction decision logic."""

        if cache_state == "hot":
            # Hot cache: be more conservative, allow higher usage
            pressure_factor = self.config.cache_aware_compaction.hot_cache_pressure_factor
            adjusted_threshold = int(threshold_tokens * pressure_factor)

            if current_tokens < adjusted_threshold:
                return False, "hot_cache_pressure_relief"

        elif cache_state == "cold":
            # Cold cache: be more aggressive to reduce cache misses
            # Allow multiple passes to catch up
            return True, "cold_cache_catchup"

        return True, "threshold_exceeded"

    def compact(self, conversation_id: int, current_tokens: int, cache_state: str = "unknown") -> CompactionResult:
        """Run compaction on a conversation."""
        logger.info(f"Starting compaction for conversation {conversation_id}")

        initial_messages = self.conversation_store.get_messages_by_conversation(conversation_id)
        initial_tokens = sum(msg.token_count for msg in initial_messages)

        summaries_created = 0
        depth_reached = 0

        # Phase 1: Leaf compaction
        leaf_result = self._run_leaf_compaction(conversation_id, cache_state)
        summaries_created += leaf_result.summaries_created
        depth_reached = max(depth_reached, 0)  # Leaf is depth 0

        # Phase 2: Condensed compaction (iterative up the tree)
        condensed_result = self._run_condensed_compaction(conversation_id, cache_state)
        summaries_created += condensed_result.summaries_created
        depth_reached = max(depth_reached, condensed_result.depth_reached)

        # Calculate final stats
        final_messages = self.conversation_store.get_messages_by_conversation(conversation_id)
        final_tokens = sum(msg.token_count for msg in final_messages)

        # Add summary tokens
        all_summaries = self.summary_store.get_summaries_by_conversation(conversation_id)
        summary_tokens = sum(s.token_count for s in all_summaries)
        final_tokens += summary_tokens

        tokens_saved = initial_tokens - final_tokens
        messages_compacted = len(initial_messages) - len(final_messages)

        logger.info(
            f"Compaction complete: {summaries_created} summaries created, "
            f"{messages_compacted} messages compacted, {tokens_saved} tokens saved"
        )

        return CompactionResult(
            summaries_created=summaries_created,
            messages_compacted=messages_compacted,
            tokens_saved=tokens_saved,
            depth_reached=depth_reached,
            cache_state=cache_state,
        )

    def _run_leaf_compaction(self, conversation_id: int, cache_state: str) -> CompactionResult:
        """Run leaf-level (depth 0) compaction on raw messages."""

        # Get messages that haven't been summarized yet
        messages = self.conversation_store.get_messages_by_conversation(conversation_id)

        # Protect fresh tail
        protected_count = self._calculate_fresh_tail_protection(messages)

        if protected_count >= len(messages) - 1:
            logger.info("All messages are in fresh tail - no leaf compaction needed")
            return CompactionResult(0, 0, 0, 0, cache_state)

        # Get messages available for compaction
        compactable_messages = messages[:-protected_count] if protected_count > 0 else messages

        # Group messages into chunks
        chunks = self._chunk_messages_for_leaf_compaction(compactable_messages)

        summaries_created = 0
        for chunk in chunks:
            if len(chunk) < self.config.leaf_min_fanout:
                continue  # Skip small chunks

            summary = self._create_leaf_summary(conversation_id, chunk)
            if summary:
                summaries_created += 1

                # Link messages to summary
                message_ids = [msg.message_id for msg in chunk]
                self.summary_store.add_summary_messages(summary.summary_id, message_ids)

        return CompactionResult(summaries_created, 0, 0, 0, cache_state)

    def _run_condensed_compaction(self, conversation_id: int, cache_state: str) -> CompactionResult:
        """Run condensed compaction (building higher levels of the DAG)."""
        summaries_created = 0
        max_depth = 0

        # Iteratively compact each depth level
        for depth in range(self.config.incremental_max_depth + 1):
            depth_summaries = self.summary_store.get_summaries_by_conversation(conversation_id, depth=depth)

            if len(depth_summaries) < self.config.condensed_min_fanout:
                break  # Not enough summaries to compact

            # Group summaries for compaction
            chunks = self._chunk_summaries_for_condensed_compaction(depth_summaries)

            for chunk in chunks:
                if len(chunk) < self.config.condensed_min_fanout_hard:
                    continue

                summary = self._create_condensed_summary(conversation_id, chunk, depth + 1)
                if summary:
                    summaries_created += 1
                    max_depth = max(max_depth, depth + 1)

                    # Link parent summaries
                    for parent_summary in chunk:
                        self.summary_store.add_summary_parent(summary.summary_id, parent_summary.summary_id)

        return CompactionResult(summaries_created, 0, 0, max_depth, cache_state)

    def _calculate_fresh_tail_protection(self, messages: list[MessageRecord]) -> int:
        """Calculate how many messages to protect as fresh tail."""
        if not messages:
            return 0

        # Start with configured count
        protected_count = self.config.fresh_tail_count

        # Apply token budget if configured
        if self.config.fresh_tail_max_tokens:
            token_budget = self.config.fresh_tail_max_tokens
            accumulated_tokens = 0

            # Walk backwards from the end
            for i in range(len(messages) - 1, -1, -1):
                if accumulated_tokens + messages[i].token_count > token_budget:
                    break
                accumulated_tokens += messages[i].token_count

            # Use the more restrictive constraint
            token_protected = len(messages) - i - 1
            protected_count = min(protected_count, token_protected)

        return min(protected_count, len(messages))

    def _chunk_messages_for_leaf_compaction(self, messages: list[MessageRecord]) -> list[list[MessageRecord]]:
        """Chunk messages into groups for leaf summarization."""
        if not messages:
            return []

        chunks = []
        current_chunk = []
        current_tokens = 0

        for message in messages:
            # Check if adding this message would exceed chunk size
            if (
                current_tokens + message.token_count > self.config.leaf_chunk_tokens
                and len(current_chunk) >= self.config.leaf_min_fanout
            ):
                chunks.append(current_chunk)
                current_chunk = [message]
                current_tokens = message.token_count
            else:
                current_chunk.append(message)
                current_tokens += message.token_count

        # Add final chunk if it's large enough
        if len(current_chunk) >= self.config.leaf_min_fanout:
            chunks.append(current_chunk)

        return chunks

    def _chunk_summaries_for_condensed_compaction(self, summaries: list[SummaryRecord]) -> list[list[SummaryRecord]]:
        """Chunk summaries into groups for condensed summarization."""
        if not summaries:
            return []

        # Simple strategy: group by creation time / token count
        chunks = []
        current_chunk = []
        current_tokens = 0
        target_tokens = self.config.leaf_chunk_tokens  # Reuse same chunk size

        for summary in summaries:
            if (
                current_tokens + summary.token_count > target_tokens
                and len(current_chunk) >= self.config.condensed_min_fanout
            ):
                chunks.append(current_chunk)
                current_chunk = [summary]
                current_tokens = summary.token_count
            else:
                current_chunk.append(summary)
                current_tokens += summary.token_count

        # Add final chunk if it's large enough
        if len(current_chunk) >= self.config.condensed_min_fanout_hard:
            chunks.append(current_chunk)

        return chunks

    def _create_leaf_summary(self, conversation_id: int, messages: list[MessageRecord]) -> SummaryRecord | None:
        """Create a leaf summary from a chunk of messages."""
        if not messages:
            return None

        # Build content for summarization
        content_parts = []
        for msg in messages:
            role_label = msg.role.upper()
            content_parts.append(f"[{role_label}]: {msg.content}")

        text_to_summarize = "\n\n".join(content_parts)

        try:
            # Generate summary
            summary_content = self.summarizer.summarize(
                text_to_summarize,
                aggressive=False,
                options=SummaryOptions(is_condensed=False, depth=0),
            )

            # Calculate metadata
            token_count = estimate_tokens(summary_content)
            earliest_at = min(msg.created_at for msg in messages)
            latest_at = max(msg.created_at for msg in messages)
            source_token_count = sum(msg.token_count for msg in messages)

            # Create summary record
            return self.summary_store.create_summary(
                CreateSummaryInput(
                    conversation_id=conversation_id,
                    kind="leaf",
                    depth=0,
                    content=summary_content,
                    token_count=token_count,
                    earliest_at=earliest_at,
                    latest_at=latest_at,
                    descendant_count=len(messages),
                    descendant_token_count=source_token_count,
                    source_message_token_count=source_token_count,
                    model=getattr(self.summarizer, "model", "unknown"),
                )
            )

        except Exception as e:
            logger.error(f"Failed to create leaf summary: {e}")
            return None

    def _create_condensed_summary(
        self, conversation_id: int, summaries: list[SummaryRecord], depth: int
    ) -> SummaryRecord | None:
        """Create a condensed summary from a chunk of summaries."""
        if not summaries:
            return None

        # Build content for summarization
        content_parts = []
        for summary in summaries:
            content_parts.append(f"=== {summary.kind.title()} Summary (Depth {summary.depth}) ===")
            content_parts.append(summary.content)
            content_parts.append("")  # Blank line

        text_to_summarize = "\n".join(content_parts)

        try:
            # Generate summary
            summary_content = self.summarizer.summarize(
                text_to_summarize,
                aggressive=False,
                options=SummaryOptions(is_condensed=True, depth=depth),
            )

            # Calculate metadata
            token_count = estimate_tokens(summary_content)
            earliest_at = min(s.earliest_at for s in summaries if s.earliest_at)
            latest_at = max(s.latest_at for s in summaries if s.latest_at)

            descendant_count = sum(s.descendant_count for s in summaries)
            descendant_token_count = sum(s.descendant_token_count for s in summaries)
            source_token_count = sum(s.token_count for s in summaries)

            # Create summary record
            return self.summary_store.create_summary(
                CreateSummaryInput(
                    conversation_id=conversation_id,
                    kind="condensed",
                    depth=depth,
                    content=summary_content,
                    token_count=token_count,
                    earliest_at=earliest_at,
                    latest_at=latest_at,
                    descendant_count=descendant_count,
                    descendant_token_count=descendant_token_count,
                    source_message_token_count=source_token_count,
                    model=getattr(self.summarizer, "model", "unknown"),
                )
            )

        except Exception as e:
            logger.error(f"Failed to create condensed summary: {e}")
            return None
