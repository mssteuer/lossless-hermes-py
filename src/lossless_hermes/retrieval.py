"""Search and retrieval for LCM.

Provides FTS5-based search across messages and summaries with
fallback to regex search when needed.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from .store.conversation import ConversationStore, MessageSearchInput
from .store.summary import SummaryStore

logger = logging.getLogger(__name__)


@dataclass
class SearchQuery:
    """Represents a search query with options."""

    query: str
    conversation_id: int | None = None
    include_messages: bool = True
    include_summaries: bool = True
    limit: int = 50
    mode: str = "full_text"  # "full_text" or "regex"


@dataclass
class SearchResult:
    """Unified search result."""

    type: str  # "message" or "summary"
    id: int | str  # message_id or summary_id
    conversation_id: int
    content: str
    snippet: str
    relevance_score: float | None = None
    metadata: dict[str, Any] | None = None


class RetrievalEngine:
    """Search and retrieval engine for LCM."""

    def __init__(self, conversation_store: ConversationStore, summary_store: SummaryStore):
        self.conversation_store = conversation_store
        self.summary_store = summary_store

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Perform unified search across messages and summaries."""
        if not query.query.strip():
            return []

        results = []

        # Search messages if requested
        if query.include_messages:
            message_results = self._search_messages(query)
            results.extend(message_results)

        # Search summaries if requested
        if query.include_summaries:
            summary_results = self._search_summaries(query)
            results.extend(summary_results)

        # Sort by relevance and limit
        results = self._rank_and_limit_results(results, query.limit)

        return results

    def _search_messages(self, query: SearchQuery) -> list[SearchResult]:
        """Search messages using the conversation store."""

        search_input = MessageSearchInput(
            conversation_id=query.conversation_id,
            query=query.query,
            mode=query.mode,
            limit=query.limit,
            sort="relevance",
        )

        message_results = self.conversation_store.search_messages(search_input)

        # Convert to unified search results
        results = []
        for msg_result in message_results:
            results.append(
                SearchResult(
                    type="message",
                    id=msg_result.message_id,
                    conversation_id=msg_result.conversation_id,
                    content="",  # Don't include full content in results
                    snippet=msg_result.snippet,
                    relevance_score=msg_result.rank,
                    metadata={
                        "role": msg_result.role,
                        "created_at": msg_result.created_at.isoformat(),
                    },
                )
            )

        return results

    def _search_summaries(self, query: SearchQuery) -> list[SearchResult]:
        """Search summaries using the summary store."""

        summary_results = self.summary_store.search_summaries(
            conversation_id=query.conversation_id, query=query.query, limit=query.limit
        )

        # Convert to unified search results
        results = []
        for sum_result in summary_results:
            results.append(
                SearchResult(
                    type="summary",
                    id=sum_result["summary_id"],
                    conversation_id=sum_result["conversation_id"],
                    content="",  # Don't include full content in results
                    snippet=sum_result["snippet"],
                    relevance_score=None,  # FTS rank not available in current implementation
                    metadata={
                        "kind": sum_result["kind"],
                        "depth": sum_result["depth"],
                        "created_at": sum_result["created_at"].isoformat(),
                    },
                )
            )

        return results

    def _rank_and_limit_results(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
        """Rank results by relevance and apply limit."""

        # Simple ranking: prefer messages over summaries, then by relevance score
        def result_score(result: SearchResult) -> float:
            base_score = 1.0 if result.type == "message" else 0.5

            if result.relevance_score is not None:
                # Lower rank values are better in FTS
                return base_score + (1.0 / (1.0 + result.relevance_score))
            else:
                return base_score

        results.sort(key=result_score, reverse=True)

        return results[:limit]

    def get_full_content(self, result: SearchResult) -> str | None:
        """Get the full content for a search result."""

        if result.type == "message":
            messages = self.conversation_store.get_messages_by_conversation(result.conversation_id)
            for msg in messages:
                if msg.message_id == result.id:
                    return msg.content

        elif result.type == "summary":
            summary = self.summary_store.get_summary(str(result.id))
            if summary:
                return summary.content

        return None

    def search_similar_conversations(
        self, query: str, exclude_conversation_id: int | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find conversations with similar content."""

        # Search across all conversations
        search_query = SearchQuery(
            query=query,
            conversation_id=None,  # Search all conversations
            include_messages=True,
            include_summaries=True,
            limit=limit * 5,  # Get more results to group by conversation
        )

        results = self.search(search_query)

        # Group by conversation and score
        conversation_scores = {}
        for result in results:
            conv_id = result.conversation_id

            # Skip excluded conversation
            if exclude_conversation_id and conv_id == exclude_conversation_id:
                continue

            if conv_id not in conversation_scores:
                conversation_scores[conv_id] = {
                    "conversation_id": conv_id,
                    "score": 0.0,
                    "result_count": 0,
                    "best_snippet": "",
                    "best_type": "",
                }

            # Add to score
            score = result.relevance_score or 1.0
            conversation_scores[conv_id]["score"] += 1.0 / (1.0 + score)
            conversation_scores[conv_id]["result_count"] += 1

            # Keep best snippet
            if not conversation_scores[conv_id]["best_snippet"]:
                conversation_scores[conv_id]["best_snippet"] = result.snippet
                conversation_scores[conv_id]["best_type"] = result.type

        # Sort by score and return top results
        similar_conversations = list(conversation_scores.values())
        similar_conversations.sort(key=lambda x: x["score"], reverse=True)

        return similar_conversations[:limit]

    def find_related_messages(self, message_id: int, conversation_id: int, limit: int = 10) -> list[SearchResult]:
        """Find messages related to a specific message."""

        # Get the source message
        messages = self.conversation_store.get_messages_by_conversation(conversation_id)
        source_message = None

        for msg in messages:
            if msg.message_id == message_id:
                source_message = msg
                break

        if not source_message:
            return []

        # Extract key terms from the message (simple approach)
        keywords = self._extract_keywords(source_message.content)

        if not keywords:
            return []

        # Search for related content
        query_text = " OR ".join(keywords[:5])  # Use top 5 keywords

        search_query = SearchQuery(
            query=query_text,
            conversation_id=conversation_id,
            include_messages=True,
            include_summaries=True,
            limit=limit + 1,  # +1 to exclude the source message
            mode="full_text",
        )

        results = self.search(search_query)

        # Filter out the source message
        filtered_results = [r for r in results if not (r.type == "message" and r.id == message_id)]

        return filtered_results[:limit]

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text (simple implementation)."""
        if not text:
            return []

        # Simple keyword extraction
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
        }

        # Extract words (simple tokenization)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        # Filter out stop words
        keywords = [w for w in words if w not in stop_words]

        # Count frequency and return most common
        word_counts = {}
        for word in keywords:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency
        sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        return [word for word, count in sorted_keywords]
