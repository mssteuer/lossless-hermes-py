"""Message identity hashing for deduplication.

Provides stable content hashing for messages to detect duplicates
and ensure consistent identity across sessions.
"""

import hashlib


def build_message_identity_hash(role: str, content: str) -> str:
    """Build a stable hash for a message based on role and content.

    This is used to identify duplicate messages and ensure consistent
    identity across sessions and imports.

    Args:
        role: Message role (system, user, assistant, tool)
        content: Message content text

    Returns:
        SHA-256 hash as hex string
    """
    if not content:
        content = ""

    # Normalize role
    role = str(role).lower().strip()

    # Normalize content (preserve structure but normalize whitespace)
    content = str(content).strip()

    # Create hash input
    hash_input = f"role:{role}\ncontent:{content}"

    # Generate SHA-256 hash
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def normalize_message_content(content: str | None) -> str:
    """Normalize message content for consistent processing."""
    if content is None:
        return ""
    return str(content).strip()
