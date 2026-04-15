"""Token estimation utilities for LCM.

Uses Unicode-aware character weighting instead of simple `text.length / 4`:
- CJK (Chinese/Japanese/Korean) characters: ~1.5 tokens/char
- Emoji / Supplementary Plane: ~2 tokens/char
- ASCII / Latin: ~0.25 tokens/char (≈ 4 chars/token)

Based on the TypeScript estimate-tokens.ts implementation.
"""

import math
from typing import Any


def is_cjk_code_point(cp: int) -> bool:
    """Detect CJK code points across all relevant Unicode ranges."""
    return (
        (0x4E00 <= cp <= 0x9FFF)  # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)  # CJK Extension A
        or (0x20000 <= cp <= 0x2A6DF)  # CJK Extension B
        or (0x2A700 <= cp <= 0x2B73F)  # CJK Extension C
        or (0x2B740 <= cp <= 0x2B81F)  # CJK Extension D
        or (0x2B820 <= cp <= 0x2CEAF)  # CJK Extension E
        or (0x2CEB0 <= cp <= 0x2EBEF)  # CJK Extension F
        or (0x3000 <= cp <= 0x303F)  # CJK Symbols and Punctuation
        or (0x3040 <= cp <= 0x30FF)  # Hiragana + Katakana
        or (0xAC00 <= cp <= 0xD7AF)  # Hangul Syllables
        or (0xFF00 <= cp <= 0xFFEF)  # Fullwidth Forms
    )


def estimate_code_point_tokens(cp: int) -> float:
    """Estimate token cost for a single Unicode code point."""
    if is_cjk_code_point(cp):
        return 1.5
    if cp > 0xFFFF:
        return 2.0
    return 0.25


def estimate_tokens(text: str) -> int:
    """Estimate text tokens using Unicode-aware character weighting."""
    if not text:
        return 0

    tokens = 0.0
    for char in text:
        cp = ord(char[0]) if char else 0
        tokens += estimate_code_point_tokens(cp)

    return math.ceil(tokens)


def truncate_text_to_estimated_tokens(text: str, max_tokens: int) -> str:
    """Truncate text so the estimated token count stays within max_tokens.

    Iterates by Unicode code point to avoid splitting surrogate pairs while
    preserving the same weighting model as estimate_tokens().
    """
    if max_tokens <= 0 or not text:
        return ""

    tokens = 0.0
    end = 0

    for char in text:
        cp = ord(char[0]) if char else 0
        next_tokens = tokens + estimate_code_point_tokens(cp)
        if math.ceil(next_tokens) > max_tokens:
            break
        tokens = next_tokens
        end += len(char)

    return text[:end]


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens for a list of OpenAI-format messages."""
    total_tokens = 0

    for message in messages:
        # Role and structure overhead
        total_tokens += 10

        # Content
        content = message.get("content", "")
        if content:
            total_tokens += estimate_tokens(str(content))

        # Tool calls
        tool_calls = message.get("tool_calls", [])
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                # Tool call overhead
                total_tokens += 5

                # Function name
                function = tool_call.get("function", {})
                if isinstance(function, dict):
                    name = function.get("name", "")
                    if name:
                        total_tokens += estimate_tokens(str(name))

                    # Function arguments
                    arguments = function.get("arguments", "")
                    if arguments:
                        total_tokens += estimate_tokens(str(arguments))

    return total_tokens


def estimate_conversation_tokens(conversation: list[dict[str, Any]]) -> int:
    """Estimate tokens for a full conversation including system overhead."""
    base_tokens = estimate_messages_tokens(conversation)

    # Add system/API overhead (response format, etc.)
    overhead = max(50, int(base_tokens * 0.05))

    return base_tokens + overhead
