"""Tests for lossless_hermes.tokens module."""

from lossless_hermes.tokens import (
    estimate_code_point_tokens,
    estimate_conversation_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    is_cjk_code_point,
    truncate_text_to_estimated_tokens,
)


class TestIsCjkCodePoint:
    def test_ascii_not_cjk(self):
        assert not is_cjk_code_point(ord("A"))
        assert not is_cjk_code_point(ord(" "))

    def test_cjk_unified_ideographs(self):
        assert is_cjk_code_point(0x4E00)  # 一
        assert is_cjk_code_point(0x9FFF)

    def test_hiragana(self):
        assert is_cjk_code_point(ord("あ"))  # 0x3042

    def test_katakana(self):
        assert is_cjk_code_point(ord("ア"))  # 0x30A2

    def test_hangul(self):
        assert is_cjk_code_point(0xAC00)  # 가

    def test_fullwidth(self):
        assert is_cjk_code_point(0xFF01)  # ！

    def test_extension_b(self):
        assert is_cjk_code_point(0x20000)


class TestEstimateCodePointTokens:
    def test_ascii(self):
        assert estimate_code_point_tokens(ord("a")) == 0.25

    def test_cjk(self):
        assert estimate_code_point_tokens(0x4E00) == 1.5

    def test_supplementary_plane(self):
        # Emoji on supplementary plane (not CJK)
        assert estimate_code_point_tokens(0x1F600) == 2.0

    def test_cjk_extension_b_is_cjk_not_supplementary(self):
        # CJK Extension B is > 0xFFFF but should be 1.5 (CJK check first)
        assert estimate_code_point_tokens(0x20000) == 1.5


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0  # type: ignore — handled gracefully

    def test_ascii_text(self):
        text = "Hello world"  # 11 chars * 0.25 = 2.75 → ceil = 3
        assert estimate_tokens(text) == 3

    def test_cjk_text(self):
        text = "你好世界"  # 4 chars * 1.5 = 6
        assert estimate_tokens(text) == 6

    def test_mixed(self):
        text = "Hello 你好"  # 6 ascii * 0.25 + space * 0.25 + 2 CJK * 1.5
        # "Hello " = 6*0.25=1.5, "你好" = 2*1.5=3.0, total=4.5 → 5
        assert estimate_tokens(text) == 5

    def test_emoji(self):
        text = "😀"  # Supplementary plane → 2.0
        assert estimate_tokens(text) == 2

    def test_long_ascii(self):
        text = "a" * 100  # 100 * 0.25 = 25
        assert estimate_tokens(text) == 25


class TestTruncateTextToEstimatedTokens:
    def test_empty(self):
        assert truncate_text_to_estimated_tokens("", 10) == ""
        assert truncate_text_to_estimated_tokens("hello", 0) == ""

    def test_no_truncation_needed(self):
        text = "Hi"  # 2 * 0.25 = 0.5 → 1 token
        assert truncate_text_to_estimated_tokens(text, 10) == text

    def test_truncates_ascii(self):
        text = "a" * 100  # 25 tokens
        result = truncate_text_to_estimated_tokens(text, 10)
        assert estimate_tokens(result) <= 10

    def test_truncates_cjk(self):
        text = "你" * 20  # 30 tokens
        result = truncate_text_to_estimated_tokens(text, 6)
        assert estimate_tokens(result) <= 6
        assert len(result) == 4  # 4 * 1.5 = 6

    def test_preserves_cjk_chars(self):
        text = "你好世界朋友"
        result = truncate_text_to_estimated_tokens(text, 3)
        assert estimate_tokens(result) <= 3
        assert len(result) == 2  # 2 * 1.5 = 3


class TestEstimateMessagesTokens:
    def test_empty(self):
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        tokens = estimate_messages_tokens(msgs)
        # 10 overhead + estimate_tokens("hello") = 10 + 2 = 12
        assert tokens == 12

    def test_multiple_messages(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        tokens = estimate_messages_tokens(msgs)
        assert tokens == 10 + 2 + 10 + 1  # 23

    def test_with_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "search", "arguments": '{"q":"test"}'}}],
            }
        ]
        tokens = estimate_messages_tokens(msgs)
        # 10 (overhead) + 0 (empty content) + 5 (tool overhead) + tokens("search") + tokens(arguments)
        assert tokens > 15

    def test_no_content(self):
        msgs = [{"role": "user"}]
        tokens = estimate_messages_tokens(msgs)
        assert tokens == 10  # Just overhead


class TestEstimateConversationTokens:
    def test_adds_overhead(self):
        msgs = [{"role": "user", "content": "hello"}]
        base = estimate_messages_tokens(msgs)
        conv = estimate_conversation_tokens(msgs)
        assert conv > base
        assert conv == base + max(50, int(base * 0.05))
