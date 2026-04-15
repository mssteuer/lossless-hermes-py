"""Tests for lossless_claw.store.identity module."""

from lossless_claw.store.identity import build_message_identity_hash, normalize_message_content


class TestBuildMessageIdentityHash:
    def test_deterministic(self):
        h1 = build_message_identity_hash("user", "hello")
        h2 = build_message_identity_hash("user", "hello")
        assert h1 == h2

    def test_different_content(self):
        h1 = build_message_identity_hash("user", "hello")
        h2 = build_message_identity_hash("user", "world")
        assert h1 != h2

    def test_different_role(self):
        h1 = build_message_identity_hash("user", "hello")
        h2 = build_message_identity_hash("assistant", "hello")
        assert h1 != h2

    def test_role_normalized(self):
        h1 = build_message_identity_hash("USER", "hello")
        h2 = build_message_identity_hash("user", "hello")
        assert h1 == h2

    def test_role_stripped(self):
        h1 = build_message_identity_hash("  user  ", "hello")
        h2 = build_message_identity_hash("user", "hello")
        assert h1 == h2

    def test_content_stripped(self):
        h1 = build_message_identity_hash("user", "  hello  ")
        h2 = build_message_identity_hash("user", "hello")
        assert h1 == h2

    def test_empty_content(self):
        h = build_message_identity_hash("user", "")
        assert isinstance(h, str) and len(h) == 64  # SHA-256 hex

    def test_none_content(self):
        h = build_message_identity_hash("user", None)
        assert isinstance(h, str) and len(h) == 64

    def test_sha256_format(self):
        h = build_message_identity_hash("user", "test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestNormalizeMessageContent:
    def test_none(self):
        assert normalize_message_content(None) == ""

    def test_strips(self):
        assert normalize_message_content("  hello  ") == "hello"

    def test_normal(self):
        assert normalize_message_content("hello world") == "hello world"

    def test_empty(self):
        assert normalize_message_content("") == ""
