"""OpenAI-compat translation helpers."""

from bkvy.api.openai_compat import _normalize_tier, _map_finish_reason, _flatten_content


class TestTierNormalization:
    def test_plain_and_prefixed(self):
        assert _normalize_tier("low") == "low"
        assert _normalize_tier("bkvy-medium") == "medium"
        assert _normalize_tier("bkvy/high") == "high"
        assert _normalize_tier("BKVY-LOW".lower()) == "low"

    def test_unknown_is_none(self):
        assert _normalize_tier("gpt-4o") is None
        assert _normalize_tier(None) is None
        assert _normalize_tier("") is None


class TestFinishReason:
    def test_truncation_maps_to_length(self):
        assert _map_finish_reason("max_tokens", None) == "length"
        assert _map_finish_reason(None, True) == "length"
        assert _map_finish_reason("MAX_TOKENS".lower(), None) == "length"

    def test_stops(self):
        assert _map_finish_reason("end_turn", None) == "stop"
        assert _map_finish_reason(None, None) == "stop"

    def test_content_filter(self):
        assert _map_finish_reason("content_filter", None) == "content_filter"


class TestContentFlatten:
    def test_string_passthrough(self):
        assert _flatten_content("hello") == "hello"

    def test_parts_joined(self):
        parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"},
                 {"type": "image_url", "image_url": {"url": "x"}}]
        assert _flatten_content(parts) == "a\nb"

    def test_none_is_empty(self):
        assert _flatten_content(None) == ""
