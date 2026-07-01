"""Provider payload construction (no network)."""

from bkvy.core.llm_client import LLMClient, ProviderAPIError


class TestGeminiPayload:
    def test_multi_turn_roles_explicit(self):
        messages = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ]
        payload = LLMClient._build_gemini_payload(messages, {})
        roles = [c["role"] for c in payload["contents"]]
        assert roles == ["user", "model", "user"]
        assert payload["systemInstruction"] == {"parts": [{"text": "be terse"}]}

    def test_stop_sequences_mapped(self):
        payload = LLMClient._build_gemini_payload(
            [{"role": "user", "content": "hi"}], {"stop": ["END"]})
        assert payload["generationConfig"]["stopSequences"] == ["END"]

    def test_thinking_disabled_sets_zero_budget(self):
        payload = LLMClient._build_gemini_payload(
            [{"role": "user", "content": "hi"}], {"disable_thinking": True})
        assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}

    def test_max_tokens_floor(self):
        payload = LLMClient._build_gemini_payload(
            [{"role": "user", "content": "hi"}], {"max_tokens": 5})
        assert payload["generationConfig"]["maxOutputTokens"] == 50


class TestProviderAPIError:
    def test_preserves_status_and_headers(self):
        e = ProviderAPIError("OpenAI API error 429: slow down",
                             status_code=429, headers={"Retry-After": "30"})
        assert e.status_code == 429
        assert e.headers["Retry-After"] == "30"
        assert "429" in str(e)
