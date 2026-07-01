"""Failure classification and rate-limit reset extraction."""

from bkvy.core.failure_classifier import FailureClassifier
from bkvy.models.circuit_states import FailureType


class TestClassification:
    def test_context_length_is_content_error_not_rate_limit(self):
        msg = ("OpenAI API error 400: This model's maximum context length is 128000 tokens. "
               "code: context_length_exceeded")
        assert FailureClassifier.classify_error(msg) == FailureType.CONTENT_ERROR

    def test_anthropic_prompt_too_long(self):
        msg = "Anthropic API error 400: prompt is too long: 210000 tokens > 200000 maximum"
        assert FailureClassifier.classify_error(msg) == FailureType.CONTENT_ERROR

    def test_status_codes_win(self):
        assert FailureClassifier.classify_error("x", 429) == FailureType.RATE_LIMIT_429
        assert FailureClassifier.classify_error("x", 503) == FailureType.SERVICE_ERROR_5XX
        assert FailureClassifier.classify_error("x", 401) == FailureType.AUTH_ERROR_4XX
        assert FailureClassifier.classify_error("x", 404) == FailureType.MODEL_ERROR

    def test_retired_model_text_is_model_error(self):
        msg = ('Anthropic API error 404: {"type":"error","error":{"type":"not_found_error",'
               '"message":"model: claude-3-sonnet-20240229"}}')
        assert FailureClassifier.classify_error(msg) == FailureType.MODEL_ERROR

    def test_openai_model_does_not_exist(self):
        msg = "OpenAI API error 404: The model `gpt-nope` does not exist or you do not have access to it."
        assert FailureClassifier.classify_error(msg) == FailureType.MODEL_ERROR

    def test_rate_limit_text(self):
        msg = "Gemini API rate limited 429: RESOURCE_EXHAUSTED quota exceeded"
        assert FailureClassifier.classify_error(msg) == FailureType.RATE_LIMIT_429

    def test_auth_text(self):
        msg = "OpenAI API error 401: invalid api key"
        assert FailureClassifier.classify_error(msg) == FailureType.AUTH_ERROR_4XX

    def test_model_error_strategy_scopes_to_combination(self):
        strategy = FailureClassifier.get_strategy(FailureType.MODEL_ERROR)
        assert strategy.should_circuit_break
        assert strategy.skip_alternatives
        assert not strategy.skip_provider


class TestResetExtraction:
    def test_gemini_retry_in(self):
        assert FailureClassifier.extract_rate_limit_reset_time("Please retry in 43.2s") == 45 - 1

    def test_openai_go_duration(self):
        secs = FailureClassifier.extract_rate_limit_reset_time("Please try again in 6m59.56s")
        assert secs == 6 * 60 + 60

    def test_openai_plain_seconds(self):
        assert FailureClassifier.extract_rate_limit_reset_time("Please try again in 20s") == 21

    def test_headers_case_insensitive(self):
        assert FailureClassifier.extract_rate_limit_reset_time("err", headers={"retry-after": "42"}) == 42
        assert FailureClassifier.extract_rate_limit_reset_time("err", headers={"Retry-After": "42"}) == 42
        assert FailureClassifier.extract_rate_limit_reset_time("err", headers={"X-RateLimit-Reset": "30"}) == 30
