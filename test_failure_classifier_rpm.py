#!/usr/bin/env python3
"""
Test that FailureClassifier distinguishes Gemini's per-minute (RPM) throttle from
the per-day (RPD) cap.

Both surface the metric "generate_content_free_tier_requests"; the RPM case carries a
short "retry in Xs" hint and must NOT bench the key until the daily reset (which would
waste the key's remaining RPD and starve the key pool). A genuine "PerDay" quotaId, or
a free-tier 429 with no short retry hint, is still treated as daily.
"""

from bkvy.core.failure_classifier import FailureClassifier

DAILY_FLOOR = 1800  # _seconds_until_gemini_daily_reset() floors at 30 min

RPM_429 = (
    'Gemini API rate limited 429: {"error": {"code": 429, "message": "You exceeded '
    'your current quota. * Quota exceeded for metric: generativelanguage.googleapis.com/'
    'generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash\\nPlease '
    'retry in 54.604646265s.", "status": "RESOURCE_EXHAUSTED"}}'
)

PER_DAY_429 = (
    'Gemini API rate limited 429: {"error": {"code": 429, "message": "Quota exceeded. '
    'quotaId: GenerateContentPerDayPerProjectPerModel-FreeTier, model: gemini-2.0-flash'
    '\\nPlease retry in 30s.", "status": "RESOURCE_EXHAUSTED"}}'
)

FREE_TIER_NO_HINT = (
    'Gemini API rate limited 429: Quota exceeded for metric: '
    'generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash'
)


def run() -> None:
    extract = FailureClassifier.extract_rate_limit_reset_time

    rpm = extract(RPM_429)
    assert rpm is not None and rpm <= 120, f"RPM throttle should honor short hint, got {rpm}"
    assert rpm == 55, f"expected int(54.6+1) = 55, got {rpm}"
    print(f"PASS: RPM throttle -> {rpm}s (not daily)")

    perday = extract(PER_DAY_429)
    assert perday >= DAILY_FLOOR, f"PerDay quotaId must be daily, got {perday}"
    print(f"PASS: PerDay quotaId -> {perday}s (daily, overrides retry hint)")

    no_hint = extract(FREE_TIER_NO_HINT)
    assert no_hint >= DAILY_FLOOR, f"free_tier with no hint must be daily, got {no_hint}"
    print(f"PASS: free_tier no-hint -> {no_hint}s (daily)")

    other = extract("Rate limited. retry after 30 seconds")
    assert other == 30, f"generic retry-after regression, got {other}"
    print(f"PASS: generic 'retry after 30 seconds' -> {other}s")

    print("\nAll failure_classifier RPM/RPD tests passed.")


if __name__ == "__main__":
    run()
