from api.rate_limit import InMemoryRateLimiter


def test_rate_limiter_blocks_after_limit():
    limiter = InMemoryRateLimiter(limit=2, window_seconds=60)
    assert limiter.allow("key") is True
    assert limiter.allow("key") is True
    assert limiter.allow("key") is False
