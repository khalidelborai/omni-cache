"""
Unit tests for Rate Limiter implementation.
"""

import pytest
import asyncio
from omnicache.core.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitExceededError,
    TokenBucket,
    rate_limited,
)


class TestTokenBucket:
    """Test cases for TokenBucket class."""

    @pytest.mark.asyncio
    async def test_initial_tokens_equals_capacity(self):
        """Test bucket starts with full capacity."""
        bucket = TokenBucket(rate=10, capacity=100)
        assert bucket.tokens == 100

    @pytest.mark.asyncio
    async def test_acquire_reduces_tokens(self):
        """Test acquiring tokens reduces available tokens."""
        bucket = TokenBucket(rate=10, capacity=100)

        allowed, _ = await bucket.acquire(10)
        assert allowed
        assert bucket.tokens == 90

    @pytest.mark.asyncio
    async def test_acquire_fails_when_insufficient(self):
        """Test acquire fails when not enough tokens."""
        bucket = TokenBucket(rate=10, capacity=5)

        allowed, wait_time = await bucket.acquire(10)
        assert not allowed
        assert wait_time > 0

    @pytest.mark.asyncio
    async def test_tokens_replenish_over_time(self):
        """Test tokens replenish based on rate."""
        bucket = TokenBucket(rate=100, capacity=100)

        # Use all tokens
        await bucket.acquire(100)
        assert bucket.tokens == 0

        # Wait for replenishment
        await asyncio.sleep(0.1)

        # Should have some tokens now
        allowed, _ = await bucket.acquire(5)
        assert allowed

    @pytest.mark.asyncio
    async def test_try_acquire_with_wait(self):
        """Test try_acquire with waiting."""
        bucket = TokenBucket(rate=100, capacity=10)

        # Use all tokens
        await bucket.acquire(10)

        # Try to acquire with wait
        result = await bucket.try_acquire(5, wait=True, max_wait=0.2)
        assert result


class TestRateLimiter:
    """Test cases for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self):
        """Test limiter allows requests under the limit."""
        config = RateLimitConfig(requests_per_second=100, burst_size=50)
        limiter = RateLimiter(config=config)

        for _ in range(10):
            allowed, _ = await limiter.check_limit()
            assert allowed

    @pytest.mark.asyncio
    async def test_rejects_requests_over_burst(self):
        """Test limiter rejects requests over burst size."""
        config = RateLimitConfig(requests_per_second=10, burst_size=5)
        limiter = RateLimiter(config=config)

        # Should allow up to burst_size
        for _ in range(5):
            allowed, _ = await limiter.check_limit()
            assert allowed

        # Should reject after that
        allowed, wait_time = await limiter.check_limit()
        assert not allowed
        assert wait_time > 0

    @pytest.mark.asyncio
    async def test_per_key_limiting(self):
        """Test per-key rate limiting."""
        config = RateLimitConfig(
            requests_per_second=100,
            burst_size=100,
            per_key=True,
            per_key_requests_per_second=10,
            per_key_burst_size=2
        )
        limiter = RateLimiter(config=config)

        # Use up per-key limit
        for _ in range(2):
            allowed, _ = await limiter.check_limit(key="test_key")
            assert allowed

        # Per-key limit should be exceeded
        allowed, _ = await limiter.check_limit(key="test_key")
        assert not allowed

        # Different key should still work
        allowed, _ = await limiter.check_limit(key="other_key")
        assert allowed

    @pytest.mark.asyncio
    async def test_disabled_limiter_allows_all(self):
        """Test disabled limiter allows all requests."""
        config = RateLimitConfig(enabled=False, burst_size=1)
        limiter = RateLimiter(config=config)

        for _ in range(100):
            allowed, _ = await limiter.check_limit()
            assert allowed

    @pytest.mark.asyncio
    async def test_raises_on_limit_exceeded(self):
        """Test raise_on_limit parameter."""
        config = RateLimitConfig(burst_size=1)
        limiter = RateLimiter(config=config)

        # First request allowed
        await limiter.check_limit(raise_on_limit=True)

        # Second should raise
        with pytest.raises(RateLimitExceededError) as exc_info:
            await limiter.check_limit(raise_on_limit=True)

        assert exc_info.value.wait_time > 0

    @pytest.mark.asyncio
    async def test_wait_for_token(self):
        """Test waiting for token to become available."""
        config = RateLimitConfig(requests_per_second=100, burst_size=1)
        limiter = RateLimiter(config=config)

        # Use the token
        await limiter.check_limit()

        # Wait for another token
        result = await limiter.wait_for_token(max_wait=0.1)
        assert result

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting limiter statistics."""
        limiter = RateLimiter()

        await limiter.check_limit()
        await limiter.check_limit()

        stats = limiter.get_stats()
        assert stats["requests"] == 2
        assert stats["allowed"] == 2
        assert stats["rejected"] == 0

    @pytest.mark.asyncio
    async def test_reset_stats(self):
        """Test resetting statistics."""
        limiter = RateLimiter()

        await limiter.check_limit()
        await limiter.reset_stats()

        stats = limiter.get_stats()
        assert stats["requests"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_key_buckets(self):
        """Test cleaning up old key buckets."""
        config = RateLimitConfig(per_key=True)
        limiter = RateLimiter(config=config)

        await limiter.check_limit(key="test_key")
        assert limiter.get_stats()["active_key_buckets"] == 1

        # Cleanup with very short max_age should remove all
        removed = await limiter.cleanup_key_buckets(max_age_seconds=0)
        assert removed == 1


class TestRateLimitedDecorator:
    """Test cases for rate_limited decorator."""

    @pytest.mark.asyncio
    async def test_decorator_allows_under_limit(self):
        """Test decorator allows calls under limit."""
        limiter = RateLimiter()

        @rate_limited(limiter)
        async def my_func(x):
            return x * 2

        result = await my_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_decorator_rejects_over_limit(self):
        """Test decorator rejects calls over limit."""
        config = RateLimitConfig(burst_size=1)
        limiter = RateLimiter(config=config)

        @rate_limited(limiter)
        async def my_func():
            return "success"

        # First call allowed
        await my_func()

        # Second should fail
        with pytest.raises(RateLimitExceededError):
            await my_func()

    @pytest.mark.asyncio
    async def test_decorator_with_key_extractor(self):
        """Test decorator with custom key extraction."""
        config = RateLimitConfig(per_key=True, per_key_burst_size=1)
        limiter = RateLimiter(config=config)

        @rate_limited(limiter, key_extractor=lambda key, value: key)
        async def set_value(key: str, value: str):
            return f"{key}={value}"

        # First call for key1
        await set_value("key1", "value1")

        # Second call for key1 should fail
        with pytest.raises(RateLimitExceededError):
            await set_value("key1", "value2")

        # But key2 should work
        result = await set_value("key2", "value1")
        assert result == "key2=value1"
