"""
Rate limiting for cache operations using token bucket algorithm.

Provides protection against cache abuse and ensures fair resource usage.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple
from collections import defaultdict


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_second: float = 100.0
    burst_size: int = 200
    enabled: bool = True
    per_key: bool = False  # Limit per cache key vs global
    per_key_requests_per_second: float = 10.0  # Rate limit per individual key
    per_key_burst_size: int = 20


class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum bucket capacity
        """
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = datetime.now()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Tuple of (success, wait_time_if_failed)
        """
        async with self._lock:
            now = datetime.now()
            elapsed = (now - self.last_update).total_seconds()

            # Add tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0.0
            else:
                wait_time = (tokens - self.tokens) / self.rate
                return False, wait_time

    async def try_acquire(self, tokens: int = 1, wait: bool = False, max_wait: float = 1.0) -> bool:
        """
        Try to acquire tokens, optionally waiting.

        Args:
            tokens: Number of tokens to acquire
            wait: Whether to wait for tokens
            max_wait: Maximum wait time in seconds

        Returns:
            True if tokens acquired, False otherwise
        """
        allowed, wait_time = await self.acquire(tokens)
        if allowed:
            return True

        if wait and wait_time <= max_wait:
            await asyncio.sleep(wait_time)
            allowed, _ = await self.acquire(tokens)
            return allowed

        return False

    def get_stats(self) -> Dict:
        """Get bucket statistics."""
        return {
            "rate": self.rate,
            "capacity": self.capacity,
            "current_tokens": self.tokens,
            "last_update": self.last_update.isoformat(),
        }


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, wait_time: float, message: Optional[str] = None):
        super().__init__(message or f"Rate limit exceeded. Retry after {wait_time:.2f}s")
        self.wait_time = wait_time


class RateLimiter:
    """
    Rate limiter for cache operations.

    Supports both global and per-key rate limiting using the token bucket algorithm.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._global_bucket = TokenBucket(
            self.config.requests_per_second,
            self.config.burst_size
        )
        self._key_buckets: Dict[str, TokenBucket] = {}
        self._key_bucket_lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "requests": 0,
            "allowed": 0,
            "rejected": 0,
            "rejected_global": 0,
            "rejected_per_key": 0,
        }

    async def check_limit(
        self,
        key: Optional[str] = None,
        tokens: int = 1,
        raise_on_limit: bool = False
    ) -> Tuple[bool, float]:
        """
        Check if operation is within rate limits.

        Args:
            key: Optional key for per-key limiting
            tokens: Number of tokens to consume
            raise_on_limit: Raise exception if limit exceeded

        Returns:
            Tuple of (allowed, wait_time_if_rejected)

        Raises:
            RateLimitExceededError: If raise_on_limit is True and limit exceeded
        """
        if not self.config.enabled:
            return True, 0.0

        self._stats["requests"] += 1

        # Per-key limiting
        if self.config.per_key and key:
            allowed, wait = await self._check_key_limit(key, tokens)
            if not allowed:
                self._stats["rejected"] += 1
                self._stats["rejected_per_key"] += 1
                if raise_on_limit:
                    raise RateLimitExceededError(wait, f"Per-key rate limit exceeded for '{key}'")
                return False, wait

        # Global limiting
        allowed, wait = await self._global_bucket.acquire(tokens)
        if allowed:
            self._stats["allowed"] += 1
        else:
            self._stats["rejected"] += 1
            self._stats["rejected_global"] += 1
            if raise_on_limit:
                raise RateLimitExceededError(wait, "Global rate limit exceeded")

        return allowed, wait

    async def _check_key_limit(self, key: str, tokens: int = 1) -> Tuple[bool, float]:
        """Check per-key rate limit."""
        async with self._key_bucket_lock:
            if key not in self._key_buckets:
                self._key_buckets[key] = TokenBucket(
                    self.config.per_key_requests_per_second,
                    self.config.per_key_burst_size
                )

        return await self._key_buckets[key].acquire(tokens)

    async def wait_for_token(
        self,
        key: Optional[str] = None,
        tokens: int = 1,
        max_wait: float = 5.0
    ) -> bool:
        """
        Wait for rate limit token to become available.

        Args:
            key: Optional key for per-key limiting
            tokens: Number of tokens needed
            max_wait: Maximum time to wait in seconds

        Returns:
            True if token acquired, False if timed out
        """
        start_time = datetime.now()

        while True:
            allowed, wait_time = await self.check_limit(key, tokens)
            if allowed:
                return True

            # Check if we've waited too long
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed + wait_time > max_wait:
                return False

            # Wait for the suggested time
            await asyncio.sleep(min(wait_time, max_wait - elapsed))

    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            **self._stats,
            "rejection_rate": self._stats["rejected"] / max(self._stats["requests"], 1),
            "config": {
                "enabled": self.config.enabled,
                "requests_per_second": self.config.requests_per_second,
                "burst_size": self.config.burst_size,
                "per_key": self.config.per_key,
            },
            "global_bucket": self._global_bucket.get_stats(),
            "active_key_buckets": len(self._key_buckets),
        }

    async def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "requests": 0,
            "allowed": 0,
            "rejected": 0,
            "rejected_global": 0,
            "rejected_per_key": 0,
        }

    async def cleanup_key_buckets(self, max_age_seconds: float = 3600.0) -> int:
        """
        Clean up old per-key buckets.

        Args:
            max_age_seconds: Maximum age for unused buckets

        Returns:
            Number of buckets removed
        """
        now = datetime.now()
        removed = 0

        async with self._key_bucket_lock:
            keys_to_remove = []
            for key, bucket in self._key_buckets.items():
                age = (now - bucket.last_update).total_seconds()
                if age > max_age_seconds:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._key_buckets[key]
                removed += 1

        return removed


# Convenience function for creating rate-limited wrappers
def rate_limited(
    limiter: RateLimiter,
    key_extractor: Optional[callable] = None
):
    """
    Decorator to add rate limiting to async functions.

    Args:
        limiter: RateLimiter instance
        key_extractor: Function to extract cache key from arguments

    Usage:
        limiter = RateLimiter()

        @rate_limited(limiter)
        async def my_function():
            ...

        @rate_limited(limiter, key_extractor=lambda key, value: key)
        async def set_value(key: str, value: Any):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            key = None
            if key_extractor:
                try:
                    key = key_extractor(*args, **kwargs)
                except Exception:
                    pass

            allowed, wait_time = await limiter.check_limit(key)
            if not allowed:
                raise RateLimitExceededError(
                    wait_time,
                    f"Rate limit exceeded for {func.__name__}"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator
