"""
TTL (Time To Live) strategy implementation.

Implements time-based expiration strategy with automatic cleanup
of expired entries and proactive expiration management.
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from datetime import datetime, timedelta
from collections import defaultdict

from omnicache.models.strategy import Strategy

if TYPE_CHECKING:
    from omnicache.models.cache import Cache


class TTLStrategy(Strategy):
    """
    Time To Live expiration strategy.

    Manages entry expiration based on TTL values and provides
    proactive cleanup of expired entries.
    """

    def __init__(
        self,
        default_ttl: Optional[float] = None,
        cleanup_interval: float = 60.0,
        batch_cleanup_size: int = 100,
        **kwargs: Any
    ) -> None:
        """
        Initialize TTL strategy.

        Args:
            default_ttl: Default TTL in seconds for entries without explicit TTL
            cleanup_interval: How often to run cleanup in seconds
            batch_cleanup_size: Maximum entries to clean up in one batch
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="ttl",
            default_ttl=default_ttl,
            cleanup_interval=cleanup_interval,
            batch_cleanup_size=batch_cleanup_size,
            **kwargs
        )
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.batch_cleanup_size = batch_cleanup_size

        # Track expiration times
        self._expiry_times: Dict[str, float] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

        # Statistics
        self._total_expirations = 0
        self._cleanup_runs = 0
        self._entries_cleaned = 0
        self._cleanup_errors = 0

    async def initialize(self, cache: 'Cache') -> None:
        """Initialize strategy with cache context."""
        await super().initialize(cache)

        # Initialize expiry tracking for existing entries
        if hasattr(cache.backend, '_entries'):
            for key, entry in cache.backend._entries.items():
                if entry.expires_at:
                    self._expiry_times[key] = entry.expires_at.timestamp()

        # Start cleanup task
        if self.cleanup_interval > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop(cache))

    async def shutdown(self) -> None:
        """Cleanup strategy resources."""
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._expiry_times.clear()
        await super().shutdown()

    async def should_evict(self, cache: 'Cache') -> bool:
        """
        TTL strategy doesn't evict based on size, only on expiration.

        Args:
            cache: Cache instance to check

        Returns:
            False (TTL strategy uses expiration, not eviction)
        """
        return False

    async def select_eviction_candidates(self, cache: 'Cache', count: int = 1) -> List[str]:
        """
        TTL strategy doesn't select candidates for eviction.

        Args:
            cache: Cache instance
            count: Number of entries to select

        Returns:
            Empty list (TTL strategy uses expiration)
        """
        return []

    async def should_expire(self, entry: Any) -> bool:
        """
        Determine if an entry should expire based on TTL.

        Args:
            entry: Cache entry to check

        Returns:
            True if entry has expired, False otherwise
        """
        if not hasattr(entry, 'expires_at') or entry.expires_at is None:
            return False

        return datetime.now() >= entry.expires_at

    async def on_access(self, key: str, entry: Any) -> None:
        """
        Handle entry access event.

        Args:
            key: Accessed entry key
            entry: Cache entry that was accessed
        """
        # Update expiry tracking if entry has TTL
        if hasattr(entry, 'expires_at') and entry.expires_at:
            self._expiry_times[key] = entry.expires_at.timestamp()

    async def on_insert(self, key: str, entry: Any) -> None:
        """
        Handle entry insertion event.

        Args:
            key: Inserted entry key
            entry: Cache entry that was inserted
        """
        # Set up expiry tracking
        if hasattr(entry, 'expires_at') and entry.expires_at:
            self._expiry_times[key] = entry.expires_at.timestamp()
        elif self.default_ttl:
            # Apply default TTL if no explicit TTL set
            expiry_time = datetime.now() + timedelta(seconds=self.default_ttl)
            self._expiry_times[key] = expiry_time.timestamp()

    async def on_update(self, key: str, old_entry: Any, new_entry: Any) -> None:
        """
        Handle entry update event.

        Args:
            key: Updated entry key
            old_entry: Previous cache entry
            new_entry: New cache entry
        """
        # Update expiry tracking
        if hasattr(new_entry, 'expires_at') and new_entry.expires_at:
            self._expiry_times[key] = new_entry.expires_at.timestamp()
        elif self.default_ttl:
            # Apply default TTL
            expiry_time = datetime.now() + timedelta(seconds=self.default_ttl)
            self._expiry_times[key] = expiry_time.timestamp()
        else:
            # Remove expiry tracking if no TTL
            self._expiry_times.pop(key, None)

    async def on_evict(self, key: str, entry: Any) -> None:
        """
        Handle entry eviction event.

        Args:
            key: Evicted entry key
            entry: Cache entry that was evicted
        """
        # Remove from expiry tracking
        self._expiry_times.pop(key, None)

    async def expire_entry(self, cache: 'Cache', key: str) -> bool:
        """
        Expire a specific cache entry.

        Args:
            cache: Cache instance
            key: Key of entry to expire

        Returns:
            True if entry was expired, False if not found or error
        """
        try:
            # Check if entry exists and is expired
            entry = await cache.backend.get_entry(key)
            if entry and await self.should_expire(entry):
                # Mark as expired and delete
                entry.mark_expired()
                await self.on_evict(key, entry)
                await cache.backend.delete(key)
                self._total_expirations += 1
                return True
            return False

        except Exception:
            self._cleanup_errors += 1
            return False

    async def cleanup_expired(self, cache: 'Cache', max_entries: Optional[int] = None) -> int:
        """
        Clean up expired entries from the cache.

        Args:
            cache: Cache instance
            max_entries: Maximum number of entries to clean up

        Returns:
            Number of entries cleaned up
        """
        current_time = datetime.now().timestamp()
        expired_keys = []

        # Find expired keys
        for key, expiry_time in list(self._expiry_times.items()):
            if current_time >= expiry_time:
                expired_keys.append(key)

            # Limit batch size
            if max_entries and len(expired_keys) >= max_entries:
                break

        # Clean up expired entries
        cleaned_count = 0
        for key in expired_keys:
            if await self.expire_entry(cache, key):
                cleaned_count += 1

        self._entries_cleaned += cleaned_count
        return cleaned_count

    async def get_expired_keys(self) -> List[str]:
        """
        Get list of keys that have expired.

        Returns:
            List of expired keys
        """
        current_time = datetime.now().timestamp()
        return [
            key for key, expiry_time in self._expiry_times.items()
            if current_time >= expiry_time
        ]

    async def get_expiring_soon(self, within_seconds: float = 60.0) -> List[str]:
        """
        Get list of keys that will expire within the specified time.

        Args:
            within_seconds: Time window in seconds

        Returns:
            List of keys expiring soon
        """
        threshold_time = datetime.now().timestamp() + within_seconds
        return [
            key for key, expiry_time in self._expiry_times.items()
            if expiry_time <= threshold_time
        ]

    async def extend_ttl(self, cache: 'Cache', key: str, additional_seconds: float) -> bool:
        """
        Extend the TTL of a cache entry.

        Args:
            cache: Cache instance
            key: Key of entry to extend
            additional_seconds: Seconds to add to TTL

        Returns:
            True if TTL was extended, False if entry not found or error
        """
        try:
            entry = await cache.backend.get_entry(key)
            if entry:
                # Extend TTL
                entry.extend_ttl(additional_seconds)

                # Update tracking
                if entry.expires_at:
                    self._expiry_times[key] = entry.expires_at.timestamp()

                return True
            return False

        except Exception:
            return False

    def get_ttl_remaining(self, key: str) -> Optional[float]:
        """
        Get remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, or None if no TTL set
        """
        if key not in self._expiry_times:
            return None

        expiry_time = self._expiry_times[key]
        current_time = datetime.now().timestamp()
        remaining = expiry_time - current_time

        return max(0.0, remaining)

    def get_expiry_distribution(self) -> Dict[str, int]:
        """
        Get distribution of entries by expiry time buckets.

        Returns:
            Dictionary mapping time ranges to entry counts
        """
        current_time = datetime.now().timestamp()
        distribution = defaultdict(int)

        for expiry_time in self._expiry_times.values():
            remaining = expiry_time - current_time

            if remaining <= 0:
                distribution["expired"] += 1
            elif remaining <= 60:
                distribution["1_minute"] += 1
            elif remaining <= 300:
                distribution["5_minutes"] += 1
            elif remaining <= 3600:
                distribution["1_hour"] += 1
            elif remaining <= 86400:
                distribution["1_day"] += 1
            else:
                distribution["longer"] += 1

        return dict(distribution)

    async def _cleanup_loop(self, cache: 'Cache') -> None:
        """Background cleanup loop for expired entries."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)

                # Run cleanup
                self._cleanup_runs += 1
                await self.cleanup_expired(cache, self.batch_cleanup_size)

            except asyncio.CancelledError:
                break
            except Exception:
                self._cleanup_errors += 1
                # Continue cleanup loop despite errors

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics for this strategy."""
        return {
            "total_expirations": self._total_expirations,
            "cleanup_runs": self._cleanup_runs,
            "entries_cleaned": self._entries_cleaned,
            "cleanup_errors": self._cleanup_errors,
            "tracked_entries": len(self._expiry_times),
            "cleanup_interval": self.cleanup_interval,
            "batch_cleanup_size": self.batch_cleanup_size,
            "default_ttl": self.default_ttl
        }

    def validate_config(self) -> None:
        """Validate strategy configuration."""
        super().validate_config()

        if self.default_ttl is not None and self.default_ttl <= 0:
            raise ValueError("default_ttl must be positive")

        if self.cleanup_interval <= 0:
            raise ValueError("cleanup_interval must be positive")

        if self.batch_cleanup_size <= 0:
            raise ValueError("batch_cleanup_size must be positive")

    def get_info(self) -> Dict[str, Any]:
        """Get strategy information with TTL-specific details."""
        base_info = super().get_info()
        base_info.update({
            "default_ttl": self.default_ttl,
            "cleanup_interval": self.cleanup_interval,
            "batch_cleanup_size": self.batch_cleanup_size,
            "tracked_entries": len(self._expiry_times),
            "expiry_distribution": self.get_expiry_distribution(),
            "cleanup_stats": self.get_cleanup_stats()
        })
        return base_info

    def __str__(self) -> str:
        return f"TTL(default_ttl={self.default_ttl}, tracked={len(self._expiry_times)})"