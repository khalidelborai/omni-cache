"""
LRU (Least Recently Used) strategy implementation.

Implements the Least Recently Used eviction strategy with efficient
access tracking and automatic eviction based on usage patterns.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from collections import OrderedDict
from datetime import datetime

from omnicache.models.strategy import Strategy

if TYPE_CHECKING:
    from omnicache.models.cache import Cache


class LRUStrategy(Strategy):
    """
    Least Recently Used eviction strategy.

    Tracks access order and evicts the least recently used entries
    when cache reaches capacity limits.
    """

    def __init__(self, max_size: Optional[int] = None, **kwargs: Any) -> None:
        """
        Initialize LRU strategy.

        Args:
            max_size: Maximum number of entries before eviction
            **kwargs: Additional configuration parameters
        """
        super().__init__(name="lru", max_size=max_size, **kwargs)
        self.max_size = max_size

        # Track access order - most recent at end
        self._access_order: OrderedDict[str, float] = OrderedDict()

        # Statistics
        self._total_accesses = 0
        self._evictions_performed = 0
        self._cache_hits = 0
        self._cache_misses = 0

    async def initialize(self, cache: 'Cache') -> None:
        """Initialize strategy with cache context."""
        await super().initialize(cache)

        # Initialize access tracking for existing entries
        if hasattr(cache.backend, '_entries'):
            for key in cache.backend._entries.keys():
                self._access_order[key] = datetime.now().timestamp()

    async def shutdown(self) -> None:
        """Cleanup strategy resources."""
        self._access_order.clear()
        await super().shutdown()

    async def should_evict(self, cache: 'Cache') -> bool:
        """
        Determine if eviction should occur.

        Args:
            cache: Cache instance to check

        Returns:
            True if eviction should occur, False otherwise
        """
        if self.max_size is None:
            return False

        current_size = await cache.backend.get_size()
        return current_size >= self.max_size

    async def select_eviction_candidates(self, cache: 'Cache', count: int = 1) -> List[str]:
        """
        Select entries for eviction based on LRU order.

        Args:
            cache: Cache instance
            count: Number of entries to select

        Returns:
            List of cache entry keys to evict (least recently used first)
        """
        candidates = []

        # Get keys in LRU order (oldest first)
        lru_keys = list(self._access_order.keys())

        # Verify keys still exist in cache
        valid_candidates = []
        for key in lru_keys:
            if await cache.backend.exists(key):
                valid_candidates.append(key)
            else:
                # Clean up tracking for non-existent keys
                self._access_order.pop(key, None)

        # Return the requested number of candidates
        return valid_candidates[:count]

    async def should_expire(self, entry: Any) -> bool:
        """
        Determine if an entry should expire.

        LRU strategy doesn't handle TTL expiration directly.

        Args:
            entry: Cache entry to check

        Returns:
            False (LRU doesn't expire entries based on time)
        """
        return False

    async def on_access(self, key: str, entry: Any) -> None:
        """
        Handle entry access event.

        Updates the access order for LRU tracking.

        Args:
            key: Accessed entry key
            entry: Cache entry that was accessed
        """
        self._total_accesses += 1
        self._cache_hits += 1

        # Update access time and move to end (most recent)
        current_time = datetime.now().timestamp()
        self._access_order.pop(key, None)  # Remove if exists
        self._access_order[key] = current_time

    async def on_insert(self, key: str, entry: Any) -> None:
        """
        Handle entry insertion event.

        Args:
            key: Inserted entry key
            entry: Cache entry that was inserted
        """
        # Track new entry
        current_time = datetime.now().timestamp()
        self._access_order[key] = current_time

        # Check if eviction is needed
        cache = getattr(self, '_cache', None)
        if cache and await self.should_evict(cache):
            await self._perform_eviction(cache)

    async def on_update(self, key: str, old_entry: Any, new_entry: Any) -> None:
        """
        Handle entry update event.

        Args:
            key: Updated entry key
            old_entry: Previous cache entry
            new_entry: New cache entry
        """
        # Treat update as access
        await self.on_access(key, new_entry)

    async def on_evict(self, key: str, entry: Any) -> None:
        """
        Handle entry eviction event.

        Args:
            key: Evicted entry key
            entry: Cache entry that was evicted
        """
        self._evictions_performed += 1

        # Remove from tracking
        self._access_order.pop(key, None)

    async def _perform_eviction(self, cache: 'Cache') -> None:
        """Perform eviction of least recently used entries."""
        try:
            # Determine how many entries to evict
            current_size = await cache.backend.get_size()
            if self.max_size and current_size >= self.max_size:
                evict_count = current_size - self.max_size + 1

                # Get eviction candidates
                candidates = await self.select_eviction_candidates(cache, evict_count)

                # Evict selected entries
                for key in candidates:
                    try:
                        entry = await cache.backend.get_entry(key)
                        if entry:
                            await self.on_evict(key, entry)
                            await cache.backend.delete(key)
                    except Exception:
                        # Continue with other evictions
                        continue

        except Exception:
            # Don't let eviction errors break the cache operation
            pass

    def get_access_stats(self) -> Dict[str, Any]:
        """Get access statistics for this strategy."""
        return {
            "total_accesses": self._total_accesses,
            "evictions_performed": self._evictions_performed,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "tracked_entries": len(self._access_order),
            "max_size": self.max_size
        }

    def get_access_order(self) -> List[str]:
        """
        Get current access order.

        Returns:
            List of keys in access order (least recent first)
        """
        return list(self._access_order.keys())

    def get_least_recent_key(self) -> Optional[str]:
        """
        Get the least recently used key.

        Returns:
            Key of least recently used entry, or None if empty
        """
        if self._access_order:
            return next(iter(self._access_order))
        return None

    def get_most_recent_key(self) -> Optional[str]:
        """
        Get the most recently used key.

        Returns:
            Key of most recently used entry, or None if empty
        """
        if self._access_order:
            return next(reversed(self._access_order))
        return None

    def clear_tracking(self) -> None:
        """Clear all access tracking data."""
        self._access_order.clear()

    def validate_config(self) -> None:
        """Validate strategy configuration."""
        super().validate_config()

        if self.max_size is not None and self.max_size <= 0:
            raise ValueError("max_size must be positive")

    def get_info(self) -> Dict[str, Any]:
        """Get strategy information with LRU-specific details."""
        base_info = super().get_info()
        base_info.update({
            "max_size": self.max_size,
            "tracked_entries": len(self._access_order),
            "least_recent_key": self.get_least_recent_key(),
            "most_recent_key": self.get_most_recent_key(),
            "access_stats": self.get_access_stats()
        })
        return base_info

    def __str__(self) -> str:
        return f"LRU(max_size={self.max_size}, tracked={len(self._access_order)})"