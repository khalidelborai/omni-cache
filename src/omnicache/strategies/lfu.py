"""
LFU (Least Frequently Used) strategy implementation.

Implements the Least Frequently Used eviction strategy with frequency
tracking and automatic eviction based on usage frequency patterns.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from collections import defaultdict
from datetime import datetime
import heapq

from omnicache.models.strategy import Strategy

if TYPE_CHECKING:
    from omnicache.models.cache import Cache


class LFUStrategy(Strategy):
    """
    Least Frequently Used eviction strategy.

    Tracks access frequency and evicts the least frequently used entries
    when cache reaches capacity limits.
    """

    def __init__(self, max_size: Optional[int] = None, **kwargs: Any) -> None:
        """
        Initialize LFU strategy.

        Args:
            max_size: Maximum number of entries before eviction
            **kwargs: Additional configuration parameters
        """
        super().__init__(name="lfu", max_size=max_size, **kwargs)
        self.max_size = max_size

        # Track access frequency
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._access_times: Dict[str, float] = {}  # For tie-breaking

        # Statistics
        self._total_accesses = 0
        self._evictions_performed = 0
        self._cache_hits = 0
        self._cache_misses = 0

    async def initialize(self, cache: 'Cache') -> None:
        """Initialize strategy with cache context."""
        await super().initialize(cache)

        # Initialize frequency tracking for existing entries
        if hasattr(cache.backend, '_entries'):
            current_time = datetime.now().timestamp()
            for key in cache.backend._entries.keys():
                self._access_counts[key] = 1  # Start with 1 access
                self._access_times[key] = current_time

    async def shutdown(self) -> None:
        """Cleanup strategy resources."""
        self._access_counts.clear()
        self._access_times.clear()
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
        Select entries for eviction based on LFU order.

        Args:
            cache: Cache instance
            count: Number of entries to select

        Returns:
            List of cache entry keys to evict (least frequently used first)
        """
        # Build a list of (frequency, last_access_time, key) for sorting
        candidates = []

        for key, frequency in self._access_counts.items():
            if await cache.backend.exists(key):
                last_access = self._access_times.get(key, 0)
                # Use heap with (frequency, last_access_time, key)
                # Lower frequency = higher priority for eviction
                # If frequencies are equal, use older access time
                candidates.append((frequency, last_access, key))
            else:
                # Clean up tracking for non-existent keys
                self._access_counts.pop(key, None)
                self._access_times.pop(key, None)

        # Sort by frequency (ascending), then by access time (ascending)
        candidates.sort(key=lambda x: (x[0], x[1]))

        # Return the requested number of keys
        return [key for _, _, key in candidates[:count]]

    async def should_expire(self, entry: Any) -> bool:
        """
        Determine if an entry should expire.

        LFU strategy doesn't handle TTL expiration directly.

        Args:
            entry: Cache entry to check

        Returns:
            False (LFU doesn't expire entries based on time)
        """
        return False

    async def on_access(self, key: str, entry: Any) -> None:
        """
        Handle entry access event.

        Updates the access frequency for LFU tracking.

        Args:
            key: Accessed entry key
            entry: Cache entry that was accessed
        """
        self._total_accesses += 1
        self._cache_hits += 1

        # Increment access count
        self._access_counts[key] += 1
        self._access_times[key] = datetime.now().timestamp()

    async def on_insert(self, key: str, entry: Any) -> None:
        """
        Handle entry insertion event.

        Args:
            key: Inserted entry key
            entry: Cache entry that was inserted
        """
        # Initialize frequency tracking for new entry
        self._access_counts[key] = 1  # Start with 1 access
        self._access_times[key] = datetime.now().timestamp()

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
        self._access_counts.pop(key, None)
        self._access_times.pop(key, None)

    async def _perform_eviction(self, cache: 'Cache') -> None:
        """Perform eviction of least frequently used entries."""
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
        frequencies = list(self._access_counts.values())
        return {
            "total_accesses": self._total_accesses,
            "evictions_performed": self._evictions_performed,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "tracked_entries": len(self._access_counts),
            "max_size": self.max_size,
            "avg_frequency": sum(frequencies) / len(frequencies) if frequencies else 0,
            "max_frequency": max(frequencies) if frequencies else 0,
            "min_frequency": min(frequencies) if frequencies else 0
        }

    def get_frequency_distribution(self) -> Dict[int, int]:
        """
        Get distribution of access frequencies.

        Returns:
            Dictionary mapping frequency to count of entries with that frequency
        """
        distribution = defaultdict(int)
        for frequency in self._access_counts.values():
            distribution[frequency] += 1
        return dict(distribution)

    def get_key_frequency(self, key: str) -> int:
        """
        Get access frequency for a specific key.

        Args:
            key: Cache key to check

        Returns:
            Access frequency for the key (0 if not tracked)
        """
        return self._access_counts.get(key, 0)

    def get_least_frequent_keys(self, count: int = 1) -> List[str]:
        """
        Get the least frequently used keys.

        Args:
            count: Number of keys to return

        Returns:
            List of keys with lowest access frequency
        """
        if not self._access_counts:
            return []

        # Create heap of (frequency, last_access_time, key)
        heap = []
        for key, frequency in self._access_counts.items():
            last_access = self._access_times.get(key, 0)
            heapq.heappush(heap, (frequency, last_access, key))

        # Extract the least frequent keys
        result = []
        for _ in range(min(count, len(heap))):
            if heap:
                _, _, key = heapq.heappop(heap)
                result.append(key)

        return result

    def get_most_frequent_keys(self, count: int = 1) -> List[str]:
        """
        Get the most frequently used keys.

        Args:
            count: Number of keys to return

        Returns:
            List of keys with highest access frequency
        """
        if not self._access_counts:
            return []

        # Sort by frequency (descending), then by access time (descending)
        sorted_items = sorted(
            [(freq, self._access_times.get(key, 0), key)
             for key, freq in self._access_counts.items()],
            key=lambda x: (x[0], x[1]),
            reverse=True
        )

        return [key for _, _, key in sorted_items[:count]]

    def clear_tracking(self) -> None:
        """Clear all frequency tracking data."""
        self._access_counts.clear()
        self._access_times.clear()

    def validate_config(self) -> None:
        """Validate strategy configuration."""
        super().validate_config()

        if self.max_size is not None and self.max_size <= 0:
            raise ValueError("max_size must be positive")

    def get_info(self) -> Dict[str, Any]:
        """Get strategy information with LFU-specific details."""
        base_info = super().get_info()
        base_info.update({
            "max_size": self.max_size,
            "tracked_entries": len(self._access_counts),
            "frequency_distribution": self.get_frequency_distribution(),
            "least_frequent_keys": self.get_least_frequent_keys(3),
            "most_frequent_keys": self.get_most_frequent_keys(3),
            "access_stats": self.get_access_stats()
        })
        return base_info

    def __str__(self) -> str:
        return f"LFU(max_size={self.max_size}, tracked={len(self._access_counts)})"