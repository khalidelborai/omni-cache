"""
Size-based strategy implementation.

Implements size-based eviction strategy with memory usage tracking
and automatic eviction when size limits are exceeded.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from collections import defaultdict

from omnicache.models.strategy import Strategy

if TYPE_CHECKING:
    from omnicache.models.cache import Cache


class SizeStrategy(Strategy):
    """
    Size-based eviction strategy.

    Tracks memory usage and evicts entries when size limits are exceeded,
    prioritizing larger entries and older entries for eviction.
    """

    def __init__(
        self,
        max_size_bytes: Optional[int] = None,
        eviction_threshold: float = 0.9,
        eviction_batch_size: int = 10,
        **kwargs: Any
    ) -> None:
        """
        Initialize Size strategy.

        Args:
            max_size_bytes: Maximum total cache size in bytes
            eviction_threshold: Size threshold (0.0-1.0) to trigger eviction
            eviction_batch_size: Number of entries to evict at once
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="size",
            max_size_bytes=max_size_bytes,
            eviction_threshold=eviction_threshold,
            eviction_batch_size=eviction_batch_size,
            **kwargs
        )
        self.max_size_bytes = max_size_bytes
        self.eviction_threshold = eviction_threshold
        self.eviction_batch_size = eviction_batch_size

        # Track entry sizes and access times
        self._entry_sizes: Dict[str, int] = {}
        self._access_times: Dict[str, float] = {}

        # Statistics
        self._total_bytes_stored = 0
        self._total_bytes_evicted = 0
        self._evictions_performed = 0
        self._size_violations = 0

    async def initialize(self, cache: 'Cache') -> None:
        """Initialize strategy with cache context."""
        await super().initialize(cache)

        # Initialize size tracking for existing entries
        if hasattr(cache.backend, '_entries'):
            current_time = datetime.now().timestamp()
            for key, entry in cache.backend._entries.items():
                size_bytes = entry.size_bytes if hasattr(entry, 'size_bytes') else 0
                self._entry_sizes[key] = size_bytes
                self._access_times[key] = current_time
                self._total_bytes_stored += size_bytes

    async def shutdown(self) -> None:
        """Cleanup strategy resources."""
        self._entry_sizes.clear()
        self._access_times.clear()
        await super().shutdown()

    async def should_evict(self, cache: 'Cache') -> bool:
        """
        Determine if eviction should occur based on size limits.

        Args:
            cache: Cache instance to check

        Returns:
            True if eviction should occur, False otherwise
        """
        if self.max_size_bytes is None:
            return False

        current_size = await self.get_current_size_bytes()
        threshold_size = self.max_size_bytes * self.eviction_threshold

        return current_size >= threshold_size

    async def select_eviction_candidates(self, cache: 'Cache', count: int = 1) -> List[str]:
        """
        Select entries for eviction based on size and access patterns.

        Prioritizes entries by a combination of:
        1. Large size (to free more memory)
        2. Older access time (LRU-like behavior)

        Args:
            cache: Cache instance
            count: Number of entries to select

        Returns:
            List of cache entry keys to evict
        """
        candidates = []

        # Calculate eviction scores for all entries
        current_time = datetime.now().timestamp()

        for key, size_bytes in self._entry_sizes.items():
            if await cache.backend.exists(key):
                access_time = self._access_times.get(key, 0)
                age_seconds = max(1, current_time - access_time)

                # Score = size * age_factor
                # Larger and older entries get higher scores
                age_factor = min(age_seconds / 3600, 24)  # Cap at 24 hours
                score = size_bytes * (1 + age_factor)

                candidates.append((score, key, size_bytes, access_time))
            else:
                # Clean up tracking for non-existent keys
                self._entry_sizes.pop(key, None)
                self._access_times.pop(key, None)

        # Sort by score (descending) - highest scores evicted first
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Return the requested number of keys
        return [key for _, key, _, _ in candidates[:count]]

    async def should_expire(self, entry: Any) -> bool:
        """
        Size strategy doesn't handle TTL expiration directly.

        Args:
            entry: Cache entry to check

        Returns:
            False (Size strategy focuses on memory management)
        """
        return False

    async def on_access(self, key: str, entry: Any) -> None:
        """
        Handle entry access event.

        Args:
            key: Accessed entry key
            entry: Cache entry that was accessed
        """
        # Update access time
        self._access_times[key] = datetime.now().timestamp()

        # Update size tracking if changed
        if hasattr(entry, 'size_bytes'):
            old_size = self._entry_sizes.get(key, 0)
            new_size = entry.size_bytes

            if old_size != new_size:
                self._entry_sizes[key] = new_size
                self._total_bytes_stored += (new_size - old_size)

    async def on_insert(self, key: str, entry: Any) -> None:
        """
        Handle entry insertion event.

        Args:
            key: Inserted entry key
            entry: Cache entry that was inserted
        """
        # Track size and access time
        size_bytes = entry.size_bytes if hasattr(entry, 'size_bytes') else 0
        self._entry_sizes[key] = size_bytes
        self._access_times[key] = datetime.now().timestamp()
        self._total_bytes_stored += size_bytes

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
        # Update size tracking
        old_size = self._entry_sizes.get(key, 0)
        new_size = new_entry.size_bytes if hasattr(new_entry, 'size_bytes') else 0

        self._entry_sizes[key] = new_size
        self._access_times[key] = datetime.now().timestamp()
        self._total_bytes_stored += (new_size - old_size)

        # Check if eviction is needed after size increase
        cache = getattr(self, '_cache', None)
        if cache and await self.should_evict(cache):
            await self._perform_eviction(cache)

    async def on_evict(self, key: str, entry: Any) -> None:
        """
        Handle entry eviction event.

        Args:
            key: Evicted entry key
            entry: Cache entry that was evicted
        """
        self._evictions_performed += 1

        # Update size tracking
        size_bytes = self._entry_sizes.pop(key, 0)
        self._access_times.pop(key, None)
        self._total_bytes_stored -= size_bytes
        self._total_bytes_evicted += size_bytes

    async def get_current_size_bytes(self) -> int:
        """
        Get current total size in bytes.

        Returns:
            Current cache size in bytes
        """
        return self._total_bytes_stored

    async def get_size_stats(self) -> Dict[str, Any]:
        """Get detailed size statistics."""
        sizes = list(self._entry_sizes.values())

        return {
            "total_bytes": self._total_bytes_stored,
            "total_bytes_evicted": self._total_bytes_evicted,
            "evictions_performed": self._evictions_performed,
            "size_violations": self._size_violations,
            "tracked_entries": len(self._entry_sizes),
            "max_size_bytes": self.max_size_bytes,
            "eviction_threshold": self.eviction_threshold,
            "avg_entry_size": sum(sizes) / len(sizes) if sizes else 0,
            "max_entry_size": max(sizes) if sizes else 0,
            "min_entry_size": min(sizes) if sizes else 0,
            "size_utilization": (
                self._total_bytes_stored / self.max_size_bytes
                if self.max_size_bytes else 0
            )
        }

    async def get_largest_entries(self, count: int = 10) -> List[Tuple[str, int]]:
        """
        Get the largest cache entries.

        Args:
            count: Number of entries to return

        Returns:
            List of (key, size_bytes) tuples sorted by size
        """
        entries = [(key, size) for key, size in self._entry_sizes.items()]
        entries.sort(key=lambda x: x[1], reverse=True)
        return entries[:count]

    async def get_size_distribution(self) -> Dict[str, int]:
        """
        Get distribution of entries by size buckets.

        Returns:
            Dictionary mapping size ranges to entry counts
        """
        distribution = defaultdict(int)

        for size_bytes in self._entry_sizes.values():
            if size_bytes < 1024:
                distribution["< 1KB"] += 1
            elif size_bytes < 10 * 1024:
                distribution["1-10KB"] += 1
            elif size_bytes < 100 * 1024:
                distribution["10-100KB"] += 1
            elif size_bytes < 1024 * 1024:
                distribution["100KB-1MB"] += 1
            elif size_bytes < 10 * 1024 * 1024:
                distribution["1-10MB"] += 1
            else:
                distribution["> 10MB"] += 1

        return dict(distribution)

    async def _perform_eviction(self, cache: 'Cache') -> None:
        """Perform size-based eviction."""
        try:
            if not self.max_size_bytes:
                return

            current_size = await self.get_current_size_bytes()

            # Continue evicting until under threshold
            while current_size > self.max_size_bytes * self.eviction_threshold:
                # Get eviction candidates
                candidates = await self.select_eviction_candidates(
                    cache, self.eviction_batch_size
                )

                if not candidates:
                    break  # No more candidates

                # Evict selected entries
                evicted_any = False
                for key in candidates:
                    try:
                        entry = await cache.backend.get_entry(key)
                        if entry:
                            await self.on_evict(key, entry)
                            await cache.backend.delete(key)
                            evicted_any = True
                    except Exception:
                        # Continue with other evictions
                        continue

                if not evicted_any:
                    break  # Couldn't evict anything

                # Update current size
                current_size = await self.get_current_size_bytes()

                # Prevent infinite loop
                if current_size <= 0:
                    break

        except Exception:
            # Don't let eviction errors break the cache operation
            pass

    def get_memory_pressure(self) -> float:
        """
        Get current memory pressure (0.0 = no pressure, 1.0 = at limit).

        Returns:
            Memory pressure ratio
        """
        if not self.max_size_bytes:
            return 0.0

        return min(1.0, self._total_bytes_stored / self.max_size_bytes)

    def is_over_threshold(self) -> bool:
        """
        Check if cache is over the eviction threshold.

        Returns:
            True if over threshold, False otherwise
        """
        if not self.max_size_bytes:
            return False

        threshold_size = self.max_size_bytes * self.eviction_threshold
        return self._total_bytes_stored >= threshold_size

    def bytes_to_free(self) -> int:
        """
        Calculate how many bytes need to be freed to reach threshold.

        Returns:
            Bytes to free (0 if under threshold)
        """
        if not self.max_size_bytes:
            return 0

        threshold_size = self.max_size_bytes * self.eviction_threshold
        if self._total_bytes_stored > threshold_size:
            return self._total_bytes_stored - threshold_size
        return 0

    def validate_config(self) -> None:
        """Validate strategy configuration."""
        super().validate_config()

        if self.max_size_bytes is not None and self.max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive")

        if not (0.0 <= self.eviction_threshold <= 1.0):
            raise ValueError("eviction_threshold must be between 0.0 and 1.0")

        if self.eviction_batch_size <= 0:
            raise ValueError("eviction_batch_size must be positive")

    def get_info(self) -> Dict[str, Any]:
        """Get strategy information with size-specific details."""
        base_info = super().get_info()

        # Get size stats synchronously for info
        sizes = list(self._entry_sizes.values())
        size_stats = {
            "total_bytes": self._total_bytes_stored,
            "total_bytes_evicted": self._total_bytes_evicted,
            "evictions_performed": self._evictions_performed,
            "tracked_entries": len(self._entry_sizes),
            "max_size_bytes": self.max_size_bytes,
            "eviction_threshold": self.eviction_threshold,
            "avg_entry_size": sum(sizes) / len(sizes) if sizes else 0,
            "memory_pressure": self.get_memory_pressure(),
            "over_threshold": self.is_over_threshold(),
            "bytes_to_free": self.bytes_to_free()
        }

        base_info.update({
            "max_size_bytes": self.max_size_bytes,
            "eviction_threshold": self.eviction_threshold,
            "eviction_batch_size": self.eviction_batch_size,
            "size_stats": size_stats
        })
        return base_info

    def __str__(self) -> str:
        size_mb = (self.max_size_bytes / (1024 * 1024)) if self.max_size_bytes else "unlimited"
        current_mb = self._total_bytes_stored / (1024 * 1024)
        return f"Size(max={size_mb}MB, current={current_mb:.1f}MB)"