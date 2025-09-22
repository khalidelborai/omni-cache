"""
Priority-based strategy implementation.

Implements priority-based eviction strategy with configurable priority
levels and automatic eviction of lowest priority entries.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from collections import defaultdict
import heapq

from omnicache.models.strategy import Strategy

if TYPE_CHECKING:
    from omnicache.models.cache import Cache


class PriorityStrategy(Strategy):
    """
    Priority-based eviction strategy.

    Tracks entry priorities and evicts lowest priority entries when
    cache reaches capacity limits. Uses creation time for tie-breaking.
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        default_priority: float = 0.5,
        priority_levels: Optional[Dict[str, float]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize Priority strategy.

        Args:
            max_size: Maximum number of entries before eviction
            default_priority: Default priority for entries (0.0-1.0)
            priority_levels: Named priority levels mapping
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            name="priority",
            max_size=max_size,
            default_priority=default_priority,
            priority_levels=priority_levels,
            **kwargs
        )
        self.max_size = max_size
        self.default_priority = default_priority
        self.priority_levels = priority_levels or {
            "critical": 1.0,
            "high": 0.8,
            "normal": 0.5,
            "low": 0.2,
            "minimal": 0.1
        }

        # Track entry priorities and metadata
        self._entry_priorities: Dict[str, float] = {}
        self._creation_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)

        # Statistics
        self._total_accesses = 0
        self._evictions_performed = 0
        self._priority_changes = 0

    async def initialize(self, cache: 'Cache') -> None:
        """Initialize strategy with cache context."""
        await super().initialize(cache)

        # Initialize priority tracking for existing entries
        if hasattr(cache.backend, '_entries'):
            current_time = datetime.now().timestamp()
            for key, entry in cache.backend._entries.items():
                priority = getattr(entry, 'priority', self.default_priority)
                self._entry_priorities[key] = priority
                self._creation_times[key] = current_time
                self._access_counts[key] = getattr(entry, 'access_count', 1)

    async def shutdown(self) -> None:
        """Cleanup strategy resources."""
        self._entry_priorities.clear()
        self._creation_times.clear()
        self._access_counts.clear()
        await super().shutdown()

    async def should_evict(self, cache: 'Cache') -> bool:
        """
        Determine if eviction should occur based on size limits.

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
        Select entries for eviction based on priority.

        Selects entries with lowest priority first, using creation time
        and access count for tie-breaking.

        Args:
            cache: Cache instance
            count: Number of entries to select

        Returns:
            List of cache entry keys to evict (lowest priority first)
        """
        candidates = []

        # Build eviction candidates with priority-based scoring
        for key, priority in self._entry_priorities.items():
            if await cache.backend.exists(key):
                creation_time = self._creation_times.get(key, 0)
                access_count = self._access_counts.get(key, 1)

                # Lower priority = higher eviction score
                # Older entries (lower creation_time) = higher eviction score
                # Less accessed entries = higher eviction score
                base_score = 1.0 - priority  # Invert priority (0.0 = high priority, 1.0 = low priority)

                # Age factor (older entries have slightly higher eviction score)
                current_time = datetime.now().timestamp()
                age_factor = (current_time - creation_time) / 86400  # Days
                age_score = min(age_factor * 0.1, 0.2)  # Cap at 0.2

                # Access factor (less accessed entries have higher eviction score)
                access_score = max(0.0, 0.1 - (access_count * 0.01))  # Decrease score with more accesses

                final_score = base_score + age_score + access_score

                candidates.append((final_score, creation_time, key, priority))
            else:
                # Clean up tracking for non-existent keys
                self._entry_priorities.pop(key, None)
                self._creation_times.pop(key, None)
                self._access_counts.pop(key, None)

        # Sort by eviction score (descending), then by creation time (ascending for older first)
        candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        # Return the requested number of keys
        return [key for _, _, key, _ in candidates[:count]]

    async def should_expire(self, entry: Any) -> bool:
        """
        Priority strategy doesn't handle TTL expiration directly.

        Args:
            entry: Cache entry to check

        Returns:
            False (Priority strategy focuses on priority management)
        """
        return False

    async def on_access(self, key: str, entry: Any) -> None:
        """
        Handle entry access event.

        Args:
            key: Accessed entry key
            entry: Cache entry that was accessed
        """
        self._total_accesses += 1
        self._access_counts[key] += 1

        # Update priority tracking if changed
        if hasattr(entry, 'priority'):
            old_priority = self._entry_priorities.get(key, self.default_priority)
            new_priority = entry.priority

            if old_priority != new_priority:
                self._entry_priorities[key] = new_priority
                self._priority_changes += 1

    async def on_insert(self, key: str, entry: Any) -> None:
        """
        Handle entry insertion event.

        Args:
            key: Inserted entry key
            entry: Cache entry that was inserted
        """
        # Track priority and metadata
        priority = getattr(entry, 'priority', self.default_priority)
        self._entry_priorities[key] = priority
        self._creation_times[key] = datetime.now().timestamp()
        self._access_counts[key] = 1

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
        # Update priority tracking
        new_priority = getattr(new_entry, 'priority', self.default_priority)
        old_priority = self._entry_priorities.get(key, self.default_priority)

        if old_priority != new_priority:
            self._entry_priorities[key] = new_priority
            self._priority_changes += 1

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
        self._entry_priorities.pop(key, None)
        self._creation_times.pop(key, None)
        self._access_counts.pop(key, None)

    async def set_priority(self, cache: 'Cache', key: str, priority: float) -> bool:
        """
        Set priority for a specific cache entry.

        Args:
            cache: Cache instance
            key: Cache entry key
            priority: New priority value (0.0-1.0)

        Returns:
            True if priority was set, False if entry not found or error
        """
        try:
            # Validate priority
            if not (0.0 <= priority <= 1.0):
                raise ValueError("Priority must be between 0.0 and 1.0")

            # Update priority tracking
            if key in self._entry_priorities:
                old_priority = self._entry_priorities[key]
                self._entry_priorities[key] = priority

                if old_priority != priority:
                    self._priority_changes += 1

                # Update the entry in the backend if possible
                entry = await cache.backend.get_entry(key)
                if entry and hasattr(entry, 'update_priority'):
                    entry.update_priority(priority)

                return True

            return False

        except Exception:
            return False

    async def set_priority_by_name(self, cache: 'Cache', key: str, priority_name: str) -> bool:
        """
        Set priority for a cache entry using named priority level.

        Args:
            cache: Cache instance
            key: Cache entry key
            priority_name: Named priority level

        Returns:
            True if priority was set, False if name not found or error
        """
        if priority_name not in self.priority_levels:
            return False

        priority_value = self.priority_levels[priority_name]
        return await self.set_priority(cache, key, priority_value)

    async def get_entries_by_priority(self, min_priority: float, max_priority: float = 1.0) -> List[str]:
        """
        Get cache entries within a priority range.

        Args:
            min_priority: Minimum priority (inclusive)
            max_priority: Maximum priority (inclusive)

        Returns:
            List of keys matching priority criteria
        """
        matching_keys = []

        for key, priority in self._entry_priorities.items():
            if min_priority <= priority <= max_priority:
                matching_keys.append(key)

        return matching_keys

    async def get_highest_priority_keys(self, count: int = 10) -> List[str]:
        """
        Get the highest priority cache entries.

        Args:
            count: Number of entries to return

        Returns:
            List of keys sorted by priority (highest first)
        """
        # Sort by priority (descending), then by access count (descending)
        entries = [
            (priority, self._access_counts.get(key, 0), key)
            for key, priority in self._entry_priorities.items()
        ]
        entries.sort(key=lambda x: (x[0], x[1]), reverse=True)

        return [key for _, _, key in entries[:count]]

    async def get_lowest_priority_keys(self, count: int = 10) -> List[str]:
        """
        Get the lowest priority cache entries.

        Args:
            count: Number of entries to return

        Returns:
            List of keys sorted by priority (lowest first)
        """
        # Sort by priority (ascending), then by creation time (ascending for older first)
        entries = [
            (priority, self._creation_times.get(key, 0), key)
            for key, priority in self._entry_priorities.items()
        ]
        entries.sort(key=lambda x: (x[0], x[1]))

        return [key for _, _, key in entries[:count]]

    def get_priority_distribution(self) -> Dict[str, int]:
        """
        Get distribution of entries by priority ranges.

        Returns:
            Dictionary mapping priority ranges to entry counts
        """
        distribution = defaultdict(int)

        for priority in self._entry_priorities.values():
            if priority >= 0.9:
                distribution["critical (0.9-1.0)"] += 1
            elif priority >= 0.7:
                distribution["high (0.7-0.9)"] += 1
            elif priority >= 0.4:
                distribution["normal (0.4-0.7)"] += 1
            elif priority >= 0.2:
                distribution["low (0.2-0.4)"] += 1
            else:
                distribution["minimal (0.0-0.2)"] += 1

        return dict(distribution)

    async def _perform_eviction(self, cache: 'Cache') -> None:
        """Perform priority-based eviction."""
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

    def get_priority_stats(self) -> Dict[str, Any]:
        """Get priority statistics for this strategy."""
        priorities = list(self._entry_priorities.values())
        access_counts = list(self._access_counts.values())

        return {
            "total_accesses": self._total_accesses,
            "evictions_performed": self._evictions_performed,
            "priority_changes": self._priority_changes,
            "tracked_entries": len(self._entry_priorities),
            "max_size": self.max_size,
            "default_priority": self.default_priority,
            "avg_priority": sum(priorities) / len(priorities) if priorities else 0,
            "max_priority": max(priorities) if priorities else 0,
            "min_priority": min(priorities) if priorities else 0,
            "avg_access_count": sum(access_counts) / len(access_counts) if access_counts else 0,
            "priority_distribution": self.get_priority_distribution()
        }

    def validate_config(self) -> None:
        """Validate strategy configuration."""
        super().validate_config()

        if self.max_size is not None and self.max_size <= 0:
            raise ValueError("max_size must be positive")

        if not (0.0 <= self.default_priority <= 1.0):
            raise ValueError("default_priority must be between 0.0 and 1.0")

        # Validate priority levels
        for name, priority in self.priority_levels.items():
            if not (0.0 <= priority <= 1.0):
                raise ValueError(f"Priority level '{name}' must be between 0.0 and 1.0")

    def get_info(self) -> Dict[str, Any]:
        """Get strategy information with priority-specific details."""
        base_info = super().get_info()
        base_info.update({
            "max_size": self.max_size,
            "default_priority": self.default_priority,
            "priority_levels": self.priority_levels,
            "tracked_entries": len(self._entry_priorities),
            "priority_stats": self.get_priority_stats()
        })
        return base_info

    def __str__(self) -> str:
        return f"Priority(max_size={self.max_size}, tracked={len(self._entry_priorities)})"