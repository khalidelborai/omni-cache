"""
ARC (Adaptive Replacement Cache) Strategy implementation.

This module implements the ARC algorithm, which adaptively balances between
LRU and LFU strategies based on the workload characteristics.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set
from enum import Enum
import time
from collections import OrderedDict


class ARCListType(Enum):
    """Types of ARC lists."""
    T1 = "t1"  # Recent items (LRU ghost)
    T2 = "t2"  # Frequent items (LFU ghost)
    B1 = "b1"  # Ghost list for T1
    B2 = "b2"  # Ghost list for T2


@dataclass
class ARCEntry:
    """Entry in ARC cache."""
    key: str
    value: Any
    access_count: int = 1
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    list_type: ARCListType = ARCListType.T1

    def update_access(self) -> None:
        """Update access information."""
        self.access_count += 1
        self.last_access = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "access_count": self.access_count,
            "last_access": self.last_access,
            "created_at": self.created_at,
            "list_type": self.list_type.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ARCEntry':
        """Create from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            access_count=data.get("access_count", 1),
            last_access=data.get("last_access", time.time()),
            created_at=data.get("created_at", time.time()),
            list_type=ARCListType(data.get("list_type", "t1")),
        )


class ARCStrategy:
    """
    ARC (Adaptive Replacement Cache) Strategy implementation.

    ARC adaptively balances between recency (LRU) and frequency (LFU)
    based on the current workload characteristics. It maintains four lists:
    - T1: Recent items (LRU-like)
    - T2: Frequent items (LFU-like)
    - B1: Ghost list for T1 (tracks recently evicted recency items)
    - B2: Ghost list for T2 (tracks recently evicted frequency items)

    The algorithm adapts the target size of T1 based on ghost list hits
    to optimize for the current access pattern.
    """

    def __init__(
        self,
        name: str = "arc",
        max_size: Optional[int] = None,
        capacity: Optional[int] = None,
        target_t1_size: int = 0,
        adaptation_parameter: float = 1.0
    ):
        """
        Initialize ARC strategy.

        Args:
            name: Strategy name
            max_size: Maximum cache size (legacy parameter)
            capacity: Cache capacity (preferred parameter)
            target_t1_size: Initial target size for T1
            adaptation_parameter: Delta for target size adjustment
        """
        # Handle capacity vs max_size parameter
        if capacity is not None and max_size is not None:
            raise ValueError("Cannot specify both capacity and max_size")
        elif capacity is not None:
            self.max_size = capacity
        elif max_size is not None:
            self.max_size = max_size
        else:
            self.max_size = 1000

        self.name = name
        self.target_t1_size = target_t1_size
        self.adaptation_parameter = adaptation_parameter

        # Cache lists
        self._t1: OrderedDict = OrderedDict()  # Recent items
        self._t2: OrderedDict = OrderedDict()  # Frequent items
        self._b1: OrderedDict = OrderedDict()  # Ghost list for T1
        self._b2: OrderedDict = OrderedDict()  # Ghost list for T2

        # Entry mapping
        self._entries: Dict[str, ARCEntry] = {}

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.adaptations = 0
        self.t1_hits = 0
        self.t2_hits = 0
        self.b1_hits = 0
        self.b2_hits = 0

        # Metadata
        self.created_at = time.time()
        self.updated_at = time.time()

        self._post_init()

    def _post_init(self):
        """Post-initialization setup."""
        if self.max_size <= 0:
            raise ValueError("Max size must be positive")

        # Initialize target T1 size to half of cache size
        if self.target_t1_size == 0:
            self.target_t1_size = self.max_size // 2

        # Validate target size
        if not 0 <= self.target_t1_size <= self.max_size:
            raise ValueError("Target T1 size must be between 0 and max_size")

    @property
    def capacity(self) -> int:
        """Get cache capacity (alias for max_size)."""
        return self.max_size

    @property
    def p(self) -> int:
        """Get adaptive parameter p (alias for target_t1_size)."""
        return self.target_t1_size

    @property
    def current_size(self) -> int:
        """Get current cache size (T1 + T2)."""
        return len(self._t1) + len(self._t2)

    @property
    def t1_size(self) -> int:
        """Get current T1 size."""
        return len(self._t1)

    @property
    def t2_size(self) -> int:
        """Get current T2 size."""
        return len(self._t2)

    @property
    def b1_size(self) -> int:
        """Get current B1 size."""
        return len(self._b1)

    @property
    def b2_size(self) -> int:
        """Get current B2 size."""
        return len(self._b2)

    @property
    def total_size(self) -> int:
        """Get total size (T1 + T2 + B1 + B2)."""
        return self.current_size + self.b1_size + self.b2_size

    @property
    def hit_ratio(self) -> float:
        """Calculate hit ratio."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def is_full(self) -> bool:
        """Check if cache is full."""
        return self.current_size >= self.max_size

    def get(self, key: str) -> Optional[Any]:
        """
        Get value for key.

        Args:
            key: Cache key

        Returns:
            Value if found, None otherwise
        """
        # Check T1 (recent items)
        if key in self._t1:
            entry = self._entries[key]
            entry.update_access()

            # Move from T1 to T2 (promote to frequent)
            self._t1.pop(key)
            self._t2[key] = entry
            entry.list_type = ARCListType.T2

            self.hits += 1
            self.t1_hits += 1
            self.updated_at = time.time()
            return entry.value

        # Check T2 (frequent items)
        if key in self._t2:
            entry = self._entries[key]
            entry.update_access()

            # Move to end (most recently used in T2)
            self._t2.move_to_end(key)

            self.hits += 1
            self.t2_hits += 1
            self.updated_at = time.time()
            return entry.value

        # Check B1 (ghost list for T1)
        if key in self._b1:
            self.b1_hits += 1
            self._adapt_on_b1_hit()
            # Remove from B1 (will be added to cache below)
            self._b1.pop(key)

        # Check B2 (ghost list for T2)
        elif key in self._b2:
            self.b2_hits += 1
            self._adapt_on_b2_hit()
            # Remove from B2 (will be added to cache below)
            self._b2.pop(key)

        # Cache miss
        self.misses += 1
        self.updated_at = time.time()
        return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value for key.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live (not used in ARC)
        """
        # If already in cache, update value
        if key in self._entries:
            entry = self._entries[key]
            entry.value = value
            entry.update_access()

            # Move to appropriate position
            if key in self._t1:
                self._t1.move_to_end(key)
            elif key in self._t2:
                self._t2.move_to_end(key)

            self.updated_at = time.time()
            return

        # Create new entry
        entry = ARCEntry(key=key, value=value)
        self._entries[key] = entry

        # Check if we need to make space
        if self.current_size >= self.max_size:
            self._replace()

        # Determine placement based on ghost list history
        if key in self._b1:
            # Was in B1, add to T2 (frequent)
            self._b1.pop(key)
            self._t2[key] = entry
            entry.list_type = ARCListType.T2
        elif key in self._b2:
            # Was in B2, add to T2 (frequent)
            self._b2.pop(key)
            self._t2[key] = entry
            entry.list_type = ARCListType.T2
        else:
            # New item, add to T1 (recent)
            self._t1[key] = entry
            entry.list_type = ARCListType.T1

        self.updated_at = time.time()

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        if key not in self._entries:
            return False

        entry = self._entries.pop(key)

        # Remove from appropriate list
        if key in self._t1:
            self._t1.pop(key)
        elif key in self._t2:
            self._t2.pop(key)
        elif key in self._b1:
            self._b1.pop(key)
        elif key in self._b2:
            self._b2.pop(key)

        self.updated_at = time.time()
        return True

    def _replace(self) -> None:
        """Replace an item to make space."""
        # Case 1: T1 is not empty and either T1 size > target or (T1 size = target and key in B2)
        if (self.t1_size > 0 and
            (self.t1_size > self.target_t1_size or
             (self.t1_size == self.target_t1_size and self.b2_size > 0))):

            # Evict LRU from T1 to B1
            lru_key, entry = self._t1.popitem(last=False)
            self._b1[lru_key] = entry
            entry.list_type = ARCListType.B1

        else:
            # Evict LRU from T2 to B2
            if self.t2_size > 0:
                lru_key, entry = self._t2.popitem(last=False)
                self._b2[lru_key] = entry
                entry.list_type = ARCListType.B2

        self.evictions += 1

        # Maintain ghost list sizes
        self._maintain_ghost_lists()

    def _adapt_on_b1_hit(self) -> None:
        """Adapt target T1 size on B1 hit."""
        # Increase target T1 size (favor recency)
        delta = max(1, self.b2_size // self.b1_size) if self.b1_size > 0 else 1
        self.target_t1_size = min(self.max_size, self.target_t1_size + delta)
        self.adaptations += 1

    def _adapt_on_b2_hit(self) -> None:
        """Adapt target T1 size on B2 hit."""
        # Decrease target T1 size (favor frequency)
        delta = max(1, self.b1_size // self.b2_size) if self.b2_size > 0 else 1
        self.target_t1_size = max(0, self.target_t1_size - delta)
        self.adaptations += 1

    def _maintain_ghost_lists(self) -> None:
        """Maintain ghost list sizes."""
        # Limit B1 size
        while self.b1_size > self.max_size:
            oldest_key = next(iter(self._b1))
            self._b1.pop(oldest_key)
            if oldest_key in self._entries:
                del self._entries[oldest_key]

        # Limit B2 size
        while self.b2_size > self.max_size:
            oldest_key = next(iter(self._b2))
            self._b2.pop(oldest_key)
            if oldest_key in self._entries:
                del self._entries[oldest_key]

        # Limit total size of ghost lists
        max_ghost_size = 2 * self.max_size
        while self.b1_size + self.b2_size > max_ghost_size:
            if self.b1_size > self.b2_size and self.b1_size > 0:
                oldest_key = next(iter(self._b1))
                self._b1.pop(oldest_key)
                if oldest_key in self._entries:
                    del self._entries[oldest_key]
            elif self.b2_size > 0:
                oldest_key = next(iter(self._b2))
                self._b2.pop(oldest_key)
                if oldest_key in self._entries:
                    del self._entries[oldest_key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._t1.clear()
        self._t2.clear()
        self._b1.clear()
        self._b2.clear()
        self._entries.clear()

        # Reset adaptive parameter to 0 as expected by contract tests
        self.target_t1_size = 0

        # Reset statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.adaptations = 0
        self.t1_hits = 0
        self.t2_hits = 0
        self.b1_hits = 0
        self.b2_hits = 0

        self.updated_at = time.time()

    def reset(self) -> None:
        """Reset cache (alias for clear)."""
        self.clear()

    def keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self._t1.keys()) + list(self._t2.keys())

    def items(self) -> List[tuple]:
        """Get all cache items."""
        items = []
        for key in self._t1:
            items.append((key, self._entries[key].value))
        for key in self._t2:
            items.append((key, self._entries[key].value))
        return items

    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an entry."""
        if key not in self._entries:
            return None

        entry = self._entries[key]
        info = entry.to_dict()

        # Add list location
        if key in self._t1:
            info["current_list"] = "T1"
            info["position"] = list(self._t1.keys()).index(key)
        elif key in self._t2:
            info["current_list"] = "T2"
            info["position"] = list(self._t2.keys()).index(key)
        elif key in self._b1:
            info["current_list"] = "B1"
            info["position"] = list(self._b1.keys()).index(key)
        elif key in self._b2:
            info["current_list"] = "B2"
            info["position"] = list(self._b2.keys()).index(key)

        return info

    def get_list_contents(self) -> Dict[str, List[str]]:
        """Get contents of all lists."""
        return {
            "T1": list(self._t1.keys()),
            "T2": list(self._t2.keys()),
            "B1": list(self._b1.keys()),
            "B2": list(self._b2.keys()),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "name": self.name,
            "max_size": self.max_size,
            "current_size": self.current_size,
            "target_t1_size": self.target_t1_size,
            "adaptation_parameter": self.adaptation_parameter,

            # List sizes
            "t1_size": self.t1_size,
            "t2_size": self.t2_size,
            "b1_size": self.b1_size,
            "b2_size": self.b2_size,
            "total_size": self.total_size,

            # Hit statistics
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hit_ratio,
            "t1_hits": self.t1_hits,
            "t2_hits": self.t2_hits,
            "b1_hits": self.b1_hits,
            "b2_hits": self.b2_hits,

            # Other statistics
            "evictions": self.evictions,
            "adaptations": self.adaptations,
            "is_full": self.is_full,

            # Ratios
            "t1_ratio": self.t1_size / self.max_size if self.max_size > 0 else 0,
            "t2_ratio": self.t2_size / self.max_size if self.max_size > 0 else 0,
            "target_t1_ratio": self.target_t1_size / self.max_size if self.max_size > 0 else 0,

            # Timestamps
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def get_efficiency_metrics(self) -> Dict[str, Any]:
        """Get efficiency metrics for the ARC algorithm."""
        total_cache_hits = self.t1_hits + self.t2_hits
        total_ghost_hits = self.b1_hits + self.b2_hits
        total_accesses = self.hits + self.misses

        return {
            "cache_hit_ratio": total_cache_hits / total_accesses if total_accesses > 0 else 0,
            "ghost_hit_ratio": total_ghost_hits / total_accesses if total_accesses > 0 else 0,
            "recency_hit_ratio": self.t1_hits / total_cache_hits if total_cache_hits > 0 else 0,
            "frequency_hit_ratio": self.t2_hits / total_cache_hits if total_cache_hits > 0 else 0,
            "adaptation_rate": self.adaptations / total_accesses if total_accesses > 0 else 0,
            "eviction_rate": self.evictions / total_accesses if total_accesses > 0 else 0,
            "target_accuracy": abs(self.t1_size - self.target_t1_size) / self.max_size if self.max_size > 0 else 0,
        }

    def optimize_parameters(self) -> None:
        """Optimize ARC parameters based on access patterns."""
        metrics = self.get_efficiency_metrics()

        # Adjust adaptation parameter based on hit patterns
        if metrics["recency_hit_ratio"] > 0.7:
            # Favor recency, increase adaptation sensitivity
            self.adaptation_parameter = min(2.0, self.adaptation_parameter * 1.1)
        elif metrics["frequency_hit_ratio"] > 0.7:
            # Favor frequency, decrease adaptation sensitivity
            self.adaptation_parameter = max(0.5, self.adaptation_parameter * 0.9)

        # Reset target if severely misaligned
        if metrics["target_accuracy"] > 0.5 and self.adaptations > 100:
            self.target_t1_size = self.max_size // 2
            self.adaptations = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary representation."""
        return {
            "name": self.name,
            "type": "arc",
            "max_size": self.max_size,
            "target_t1_size": self.target_t1_size,
            "adaptation_parameter": self.adaptation_parameter,
            "statistics": self.get_statistics(),
            "efficiency_metrics": self.get_efficiency_metrics(),
            "list_contents": self.get_list_contents(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ARCStrategy':
        """Create strategy from dictionary representation."""
        strategy = cls(
            name=data.get("name", "arc"),
            max_size=data.get("max_size", 1000),
            target_t1_size=data.get("target_t1_size", 0),
            adaptation_parameter=data.get("adaptation_parameter", 1.0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )

        # Restore statistics if available
        stats = data.get("statistics", {})
        strategy.hits = stats.get("hits", 0)
        strategy.misses = stats.get("misses", 0)
        strategy.evictions = stats.get("evictions", 0)
        strategy.adaptations = stats.get("adaptations", 0)
        strategy.t1_hits = stats.get("t1_hits", 0)
        strategy.t2_hits = stats.get("t2_hits", 0)
        strategy.b1_hits = stats.get("b1_hits", 0)
        strategy.b2_hits = stats.get("b2_hits", 0)

        return strategy

    def __str__(self) -> str:
        """String representation of the strategy."""
        return (f"ARCStrategy(size={self.current_size}/{self.max_size}, "
                f"T1={self.t1_size}, T2={self.t2_size}, target_T1={self.target_t1_size})")

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"ARCStrategy(name='{self.name}', max_size={self.max_size}, "
                f"current_size={self.current_size}, hit_ratio={self.hit_ratio:.3f})")

    def __eq__(self, other) -> bool:
        """Check equality based on name and configuration."""
        if not isinstance(other, ARCStrategy):
            return False
        return (self.name == other.name and
                self.max_size == other.max_size and
                self.target_t1_size == other.target_t1_size)

    def __hash__(self) -> int:
        """Hash based on name and configuration."""
        return hash((self.name, self.max_size, self.target_t1_size))

    def on_access(self, entry: 'CacheEntry') -> Optional['CacheEntry']:
        """
        Handle cache entry access according to ARC algorithm.

        This is the main method that implements the ARC algorithm logic.
        It handles cache hits, misses, and evictions while adapting the
        cache partitioning based on access patterns.

        Args:
            entry: The cache entry being accessed

        Returns:
            Entry to be evicted if cache is full, None otherwise
        """
        from omnicache.models.entry import CacheEntry

        key = entry.key.full_key if hasattr(entry.key, 'full_key') else str(entry.key)

        # Case 1: Cache hit in T1 (recent cache)
        if key in self._t1:
            # Move from T1 to T2 (promote to frequent)
            existing_entry = self._entries[key]
            existing_entry.update_access()  # Update access stats

            self._t1.pop(key)
            self._t2[key] = existing_entry
            existing_entry.list_type = ARCListType.T2

            self.hits += 1
            self.t1_hits += 1
            self.updated_at = time.time()
            return None

        # Case 2: Cache hit in T2 (frequent cache)
        if key in self._t2:
            # Move to end (most recently used in T2)
            existing_entry = self._entries[key]
            existing_entry.update_access()  # Update access stats

            self._t2.move_to_end(key)

            self.hits += 1
            self.t2_hits += 1
            self.updated_at = time.time()
            return None

        # Case 3: Hit in B1 (ghost list for T1) - adapt and cache
        if key in self._b1:
            self.b1_hits += 1
            self._adapt_on_b1_hit()

            # Remove from B1
            self._b1.pop(key)

            # Add to cache (will go to T2 since it was in ghost list)
            evicted_entry = self._add_to_cache(entry, ARCListType.T2)
            return evicted_entry

        # Case 4: Hit in B2 (ghost list for T2) - adapt and cache
        if key in self._b2:
            self.b2_hits += 1
            self._adapt_on_b2_hit()

            # Remove from B2
            self._b2.pop(key)

            # Add to cache (will go to T2 since it was in ghost list)
            evicted_entry = self._add_to_cache(entry, ARCListType.T2)
            return evicted_entry

        # Case 5: Complete miss - new entry
        self.misses += 1
        self.updated_at = time.time()

        # Add to T1 (recent cache) for new entries
        evicted_entry = self._add_to_cache(entry, ARCListType.T1)
        return evicted_entry

    def _add_to_cache(self, entry: 'CacheEntry', target_list: ARCListType) -> Optional['CacheEntry']:
        """
        Add entry to cache, handling eviction if necessary.

        Args:
            entry: Entry to add
            target_list: Which list to add to (T1 or T2)

        Returns:
            Evicted entry if cache was full, None otherwise
        """
        key = entry.key.full_key if hasattr(entry.key, 'full_key') else str(entry.key)
        evicted_entry = None

        # Convert CacheEntry to ARCEntry for internal storage
        arc_entry = ARCEntry(
            key=key,
            value=entry.value.data if hasattr(entry.value, 'data') else entry.value,
            list_type=target_list
        )

        self._entries[key] = arc_entry

        # Check if cache is full and eviction is needed
        if self.current_size >= self.max_size:
            evicted_entry = self._evict_for_space()

        # Add to appropriate list
        if target_list == ARCListType.T1:
            self._t1[key] = arc_entry
        else:  # T2
            self._t2[key] = arc_entry

        arc_entry.list_type = target_list
        return evicted_entry

    def _evict_for_space(self) -> Optional['CacheEntry']:
        """
        Evict an entry to make space according to ARC replacement policy.

        Returns:
            Evicted entry if any, None otherwise
        """
        from omnicache.models.entry import CacheEntry
        from omnicache.models.key import Key
        from omnicache.models.value import Value

        evicted_entry = None

        # Case 1: T1 is not empty and should be evicted from
        if (self.t1_size > 0 and
            (self.t1_size > self.target_t1_size or
             (self.t1_size == self.target_t1_size and self.b2_size > 0))):

            # Evict LRU from T1 to B1
            lru_key, arc_entry = self._t1.popitem(last=False)
            self._b1[lru_key] = arc_entry
            arc_entry.list_type = ARCListType.B1

            # Create CacheEntry to return
            key_obj = Key(value=lru_key)
            value_obj = Value(data=arc_entry.value)
            evicted_entry = CacheEntry(key=key_obj, value=value_obj)
            evicted_entry.mark_evicted()

        else:
            # Evict LRU from T2 to B2
            if self.t2_size > 0:
                lru_key, arc_entry = self._t2.popitem(last=False)
                self._b2[lru_key] = arc_entry
                arc_entry.list_type = ARCListType.B2

                # Create CacheEntry to return
                key_obj = Key(value=lru_key)
                value_obj = Value(data=arc_entry.value)
                evicted_entry = CacheEntry(key=key_obj, value=value_obj)
                evicted_entry.mark_evicted()

        self.evictions += 1

        # Maintain ghost list sizes
        self._maintain_ghost_lists()

        return evicted_entry

    @property
    def b1(self) -> Dict[str, Any]:
        """Get B1 ghost list contents (for contract test compatibility)."""
        return dict(self._b1)

    @property
    def b2(self) -> Dict[str, Any]:
        """Get B2 ghost list contents (for contract test compatibility)."""
        return dict(self._b2)

    @property
    def t1(self) -> Dict[str, Any]:
        """Get T1 cache contents (for contract test compatibility)."""
        return dict(self._t1)

    @property
    def t2(self) -> Dict[str, Any]:
        """Get T2 cache contents (for contract test compatibility)."""
        return dict(self._t2)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics (for contract test compatibility)."""
        return self.get_statistics()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics (for contract test compatibility)."""
        return self.get_statistics()

    async def initialize(self, cache: 'Cache') -> None:
        """Initialize strategy with cache context."""
        # Initialize base strategy if it exists
        if hasattr(super(), 'initialize'):
            await super().initialize(cache)

        # ARC strategy is self-contained and doesn't need special initialization
        pass

    async def shutdown(self) -> None:
        """Cleanup strategy resources."""
        # Clear all internal data structures
        self._t1.clear()
        self._t2.clear()
        self._b1.clear()
        self._b2.clear()
        self._entries.clear()

        # Call parent shutdown if it exists
        if hasattr(super(), 'shutdown'):
            await super().shutdown()