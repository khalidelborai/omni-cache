"""
Power caching utilities.

Advanced caching features for high-performance applications:
- Request coalescing (deduplicate concurrent requests)
- Multi-level caching (L1 memory + L2 external)
- Cache versioning (schema migration support)
- Negative caching (cache "not found" results)
- Memoization (function result caching)
- Write-behind (async write-through)
- Prefetching (predictive loading)
"""

import asyncio
import hashlib
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import (
    Any, Awaitable, Callable, Dict, Generic, List, Optional,
    Set, Tuple, TypeVar, Union
)
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Request Coalescing
# =============================================================================

class RequestCoalescer:
    """
    Coalesces concurrent requests for the same key.

    When multiple requests arrive for the same key simultaneously,
    only one actually executes while others wait for the result.
    This prevents thundering herd and reduces backend load.
    """

    def __init__(self):
        self._pending: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], Awaitable[T]]
    ) -> T:
        """
        Get result for key, computing only once for concurrent requests.

        Args:
            key: Unique key for the request
            compute_func: Async function to compute the result

        Returns:
            Computed result (shared among all concurrent requests)
        """
        async with self._lock:
            if key in self._pending:
                # Another request is already computing this
                future = self._pending[key]
            else:
                # We're the first - create future and compute
                future = asyncio.get_event_loop().create_future()
                self._pending[key] = future

                # Start computation in background
                asyncio.create_task(self._compute_and_resolve(key, compute_func, future))

        # Wait for result
        return await future

    async def _compute_and_resolve(
        self,
        key: str,
        compute_func: Callable[[], Awaitable[T]],
        future: asyncio.Future
    ) -> None:
        """Compute result and resolve the future."""
        try:
            result = await compute_func()
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            async with self._lock:
                self._pending.pop(key, None)


# Global coalescer
request_coalescer = RequestCoalescer()


def coalesce_requests(key_func: Optional[Callable[..., str]] = None):
    """
    Decorator to coalesce concurrent requests.

    Args:
        key_func: Function to generate unique key from arguments

    Example:
        @coalesce_requests(key_func=lambda user_id: f"user:{user_id}")
        async def fetch_user(user_id: int):
            # Only one concurrent call per user_id
            return await db.get_user(user_id)
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")
                key = ":".join(key_parts)

            async def compute():
                return await func(*args, **kwargs)

            return await request_coalescer.get_or_compute(key, compute)

        return wrapper
    return decorator


# =============================================================================
# Multi-Level Caching
# =============================================================================

@dataclass
class CacheLevel:
    """Configuration for a cache level."""
    name: str
    cache: Any  # Cache instance
    ttl: Optional[float] = None
    write_through: bool = True  # Write to this level on set
    read_through: bool = True   # Try to read from this level


class MultiLevelCache:
    """
    Multi-level cache with automatic promotion/demotion.

    Typically used as L1 (memory) + L2 (Redis/disk) configuration.
    Reads check L1 first, then L2. Writes go to both levels.
    Cache misses in L1 that hit L2 promote the value to L1.
    """

    def __init__(self, levels: List[CacheLevel]):
        if not levels:
            raise ValueError("At least one cache level required")
        self.levels = levels
        self._stats = {
            level.name: {"hits": 0, "misses": 0, "promotions": 0}
            for level in levels
        }

    async def get(self, key: str, promote: bool = True) -> Optional[Any]:
        """
        Get value from cache, checking levels in order.

        Args:
            key: Cache key
            promote: If True, promote to higher levels on hit

        Returns:
            Cached value or None
        """
        hit_level_idx = None
        value = None

        # Check each level in order
        for idx, level in enumerate(self.levels):
            if not level.read_through:
                continue

            try:
                value = await level.cache.get(key)
                if value is not None:
                    self._stats[level.name]["hits"] += 1
                    hit_level_idx = idx
                    break
                else:
                    self._stats[level.name]["misses"] += 1
            except Exception as e:
                logger.warning(f"Error reading from {level.name}: {e}")
                continue

        # Promote to higher levels if found in lower level
        if value is not None and hit_level_idx is not None and promote:
            for idx in range(hit_level_idx):
                level = self.levels[idx]
                if level.write_through:
                    try:
                        if level.ttl:
                            await level.cache.set(key, value, ttl=level.ttl)
                        else:
                            await level.cache.set(key, value)
                        self._stats[level.name]["promotions"] += 1
                    except Exception as e:
                        logger.warning(f"Error promoting to {level.name}: {e}")

        return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        levels: Optional[List[str]] = None
    ) -> None:
        """
        Set value in cache levels.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL (overrides level-specific TTL)
            levels: Specific levels to write to (None = all write_through levels)
        """
        for level in self.levels:
            if not level.write_through:
                continue

            if levels and level.name not in levels:
                continue

            try:
                effective_ttl = ttl or level.ttl
                if effective_ttl:
                    await level.cache.set(key, value, ttl=effective_ttl)
                else:
                    await level.cache.set(key, value)
            except Exception as e:
                logger.warning(f"Error writing to {level.name}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete from all cache levels."""
        deleted = False
        for level in self.levels:
            try:
                if await level.cache.delete(key):
                    deleted = True
            except Exception as e:
                logger.warning(f"Error deleting from {level.name}: {e}")
        return deleted

    async def invalidate_all(self) -> None:
        """Clear all cache levels."""
        for level in self.levels:
            try:
                if hasattr(level.cache, 'clear'):
                    await level.cache.clear()
            except Exception as e:
                logger.warning(f"Error clearing {level.name}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all levels."""
        return {
            "levels": self._stats,
            "summary": {
                "total_hits": sum(s["hits"] for s in self._stats.values()),
                "total_misses": sum(s["misses"] for s in self._stats.values()),
                "total_promotions": sum(s["promotions"] for s in self._stats.values()),
            }
        }


def multilevel_cache(
    l1_cache_name: str = "l1_memory",
    l2_cache_name: str = "l2_redis",
    l1_ttl: float = 60,
    l2_ttl: float = 3600,
    key_func: Optional[Callable[..., str]] = None
):
    """
    Decorator for multi-level caching.

    Args:
        l1_cache_name: Name of L1 (fast) cache
        l2_cache_name: Name of L2 (persistent) cache
        l1_ttl: TTL for L1 cache
        l2_ttl: TTL for L2 cache
        key_func: Custom key generation function

    Example:
        @multilevel_cache(l1_ttl=30, l2_ttl=300)
        async def get_user(user_id: int):
            return await db.fetch_user(user_id)
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        _ml_cache: Optional[MultiLevelCache] = None

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            nonlocal _ml_cache

            from omnicache.core.manager import manager

            # Initialize multi-level cache on first call
            if _ml_cache is None:
                l1 = await manager.get_cache(l1_cache_name, auto_create=True)
                l2 = await manager.get_cache(l2_cache_name, auto_create=True)
                _ml_cache = MultiLevelCache([
                    CacheLevel(name="L1", cache=l1, ttl=l1_ttl),
                    CacheLevel(name="L2", cache=l2, ttl=l2_ttl),
                ])

            # Generate key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")
                cache_key = ":".join(key_parts)

            # Try cache
            cached = await _ml_cache.get(cache_key)
            if cached is not None:
                return cached

            # Execute and cache
            result = await func(*args, **kwargs)
            await _ml_cache.set(cache_key, result)

            return result

        wrapper._ml_cache = lambda: _ml_cache
        return wrapper

    return decorator


# =============================================================================
# Cache Versioning
# =============================================================================

@dataclass
class VersionedValue:
    """Wrapper for versioned cache values."""
    value: Any
    version: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VersionedCache:
    """
    Cache wrapper that supports versioning.

    When the version changes, old cached values are automatically
    invalidated. Useful for schema migrations and breaking changes.
    """

    VERSION_PREFIX = "__v:"

    def __init__(self, cache: Any, version: str = "1.0"):
        self._cache = cache
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    def set_version(self, new_version: str) -> None:
        """Update cache version (invalidates all old entries on read)."""
        self._version = new_version

    def _version_key(self, key: str) -> str:
        """Generate versioned key."""
        return f"{self.VERSION_PREFIX}{self._version}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value if version matches."""
        versioned_key = self._version_key(key)
        data = await self._cache.get(versioned_key)

        if data is None:
            return None

        # Handle VersionedValue wrapper
        if isinstance(data, dict) and "__versioned__" in data:
            if data.get("version") != self._version:
                # Version mismatch - treat as miss
                return None
            return data.get("value")

        return data

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set value with current version."""
        versioned_key = self._version_key(key)

        wrapped = {
            "__versioned__": True,
            "value": value,
            "version": self._version,
            "created_at": time.time(),
            "metadata": metadata or {}
        }

        if ttl:
            await self._cache.set(versioned_key, wrapped, ttl=ttl)
        else:
            await self._cache.set(versioned_key, wrapped)

    async def delete(self, key: str) -> bool:
        """Delete versioned key."""
        versioned_key = self._version_key(key)
        return await self._cache.delete(versioned_key)

    def __getattr__(self, name: str) -> Any:
        """Proxy other methods to underlying cache."""
        return getattr(self._cache, name)


def versioned_cache(
    cache_name: str = "default",
    version: str = "1.0",
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None
):
    """
    Decorator for versioned caching.

    When you change the version, old cached values are ignored.

    Args:
        cache_name: Name of cache to use
        version: Cache version (change to invalidate old data)
        ttl: Time to live
        key_func: Custom key generation function

    Example:
        @versioned_cache(version="2.0")  # Change version to invalidate
        async def get_user_v2(user_id: int):
            return await fetch_user_new_format(user_id)
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        _vcache: Optional[VersionedCache] = None

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            nonlocal _vcache

            from omnicache.core.manager import manager

            if _vcache is None:
                cache = await manager.get_cache(cache_name, auto_create=True)
                _vcache = VersionedCache(cache, version)

            # Generate key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")
                cache_key = ":".join(key_parts)

            # Try cache
            cached = await _vcache.get(cache_key)
            if cached is not None:
                return cached

            # Execute and cache
            result = await func(*args, **kwargs)
            await _vcache.set(cache_key, result, ttl=ttl)

            return result

        wrapper.set_version = lambda v: setattr(_vcache, '_version', v) if _vcache else None
        return wrapper

    return decorator


# =============================================================================
# Negative Caching
# =============================================================================

class NegativeMarker:
    """Marker for negative cache entries."""
    __slots__ = ('reason', 'timestamp')

    def __init__(self, reason: str = "not_found"):
        self.reason = reason
        self.timestamp = time.time()


class NegativeCache:
    """
    Cache wrapper that supports negative caching.

    Caches "not found" results to prevent repeated lookups
    for non-existent items.
    """

    NEGATIVE_MARKER = "__NEGATIVE__"

    def __init__(
        self,
        cache: Any,
        negative_ttl: float = 60,
        positive_ttl: Optional[float] = None
    ):
        self._cache = cache
        self._negative_ttl = negative_ttl
        self._positive_ttl = positive_ttl

    async def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """
        Get value from cache.

        Returns:
            Tuple of (value, is_negative_cached)
            - (value, False) if positive cache hit
            - (None, True) if negative cache hit
            - (None, False) if cache miss
        """
        data = await self._cache.get(key)

        if data is None:
            return None, False

        if isinstance(data, dict) and data.get("__negative__"):
            return None, True

        return data, False

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Set positive cache value."""
        effective_ttl = ttl or self._positive_ttl
        if effective_ttl:
            await self._cache.set(key, value, ttl=effective_ttl)
        else:
            await self._cache.set(key, value)

    async def set_negative(
        self,
        key: str,
        reason: str = "not_found",
        ttl: Optional[float] = None
    ) -> None:
        """Set negative cache marker."""
        marker = {
            "__negative__": True,
            "reason": reason,
            "timestamp": time.time()
        }
        effective_ttl = ttl or self._negative_ttl
        await self._cache.set(key, marker, ttl=effective_ttl)

    async def delete(self, key: str) -> bool:
        """Delete key (positive or negative)."""
        return await self._cache.delete(key)


def negative_cache(
    cache_name: str = "default",
    positive_ttl: Optional[float] = None,
    negative_ttl: float = 60,
    not_found_value: Any = None,
    key_func: Optional[Callable[..., str]] = None
):
    """
    Decorator for negative caching.

    Caches both found and not-found results to prevent
    repeated lookups for non-existent items.

    Args:
        cache_name: Name of cache to use
        positive_ttl: TTL for found results
        negative_ttl: TTL for not-found results
        not_found_value: Value that indicates "not found"
        key_func: Custom key generation function

    Example:
        @negative_cache(positive_ttl=300, negative_ttl=60, not_found_value=None)
        async def get_user(user_id: int):
            return await db.fetch_user(user_id)  # Returns None if not found
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        _ncache: Optional[NegativeCache] = None

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            nonlocal _ncache

            from omnicache.core.manager import manager

            if _ncache is None:
                cache = await manager.get_cache(cache_name, auto_create=True)
                _ncache = NegativeCache(cache, negative_ttl, positive_ttl)

            # Generate key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")
                cache_key = ":".join(key_parts)

            # Try cache
            value, is_negative = await _ncache.get(cache_key)

            if is_negative:
                # Cached as not-found
                return not_found_value

            if value is not None:
                # Positive cache hit
                return value

            # Cache miss - execute function
            result = await func(*args, **kwargs)

            if result == not_found_value:
                # Cache as negative
                await _ncache.set_negative(cache_key)
            else:
                # Cache as positive
                await _ncache.set(cache_key, result)

            return result

        return wrapper

    return decorator


# =============================================================================
# Memoization
# =============================================================================

class LRUMemoizer(Generic[T]):
    """
    LRU memoization cache.

    In-memory function result cache with LRU eviction.
    """

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Tuple[T, float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[T]:
        """Get memoized value."""
        async with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key][0]
            self._misses += 1
            return None

    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set memoized value."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)

            expires_at = time.time() + ttl if ttl else None
            self._cache[key] = (value, expires_at)

    async def clear(self) -> None:
        """Clear all memoized values."""
        async with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get memoization statistics."""
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        }


def memoize(
    maxsize: int = 128,
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None
):
    """
    In-memory memoization decorator with LRU eviction.

    Args:
        maxsize: Maximum number of cached results
        ttl: Optional TTL for cached results
        key_func: Custom key generation function

    Example:
        @memoize(maxsize=256, ttl=60)
        async def expensive_computation(x: int, y: int):
            return await heavy_calculation(x, y)
    """
    memoizer = LRUMemoizer(maxsize)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")
                key = ":".join(key_parts)

            # Check memoized
            cached = await memoizer.get(key)
            if cached is not None:
                return cached

            # Execute and memoize
            result = await func(*args, **kwargs)
            await memoizer.set(key, result, ttl)

            return result

        wrapper.cache_clear = memoizer.clear
        wrapper.cache_stats = memoizer.stats
        return wrapper

    return decorator


# =============================================================================
# Write-Behind Cache
# =============================================================================

class WriteBehindCache:
    """
    Write-behind (write-back) cache implementation.

    Writes are queued and flushed to the backend asynchronously,
    improving write performance at the cost of potential data loss.
    """

    def __init__(
        self,
        cache: Any,
        backend_writer: Callable[[str, Any], Awaitable[None]],
        flush_interval: float = 1.0,
        batch_size: int = 100
    ):
        self._cache = cache
        self._writer = backend_writer
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._write_queue: Dict[str, Any] = {}
        self._queue_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the write-behind flush task."""
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Stop and flush remaining writes."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush()

    async def get(self, key: str) -> Optional[Any]:
        """Get from cache."""
        return await self._cache.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Set in cache and queue for backend write."""
        # Write to cache immediately
        if ttl:
            await self._cache.set(key, value, ttl=ttl)
        else:
            await self._cache.set(key, value)

        # Queue for backend write
        async with self._queue_lock:
            self._write_queue[key] = value

            # Flush if batch is full
            if len(self._write_queue) >= self._batch_size:
                await self._flush()

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            await asyncio.sleep(self._flush_interval)
            await self._flush()

    async def _flush(self) -> None:
        """Flush queued writes to backend."""
        async with self._queue_lock:
            if not self._write_queue:
                return

            queue = self._write_queue.copy()
            self._write_queue.clear()

        # Write to backend
        for key, value in queue.items():
            try:
                await self._writer(key, value)
            except Exception as e:
                logger.error(f"Write-behind failed for {key}: {e}")
                # Re-queue failed writes
                async with self._queue_lock:
                    if key not in self._write_queue:
                        self._write_queue[key] = value


# =============================================================================
# Prefetching
# =============================================================================

class Prefetcher:
    """
    Predictive cache prefetching.

    Tracks access patterns and prefetches likely-needed items.
    """

    def __init__(
        self,
        cache: Any,
        loader: Callable[[str], Awaitable[Any]],
        max_predictions: int = 10,
        confidence_threshold: float = 0.3
    ):
        self._cache = cache
        self._loader = loader
        self._max_predictions = max_predictions
        self._confidence_threshold = confidence_threshold

        # Access pattern tracking
        self._sequences: List[List[str]] = []
        self._current_sequence: List[str] = []
        self._co_occurrences: Dict[str, Dict[str, int]] = {}
        self._access_counts: Dict[str, int] = {}

    def record_access(self, key: str) -> None:
        """Record an access for pattern learning."""
        self._access_counts[key] = self._access_counts.get(key, 0) + 1

        # Update current sequence
        self._current_sequence.append(key)
        if len(self._current_sequence) > 10:
            self._current_sequence.pop(0)

        # Update co-occurrences
        for prev_key in self._current_sequence[:-1]:
            if prev_key not in self._co_occurrences:
                self._co_occurrences[prev_key] = {}
            self._co_occurrences[prev_key][key] = \
                self._co_occurrences[prev_key].get(key, 0) + 1

    def predict_next(self, current_key: str) -> List[str]:
        """Predict next likely accessed keys."""
        if current_key not in self._co_occurrences:
            return []

        co_occs = self._co_occurrences[current_key]
        total = sum(co_occs.values())

        predictions = []
        for key, count in co_occs.items():
            confidence = count / total
            if confidence >= self._confidence_threshold:
                predictions.append((key, confidence))

        # Sort by confidence and limit
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in predictions[:self._max_predictions]]

    async def prefetch(self, current_key: str) -> int:
        """Prefetch predicted keys."""
        predictions = self.predict_next(current_key)
        prefetched = 0

        for key in predictions:
            # Check if already cached
            cached = await self._cache.get(key)
            if cached is None:
                try:
                    value = await self._loader(key)
                    if value is not None:
                        await self._cache.set(key, value)
                        prefetched += 1
                except Exception as e:
                    logger.debug(f"Prefetch failed for {key}: {e}")

        return prefetched


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Request Coalescing
    "RequestCoalescer",
    "request_coalescer",
    "coalesce_requests",

    # Multi-Level Caching
    "CacheLevel",
    "MultiLevelCache",
    "multilevel_cache",

    # Versioning
    "VersionedValue",
    "VersionedCache",
    "versioned_cache",

    # Negative Caching
    "NegativeMarker",
    "NegativeCache",
    "negative_cache",

    # Memoization
    "LRUMemoizer",
    "memoize",

    # Write-Behind
    "WriteBehindCache",

    # Prefetching
    "Prefetcher",
]
