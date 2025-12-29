"""
Advanced caching patterns and utilities.

Provides sophisticated caching patterns including:
- Tag-based invalidation
- Stale-while-revalidate (background refresh)
- Cache stampede prevention (distributed locking)
- Batch operations
- Event hooks
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import (
    Any, Callable, Dict, List, Optional, Set, TypeVar, Union,
    Awaitable, Tuple
)
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Cache Events
# =============================================================================

class CacheEventType(Enum):
    """Types of cache events."""
    HIT = "hit"
    MISS = "miss"
    SET = "set"
    DELETE = "delete"
    EVICT = "evict"
    EXPIRE = "expire"
    REFRESH = "refresh"
    ERROR = "error"


@dataclass
class CacheEvent:
    """Represents a cache event."""
    event_type: CacheEventType
    cache_name: str
    key: str
    timestamp: datetime = field(default_factory=datetime.now)
    value: Any = None
    ttl: Optional[float] = None
    tags: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheEventEmitter:
    """
    Event emitter for cache operations.

    Allows registering callbacks for cache events.
    """

    def __init__(self):
        self._listeners: Dict[CacheEventType, List[Callable]] = {
            event_type: [] for event_type in CacheEventType
        }
        self._global_listeners: List[Callable] = []

    def on(self, event_type: CacheEventType, callback: Callable) -> None:
        """Register a callback for a specific event type."""
        self._listeners[event_type].append(callback)

    def on_hit(self, callback: Callable) -> None:
        """Register callback for cache hits."""
        self.on(CacheEventType.HIT, callback)

    def on_miss(self, callback: Callable) -> None:
        """Register callback for cache misses."""
        self.on(CacheEventType.MISS, callback)

    def on_set(self, callback: Callable) -> None:
        """Register callback for cache sets."""
        self.on(CacheEventType.SET, callback)

    def on_evict(self, callback: Callable) -> None:
        """Register callback for cache evictions."""
        self.on(CacheEventType.EVICT, callback)

    def on_any(self, callback: Callable) -> None:
        """Register callback for all events."""
        self._global_listeners.append(callback)

    async def emit(self, event: CacheEvent) -> None:
        """Emit an event to all registered listeners."""
        # Call specific listeners
        for callback in self._listeners[event.event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

        # Call global listeners
        for callback in self._global_listeners:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Global event callback error: {e}")

    def remove_listener(self, event_type: CacheEventType, callback: Callable) -> bool:
        """Remove a specific listener."""
        try:
            self._listeners[event_type].remove(callback)
            return True
        except ValueError:
            return False

    def clear_listeners(self, event_type: Optional[CacheEventType] = None) -> None:
        """Clear listeners for a specific event type or all events."""
        if event_type:
            self._listeners[event_type] = []
        else:
            for et in CacheEventType:
                self._listeners[et] = []
            self._global_listeners = []


# Global event emitter
cache_events = CacheEventEmitter()


# =============================================================================
# Tag-Based Invalidation
# =============================================================================

class TagManager:
    """
    Manages cache tags for group invalidation.

    Tags allow invalidating multiple cache entries at once
    by associating them with common tags.
    """

    def __init__(self):
        self._tag_to_keys: Dict[str, Set[str]] = {}
        self._key_to_tags: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def add_tags(self, cache_name: str, key: str, tags: List[str]) -> None:
        """Associate tags with a cache key."""
        full_key = f"{cache_name}:{key}"

        async with self._lock:
            # Add to key -> tags mapping
            if full_key not in self._key_to_tags:
                self._key_to_tags[full_key] = set()
            self._key_to_tags[full_key].update(tags)

            # Add to tag -> keys mapping
            for tag in tags:
                if tag not in self._tag_to_keys:
                    self._tag_to_keys[tag] = set()
                self._tag_to_keys[tag].add(full_key)

    async def get_keys_by_tag(self, tag: str) -> Set[str]:
        """Get all keys associated with a tag."""
        async with self._lock:
            return self._tag_to_keys.get(tag, set()).copy()

    async def get_tags_for_key(self, cache_name: str, key: str) -> Set[str]:
        """Get all tags for a specific key."""
        full_key = f"{cache_name}:{key}"
        async with self._lock:
            return self._key_to_tags.get(full_key, set()).copy()

    async def remove_key(self, cache_name: str, key: str) -> None:
        """Remove a key from all tag associations."""
        full_key = f"{cache_name}:{key}"

        async with self._lock:
            if full_key in self._key_to_tags:
                tags = self._key_to_tags.pop(full_key)
                for tag in tags:
                    if tag in self._tag_to_keys:
                        self._tag_to_keys[tag].discard(full_key)
                        if not self._tag_to_keys[tag]:
                            del self._tag_to_keys[tag]

    async def invalidate_tag(self, tag: str) -> List[str]:
        """
        Get all keys to invalidate for a tag and clean up.

        Returns list of full keys (cache_name:key format).
        """
        async with self._lock:
            keys = list(self._tag_to_keys.get(tag, set()))

            # Clean up
            if tag in self._tag_to_keys:
                for full_key in self._tag_to_keys[tag]:
                    if full_key in self._key_to_tags:
                        self._key_to_tags[full_key].discard(tag)
                        if not self._key_to_tags[full_key]:
                            del self._key_to_tags[full_key]
                del self._tag_to_keys[tag]

            return keys

    def get_stats(self) -> Dict[str, Any]:
        """Get tag manager statistics."""
        return {
            "total_tags": len(self._tag_to_keys),
            "total_tagged_keys": len(self._key_to_tags),
            "tags": {tag: len(keys) for tag, keys in self._tag_to_keys.items()}
        }


# Global tag manager
tag_manager = TagManager()


# =============================================================================
# Cache Stampede Prevention
# =============================================================================

class CacheLock:
    """
    Distributed lock for cache stampede prevention.

    Prevents multiple concurrent requests from regenerating
    the same cache entry simultaneously.
    """

    def __init__(self, default_timeout: float = 30.0):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock_times: Dict[str, float] = {}
        self._default_timeout = default_timeout
        self._manager_lock = asyncio.Lock()

    async def acquire(self, key: str, timeout: Optional[float] = None) -> bool:
        """
        Acquire a lock for a cache key.

        Returns True if lock was acquired, False if timed out.
        """
        timeout = timeout or self._default_timeout

        async with self._manager_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()

        lock = self._locks[key]

        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            self._lock_times[key] = time.time()
            return True
        except asyncio.TimeoutError:
            return False

    def release(self, key: str) -> None:
        """Release a lock for a cache key."""
        if key in self._locks and self._locks[key].locked():
            self._locks[key].release()
            self._lock_times.pop(key, None)

    async def is_locked(self, key: str) -> bool:
        """Check if a key is currently locked."""
        return key in self._locks and self._locks[key].locked()

    async def cleanup_stale_locks(self, max_age: float = 60.0) -> int:
        """Clean up locks that have been held too long."""
        current_time = time.time()
        cleaned = 0

        async with self._manager_lock:
            stale_keys = [
                key for key, lock_time in self._lock_times.items()
                if current_time - lock_time > max_age
            ]

            for key in stale_keys:
                if key in self._locks and self._locks[key].locked():
                    self._locks[key].release()
                self._lock_times.pop(key, None)
                cleaned += 1

        return cleaned


# Global cache lock
cache_lock = CacheLock()


# =============================================================================
# Stale-While-Revalidate
# =============================================================================

@dataclass
class SWREntry:
    """Entry with stale-while-revalidate metadata."""
    value: Any
    created_at: float
    stale_at: float
    expires_at: float
    is_refreshing: bool = False


class StaleWhileRevalidate:
    """
    Implements stale-while-revalidate caching pattern.

    Returns stale data immediately while refreshing in the background.
    """

    def __init__(self):
        self._entries: Dict[str, SWREntry] = {}
        self._refresh_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def get(
        self,
        cache: Any,
        key: str,
        loader: Callable[[], Awaitable[T]],
        ttl: float = 300,
        stale_ttl: float = 60
    ) -> T:
        """
        Get value with stale-while-revalidate semantics.

        Args:
            cache: Cache instance to use
            key: Cache key
            loader: Async function to load fresh data
            ttl: Time until data becomes stale
            stale_ttl: Additional time to serve stale data while refreshing

        Returns:
            Cached or fresh value
        """
        current_time = time.time()

        # Try to get from cache
        cached = await cache.get(key)

        if cached is not None:
            # Check if we have SWR metadata
            async with self._lock:
                entry = self._entries.get(key)

            if entry:
                # Data is fresh
                if current_time < entry.stale_at:
                    return cached

                # Data is stale but not expired - return stale and refresh
                if current_time < entry.expires_at:
                    await self._trigger_refresh(cache, key, loader, ttl, stale_ttl)
                    return cached

                # Data is expired - must refresh synchronously
            else:
                # No metadata, treat as fresh
                return cached

        # No cached data or expired - load fresh
        return await self._load_and_cache(cache, key, loader, ttl, stale_ttl)

    async def _trigger_refresh(
        self,
        cache: Any,
        key: str,
        loader: Callable,
        ttl: float,
        stale_ttl: float
    ) -> None:
        """Trigger background refresh if not already refreshing."""
        async with self._lock:
            entry = self._entries.get(key)
            if entry and entry.is_refreshing:
                return

            if entry:
                entry.is_refreshing = True

        # Start background refresh
        async def refresh():
            try:
                await self._load_and_cache(cache, key, loader, ttl, stale_ttl)
            except Exception as e:
                logger.error(f"SWR refresh failed for {key}: {e}")
            finally:
                async with self._lock:
                    if key in self._entries:
                        self._entries[key].is_refreshing = False
                    self._refresh_tasks.pop(key, None)

        task = asyncio.create_task(refresh())
        self._refresh_tasks[key] = task

    async def _load_and_cache(
        self,
        cache: Any,
        key: str,
        loader: Callable,
        ttl: float,
        stale_ttl: float
    ) -> Any:
        """Load fresh data and cache it."""
        value = await loader()
        current_time = time.time()

        # Store in cache
        await cache.set(key, value, ttl=ttl + stale_ttl)

        # Store SWR metadata
        async with self._lock:
            self._entries[key] = SWREntry(
                value=value,
                created_at=current_time,
                stale_at=current_time + ttl,
                expires_at=current_time + ttl + stale_ttl
            )

        return value

    async def invalidate(self, key: str) -> None:
        """Invalidate SWR entry."""
        async with self._lock:
            self._entries.pop(key, None)
            task = self._refresh_tasks.pop(key, None)
            if task and not task.done():
                task.cancel()


# Global SWR manager
swr_manager = StaleWhileRevalidate()


# =============================================================================
# Decorators
# =============================================================================

def cached_with_tags(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    tags: Optional[List[str]] = None,
    tag_func: Optional[Callable[..., List[str]]] = None,
    key_func: Optional[Callable[..., str]] = None
):
    """
    Caching decorator with tag support for group invalidation.

    Args:
        cache_name: Name of cache to use
        ttl: Time to live
        tags: Static list of tags
        tag_func: Function to generate tags dynamically
        key_func: Custom key generation function

    Example:
        @cached_with_tags(
            cache_name="products",
            ttl=3600,
            tags=["products"],
            tag_func=lambda product_id: [f"product:{product_id}"]
        )
        async def get_product(product_id: int):
            return await db.fetch_product(product_id)

        # Invalidate all products
        await invalidate_by_tag("products")

        # Invalidate specific product
        await invalidate_by_tag("product:123")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from omnicache.core.manager import manager

            cache = await manager.get_cache(cache_name, auto_create=True)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")
                cache_key = ":".join(key_parts)

            # Try to get from cache
            cached = await cache.get(cache_key)
            if cached is not None:
                await cache_events.emit(CacheEvent(
                    event_type=CacheEventType.HIT,
                    cache_name=cache_name,
                    key=cache_key
                ))
                return cached

            # Cache miss
            await cache_events.emit(CacheEvent(
                event_type=CacheEventType.MISS,
                cache_name=cache_name,
                key=cache_key
            ))

            # Execute function
            result = await func(*args, **kwargs)

            # Determine tags
            all_tags = list(tags or [])
            if tag_func:
                dynamic_tags = tag_func(*args, **kwargs)
                all_tags.extend(dynamic_tags)

            # Cache result
            if ttl:
                await cache.set(cache_key, result, ttl=ttl)
            else:
                await cache.set(cache_key, result)

            # Register tags
            if all_tags:
                await tag_manager.add_tags(cache_name, cache_key, all_tags)

            await cache_events.emit(CacheEvent(
                event_type=CacheEventType.SET,
                cache_name=cache_name,
                key=cache_key,
                value=result,
                ttl=ttl,
                tags=all_tags
            ))

            return result

        return wrapper
    return decorator


def cached_with_lock(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    lock_timeout: float = 30.0,
    key_func: Optional[Callable[..., str]] = None
):
    """
    Caching decorator with stampede prevention.

    Only one request will compute the value while others wait.

    Args:
        cache_name: Name of cache to use
        ttl: Time to live
        lock_timeout: Maximum time to wait for lock
        key_func: Custom key generation function

    Example:
        @cached_with_lock(cache_name="expensive", ttl=300, lock_timeout=10)
        async def expensive_computation(param: str):
            # Only one concurrent call will execute this
            return await heavy_database_query(param)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from omnicache.core.manager import manager

            cache = await manager.get_cache(cache_name, auto_create=True)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")
                cache_key = ":".join(key_parts)

            lock_key = f"lock:{cache_name}:{cache_key}"

            # Try to get from cache first (no lock needed for reads)
            cached = await cache.get(cache_key)
            if cached is not None:
                return cached

            # Acquire lock for computation
            acquired = await cache_lock.acquire(lock_key, timeout=lock_timeout)

            try:
                # Double-check cache after acquiring lock
                cached = await cache.get(cache_key)
                if cached is not None:
                    return cached

                # Execute function
                result = await func(*args, **kwargs)

                # Cache result
                if ttl:
                    await cache.set(cache_key, result, ttl=ttl)
                else:
                    await cache.set(cache_key, result)

                return result
            finally:
                if acquired:
                    cache_lock.release(lock_key)

        return wrapper
    return decorator


def cached_swr(
    cache_name: str = "default",
    ttl: float = 300,
    stale_ttl: float = 60,
    key_func: Optional[Callable[..., str]] = None
):
    """
    Caching decorator with stale-while-revalidate pattern.

    Returns stale data immediately while refreshing in background.

    Args:
        cache_name: Name of cache to use
        ttl: Time until data becomes stale
        stale_ttl: Additional time to serve stale data
        key_func: Custom key generation function

    Example:
        @cached_swr(cache_name="api", ttl=60, stale_ttl=300)
        async def get_external_api_data():
            # Returns cached data (even if stale) immediately
            # Refreshes in background when stale
            return await external_api.fetch()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from omnicache.core.manager import manager

            cache = await manager.get_cache(cache_name, auto_create=True)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")
                cache_key = ":".join(key_parts)

            # Create loader function
            async def loader():
                return await func(*args, **kwargs)

            return await swr_manager.get(cache, cache_key, loader, ttl, stale_ttl)

        return wrapper
    return decorator


def cached_batch(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    key_prefix: str = "",
    id_param: str = "ids"
):
    """
    Batch caching decorator for bulk operations.

    Fetches cached items individually and only loads missing ones.

    Args:
        cache_name: Name of cache to use
        ttl: Time to live
        key_prefix: Prefix for cache keys
        id_param: Name of the parameter containing IDs

    Example:
        @cached_batch(cache_name="users", ttl=300, key_prefix="user", id_param="user_ids")
        async def get_users(user_ids: List[int]):
            # Only called for IDs not in cache
            return await db.fetch_users(user_ids)

        # First call: fetches all from DB
        users = await get_users([1, 2, 3])

        # Second call: fetches 4, 5 from DB, 1, 2, 3 from cache
        users = await get_users([1, 2, 3, 4, 5])
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from omnicache.core.manager import manager

            cache = await manager.get_cache(cache_name, auto_create=True)

            # Get IDs from kwargs or first positional arg
            ids = kwargs.get(id_param) or (args[0] if args else [])
            if not ids:
                return []

            # Generate cache keys
            prefix = f"{key_prefix}:" if key_prefix else ""
            id_to_key = {id_: f"{prefix}{id_}" for id_ in ids}

            # Fetch cached items
            results = {}
            missing_ids = []

            for id_, key in id_to_key.items():
                cached = await cache.get(key)
                if cached is not None:
                    results[id_] = cached
                else:
                    missing_ids.append(id_)

            # Fetch missing items
            if missing_ids:
                # Call function with only missing IDs
                if id_param in kwargs:
                    kwargs[id_param] = missing_ids
                    fresh_items = await func(*args, **kwargs)
                else:
                    fresh_items = await func(missing_ids, *args[1:], **kwargs)

                # Cache and add to results
                for item in fresh_items:
                    # Assume item has 'id' attribute or is a dict with 'id' key
                    if hasattr(item, 'id'):
                        item_id = item.id
                    elif isinstance(item, dict) and 'id' in item:
                        item_id = item['id']
                    else:
                        continue

                    key = id_to_key.get(item_id, f"{prefix}{item_id}")

                    if ttl:
                        await cache.set(key, item, ttl=ttl)
                    else:
                        await cache.set(key, item)

                    results[item_id] = item

            # Return in original order
            return [results.get(id_) for id_ in ids if id_ in results]

        return wrapper
    return decorator


# =============================================================================
# Invalidation Functions
# =============================================================================

async def invalidate_by_tag(tag: str, cache_name: Optional[str] = None) -> int:
    """
    Invalidate all cache entries with a specific tag.

    Args:
        tag: Tag to invalidate
        cache_name: Optional cache name filter

    Returns:
        Number of entries invalidated
    """
    from omnicache.core.manager import manager

    keys = await tag_manager.invalidate_tag(tag)
    invalidated = 0

    for full_key in keys:
        parts = full_key.split(":", 1)
        if len(parts) == 2:
            key_cache_name, key = parts

            if cache_name and key_cache_name != cache_name:
                continue

            try:
                cache = await manager.get_cache(key_cache_name)
                if cache:
                    await cache.delete(key)
                    invalidated += 1

                    await cache_events.emit(CacheEvent(
                        event_type=CacheEventType.DELETE,
                        cache_name=key_cache_name,
                        key=key,
                        metadata={"reason": "tag_invalidation", "tag": tag}
                    ))
            except Exception as e:
                logger.error(f"Failed to invalidate {full_key}: {e}")

    return invalidated


async def invalidate_by_pattern(
    pattern: str,
    cache_name: str = "default"
) -> int:
    """
    Invalidate cache entries matching a pattern.

    Args:
        pattern: Glob-style pattern (e.g., "user:*", "product:123:*")
        cache_name: Cache to invalidate from

    Returns:
        Number of entries invalidated
    """
    from omnicache.core.manager import manager

    try:
        return await manager.clear_cache(cache_name, pattern=pattern)
    except Exception as e:
        logger.error(f"Failed to invalidate pattern {pattern}: {e}")
        return 0


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Events
    "CacheEventType",
    "CacheEvent",
    "CacheEventEmitter",
    "cache_events",

    # Tags
    "TagManager",
    "tag_manager",

    # Locking
    "CacheLock",
    "cache_lock",

    # SWR
    "StaleWhileRevalidate",
    "swr_manager",

    # Decorators
    "cached_with_tags",
    "cached_with_lock",
    "cached_swr",
    "cached_batch",

    # Invalidation
    "invalidate_by_tag",
    "invalidate_by_pattern",
]
