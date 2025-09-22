"""
Redis backend implementation.

Redis-based storage backend with full feature support including TTL,
pattern matching, and cluster support.
"""

import json
import fnmatch
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

from omnicache.models.backend import Backend
from omnicache.models.key import Key
from omnicache.models.value import Value, SerializerType
from omnicache.models.entry import CacheEntry
from omnicache.models.result import ClearResult
from omnicache.core.exceptions import CacheBackendError


class RedisBackend(Backend):
    """
    Redis storage backend.

    Provides distributed cache storage with persistence,
    clustering support, and Redis-native operations.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "omnicache:",
        **config: Any
    ) -> None:
        """
        Initialize Redis backend.

        Args:
            url: Redis connection URL
            host: Redis host (overrides URL)
            port: Redis port (overrides URL)
            db: Redis database number
            password: Redis password
            key_prefix: Prefix for all cache keys
            **config: Additional Redis configuration
        """
        if not REDIS_AVAILABLE:
            raise CacheBackendError("Redis not available. Install with: pip install redis")

        # Build connection parameters
        if host and port:
            redis_config = {
                "host": host,
                "port": port,
                "db": db,
                "password": password,
                **config
            }
        else:
            redis_config = {"url": url, **config}

        super().__init__(
            name="redis",
            url=url,
            key_prefix=key_prefix,
            **redis_config
        )

        self.key_prefix = key_prefix
        self._redis: Optional[Redis] = None
        self._redis_config = redis_config

        # Statistics
        self._total_gets = 0
        self._total_sets = 0
        self._total_deletes = 0
        self._total_hits = 0
        self._total_misses = 0

    async def initialize(self) -> None:
        """Initialize the Redis connection."""
        try:
            # Create Redis connection
            if "url" in self._redis_config:
                self._redis = redis.from_url(**self._redis_config)
            else:
                self._redis = Redis(**self._redis_config)

            # Test connection
            await self._redis.ping()

            await super().initialize()
            self._clear_error()

        except Exception as e:
            self._record_error(f"Failed to connect to Redis: {str(e)}")
            raise CacheBackendError(f"Redis backend initialization failed: {str(e)}")

    async def shutdown(self) -> None:
        """Shutdown the Redis connection."""
        try:
            if self._redis:
                await self._redis.close()
                self._redis = None

            await super().shutdown()
            self._clear_error()

        except Exception as e:
            self._record_error(f"Failed to shutdown Redis: {str(e)}")
            raise CacheBackendError(f"Redis backend shutdown failed: {str(e)}")

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
        priority: Optional[float] = None
    ) -> None:
        """Store a cache entry in Redis."""
        try:
            if not self._redis:
                raise CacheBackendError("Redis not initialized")

            self._total_sets += 1

            # Create Key and Value objects
            cache_key = Key(value=key, tags=set(tags) if tags else None)
            cache_value = Value(data=value)

            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=cache_value,
                ttl=ttl,
                priority=priority or 0.5
            )

            # Prepare data for Redis
            redis_key = self._make_redis_key(key)
            entry_data = self._serialize_entry(entry)

            # Store in Redis with TTL
            if ttl:
                await self._redis.setex(redis_key, int(ttl), entry_data)
            else:
                await self._redis.set(redis_key, entry_data)

            # Store tags if present
            if tags:
                await self._store_tags(key, tags)

            self._clear_error()

        except Exception as e:
            self._record_error(f"Failed to set key '{key}': {str(e)}")
            raise CacheBackendError(f"Redis backend set failed: {str(e)}")

    async def get(self, key: str) -> Any:
        """Retrieve a cache entry value from Redis."""
        try:
            if not self._redis:
                raise CacheBackendError("Redis not initialized")

            self._total_gets += 1

            redis_key = self._make_redis_key(key)
            entry_data = await self._redis.get(redis_key)

            if entry_data is None:
                self._total_misses += 1
                return None

            # Deserialize entry
            entry = self._deserialize_entry(entry_data)

            # Check if expired (Redis should handle this, but double-check)
            if entry.is_expired():
                await self.delete(key)
                self._total_misses += 1
                return None

            # Record access
            entry.access()
            self._total_hits += 1
            self._clear_error()

            return entry.value.data

        except Exception as e:
            self._record_error(f"Failed to get key '{key}': {str(e)}")
            raise CacheBackendError(f"Redis backend get failed: {str(e)}")

    async def delete(self, key: str) -> bool:
        """Delete a cache entry from Redis."""
        try:
            if not self._redis:
                raise CacheBackendError("Redis not initialized")

            self._total_deletes += 1

            redis_key = self._make_redis_key(key)
            result = await self._redis.delete(redis_key)

            # Clean up tags
            await self._remove_tags(key)

            self._clear_error()
            return result > 0

        except Exception as e:
            self._record_error(f"Failed to delete key '{key}': {str(e)}")
            raise CacheBackendError(f"Redis backend delete failed: {str(e)}")

    async def get_entry(self, key: str) -> Optional[CacheEntry]:
        """Get complete cache entry with metadata from Redis."""
        try:
            if not self._redis:
                raise CacheBackendError("Redis not initialized")

            redis_key = self._make_redis_key(key)
            entry_data = await self._redis.get(redis_key)

            if entry_data is None:
                return None

            # Deserialize entry
            entry = self._deserialize_entry(entry_data)

            # Check if expired
            if entry.is_expired():
                await self.delete(key)
                return None

            # Record access
            entry.access()
            self._clear_error()

            return entry

        except Exception as e:
            self._record_error(f"Failed to get entry '{key}': {str(e)}")
            raise CacheBackendError(f"Redis backend get_entry failed: {str(e)}")

    async def clear(
        self,
        pattern: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> ClearResult:
        """Clear cache entries from Redis."""
        try:
            if not self._redis:
                raise CacheBackendError("Redis not initialized")

            cleared_keys = []
            errors = []

            if pattern is None and tags is None:
                # Clear all cache keys
                redis_pattern = self._make_redis_key("*")
                keys = await self._redis.keys(redis_pattern)

                if keys:
                    deleted = await self._redis.delete(*keys)
                    cleared_keys = [self._extract_cache_key(k.decode()) for k in keys]
            else:
                # Get all cache keys and filter
                redis_pattern = self._make_redis_key("*")
                redis_keys = await self._redis.keys(redis_pattern)

                keys_to_delete = []

                for redis_key in redis_keys:
                    cache_key = self._extract_cache_key(redis_key.decode())
                    should_clear = True

                    # Check pattern match
                    if pattern and not fnmatch.fnmatch(cache_key, pattern):
                        should_clear = False

                    # Check tag match
                    if tags and should_clear:
                        entry_tags = await self._get_tags(cache_key)
                        if not set(tags).issubset(entry_tags):
                            should_clear = False

                    if should_clear:
                        keys_to_delete.append(redis_key)
                        cleared_keys.append(cache_key)

                # Delete selected keys
                if keys_to_delete:
                    await self._redis.delete(*keys_to_delete)

            # Clean up tag storage
            if tags is None:
                await self._clear_all_tags()

            self._clear_error()

            return ClearResult(
                cleared_count=len(cleared_keys),
                pattern=pattern,
                tags=set(tags) if tags else None,
                error_count=len(errors),
                errors=errors
            )

        except Exception as e:
            self._record_error(f"Failed to clear entries: {str(e)}")
            raise CacheBackendError(f"Redis backend clear failed: {str(e)}")

    async def exists(self, key: str) -> bool:
        """Check if a cache entry exists in Redis."""
        try:
            if not self._redis:
                raise CacheBackendError("Redis not initialized")

            redis_key = self._make_redis_key(key)
            result = await self._redis.exists(redis_key)
            return result > 0

        except Exception as e:
            self._record_error(f"Failed to check existence of key '{key}': {str(e)}")
            raise CacheBackendError(f"Redis backend exists failed: {str(e)}")

    async def get_size(self) -> int:
        """Get the number of cache entries in Redis."""
        try:
            if not self._redis:
                raise CacheBackendError("Redis not initialized")

            redis_pattern = self._make_redis_key("*")
            keys = await self._redis.keys(redis_pattern)
            return len(keys)

        except Exception as e:
            self._record_error(f"Failed to get size: {str(e)}")
            raise CacheBackendError(f"Redis backend get_size failed: {str(e)}")

    async def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes."""
        try:
            if not self._redis:
                raise CacheBackendError("Redis not initialized")

            # Use Redis MEMORY USAGE command if available
            redis_pattern = self._make_redis_key("*")
            keys = await self._redis.keys(redis_pattern)

            total_bytes = 0
            for key in keys:
                try:
                    # Try to get memory usage for each key
                    usage = await self._redis.memory_usage(key)
                    if usage:
                        total_bytes += usage
                except:
                    # Fallback: estimate based on serialized size
                    data = await self._redis.get(key)
                    if data:
                        total_bytes += len(data)

            return total_bytes

        except Exception as e:
            self._record_error(f"Failed to calculate memory usage: {str(e)}")
            raise CacheBackendError(f"Redis backend get_memory_usage failed: {str(e)}")

    async def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get list of cache keys matching optional pattern."""
        try:
            if not self._redis:
                raise CacheBackendError("Redis not initialized")

            if pattern:
                redis_pattern = self._make_redis_key(pattern)
            else:
                redis_pattern = self._make_redis_key("*")

            redis_keys = await self._redis.keys(redis_pattern)
            return [self._extract_cache_key(k.decode()) for k in redis_keys]

        except Exception as e:
            self._record_error(f"Failed to get keys: {str(e)}")
            raise CacheBackendError(f"Redis backend get_keys failed: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get backend statistics."""
        return {
            "total_gets": self._total_gets,
            "total_sets": self._total_sets,
            "total_deletes": self._total_deletes,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": self._total_hits / max(self._total_gets, 1),
            "key_prefix": self.key_prefix,
            "redis_available": REDIS_AVAILABLE
        }

    def _make_redis_key(self, cache_key: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.key_prefix}{cache_key}"

    def _extract_cache_key(self, redis_key: str) -> str:
        """Extract cache key from Redis key."""
        if redis_key.startswith(self.key_prefix):
            return redis_key[len(self.key_prefix):]
        return redis_key

    def _serialize_entry(self, entry: CacheEntry) -> str:
        """Serialize cache entry for Redis storage."""
        # Convert entry to dict and serialize as JSON
        entry_dict = entry.to_dict()
        return json.dumps(entry_dict, default=str)

    def _deserialize_entry(self, data: Union[str, bytes]) -> CacheEntry:
        """Deserialize cache entry from Redis data."""
        if isinstance(data, bytes):
            data = data.decode('utf-8')

        entry_dict = json.loads(data)

        # Reconstruct Key object
        key_data = entry_dict["key"]
        cache_key = Key.from_dict(key_data)

        # Reconstruct Value object
        value_data = entry_dict["value"]
        serializer_type = SerializerType(value_data["serializer_type"])
        cache_value = Value.from_serialized(
            serialized_data=value_data.get("serialized_data", ""),
            serializer_type=serializer_type,
            content_type=value_data.get("content_type"),
            version=value_data.get("version", 1)
        )

        # Create CacheEntry
        entry = CacheEntry(
            key=cache_key,
            value=cache_value,
            ttl=entry_dict.get("ttl_remaining"),
            priority=entry_dict.get("priority", 0.5)
        )

        return entry

    async def _store_tags(self, key: str, tags: List[str]) -> None:
        """Store tag associations for a key."""
        if not self._redis or not tags:
            return

        tag_key = f"{self.key_prefix}tags:{key}"
        await self._redis.sadd(tag_key, *tags)

    async def _get_tags(self, key: str) -> set:
        """Get tags for a key."""
        if not self._redis:
            return set()

        tag_key = f"{self.key_prefix}tags:{key}"
        tags = await self._redis.smembers(tag_key)
        return {tag.decode() for tag in tags}

    async def _remove_tags(self, key: str) -> None:
        """Remove tag associations for a key."""
        if not self._redis:
            return

        tag_key = f"{self.key_prefix}tags:{key}"
        await self._redis.delete(tag_key)

    async def _clear_all_tags(self) -> None:
        """Clear all tag associations."""
        if not self._redis:
            return

        tag_pattern = f"{self.key_prefix}tags:*"
        tag_keys = await self._redis.keys(tag_pattern)
        if tag_keys:
            await self._redis.delete(*tag_keys)

    def __str__(self) -> str:
        return f"Redis({self.config.get('url', 'localhost:6379')})"