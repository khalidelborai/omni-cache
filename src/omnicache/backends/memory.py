"""
Memory backend implementation.

In-memory storage backend with TTL support, pattern matching,
and comprehensive entry management.
"""

import asyncio
import fnmatch
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from omnicache.models.backend import Backend, BackendStatus
from omnicache.models.key import Key
from omnicache.models.value import Value
from omnicache.models.entry import CacheEntry
from omnicache.models.result import ClearResult
from omnicache.core.exceptions import CacheBackendError


class MemoryBackend(Backend):
    """
    In-memory storage backend.

    Provides fast, local cache storage with TTL support,
    pattern-based operations, and memory usage tracking.
    """

    def __init__(self, max_size: Optional[int] = None, **config: Any) -> None:
        """
        Initialize memory backend.

        Args:
            max_size: Maximum number of entries (None = unlimited)
            **config: Additional configuration parameters
        """
        super().__init__(name="memory", max_size=max_size, **config)
        self.max_size = max_size

        # Storage
        self._entries: Dict[str, CacheEntry] = {}
        self._expiry_tasks: Dict[str, asyncio.Task] = {}

        # Statistics
        self._total_gets = 0
        self._total_sets = 0
        self._total_deletes = 0
        self._total_hits = 0
        self._total_misses = 0

    async def initialize(self) -> None:
        """Initialize the memory backend."""
        try:
            await super().initialize()
            self._clear_error()
        except Exception as e:
            self._record_error(f"Failed to initialize memory backend: {str(e)}")
            raise CacheBackendError(f"Memory backend initialization failed: {str(e)}")

    async def shutdown(self) -> None:
        """Shutdown the memory backend and cleanup resources."""
        try:
            # Cancel all expiry tasks
            for task in self._expiry_tasks.values():
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if self._expiry_tasks:
                await asyncio.gather(*self._expiry_tasks.values(), return_exceptions=True)

            # Clear storage
            self._entries.clear()
            self._expiry_tasks.clear()

            await super().shutdown()
            self._clear_error()
        except Exception as e:
            self._record_error(f"Failed to shutdown memory backend: {str(e)}")
            raise CacheBackendError(f"Memory backend shutdown failed: {str(e)}")

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
        priority: Optional[float] = None
    ) -> None:
        """Store a cache entry."""
        try:
            self._total_sets += 1

            # Check size limit
            if self.max_size and len(self._entries) >= self.max_size and key not in self._entries:
                raise CacheBackendError(f"Memory backend full (max_size={self.max_size})")

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

            # Cancel existing expiry task if updating
            if key in self._expiry_tasks:
                self._expiry_tasks[key].cancel()
                del self._expiry_tasks[key]

            # Store entry
            self._entries[key] = entry

            # Set up TTL expiry if specified
            if ttl is not None:
                self._expiry_tasks[key] = asyncio.create_task(
                    self._expire_after_ttl(key, ttl)
                )

            self._clear_error()

        except Exception as e:
            self._record_error(f"Failed to set key '{key}': {str(e)}")
            raise CacheBackendError(f"Memory backend set failed: {str(e)}")

    async def get(self, key: str) -> Any:
        """Retrieve a cache entry value."""
        try:
            self._total_gets += 1

            if key not in self._entries:
                self._total_misses += 1
                return None

            entry = self._entries[key]

            # Check if expired
            if entry.is_expired():
                await self._remove_expired_entry(key)
                self._total_misses += 1
                return None

            # Record access
            entry.access()
            self._total_hits += 1
            self._clear_error()

            return entry.value.data

        except Exception as e:
            self._record_error(f"Failed to get key '{key}': {str(e)}")
            raise CacheBackendError(f"Memory backend get failed: {str(e)}")

    async def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        try:
            self._total_deletes += 1

            if key not in self._entries:
                return False

            # Cancel expiry task
            if key in self._expiry_tasks:
                self._expiry_tasks[key].cancel()
                del self._expiry_tasks[key]

            # Remove entry
            del self._entries[key]
            self._clear_error()

            return True

        except Exception as e:
            self._record_error(f"Failed to delete key '{key}': {str(e)}")
            raise CacheBackendError(f"Memory backend delete failed: {str(e)}")

    async def get_entry(self, key: str) -> Optional[CacheEntry]:
        """Get complete cache entry with metadata."""
        try:
            if key not in self._entries:
                return None

            entry = self._entries[key]

            # Check if expired
            if entry.is_expired():
                await self._remove_expired_entry(key)
                return None

            # Record access
            entry.access()
            self._clear_error()

            return entry

        except Exception as e:
            self._record_error(f"Failed to get entry '{key}': {str(e)}")
            raise CacheBackendError(f"Memory backend get_entry failed: {str(e)}")

    async def clear(
        self,
        pattern: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> ClearResult:
        """Clear cache entries."""
        try:
            cleared_keys = []
            errors = []

            # Get keys to clear
            keys_to_clear = []

            for key, entry in self._entries.items():
                should_clear = True

                # Check pattern match
                if pattern and not fnmatch.fnmatch(key, pattern):
                    should_clear = False

                # Check tag match
                if tags and not entry.key.matches_tags(set(tags)):
                    should_clear = False

                if should_clear:
                    keys_to_clear.append(key)

            # Clear selected entries
            for key in keys_to_clear:
                try:
                    if await self.delete(key):
                        cleared_keys.append(key)
                except Exception as e:
                    errors.append(f"Failed to clear key '{key}': {str(e)}")

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
            raise CacheBackendError(f"Memory backend clear failed: {str(e)}")

    async def exists(self, key: str) -> bool:
        """Check if a cache entry exists."""
        try:
            if key not in self._entries:
                return False

            entry = self._entries[key]

            # Check if expired
            if entry.is_expired():
                await self._remove_expired_entry(key)
                return False

            return True

        except Exception as e:
            self._record_error(f"Failed to check existence of key '{key}': {str(e)}")
            raise CacheBackendError(f"Memory backend exists failed: {str(e)}")

    async def get_size(self) -> int:
        """Get the number of entries in the cache."""
        try:
            # Clean up expired entries first
            await self._cleanup_expired()
            return len(self._entries)
        except Exception as e:
            self._record_error(f"Failed to get size: {str(e)}")
            raise CacheBackendError(f"Memory backend get_size failed: {str(e)}")

    async def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes."""
        try:
            total_bytes = 0

            for entry in self._entries.values():
                total_bytes += entry.size_bytes

            # Add overhead for data structures
            overhead = len(self._entries) * 100  # Rough estimate
            return total_bytes + overhead

        except Exception as e:
            self._record_error(f"Failed to calculate memory usage: {str(e)}")
            raise CacheBackendError(f"Memory backend get_memory_usage failed: {str(e)}")

    async def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get list of keys matching optional pattern."""
        try:
            # Clean up expired entries first
            await self._cleanup_expired()

            if pattern is None:
                return list(self._entries.keys())

            return [key for key in self._entries.keys() if fnmatch.fnmatch(key, pattern)]

        except Exception as e:
            self._record_error(f"Failed to get keys: {str(e)}")
            raise CacheBackendError(f"Memory backend get_keys failed: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get backend statistics."""
        return {
            "total_gets": self._total_gets,
            "total_sets": self._total_sets,
            "total_deletes": self._total_deletes,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": self._total_hits / max(self._total_gets, 1),
            "entry_count": len(self._entries),
            "active_expiry_tasks": len(self._expiry_tasks),
            "max_size": self.max_size
        }

    async def _expire_after_ttl(self, key: str, ttl: float) -> None:
        """Expire a key after TTL seconds."""
        try:
            await asyncio.sleep(ttl)
            await self._remove_expired_entry(key)
        except asyncio.CancelledError:
            # Task was cancelled (normal when key is updated/deleted)
            pass
        except Exception as e:
            self._record_error(f"Error expiring key '{key}': {str(e)}")

    async def _remove_expired_entry(self, key: str) -> None:
        """Remove an expired entry."""
        if key in self._entries:
            self._entries[key].mark_expired()
            del self._entries[key]

        if key in self._expiry_tasks:
            del self._expiry_tasks[key]

    async def _cleanup_expired(self) -> None:
        """Clean up all expired entries."""
        expired_keys = []

        for key, entry in self._entries.items():
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            await self._remove_expired_entry(key)

    def __str__(self) -> str:
        return f"Memory(entries={len(self._entries)}, max_size={self.max_size})"