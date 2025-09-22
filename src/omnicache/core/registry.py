"""
Cache Registry implementation.

Global registry for managing cache instances with lifecycle tracking,
configuration management, and centralized access.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING
from datetime import datetime
from collections import defaultdict
from threading import RLock

from omnicache.core.exceptions import (
    CacheError,
    CacheNotFoundError,
    CacheConfigurationError
)

if TYPE_CHECKING:
    from omnicache.models.cache import Cache


class CacheRegistry:
    """
    Global registry for cache instances.

    Provides centralized management of cache instances with lifecycle
    tracking, configuration persistence, and thread-safe operations.
    """

    _instance: Optional['CacheRegistry'] = None
    _lock = RLock()

    def __new__(cls) -> 'CacheRegistry':
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize registry (called only once due to singleton)."""
        if hasattr(self, '_initialized'):
            return

        # Cache storage
        self._caches: Dict[str, 'Cache'] = {}
        self._cache_configs: Dict[str, Dict[str, Any]] = {}
        self._cache_metadata: Dict[str, Dict[str, Any]] = {}

        # Lifecycle tracking
        self._creation_times: Dict[str, datetime] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._last_accessed: Dict[str, datetime] = {}

        # State tracking
        self._is_shutting_down = False
        self._registry_lock = RLock()

        # Mark as initialized
        self._initialized = True

    @property
    def cache_count(self) -> int:
        """Get total number of registered caches."""
        with self._registry_lock:
            return len(self._caches)

    @property
    def cache_names(self) -> List[str]:
        """Get list of all registered cache names."""
        with self._registry_lock:
            return list(self._caches.keys())

    def register(self, cache: 'Cache') -> None:
        """
        Register a cache instance.

        Args:
            cache: Cache instance to register

        Raises:
            CacheError: If cache name already exists
            ValueError: If cache is invalid
        """
        if not cache or not hasattr(cache, 'name'):
            raise ValueError("Invalid cache instance")

        name = cache.name

        with self._registry_lock:
            if self._is_shutting_down:
                raise CacheError("Registry is shutting down")

            if name in self._caches:
                raise CacheError(f"Cache with name '{name}' already exists")

            # Register cache
            self._caches[name] = cache
            self._creation_times[name] = datetime.now()
            self._access_counts[name] = 0
            self._last_accessed[name] = datetime.now()

            # Store configuration and metadata
            self._cache_configs[name] = self._extract_config(cache)
            self._cache_metadata[name] = self._extract_metadata(cache)

    def unregister(self, name: str) -> bool:
        """
        Unregister a cache instance.

        Args:
            name: Name of cache to unregister

        Returns:
            True if cache was unregistered, False if not found
        """
        with self._registry_lock:
            if name in self._caches:
                # Remove all tracking data
                del self._caches[name]
                del self._creation_times[name]
                del self._access_counts[name]
                del self._last_accessed[name]
                self._cache_configs.pop(name, None)
                self._cache_metadata.pop(name, None)
                return True

            return False

    def get(self, name: str) -> Optional['Cache']:
        """
        Get a cache instance by name.

        Args:
            name: Cache name

        Returns:
            Cache instance or None if not found
        """
        with self._registry_lock:
            if name in self._caches:
                # Update access tracking
                self._access_counts[name] += 1
                self._last_accessed[name] = datetime.now()
                return self._caches[name]

            return None

    def get_or_raise(self, name: str) -> 'Cache':
        """
        Get a cache instance by name or raise exception.

        Args:
            name: Cache name

        Returns:
            Cache instance

        Raises:
            CacheNotFoundError: If cache not found
        """
        cache = self.get(name)
        if cache is None:
            raise CacheNotFoundError(f"Cache '{name}' not found")
        return cache

    def exists(self, name: str) -> bool:
        """
        Check if a cache exists.

        Args:
            name: Cache name

        Returns:
            True if cache exists, False otherwise
        """
        with self._registry_lock:
            return name in self._caches

    def list_caches(self) -> List[Dict[str, Any]]:
        """
        Get list of all caches with metadata.

        Returns:
            List of cache information dictionaries
        """
        with self._registry_lock:
            cache_list = []

            for name, cache in self._caches.items():
                info = {
                    "name": name,
                    "status": cache.status,
                    "strategy": str(cache.strategy),
                    "backend": str(cache.backend),
                    "created_at": self._creation_times[name].isoformat(),
                    "access_count": self._access_counts[name],
                    "last_accessed": self._last_accessed[name].isoformat(),
                    "namespace": cache.namespace,
                    "max_size": cache.max_size,
                    "default_ttl": cache.default_ttl
                }

                # Add metadata if available
                if name in self._cache_metadata:
                    info.update(self._cache_metadata[name])

                cache_list.append(info)

            return cache_list

    def get_cache_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific cache.

        Args:
            name: Cache name

        Returns:
            Cache information dictionary or None if not found
        """
        cache = self.get(name)
        if not cache:
            return None

        with self._registry_lock:
            info = cache.get_info()
            info.update({
                "registry_created_at": self._creation_times[name].isoformat(),
                "registry_access_count": self._access_counts[name],
                "registry_last_accessed": self._last_accessed[name].isoformat()
            })

            return info

    async def initialize_all(self) -> None:
        """Initialize all registered caches."""
        with self._registry_lock:
            caches_to_init = list(self._caches.values())

        # Initialize caches outside of lock to avoid deadlock
        for cache in caches_to_init:
            try:
                if hasattr(cache, 'initialize'):
                    await cache.initialize()
            except Exception as e:
                # Log error but continue with other caches
                print(f"Failed to initialize cache '{cache.name}': {e}")

    async def shutdown_all(self) -> None:
        """Shutdown all registered caches and the registry."""
        with self._registry_lock:
            self._is_shutting_down = True
            caches_to_shutdown = list(self._caches.values())

        # Shutdown caches outside of lock
        for cache in caches_to_shutdown:
            try:
                if hasattr(cache, 'shutdown'):
                    await cache.shutdown()
            except Exception as e:
                # Log error but continue with other caches
                print(f"Failed to shutdown cache '{cache.name}': {e}")

        # Clear registry
        with self._registry_lock:
            self._caches.clear()
            self._cache_configs.clear()
            self._cache_metadata.clear()
            self._creation_times.clear()
            self._access_counts.clear()
            self._last_accessed.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry-wide statistics.

        Returns:
            Dictionary containing registry statistics
        """
        with self._registry_lock:
            total_accesses = sum(self._access_counts.values())
            creation_times = list(self._creation_times.values())

            stats = {
                "total_caches": len(self._caches),
                "total_accesses": total_accesses,
                "is_shutting_down": self._is_shutting_down,
                "oldest_cache": min(creation_times).isoformat() if creation_times else None,
                "newest_cache": max(creation_times).isoformat() if creation_times else None
            }

            # Add cache status distribution
            status_counts = defaultdict(int)
            for cache in self._caches.values():
                status_counts[cache.status] += 1

            stats["status_distribution"] = dict(status_counts)

            return stats

    def find_caches(self, **criteria: Any) -> List['Cache']:
        """
        Find caches matching specific criteria.

        Args:
            **criteria: Search criteria (status, strategy, backend, etc.)

        Returns:
            List of matching cache instances
        """
        matching_caches = []

        with self._registry_lock:
            for cache in self._caches.values():
                matches = True

                for key, value in criteria.items():
                    if hasattr(cache, key):
                        cache_value = getattr(cache, key)
                        if callable(cache_value):
                            cache_value = cache_value()

                        if cache_value != value:
                            matches = False
                            break

                if matches:
                    matching_caches.append(cache)

        return matching_caches

    def update_cache_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a cache.

        Args:
            name: Cache name
            metadata: Metadata dictionary to update

        Returns:
            True if updated, False if cache not found
        """
        with self._registry_lock:
            if name in self._cache_metadata:
                self._cache_metadata[name].update(metadata)
                return True
            return False

    def get_cache_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a cache.

        Args:
            name: Cache name

        Returns:
            Configuration dictionary or None if not found
        """
        with self._registry_lock:
            return self._cache_configs.get(name)

    def _extract_config(self, cache: 'Cache') -> Dict[str, Any]:
        """Extract configuration from cache instance."""
        return {
            "strategy": str(cache.strategy),
            "backend": str(cache.backend),
            "max_size": cache.max_size,
            "default_ttl": cache.default_ttl,
            "namespace": cache.namespace
        }

    def _extract_metadata(self, cache: 'Cache') -> Dict[str, Any]:
        """Extract metadata from cache instance."""
        return {
            "class_name": cache.__class__.__name__,
            "module": cache.__class__.__module__
        }

    def __len__(self) -> int:
        """Get number of registered caches."""
        return self.cache_count

    def __contains__(self, name: str) -> bool:
        """Check if cache exists using 'in' operator."""
        return self.exists(name)

    def __iter__(self):
        """Iterate over cache names."""
        with self._registry_lock:
            return iter(list(self._caches.keys()))

    def __getitem__(self, name: str) -> 'Cache':
        """Get cache using bracket notation."""
        return self.get_or_raise(name)


# Global registry instance
registry = CacheRegistry()


# Convenience functions for global access
def get_cache(name: str) -> Optional['Cache']:
    """Get cache from global registry."""
    return registry.get(name)


def register_cache(cache: 'Cache') -> None:
    """Register cache in global registry."""
    registry.register(cache)


def unregister_cache(name: str) -> bool:
    """Unregister cache from global registry."""
    return registry.unregister(name)


def list_caches() -> List[Dict[str, Any]]:
    """List all caches in global registry."""
    return registry.list_caches()


def cache_exists(name: str) -> bool:
    """Check if cache exists in global registry."""
    return registry.exists(name)