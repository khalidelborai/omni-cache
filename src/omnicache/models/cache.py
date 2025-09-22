"""
Cache entity model.

Defines the core Cache class that manages key-value pairs with associated metadata
and strategy enforcement.
"""

import re
from datetime import datetime
from typing import Optional, Any, Dict, List, Union
from enum import Enum

from omnicache.core.exceptions import (
    CacheError,
    CacheConfigurationError,
    CacheKeyError,
)


class CacheStatus(Enum):
    """Cache status enumeration."""
    INITIALIZING = "INITIALIZING"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    MAINTENANCE = "MAINTENANCE"
    SHUTDOWN = "SHUTDOWN"


class Cache:
    """
    Central storage abstraction that manages key-value pairs with associated
    metadata and strategy enforcement.
    """

    # Valid name pattern: alphanumeric, hyphens, underscores
    _NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

    # Valid namespace pattern: alphanumeric, underscores (empty string allowed)
    _NAMESPACE_PATTERN = re.compile(r'^[a-zA-Z0-9_]*$')

    def __init__(
        self,
        name: str,
        strategy: Optional[Any] = None,
        backend: Optional[Any] = None,
        max_size: Optional[int] = None,
        default_ttl: Optional[float] = None,
        namespace: str = "",
        **kwargs: Any
    ) -> None:
        """
        Initialize a new Cache instance.

        Args:
            name: Unique identifier for the cache instance
            strategy: Eviction and expiration strategy (defaults to LRU)
            backend: Storage backend implementation (defaults to Memory)
            max_size: Maximum number of entries (None = unlimited)
            default_ttl: Default time-to-live in seconds
            namespace: Key namespace for multi-tenant scenarios
            **kwargs: Additional configuration parameters

        Raises:
            ValueError: If validation rules are violated
            CacheError: If cache with same name already exists
        """
        # Validate input parameters
        self._validate_name(name)
        self._validate_namespace(namespace)
        self._validate_max_size(max_size)
        self._validate_default_ttl(default_ttl)

        # Check for duplicate cache names
        self._check_duplicate_name(name)

        # Core attributes
        self._name = name
        self._namespace = namespace
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._status = CacheStatus.INITIALIZING

        # Timestamps
        self._created_at = datetime.now()
        self._updated_at = datetime.now()

        # Initialize strategy (will be set to default LRU if None)
        self._strategy = strategy or self._create_default_strategy()

        # Initialize backend (will be set to default Memory if None)
        self._backend = backend or self._create_default_backend()

        # Initialize statistics
        self._statistics = self._create_statistics()

        # Register cache instance
        self._register_cache()

    @property
    def name(self) -> str:
        """Get cache name."""
        return self._name

    @property
    def namespace(self) -> str:
        """Get cache namespace."""
        return self._namespace

    @property
    def max_size(self) -> Optional[int]:
        """Get maximum cache size."""
        return self._max_size

    @property
    def default_ttl(self) -> Optional[float]:
        """Get default TTL."""
        return self._default_ttl

    @property
    def status(self) -> str:
        """Get current cache status."""
        return self._status.value

    @property
    def strategy(self) -> Any:
        """Get cache strategy."""
        return self._strategy

    @property
    def backend(self) -> Any:
        """Get cache backend."""
        return self._backend

    @property
    def statistics(self) -> Any:
        """Get cache statistics."""
        return self._statistics

    @property
    def created_at(self) -> datetime:
        """Get cache creation timestamp."""
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp."""
        return self._updated_at

    async def initialize(self) -> None:
        """
        Initialize the cache and its components.

        Transitions from INITIALIZING to ACTIVE state.
        """
        try:
            # Initialize backend connection
            await self._backend.initialize()

            # Initialize strategy
            await self._strategy.initialize(self)

            # Initialize statistics
            await self._statistics.initialize()

            # Transition to active state
            self._status = CacheStatus.ACTIVE
            self._updated_at = datetime.now()

        except Exception as e:
            # Remain in initializing state on failure
            raise CacheError(f"Failed to initialize cache '{self._name}': {str(e)}")

    async def shutdown(self) -> None:
        """
        Shutdown the cache and cleanup resources.

        Transitions to SHUTDOWN state.
        """
        try:
            # Shutdown components
            if self._backend:
                await self._backend.shutdown()

            if self._strategy:
                await self._strategy.shutdown()

            if self._statistics:
                await self._statistics.shutdown()

            # Transition to shutdown state
            self._status = CacheStatus.SHUTDOWN
            self._updated_at = datetime.now()

            # Unregister cache
            self._unregister_cache()

        except Exception as e:
            raise CacheError(f"Failed to shutdown cache '{self._name}': {str(e)}")

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
        priority: Optional[float] = None
    ) -> None:
        """
        Set a cache entry.

        Args:
            key: Cache entry key
            value: Value to store
            ttl: Time-to-live in seconds (uses default_ttl if None)
            tags: Optional tags for bulk operations
            priority: Entry priority for eviction (0.0-1.0)

        Raises:
            CacheKeyError: If key is invalid
            ValueError: If parameters are invalid
        """
        self._validate_key(key)
        if ttl is not None:
            self._validate_ttl(ttl)
        if priority is not None:
            self._validate_priority(priority)

        # Use default TTL if not specified
        effective_ttl = ttl if ttl is not None else self._default_ttl

        # Delegate to backend
        await self._backend.set(key, value, effective_ttl, tags, priority)

        # Update timestamp
        self._updated_at = datetime.now()

    async def get(self, key: str) -> Any:
        """
        Get a cache entry value.

        Args:
            key: Cache entry key

        Returns:
            The cached value or None if not found/expired

        Raises:
            CacheKeyError: If key is invalid
        """
        self._validate_key(key)

        # Delegate to backend
        return await self._backend.get(key)

    async def delete(self, key: str) -> bool:
        """
        Delete a cache entry.

        Args:
            key: Cache entry key

        Returns:
            True if entry was deleted, False if not found

        Raises:
            CacheKeyError: If key is invalid
        """
        self._validate_key(key)

        # Delegate to backend
        result = await self._backend.delete(key)

        # Update timestamp
        self._updated_at = datetime.now()

        return result

    async def get_entry(self, key: str) -> Any:
        """
        Get complete cache entry with metadata.

        Args:
            key: Cache entry key

        Returns:
            CacheEntry object or None if not found

        Raises:
            CacheKeyError: If key is invalid
        """
        self._validate_key(key)

        # Delegate to backend
        return await self._backend.get_entry(key)

    async def clear(self, pattern: Optional[str] = None, tags: Optional[List[str]] = None) -> Any:
        """
        Clear cache entries.

        Args:
            pattern: Key pattern for selective clearing
            tags: Tags for selective clearing

        Returns:
            ClearResult with number of entries cleared
        """
        result = await self._backend.clear(pattern, tags)

        # Update timestamp
        self._updated_at = datetime.now()

        return result

    async def get_statistics(self) -> Any:
        """Get current cache statistics."""
        return await self._statistics.get_current_stats()

    def get_info(self) -> Dict[str, Any]:
        """
        Get cache information dictionary.

        Returns:
            Dictionary containing all cache metadata
        """
        return {
            "name": self._name,
            "namespace": self._namespace,
            "max_size": self._max_size,
            "default_ttl": self._default_ttl,
            "status": self.status,
            "strategy": str(self._strategy),
            "backend": str(self._backend),
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "entry_count": self._statistics.entry_count if self._statistics else 0,
            "statistics": self._statistics.to_dict() if self._statistics else {}
        }

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> 'Cache':
        """
        Create cache from configuration dictionary.

        Args:
            name: Cache name
            config: Configuration dictionary

        Returns:
            Cache instance
        """
        # Extract strategy configuration
        strategy_config = config.get('strategy', 'lru')
        strategy_params = config.get('parameters', {})

        # Create strategy instance based on configuration
        strategy = cls._create_strategy_from_config(strategy_config, strategy_params)

        # Create cache with configuration
        return cls(
            name=name,
            strategy=strategy,
            max_size=config.get('max_size'),
            default_ttl=config.get('default_ttl'),
            namespace=config.get('namespace', ''),
        )

    async def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update cache configuration.

        Args:
            config: New configuration dictionary
        """
        # Update strategy if specified
        if 'strategy' in config:
            strategy_config = config['strategy']
            strategy_params = config.get('parameters', {})
            new_strategy = self._create_strategy_from_config(strategy_config, strategy_params)
            await self.set_strategy(new_strategy)

        # Update other parameters
        if 'max_size' in config:
            self._validate_max_size(config['max_size'])
            self._max_size = config['max_size']

        if 'default_ttl' in config:
            self._validate_default_ttl(config['default_ttl'])
            self._default_ttl = config['default_ttl']

        self._updated_at = datetime.now()

    async def set_strategy(self, new_strategy: Any) -> None:
        """
        Change cache strategy at runtime.

        Args:
            new_strategy: New strategy instance
        """
        # Shutdown old strategy
        if self._strategy:
            await self._strategy.shutdown()

        # Set new strategy
        self._strategy = new_strategy
        await self._strategy.initialize(self)

        self._updated_at = datetime.now()

    # Private validation methods
    @staticmethod
    def _validate_name(name: str) -> None:
        """Validate cache name."""
        if not name:
            raise ValueError("Cache name cannot be empty")

        if not Cache._NAME_PATTERN.match(name):
            raise ValueError("Cache name contains invalid characters")

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        """Validate cache namespace."""
        if not Cache._NAMESPACE_PATTERN.match(namespace):
            raise ValueError("Namespace contains invalid characters")

    @staticmethod
    def _validate_max_size(max_size: Optional[int]) -> None:
        """Validate max_size parameter."""
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive")

    @staticmethod
    def _validate_default_ttl(default_ttl: Optional[float]) -> None:
        """Validate default_ttl parameter."""
        if default_ttl is not None and default_ttl <= 0:
            raise ValueError("default_ttl must be positive")

    @staticmethod
    def _validate_key(key: str) -> None:
        """Validate cache key."""
        if not key:
            raise CacheKeyError("Key cannot be empty")

    @staticmethod
    def _validate_ttl(ttl: float) -> None:
        """Validate TTL value."""
        if ttl <= 0:
            raise ValueError("TTL must be positive")

    @staticmethod
    def _validate_priority(priority: float) -> None:
        """Validate priority value."""
        if not (0.0 <= priority <= 1.0):
            raise ValueError("Priority must be between 0.0 and 1.0")

    def _check_duplicate_name(self, name: str) -> None:
        """Check if cache name already exists."""
        # This will be implemented when we create the CacheRegistry
        pass

    def _create_default_strategy(self) -> Any:
        """Create default LRU strategy."""
        # Will be implemented when strategy classes are available
        from omnicache.strategies.lru import LRUStrategy
        return LRUStrategy(max_size=self._max_size or 1000)

    def _create_default_backend(self) -> Any:
        """Create default memory backend."""
        # Will be implemented when backend classes are available
        from omnicache.backends.memory import MemoryBackend
        return MemoryBackend()

    def _create_statistics(self) -> Any:
        """Create statistics instance."""
        # Will be implemented when statistics class is available
        from omnicache.models.statistics import Statistics
        return Statistics(cache_name=self._name)

    @staticmethod
    def _create_strategy_from_config(strategy_type: str, params: Dict[str, Any]) -> Any:
        """Create strategy instance from configuration."""
        # Will be implemented when all strategy classes are available
        if strategy_type == 'lru':
            from omnicache.strategies.lru import LRUStrategy
            return LRUStrategy(**params)
        elif strategy_type == 'lfu':
            from omnicache.strategies.lfu import LFUStrategy
            return LFUStrategy(**params)
        elif strategy_type == 'ttl':
            from omnicache.strategies.ttl import TTLStrategy
            return TTLStrategy(**params)
        elif strategy_type == 'size':
            from omnicache.strategies.size import SizeStrategy
            return SizeStrategy(**params)
        else:
            raise CacheConfigurationError(f"Unknown strategy type: {strategy_type}")

    def _register_cache(self) -> None:
        """Register cache in global registry."""
        # Will be implemented when CacheRegistry is available
        pass

    def _unregister_cache(self) -> None:
        """Unregister cache from global registry."""
        # Will be implemented when CacheRegistry is available
        pass