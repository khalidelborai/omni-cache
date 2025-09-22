"""
Strategy entity model.

Defines the base Strategy class and common strategy implementations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from omnicache.models.cache import Cache


class StrategyType(Enum):
    """Strategy type enumeration."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    SIZE = "size"
    CUSTOM = "custom"


class Strategy(ABC):
    """
    Abstract base class for cache eviction and expiration strategies.

    Defines the interface that all strategy implementations must follow.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize strategy.

        Args:
            name: Strategy name identifier
            **kwargs: Strategy-specific configuration parameters
        """
        self.name = name
        self.config = kwargs
        self.created_at = datetime.now()
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if strategy is initialized."""
        return self._is_initialized

    @abstractmethod
    async def initialize(self, cache: 'Cache') -> None:
        """
        Initialize strategy with cache context.

        Args:
            cache: Cache instance this strategy will manage
        """
        self._is_initialized = True

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup strategy resources."""
        self._is_initialized = False

    @abstractmethod
    async def should_evict(self, cache: 'Cache') -> bool:
        """
        Determine if eviction should occur.

        Args:
            cache: Cache instance to check

        Returns:
            True if eviction should occur, False otherwise
        """
        pass

    @abstractmethod
    async def select_eviction_candidates(self, cache: 'Cache', count: int = 1) -> list:
        """
        Select entries for eviction.

        Args:
            cache: Cache instance
            count: Number of entries to select

        Returns:
            List of cache entry keys to evict
        """
        pass

    @abstractmethod
    async def should_expire(self, entry: Any) -> bool:
        """
        Determine if an entry should expire.

        Args:
            entry: Cache entry to check

        Returns:
            True if entry should expire, False otherwise
        """
        pass

    async def on_access(self, key: str, entry: Any) -> None:
        """
        Handle entry access event.

        Args:
            key: Accessed entry key
            entry: Cache entry that was accessed
        """
        pass

    async def on_insert(self, key: str, entry: Any) -> None:
        """
        Handle entry insertion event.

        Args:
            key: Inserted entry key
            entry: Cache entry that was inserted
        """
        pass

    async def on_update(self, key: str, old_entry: Any, new_entry: Any) -> None:
        """
        Handle entry update event.

        Args:
            key: Updated entry key
            old_entry: Previous cache entry
            new_entry: New cache entry
        """
        pass

    async def on_evict(self, key: str, entry: Any) -> None:
        """
        Handle entry eviction event.

        Args:
            key: Evicted entry key
            entry: Cache entry that was evicted
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get strategy information.

        Returns:
            Dictionary containing strategy metadata
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config,
            "is_initialized": self._is_initialized,
            "created_at": self.created_at.isoformat()
        }

    def validate_config(self) -> None:
        """
        Validate strategy configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def __str__(self) -> str:
        config_str = ", ".join(f"{k}={v}" for k, v in self.config.items())
        return f"{self.__class__.__name__}({config_str})" if config_str else self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', config={self.config})>"