"""
Backend entity model.

Defines the abstract Backend class that all storage backends must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from omnicache.models.entry import CacheEntry
    from omnicache.models.result import ClearResult


class BackendType(Enum):
    """Backend type enumeration."""
    MEMORY = "memory"
    REDIS = "redis"
    FILESYSTEM = "filesystem"
    CUSTOM = "custom"


class BackendStatus(Enum):
    """Backend status enumeration."""
    INITIALIZING = "initializing"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class Backend(ABC):
    """
    Abstract base class for cache storage backends.

    Defines the interface that all backend implementations must follow
    for storing and retrieving cache entries.
    """

    def __init__(self, name: str, **config: Any) -> None:
        """
        Initialize backend.

        Args:
            name: Backend name identifier
            **config: Backend-specific configuration parameters
        """
        self.name = name
        self.config = config
        self.created_at = datetime.now()
        self._status = BackendStatus.INITIALIZING
        self._is_initialized = False
        self._connection_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None

    @property
    def status(self) -> BackendStatus:
        """Get current backend status."""
        return self._status

    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._is_initialized

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self._status == BackendStatus.CONNECTED

    @property
    def connection_count(self) -> int:
        """Get connection attempt count."""
        return self._connection_count

    @property
    def error_count(self) -> int:
        """Get error count."""
        return self._error_count

    @property
    def last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend connection and resources.

        Raises:
            BackendError: If initialization fails
        """
        self._connection_count += 1
        self._is_initialized = True
        self._status = BackendStatus.CONNECTED

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the backend and cleanup resources.

        Raises:
            BackendError: If shutdown fails
        """
        self._is_initialized = False
        self._status = BackendStatus.DISCONNECTED

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
        priority: Optional[float] = None
    ) -> None:
        """
        Store a cache entry.

        Args:
            key: Cache entry key
            value: Value to store
            ttl: Time-to-live in seconds
            tags: Optional tags for bulk operations
            priority: Entry priority for eviction

        Raises:
            BackendError: If storage operation fails
        """
        pass

    @abstractmethod
    async def get(self, key: str) -> Any:
        """
        Retrieve a cache entry value.

        Args:
            key: Cache entry key

        Returns:
            The cached value or None if not found/expired

        Raises:
            BackendError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a cache entry.

        Args:
            key: Cache entry key

        Returns:
            True if entry was deleted, False if not found

        Raises:
            BackendError: If deletion operation fails
        """
        pass

    @abstractmethod
    async def get_entry(self, key: str) -> Optional['CacheEntry']:
        """
        Get complete cache entry with metadata.

        Args:
            key: Cache entry key

        Returns:
            CacheEntry object or None if not found

        Raises:
            BackendError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def clear(
        self,
        pattern: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> 'ClearResult':
        """
        Clear cache entries.

        Args:
            pattern: Key pattern for selective clearing
            tags: Tags for selective clearing

        Returns:
            ClearResult with number of entries cleared

        Raises:
            BackendError: If clear operation fails
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a cache entry exists.

        Args:
            key: Cache entry key

        Returns:
            True if entry exists, False otherwise

        Raises:
            BackendError: If check operation fails
        """
        pass

    @abstractmethod
    async def get_size(self) -> int:
        """
        Get the number of entries in the cache.

        Returns:
            Number of cache entries

        Raises:
            BackendError: If size operation fails
        """
        pass

    @abstractmethod
    async def get_memory_usage(self) -> int:
        """
        Get approximate memory usage in bytes.

        Returns:
            Memory usage in bytes

        Raises:
            BackendError: If memory usage calculation fails
        """
        pass

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the backend.

        Returns:
            Dictionary containing health status information
        """
        try:
            # Basic connectivity test
            test_key = f"__health_check_{datetime.now().timestamp()}"
            await self.set(test_key, "test", ttl=1.0)
            value = await self.get(test_key)
            await self.delete(test_key)

            is_healthy = value == "test"

            return {
                "healthy": is_healthy,
                "status": self._status.value,
                "connection_count": self._connection_count,
                "error_count": self._error_count,
                "last_error": self._last_error,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._record_error(str(e))
            return {
                "healthy": False,
                "status": self._status.value,
                "connection_count": self._connection_count,
                "error_count": self._error_count,
                "last_error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get list of keys matching optional pattern.

        Args:
            pattern: Optional glob pattern to filter keys

        Returns:
            List of matching keys

        Note:
            Default implementation returns empty list.
            Backends should override for efficiency.
        """
        return []

    def get_info(self) -> Dict[str, Any]:
        """
        Get backend information.

        Returns:
            Dictionary containing backend metadata
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "status": self._status.value,
            "is_initialized": self._is_initialized,
            "is_connected": self.is_connected,
            "connection_count": self._connection_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "config": self.config,
            "created_at": self.created_at.isoformat()
        }

    def _record_error(self, error_message: str) -> None:
        """Record an error for tracking."""
        self._error_count += 1
        self._last_error = error_message
        if self._status == BackendStatus.CONNECTED:
            self._status = BackendStatus.ERROR

    def _clear_error(self) -> None:
        """Clear error state."""
        self._last_error = None
        if self._status == BackendStatus.ERROR:
            self._status = BackendStatus.CONNECTED

    def validate_config(self) -> None:
        """
        Validate backend configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def __str__(self) -> str:
        config_str = ", ".join(f"{k}={v}" for k, v in self.config.items())
        return f"{self.__class__.__name__}({config_str})" if config_str else self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', status={self._status.value})>"