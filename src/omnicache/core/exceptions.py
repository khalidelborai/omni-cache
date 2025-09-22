"""
Exception classes for OmniCache.

Defines all custom exceptions used throughout the caching library.
"""

from typing import Optional, Any


class CacheError(Exception):
    """Base exception for all cache-related errors."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CacheNotFoundError(CacheError):
    """Raised when a requested cache is not found."""

    def __init__(self, cache_name: str) -> None:
        super().__init__(f"Cache '{cache_name}' not found")
        self.cache_name = cache_name


class CacheKeyError(CacheError):
    """Raised when there's an issue with a cache key."""

    def __init__(self, message: str, key: Optional[str] = None) -> None:
        super().__init__(message)
        self.key = key


class CacheBackendError(CacheError):
    """Raised when there's an issue with the cache backend."""

    def __init__(self, message: str, backend_type: Optional[str] = None) -> None:
        super().__init__(message)
        self.backend_type = backend_type


class CacheStrategyError(CacheError):
    """Raised when there's an issue with the cache strategy."""

    def __init__(self, message: str, strategy_type: Optional[str] = None) -> None:
        super().__init__(message)
        self.strategy_type = strategy_type


class CacheSerializationError(CacheError):
    """Raised when there's an issue with serialization/deserialization."""

    def __init__(self, message: str, serializer_type: Optional[str] = None) -> None:
        super().__init__(message)
        self.serializer_type = serializer_type


class CacheCapacityError(CacheError):
    """Raised when cache capacity limits are exceeded."""

    def __init__(self, message: str, current_size: Optional[int] = None, max_size: Optional[int] = None) -> None:
        super().__init__(message)
        self.current_size = current_size
        self.max_size = max_size


class CacheTimeoutError(CacheError):
    """Raised when cache operations timeout."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class CacheConfigurationError(CacheError):
    """Raised when there's an issue with cache configuration."""

    def __init__(self, message: str, config_key: Optional[str] = None, config_value: Optional[Any] = None) -> None:
        super().__init__(message)
        self.config_key = config_key
        self.config_value = config_value