"""
OmniCache: Universal caching library with modern strategies and framework integration.

This library provides a flexible caching solution that supports multiple eviction
strategies, storage backends, and seamless integration with web frameworks like FastAPI.
"""

__version__ = "0.1.0"
__author__ = "OmniCache Team"
__email__ = "elboraikhalid@gmail.com"

# Core exports that users will commonly import
from omnicache.core.cache import Cache
from omnicache.core.exceptions import (
    CacheError,
    CacheNotFoundError,
    CacheKeyError,
    CacheBackendError,
    CacheStrategyError,
)

# Strategy exports
from omnicache.strategies.lru import LRUStrategy
from omnicache.strategies.lfu import LFUStrategy
from omnicache.strategies.ttl import TTLStrategy
from omnicache.strategies.size import SizeStrategy

# Backend exports
from omnicache.backends.memory import MemoryBackend
from omnicache.backends.redis import RedisBackend
from omnicache.backends.filesystem import FileSystemBackend

__all__ = [
    # Core
    "Cache",
    # Exceptions
    "CacheError",
    "CacheNotFoundError",
    "CacheKeyError",
    "CacheBackendError",
    "CacheStrategyError",
    # Strategies
    "LRUStrategy",
    "LFUStrategy",
    "TTLStrategy",
    "SizeStrategy",
    # Backends
    "MemoryBackend",
    "RedisBackend",
    "FileSystemBackend",
]