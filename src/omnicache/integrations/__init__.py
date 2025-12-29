"""
Framework integrations for OmniCache.

Provides seamless integration with popular web frameworks and libraries.
"""

# FastAPI integration - Basic decorators
from omnicache.integrations.fastapi import (
    cache,
    enterprise_cache,
    secure_cache,
    cache_response,
    CacheMiddleware,
    EnterpriseMonitoringMiddleware,
    get_cache_stats,
    clear_cache,
    invalidate_cache_key,
    get_enterprise_cache_stats,
    create_enterprise_fastapi_cache,
)

# Extended FastAPI integration - Lifespan and admin
from omnicache.integrations.fastapi_ext import (
    OmniCacheConfig,
    OmniCacheState,
    create_lifespan,
    get_cache,
    cache_dependency,
    get_omnicache_state,
    get_warmer,
    create_admin_router,
    CircuitBreakerMiddleware,
    setup_omnicache,
)

# FastAPI patterns - Advanced caching patterns
from omnicache.integrations.fastapi_patterns import (
    cached_endpoint,
    cached_unless_error,
    invalidates_cache,
    CacheAside,
    cache_aside,
    CacheEventBroadcaster,
    event_broadcaster,
    create_cache_events_router,
    create_invalidation_router,
)

# Core patterns - Framework-agnostic
from omnicache.core.patterns import (
    CacheEventType,
    CacheEvent,
    cache_events,
    tag_manager,
    cache_lock,
    swr_manager,
    cached_with_tags,
    cached_with_lock,
    cached_swr,
    cached_batch,
    invalidate_by_tag,
    invalidate_by_pattern,
)

# Compression
from omnicache.core.compression import (
    CompressionAlgorithm,
    CompressionConfig,
    CompressionMiddleware,
    CompressedCache,
    JSONSerializer,
    PickleSerializer,
    ZlibCompressor,
    GzipCompressor,
    LzmaCompressor,
    compressed_cache,
)

# Power features
from omnicache.core.power import (
    # Request coalescing
    RequestCoalescer,
    request_coalescer,
    coalesce_requests,
    # Multi-level caching
    CacheLevel,
    MultiLevelCache,
    multilevel_cache,
    # Versioning
    VersionedCache,
    versioned_cache,
    # Negative caching
    NegativeCache,
    negative_cache,
    # Memoization
    LRUMemoizer,
    memoize,
    # Write-behind
    WriteBehindCache,
    # Prefetching
    Prefetcher,
)

__all__ = [
    # === Basic Decorators ===
    "cache",
    "enterprise_cache",
    "secure_cache",
    "cache_response",

    # === Middleware ===
    "CacheMiddleware",
    "EnterpriseMonitoringMiddleware",
    "CircuitBreakerMiddleware",

    # === Utility Functions ===
    "get_cache_stats",
    "clear_cache",
    "invalidate_cache_key",
    "get_enterprise_cache_stats",
    "create_enterprise_fastapi_cache",

    # === Lifespan & Setup ===
    "OmniCacheConfig",
    "OmniCacheState",
    "create_lifespan",
    "setup_omnicache",

    # === Dependencies ===
    "get_cache",
    "cache_dependency",
    "get_omnicache_state",
    "get_warmer",

    # === Admin ===
    "create_admin_router",

    # === Advanced Decorators ===
    "cached_endpoint",
    "cached_unless_error",
    "invalidates_cache",
    "cached_with_tags",
    "cached_with_lock",
    "cached_swr",
    "cached_batch",

    # === Cache Aside ===
    "CacheAside",
    "cache_aside",

    # === Events ===
    "CacheEventType",
    "CacheEvent",
    "cache_events",
    "CacheEventBroadcaster",
    "event_broadcaster",
    "create_cache_events_router",

    # === Invalidation ===
    "tag_manager",
    "invalidate_by_tag",
    "invalidate_by_pattern",
    "create_invalidation_router",

    # === Locking & SWR ===
    "cache_lock",
    "swr_manager",

    # === Compression ===
    "CompressionAlgorithm",
    "CompressionConfig",
    "CompressionMiddleware",
    "CompressedCache",
    "JSONSerializer",
    "PickleSerializer",
    "ZlibCompressor",
    "GzipCompressor",
    "LzmaCompressor",
    "compressed_cache",

    # === Request Coalescing ===
    "RequestCoalescer",
    "request_coalescer",
    "coalesce_requests",

    # === Multi-Level Caching ===
    "CacheLevel",
    "MultiLevelCache",
    "multilevel_cache",

    # === Versioning ===
    "VersionedCache",
    "versioned_cache",

    # === Negative Caching ===
    "NegativeCache",
    "negative_cache",

    # === Memoization ===
    "LRUMemoizer",
    "memoize",

    # === Write-Behind ===
    "WriteBehindCache",

    # === Prefetching ===
    "Prefetcher",
]
