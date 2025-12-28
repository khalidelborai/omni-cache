"""
Framework integrations for OmniCache.

Provides seamless integration with popular web frameworks and libraries.
"""

# FastAPI integration
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

# Extended FastAPI integration
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

__all__ = [
    # Basic decorators
    "cache",
    "enterprise_cache",
    "secure_cache",
    "cache_response",

    # Middleware
    "CacheMiddleware",
    "EnterpriseMonitoringMiddleware",
    "CircuitBreakerMiddleware",

    # Utility functions
    "get_cache_stats",
    "clear_cache",
    "invalidate_cache_key",
    "get_enterprise_cache_stats",
    "create_enterprise_fastapi_cache",

    # Extended integration
    "OmniCacheConfig",
    "OmniCacheState",
    "create_lifespan",
    "get_cache",
    "cache_dependency",
    "get_omnicache_state",
    "get_warmer",
    "create_admin_router",
    "setup_omnicache",
]
