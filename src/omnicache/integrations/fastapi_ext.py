"""
Extended FastAPI integration for OmniCache.

Provides comprehensive FastAPI integration including:
- Lifespan management with cache warming
- Admin router with cache management endpoints
- Health check endpoints (Kubernetes-ready)
- Circuit breaker integration
- Prometheus metrics endpoint
- Dependency injection helpers
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, AsyncIterator, Union
import logging

try:
    from fastapi import FastAPI, APIRouter, Request, Response, Depends, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse, PlainTextResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = Any
    APIRouter = Any
    Request = Any
    Response = Any
    Depends = Any
    HTTPException = Any
    BackgroundTasks = Any
    JSONResponse = Any
    PlainTextResponse = Any

from omnicache.core.manager import manager
from omnicache.core.registry import registry
from omnicache.core.warmer import CacheWarmer, WarmupConfig, WarmupStrategy, WarmupItem
from omnicache.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, circuit_breaker_registry
from omnicache.core.rate_limiter import RateLimiter, RateLimitConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OmniCacheConfig:
    """Configuration for OmniCache FastAPI integration."""

    # Cache warming
    enable_warming: bool = True
    warming_strategy: WarmupStrategy = WarmupStrategy.PROGRESSIVE
    warming_batch_size: int = 100
    warming_delay: float = 0.05
    warming_timeout: float = 300.0

    # Health checks
    enable_health_checks: bool = True
    health_check_timeout: float = 5.0

    # Metrics
    enable_metrics: bool = True
    metrics_prefix: str = "omnicache"

    # Circuit breaker
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0

    # Rate limiting
    enable_rate_limiting: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: float = 60.0

    # Admin API
    admin_prefix: str = "/admin/cache"
    require_auth: bool = False

    # Warming data sources (populated at runtime)
    warming_sources: Dict[str, Callable] = field(default_factory=dict)


# =============================================================================
# State Management
# =============================================================================

class OmniCacheState:
    """
    Shared state for OmniCache FastAPI integration.

    Store this in app.state.omnicache for access across the application.
    """

    def __init__(self, config: Optional[OmniCacheConfig] = None):
        self.config = config or OmniCacheConfig()
        self.warmer: Optional[CacheWarmer] = None
        self.circuit_breaker: Optional[CircuitBreaker] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.is_ready: bool = False
        self.startup_time: Optional[datetime] = None
        self._warmup_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self) -> None:
        """Initialize all components."""
        self.startup_time = datetime.now()

        # Initialize cache warmer
        if self.config.enable_warming:
            self.warmer = CacheWarmer(WarmupConfig(
                strategy=self.config.warming_strategy,
                batch_size=self.config.warming_batch_size,
                delay_between_batches=self.config.warming_delay,
                timeout=self.config.warming_timeout
            ))

        # Initialize circuit breaker
        if self.config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
                failure_threshold=self.config.circuit_breaker_threshold,
                recovery_timeout=self.config.circuit_breaker_timeout
            ))

        # Initialize rate limiter
        if self.config.enable_rate_limiting:
            self.rate_limiter = RateLimiter(RateLimitConfig(
                requests_per_second=self.config.rate_limit_requests / self.config.rate_limit_window,
                burst_size=self.config.rate_limit_requests
            ))

        # Initialize manager
        await manager.initialize()

        self.is_ready = True
        logger.info("OmniCache initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown all components."""
        self.is_ready = False

        # Cancel warmup tasks
        for task in self._warmup_tasks.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Shutdown manager
        await manager.shutdown()

        logger.info("OmniCache shutdown complete")


# =============================================================================
# Lifespan Management
# =============================================================================

def create_lifespan(
    config: Optional[OmniCacheConfig] = None,
    on_startup: Optional[Callable] = None,
    on_shutdown: Optional[Callable] = None,
    warm_caches: Optional[Dict[str, Callable]] = None
):
    """
    Create a lifespan context manager for FastAPI with OmniCache integration.

    Args:
        config: OmniCache configuration
        on_startup: Additional startup callback
        on_shutdown: Additional shutdown callback
        warm_caches: Dictionary of cache names to warming data sources

    Returns:
        Async context manager for FastAPI lifespan

    Example:
        ```python
        async def get_products():
            async for product in db.fetch_products():
                yield (f"product:{product.id}", product.dict(), 3600)

        lifespan = create_lifespan(
            config=OmniCacheConfig(enable_warming=True),
            warm_caches={"products": get_products}
        )

        app = FastAPI(lifespan=lifespan)
        ```
    """
    config = config or OmniCacheConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Initialize state
        state = OmniCacheState(config)
        app.state.omnicache = state

        # Initialize OmniCache
        await state.initialize()

        # Run custom startup
        if on_startup:
            if asyncio.iscoroutinefunction(on_startup):
                await on_startup(app)
            else:
                on_startup(app)

        # Start cache warming
        if warm_caches and state.warmer:
            for cache_name, data_source in warm_caches.items():
                try:
                    cache = await manager.get_cache(cache_name, auto_create=True)

                    # Get data iterator
                    if asyncio.iscoroutinefunction(data_source):
                        iterator = data_source()
                    elif callable(data_source):
                        result = data_source()
                        if hasattr(result, '__aiter__'):
                            iterator = result
                        elif isinstance(result, dict):
                            await state.warmer.warm_from_dict(cache, result, cache_name=cache_name)
                            continue
                        else:
                            iterator = result
                    else:
                        continue

                    # Start background warming
                    await state.warmer.warm_background(cache, iterator, cache_name=cache_name)
                    logger.info(f"Started warming cache: {cache_name}")

                except Exception as e:
                    logger.error(f"Failed to start warming for {cache_name}: {e}")

        yield

        # Run custom shutdown
        if on_shutdown:
            if asyncio.iscoroutinefunction(on_shutdown):
                await on_shutdown(app)
            else:
                on_shutdown(app)

        # Shutdown OmniCache
        await state.shutdown()

    return lifespan


# =============================================================================
# Dependency Injection
# =============================================================================

async def get_cache(cache_name: str = "default"):
    """
    FastAPI dependency for getting a cache instance.

    Example:
        ```python
        @app.get("/items/{item_id}")
        async def get_item(item_id: str, cache = Depends(lambda: get_cache("items"))):
            cached = await cache.get(item_id)
            if cached:
                return cached
            # ... fetch and cache
        ```
    """
    cache = await manager.get_cache(cache_name, auto_create=True)
    if not cache:
        raise HTTPException(500, f"Cache '{cache_name}' not available")
    return cache


def cache_dependency(cache_name: str = "default"):
    """
    Create a cache dependency for a specific cache name.

    Example:
        ```python
        items_cache = cache_dependency("items")

        @app.get("/items/{item_id}")
        async def get_item(item_id: str, cache = Depends(items_cache)):
            return await cache.get(item_id)
        ```
    """
    async def _get_cache():
        return await get_cache(cache_name)
    return _get_cache


async def get_omnicache_state(request: Request) -> OmniCacheState:
    """Get OmniCache state from request."""
    state = getattr(request.app.state, 'omnicache', None)
    if not state:
        raise HTTPException(500, "OmniCache not initialized")
    return state


async def get_warmer(request: Request) -> CacheWarmer:
    """Get cache warmer from request."""
    state = await get_omnicache_state(request)
    if not state.warmer:
        raise HTTPException(500, "Cache warmer not enabled")
    return state.warmer


# =============================================================================
# Admin Router
# =============================================================================

def create_admin_router(
    prefix: str = "/admin/cache",
    tags: List[str] = None,
    require_auth: bool = False,
    auth_dependency: Optional[Callable] = None
) -> APIRouter:
    """
    Create an admin router for cache management.

    Args:
        prefix: URL prefix for admin endpoints
        tags: OpenAPI tags
        require_auth: Whether to require authentication
        auth_dependency: Custom auth dependency

    Returns:
        FastAPI APIRouter with cache management endpoints
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required")

    router = APIRouter(prefix=prefix, tags=tags or ["cache-admin"])

    dependencies = []
    if require_auth and auth_dependency:
        dependencies.append(Depends(auth_dependency))

    # -------------------------------------------------------------------------
    # Health Endpoints
    # -------------------------------------------------------------------------

    @router.get("/health", dependencies=dependencies)
    async def health_check(request: Request):
        """
        Comprehensive health check for all caches.

        Returns overall health status and individual cache health.
        """
        try:
            health = await manager.health_check()
            status_code = 200 if health.get("healthy", False) else 503
            return JSONResponse(content=health, status_code=status_code)
        except Exception as e:
            return JSONResponse(
                content={"healthy": False, "error": str(e)},
                status_code=503
            )

    @router.get("/health/ready", dependencies=dependencies)
    async def readiness_check(request: Request):
        """
        Kubernetes readiness probe.

        Returns 200 if the cache system is ready to serve requests.
        """
        try:
            state = await get_omnicache_state(request)
            readiness = await manager.readiness_check()

            # Check warmup status if warming is enabled
            warmup_ready = True
            if state.warmer:
                all_status = state.warmer.get_all_status()
                for name, status in all_status.items():
                    if status.get("in_progress", False):
                        warmup_ready = False
                        break

            is_ready = readiness.get("ready", False) and state.is_ready

            return JSONResponse(
                content={
                    "ready": is_ready,
                    "warmup_complete": warmup_ready,
                    "details": readiness
                },
                status_code=200 if is_ready else 503
            )
        except Exception as e:
            return JSONResponse(
                content={"ready": False, "error": str(e)},
                status_code=503
            )

    @router.get("/health/live", dependencies=dependencies)
    async def liveness_check(request: Request):
        """
        Kubernetes liveness probe.

        Returns 200 if the application is alive.
        """
        try:
            liveness = await manager.liveness_check()
            return JSONResponse(content=liveness, status_code=200)
        except Exception as e:
            return JSONResponse(
                content={"alive": False, "error": str(e)},
                status_code=503
            )

    # -------------------------------------------------------------------------
    # Cache Management Endpoints
    # -------------------------------------------------------------------------

    @router.get("/caches", dependencies=dependencies)
    async def list_caches():
        """List all registered caches with metadata."""
        return {
            "caches": registry.list_caches(),
            "total": len(registry),
            "statistics": registry.get_statistics()
        }

    @router.get("/caches/{cache_name}", dependencies=dependencies)
    async def get_cache_info(cache_name: str):
        """Get detailed information about a specific cache."""
        info = registry.get_cache_info(cache_name)
        if not info:
            raise HTTPException(404, f"Cache '{cache_name}' not found")
        return info

    @router.get("/caches/{cache_name}/stats", dependencies=dependencies)
    async def get_cache_stats(cache_name: str):
        """Get statistics for a specific cache."""
        try:
            stats = await manager.get_cache_stats(cache_name)
            return stats
        except Exception as e:
            raise HTTPException(404, str(e))

    @router.delete("/caches/{cache_name}", dependencies=dependencies)
    async def clear_cache(cache_name: str, pattern: Optional[str] = None):
        """Clear all entries from a cache or entries matching a pattern."""
        try:
            cleared = await manager.clear_cache(cache_name, pattern=pattern)
            return {"cleared": cleared, "cache": cache_name, "pattern": pattern}
        except Exception as e:
            raise HTTPException(500, str(e))

    @router.delete("/caches/{cache_name}/keys/{key:path}", dependencies=dependencies)
    async def invalidate_key(cache_name: str, key: str):
        """Invalidate a specific cache key."""
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise HTTPException(404, f"Cache '{cache_name}' not found")

            deleted = await cache.delete(key)
            return {"deleted": deleted, "key": key, "cache": cache_name}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, str(e))

    # -------------------------------------------------------------------------
    # Warming Endpoints
    # -------------------------------------------------------------------------

    @router.get("/warming/status", dependencies=dependencies)
    async def get_warming_status(request: Request):
        """Get warmup status for all caches."""
        state = await get_omnicache_state(request)
        if not state.warmer:
            return {"enabled": False, "message": "Cache warming not enabled"}

        return {
            "enabled": True,
            "status": state.warmer.get_all_status()
        }

    @router.get("/warming/status/{cache_name}", dependencies=dependencies)
    async def get_cache_warming_status(cache_name: str, request: Request):
        """Get warmup status for a specific cache."""
        state = await get_omnicache_state(request)
        if not state.warmer:
            raise HTTPException(400, "Cache warming not enabled")

        return state.warmer.get_warmup_status(cache_name)

    @router.post("/warming/{cache_name}", dependencies=dependencies)
    async def trigger_warming(
        cache_name: str,
        request: Request,
        background_tasks: BackgroundTasks,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Trigger cache warming manually.

        If data is provided, warm from that data.
        Otherwise, use registered warming source if available.
        """
        state = await get_omnicache_state(request)
        if not state.warmer:
            raise HTTPException(400, "Cache warming not enabled")

        cache = await manager.get_cache(cache_name, auto_create=True)
        if not cache:
            raise HTTPException(404, f"Cache '{cache_name}' not found")

        async def do_warming():
            try:
                if data:
                    await state.warmer.warm_from_dict(cache, data, cache_name=cache_name)
                elif cache_name in state.config.warming_sources:
                    source = state.config.warming_sources[cache_name]
                    iterator = source() if callable(source) else source
                    if isinstance(iterator, dict):
                        await state.warmer.warm_from_dict(cache, iterator, cache_name=cache_name)
                    else:
                        await state.warmer.warm_from_iterator(cache, iterator, cache_name=cache_name)
                else:
                    logger.warning(f"No warming source for cache: {cache_name}")
            except Exception as e:
                logger.error(f"Warming failed for {cache_name}: {e}")

        background_tasks.add_task(do_warming)

        return {
            "status": "warming_started",
            "cache": cache_name,
            "has_data": data is not None
        }

    @router.delete("/warming/{cache_name}", dependencies=dependencies)
    async def cancel_warming(cache_name: str, request: Request):
        """Cancel ongoing cache warming."""
        state = await get_omnicache_state(request)
        if not state.warmer:
            raise HTTPException(400, "Cache warming not enabled")

        cancelled = await state.warmer.cancel_warmup(cache_name)
        return {"cancelled": cancelled, "cache": cache_name}

    # -------------------------------------------------------------------------
    # Circuit Breaker Endpoints
    # -------------------------------------------------------------------------

    @router.get("/circuit-breakers", dependencies=dependencies)
    async def list_circuit_breakers():
        """List all circuit breakers and their states."""
        breakers = circuit_breaker_registry.list_all()
        return {
            "circuit_breakers": [
                {
                    "name": name,
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "success_count": cb.success_count,
                    "is_closed": cb.is_closed
                }
                for name, cb in breakers.items()
            ]
        }

    @router.post("/circuit-breakers/{name}/reset", dependencies=dependencies)
    async def reset_circuit_breaker(name: str):
        """Reset a circuit breaker to closed state."""
        cb = circuit_breaker_registry.get(name)
        if not cb:
            raise HTTPException(404, f"Circuit breaker '{name}' not found")

        cb.reset()
        return {"reset": True, "name": name, "state": cb.state.value}

    # -------------------------------------------------------------------------
    # Metrics Endpoints
    # -------------------------------------------------------------------------

    @router.get("/metrics", dependencies=dependencies)
    async def get_metrics():
        """Get cache metrics in JSON format."""
        metrics = {
            "registry": registry.get_statistics(),
            "caches": {}
        }

        for cache_name in registry.cache_names:
            try:
                stats = await manager.get_cache_stats(cache_name)
                metrics["caches"][cache_name] = stats
            except Exception:
                metrics["caches"][cache_name] = {"error": "Failed to get stats"}

        return metrics

    @router.get("/metrics/prometheus", dependencies=dependencies, response_class=PlainTextResponse)
    async def get_prometheus_metrics():
        """Get cache metrics in Prometheus format."""
        try:
            from omnicache.analytics.prometheus import PrometheusMetrics

            exporter = PrometheusMetrics()

            # Collect metrics from all caches
            for cache_name in registry.cache_names:
                try:
                    cache = registry.get(cache_name)
                    if cache:
                        stats = await manager.get_cache_stats(cache_name)

                        # Update Prometheus metrics
                        exporter.record_operation(cache_name, "get", stats.get("hits", 0) > 0, 0)

                except Exception:
                    pass

            return exporter.export()

        except ImportError:
            raise HTTPException(500, "Prometheus metrics not available")

    return router


# =============================================================================
# Middleware
# =============================================================================

class CircuitBreakerMiddleware:
    """
    Middleware that wraps cache operations with circuit breaker protection.
    """

    def __init__(
        self,
        app,
        circuit_breaker: Optional[CircuitBreaker] = None,
        exclude_paths: Optional[List[str]] = None
    ):
        self.app = app
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.exclude_paths = set(exclude_paths or ["/health", "/ready", "/live"])

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Skip excluded paths
        if any(path.startswith(p) for p in self.exclude_paths):
            await self.app(scope, receive, send)
            return

        # Check circuit breaker
        if not self.circuit_breaker.is_closed:
            response = JSONResponse(
                content={
                    "error": "Service temporarily unavailable",
                    "circuit_breaker": "open"
                },
                status_code=503
            )
            await response(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise


# =============================================================================
# Convenience Functions
# =============================================================================

def setup_omnicache(
    app: FastAPI,
    config: Optional[OmniCacheConfig] = None,
    include_admin: bool = True,
    admin_prefix: str = "/admin/cache",
    warm_caches: Optional[Dict[str, Callable]] = None
) -> None:
    """
    Quick setup for OmniCache with FastAPI.

    This is a convenience function that sets up lifespan, admin routes,
    and optionally cache warming in one call.

    Args:
        app: FastAPI application instance
        config: OmniCache configuration
        include_admin: Whether to include admin endpoints
        admin_prefix: Prefix for admin endpoints
        warm_caches: Dictionary of cache names to warming data sources

    Example:
        ```python
        from fastapi import FastAPI
        from omnicache.integrations.fastapi_ext import setup_omnicache, OmniCacheConfig

        app = FastAPI()

        async def product_loader():
            async for product in db.fetch_all_products():
                yield (f"product:{product.id}", product, 3600)

        setup_omnicache(
            app,
            config=OmniCacheConfig(enable_warming=True),
            warm_caches={"products": product_loader}
        )
        ```
    """
    config = config or OmniCacheConfig()

    # Store config in app state
    if not hasattr(app.state, 'omnicache_config'):
        app.state.omnicache_config = config

    # Add admin router
    if include_admin:
        admin_router = create_admin_router(
            prefix=admin_prefix,
            require_auth=config.require_auth
        )
        app.include_router(admin_router)

    # Register startup/shutdown events
    @app.on_event("startup")
    async def startup():
        state = OmniCacheState(config)
        app.state.omnicache = state
        await state.initialize()

        # Start warming
        if warm_caches and state.warmer:
            for cache_name, data_source in warm_caches.items():
                try:
                    cache = await manager.get_cache(cache_name, auto_create=True)
                    if callable(data_source):
                        result = data_source()
                        if isinstance(result, dict):
                            await state.warmer.warm_from_dict(cache, result, cache_name=cache_name)
                        else:
                            await state.warmer.warm_background(cache, result, cache_name=cache_name)
                except Exception as e:
                    logger.error(f"Failed to warm {cache_name}: {e}")

    @app.on_event("shutdown")
    async def shutdown():
        state = getattr(app.state, 'omnicache', None)
        if state:
            await state.shutdown()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "OmniCacheConfig",
    "OmniCacheState",

    # Lifespan
    "create_lifespan",

    # Dependencies
    "get_cache",
    "cache_dependency",
    "get_omnicache_state",
    "get_warmer",

    # Router
    "create_admin_router",

    # Middleware
    "CircuitBreakerMiddleware",

    # Setup
    "setup_omnicache",
]
