"""
FastAPI-specific caching patterns and decorators.

Provides request-aware caching patterns including:
- Conditional caching based on request/response
- User-scoped caching
- Request-aware cache keys
- Cache invalidation endpoints
- Real-time cache events via WebSocket
"""

import asyncio
import json
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Union
from datetime import datetime
import logging

try:
    from fastapi import Request, Response, HTTPException, WebSocket, APIRouter
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    Request = Any
    Response = Any
    HTTPException = Any
    WebSocket = Any
    APIRouter = Any
    JSONResponse = Any

from omnicache.core.patterns import (
    cache_events,
    CacheEvent,
    CacheEventType,
    tag_manager,
    invalidate_by_tag,
    invalidate_by_pattern,
)
from omnicache.core.manager import manager

logger = logging.getLogger(__name__)


# =============================================================================
# Request-Aware Caching
# =============================================================================

def cached_endpoint(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    vary_on_headers: Optional[List[str]] = None,
    vary_on_query: Optional[List[str]] = None,
    vary_on_user: bool = False,
    user_id_header: str = "X-User-ID",
    condition: Optional[Callable[[Request], bool]] = None,
    unless: Optional[Callable[[Request, Any], bool]] = None,
    tags: Optional[List[str]] = None,
    tag_func: Optional[Callable[..., List[str]]] = None
):
    """
    FastAPI endpoint caching with full request context.

    Args:
        cache_name: Name of cache to use
        ttl: Time to live in seconds
        vary_on_headers: Headers to include in cache key
        vary_on_query: Query params to include in cache key
        vary_on_user: Whether to create per-user cache entries
        user_id_header: Header containing user ID
        condition: Function to determine if request should be cached
        unless: Function to determine if response should NOT be cached
        tags: Static tags for invalidation
        tag_func: Dynamic tag generation function

    Example:
        @app.get("/api/products")
        @cached_endpoint(
            cache_name="products",
            ttl=300,
            vary_on_query=["category", "page"],
            tags=["products"]
        )
        async def list_products(request: Request, category: str = None, page: int = 1):
            return await db.fetch_products(category, page)

        @app.get("/api/user/profile")
        @cached_endpoint(
            cache_name="profiles",
            ttl=600,
            vary_on_user=True,
            condition=lambda r: r.headers.get("Authorization") is not None
        )
        async def get_profile(request: Request):
            return await get_user_profile(request)
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request in arguments
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                for v in kwargs.values():
                    if isinstance(v, Request):
                        request = v
                        break

            # Check condition
            if condition and request and not condition(request):
                return await func(*args, **kwargs)

            cache = await manager.get_cache(cache_name, auto_create=True)

            # Build cache key
            key_parts = [func.__name__, request.url.path if request else ""]

            # Add user context
            if vary_on_user and request:
                user_id = request.headers.get(user_id_header, "anonymous")
                key_parts.append(f"user:{user_id}")

            # Add headers
            if vary_on_headers and request:
                for header in vary_on_headers:
                    value = request.headers.get(header, "")
                    key_parts.append(f"h:{header}={value}")

            # Add query params
            if vary_on_query and request:
                for param in vary_on_query:
                    value = request.query_params.get(param, "")
                    key_parts.append(f"q:{param}={value}")
            elif request and not vary_on_query:
                # Include all query params if not specified
                for k, v in sorted(request.query_params.items()):
                    key_parts.append(f"q:{k}={v}")

            # Add path params from kwargs
            for k, v in sorted(kwargs.items()):
                if k != "request" and not hasattr(v, "headers"):
                    key_parts.append(f"p:{k}={v}")

            cache_key = ":".join(str(p) for p in key_parts)

            # Try cache
            cached = await cache.get(cache_key)
            if cached is not None:
                await cache_events.emit(CacheEvent(
                    event_type=CacheEventType.HIT,
                    cache_name=cache_name,
                    key=cache_key
                ))
                return cached

            # Execute function
            result = await func(*args, **kwargs)

            # Check unless condition
            if unless and request and unless(request, result):
                return result

            # Determine tags
            all_tags = list(tags or [])
            if tag_func:
                all_tags.extend(tag_func(*args, **kwargs))

            # Cache result
            if ttl:
                await cache.set(cache_key, result, ttl=ttl)
            else:
                await cache.set(cache_key, result)

            # Register tags
            if all_tags:
                await tag_manager.add_tags(cache_name, cache_key, all_tags)

            await cache_events.emit(CacheEvent(
                event_type=CacheEventType.SET,
                cache_name=cache_name,
                key=cache_key,
                ttl=ttl,
                tags=all_tags
            ))

            return result

        return wrapper
    return decorator


def cached_unless_error(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    error_ttl: Optional[float] = None
):
    """
    Cache successful responses, optionally cache errors briefly.

    Args:
        cache_name: Name of cache to use
        ttl: TTL for successful responses
        error_ttl: TTL for error responses (None = don't cache errors)

    Example:
        @app.get("/api/external")
        @cached_unless_error(cache_name="external", ttl=300, error_ttl=30)
        async def fetch_external():
            # Success cached for 5 min, errors cached for 30 sec
            return await external_api.fetch()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = await manager.get_cache(cache_name, auto_create=True)

            # Generate key
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args if not hasattr(arg, 'headers'))
            for k, v in sorted(kwargs.items()):
                if not hasattr(v, 'headers'):
                    key_parts.append(f"{k}={v}")
            cache_key = ":".join(key_parts)

            # Check cache
            cached = await cache.get(cache_key)
            if cached is not None:
                if isinstance(cached, dict) and cached.get("__is_error__"):
                    raise HTTPException(
                        status_code=cached.get("status_code", 500),
                        detail=cached.get("detail", "Cached error")
                    )
                return cached

            # Execute
            try:
                result = await func(*args, **kwargs)

                if ttl:
                    await cache.set(cache_key, result, ttl=ttl)
                else:
                    await cache.set(cache_key, result)

                return result

            except HTTPException as e:
                if error_ttl:
                    error_data = {
                        "__is_error__": True,
                        "status_code": e.status_code,
                        "detail": e.detail
                    }
                    await cache.set(cache_key, error_data, ttl=error_ttl)
                raise

        return wrapper
    return decorator


def invalidates_cache(
    tags: Optional[List[str]] = None,
    tag_func: Optional[Callable[..., List[str]]] = None,
    patterns: Optional[List[str]] = None,
    pattern_func: Optional[Callable[..., List[str]]] = None,
    cache_name: str = "default"
):
    """
    Decorator to invalidate cache after successful execution.

    Use on POST/PUT/DELETE endpoints that modify data.

    Args:
        tags: Static tags to invalidate
        tag_func: Dynamic tag generation
        patterns: Static patterns to invalidate
        pattern_func: Dynamic pattern generation
        cache_name: Cache to invalidate from

    Example:
        @app.post("/api/products")
        @invalidates_cache(tags=["products", "catalog"])
        async def create_product(product: ProductCreate):
            return await db.create_product(product)

        @app.put("/api/products/{product_id}")
        @invalidates_cache(
            tag_func=lambda product_id, **kw: [f"product:{product_id}"],
            patterns=["products:list:*"]
        )
        async def update_product(product_id: int, product: ProductUpdate):
            return await db.update_product(product_id, product)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute function first
            result = await func(*args, **kwargs)

            # Collect tags to invalidate
            all_tags = list(tags or [])
            if tag_func:
                all_tags.extend(tag_func(*args, **kwargs))

            # Invalidate tags
            for tag in all_tags:
                try:
                    await invalidate_by_tag(tag, cache_name)
                    logger.debug(f"Invalidated tag: {tag}")
                except Exception as e:
                    logger.error(f"Failed to invalidate tag {tag}: {e}")

            # Collect patterns to invalidate
            all_patterns = list(patterns or [])
            if pattern_func:
                all_patterns.extend(pattern_func(*args, **kwargs))

            # Invalidate patterns
            for pattern in all_patterns:
                try:
                    await invalidate_by_pattern(pattern, cache_name)
                    logger.debug(f"Invalidated pattern: {pattern}")
                except Exception as e:
                    logger.error(f"Failed to invalidate pattern {pattern}: {e}")

            return result

        return wrapper
    return decorator


# =============================================================================
# Cache Aside Pattern
# =============================================================================

class CacheAside:
    """
    Cache-aside (lazy loading) pattern implementation.

    Provides explicit control over cache operations with
    read-through and write-through capabilities.
    """

    def __init__(self, cache_name: str = "default", default_ttl: float = 300):
        self.cache_name = cache_name
        self.default_ttl = default_ttl
        self._cache = None

    async def _get_cache(self):
        if not self._cache:
            self._cache = await manager.get_cache(self.cache_name, auto_create=True)
        return self._cache

    async def get_or_set(
        self,
        key: str,
        loader: Callable[[], Any],
        ttl: Optional[float] = None
    ) -> Any:
        """
        Get from cache or load and set.

        Args:
            key: Cache key
            loader: Function to load data if not cached
            ttl: Optional TTL override
        """
        cache = await self._get_cache()

        # Try cache
        cached = await cache.get(key)
        if cached is not None:
            return cached

        # Load and cache
        if asyncio.iscoroutinefunction(loader):
            value = await loader()
        else:
            value = loader()

        await cache.set(key, value, ttl=ttl or self.default_ttl)
        return value

    async def get(self, key: str) -> Optional[Any]:
        """Get from cache without loading."""
        cache = await self._get_cache()
        return await cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set in cache."""
        cache = await self._get_cache()
        await cache.set(key, value, ttl=ttl or self.default_ttl)

    async def delete(self, key: str) -> bool:
        """Delete from cache."""
        cache = await self._get_cache()
        return await cache.delete(key)

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys from cache."""
        cache = await self._get_cache()
        results = {}
        for key in keys:
            value = await cache.get(key)
            if value is not None:
                results[key] = value
        return results

    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[float] = None
    ) -> None:
        """Set multiple items in cache."""
        cache = await self._get_cache()
        for key, value in items.items():
            await cache.set(key, value, ttl=ttl or self.default_ttl)

    async def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys from cache."""
        cache = await self._get_cache()
        deleted = 0
        for key in keys:
            if await cache.delete(key):
                deleted += 1
        return deleted


def cache_aside(cache_name: str = "default", default_ttl: float = 300):
    """
    FastAPI dependency for cache-aside pattern.

    Example:
        @app.get("/users/{user_id}")
        async def get_user(
            user_id: int,
            cache: CacheAside = Depends(cache_aside("users", ttl=600))
        ):
            return await cache.get_or_set(
                f"user:{user_id}",
                lambda: db.fetch_user(user_id)
            )
    """
    def dependency():
        return CacheAside(cache_name, default_ttl)
    return dependency


# =============================================================================
# WebSocket Cache Events
# =============================================================================

class CacheEventBroadcaster:
    """
    Broadcasts cache events to connected WebSocket clients.

    Useful for real-time cache monitoring dashboards.
    """

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._subscriptions: Dict[WebSocket, Set[str]] = {}
        self._started = False

    async def connect(
        self,
        websocket: WebSocket,
        event_types: Optional[List[str]] = None
    ) -> None:
        """Accept WebSocket connection and subscribe to events."""
        await websocket.accept()
        self._connections.add(websocket)

        # Subscribe to specific event types or all
        if event_types:
            self._subscriptions[websocket] = set(event_types)
        else:
            self._subscriptions[websocket] = {e.value for e in CacheEventType}

        if not self._started:
            self._start_listener()

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection."""
        self._connections.discard(websocket)
        self._subscriptions.pop(websocket, None)

    def _start_listener(self) -> None:
        """Start listening to cache events."""
        self._started = True

        async def broadcast_event(event: CacheEvent):
            if not self._connections:
                return

            message = {
                "type": event.event_type.value,
                "cache": event.cache_name,
                "key": event.key,
                "timestamp": event.timestamp.isoformat(),
                "tags": event.tags,
                "metadata": event.metadata
            }

            dead_connections = set()

            for ws in self._connections:
                subscribed = self._subscriptions.get(ws, set())
                if event.event_type.value in subscribed:
                    try:
                        await ws.send_json(message)
                    except Exception:
                        dead_connections.add(ws)

            # Clean up dead connections
            for ws in dead_connections:
                self.disconnect(ws)

        cache_events.on_any(broadcast_event)


# Global broadcaster
event_broadcaster = CacheEventBroadcaster()


def create_cache_events_router(prefix: str = "/cache/events") -> APIRouter:
    """
    Create router for cache event WebSocket endpoint.

    Example:
        app.include_router(create_cache_events_router())

        # Connect via WebSocket: ws://localhost:8000/cache/events
        # Optionally filter: ws://localhost:8000/cache/events?types=hit,miss
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required")

    router = APIRouter(prefix=prefix, tags=["cache-events"])

    @router.websocket("")
    async def cache_events_ws(websocket: WebSocket, types: Optional[str] = None):
        """WebSocket endpoint for real-time cache events."""
        event_types = types.split(",") if types else None

        await event_broadcaster.connect(websocket, event_types)

        try:
            while True:
                # Keep connection alive, handle client messages
                data = await websocket.receive_text()

                if data == "ping":
                    await websocket.send_text("pong")
                elif data.startswith("subscribe:"):
                    # Subscribe to additional event types
                    new_types = data.split(":")[1].split(",")
                    current = event_broadcaster._subscriptions.get(websocket, set())
                    current.update(new_types)
                    event_broadcaster._subscriptions[websocket] = current
                elif data.startswith("unsubscribe:"):
                    # Unsubscribe from event types
                    remove_types = data.split(":")[1].split(",")
                    current = event_broadcaster._subscriptions.get(websocket, set())
                    current.difference_update(remove_types)
                    event_broadcaster._subscriptions[websocket] = current

        except Exception:
            pass
        finally:
            event_broadcaster.disconnect(websocket)

    return router


# =============================================================================
# Invalidation Router
# =============================================================================

def create_invalidation_router(
    prefix: str = "/cache/invalidate",
    require_auth: bool = True,
    auth_dependency: Optional[Callable] = None
) -> APIRouter:
    """
    Create router for cache invalidation endpoints.

    Example:
        app.include_router(create_invalidation_router())

        # Invalidate by tag: POST /cache/invalidate/tag/products
        # Invalidate by pattern: POST /cache/invalidate/pattern?pattern=user:*
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required")

    from fastapi import Depends

    router = APIRouter(prefix=prefix, tags=["cache-invalidation"])

    dependencies = []
    if require_auth and auth_dependency:
        dependencies.append(Depends(auth_dependency))

    @router.post("/tag/{tag}", dependencies=dependencies)
    async def invalidate_tag_endpoint(tag: str, cache_name: Optional[str] = None):
        """Invalidate all entries with a specific tag."""
        count = await invalidate_by_tag(tag, cache_name)
        return {"invalidated": count, "tag": tag, "cache": cache_name}

    @router.post("/tags", dependencies=dependencies)
    async def invalidate_tags_endpoint(
        tags: List[str],
        cache_name: Optional[str] = None
    ):
        """Invalidate multiple tags at once."""
        total = 0
        results = {}
        for tag in tags:
            count = await invalidate_by_tag(tag, cache_name)
            results[tag] = count
            total += count
        return {"total_invalidated": total, "by_tag": results}

    @router.post("/pattern", dependencies=dependencies)
    async def invalidate_pattern_endpoint(
        pattern: str,
        cache_name: str = "default"
    ):
        """Invalidate entries matching a pattern."""
        count = await invalidate_by_pattern(pattern, cache_name)
        return {"invalidated": count, "pattern": pattern, "cache": cache_name}

    @router.get("/tags", dependencies=dependencies)
    async def list_tags():
        """List all tags and their key counts."""
        return tag_manager.get_stats()

    return router


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Decorators
    "cached_endpoint",
    "cached_unless_error",
    "invalidates_cache",

    # Cache Aside
    "CacheAside",
    "cache_aside",

    # WebSocket
    "CacheEventBroadcaster",
    "event_broadcaster",
    "create_cache_events_router",

    # Invalidation
    "create_invalidation_router",
]
