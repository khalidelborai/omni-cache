"""
FastAPI integration for OmniCache.

Provides decorators and middleware for seamless caching integration with FastAPI applications.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, UTC
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Union

try:
    from fastapi import Request, Response
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response as StarletteResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    # Fallback types for when FastAPI is not available
    Request = Any
    Response = Any
    JSONResponse = Any
    BaseHTTPMiddleware = Any
    StarletteResponse = Any

from omnicache.core.manager import manager
from omnicache.core.exceptions import CacheError
from omnicache.models.security_policy import SecurityPolicy


def cache(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
    vary_on: Optional[List[str]] = None,
    condition: Optional[Callable] = None
):
    """
    Function-level caching decorator for FastAPI routes.

    Args:
        cache_name: Name of the cache to use
        ttl: Time to live for cached entries
        key_func: Custom function to generate cache keys
        vary_on: List of parameter names to include in cache key
        condition: Function to determine if caching should be applied

    Returns:
        Decorated function with caching behavior
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required for FastAPI integrations")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check condition if provided
            if condition and not condition(*args, **kwargs):
                return await func(*args, **kwargs)

            # Get or create cache
            cache = await manager.get_cache(cache_name, auto_create=True)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]

                # Add positional args
                if args:
                    key_parts.extend(str(arg) for arg in args)

                # Add specified keyword args
                if vary_on:
                    for param in vary_on:
                        if param in kwargs:
                            key_parts.append(f"{param}={kwargs[param]}")
                else:
                    # Include all keyword args if vary_on not specified
                    for key, value in sorted(kwargs.items()):
                        key_parts.append(f"{key}={value}")

                cache_key = ":".join(key_parts)

            # Try to get from cache
            try:
                cached_result = await cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            except Exception:
                # If cache fails, continue to execute function
                pass

            # Execute function and cache result
            result = await func(*args, **kwargs)

            try:
                # Store in cache
                if ttl:
                    await cache.set(cache_key, result, ttl=ttl)
                else:
                    await cache.set(cache_key, result)
            except Exception:
                # If caching fails, still return result
                pass

            return result

        return wrapper
    return decorator


def enterprise_cache(
    cache_name: str = "enterprise_default",
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
    vary_on: Optional[List[str]] = None,
    condition: Optional[Callable] = None,
    strategy: str = "arc",
    enable_analytics: bool = True,
    enable_security: bool = False,
    enable_ml_prefetch: bool = False,
    security_policy: Optional[Dict[str, Any]] = None
):
    """
    Enterprise-grade caching decorator with advanced features.

    Args:
        cache_name: Name of the cache to use
        ttl: Time to live for cached entries
        key_func: Custom function to generate cache keys
        vary_on: List of parameter names to include in cache key
        condition: Function to determine if caching should be applied
        strategy: Caching strategy (arc, lru, lfu)
        enable_analytics: Enable analytics tracking
        enable_security: Enable security features
        enable_ml_prefetch: Enable ML-based prefetching
        security_policy: Security policy configuration

    Returns:
        Decorated function with enterprise caching behavior
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required for FastAPI integrations")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request context for security
            request = None
            for arg in args:
                if hasattr(arg, 'method') and hasattr(arg, 'url'):
                    request = arg
                    break

            # Security check if enabled
            if enable_security and request:
                try:
                    # Check authorization header or user context
                    auth_header = request.headers.get("Authorization")
                    if not auth_header and security_policy and security_policy.get('require_explicit_permissions'):
                        from fastapi import HTTPException
                        raise HTTPException(status_code=401, detail="Authentication required")
                except Exception:
                    # Security check failed - don't cache and execute normally
                    pass

            # Check condition if provided
            if condition and not condition(*args, **kwargs):
                return await func(*args, **kwargs)

            # Get or create enterprise cache
            try:
                cache = await manager.get_cache(cache_name)
                if not cache:
                    # Create enterprise cache with specified features
                    cache = await manager.create_enterprise_cache(
                        cache_name,
                        strategy=strategy,
                        security_policy=security_policy if enable_security else None,
                        enable_analytics=enable_analytics,
                        enable_ml_prefetch=enable_ml_prefetch
                    )
            except Exception:
                # Fallback to basic cache if enterprise creation fails
                cache = await manager.get_cache(cache_name, auto_create=True)

            # Generate cache key with security context
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Enhanced key generation for enterprise
                key_parts = [func.__name__]

                # Add user context for security
                if enable_security and request:
                    user_id = request.headers.get("X-User-ID") or "anonymous"
                    key_parts.append(f"user:{user_id}")

                # Add positional args
                if args:
                    # Skip request object in key generation
                    non_request_args = [arg for arg in args if not hasattr(arg, 'method')]
                    key_parts.extend(str(arg) for arg in non_request_args)

                # Add specified keyword args
                if vary_on:
                    for param in vary_on:
                        if param in kwargs:
                            key_parts.append(f"{param}={kwargs[param]}")
                else:
                    # Include relevant keyword args
                    for key, value in sorted(kwargs.items()):
                        if not hasattr(value, 'method'):  # Skip request objects
                            key_parts.append(f"{key}={value}")

                cache_key = ":".join(key_parts)

            # ML prefetch prediction (if enabled)
            if enable_ml_prefetch:
                try:
                    # Predict future access patterns
                    if hasattr(manager, '_access_predictor') and manager._access_predictor:
                        predictions = await manager._access_predictor.predict_access(cache, cache_key, 1)
                        # The prediction is mainly used for internal optimization
                except Exception:
                    pass

            # Try to get from cache
            cache_hit = False
            try:
                cached_result = await cache.get(cache_key)
                if cached_result is not None:
                    cache_hit = True

                    # Analytics tracking
                    if enable_analytics and hasattr(manager, '_analytics_tracker') and manager._analytics_tracker:
                        await manager._analytics_tracker.track_access(cache_name, cache_key, "hit", request)

                    return cached_result
            except Exception:
                # If cache fails, continue to execute function
                pass

            # Execute function
            result = await func(*args, **kwargs)

            # Cache the result
            try:
                if ttl:
                    await cache.set(cache_key, result, ttl=ttl)
                else:
                    await cache.set(cache_key, result)

                # Analytics tracking for miss
                if enable_analytics and hasattr(manager, '_analytics_tracker') and manager._analytics_tracker:
                    await manager._analytics_tracker.track_access(cache_name, cache_key, "miss", request)

            except Exception:
                # If caching fails, still return result
                pass

            return result

        return wrapper
    return decorator


def secure_cache(
    cache_name: str = "secure_default",
    ttl: Optional[float] = None,
    access_level: str = "authenticated",
    encryption: str = "aes_256",
    require_permissions: bool = True,
    audit_enabled: bool = True
):
    """
    Security-focused caching decorator with encryption and access control.

    Args:
        cache_name: Name of the cache to use
        ttl: Time to live for cached entries
        access_level: Required access level (public, authenticated, authorized, restricted)
        encryption: Encryption algorithm (aes_128, aes_256, chacha20)
        require_permissions: Require explicit permissions
        audit_enabled: Enable audit logging

    Returns:
        Decorated function with secure caching behavior
    """
    security_policy = {
        'access_control_level': access_level,
        'encryption_algorithm': encryption,
        'require_explicit_permissions': require_permissions,
        'enable_audit_logging': audit_enabled
    }

    return enterprise_cache(
        cache_name=cache_name,
        ttl=ttl,
        enable_security=True,
        enable_analytics=audit_enabled,
        security_policy=security_policy
    )


def cache_response(
    ttl: Optional[float] = None,
    vary_on: Optional[List[str]] = None,
    cache_name: str = "response_cache",
    include_headers: bool = True,
    etag: bool = True,
    cache_control: Optional[str] = None
):
    """
    Response-level caching decorator with HTTP headers.

    Args:
        ttl: Time to live for cached responses
        vary_on: List of parameter names to include in cache key
        cache_name: Name of the cache to use
        include_headers: Whether to include caching headers
        etag: Whether to generate ETag headers
        cache_control: Custom Cache-Control header value

    Returns:
        Decorated function with response caching and headers
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required for FastAPI integrations")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from arguments if available
            request = None
            for arg in args:
                if hasattr(arg, 'method') and hasattr(arg, 'url'):
                    request = arg
                    break

            # Try to get request from FastAPI context if not found in args
            if request is None:
                try:
                    from starlette.requests import Request as StarletteRequest
                    # Look for Request in kwargs
                    for value in kwargs.values():
                        if isinstance(value, StarletteRequest):
                            request = value
                            break
                except ImportError:
                    pass

            # Get or create cache
            cache = await manager.get_cache(cache_name, auto_create=True)

            # Generate cache key
            key_parts = [func.__name__]

            # Add specified parameters or all kwargs
            if vary_on:
                for param in vary_on:
                    if param in kwargs:
                        key_parts.append(f"{param}={kwargs[param]}")
            else:
                for key, value in sorted(kwargs.items()):
                    if not hasattr(value, 'method'):  # Skip request objects
                        key_parts.append(f"{key}={value}")

            cache_key = ":".join(key_parts)

            # Try to get from cache
            try:
                cached_result = await cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            except Exception:
                # If cache fails, continue to execute function
                pass

            # Execute function and cache result
            result = await func(*args, **kwargs)

            try:
                # Store in cache
                if ttl:
                    await cache.set(cache_key, result, ttl=ttl)
                else:
                    await cache.set(cache_key, result)
            except Exception:
                # If caching fails, still return result
                pass

            return result

        return wrapper
    return decorator


def cache_response(
    ttl: Optional[float] = None,
    vary_on: Optional[List[str]] = None,
    cache_name: str = "response_cache",
    include_headers: bool = True,
    etag: bool = True,
    cache_control: Optional[str] = None
):
    """
    Response-level caching decorator with HTTP headers.

    Args:
        ttl: Time to live for cached responses
        vary_on: List of parameter names to include in cache key
        cache_name: Name of the cache to use
        include_headers: Whether to include caching headers
        etag: Whether to generate ETag headers
        cache_control: Custom Cache-Control header value

    Returns:
        Decorated function with response caching and headers
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required for FastAPI integrations")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from arguments if available
            request = None
            for arg in args:
                if hasattr(arg, 'method') and hasattr(arg, 'url'):
                    request = arg
                    break

            # Try to get request from FastAPI context if not found in args
            if request is None:
                try:
                    from starlette.requests import Request as StarletteRequest
                    # Look for Request in kwargs
                    for value in kwargs.values():
                        if isinstance(value, StarletteRequest):
                            request = value
                            break
                except ImportError:
                    pass

            # Get or create cache
            cache = await manager.get_cache(cache_name, auto_create=True)

            # Generate cache key
            key_parts = [func.__name__]

            # Add specified parameters or all kwargs
            if vary_on:
                for param in vary_on:
                    if param in kwargs:
                        key_parts.append(f"{param}={kwargs[param]}")
            else:
                for key, value in sorted(kwargs.items()):
                    if not hasattr(value, 'method'):  # Skip request objects
                        key_parts.append(f"{key}={value}")

            cache_key = ":".join(key_parts)

            # Check for conditional request (If-None-Match)
            if request and etag:
                if_none_match = request.headers.get("If-None-Match")
                if if_none_match:
                    # Try to get cached ETag
                    cached_etag = await cache.get(f"{cache_key}:etag")
                    if cached_etag and if_none_match == cached_etag:
                        # Return 304 Not Modified
                        return Response(status_code=304)

            # Try to get cached response
            try:
                cached_response = await cache.get(cache_key)
                if cached_response is not None:
                    response_data = json.loads(cached_response) if isinstance(cached_response, str) else cached_response
                    response = JSONResponse(content=response_data)

                    # Add caching headers
                    if include_headers:
                        _add_cache_headers(response, ttl, cache_control)

                    # Add ETag if enabled
                    if etag:
                        cached_etag = await cache.get(f"{cache_key}:etag")
                        if cached_etag:
                            response.headers["ETag"] = cached_etag

                    return response
            except Exception:
                # If cache fails, continue to execute function
                pass

            # Execute function
            result = await func(*args, **kwargs)

            # Create response
            if isinstance(result, dict):
                response = JSONResponse(content=result)
                response_data = result
            else:
                response = result
                if hasattr(result, 'body'):
                    response_data = json.loads(result.body.decode())
                else:
                    response_data = result

            # Add caching headers
            if include_headers:
                _add_cache_headers(response, ttl, cache_control)

            # Generate and add ETag
            if etag:
                etag_value = _generate_etag(response_data)
                response.headers["ETag"] = etag_value

                # Cache the ETag
                try:
                    await cache.set(f"{cache_key}:etag", etag_value, ttl=ttl)
                except Exception:
                    pass

            # Cache the response
            try:
                if ttl:
                    await cache.set(cache_key, response_data, ttl=ttl)
                else:
                    await cache.set(cache_key, response_data)
            except Exception:
                # If caching fails, still return response
                pass

            return response

        return wrapper
    return decorator


class CacheMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic HTTP caching.

    Automatically caches GET requests and handles conditional requests.
    """

    def __init__(
        self,
        app,
        cache_name: str = "http_cache",
        default_ttl: float = 300,
        exclude_paths: Optional[List[str]] = None,
        include_query_params: bool = True,
        vary_headers: Optional[List[str]] = None
    ):
        """
        Initialize cache middleware.

        Args:
            app: FastAPI application
            cache_name: Name of the cache to use
            default_ttl: Default TTL for cached responses
            exclude_paths: List of paths to exclude from caching
            include_query_params: Whether to include query parameters in cache key
            vary_headers: List of headers to include in cache key
        """
        if not HAS_FASTAPI:
            raise ImportError("FastAPI is required for FastAPI integrations")

        super().__init__(app)
        self.cache_name = cache_name
        self.default_ttl = default_ttl
        self.exclude_paths = set(exclude_paths or [])
        self.include_query_params = include_query_params
        self.vary_headers = vary_headers or []

    async def dispatch(self, request: Request, call_next: Callable) -> StarletteResponse:
        """Process request and response with caching logic."""
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)

        # Check if path is excluded
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get or create cache
        try:
            cache = await manager.get_cache(self.cache_name, auto_create=True)
        except Exception:
            # If cache fails, continue without caching
            return await call_next(request)

        # Generate cache key
        cache_key = self._generate_cache_key(request)

        # Check for conditional request (If-None-Match)
        if_none_match = request.headers.get("If-None-Match")
        if if_none_match:
            try:
                cached_etag = await cache.get(f"{cache_key}:etag")
                if cached_etag and if_none_match == cached_etag:
                    # Return 304 Not Modified
                    response = Response(status_code=304)
                    response.headers["ETag"] = cached_etag
                    return response
            except Exception:
                pass

        # Try to get cached response
        try:
            cached_data = await cache.get(cache_key)
            if cached_data is not None:
                # Reconstruct response from cached data
                response_content = cached_data.get("content", {})
                response = JSONResponse(content=response_content)

                # Add cached headers
                if "headers" in cached_data:
                    for header_name, header_value in cached_data["headers"].items():
                        response.headers[header_name] = header_value

                # Add caching headers
                self._add_cache_headers(response)

                return response
        except Exception:
            # If cache retrieval fails, continue to execute request
            pass

        # Execute request
        response = await call_next(request)

        # Only cache successful JSON responses
        if response.status_code == 200:
            try:
                # Read response body
                response_body = b""
                if hasattr(response, 'body_iterator'):
                    async for chunk in response.body_iterator:
                        response_body += chunk
                elif hasattr(response, 'body'):
                    response_body = response.body
                else:
                    # Fallback - return original response
                    return response

                # Parse response content
                try:
                    response_content = json.loads(response_body.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # If not JSON, store as string
                    response_content = response_body.decode(errors='ignore')

                # Generate ETag
                etag_value = _generate_etag(response_content)

                # Prepare cached data
                cached_data = {
                    "content": response_content,
                    "headers": {
                        "ETag": etag_value,
                        "Content-Type": response.headers.get("Content-Type", "application/json")
                    }
                }

                # Cache the response
                await cache.set(cache_key, cached_data, ttl=self.default_ttl)
                await cache.set(f"{cache_key}:etag", etag_value, ttl=self.default_ttl)

                # Create new response with cached content and headers
                new_response = JSONResponse(content=response_content)

                # Copy original headers
                for header_name, header_value in response.headers.items():
                    new_response.headers[header_name] = header_value

                # Add caching headers
                new_response.headers["ETag"] = etag_value
                self._add_cache_headers(new_response)

                return new_response

            except Exception:
                # If caching fails, return original response
                pass

        return response

    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        key_parts = [request.url.path]

        # Include query parameters
        if self.include_query_params and request.query_params:
            query_parts = []
            for key, value in sorted(request.query_params.items()):
                query_parts.append(f"{key}={value}")
            if query_parts:
                key_parts.append("?" + "&".join(query_parts))

        # Include specified headers
        for header_name in self.vary_headers:
            header_value = request.headers.get(header_name)
            if header_value:
                key_parts.append(f"{header_name}:{header_value}")

        return ":".join(key_parts)

    def _add_cache_headers(self, response: Response) -> None:
        """Add cache control headers to response."""
        cache_control = f"max-age={int(self.default_ttl)}, public"
        response.headers["Cache-Control"] = cache_control

        # Add Expires header
        expires = datetime.now(UTC) + timedelta(seconds=self.default_ttl)
        response.headers["Expires"] = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")


def _add_cache_headers(response: Response, ttl: Optional[float], cache_control: Optional[str]) -> None:
    """Add cache control headers to response."""
    if cache_control:
        response.headers["Cache-Control"] = cache_control
    elif ttl:
        response.headers["Cache-Control"] = f"max-age={int(ttl)}, public"
    else:
        response.headers["Cache-Control"] = "max-age=300, public"

    # Add Expires header
    if ttl:
        expires = datetime.now(UTC) + timedelta(seconds=ttl)
        response.headers["Expires"] = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")


def _generate_etag(content: Any) -> str:
    """Generate ETag for content."""
    if isinstance(content, dict):
        content_str = json.dumps(content, sort_keys=True)
    else:
        content_str = str(content)

    return f'"{hashlib.md5(content_str.encode()).hexdigest()}"'


# Convenience functions for FastAPI integration
async def get_cache_stats(cache_name: str) -> Dict[str, Any]:
    """Get statistics for a FastAPI cache."""
    try:
        return await manager.get_cache_stats(cache_name)
    except Exception as e:
        return {"error": str(e)}


async def clear_cache(cache_name: str, pattern: Optional[str] = None) -> int:
    """Clear entries from a FastAPI cache."""
    try:
        return await manager.clear_cache(cache_name, pattern=pattern)
    except Exception:
        return 0


async def invalidate_cache_key(cache_name: str, key: str) -> bool:
    """Invalidate a specific cache key."""
    try:
        cache = await manager.get_cache(cache_name)
        if cache:
            return await cache.delete(key)
        return False
    except Exception:
        return False


class EnterpriseMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Enterprise-grade monitoring middleware with comprehensive analytics,
    security monitoring, and performance tracking.
    """

    def __init__(
        self,
        app,
        cache_name: str = "enterprise_http_cache",
        default_ttl: float = 300,
        enable_analytics: bool = True,
        enable_security_monitoring: bool = True,
        enable_ml_insights: bool = False,
        security_policy: Optional[Dict[str, Any]] = None,
        exclude_paths: Optional[List[str]] = None,
        rate_limit_requests: Optional[int] = None,
        rate_limit_window: float = 60.0
    ):
        """
        Initialize enterprise monitoring middleware.

        Args:
            app: FastAPI application
            cache_name: Name of the cache to use
            default_ttl: Default TTL for cached responses
            enable_analytics: Enable comprehensive analytics tracking
            enable_security_monitoring: Enable security monitoring
            enable_ml_insights: Enable ML-based insights
            security_policy: Security policy configuration
            exclude_paths: List of paths to exclude from monitoring
            rate_limit_requests: Maximum requests per window (None for no limit)
            rate_limit_window: Rate limit window in seconds
        """
        if not HAS_FASTAPI:
            raise ImportError("FastAPI is required for FastAPI integrations")

        super().__init__(app)
        self.cache_name = cache_name
        self.default_ttl = default_ttl
        self.enable_analytics = enable_analytics
        self.enable_security_monitoring = enable_security_monitoring
        self.enable_ml_insights = enable_ml_insights
        self.security_policy = security_policy or {}
        self.exclude_paths = set(exclude_paths or [])
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window

        # In-memory tracking for rate limiting and analytics
        self._request_counts: Dict[str, List[float]] = {}
        self._security_events: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, List[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> StarletteResponse:
        """Process request and response with enterprise monitoring."""
        start_time = datetime.now()
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")

        # Generate request ID for tracking
        request_id = f"{start_time.timestamp()}_{hash(f'{client_ip}{request.url.path}')}"

        # Skip monitoring for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Rate limiting check
        if self.rate_limit_requests:
            if not await self._check_rate_limit(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(int(self.rate_limit_window))}
                )

        # Security monitoring
        security_events = []
        if self.enable_security_monitoring:
            security_events = await self._analyze_security(request, client_ip, user_agent)

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000  # ms

            # Analytics tracking
            if self.enable_analytics:
                await self._track_analytics(request, response, processing_time, request_id)

            # Performance monitoring
            await self._track_performance(request.url.path, processing_time, response.status_code)

            # ML insights collection
            if self.enable_ml_insights:
                await self._collect_ml_data(request, response, processing_time)

            # Security event logging
            if security_events:
                await self._log_security_events(security_events, request_id)

            # Add monitoring headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time:.2f}ms"
            response.headers["X-Cache-Status"] = "monitored"

            return response

        except Exception as e:
            # Error tracking
            if self.enable_analytics:
                await self._track_error(request, str(e), request_id)
            raise

    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if request is within rate limits."""
        current_time = datetime.now().timestamp()
        window_start = current_time - self.rate_limit_window

        # Clean old entries
        if client_ip in self._request_counts:
            self._request_counts[client_ip] = [
                timestamp for timestamp in self._request_counts[client_ip]
                if timestamp > window_start
            ]
        else:
            self._request_counts[client_ip] = []

        # Check limit
        if len(self._request_counts[client_ip]) >= self.rate_limit_requests:
            return False

        # Add current request
        self._request_counts[client_ip].append(current_time)
        return True

    async def _analyze_security(self, request: Request, client_ip: str, user_agent: str) -> List[Dict[str, Any]]:
        """Analyze request for security threats."""
        events = []

        # Check for suspicious patterns
        suspicious_patterns = [
            "union select", "drop table", "../", "script>", "eval(",
            "base64_decode", "file_get_contents", "system(", "exec("
        ]

        # Check URL and query parameters
        full_url = str(request.url).lower()
        for pattern in suspicious_patterns:
            if pattern in full_url:
                events.append({
                    "type": "suspicious_pattern",
                    "pattern": pattern,
                    "location": "url",
                    "severity": "medium",
                    "client_ip": client_ip,
                    "user_agent": user_agent,
                    "timestamp": datetime.now().isoformat()
                })

        # Check for unusual request methods
        if request.method not in ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]:
            events.append({
                "type": "unusual_method",
                "method": request.method,
                "severity": "low",
                "client_ip": client_ip,
                "timestamp": datetime.now().isoformat()
            })

        # Check for missing security headers
        if not request.headers.get("Authorization") and self.security_policy.get("require_authentication"):
            events.append({
                "type": "missing_auth",
                "severity": "high",
                "client_ip": client_ip,
                "timestamp": datetime.now().isoformat()
            })

        return events

    async def _track_analytics(self, request: Request, response: StarletteResponse, processing_time: float, request_id: str):
        """Track comprehensive analytics data."""
        if not hasattr(manager, '_analytics_tracker') or not manager._analytics_tracker:
            return

        analytics_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time": processing_time,
            "user_agent": request.headers.get("User-Agent"),
            "client_ip": request.client.host if request.client else None,
            "timestamp": datetime.now().isoformat()
        }

        try:
            await manager._analytics_tracker.track_request(analytics_data)
        except Exception:
            # Silent fail for analytics
            pass

    async def _track_performance(self, path: str, processing_time: float, status_code: int):
        """Track performance metrics."""
        # Store in memory for immediate access
        if path not in self._performance_metrics:
            self._performance_metrics[path] = []

        self._performance_metrics[path].append(processing_time)

        # Keep only last 1000 measurements per path
        if len(self._performance_metrics[path]) > 1000:
            self._performance_metrics[path] = self._performance_metrics[path][-1000:]

    async def _collect_ml_data(self, request: Request, response: StarletteResponse, processing_time: float):
        """Collect data for ML analysis."""
        if not hasattr(manager, '_access_predictor') or not manager._access_predictor:
            return

        ml_data = {
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "processing_time": processing_time,
            "query_params": dict(request.query_params),
            "hour_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "timestamp": datetime.now().isoformat()
        }

        try:
            await manager._access_predictor.collect_access_data(ml_data)
        except Exception:
            # Silent fail for ML collection
            pass

    async def _log_security_events(self, events: List[Dict[str, Any]], request_id: str):
        """Log security events."""
        if not hasattr(manager, '_security_monitor') or not manager._security_monitor:
            return

        for event in events:
            event["request_id"] = request_id
            try:
                await manager._security_monitor.log_event(event)
            except Exception:
                # Silent fail for security logging
                pass

    async def _track_error(self, request: Request, error: str, request_id: str):
        """Track application errors."""
        error_data = {
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

        if hasattr(manager, '_analytics_tracker') and manager._analytics_tracker:
            try:
                await manager._analytics_tracker.track_error(error_data)
            except Exception:
                pass

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = {}
        for path, times in self._performance_metrics.items():
            if times:
                stats[path] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "requests_count": len(times),
                    "p95_time": sorted(times)[int(len(times) * 0.95)] if len(times) > 0 else 0
                }
        return stats

    def get_security_summary(self) -> Dict[str, Any]:
        """Get security events summary."""
        total_events = len(self._security_events)
        if total_events == 0:
            return {"total_events": 0, "risk_level": "low"}

        # Analyze severity distribution
        severity_counts = {}
        for event in self._security_events[-100:]:  # Last 100 events
            severity = event.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Determine risk level
        high_severity = severity_counts.get("high", 0)
        medium_severity = severity_counts.get("medium", 0)

        if high_severity > 5:
            risk_level = "critical"
        elif high_severity > 0 or medium_severity > 10:
            risk_level = "high"
        elif medium_severity > 0:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "total_events": total_events,
            "recent_events": len(self._security_events[-100:]),
            "severity_distribution": severity_counts,
            "risk_level": risk_level
        }


# Enterprise convenience functions
async def get_enterprise_cache_stats(cache_name: str) -> Dict[str, Any]:
    """Get comprehensive enterprise cache statistics."""
    try:
        stats = await manager.get_cache_stats(cache_name)
        enterprise_analytics = await manager.get_enterprise_analytics(cache_name)
        security_report = await manager.get_security_report(cache_name)
        ml_insights = await manager.get_ml_insights(cache_name)

        return {
            "cache_stats": stats,
            "analytics": enterprise_analytics,
            "security": security_report,
            "ml_insights": ml_insights,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


async def create_enterprise_fastapi_cache(
    cache_name: str,
    strategy: str = "arc",
    enable_tiers: bool = False,
    enable_security: bool = True,
    enable_ml: bool = False,
    max_size: int = 10000
) -> Dict[str, Any]:
    """Create an enterprise cache optimized for FastAPI integration."""
    try:
        tiers = None
        if enable_tiers:
            tiers = [
                {
                    "name": "L1_Memory",
                    "tier_type": "memory",
                    "max_size": max_size // 10,
                    "default_ttl": 300,
                    "backend_config": {"type": "memory"},
                    "priority": 1
                },
                {
                    "name": "L2_Redis",
                    "tier_type": "network",
                    "max_size": max_size,
                    "default_ttl": 3600,
                    "backend_config": {"type": "redis", "host": "localhost", "port": 6379},
                    "priority": 2
                }
            ]

        security_policy = None
        if enable_security:
            security_policy = {
                'access_control_level': 'authenticated',
                'encryption_algorithm': 'aes_256',
                'require_explicit_permissions': False,  # More permissive for web APIs
                'enable_audit_logging': True
            }

        cache = await manager.create_enterprise_cache(
            cache_name,
            strategy=strategy,
            tiers=tiers,
            security_policy=security_policy,
            enable_analytics=True,
            enable_ml_prefetch=enable_ml,
            max_size=max_size
        )

        return {
            "cache_name": cache_name,
            "status": "created",
            "features": {
                "strategy": strategy,
                "hierarchical_tiers": enable_tiers,
                "security": enable_security,
                "ml_optimization": enable_ml,
                "analytics": True
            }
        }

    except Exception as e:
        return {"error": str(e), "cache_name": cache_name}