"""
Unit tests for FastAPI extended integration.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class TestOmniCacheConfig:
    """Tests for OmniCacheConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from omnicache.integrations.fastapi_ext import OmniCacheConfig
        from omnicache.core.warmer import WarmupStrategy

        config = OmniCacheConfig()

        assert config.enable_warming is True
        assert config.warming_strategy == WarmupStrategy.PROGRESSIVE
        assert config.warming_batch_size == 100
        assert config.enable_health_checks is True
        assert config.enable_metrics is True
        assert config.enable_circuit_breaker is True
        assert config.admin_prefix == "/admin/cache"

    def test_custom_config(self):
        """Test custom configuration."""
        from omnicache.integrations.fastapi_ext import OmniCacheConfig
        from omnicache.core.warmer import WarmupStrategy

        config = OmniCacheConfig(
            enable_warming=False,
            warming_strategy=WarmupStrategy.EAGER,
            warming_batch_size=50,
            rate_limit_requests=200,
            admin_prefix="/api/cache"
        )

        assert config.enable_warming is False
        assert config.warming_strategy == WarmupStrategy.EAGER
        assert config.warming_batch_size == 50
        assert config.rate_limit_requests == 200
        assert config.admin_prefix == "/api/cache"


class TestOmniCacheState:
    """Tests for OmniCacheState."""

    def test_initial_state(self):
        """Test initial state values."""
        from omnicache.integrations.fastapi_ext import OmniCacheState, OmniCacheConfig

        state = OmniCacheState()

        assert state.warmer is None
        assert state.circuit_breaker is None
        assert state.rate_limiter is None
        assert state.is_ready is False
        assert state.startup_time is None

    def test_state_with_config(self):
        """Test state with custom config."""
        from omnicache.integrations.fastapi_ext import OmniCacheState, OmniCacheConfig

        config = OmniCacheConfig(enable_rate_limiting=True)
        state = OmniCacheState(config)

        assert state.config.enable_rate_limiting is True

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test state initialization."""
        from omnicache.integrations.fastapi_ext import OmniCacheState, OmniCacheConfig

        config = OmniCacheConfig(
            enable_warming=True,
            enable_circuit_breaker=True,
            enable_rate_limiting=True
        )
        state = OmniCacheState(config)

        with patch('omnicache.integrations.fastapi_ext.manager') as mock_manager:
            mock_manager.initialize = AsyncMock()
            await state.initialize()

        assert state.is_ready is True
        assert state.startup_time is not None
        assert state.warmer is not None
        assert state.circuit_breaker is not None
        assert state.rate_limiter is not None

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test state shutdown."""
        from omnicache.integrations.fastapi_ext import OmniCacheState

        state = OmniCacheState()
        state.is_ready = True

        with patch('omnicache.integrations.fastapi_ext.manager') as mock_manager:
            mock_manager.shutdown = AsyncMock()
            await state.shutdown()

        assert state.is_ready is False


class TestCacheDependency:
    """Tests for cache dependency functions."""

    @pytest.mark.asyncio
    async def test_cache_dependency_factory(self):
        """Test cache_dependency factory function."""
        from omnicache.integrations.fastapi_ext import cache_dependency

        items_cache_dep = cache_dependency("items")

        assert callable(items_cache_dep)

    @pytest.mark.asyncio
    async def test_get_cache(self):
        """Test get_cache function."""
        from omnicache.integrations.fastapi_ext import get_cache

        mock_cache = MagicMock()

        with patch('omnicache.integrations.fastapi_ext.manager') as mock_manager:
            mock_manager.get_cache = AsyncMock(return_value=mock_cache)

            cache = await get_cache("test_cache")

            assert cache == mock_cache
            mock_manager.get_cache.assert_called_once_with("test_cache", auto_create=True)


class TestCreateLifespan:
    """Tests for create_lifespan function."""

    def test_lifespan_creation(self):
        """Test lifespan context manager creation."""
        from omnicache.integrations.fastapi_ext import create_lifespan, OmniCacheConfig

        config = OmniCacheConfig()
        lifespan = create_lifespan(config=config)

        # Should return an async context manager
        assert callable(lifespan)

    def test_lifespan_with_warming_sources(self):
        """Test lifespan with cache warming sources."""
        from omnicache.integrations.fastapi_ext import create_lifespan

        async def product_loader():
            yield ("product:1", {"id": 1, "name": "Test"}, 3600)

        lifespan = create_lifespan(
            warm_caches={"products": product_loader}
        )

        assert callable(lifespan)

    def test_lifespan_with_callbacks(self):
        """Test lifespan with startup/shutdown callbacks."""
        from omnicache.integrations.fastapi_ext import create_lifespan

        startup_called = False
        shutdown_called = False

        def on_startup(app):
            nonlocal startup_called
            startup_called = True

        def on_shutdown(app):
            nonlocal shutdown_called
            shutdown_called = True

        lifespan = create_lifespan(
            on_startup=on_startup,
            on_shutdown=on_shutdown
        )

        assert callable(lifespan)


class TestCreateAdminRouter:
    """Tests for create_admin_router function."""

    def test_router_creation(self):
        """Test admin router creation."""
        from omnicache.integrations.fastapi_ext import create_admin_router

        router = create_admin_router()

        assert router is not None
        assert router.prefix == "/admin/cache"

    def test_router_custom_prefix(self):
        """Test router with custom prefix."""
        from omnicache.integrations.fastapi_ext import create_admin_router

        router = create_admin_router(prefix="/api/v1/cache")

        assert router.prefix == "/api/v1/cache"

    def test_router_with_tags(self):
        """Test router with custom tags."""
        from omnicache.integrations.fastapi_ext import create_admin_router

        router = create_admin_router(tags=["cache", "admin"])

        assert "cache" in router.tags
        assert "admin" in router.tags

    def test_router_endpoints_exist(self):
        """Test that all expected endpoints are registered."""
        from omnicache.integrations.fastapi_ext import create_admin_router

        router = create_admin_router()

        # Get all route paths
        paths = [route.path for route in router.routes]

        # Check health endpoints
        assert "/health" in paths
        assert "/health/ready" in paths
        assert "/health/live" in paths

        # Check cache management endpoints
        assert "/caches" in paths
        assert "/caches/{cache_name}" in paths
        assert "/caches/{cache_name}/stats" in paths

        # Check warming endpoints
        assert "/warming/status" in paths
        assert "/warming/status/{cache_name}" in paths
        assert "/warming/{cache_name}" in paths

        # Check circuit breaker endpoints
        assert "/circuit-breakers" in paths

        # Check metrics endpoints
        assert "/metrics" in paths
        assert "/metrics/prometheus" in paths


class TestCircuitBreakerMiddleware:
    """Tests for CircuitBreakerMiddleware."""

    def test_middleware_creation(self):
        """Test middleware creation."""
        from omnicache.integrations.fastapi_ext import CircuitBreakerMiddleware
        from omnicache.core.circuit_breaker import CircuitBreaker

        mock_app = MagicMock()
        cb = CircuitBreaker()

        middleware = CircuitBreakerMiddleware(mock_app, circuit_breaker=cb)

        assert middleware.app == mock_app
        assert middleware.circuit_breaker == cb

    def test_middleware_default_excludes(self):
        """Test default excluded paths."""
        from omnicache.integrations.fastapi_ext import CircuitBreakerMiddleware

        mock_app = MagicMock()
        middleware = CircuitBreakerMiddleware(mock_app)

        assert "/health" in middleware.exclude_paths
        assert "/ready" in middleware.exclude_paths
        assert "/live" in middleware.exclude_paths

    def test_middleware_custom_excludes(self):
        """Test custom excluded paths."""
        from omnicache.integrations.fastapi_ext import CircuitBreakerMiddleware

        mock_app = MagicMock()
        middleware = CircuitBreakerMiddleware(
            mock_app,
            exclude_paths=["/api/status", "/metrics"]
        )

        assert "/api/status" in middleware.exclude_paths
        assert "/metrics" in middleware.exclude_paths


class TestSetupOmnicache:
    """Tests for setup_omnicache convenience function."""

    def test_setup_basic(self):
        """Test basic setup."""
        from omnicache.integrations.fastapi_ext import setup_omnicache

        # Create a mock FastAPI app
        mock_app = MagicMock()
        mock_app.state = MagicMock()
        mock_app.include_router = MagicMock()
        mock_app.on_event = MagicMock(return_value=lambda f: f)

        setup_omnicache(mock_app)

        # Verify router was included
        mock_app.include_router.assert_called_once()

    def test_setup_without_admin(self):
        """Test setup without admin router."""
        from omnicache.integrations.fastapi_ext import setup_omnicache

        mock_app = MagicMock()
        mock_app.state = MagicMock()
        mock_app.include_router = MagicMock()
        mock_app.on_event = MagicMock(return_value=lambda f: f)

        setup_omnicache(mock_app, include_admin=False)

        # Verify router was NOT included
        mock_app.include_router.assert_not_called()

    def test_setup_with_config(self):
        """Test setup with custom config."""
        from omnicache.integrations.fastapi_ext import setup_omnicache, OmniCacheConfig

        mock_app = MagicMock()
        mock_app.state = MagicMock()
        mock_app.include_router = MagicMock()
        mock_app.on_event = MagicMock(return_value=lambda f: f)

        config = OmniCacheConfig(admin_prefix="/cache-admin")

        setup_omnicache(mock_app, config=config)

        # Verify config was stored
        assert mock_app.state.omnicache_config == config


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_accessible(self):
        """Test that all exports are accessible."""
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

        # All imports should succeed
        assert OmniCacheConfig is not None
        assert OmniCacheState is not None
        assert create_lifespan is not None
        assert get_cache is not None
        assert cache_dependency is not None
        assert get_omnicache_state is not None
        assert get_warmer is not None
        assert create_admin_router is not None
        assert CircuitBreakerMiddleware is not None
        assert setup_omnicache is not None

    def test_integrations_init_exports(self):
        """Test that integrations __init__ exports work."""
        from omnicache.integrations import (
            # Basic decorators
            cache,
            enterprise_cache,
            secure_cache,
            cache_response,
            # Middleware
            CacheMiddleware,
            CircuitBreakerMiddleware,
            # Extended
            OmniCacheConfig,
            create_lifespan,
            create_admin_router,
            setup_omnicache,
        )

        assert cache is not None
        assert enterprise_cache is not None
        assert secure_cache is not None
        assert cache_response is not None
        assert CacheMiddleware is not None
        assert CircuitBreakerMiddleware is not None
        assert OmniCacheConfig is not None
        assert create_lifespan is not None
        assert create_admin_router is not None
        assert setup_omnicache is not None
