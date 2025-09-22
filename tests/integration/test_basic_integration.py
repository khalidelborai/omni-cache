"""
Integration test for basic library usage scenario.

Tests the complete basic usage workflow from the quickstart guide.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
import asyncio
from omnicache import Cache, LRUStrategy, RedisBackend, FileSystemBackend


class TestBasicIntegration:
    """Test basic library integration according to quickstart scenario 1."""

    @pytest.mark.asyncio
    async def test_basic_cache_creation_and_usage(self):
        """Test creating a cache with default settings and basic operations."""
        # Create cache with default settings (in-memory, LRU strategy)
        cache = Cache(name="my_cache")
        await cache.initialize()

        # Store and retrieve data
        await cache.set("user:123", {"name": "John", "age": 30})
        user_data = await cache.get("user:123")

        assert user_data == {"name": "John", "age": 30}

    @pytest.mark.asyncio
    async def test_cache_with_strategy_configuration(self):
        """Test creating a cache with specific strategy and limits."""
        # Create cache with specific strategy and limits
        cache = Cache(
            name="api_cache",
            strategy=LRUStrategy(max_size=1000),
            default_ttl=300  # 5 minutes
        )
        await cache.initialize()

        # Verify configuration
        assert cache.name == "api_cache"
        assert isinstance(cache.strategy, LRUStrategy)
        assert cache.default_ttl == 300

        # Test TTL functionality
        await cache.set("temp_data", "expires_soon")

        # Should be available immediately
        result = await cache.get("temp_data")
        assert result == "expires_soon"

        # Test that TTL is set correctly (detailed verification in contract tests)
        entry = await cache.get_entry("temp_data")
        assert entry.ttl == 300

    @pytest.mark.asyncio
    async def test_redis_backend_integration(self):
        """Test creating a cache with Redis backend."""
        # Redis backend for persistence
        redis_cache = Cache(
            name="persistent_cache",
            backend=RedisBackend(host="localhost", port=6379)
        )

        # Should be able to initialize (may fail if Redis not available, which is expected)
        try:
            await redis_cache.initialize()

            # If Redis is available, test basic operations
            await redis_cache.set("redis_test_key", "redis_test_value")
            value = await redis_cache.get("redis_test_key")
            assert value == "redis_test_value"

        except Exception as e:
            # Redis not available - this is acceptable for development
            assert "connection" in str(e).lower() or "redis" in str(e).lower()

    @pytest.mark.asyncio
    async def test_filesystem_backend_integration(self):
        """Test creating a cache with file system backend."""
        # File system backend for local persistence
        file_cache = Cache(
            name="file_cache",
            backend=FileSystemBackend(directory="/tmp/omnicache_test")
        )
        await file_cache.initialize()

        # Test basic operations
        await file_cache.set("file_test_key", "file_test_value")
        value = await file_cache.get("file_test_key")
        assert value == "file_test_value"

        # Test persistence across cache instances
        file_cache2 = Cache(
            name="file_cache2",
            backend=FileSystemBackend(directory="/tmp/omnicache_test")
        )
        await file_cache2.initialize()

        # Data should persist in filesystem backend
        persistent_value = await file_cache2.get("file_test_key")
        assert persistent_value == "file_test_value"

    @pytest.mark.asyncio
    async def test_minimal_configuration_requirement(self):
        """Test that minimal configuration requirement is met with sensible defaults."""
        # Should work with just a name
        cache = Cache(name="minimal_cache")
        await cache.initialize()

        # Should have sensible defaults
        assert cache.name == "minimal_cache"
        assert cache.strategy is not None
        assert cache.backend is not None
        assert cache.namespace == ""
        assert cache.max_size is None  # Unlimited
        assert cache.default_ttl is None  # No default TTL

        # Should be functional
        await cache.set("test", "value")
        result = await cache.get("test")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_multiple_cache_instances(self):
        """Test creating and managing multiple cache instances."""
        cache1 = Cache(name="cache_1")
        cache2 = Cache(name="cache_2")
        cache3 = Cache(name="cache_3")

        await cache1.initialize()
        await cache2.initialize()
        await cache3.initialize()

        # Each cache should be independent
        await cache1.set("key", "value1")
        await cache2.set("key", "value2")
        await cache3.set("key", "value3")

        assert await cache1.get("key") == "value1"
        assert await cache2.get("key") == "value2"
        assert await cache3.get("key") == "value3"

    @pytest.mark.asyncio
    async def test_cache_lifecycle_management(self):
        """Test complete cache lifecycle from creation to shutdown."""
        cache = Cache(name="lifecycle_cache")

        # Initial state
        assert cache.status == "INITIALIZING"

        # Initialize
        await cache.initialize()
        assert cache.status == "ACTIVE"

        # Use cache
        await cache.set("lifecycle_test", "active_data")
        assert await cache.get("lifecycle_test") == "active_data"

        # Shutdown
        await cache.shutdown()
        assert cache.status == "SHUTDOWN"

    @pytest.mark.asyncio
    async def test_error_handling_graceful_degradation(self):
        """Test that errors are handled gracefully."""
        cache = Cache(name="error_test_cache")
        await cache.initialize()

        # Test invalid operations
        with pytest.raises(Exception):
            await cache.set("", "empty_key_should_fail")

        with pytest.raises(Exception):
            await cache.set("key", "value", ttl=-1)

        # Cache should still be functional after errors
        await cache.set("valid_key", "valid_value")
        assert await cache.get("valid_key") == "valid_value"