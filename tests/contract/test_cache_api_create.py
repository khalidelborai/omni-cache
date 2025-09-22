"""
Contract tests for cache creation API.

Tests the cache creation endpoint according to the OpenAPI specification.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
from omnicache import Cache
from omnicache.core.exceptions import CacheError


class TestCacheCreationAPI:
    """Test cache creation functionality according to API contract."""

    def test_create_cache_with_default_settings(self):
        """Test creating a cache with minimal configuration."""
        # This test will fail until Cache class is properly implemented
        cache = Cache(name="test_cache")

        assert cache.name == "test_cache"
        assert cache.strategy is not None
        assert cache.backend is not None
        assert cache.max_size is None  # Unlimited by default
        assert cache.default_ttl is None  # No TTL by default
        assert cache.namespace == ""  # Empty namespace by default

    def test_create_cache_with_lru_strategy(self):
        """Test creating a cache with LRU strategy."""
        from omnicache.strategies.lru import LRUStrategy

        strategy = LRUStrategy(max_size=1000)
        cache = Cache(name="lru_cache", strategy=strategy)

        assert cache.name == "lru_cache"
        assert isinstance(cache.strategy, LRUStrategy)
        assert cache.strategy.max_size == 1000

    def test_create_cache_with_custom_ttl(self):
        """Test creating a cache with default TTL."""
        cache = Cache(name="ttl_cache", default_ttl=300)

        assert cache.name == "ttl_cache"
        assert cache.default_ttl == 300

    def test_create_cache_with_namespace(self):
        """Test creating a cache with namespace."""
        cache = Cache(name="namespaced_cache", namespace="tenant_1")

        assert cache.name == "namespaced_cache"
        assert cache.namespace == "tenant_1"

    def test_create_cache_with_redis_backend(self):
        """Test creating a cache with Redis backend."""
        from omnicache.backends.redis import RedisBackend

        backend = RedisBackend(host="localhost", port=6379)
        cache = Cache(name="redis_cache", backend=backend)

        assert cache.name == "redis_cache"
        assert isinstance(cache.backend, RedisBackend)

    def test_create_cache_duplicate_name_raises_error(self):
        """Test that creating a cache with duplicate name raises error."""
        # Create first cache
        cache1 = Cache(name="duplicate_test")

        # Attempting to create another cache with same name should fail
        with pytest.raises(CacheError, match="Cache with name 'duplicate_test' already exists"):
            cache2 = Cache(name="duplicate_test")

    def test_create_cache_invalid_name_raises_error(self):
        """Test that invalid cache names raise validation errors."""
        # Empty name should fail
        with pytest.raises(ValueError, match="Cache name cannot be empty"):
            Cache(name="")

        # Name with invalid characters should fail
        with pytest.raises(ValueError, match="Cache name contains invalid characters"):
            Cache(name="cache with spaces")

        # Name with special characters should fail
        with pytest.raises(ValueError, match="Cache name contains invalid characters"):
            Cache(name="cache@special#chars")

    def test_create_cache_invalid_ttl_raises_error(self):
        """Test that invalid TTL values raise validation errors."""
        # Negative TTL should fail
        with pytest.raises(ValueError, match="default_ttl must be positive"):
            Cache(name="invalid_ttl", default_ttl=-1)

        # Zero TTL should fail
        with pytest.raises(ValueError, match="default_ttl must be positive"):
            Cache(name="zero_ttl", default_ttl=0)

    def test_create_cache_invalid_max_size_raises_error(self):
        """Test that invalid max_size values raise validation errors."""
        # Negative max_size should fail
        with pytest.raises(ValueError, match="max_size must be positive"):
            Cache(name="invalid_size", max_size=-1)

        # Zero max_size should fail
        with pytest.raises(ValueError, match="max_size must be positive"):
            Cache(name="zero_size", max_size=0)

    @pytest.mark.asyncio
    async def test_cache_lifecycle_creation_to_active(self):
        """Test cache state transitions from creation to active."""
        cache = Cache(name="lifecycle_test")

        # Cache should start in INITIALIZING state
        assert cache.status == "INITIALIZING"

        # After initialization, should be ACTIVE
        await cache.initialize()
        assert cache.status == "ACTIVE"

        # Should have creation timestamp
        assert cache.created_at is not None
        assert cache.updated_at is not None

    @pytest.mark.asyncio
    async def test_cache_statistics_initialized_on_creation(self):
        """Test that cache statistics are properly initialized."""
        cache = Cache(name="stats_test")
        await cache.initialize()

        stats = cache.statistics
        assert stats.cache_name == "stats_test"
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.eviction_count == 0
        assert stats.error_count == 0
        assert stats.entry_count == 0
        assert stats.total_size_bytes == 0

    def test_cache_registry_tracks_created_caches(self):
        """Test that created caches are tracked in global registry."""
        from omnicache.core.registry import CacheRegistry

        # Clear registry for clean test
        CacheRegistry.clear()

        cache1 = Cache(name="registry_test_1")
        cache2 = Cache(name="registry_test_2")

        # Both caches should be in registry
        assert "registry_test_1" in CacheRegistry.list_caches()
        assert "registry_test_2" in CacheRegistry.list_caches()

        # Should be able to retrieve by name
        retrieved_cache1 = CacheRegistry.get_cache("registry_test_1")
        assert retrieved_cache1 is cache1