"""
Contract tests for cache listing API.

Tests the cache listing and retrieval endpoints according to the OpenAPI specification.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
from omnicache import Cache
from omnicache.core.exceptions import CacheNotFoundError


class TestCacheListingAPI:
    """Test cache listing and retrieval functionality according to API contract."""

    def setup_method(self):
        """Set up test environment before each test."""
        from omnicache.core.registry import CacheRegistry
        # Clear registry for clean tests
        CacheRegistry.clear()

    def test_list_caches_empty_registry(self):
        """Test listing caches when no caches exist."""
        from omnicache.core.registry import CacheRegistry

        caches = CacheRegistry.list_caches()
        assert caches == []

    def test_list_caches_with_multiple_caches(self):
        """Test listing multiple caches."""
        from omnicache.core.registry import CacheRegistry

        # Create multiple caches
        cache1 = Cache(name="cache_1")
        cache2 = Cache(name="cache_2")
        cache3 = Cache(name="cache_3")

        cache_names = CacheRegistry.list_caches()
        assert len(cache_names) == 3
        assert "cache_1" in cache_names
        assert "cache_2" in cache_names
        assert "cache_3" in cache_names

    def test_list_caches_filtered_by_namespace(self):
        """Test listing caches filtered by namespace."""
        from omnicache.core.registry import CacheRegistry

        # Create caches with different namespaces
        cache1 = Cache(name="cache_1", namespace="tenant_a")
        cache2 = Cache(name="cache_2", namespace="tenant_b")
        cache3 = Cache(name="cache_3", namespace="tenant_a")
        cache4 = Cache(name="cache_4", namespace="")  # Default namespace

        # Filter by tenant_a namespace
        tenant_a_caches = CacheRegistry.list_caches(namespace="tenant_a")
        assert len(tenant_a_caches) == 2
        assert "cache_1" in tenant_a_caches
        assert "cache_3" in tenant_a_caches

        # Filter by tenant_b namespace
        tenant_b_caches = CacheRegistry.list_caches(namespace="tenant_b")
        assert len(tenant_b_caches) == 1
        assert "cache_2" in tenant_b_caches

        # Filter by default namespace
        default_caches = CacheRegistry.list_caches(namespace="")
        assert len(default_caches) == 1
        assert "cache_4" in default_caches

    def test_get_cache_by_name_success(self):
        """Test retrieving a cache by name successfully."""
        from omnicache.core.registry import CacheRegistry

        # Create a cache
        original_cache = Cache(name="get_test_cache")

        # Retrieve it by name
        retrieved_cache = CacheRegistry.get_cache("get_test_cache")

        assert retrieved_cache is original_cache
        assert retrieved_cache.name == "get_test_cache"

    def test_get_cache_by_name_not_found(self):
        """Test retrieving a non-existent cache raises error."""
        from omnicache.core.registry import CacheRegistry

        # Attempt to get non-existent cache
        with pytest.raises(CacheNotFoundError, match="Cache 'nonexistent' not found"):
            CacheRegistry.get_cache("nonexistent")

    def test_get_cache_info_includes_all_metadata(self):
        """Test that cache info includes all required metadata."""
        cache = Cache(
            name="info_test_cache",
            namespace="test_namespace",
            default_ttl=300,
            max_size=1000
        )

        cache_info = cache.get_info()

        # Verify all required fields are present
        assert cache_info["name"] == "info_test_cache"
        assert cache_info["namespace"] == "test_namespace"
        assert cache_info["default_ttl"] == 300
        assert cache_info["max_size"] == 1000
        assert cache_info["strategy"] is not None
        assert cache_info["backend"] is not None
        assert cache_info["status"] in ["INITIALIZING", "ACTIVE", "DEGRADED", "MAINTENANCE", "SHUTDOWN"]
        assert "created_at" in cache_info
        assert "updated_at" in cache_info
        assert "entry_count" in cache_info

    @pytest.mark.asyncio
    async def test_get_cache_info_includes_statistics(self):
        """Test that cache info includes current statistics."""
        cache = Cache(name="stats_info_test")
        await cache.initialize()

        # Add some data to generate statistics
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.get("key1")  # Hit
        await cache.get("nonexistent")  # Miss

        cache_info = cache.get_info()

        assert cache_info["statistics"]["hit_count"] >= 1
        assert cache_info["statistics"]["miss_count"] >= 1
        assert cache_info["statistics"]["entry_count"] >= 2

    def test_list_caches_returns_cache_summaries(self):
        """Test that list_caches returns proper cache summaries."""
        from omnicache.core.registry import CacheRegistry

        # Create caches with different configurations
        cache1 = Cache(name="summary_test_1", namespace="ns1", default_ttl=300)
        cache2 = Cache(name="summary_test_2", namespace="ns2", max_size=500)

        cache_summaries = CacheRegistry.list_caches_with_info()

        assert len(cache_summaries) == 2

        # Check that each summary contains required fields
        for summary in cache_summaries:
            assert "name" in summary
            assert "namespace" in summary
            assert "status" in summary
            assert "entry_count" in summary
            assert "created_at" in summary

    def test_cache_deletion_removes_from_registry(self):
        """Test that deleting a cache removes it from the registry."""
        from omnicache.core.registry import CacheRegistry

        # Create a cache
        cache = Cache(name="deletion_test")
        assert "deletion_test" in CacheRegistry.list_caches()

        # Delete the cache
        CacheRegistry.delete_cache("deletion_test")

        # Verify it's removed from registry
        assert "deletion_test" not in CacheRegistry.list_caches()

        # Verify attempting to get it raises error
        with pytest.raises(CacheNotFoundError):
            CacheRegistry.get_cache("deletion_test")

    def test_delete_nonexistent_cache_raises_error(self):
        """Test that deleting a non-existent cache raises error."""
        from omnicache.core.registry import CacheRegistry

        with pytest.raises(CacheNotFoundError, match="Cache 'nonexistent' not found"):
            CacheRegistry.delete_cache("nonexistent")

    @pytest.mark.asyncio
    async def test_cache_status_updates_in_listing(self):
        """Test that cache status updates are reflected in listings."""
        from omnicache.core.registry import CacheRegistry

        cache = Cache(name="status_test")

        # Initially should be INITIALIZING
        cache_info = CacheRegistry.get_cache("status_test").get_info()
        assert cache_info["status"] == "INITIALIZING"

        # After initialization should be ACTIVE
        await cache.initialize()
        cache_info = CacheRegistry.get_cache("status_test").get_info()
        assert cache_info["status"] == "ACTIVE"

        # After shutdown should be SHUTDOWN
        await cache.shutdown()
        cache_info = CacheRegistry.get_cache("status_test").get_info()
        assert cache_info["status"] == "SHUTDOWN"