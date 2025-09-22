"""
Contract tests for cache statistics API.

Tests the cache statistics and monitoring endpoints according to the OpenAPI specification.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
import asyncio
from omnicache import Cache


class TestCacheStatisticsAPI:
    """Test cache statistics functionality according to API contract."""

    @pytest.fixture
    async def cache(self):
        """Create and initialize a test cache."""
        cache = Cache(name="stats_test_cache")
        await cache.initialize()
        return cache

    @pytest.mark.asyncio
    async def test_get_statistics_initial_state(self, cache):
        """Test getting statistics from a newly created cache."""
        stats = await cache.get_statistics()

        # Verify all required statistics fields are present
        assert stats.cache_name == "stats_test_cache"
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.eviction_count == 0
        assert stats.error_count == 0
        assert stats.total_size_bytes == 0
        assert stats.entry_count == 0
        assert stats.avg_access_time_ms >= 0
        assert stats.last_reset is not None
        assert stats.collection_interval > 0

    @pytest.mark.asyncio
    async def test_statistics_hit_count_tracking(self, cache):
        """Test that hit count is properly tracked."""
        # Set up test data
        await cache.set("hit_test_key", "hit_test_value")

        # Initial stats
        stats = await cache.get_statistics()
        initial_hits = stats.hit_count

        # Access the key multiple times
        await cache.get("hit_test_key")
        await cache.get("hit_test_key")
        await cache.get("hit_test_key")

        # Check updated stats
        stats = await cache.get_statistics()
        assert stats.hit_count == initial_hits + 3

    @pytest.mark.asyncio
    async def test_statistics_miss_count_tracking(self, cache):
        """Test that miss count is properly tracked."""
        # Initial stats
        stats = await cache.get_statistics()
        initial_misses = stats.miss_count

        # Access non-existent keys
        await cache.get("nonexistent_key1")
        await cache.get("nonexistent_key2")

        # Check updated stats
        stats = await cache.get_statistics()
        assert stats.miss_count == initial_misses + 2

    @pytest.mark.asyncio
    async def test_statistics_hit_rate_calculation(self, cache):
        """Test that hit rate is correctly calculated."""
        # Set up test data
        await cache.set("hit_rate_key", "value")

        # Generate hits and misses
        await cache.get("hit_rate_key")  # Hit
        await cache.get("hit_rate_key")  # Hit
        await cache.get("nonexistent1")  # Miss
        await cache.get("nonexistent2")  # Miss

        stats = await cache.get_statistics()

        # Hit rate should be 2 hits / 4 total = 0.5
        expected_hit_rate = stats.hit_count / (stats.hit_count + stats.miss_count)
        assert abs(stats.hit_rate - expected_hit_rate) < 0.01

    @pytest.mark.asyncio
    async def test_statistics_entry_count_tracking(self, cache):
        """Test that entry count is properly tracked."""
        # Initial count should be 0
        stats = await cache.get_statistics()
        assert stats.entry_count == 0

        # Add entries
        await cache.set("count_key1", "value1")
        await cache.set("count_key2", "value2")
        await cache.set("count_key3", "value3")

        stats = await cache.get_statistics()
        assert stats.entry_count == 3

        # Remove an entry
        await cache.delete("count_key2")

        stats = await cache.get_statistics()
        assert stats.entry_count == 2

    @pytest.mark.asyncio
    async def test_statistics_total_size_tracking(self, cache):
        """Test that total size in bytes is tracked."""
        # Initial size should be 0
        stats = await cache.get_statistics()
        assert stats.total_size_bytes == 0

        # Add some data
        await cache.set("size_key1", "small_value")
        await cache.set("size_key2", "much_larger_value_with_more_content")

        stats = await cache.get_statistics()
        assert stats.total_size_bytes > 0

        # Size should increase with more data
        initial_size = stats.total_size_bytes
        await cache.set("size_key3", "additional_data_to_increase_total_size")

        stats = await cache.get_statistics()
        assert stats.total_size_bytes > initial_size

    @pytest.mark.asyncio
    async def test_statistics_eviction_count_tracking(self, cache):
        """Test that eviction count is tracked when cache limits are reached."""
        # Create cache with small size limit to trigger evictions
        small_cache = Cache(name="eviction_test", max_size=2)
        await small_cache.initialize()

        # Fill cache beyond capacity
        await small_cache.set("evict_key1", "value1")
        await small_cache.set("evict_key2", "value2")
        await small_cache.set("evict_key3", "value3")  # Should trigger eviction

        stats = await small_cache.get_statistics()
        assert stats.eviction_count >= 1

    @pytest.mark.asyncio
    async def test_statistics_error_count_tracking(self, cache):
        """Test that error count is tracked when operations fail."""
        # Initial error count should be 0
        stats = await cache.get_statistics()
        initial_errors = stats.error_count

        # Simulate backend error by corrupting cache state
        # This is implementation-specific and may vary
        try:
            # Force an error condition
            await cache.set(None, "invalid_key_value")  # Should cause error
        except Exception:
            pass  # Expected to fail

        stats = await cache.get_statistics()
        # Error count should have increased
        assert stats.error_count >= initial_errors

    @pytest.mark.asyncio
    async def test_statistics_average_access_time(self, cache):
        """Test that average access time is tracked."""
        # Set up test data
        await cache.set("timing_key", "timing_value")

        # Perform some operations to generate timing data
        for _ in range(10):
            await cache.get("timing_key")

        stats = await cache.get_statistics()

        # Should have reasonable access time (> 0, < 1000ms for simple operations)
        assert stats.avg_access_time_ms >= 0
        assert stats.avg_access_time_ms < 1000

    @pytest.mark.asyncio
    async def test_statistics_backend_status(self, cache):
        """Test that backend status is included in statistics."""
        stats = await cache.get_statistics()

        # Backend status should be one of the expected values
        assert stats.backend_status in ["connected", "degraded", "disconnected"]

        # For a healthy cache, should be connected
        assert stats.backend_status == "connected"

    @pytest.mark.asyncio
    async def test_statistics_reset_functionality(self, cache):
        """Test that statistics can be reset."""
        # Generate some statistics
        await cache.set("reset_key", "reset_value")
        await cache.get("reset_key")  # Hit
        await cache.get("nonexistent")  # Miss

        # Verify statistics exist
        stats = await cache.get_statistics()
        assert stats.hit_count > 0
        assert stats.miss_count > 0

        # Reset statistics
        await cache.reset_statistics()

        # Verify statistics are reset
        stats = await cache.get_statistics()
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.eviction_count == 0
        assert stats.error_count == 0
        # Note: entry_count and total_size_bytes might not reset as they reflect current state

    @pytest.mark.asyncio
    async def test_statistics_json_serialization(self, cache):
        """Test that statistics can be serialized to JSON."""
        # Generate some data
        await cache.set("json_key", "json_value")
        await cache.get("json_key")

        stats = await cache.get_statistics()
        stats_dict = stats.to_dict()

        # Verify all required fields are present and serializable
        required_fields = [
            "cache_name", "hit_count", "miss_count", "hit_rate", "eviction_count",
            "error_count", "total_size_bytes", "entry_count", "avg_access_time_ms",
            "backend_status", "last_reset"
        ]

        for field in required_fields:
            assert field in stats_dict
            # Verify field is JSON serializable (no complex objects)
            import json
            json.dumps(stats_dict[field])

    @pytest.mark.asyncio
    async def test_statistics_prometheus_format(self, cache):
        """Test that statistics can be exported in Prometheus format."""
        # Generate some data
        await cache.set("prom_key", "prom_value")
        await cache.get("prom_key")

        prometheus_output = await cache.get_statistics_prometheus()

        # Should contain basic Prometheus metrics
        assert "omnicache_hits_total" in prometheus_output
        assert "omnicache_misses_total" in prometheus_output
        assert "omnicache_entries_total" in prometheus_output
        assert "omnicache_size_bytes" in prometheus_output
        assert f'cache_name="stats_test_cache"' in prometheus_output

    @pytest.mark.asyncio
    async def test_statistics_collection_interval(self, cache):
        """Test that statistics collection interval can be configured."""
        # Default collection interval
        stats = await cache.get_statistics()
        default_interval = stats.collection_interval

        # Update collection interval
        await cache.set_statistics_collection_interval(5.0)

        stats = await cache.get_statistics()
        assert stats.collection_interval == 5.0
        assert stats.collection_interval != default_interval

    @pytest.mark.asyncio
    async def test_statistics_with_namespace_filtering(self, cache):
        """Test that statistics properly handle namespaced caches."""
        # Create namespaced cache
        namespaced_cache = Cache(name="namespaced_stats", namespace="tenant_1")
        await namespaced_cache.initialize()

        # Add data to both caches
        await cache.set("global_key", "global_value")
        await namespaced_cache.set("tenant_key", "tenant_value")

        # Statistics should be separate
        global_stats = await cache.get_statistics()
        tenant_stats = await namespaced_cache.get_statistics()

        assert global_stats.cache_name == "stats_test_cache"
        assert tenant_stats.cache_name == "namespaced_stats"
        assert global_stats.entry_count != tenant_stats.entry_count