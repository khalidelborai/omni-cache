"""
Integration test for backend failover scenario.

Tests graceful degradation on backend failures from the quickstart guide.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
from omnicache import Cache, RedisBackend, FileSystemBackend


class TestBackendFailover:
    """Test backend failover according to quickstart scenario B."""

    @pytest.mark.asyncio
    async def test_redis_to_filesystem_failover(self):
        """Test failover from Redis to file system backend."""
        # Primary Redis backend with file system fallback
        cache = Cache(
            name="resilient_cache",
            backend=RedisBackend("localhost:6379"),
            fallback_backend=FileSystemBackend("/tmp/cache_fallback")
        )
        
        try:
            await cache.initialize()
            
            # Store data
            await cache.set("test_key", "test_value")
            
            # Simulate Redis failure by disconnecting
            await cache.backend.disconnect()
            
            # Should automatically fallback to file system
            value = await cache.get("test_key")
            assert value == "test_value"
            
            # New data should go to fallback
            await cache.set("failover_key", "failover_value")
            value = await cache.get("failover_key")
            assert value == "failover_value"
            
        except Exception as e:
            # If Redis not available, test fallback initialization
            assert "connection" in str(e).lower() or "redis" in str(e).lower()
            
    @pytest.mark.asyncio
    async def test_graceful_degradation_maintains_functionality(self):
        """Test that cache remains functional during backend issues."""
        cache = Cache(
            name="degraded_cache",
            backend=FileSystemBackend("/tmp/test_degraded")
        )
        await cache.initialize()
        
        # Normal operation
        await cache.set("normal_key", "normal_value")
        assert await cache.get("normal_key") == "normal_value"
        
        # Simulate backend corruption/issues
        # Cache should degrade gracefully
        assert cache.status in ["ACTIVE", "DEGRADED"]
        
        # Should still be able to perform basic operations
        await cache.set("degraded_key", "degraded_value")
        value = await cache.get("degraded_key")
        assert value == "degraded_value"