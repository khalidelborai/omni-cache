"""
Integration test for strategy switching scenario.

Tests runtime strategy changes from the quickstart guide.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
from omnicache import Cache, LRUStrategy, LFUStrategy, TTLStrategy


class TestStrategySwitching:
    """Test strategy switching according to quickstart scenario 4."""

    @pytest.mark.asyncio
    async def test_runtime_strategy_switching(self):
        """Test switching strategies without losing data."""
        cache = Cache(name="flexible_cache", strategy=LRUStrategy(max_size=1000))
        await cache.initialize()

        # Populate cache
        for i in range(500):
            await cache.set(f"key:{i}", f"value:{i}")

        # Verify data exists
        assert await cache.get("key:100") == "value:100"

        # Switch to LFU strategy
        await cache.set_strategy(LFUStrategy(max_size=1000))

        # Existing data should remain
        assert await cache.get("key:100") == "value:100"

        # New eviction behavior should be applied
        await cache.set("new_key", "new_value")
        assert await cache.get("new_key") == "new_value"

    @pytest.mark.asyncio  
    async def test_configuration_based_strategy_switching(self):
        """Test changing strategy via configuration update."""
        # Strategy from configuration
        config = {
            "strategy": "ttl",
            "parameters": {"default_ttl": 600, "cleanup_interval": 60}
        }

        cache = Cache.from_config("dynamic_cache", config)
        await cache.initialize()

        # Verify initial strategy
        assert isinstance(cache.strategy, TTLStrategy)

        # Change strategy via configuration update
        new_config = {
            "strategy": "size", 
            "parameters": {"max_size": 2000}
        }
        await cache.update_config(new_config)

        # Strategy should have changed
        assert cache.strategy.max_size == 2000