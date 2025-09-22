"""
Integration test for hierarchical tier management.

This test validates that the tier management system correctly handles
L1/L2/L3 cache promotion, demotion, and hierarchical operations.
"""

import pytest
import asyncio
import time
from typing import Dict, List, Any
from omnicache.core.cache import Cache
from omnicache.strategies.lru import LRUStrategy
from omnicache.backends.memory import MemoryBackend
from omnicache.backends.redis import RedisBackend
from omnicache.enterprise.tier_manager import TierManager, TierConfig
from omnicache.enterprise.hierarchy import HierarchicalCache


@pytest.mark.integration
class TestTierManagement:
    """Integration tests for hierarchical tier management."""

    @pytest.fixture
    async def tier_config(self):
        """Configuration for three-tier cache hierarchy."""
        return TierConfig(
            l1_capacity=50,
            l1_promotion_threshold=5,
            l2_capacity=200,
            l2_promotion_threshold=15,
            l3_capacity=1000,
            demotion_period=60,
            promotion_cooldown=10
        )

    @pytest.fixture
    async def tier_manager(self, tier_config):
        """Create tier manager with L1/L2/L3 hierarchy."""
        # L1: Fast memory cache
        l1_backend = MemoryBackend()
        l1_strategy = LRUStrategy(capacity=tier_config.l1_capacity)
        l1_cache = Cache(backend=l1_backend, strategy=l1_strategy, name="l1_cache")

        # L2: Larger memory cache
        l2_backend = MemoryBackend()
        l2_strategy = LRUStrategy(capacity=tier_config.l2_capacity)
        l2_cache = Cache(backend=l2_backend, strategy=l2_strategy, name="l2_cache")

        # L3: Persistent storage cache
        l3_backend = RedisBackend(host="localhost", port=6379, db=1)
        l3_strategy = LRUStrategy(capacity=tier_config.l3_capacity)
        l3_cache = Cache(backend=l3_backend, strategy=l3_strategy, name="l3_cache")

        manager = TierManager(
            l1_cache=l1_cache,
            l2_cache=l2_cache,
            l3_cache=l3_cache,
            config=tier_config
        )

        await manager.initialize()
        return manager

    @pytest.fixture
    async def hierarchical_cache(self, tier_manager):
        """Create hierarchical cache for end-to-end testing."""
        return HierarchicalCache(tier_manager=tier_manager)

    async def test_tier_promotion_workflow(self, tier_manager, tier_config):
        """Test L3 -> L2 -> L1 promotion based on access patterns."""
        key = "hot_data"
        value = {"content": "frequently accessed data", "size": 1024}

        # Initially store in L3
        await tier_manager.set(key, value, tier="l3")

        # Verify initial placement
        assert await tier_manager.exists(key, tier="l3")
        assert not await tier_manager.exists(key, tier="l2")
        assert not await tier_manager.exists(key, tier="l1")

        # Access multiple times to trigger L3 -> L2 promotion
        for _ in range(tier_config.l2_promotion_threshold + 1):
            result = await tier_manager.get(key)
            assert result == value
            await asyncio.sleep(0.1)  # Small delay to simulate real access

        # Wait for promotion to process
        await asyncio.sleep(1)

        # Verify L3 -> L2 promotion
        assert await tier_manager.exists(key, tier="l2")
        tier_stats = await tier_manager.get_tier_stats()
        assert tier_stats["l2"]["promotions"] == 1

        # Access more to trigger L2 -> L1 promotion
        for _ in range(tier_config.l1_promotion_threshold + 1):
            result = await tier_manager.get(key)
            assert result == value
            await asyncio.sleep(0.1)

        await asyncio.sleep(1)

        # Verify L2 -> L1 promotion
        assert await tier_manager.exists(key, tier="l1")
        tier_stats = await tier_manager.get_tier_stats()
        assert tier_stats["l1"]["promotions"] == 1

        # Verify data consistency across tiers
        l1_data = await tier_manager.get(key, tier="l1")
        l2_data = await tier_manager.get(key, tier="l2")
        assert l1_data == l2_data == value

    async def test_tier_demotion_workflow(self, tier_manager, tier_config):
        """Test L1 -> L2 -> L3 demotion based on inactivity."""
        key = "cooling_data"
        value = {"content": "data becoming cold", "size": 512}

        # Store directly in L1 (simulate promoted data)
        await tier_manager.set(key, value, tier="l1")
        assert await tier_manager.exists(key, tier="l1")

        # Simulate passage of time without access
        await tier_manager._simulate_time_passage(tier_config.demotion_period + 1)

        # Trigger demotion check
        await tier_manager.process_demotions()

        # Verify L1 -> L2 demotion
        assert not await tier_manager.exists(key, tier="l1")
        assert await tier_manager.exists(key, tier="l2")

        # Continue time passage for L2 -> L3 demotion
        await tier_manager._simulate_time_passage(tier_config.demotion_period + 1)
        await tier_manager.process_demotions()

        # Verify L2 -> L3 demotion
        assert not await tier_manager.exists(key, tier="l2")
        assert await tier_manager.exists(key, tier="l3")

        tier_stats = await tier_manager.get_tier_stats()
        assert tier_stats["l1"]["demotions"] == 1
        assert tier_stats["l2"]["demotions"] == 1

    async def test_hierarchical_cache_operations(self, hierarchical_cache):
        """Test end-to-end operations through hierarchical cache interface."""
        test_data = {
            "user_123": {"name": "John Doe", "preferences": {"theme": "dark"}},
            "session_456": {"token": "abc123", "expires": 1234567890},
            "config_789": {"max_connections": 100, "timeout": 30}
        }

        # Store data through hierarchical interface
        for key, value in test_data.items():
            await hierarchical_cache.set(key, value)

        # Verify all data can be retrieved
        for key, expected_value in test_data.items():
            result = await hierarchical_cache.get(key)
            assert result == expected_value

        # Test bulk operations
        bulk_keys = list(test_data.keys())
        bulk_results = await hierarchical_cache.get_many(bulk_keys)

        for key in bulk_keys:
            assert bulk_results[key] == test_data[key]

        # Test cache invalidation across tiers
        await hierarchical_cache.delete("user_123")

        # Verify deletion across all tiers
        result = await hierarchical_cache.get("user_123")
        assert result is None

    async def test_tier_capacity_management(self, tier_manager, tier_config):
        """Test tier capacity limits and eviction policies."""
        # Fill L1 to capacity
        l1_data = {}
        for i in range(tier_config.l1_capacity + 10):  # Exceed capacity
            key = f"l1_key_{i}"
            value = f"l1_value_{i}"
            l1_data[key] = value
            await tier_manager.set(key, value, tier="l1")

        # Verify L1 respects capacity
        l1_stats = await tier_manager.get_tier_stats()
        assert l1_stats["l1"]["size"] <= tier_config.l1_capacity

        # Verify evicted items are demoted to L2
        evicted_count = len(l1_data) - tier_config.l1_capacity
        l2_stats = await tier_manager.get_tier_stats()
        assert l2_stats["l2"]["size"] >= evicted_count

    async def test_concurrent_tier_operations(self, tier_manager):
        """Test thread safety and concurrent operations across tiers."""
        async def worker(worker_id: int, operations: int):
            """Worker function for concurrent testing."""
            for i in range(operations):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"

                # Random tier assignment
                tier = ["l1", "l2", "l3"][i % 3]
                await tier_manager.set(key, value, tier=tier)

                # Immediate read-back verification
                result = await tier_manager.get(key)
                assert result == value

        # Run concurrent workers
        workers = [worker(i, 50) for i in range(5)]
        await asyncio.gather(*workers)

        # Verify system state consistency
        tier_stats = await tier_manager.get_tier_stats()
        total_items = sum(stats["size"] for stats in tier_stats.values())
        assert total_items == 250  # 5 workers * 50 operations

    async def test_tier_monitoring_metrics(self, tier_manager):
        """Test comprehensive tier monitoring and metrics collection."""
        # Generate diverse access patterns
        patterns = {
            "hot": {"keys": [f"hot_{i}" for i in range(10)], "accesses": 20},
            "warm": {"keys": [f"warm_{i}" for i in range(20)], "accesses": 5},
            "cold": {"keys": [f"cold_{i}" for i in range(50)], "accesses": 1}
        }

        for pattern_name, pattern_data in patterns.items():
            for key in pattern_data["keys"]:
                value = f"{pattern_name}_data_{key}"
                await tier_manager.set(key, value)

                # Generate access pattern
                for _ in range(pattern_data["accesses"]):
                    await tier_manager.get(key)

        # Allow time for tier adjustments
        await asyncio.sleep(2)

        # Collect comprehensive metrics
        metrics = await tier_manager.get_comprehensive_metrics()

        # Verify metric categories
        assert "tier_distribution" in metrics
        assert "promotion_stats" in metrics
        assert "access_patterns" in metrics
        assert "performance_metrics" in metrics

        # Verify tier distribution makes sense
        distribution = metrics["tier_distribution"]
        assert distribution["l1"] > 0  # Hot data should be in L1
        assert distribution["l2"] >= 0
        assert distribution["l3"] >= 0

        # Verify access pattern detection
        patterns_detected = metrics["access_patterns"]
        assert "hot_keys" in patterns_detected
        assert "promotion_candidates" in patterns_detected

    async def test_tier_failure_recovery(self, tier_manager):
        """Test tier failure scenarios and recovery mechanisms."""
        key = "resilient_data"
        value = {"critical": True, "backup_required": True}

        # Store data in all tiers
        await tier_manager.set(key, value, replicate_to_all=True)

        # Simulate L1 failure
        await tier_manager.simulate_tier_failure("l1")

        # Verify data still accessible from L2/L3
        result = await tier_manager.get(key)
        assert result == value

        # Verify request routed to healthy tiers
        access_stats = await tier_manager.get_access_stats()
        assert access_stats["l2"]["requests"] > 0 or access_stats["l3"]["requests"] > 0

        # Simulate L1 recovery
        await tier_manager.recover_tier("l1")

        # Verify normal operation resumed
        await tier_manager.set("recovery_test", "test_value")
        result = await tier_manager.get("recovery_test")
        assert result == "test_value"

    async def test_enterprise_tier_policies(self, tier_manager):
        """Test enterprise-specific tier management policies."""
        # Test priority-based placement
        high_priority_data = {"priority": "high", "sla": "99.9%"}
        await tier_manager.set("critical_config", high_priority_data, priority="high")

        # High priority should go directly to L1
        assert await tier_manager.exists("critical_config", tier="l1")

        # Test data classification-based tiering
        pii_data = {"ssn": "123-45-6789", "classification": "pii"}
        await tier_manager.set("user_pii", pii_data, classification="pii")

        # PII should be in encrypted tier (L3 with encryption)
        tier_info = await tier_manager.get_key_tier_info("user_pii")
        assert tier_info["tier"] == "l3"
        assert tier_info["encrypted"] is True

        # Test compliance policies
        retention_data = {"created": time.time(), "retention_days": 90}
        await tier_manager.set("temp_data", retention_data, retention_policy="90_days")

        # Verify retention tracking
        retention_info = await tier_manager.get_retention_info("temp_data")
        assert retention_info["expires_at"] is not None
        assert retention_info["auto_delete"] is True

    @pytest.mark.parametrize("load_scenario", [
        {"name": "burst_load", "ops": 1000, "duration": 5},
        {"name": "sustained_load", "ops": 5000, "duration": 30},
        {"name": "mixed_patterns", "ops": 2000, "duration": 15}
    ])
    async def test_tier_performance_under_load(self, tier_manager, load_scenario):
        """Test tier performance under various load scenarios."""
        ops = load_scenario["ops"]
        duration = load_scenario["duration"]

        start_time = time.time()

        async def load_generator():
            """Generate load according to scenario."""
            for i in range(ops):
                key = f"load_key_{i}"
                value = f"load_value_{i}"

                # Mixed operations
                if i % 3 == 0:
                    await tier_manager.set(key, value)
                else:
                    await tier_manager.get(key)

                # Yield control to avoid blocking
                if i % 100 == 0:
                    await asyncio.sleep(0.01)

        # Run load test
        await asyncio.wait_for(load_generator(), timeout=duration + 10)

        elapsed_time = time.time() - start_time

        # Collect performance metrics
        perf_metrics = await tier_manager.get_performance_metrics()

        # Verify performance benchmarks
        assert perf_metrics["avg_response_time"] < 0.1  # < 100ms average
        assert perf_metrics["operations_per_second"] > ops / (duration * 2)  # Reasonable throughput
        assert perf_metrics["error_rate"] < 0.01  # < 1% errors

        # Verify tier system remained stable
        tier_stats = await tier_manager.get_tier_stats()
        for tier_name, stats in tier_stats.items():
            assert stats["errors"] == 0
            assert stats["availability"] > 0.99