"""
Integration test for ARC vs LRU performance comparison.

This test validates that ARC strategy provides superior performance
compared to LRU in workloads with mixed access patterns.
"""

import pytest
import time
import random
from typing import List, Dict
from omnicache.core.cache import Cache
from omnicache.strategies.arc import ARCStrategy
from omnicache.strategies.lru import LRUStrategy
from omnicache.backends.memory import MemoryBackend


@pytest.mark.integration
class TestARCPerformance:
    """Integration tests for ARC strategy performance."""

    @pytest.fixture
    def cache_capacity(self):
        """Cache capacity for performance tests."""
        return 100

    @pytest.fixture
    def arc_cache(self, cache_capacity):
        """Create ARC-based cache for testing."""
        backend = MemoryBackend()
        strategy = ARCStrategy(capacity=cache_capacity)
        return Cache(backend=backend, strategy=strategy, name="arc_test_cache")

    @pytest.fixture
    def lru_cache(self, cache_capacity):
        """Create LRU-based cache for testing."""
        backend = MemoryBackend()
        strategy = LRUStrategy(capacity=cache_capacity)
        return Cache(backend=backend, strategy=strategy, name="lru_test_cache")

    def generate_recent_access_pattern(self, num_operations: int = 500) -> List[str]:
        """Generate access pattern favoring recent items."""
        keys = [f"recent_key_{i}" for i in range(50)]
        pattern = []

        for _ in range(num_operations):
            # 80% chance of accessing recent 20% of keys
            if random.random() < 0.8:
                key = random.choice(keys[:10])  # Recent 20%
            else:
                key = random.choice(keys)  # Any key

            pattern.append(key)

        return pattern

    def generate_frequent_access_pattern(self, num_operations: int = 500) -> List[str]:
        """Generate access pattern favoring frequently accessed items."""
        keys = [f"freq_key_{i}" for i in range(50)]
        pattern = []

        # Create frequency distribution (Zipf-like)
        weights = [1.0 / (i + 1) for i in range(len(keys))]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        for _ in range(num_operations):
            key = random.choices(keys, weights=probabilities)[0]
            pattern.append(key)

        return pattern

    def generate_mixed_access_pattern(self, num_operations: int = 1000) -> List[str]:
        """Generate mixed access pattern (recent + frequent)."""
        recent_pattern = self.generate_recent_access_pattern(num_operations // 2)
        frequent_pattern = self.generate_frequent_access_pattern(num_operations // 2)

        # Interleave patterns
        mixed_pattern = []
        for i in range(max(len(recent_pattern), len(frequent_pattern))):
            if i < len(recent_pattern):
                mixed_pattern.append(recent_pattern[i])
            if i < len(frequent_pattern):
                mixed_pattern.append(frequent_pattern[i])

        return mixed_pattern

    async def run_workload(self, cache: Cache, access_pattern: List[str]) -> Dict[str, float]:
        """Run workload on cache and collect performance metrics."""
        hits = 0
        misses = 0
        start_time = time.time()

        for key in access_pattern:
            # Try to get from cache
            value = await cache.get(key)

            if value is not None:
                hits += 1
            else:
                misses += 1
                # Cache miss - simulate data retrieval and store
                await cache.set(key, f"value_for_{key}")

        end_time = time.time()

        total_operations = hits + misses
        hit_ratio = hits / total_operations if total_operations > 0 else 0
        throughput = total_operations / (end_time - start_time)

        return {
            "hits": hits,
            "misses": misses,
            "hit_ratio": hit_ratio,
            "throughput": throughput,
            "total_time": end_time - start_time
        }

    @pytest.mark.benchmark
    async def test_arc_vs_lru_recent_pattern(self, arc_cache, lru_cache):
        """Test ARC vs LRU on recent access pattern."""
        pattern = self.generate_recent_access_pattern(1000)

        # Run workloads
        arc_metrics = await self.run_workload(arc_cache, pattern)
        lru_metrics = await self.run_workload(lru_cache, pattern)

        print(f"Recent Pattern Results:")
        print(f"ARC Hit Ratio: {arc_metrics['hit_ratio']:.3f}")
        print(f"LRU Hit Ratio: {lru_metrics['hit_ratio']:.3f}")

        # ARC should perform at least as well as LRU for recent patterns
        assert arc_metrics['hit_ratio'] >= lru_metrics['hit_ratio'] * 0.95

    @pytest.mark.benchmark
    async def test_arc_vs_lru_frequent_pattern(self, arc_cache, lru_cache):
        """Test ARC vs LRU on frequent access pattern."""
        pattern = self.generate_frequent_access_pattern(1000)

        arc_metrics = await self.run_workload(arc_cache, pattern)
        lru_metrics = await self.run_workload(lru_cache, pattern)

        print(f"Frequent Pattern Results:")
        print(f"ARC Hit Ratio: {arc_metrics['hit_ratio']:.3f}")
        print(f"LRU Hit Ratio: {lru_metrics['hit_ratio']:.3f}")

        # ARC should significantly outperform LRU for frequent patterns
        assert arc_metrics['hit_ratio'] > lru_metrics['hit_ratio']

    @pytest.mark.benchmark
    async def test_arc_vs_lru_mixed_pattern(self, arc_cache, lru_cache):
        """Test ARC vs LRU on mixed access pattern (main test)."""
        pattern = self.generate_mixed_access_pattern(2000)

        arc_metrics = await self.run_workload(arc_cache, pattern)
        lru_metrics = await self.run_workload(lru_cache, pattern)

        print(f"Mixed Pattern Results:")
        print(f"ARC Hit Ratio: {arc_metrics['hit_ratio']:.3f}")
        print(f"LRU Hit Ratio: {lru_metrics['hit_ratio']:.3f}")
        print(f"ARC Improvement: {((arc_metrics['hit_ratio'] / lru_metrics['hit_ratio']) - 1) * 100:.1f}%")

        # ARC should provide >10% improvement as specified in tasks.md
        improvement_ratio = arc_metrics['hit_ratio'] / lru_metrics['hit_ratio']
        assert improvement_ratio > 1.10, f"ARC improvement {improvement_ratio:.3f} < required 1.10"

    async def test_arc_adaptation_behavior(self, arc_cache):
        """Test that ARC adapts its behavior to access patterns."""
        # Get initial adaptive parameter
        initial_p = arc_cache.strategy.p

        # Phase 1: Recent access pattern
        recent_pattern = self.generate_recent_access_pattern(500)
        await self.run_workload(arc_cache, recent_pattern)
        p_after_recent = arc_cache.strategy.p

        # Phase 2: Frequent access pattern
        frequent_pattern = self.generate_frequent_access_pattern(500)
        await self.run_workload(arc_cache, frequent_pattern)
        p_after_frequent = arc_cache.strategy.p

        print(f"ARC Parameter Evolution:")
        print(f"Initial p: {initial_p}")
        print(f"After recent pattern p: {p_after_recent}")
        print(f"After frequent pattern p: {p_after_frequent}")

        # Parameter should adapt to workload changes
        # Exact behavior depends on implementation, but p should change
        assert p_after_recent != initial_p or p_after_frequent != p_after_recent

    async def test_arc_memory_efficiency(self, arc_cache, lru_cache):
        """Test that ARC doesn't use significantly more memory than LRU."""
        # Fill both caches to capacity
        for i in range(100):
            await arc_cache.set(f"key_{i}", f"value_{i}")
            await lru_cache.set(f"key_{i}", f"value_{i}")

        # Check cache sizes
        arc_size = len(arc_cache.backend._cache)
        lru_size = len(lru_cache.backend._cache)

        print(f"ARC Cache Size: {arc_size}")
        print(f"LRU Cache Size: {lru_size}")

        # Both should respect capacity limits
        assert arc_size <= 100
        assert lru_size <= 100

        # ARC should not use significantly more memory
        # (Ghost lists are metadata, not full values)
        assert arc_size <= lru_size * 1.2  # Allow 20% overhead for metadata

    @pytest.mark.benchmark
    async def test_arc_throughput_performance(self, arc_cache, lru_cache):
        """Test that ARC throughput is competitive with LRU."""
        pattern = self.generate_mixed_access_pattern(5000)

        arc_metrics = await self.run_workload(arc_cache, pattern)
        lru_metrics = await self.run_workload(lru_cache, pattern)

        print(f"Throughput Comparison:")
        print(f"ARC Throughput: {arc_metrics['throughput']:.0f} ops/sec")
        print(f"LRU Throughput: {lru_metrics['throughput']:.0f} ops/sec")

        # ARC should not be significantly slower than LRU
        throughput_ratio = arc_metrics['throughput'] / lru_metrics['throughput']
        assert throughput_ratio > 0.8, f"ARC throughput too slow: {throughput_ratio:.3f}"

    async def test_arc_edge_cases(self, arc_cache):
        """Test ARC behavior in edge cases."""
        # Test with very small cache
        small_cache = Cache(
            backend=MemoryBackend(),
            strategy=ARCStrategy(capacity=2),
            name="small_arc_cache"
        )

        # Should handle capacity of 2 gracefully
        await small_cache.set("key1", "value1")
        await small_cache.set("key2", "value2")
        await small_cache.set("key3", "value3")  # Should evict

        assert len(small_cache.backend._cache) <= 2

        # Test with single capacity
        tiny_cache = Cache(
            backend=MemoryBackend(),
            strategy=ARCStrategy(capacity=1),
            name="tiny_arc_cache"
        )

        await tiny_cache.set("key1", "value1")
        assert await tiny_cache.get("key1") == "value1"

        await tiny_cache.set("key2", "value2")
        # key1 should be evicted
        assert await tiny_cache.get("key2") == "value2"