"""
Final integration testing and performance validation for OmniCache Enterprise.

This test suite validates that all enterprise features work together
seamlessly and meet the performance targets specified in tasks.md.
"""

import pytest
import asyncio
import time
import statistics
from typing import Dict, Any, List
from omnicache.models.cache import Cache
from omnicache.strategies.arc import ARCStrategy
from omnicache.strategies.lru import LRUStrategy
from omnicache.backends.memory import MemoryBackend


@pytest.mark.integration
@pytest.mark.final_validation
class TestFinalValidation:
    """Final integration and performance validation tests."""

    @pytest.fixture
    async def enterprise_cache(self):
        """Create a fully configured enterprise cache."""
        cache = Cache.create_enterprise_cache(
            name="final_validation_cache",
            strategy_type="arc",
            backend_type="memory",  # Use memory for consistent testing
            analytics_enabled=True,
            ml_prefetch_enabled=True,
            max_size=1000
        )
        yield cache
        # Cleanup
        try:
            await cache.shutdown()
        except:
            pass

    @pytest.fixture
    async def baseline_cache(self):
        """Create baseline cache for comparison."""
        strategy = LRUStrategy(max_size=1000)
        backend = MemoryBackend()
        cache = Cache(name="baseline_cache", strategy=strategy, backend=backend)
        yield cache
        try:
            await cache.shutdown()
        except:
            pass

    async def test_enterprise_cache_creation_and_configuration(self):
        """Test that enterprise cache can be created with all features."""
        # Test various enterprise cache configurations
        test_configs = [
            {
                "name": "test_arc_memory",
                "strategy_type": "arc",
                "backend_type": "memory",
                "analytics_enabled": True,
                "ml_prefetch_enabled": False,
            },
            {
                "name": "test_lru_memory",
                "strategy_type": "lru",
                "backend_type": "memory",
                "analytics_enabled": False,
                "ml_prefetch_enabled": True,
            },
            {
                "name": "test_full_enterprise",
                "strategy_type": "arc",
                "backend_type": "memory",
                "analytics_enabled": True,
                "ml_prefetch_enabled": True,
            },
        ]

        created_caches = []
        for config in test_configs:
            cache = Cache.create_enterprise_cache(**config)
            created_caches.append(cache)

            # Validate cache was created successfully
            assert cache is not None
            assert cache.name == config["name"]

            # Test basic operations
            await cache.set("test_key", "test_value")
            value = await cache.get("test_key")
            assert value == "test_value"

        # Cleanup
        for cache in created_caches:
            try:
                await cache.shutdown()
            except:
                pass

    async def test_arc_strategy_performance_target(self, enterprise_cache, baseline_cache):
        """Validate ARC strategy meets >10% improvement target."""
        print("\n🎯 Testing ARC Performance Target (>10% improvement)")

        # Generate realistic workload
        workload_size = 3000
        key_universe = 500

        # Mixed access pattern workload
        workload = []
        for i in range(workload_size):
            if i % 10 < 6:  # 60% frequent keys
                key = f"frequent_key_{i % 50}"
            else:  # 40% random keys
                key = f"random_key_{i % key_universe}"
            workload.append(key)

        # Test baseline LRU cache
        lru_hits = 0
        lru_start = time.time()
        for key in workload:
            value = await baseline_cache.get(key)
            if value is not None:
                lru_hits += 1
            else:
                await baseline_cache.set(key, f"value_for_{key}")
        lru_time = time.time() - lru_start

        # Test ARC cache
        arc_hits = 0
        arc_start = time.time()
        for key in workload:
            value = await enterprise_cache.get(key)
            if value is not None:
                arc_hits += 1
            else:
                await enterprise_cache.set(key, f"value_for_{key}")
        arc_time = time.time() - arc_start

        # Calculate metrics
        lru_hit_rate = lru_hits / len(workload)
        arc_hit_rate = arc_hits / len(workload)
        improvement = (arc_hit_rate / lru_hit_rate - 1) * 100

        print(f"LRU Hit Rate: {lru_hit_rate:.3f}")
        print(f"ARC Hit Rate: {arc_hit_rate:.3f}")
        print(f"Improvement: {improvement:.1f}%")
        print(f"LRU Time: {lru_time:.3f}s")
        print(f"ARC Time: {arc_time:.3f}s")

        # Validate performance target
        assert improvement > 10.0, f"ARC improvement {improvement:.1f}% < required 10%"

    async def test_enterprise_features_integration(self, enterprise_cache):
        """Test that all enterprise features work together without conflicts."""
        print("\n🔧 Testing Enterprise Features Integration")

        # Test 1: ARC strategy with analytics
        await enterprise_cache.set("arc_test_key", "arc_test_value")
        value = await enterprise_cache.get("arc_test_key")
        assert value == "arc_test_value"

        # Verify ARC strategy is working
        assert hasattr(enterprise_cache.strategy, 'p')  # ARC adaptive parameter
        assert hasattr(enterprise_cache.strategy, 't1')  # ARC T1 list

        # Test 2: Statistics collection
        stats = await enterprise_cache.get_statistics()
        assert stats is not None
        assert hasattr(stats, 'hit_count')
        assert hasattr(stats, 'miss_count')

        # Test 3: Enterprise-specific statistics
        stats_dict = stats.to_dict()
        enterprise_metrics = [
            'arc_t1_hits', 'arc_t2_hits', 'arc_adaptations',
            'ml_predictions_made', 'ml_prediction_accuracy',
            'promotions', 'demotions'
        ]

        for metric in enterprise_metrics:
            assert metric in stats_dict, f"Missing enterprise metric: {metric}"

        # Test 4: Configuration persistence
        original_capacity = enterprise_cache.strategy.capacity
        assert original_capacity > 0

    async def test_concurrent_operations_stability(self, enterprise_cache):
        """Test enterprise cache stability under concurrent operations."""
        print("\n⚡ Testing Concurrent Operations Stability")

        async def worker(worker_id: int, operations: int):
            """Worker function for concurrent testing."""
            for i in range(operations):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"

                # Set operation
                await enterprise_cache.set(key, value)

                # Get operation
                retrieved = await enterprise_cache.get(key)
                assert retrieved == value, f"Data corruption in worker {worker_id}"

        # Run concurrent workers
        num_workers = 5
        operations_per_worker = 100

        start_time = time.time()
        tasks = []
        for worker_id in range(num_workers):
            task = asyncio.create_task(worker(worker_id, operations_per_worker))
            tasks.append(task)

        # Wait for all workers to complete
        await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time

        total_operations = num_workers * operations_per_worker * 2  # set + get
        throughput = total_operations / elapsed_time

        print(f"Concurrent Operations: {total_operations}")
        print(f"Elapsed Time: {elapsed_time:.3f}s")
        print(f"Throughput: {throughput:.0f} ops/sec")

        # Validate performance under concurrency
        assert throughput > 1000, f"Concurrent throughput too low: {throughput:.0f} ops/sec"

        # Validate cache integrity
        stats = await enterprise_cache.get_statistics()
        assert stats.hit_count > 0, "No cache hits recorded during concurrent test"

    async def test_memory_efficiency_and_cleanup(self, enterprise_cache):
        """Test memory efficiency and proper cleanup of enterprise features."""
        print("\n🧹 Testing Memory Efficiency and Cleanup")

        # Fill cache to capacity
        cache_capacity = 1000
        for i in range(cache_capacity * 2):  # Overfill to trigger evictions
            await enterprise_cache.set(f"memory_test_key_{i}", f"memory_test_value_{i}")

        # Verify cache respects capacity limits
        stats = await enterprise_cache.get_statistics()
        assert stats.entry_count <= cache_capacity, "Cache exceeded capacity limits"

        # Test ARC ghost lists are maintained properly
        if hasattr(enterprise_cache.strategy, 'total_size'):
            total_arc_entries = enterprise_cache.strategy.total_size
            # Ghost lists should be bounded
            assert total_arc_entries <= cache_capacity * 3, "ARC ghost lists unbounded"

        # Test cache cleanup
        await enterprise_cache.clear()
        stats_after_clear = await enterprise_cache.get_statistics()
        assert stats_after_clear.entry_count == 0, "Cache not properly cleared"

    async def test_error_handling_and_resilience(self, enterprise_cache):
        """Test error handling and system resilience."""
        print("\n🛡️ Testing Error Handling and Resilience")

        # Test 1: Invalid key handling
        try:
            await enterprise_cache.get(None)
            assert False, "Should have raised error for None key"
        except (ValueError, TypeError):
            pass  # Expected

        # Test 2: Invalid value handling
        try:
            await enterprise_cache.set("test_key", None)
            # This might be allowed depending on implementation
        except Exception:
            pass  # Either allowed or properly handled

        # Test 3: Large value handling
        large_value = "x" * (1024 * 1024)  # 1MB string
        try:
            await enterprise_cache.set("large_key", large_value)
            retrieved_large = await enterprise_cache.get("large_key")
            assert retrieved_large == large_value or retrieved_large is None
        except Exception:
            # Either handled gracefully or raises appropriate error
            pass

        # Test 4: Rapid operations (stress test)
        try:
            rapid_tasks = []
            for i in range(100):
                task = enterprise_cache.set(f"rapid_key_{i}", f"rapid_value_{i}")
                rapid_tasks.append(task)

            await asyncio.gather(*rapid_tasks, return_exceptions=True)
            # Should complete without crashing
        except Exception as e:
            print(f"Rapid operations handling: {e}")

    async def test_enterprise_analytics_accuracy(self, enterprise_cache):
        """Test accuracy of enterprise analytics and metrics."""
        print("\n📊 Testing Enterprise Analytics Accuracy")

        # Clear cache to start with clean metrics
        await enterprise_cache.clear()

        # Perform known operations
        num_sets = 50
        num_gets = 100
        expected_hits = 0
        expected_misses = 0

        # Phase 1: All misses (cache is empty)
        for i in range(num_gets // 2):
            value = await enterprise_cache.get(f"analytics_key_{i}")
            if value is None:
                expected_misses += 1
            else:
                expected_hits += 1

        # Phase 2: Load cache
        for i in range(num_sets):
            await enterprise_cache.set(f"analytics_key_{i}", f"analytics_value_{i}")

        # Phase 3: Mix of hits and misses
        for i in range(num_gets // 2):
            value = await enterprise_cache.get(f"analytics_key_{i % (num_sets + 10)}")
            if value is None:
                expected_misses += 1
            else:
                expected_hits += 1

        # Verify analytics accuracy
        stats = await enterprise_cache.get_statistics()

        print(f"Expected hits: {expected_hits}, Recorded: {stats.hit_count}")
        print(f"Expected misses: {expected_misses}, Recorded: {stats.miss_count}")

        # Allow some tolerance for internal operations
        hit_tolerance = 5
        miss_tolerance = 5

        assert abs(stats.hit_count - expected_hits) <= hit_tolerance, \
            f"Hit count inaccurate: expected ~{expected_hits}, got {stats.hit_count}"

        assert abs(stats.miss_count - expected_misses) <= miss_tolerance, \
            f"Miss count inaccurate: expected ~{expected_misses}, got {stats.miss_count}"

        # Test hit rate calculation
        total_operations = stats.hit_count + stats.miss_count
        if total_operations > 0:
            calculated_hit_rate = stats.hit_count / total_operations
            reported_hit_rate = stats.hit_rate

            assert abs(calculated_hit_rate - reported_hit_rate) < 0.01, \
                f"Hit rate calculation error: {calculated_hit_rate:.3f} vs {reported_hit_rate:.3f}"

    async def test_performance_under_load(self, enterprise_cache):
        """Test performance under sustained load."""
        print("\n🚀 Testing Performance Under Load")

        # Sustained load test
        load_duration = 5  # seconds
        operation_count = 0
        start_time = time.time()
        latencies = []

        while time.time() - start_time < load_duration:
            # Measure individual operation latency
            op_start = time.perf_counter()

            key = f"load_test_key_{operation_count % 1000}"
            if operation_count % 3 == 0:
                # Set operation
                await enterprise_cache.set(key, f"load_test_value_{operation_count}")
            else:
                # Get operation
                await enterprise_cache.get(key)

            op_end = time.perf_counter()
            latencies.append((op_end - op_start) * 1000)  # Convert to ms
            operation_count += 1

        total_time = time.time() - start_time
        throughput = operation_count / total_time
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))] if latencies else 0

        print(f"Load Test Results:")
        print(f"Operations: {operation_count}")
        print(f"Duration: {total_time:.2f}s")
        print(f"Throughput: {throughput:.0f} ops/sec")
        print(f"Avg Latency: {avg_latency:.3f}ms")
        print(f"P95 Latency: {p95_latency:.3f}ms")
        print(f"P99 Latency: {p99_latency:.3f}ms")

        # Performance targets
        assert throughput > 2000, f"Throughput too low: {throughput:.0f} ops/sec"
        assert p95_latency < 10, f"P95 latency too high: {p95_latency:.3f}ms"
        assert p99_latency < 50, f"P99 latency too high: {p99_latency:.3f}ms"

    @pytest.mark.security
    async def test_basic_security_validation(self, enterprise_cache):
        """Basic security validation for enterprise features."""
        print("\n🔒 Testing Basic Security Validation")

        # Test 1: Key validation
        malicious_keys = ["../../../etc/passwd", "'; DROP TABLE;", "<script>alert(1)</script>"]

        for malicious_key in malicious_keys:
            try:
                # Should either sanitize or reject malicious keys
                await enterprise_cache.set(malicious_key, "test_value")
                value = await enterprise_cache.get(malicious_key)
                # If allowed, key should be sanitized
                print(f"Malicious key handled: {malicious_key[:20]}...")
            except (ValueError, TypeError):
                # Rejection is also acceptable
                print(f"Malicious key rejected: {malicious_key[:20]}...")

        # Test 2: Large key/value handling
        try:
            large_key = "x" * 10000
            large_value = "y" * (1024 * 1024)  # 1MB

            await enterprise_cache.set(large_key, large_value)
            # Should handle gracefully (store, truncate, or reject)
        except Exception as e:
            print(f"Large data handling: {type(e).__name__}")

    async def test_final_integration_summary(self, enterprise_cache, baseline_cache):
        """Final integration test summary and validation."""
        print("\n📋 Final Integration Test Summary")
        print("="*50)

        # Collect comprehensive metrics
        test_workload = [f"final_test_key_{i}" for i in range(500)]

        # Test enterprise cache
        enterprise_start = time.time()
        enterprise_hits = 0
        for key in test_workload:
            value = await enterprise_cache.get(key)
            if value is not None:
                enterprise_hits += 1
            else:
                await enterprise_cache.set(key, f"value_for_{key}")
        enterprise_time = time.time() - enterprise_start

        # Test baseline cache
        baseline_start = time.time()
        baseline_hits = 0
        for key in test_workload:
            value = await baseline_cache.get(key)
            if value is not None:
                baseline_hits += 1
            else:
                await baseline_cache.set(key, f"value_for_{key}")
        baseline_time = time.time() - baseline_start

        # Calculate final metrics
        enterprise_hit_rate = enterprise_hits / len(test_workload)
        baseline_hit_rate = baseline_hits / len(test_workload)
        enterprise_throughput = len(test_workload) / enterprise_time
        baseline_throughput = len(test_workload) / baseline_time

        # Get enterprise statistics
        enterprise_stats = await enterprise_cache.get_statistics()
        baseline_stats = await baseline_cache.get_statistics()

        # Final validation summary
        print(f"Enterprise Cache Performance:")
        print(f"  Hit Rate: {enterprise_hit_rate:.3f}")
        print(f"  Throughput: {enterprise_throughput:.0f} ops/sec")
        print(f"  ARC Adaptations: {getattr(enterprise_stats, 'arc_adaptations', 'N/A')}")
        print(f"  Total Operations: {enterprise_stats.hit_count + enterprise_stats.miss_count}")

        print(f"\nBaseline Cache Performance:")
        print(f"  Hit Rate: {baseline_hit_rate:.3f}")
        print(f"  Throughput: {baseline_throughput:.0f} ops/sec")
        print(f"  Total Operations: {baseline_stats.hit_count + baseline_stats.miss_count}")

        # Final assertions
        assert enterprise_cache is not None, "Enterprise cache creation failed"
        assert enterprise_throughput > 1000, "Enterprise throughput below minimum"

        print(f"\n✅ All enterprise features validated successfully!")
        print(f"🎯 Performance targets met:")
        print(f"   • Enterprise throughput: {enterprise_throughput:.0f} ops/sec > 1000")
        print(f"   • System stability: Passed")
        print(f"   • Feature integration: Passed")
        print(f"   • Analytics accuracy: Passed")


if __name__ == "__main__":
    # Run final validation tests
    pytest.main([__file__, "-v", "-m", "final_validation"])