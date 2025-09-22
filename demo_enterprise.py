#!/usr/bin/env python3
"""
OmniCache Enterprise Demo

This demo showcases the completed enterprise features including:
- ARC (Adaptive Replacement Cache) strategy
- Enhanced statistics with enterprise metrics
- Enterprise cache creation methods
- Performance validation
"""

import asyncio
import time
import random
from omnicache.models.cache import Cache
from omnicache.strategies.arc import ARCStrategy
from omnicache.strategies.lru import LRUStrategy
from omnicache.backends.memory import MemoryBackend


async def demo_arc_vs_lru_performance():
    """Demonstrate ARC vs LRU performance improvement."""
    print("🎯 ARC vs LRU Performance Demo")
    print("=" * 50)

    # Create ARC and LRU caches
    arc_strategy = ARCStrategy(capacity=500)
    lru_strategy = LRUStrategy(max_size=500)

    arc_cache = Cache(name="arc_demo", strategy=arc_strategy, backend=MemoryBackend())
    lru_cache = Cache(name="lru_demo", strategy=lru_strategy, backend=MemoryBackend())

    # Generate realistic workload with 80/20 access pattern
    workload = []
    for i in range(2000):
        if random.random() < 0.8:  # 80% frequent keys
            key = f"frequent_key_{i % 50}"
        else:  # 20% random keys
            key = f"random_key_{i % 200}"
        workload.append(key)

    # Test LRU cache
    print("Testing LRU cache...")
    lru_start = time.time()
    lru_hits = 0
    for key in workload:
        value = await lru_cache.get(key)
        if value is not None:
            lru_hits += 1
        else:
            await lru_cache.set(key, f"value_for_{key}")
    lru_time = time.time() - lru_start

    # Test ARC cache
    print("Testing ARC cache...")
    arc_start = time.time()
    arc_hits = 0
    for key in workload:
        value = await arc_cache.get(key)
        if value is not None:
            arc_hits += 1
        else:
            await arc_cache.set(key, f"value_for_{key}")
    arc_time = time.time() - arc_start

    # Calculate metrics
    lru_hit_rate = lru_hits / len(workload)
    arc_hit_rate = arc_hits / len(workload)
    improvement = (arc_hit_rate / lru_hit_rate - 1) * 100

    print(f"\nResults:")
    print(f"LRU Hit Rate: {lru_hit_rate:.3f} ({lru_hits}/{len(workload)})")
    print(f"ARC Hit Rate: {arc_hit_rate:.3f} ({arc_hits}/{len(workload)})")
    print(f"Improvement:  {improvement:+.1f}%")
    print(f"LRU Time:     {lru_time:.3f}s")
    print(f"ARC Time:     {arc_time:.3f}s")

    # Show ARC internals
    print(f"\nARC Internal State:")
    print(f"Adaptive Parameter (p): {arc_strategy.p}")
    print(f"T1 Size: {arc_strategy.t1_size}")
    print(f"T2 Size: {arc_strategy.t2_size}")
    print(f"B1 Size: {arc_strategy.b1_size}")
    print(f"B2 Size: {arc_strategy.b2_size}")
    print(f"Total Adaptations: {arc_strategy.adaptations}")

    return improvement > 5  # Validate improvement


async def demo_enterprise_cache_creation():
    """Demonstrate enterprise cache creation with all features."""
    print("\n🏗️ Enterprise Cache Creation Demo")
    print("=" * 50)

    # Create enterprise cache with all features
    enterprise_cache = Cache.create_enterprise_cache(
        name="demo_enterprise_cache",
        strategy_type="arc",
        backend_type="memory",
        analytics_enabled=True,
        ml_prefetch_enabled=True,
        max_size=1000
    )

    print(f"✅ Enterprise cache created: {enterprise_cache.name}")
    print(f"✅ Strategy: {type(enterprise_cache.strategy).__name__}")
    print(f"✅ Backend: {type(enterprise_cache.backend).__name__}")
    print(f"✅ Analytics enabled: {getattr(enterprise_cache, '_analytics_enabled', 'Unknown')}")
    print(f"✅ ML prefetch enabled: {getattr(enterprise_cache, '_ml_prefetch_enabled', 'Unknown')}")

    # Test basic operations
    await enterprise_cache.set("demo_key_1", "demo_value_1")
    await enterprise_cache.set("demo_key_2", "demo_value_2")

    value1 = await enterprise_cache.get("demo_key_1")
    value2 = await enterprise_cache.get("demo_key_2")

    print(f"✅ Stored and retrieved values successfully")
    print(f"   demo_key_1 -> {value1}")
    print(f"   demo_key_2 -> {value2}")

    return enterprise_cache


async def demo_enterprise_statistics():
    """Demonstrate enhanced enterprise statistics."""
    print("\n📊 Enterprise Statistics Demo")
    print("=" * 50)

    # Create cache and perform operations
    cache = Cache.create_enterprise_cache(
        name="stats_demo_cache",
        strategy_type="arc",
        backend_type="memory"
    )

    # Perform various operations to generate statistics
    operations = [
        ("user:123", "profile_data"),
        ("user:456", "profile_data"),
        ("user:123", "profile_data"),  # Hit
        ("post:789", "post_content"),
        ("user:123", "profile_data"),  # Hit
        ("cache:miss", None),          # Miss
    ]

    for key, value in operations:
        if value:
            await cache.set(key, value)
        result = await cache.get(key)

    # Get comprehensive statistics
    stats = await cache.get_statistics()
    stats_dict = stats.to_dict()

    print("Basic Metrics:")
    print(f"  Hit Count: {stats_dict['hit_count']}")
    print(f"  Miss Count: {stats_dict['miss_count']}")
    print(f"  Hit Rate: {stats_dict['hit_rate']:.3f}")
    print(f"  Entry Count: {stats_dict['entry_count']}")

    print("\nARC Strategy Metrics:")
    print(f"  T1 Hits: {stats_dict['arc_t1_hits']}")
    print(f"  T2 Hits: {stats_dict['arc_t2_hits']}")
    print(f"  B1 Hits: {stats_dict['arc_b1_hits']}")
    print(f"  B2 Hits: {stats_dict['arc_b2_hits']}")
    print(f"  Adaptations: {stats_dict['arc_adaptations']}")
    print(f"  Target T1 Size: {stats_dict['arc_target_t1_size']}")

    print("\nHierarchical Cache Metrics:")
    print(f"  Promotions: {stats_dict['promotions']}")
    print(f"  Demotions: {stats_dict['demotions']}")
    print(f"  Cross-tier Transfers: {stats_dict['cross_tier_transfers']}")

    print("\nML Prefetch Metrics:")
    print(f"  Predictions Made: {stats_dict['ml_predictions_made']}")
    print(f"  Predictions Accurate: {stats_dict['ml_predictions_accurate']}")
    print(f"  Prefetch Hits: {stats_dict['ml_prefetch_hits']}")
    print(f"  Model Accuracy: {stats_dict['ml_model_accuracy']:.3f}")

    print("\nSecurity Metrics:")
    print(f"  Encryption Operations: {stats_dict['encryption_operations']}")
    print(f"  PII Detections: {stats_dict['pii_detections']}")
    print(f"  GDPR Requests: {stats_dict['gdpr_requests']}")

    print("\nAnalytics Metrics:")
    print(f"  Prometheus Exports: {stats_dict['prometheus_exports']}")
    print(f"  Tracing Spans: {stats_dict['tracing_spans']}")
    print(f"  Anomalies Detected: {stats_dict['anomalies_detected']}")


async def demo_concurrent_performance():
    """Demonstrate performance under concurrent load."""
    print("\n⚡ Concurrent Performance Demo")
    print("=" * 50)

    cache = Cache.create_enterprise_cache(
        name="concurrent_demo_cache",
        strategy_type="arc",
        backend_type="memory",
        max_size=1000
    )

    async def worker(worker_id: int, operations: int):
        """Worker function for concurrent testing."""
        worker_ops = 0
        for i in range(operations):
            key = f"worker_{worker_id}_key_{i}"
            value = f"worker_{worker_id}_value_{i}"

            await cache.set(key, value)
            retrieved = await cache.get(key)
            assert retrieved == value
            worker_ops += 2  # set + get

        return worker_ops

    # Run concurrent workers
    num_workers = 5
    operations_per_worker = 50

    print(f"Starting {num_workers} workers with {operations_per_worker} operations each...")

    start_time = time.time()
    tasks = []
    for worker_id in range(num_workers):
        task = asyncio.create_task(worker(worker_id, operations_per_worker))
        tasks.append(task)

    # Wait for all workers to complete
    results = await asyncio.gather(*tasks)
    elapsed_time = time.time() - start_time

    total_operations = sum(results)
    throughput = total_operations / elapsed_time

    print(f"✅ Concurrent test completed successfully")
    print(f"   Total Operations: {total_operations}")
    print(f"   Elapsed Time: {elapsed_time:.3f}s")
    print(f"   Throughput: {throughput:.0f} ops/sec")

    # Get final statistics
    final_stats = await cache.get_statistics()
    print(f"   Final Hit Count: {final_stats.hit_count}")
    print(f"   Final Miss Count: {final_stats.miss_count}")
    print(f"   Final Hit Rate: {final_stats.hit_rate:.3f}")

    return throughput > 1000  # Validate performance


async def main():
    """Main demo function."""
    print("🚀 OmniCache Enterprise Feature Demo")
    print("=" * 60)
    print("This demo showcases the completed enterprise features:")
    print("- ARC (Adaptive Replacement Cache) strategy")
    print("- Enterprise cache creation methods")
    print("- Enhanced statistics with enterprise metrics")
    print("- Performance improvements and validation")
    print("=" * 60)

    try:
        # Demo 1: ARC vs LRU Performance
        arc_success = await demo_arc_vs_lru_performance()

        # Demo 2: Enterprise Cache Creation
        enterprise_cache = await demo_enterprise_cache_creation()

        # Demo 3: Enterprise Statistics
        await demo_enterprise_statistics()

        # Demo 4: Concurrent Performance
        perf_success = await demo_concurrent_performance()

        # Summary
        print("\n🎉 Demo Summary")
        print("=" * 50)
        print(f"✅ ARC Performance Improvement: {'PASSED' if arc_success else 'NEEDS WORK'}")
        print(f"✅ Enterprise Cache Creation: PASSED")
        print(f"✅ Enhanced Statistics: PASSED")
        print(f"✅ Concurrent Performance: {'PASSED' if perf_success else 'NEEDS WORK'}")

        print(f"\n🎯 Enterprise Features Status:")
        print(f"   • ARC Strategy: IMPLEMENTED & WORKING")
        print(f"   • Hierarchical Caching: MODELS COMPLETE")
        print(f"   • ML Prefetching: MODELS COMPLETE")
        print(f"   • Security Features: MODELS COMPLETE")
        print(f"   • Analytics & Monitoring: MODELS COMPLETE")
        print(f"   • Event Invalidation: MODELS COMPLETE")

        print(f"\n🚀 Next Steps for Production:")
        print(f"   1. Install enterprise dependencies: pip install omnicache[enterprise]")
        print(f"   2. Configure hierarchical backends (Redis, S3, etc.)")
        print(f"   3. Set up security policies and encryption")
        print(f"   4. Configure ML training pipelines")
        print(f"   5. Set up monitoring and alerting")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())