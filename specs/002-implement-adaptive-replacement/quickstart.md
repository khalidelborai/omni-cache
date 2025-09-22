# Quickstart: Advanced Cache Strategies and Enterprise Features

**Date**: 2025-01-22
**Feature**: Advanced Cache Strategies and Enterprise Features
**Phase**: 1 - Quickstart Guide

This quickstart guide demonstrates how to use the six advanced caching features that enhance OmniCache for enterprise environments.

## Prerequisites

- OmniCache v1.0+ installed
- Python 3.11+
- Optional: Redis server for L2 tier
- Optional: Cloud storage credentials for L3 tier

## Quick Setup

### 1. Enable ARC Strategy

The Adaptive Replacement Cache automatically balances between recency (LRU) and frequency (LFU):

```python
from omnicache import create_cache

# Create cache with ARC strategy
cache = await create_cache(
    name="smart_cache",
    strategy="arc",
    max_size=10000
)

# ARC automatically adapts - no manual tuning needed!
await cache.set("user:1", {"name": "John", "role": "admin"})
await cache.set("user:2", {"name": "Jane", "role": "user"})

# Check ARC performance metrics
stats = await cache.get_statistics()
print(f"ARC adaptation factor: {stats.arc_p_value}")
print(f"T1 (recent) hits: {stats.t1_hits}")
print(f"T2 (frequent) hits: {stats.t2_hits}")
```

### 2. Configure Hierarchical Caching

Set up automatic L1 (Memory) → L2 (Redis) → L3 (S3) tiering:

```python
from omnicache import create_cache
from omnicache.backends.hierarchical import HierarchicalBackend

# Configure three-tier hierarchy
cache = await create_cache(
    name="tiered_cache",
    backend=HierarchicalBackend(
        tiers=[
            {"level": 1, "type": "memory", "capacity_gb": 1},
            {"level": 2, "type": "redis", "capacity_gb": 10, "host": "localhost"},
            {"level": 3, "type": "s3", "capacity_gb": 100, "bucket": "my-cache-bucket"}
        ],
        auto_promotion=True
    )
)

# Data automatically moves between tiers based on access patterns
await cache.set("hot_data", large_dataset)  # Starts in L1
await cache.set("warm_data", medium_dataset)  # May go to L2
await cache.set("cold_data", archive_data)  # Likely goes to L3

# Check tier utilization
tier_stats = await cache.get_tier_metrics()
print(f"L1 utilization: {tier_stats.l1_utilization_percent}%")
print(f"L2 utilization: {tier_stats.l2_utilization_percent}%")
print(f"Monthly cost: ${tier_stats.total_monthly_cost}")
```

### 3. Enable ML Prefetching

Use machine learning to predict and prefetch data before it's requested:

```python
from omnicache.ml import MLPrefetcher

# Enable ML-powered prefetching
cache = await create_cache(
    name="predictive_cache",
    strategy="lru",
    max_size=5000
)

# Configure ML prefetching
await cache.enable_ml_prefetching(
    model_type="lstm",  # or "arima", "transformer"
    prediction_horizon_hours=1.0,
    confidence_threshold=0.7,
    max_prefetch_keys=1000
)

# Normal cache operations automatically train the model
for i in range(1000):
    await cache.get(f"user:{i % 100}")  # Simulate user access patterns

# Check prefetching effectiveness
ml_stats = await cache.get_ml_metrics()
print(f"Cache miss reduction: {ml_stats.miss_reduction_percent}%")
print(f"Prediction accuracy: {ml_stats.model_accuracy}")
print(f"Prefetch hit rate: {ml_stats.prefetch_hit_rate}")
```

### 4. Configure Zero-Trust Security

Automatic encryption and PII detection for enterprise compliance:

```python
from omnicache.security import SecurityPolicy

# Configure comprehensive security
security_policy = SecurityPolicy(
    encryption_enabled=True,
    encryption_algorithm="AES-256-GCM",
    pii_detection_enabled=True,
    compliance_frameworks=["GDPR", "HIPAA"],
    key_rotation_hours=168,  # Weekly rotation
    audit_level="detailed"
)

cache = await create_cache(
    name="secure_cache",
    security_policy=security_policy
)

# Data is automatically encrypted and PII is detected
await cache.set("user_profile", {
    "name": "John Doe",
    "email": "john@example.com",  # Detected as PII, encrypted
    "ssn": "123-45-6789",        # Detected as PII, encrypted
    "preferences": {"theme": "dark"}
})

# GDPR compliance: Right to be forgotten
await cache.forget_user_data(
    subject_id="john@example.com",
    verification_token="user_verified_token"
)

# Check security metrics
security_stats = await cache.get_security_metrics()
print(f"Encrypted entries: {security_stats.encrypted_entries}")
print(f"PII detections: {security_stats.pii_detections}")
print(f"Compliance score: {security_stats.compliance_score}")
```

### 5. Set Up Real-Time Analytics

Comprehensive observability with Prometheus and Grafana:

```python
from omnicache.analytics import AnalyticsDashboard

# Enable analytics collection
cache = await create_cache(
    name="monitored_cache",
    analytics_config={
        "metrics_enabled": True,
        "tracing_enabled": True,
        "sampling_rate": 1.0,
        "custom_dimensions": ["user_type", "region"]
    }
)

# Analytics are automatically collected
dashboard = AnalyticsDashboard(cache)

# View real-time metrics
metrics = await dashboard.get_realtime_metrics(time_range="1h")
print(f"Hit rate: {metrics.hit_rate}")
print(f"P95 latency: {metrics.p95_latency_ms}ms")
print(f"Throughput: {metrics.throughput_ops_sec} ops/sec")

# Configure alerts
await dashboard.configure_alerts([
    {
        "rule_name": "high_latency",
        "metric": "p95_latency_ms",
        "condition": "gt",
        "threshold": 100,
        "severity": "high"
    },
    {
        "rule_name": "low_hit_rate",
        "metric": "hit_rate",
        "condition": "lt",
        "threshold": 0.8,
        "severity": "medium"
    }
])

# Access Grafana dashboard
print(f"Dashboard URL: {dashboard.grafana_url}")
```

### 6. Configure Event-Driven Invalidation

Reactive cache invalidation with dependency tracking:

```python
from omnicache.events import EventInvalidation

# Configure event sources
cache = await create_cache(
    name="reactive_cache",
    event_invalidation=EventInvalidation(
        sources=[
            {
                "source_type": "kafka",
                "connection_config": {
                    "bootstrap_servers": ["localhost:9092"],
                    "topics": ["user_updates", "product_changes"]
                }
            }
        ]
    )
)

# Define cache dependencies
await cache.define_dependencies([
    {
        "dependent_key": "user_profile:*",
        "dependency_keys": ["user_data:*"],
        "dependency_type": "data"
    },
    {
        "dependent_key": "user_recommendations:*",
        "dependency_keys": ["user_profile:*", "product_catalog"],
        "dependency_type": "computed"
    }
])

# Cache entries are automatically invalidated when dependencies change
await cache.set("user_data:123", user_data)
await cache.set("user_profile:123", computed_profile)
await cache.set("user_recommendations:123", recommendations)

# When user_data:123 changes via event, user_profile:123 and
# user_recommendations:123 are automatically invalidated in order

# Check invalidation metrics
invalidation_stats = await cache.get_invalidation_metrics()
print(f"Total invalidations: {invalidation_stats.total_invalidations}")
print(f"Cascading invalidations: {invalidation_stats.cascading_invalidations}")
```

## CLI Integration

All features are accessible via the enhanced CLI:

```bash
# Enable ARC strategy
omnicache cache create smart_cache --strategy arc --max-size 10000

# Configure hierarchical tiers
omnicache tiers configure smart_cache \
  --l1-memory 1GB \
  --l2-redis localhost:6379 10GB \
  --l3-s3 my-bucket 100GB

# Enable ML prefetching
omnicache ml enable smart_cache --model lstm --confidence 0.7

# Configure security
omnicache security configure smart_cache \
  --encryption AES-256-GCM \
  --pii-detection \
  --compliance GDPR,HIPAA

# View analytics
omnicache analytics dashboard smart_cache

# Set up event invalidation
omnicache events configure smart_cache \
  --source kafka://localhost:9092/user_updates
```

## FastAPI Integration

Use advanced features in FastAPI applications:

```python
from fastapi import FastAPI
from omnicache.integrations.fastapi import cache, cache_response, CacheMiddleware

app = FastAPI()

# Add enterprise cache middleware
app.add_middleware(
    CacheMiddleware,
    cache_name="enterprise_cache",
    strategy="arc",
    hierarchical_tiers=True,
    ml_prefetching=True,
    security_enabled=True,
    analytics_enabled=True
)

@app.get("/users/{user_id}")
@cache(cache_name="user_cache", strategy="arc", ttl=3600)
async def get_user(user_id: int):
    # Data automatically encrypted, analytics collected,
    # ML predictions updated, dependencies tracked
    return await fetch_user_data(user_id)
```

## Performance Expectations

With all features enabled, expect:

- **ARC Strategy**: 10-20% better hit rates than LRU/LFU
- **Hierarchical Caching**: 30-50% cost reduction through tier optimization
- **ML Prefetching**: 30-50% cache miss reduction for predictable workloads
- **Security**: <10% performance overhead with hardware encryption
- **Analytics**: <5% performance impact with sampling
- **Event Invalidation**: Sub-second invalidation latency

## Next Steps

1. **Monitor Performance**: Use the analytics dashboard to track improvements
2. **Tune ML Models**: Adjust prediction parameters based on workload
3. **Optimize Costs**: Review tier utilization and adjust capacity
4. **Security Audit**: Validate PII detection and encryption coverage
5. **Scale Out**: Configure distributed cache clusters with shared analytics

This advanced configuration provides enterprise-grade caching with automatic optimization, comprehensive security, and real-time observability - all while maintaining the simplicity of the OmniCache API.