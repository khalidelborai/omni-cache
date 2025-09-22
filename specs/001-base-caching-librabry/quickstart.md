# QuickStart Guide: OmniCache Library

**Feature**: Base Caching Library Architecture
**Phase**: 1 - User Validation
**Date**: 2025-09-22

## Overview

This quickstart guide validates the core user scenarios from the feature specification by providing step-by-step examples that demonstrate the library's capabilities.

## Prerequisites

- Python 3.11+
- Optional: Redis server for distributed caching
- FastAPI application (for web integration scenarios)

## Installation

```bash
pip install omnicache
```

## Scenario 1: Basic Library Integration

**User Story**: Developer integrates caching library with minimal configuration

### Step 1: Create a Basic Cache

```python
from omnicache import Cache

# Create cache with default settings (in-memory, LRU strategy)
cache = Cache(name="my_cache")

# Store and retrieve data
await cache.set("user:123", {"name": "John", "age": 30})
user_data = await cache.get("user:123")
print(user_data)  # {"name": "John", "age": 30}
```

### Step 2: Configure Cache Strategy

```python
from omnicache import Cache, LRUStrategy

# Create cache with specific strategy and limits
cache = Cache(
    name="api_cache",
    strategy=LRUStrategy(max_size=1000),
    default_ttl=300  # 5 minutes
)

# Data expires automatically
await cache.set("temp_data", "expires_soon")
await asyncio.sleep(301)
result = await cache.get("temp_data")  # None (expired)
```

### Step 3: Use Different Storage Backends

```python
from omnicache import Cache, RedisBackend, FileSystemBackend

# Redis backend for persistence
redis_cache = Cache(
    name="persistent_cache",
    backend=RedisBackend(host="localhost", port=6379)
)

# File system backend for local persistence
file_cache = Cache(
    name="file_cache",
    backend=FileSystemBackend(directory="/tmp/cache")
)
```

**Validation**: ✅ Minimal configuration requirement met with sensible defaults

## Scenario 2: FastAPI Integration

**User Story**: Developer adds caching to existing FastAPI application

### Step 1: Function-Level Caching

```python
from fastapi import FastAPI
from omnicache.integrations.fastapi import cache

app = FastAPI()

@app.get("/users/{user_id}")
@cache(cache_name="user_cache", ttl=1800)
async def get_user(user_id: int):
    # Simulated database call
    user_data = await fetch_user_from_database(user_id)
    return user_data

# First call: cache miss, database query
# Subsequent calls: cache hit, no database query
```

### Step 2: Response Caching with Headers

```python
from omnicache.integrations.fastapi import cache_response

@app.get("/products/{product_id}")
@cache_response(ttl=600, vary_on=["product_id"])
async def get_product(product_id: int):
    product = await product_service.get(product_id)
    return product

# Automatic HTTP cache headers added:
# Cache-Control: max-age=600
# ETag: "hash-of-response"
```

### Step 3: Middleware for Automatic Caching

```python
from omnicache.integrations.fastapi import CacheMiddleware

app.add_middleware(
    CacheMiddleware,
    cache_name="http_cache",
    default_ttl=300,
    exclude_paths=["/health", "/admin"]
)

# All GET requests automatically cached
# 304 Not Modified responses for conditional requests
```

**Validation**: ✅ Seamless FastAPI integration with decorators and middleware

## Scenario 3: CLI Management

**User Story**: Administrator uses CLI tools for cache management

### Step 1: Cache Instance Management

```bash
# Create new cache
omnicache cache create web-cache --strategy lru --max-size 10000

# List all caches
omnicache cache list

# Get cache information
omnicache cache info web-cache

# Output:
# Name: web-cache
# Strategy: LRU
# Entries: 1,234 / 10,000
# Hit Rate: 85.6%
# Memory Usage: 45.2 MB
```

### Step 2: Entry Operations

```bash
# Store cache entry
omnicache entry set web-cache "user:456" '{"name":"Alice","role":"admin"}' --ttl 3600

# Retrieve entry
omnicache entry get web-cache "user:456"

# List entries with pattern
omnicache entry list web-cache --pattern "user:*" --limit 50

# Delete specific entry
omnicache entry delete web-cache "user:456"
```

### Step 3: Monitoring and Statistics

```bash
# View real-time statistics
omnicache stats web-cache --watch --interval 5

# Clear cache selectively
omnicache clear web-cache --pattern "temp:*" --force

# Warm cache from file
omnicache warm web-cache --file user-data.json --batch-size 100
```

**Validation**: ✅ Comprehensive CLI tools for operational management

## Scenario 4: Strategy Switching

**User Story**: Developer switches caching strategies without code changes

### Step 1: Runtime Strategy Change

```python
from omnicache import Cache, LRUStrategy, LFUStrategy, TTLStrategy

cache = Cache(name="flexible_cache", strategy=LRUStrategy(max_size=1000))

# Populate cache
for i in range(500):
    await cache.set(f"key:{i}", f"value:{i}")

# Switch to LFU strategy
cache.set_strategy(LFUStrategy(max_size=1000))

# Existing data remains, new eviction behavior applied
await cache.set("new_key", "new_value")
```

### Step 2: Configuration-Based Strategy

```python
# Strategy from configuration
config = {
    "strategy": "ttl",
    "parameters": {"default_ttl": 600, "cleanup_interval": 60}
}

cache = Cache.from_config("dynamic_cache", config)

# Change strategy via configuration update
cache.update_config({"strategy": "size", "parameters": {"max_size": 2000}})
```

**Validation**: ✅ Runtime strategy switching without data loss

## Scenario 5: Custom Extension

**User Story**: Developer creates custom caching strategy

### Step 1: Implement Custom Strategy

```python
from omnicache.strategies import BaseStrategy
from typing import Any, Optional

class PriorityStrategy(BaseStrategy):
    """Evict entries based on priority scores"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size

    def should_evict(self, cache_size: int) -> bool:
        return cache_size >= self.max_size

    def select_eviction_candidates(self, entries) -> list:
        # Evict lowest priority entries first
        return sorted(entries, key=lambda e: e.priority)[:cache_size - self.max_size + 1]
```

### Step 2: Register and Use Custom Strategy

```python
from omnicache import StrategyRegistry

# Register custom strategy
StrategyRegistry.register("priority", PriorityStrategy)

# Use custom strategy
cache = Cache(
    name="priority_cache",
    strategy=PriorityStrategy(max_size=500)
)

# Store entries with priority
await cache.set("important", "critical_data", priority=1.0)
await cache.set("normal", "regular_data", priority=0.5)
await cache.set("temporary", "temp_data", priority=0.1)
```

**Validation**: ✅ Extensible architecture supports custom strategies

## Integration Testing Scenarios

### Scenario A: High-Load Performance

```python
import asyncio
import time

async def performance_test():
    cache = Cache("perf_test", strategy=LRUStrategy(max_size=10000))

    start_time = time.time()

    # Concurrent operations
    tasks = []
    for i in range(1000):
        tasks.append(cache.set(f"key:{i}", f"value:{i}"))

    await asyncio.gather(*tasks)

    # Measure get performance
    get_tasks = []
    for i in range(1000):
        get_tasks.append(cache.get(f"key:{i}"))

    results = await asyncio.gather(*get_tasks)

    duration = time.time() - start_time
    print(f"1000 set + 1000 get operations: {duration:.3f}s")
    # Expected: < 100ms for memory backend

# Validation: ✅ Sub-millisecond operations under load
```

### Scenario B: Backend Failover

```python
async def failover_test():
    # Primary Redis backend with file system fallback
    cache = Cache(
        name="resilient_cache",
        backend=RedisBackend("localhost:6379"),
        fallback_backend=FileSystemBackend("/tmp/cache_fallback")
    )

    # Store data
    await cache.set("test_key", "test_value")

    # Simulate Redis failure
    # (Redis process stops)

    # Should automatically fallback to file system
    value = await cache.get("test_key")  # Still works

    # New data goes to fallback
    await cache.set("failover_key", "failover_value")

# Validation: ✅ Graceful degradation on backend failures
```

### Scenario C: Multi-Tenant Usage

```python
async def multi_tenant_test():
    # Shared cache with namespaces
    tenant_a_cache = Cache("shared", namespace="tenant_a")
    tenant_b_cache = Cache("shared", namespace="tenant_b")

    # Isolated data
    await tenant_a_cache.set("user:1", "Alice")
    await tenant_b_cache.set("user:1", "Bob")

    # No data leakage
    assert await tenant_a_cache.get("user:1") == "Alice"
    assert await tenant_b_cache.get("user:1") == "Bob"

    # Tenant-specific statistics
    a_stats = await tenant_a_cache.get_statistics()
    b_stats = await tenant_b_cache.get_statistics()

# Validation: ✅ Namespace isolation for multi-tenant scenarios
```

## Expected Outcomes

After completing this quickstart:

1. **Basic Integration**: ✅ Library works with minimal configuration
2. **FastAPI Integration**: ✅ Decorators and middleware provide transparent caching
3. **CLI Management**: ✅ Administrative operations available via command line
4. **Strategy Flexibility**: ✅ Runtime strategy changes without code modification
5. **Extensibility**: ✅ Custom strategies integrate seamlessly
6. **Performance**: ✅ Sub-millisecond operations for in-memory backend
7. **Reliability**: ✅ Graceful degradation on backend failures
8. **Multi-tenancy**: ✅ Namespace isolation prevents data leakage

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure `pip install omnicache` completed successfully
2. **Redis Connection**: Verify Redis server is running for Redis backend tests
3. **Permission Error**: Check file system permissions for FileSystemBackend
4. **Memory Usage**: Monitor memory with large datasets, use size limits

### Validation Commands

```bash
# Verify CLI installation
omnicache --version

# Check available strategies
omnicache cache create test --strategy invalid  # Should show available options

# Validate configuration
omnicache cache info non-existent  # Should show clear error message
```

## Next Steps

After validating these scenarios:

1. **Phase 2**: Generate detailed implementation tasks
2. **TDD Implementation**: Write tests that validate these scenarios
3. **Performance Benchmarking**: Implement performance requirements
4. **Production Deployment**: Configure for production environments

This quickstart serves as the acceptance criteria for the implementation phase.