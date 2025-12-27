# OmniCache Implementation Plan

## Gap Resolution and New Features

**Generated:** 2025-12-27
**Target Version:** 0.2.0

---

## Executive Summary

This plan addresses identified gaps in the OmniCache codebase and introduces new features to enhance reliability, performance, and operability. The implementation is organized into 4 phases with clear dependencies.

---

## Phase 1: Critical Fixes (Priority: HIGH)

### 1.1 Complete Cache Registry Integration

**Problem:** Registry integration is stubbed out - duplicate cache names are not detected.

**Files to Modify:**
- `src/omnicache/models/cache.py` (lines 455-458, 530-533, 615-618)

**Implementation:**
```python
def _check_duplicate_name(self, name: str) -> None:
    """Check if cache name already exists in the global registry."""
    from omnicache.core.registry import registry
    if registry.exists(name):
        raise CacheError(f"Cache with name '{name}' already exists")

def _register_cache(self) -> None:
    """Register cache in global registry."""
    from omnicache.core.registry import registry
    registry.register(self)

def _unregister_cache(self) -> None:
    """Unregister cache from global registry."""
    from omnicache.core.registry import registry
    registry.unregister(self._name)
```

**Complexity:** Low | **Dependencies:** None

---

### 1.2 Add Thread Safety for Multi-Worker Deployments

**Problem:** No locking mechanism for concurrent access in async contexts.

**Files to Modify:**
- `src/omnicache/backends/memory.py`
- `src/omnicache/strategies/arc.py`
- `src/omnicache/core/manager.py`

**Implementation Pattern:**
```python
# In MemoryBackend.__init__
self._lock = asyncio.Lock()

# Wrap operations
async def set(self, key: str, value: Any, ...) -> None:
    async with self._lock:
        # existing implementation
```

**Complexity:** Medium | **Dependencies:** None

---

### 1.3 Fix Memory Leak in TTL Task Management

**Problem:** Completed TTL expiry tasks are not cleaned from `_expiry_tasks` dict.

**Files to Modify:**
- `src/omnicache/backends/memory.py` (lines 118-121)

**Implementation:**
```python
if ttl is not None:
    task = asyncio.create_task(self._expire_after_ttl(key, ttl))
    task.add_done_callback(lambda t, k=key: self._expiry_tasks.pop(k, None))
    self._expiry_tasks[key] = task
```

**Complexity:** Low | **Dependencies:** None

---

## Phase 2: Reliability Features (Priority: HIGH)

### 2.1 Implement Circuit Breaker Pattern

**Problem:** Backend failures cascade without protection.

**Files to Create:**
- `src/omnicache/core/circuit_breaker.py`

**Files to Modify:**
- `src/omnicache/backends/redis.py`
- `src/omnicache/backends/cloud/*.py`

**Key Components:**
```python
class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0

class CircuitBreaker:
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        # Wrap calls with circuit breaker logic
```

**Complexity:** Medium | **Dependencies:** None

---

### 2.2 Add Health Check Endpoints

**Problem:** No standardized health checking for monitoring/orchestration.

**Files to Modify:**
- `src/omnicache/core/manager.py`
- `src/omnicache/integrations/fastapi.py`

**New Methods:**
```python
async def health_check(self, cache_name: Optional[str] = None) -> Dict[str, Any]
async def readiness_check(self) -> Dict[str, Any]
async def liveness_check(self) -> Dict[str, Any]
```

**FastAPI Integration:**
```python
def create_health_router():
    router = APIRouter(prefix="/health")

    @router.get("/live")
    async def liveness(): ...

    @router.get("/ready")
    async def readiness(): ...

    @router.get("/")
    async def health(): ...
```

**Complexity:** Low | **Dependencies:** 2.1 (Circuit Breaker)

---

## Phase 3: Performance Features (Priority: MEDIUM)

### 3.1 Redis Connection Pooling

**Problem:** Single Redis connection limits concurrency.

**Files to Modify:**
- `src/omnicache/backends/redis.py`

**New Parameters:**
```python
def __init__(
    self,
    # ... existing params
    max_connections: int = 10,
    min_connections: int = 1,
    connection_timeout: float = 5.0,
    socket_timeout: float = 5.0,
    retry_on_timeout: bool = True,
):
```

**Complexity:** Medium | **Dependencies:** 1.2 (Thread Safety)

---

### 3.2 Rate Limiting for Cache Operations

**Problem:** No protection against cache abuse.

**Files to Create:**
- `src/omnicache/core/rate_limiter.py`

**Files to Modify:**
- `src/omnicache/models/cache.py`

**Key Components:**
```python
@dataclass
class RateLimitConfig:
    requests_per_second: float = 100.0
    burst_size: int = 200
    enabled: bool = True
    per_key: bool = False

class TokenBucket:
    async def acquire(self, tokens: int = 1) -> Tuple[bool, float]: ...

class RateLimiter:
    async def check_limit(self, key: Optional[str] = None) -> Tuple[bool, float]: ...
```

**Complexity:** Medium | **Dependencies:** None

---

## Phase 4: Operational Features (Priority: MEDIUM)

### 4.1 Prometheus Metrics Export

**Problem:** Statistics exist but not in exportable format.

**Files to Create:**
- `src/omnicache/analytics/prometheus.py`

**Files to Modify:**
- `src/omnicache/integrations/fastapi.py`

**Metrics Defined:**
| Metric | Type | Description |
|--------|------|-------------|
| `omnicache_operations_total` | Counter | Total cache operations |
| `omnicache_hits_total` | Counter | Cache hits |
| `omnicache_misses_total` | Counter | Cache misses |
| `omnicache_entries_current` | Gauge | Current entry count |
| `omnicache_memory_bytes` | Gauge | Memory usage |
| `omnicache_operation_duration_seconds` | Histogram | Operation latency |
| `omnicache_backend_healthy` | Gauge | Backend health status |
| `omnicache_evictions_total` | Counter | Eviction count |

**Complexity:** Medium | **Dependencies:** 2.1 (Circuit Breaker)

---

### 4.2 Cache Warming/Preloading

**Problem:** No mechanism to preload cache on startup.

**Files to Create:**
- `src/omnicache/core/warmer.py`

**Files to Modify:**
- `src/omnicache/core/manager.py`

**Key Components:**
```python
class WarmupStrategy(Enum):
    EAGER = "eager"
    LAZY = "lazy"
    SCHEDULED = "scheduled"
    PROGRESSIVE = "progressive"

class CacheWarmer:
    async def warm_from_dict(self, cache, data, ttl) -> WarmupResult
    async def warm_from_function(self, cache, keys, loader) -> WarmupResult
    async def warm_background(self, cache, data_source) -> None
```

**Complexity:** Medium | **Dependencies:** None

---

### 4.3 Unit Test Suite

**Problem:** Missing unit tests for core components.

**Files to Create:**
- `tests/unit/__init__.py`
- `tests/unit/test_arc_strategy.py`
- `tests/unit/test_lru_strategy.py`
- `tests/unit/test_memory_backend.py`
- `tests/unit/test_registry.py`
- `tests/unit/test_circuit_breaker.py`
- `tests/unit/test_rate_limiter.py`

**Complexity:** Medium | **Dependencies:** All phases

---

## Implementation Schedule

```
Week 1-2: Phase 1 (Critical Fixes)
├── Day 1: Registry Integration
├── Day 2: TTL Memory Leak Fix
├── Day 2-3: Thread Safety (Memory Backend)
└── Day 3-4: Thread Safety (ARC, Manager)

Week 2-3: Phase 2 (Reliability)
├── Day 5-6: Circuit Breaker Implementation
└── Day 7: Health Check Endpoints

Week 3-4: Phase 3 (Performance)
├── Day 8-9: Redis Connection Pooling
└── Day 10: Rate Limiting

Week 4-5: Phase 4 (Operational)
├── Day 11-12: Prometheus Metrics
├── Day 13: Cache Warming
└── Ongoing: Unit Tests
```

---

## Dependency Graph

```
1.1 Registry ─────────────────────────────────────┐
                                                   │
1.2 Thread Safety ──────────┬─────────────────────┤
                            │                      │
1.3 TTL Fix ────────────────┤                      │
                            │                      │
                            ▼                      │
2.1 Circuit Breaker ──────────┬────────────────────┤
                              │                    │
                              ▼                    │
2.2 Health Checks ────────────┤                    │
                              │                    │
                              ▼                    │
3.1 Connection Pool ──────────┤                    │
                              │                    │
3.2 Rate Limiting ────────────┤                    │
                              │                    │
                              ▼                    │
4.1 Prometheus ───────────────┤                    │
                              │                    │
4.2 Cache Warming ────────────┤                    │
                              │                    │
                              ▼                    ▼
                    4.3 Unit Tests ◄───────────────┘
```

---

## Summary Table

| Item | Phase | Complexity | Priority | New Files | Modified Files |
|------|-------|------------|----------|-----------|----------------|
| Registry Integration | 1 | Low | Critical | 0 | 1 |
| Thread Safety | 1 | Medium | Critical | 0 | 3 |
| TTL Memory Leak | 1 | Low | Critical | 0 | 1 |
| Circuit Breaker | 2 | Medium | High | 1 | 2 |
| Health Checks | 2 | Low | High | 0 | 2 |
| Connection Pooling | 3 | Medium | Medium | 0 | 1 |
| Rate Limiting | 3 | Medium | Medium | 1 | 1 |
| Prometheus Metrics | 4 | Medium | Medium | 1 | 2 |
| Cache Warming | 4 | Medium | Medium | 1 | 1 |
| Unit Tests | 4 | Medium | Medium | 7 | 0 |

**Total: 11 new files, 14 file modifications**

---

## Testing Requirements

Each feature requires:
1. **Unit tests** - Test individual components in isolation
2. **Integration tests** - Test component interactions
3. **Stress tests** - Verify behavior under load (especially thread safety, rate limiting)

---

## Breaking Changes

None expected. All changes are additive or fix existing behavior.

---

## Migration Notes

- Existing caches will automatically benefit from thread safety and memory leak fixes
- Circuit breaker and rate limiting are opt-in via configuration
- Prometheus metrics require `prometheus-client` package (already in enterprise extras)
