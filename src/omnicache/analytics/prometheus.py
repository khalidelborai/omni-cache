"""
Prometheus metrics exporter for OmniCache.

Provides standardized metrics for monitoring cache performance,
backend health, and operational statistics.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import time
from functools import wraps

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Fallback stubs
    Counter = None
    Gauge = None
    Histogram = None
    Summary = None
    CollectorRegistry = None


class PrometheusMetrics:
    """
    Prometheus metrics collector for OmniCache.

    Provides standard metrics for:
    - Cache operations (hits, misses, sets, deletes)
    - Latency measurements
    - Cache size and memory usage
    - Backend health status
    - Circuit breaker state
    - Rate limiting statistics
    """

    def __init__(
        self,
        prefix: str = "omnicache",
        registry: Optional[Any] = None,
        include_default_labels: bool = True
    ):
        """
        Initialize Prometheus metrics.

        Args:
            prefix: Metric name prefix
            registry: Custom CollectorRegistry (creates new if None)
            include_default_labels: Include default labels like hostname

        Raises:
            ImportError: If prometheus_client is not installed
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client not installed. "
                "Install with: pip install prometheus-client"
            )

        self.prefix = prefix
        self.registry = registry or CollectorRegistry()

        # Initialize all metrics
        self._init_operation_metrics()
        self._init_performance_metrics()
        self._init_size_metrics()
        self._init_health_metrics()
        self._init_rate_limit_metrics()

    def _init_operation_metrics(self) -> None:
        """Initialize cache operation metrics."""
        # Total operations counter
        self.operations_total = Counter(
            f"{self.prefix}_operations_total",
            "Total cache operations",
            ["cache_name", "operation", "status"],
            registry=self.registry
        )

        # Cache hits counter
        self.hits_total = Counter(
            f"{self.prefix}_hits_total",
            "Total cache hits",
            ["cache_name"],
            registry=self.registry
        )

        # Cache misses counter
        self.misses_total = Counter(
            f"{self.prefix}_misses_total",
            "Total cache misses",
            ["cache_name"],
            registry=self.registry
        )

        # Evictions counter
        self.evictions_total = Counter(
            f"{self.prefix}_evictions_total",
            "Total cache evictions",
            ["cache_name", "reason"],
            registry=self.registry
        )

    def _init_performance_metrics(self) -> None:
        """Initialize performance metrics."""
        # Operation latency histogram
        self.operation_duration = Histogram(
            f"{self.prefix}_operation_duration_seconds",
            "Operation latency in seconds",
            ["cache_name", "operation"],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )

        # Hit ratio gauge (calculated metric)
        self.hit_ratio = Gauge(
            f"{self.prefix}_hit_ratio",
            "Cache hit ratio (0.0 to 1.0)",
            ["cache_name"],
            registry=self.registry
        )

    def _init_size_metrics(self) -> None:
        """Initialize size and capacity metrics."""
        # Current entry count
        self.entries_current = Gauge(
            f"{self.prefix}_entries_current",
            "Current number of cache entries",
            ["cache_name"],
            registry=self.registry
        )

        # Maximum capacity
        self.entries_max = Gauge(
            f"{self.prefix}_entries_max",
            "Maximum cache capacity",
            ["cache_name"],
            registry=self.registry
        )

        # Memory usage in bytes
        self.memory_bytes = Gauge(
            f"{self.prefix}_memory_bytes",
            "Memory usage in bytes",
            ["cache_name"],
            registry=self.registry
        )

    def _init_health_metrics(self) -> None:
        """Initialize health and status metrics."""
        # Backend health (1 = healthy, 0 = unhealthy)
        self.backend_healthy = Gauge(
            f"{self.prefix}_backend_healthy",
            "Backend health status (1=healthy, 0=unhealthy)",
            ["cache_name", "backend_type"],
            registry=self.registry
        )

        # Circuit breaker state (0=closed, 1=open, 2=half-open)
        self.circuit_breaker_state = Gauge(
            f"{self.prefix}_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half-open)",
            ["cache_name"],
            registry=self.registry
        )

        # Cache status
        self.cache_status = Gauge(
            f"{self.prefix}_cache_status",
            "Cache status (1=active, 0=inactive)",
            ["cache_name", "status"],
            registry=self.registry
        )

    def _init_rate_limit_metrics(self) -> None:
        """Initialize rate limiting metrics."""
        # Rate limit rejections
        self.rate_limit_rejections = Counter(
            f"{self.prefix}_rate_limit_rejections_total",
            "Total rate limit rejections",
            ["cache_name", "limit_type"],
            registry=self.registry
        )

        # Current rate limit tokens
        self.rate_limit_tokens = Gauge(
            f"{self.prefix}_rate_limit_tokens_available",
            "Available rate limit tokens",
            ["cache_name"],
            registry=self.registry
        )

    # Recording methods

    def record_operation(
        self,
        cache_name: str,
        operation: str,
        success: bool,
        duration: float
    ) -> None:
        """
        Record a cache operation.

        Args:
            cache_name: Name of the cache
            operation: Operation type (get, set, delete, clear)
            success: Whether operation succeeded
            duration: Operation duration in seconds
        """
        status = "success" if success else "error"
        self.operations_total.labels(
            cache_name=cache_name,
            operation=operation,
            status=status
        ).inc()
        self.operation_duration.labels(
            cache_name=cache_name,
            operation=operation
        ).observe(duration)

    def record_hit(self, cache_name: str) -> None:
        """Record a cache hit."""
        self.hits_total.labels(cache_name=cache_name).inc()

    def record_miss(self, cache_name: str) -> None:
        """Record a cache miss."""
        self.misses_total.labels(cache_name=cache_name).inc()

    def record_eviction(self, cache_name: str, reason: str = "capacity") -> None:
        """Record a cache eviction."""
        self.evictions_total.labels(cache_name=cache_name, reason=reason).inc()

    def update_cache_size(self, cache_name: str, size: int, max_size: Optional[int] = None) -> None:
        """Update cache size metrics."""
        self.entries_current.labels(cache_name=cache_name).set(size)
        if max_size is not None:
            self.entries_max.labels(cache_name=cache_name).set(max_size)

    def update_memory_usage(self, cache_name: str, bytes_used: int) -> None:
        """Update memory usage metric."""
        self.memory_bytes.labels(cache_name=cache_name).set(bytes_used)

    def update_hit_ratio(self, cache_name: str, ratio: float) -> None:
        """Update hit ratio metric."""
        self.hit_ratio.labels(cache_name=cache_name).set(ratio)

    def update_backend_health(self, cache_name: str, backend_type: str, healthy: bool) -> None:
        """Update backend health status."""
        self.backend_healthy.labels(
            cache_name=cache_name,
            backend_type=backend_type
        ).set(1 if healthy else 0)

    def update_circuit_breaker_state(self, cache_name: str, state: str) -> None:
        """
        Update circuit breaker state.

        Args:
            cache_name: Cache name
            state: Circuit state ("closed", "open", "half_open")
        """
        state_values = {"closed": 0, "open": 1, "half_open": 2}
        self.circuit_breaker_state.labels(
            cache_name=cache_name
        ).set(state_values.get(state, 0))

    def update_cache_status(self, cache_name: str, status: str, active: bool) -> None:
        """Update cache status metric."""
        self.cache_status.labels(
            cache_name=cache_name,
            status=status
        ).set(1 if active else 0)

    def record_rate_limit_rejection(self, cache_name: str, limit_type: str = "global") -> None:
        """Record a rate limit rejection."""
        self.rate_limit_rejections.labels(
            cache_name=cache_name,
            limit_type=limit_type
        ).inc()

    def update_rate_limit_tokens(self, cache_name: str, tokens: float) -> None:
        """Update available rate limit tokens."""
        self.rate_limit_tokens.labels(cache_name=cache_name).set(tokens)

    # Utility methods

    def get_metrics_output(self) -> bytes:
        """Get metrics in Prometheus exposition format."""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get content type for metrics response."""
        return CONTENT_TYPE_LATEST


# Decorator for timing operations
def timed_operation(metrics: PrometheusMetrics, cache_name: str, operation: str):
    """
    Decorator to time cache operations.

    Args:
        metrics: PrometheusMetrics instance
        cache_name: Cache name for labeling
        operation: Operation name

    Usage:
        @timed_operation(metrics, "my_cache", "get")
        async def get_value(key):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            success = True
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                success = False
                raise
            finally:
                duration = time.perf_counter() - start_time
                metrics.record_operation(cache_name, operation, success, duration)
        return wrapper
    return decorator


# Global metrics instance
_metrics: Optional[PrometheusMetrics] = None


def initialize_metrics(prefix: str = "omnicache") -> PrometheusMetrics:
    """
    Initialize global metrics instance.

    Args:
        prefix: Metric name prefix

    Returns:
        PrometheusMetrics instance
    """
    global _metrics
    _metrics = PrometheusMetrics(prefix=prefix)
    return _metrics


def get_metrics() -> Optional[PrometheusMetrics]:
    """Get global metrics instance."""
    return _metrics


def shutdown_metrics() -> None:
    """Shutdown global metrics instance."""
    global _metrics
    _metrics = None
