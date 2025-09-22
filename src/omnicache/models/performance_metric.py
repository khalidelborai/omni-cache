"""
Performance Metric model for analytics and monitoring.

This module defines the performance metric model for collecting, aggregating,
and analyzing cache performance data with various aggregation methods.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import time
import statistics
import json
from collections import defaultdict, deque


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"        # Monotonically increasing values
    GAUGE = "gauge"           # Point-in-time values
    HISTOGRAM = "histogram"   # Distribution of values
    TIMER = "timer"           # Duration measurements
    RATE = "rate"             # Rate of events over time


class AggregationMethod(Enum):
    """Aggregation methods for metrics."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    P95 = "p95"
    P99 = "p99"
    COUNT = "count"
    RATE = "rate"
    STDDEV = "stddev"


class TimeWindow(Enum):
    """Time windows for metric aggregation."""
    MINUTE = 60
    FIVE_MINUTES = 300
    FIFTEEN_MINUTES = 900
    HOUR = 3600
    DAY = 86400
    WEEK = 604800


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate metric point."""
        if self.timestamp <= 0:
            raise ValueError("Timestamp must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricPoint':
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            value=data["value"],
            tags=data.get("tags", {}),
        )


@dataclass
class HistogramBucket:
    """Histogram bucket for distribution metrics."""
    upper_bound: float
    count: int = 0

    def __post_init__(self):
        """Validate bucket."""
        if self.count < 0:
            raise ValueError("Count cannot be negative")


@dataclass
class AggregatedMetric:
    """Aggregated metric result."""
    metric_name: str
    aggregation_method: AggregationMethod
    time_window: TimeWindow
    start_time: float
    end_time: float
    value: Union[int, float]
    sample_count: int
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "aggregation_method": self.aggregation_method.value,
            "time_window": self.time_window.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "value": self.value,
            "sample_count": self.sample_count,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AggregatedMetric':
        """Create from dictionary."""
        return cls(
            metric_name=data["metric_name"],
            aggregation_method=AggregationMethod(data["aggregation_method"]),
            time_window=TimeWindow(data["time_window"]),
            start_time=data["start_time"],
            end_time=data["end_time"],
            value=data["value"],
            sample_count=data["sample_count"],
            tags=data.get("tags", {}),
        )


@dataclass
class PerformanceMetric:
    """
    Performance metric model for analytics and monitoring.

    Collects, stores, and aggregates performance data with support for
    various metric types and aggregation methods.
    """

    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""

    # Data storage
    _data_points: deque = field(default_factory=lambda: deque(maxlen=10000))
    _histogram_buckets: List[HistogramBucket] = field(default_factory=list)

    # Configuration
    retention_seconds: int = 86400  # 24 hours
    max_points: int = 10000
    default_tags: Dict[str, str] = field(default_factory=dict)

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    total_points: int = 0

    def __post_init__(self):
        """Post-initialization setup."""
        if not self.name:
            raise ValueError("Metric name is required")

        if self.retention_seconds <= 0:
            raise ValueError("Retention seconds must be positive")

        if self.max_points <= 0:
            raise ValueError("Max points must be positive")

        # Initialize histogram buckets for histogram metrics
        if self.metric_type == MetricType.HISTOGRAM and not self._histogram_buckets:
            self._initialize_histogram_buckets()

    def _initialize_histogram_buckets(self):
        """Initialize default histogram buckets."""
        # Default buckets: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, +Inf
        bounds = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
        self._histogram_buckets = [HistogramBucket(bound) for bound in bounds]

    @property
    def current_size(self) -> int:
        """Get current number of data points."""
        return len(self._data_points)

    @property
    def is_counter(self) -> bool:
        """Check if this is a counter metric."""
        return self.metric_type == MetricType.COUNTER

    @property
    def is_gauge(self) -> bool:
        """Check if this is a gauge metric."""
        return self.metric_type == MetricType.GAUGE

    @property
    def is_histogram(self) -> bool:
        """Check if this is a histogram metric."""
        return self.metric_type == MetricType.HISTOGRAM

    @property
    def is_timer(self) -> bool:
        """Check if this is a timer metric."""
        return self.metric_type == MetricType.TIMER

    def record(self, value: Union[int, float], timestamp: Optional[float] = None,
              tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric value.

        Args:
            value: Metric value
            timestamp: Optional timestamp (uses current time if None)
            tags: Optional tags for this data point
        """
        if timestamp is None:
            timestamp = time.time()

        # Merge with default tags
        merged_tags = self.default_tags.copy()
        if tags:
            merged_tags.update(tags)

        point = MetricPoint(timestamp=timestamp, value=value, tags=merged_tags)

        # Add to data points
        self._data_points.append(point)
        self.total_points += 1
        self.updated_at = timestamp

        # Update histogram buckets for histogram metrics
        if self.is_histogram:
            self._update_histogram(value)

        # Clean old data
        self._cleanup_old_data()

    def _update_histogram(self, value: float) -> None:
        """Update histogram buckets with new value."""
        for bucket in self._histogram_buckets:
            if value <= bucket.upper_bound:
                bucket.count += 1

    def _cleanup_old_data(self) -> None:
        """Remove data points older than retention period."""
        cutoff_time = time.time() - self.retention_seconds
        while self._data_points and self._data_points[0].timestamp < cutoff_time:
            self._data_points.popleft()

    def get_latest_value(self) -> Optional[Union[int, float]]:
        """Get the most recent metric value."""
        if not self._data_points:
            return None
        return self._data_points[-1].value

    def get_points_in_window(self, start_time: float, end_time: float) -> List[MetricPoint]:
        """Get data points within a time window."""
        return [
            point for point in self._data_points
            if start_time <= point.timestamp <= end_time
        ]

    def aggregate(self, method: AggregationMethod, time_window: TimeWindow,
                 end_time: Optional[float] = None, tags: Optional[Dict[str, str]] = None) -> Optional[AggregatedMetric]:
        """
        Aggregate metric values over a time window.

        Args:
            method: Aggregation method
            time_window: Time window for aggregation
            end_time: End of time window (uses current time if None)
            tags: Optional tag filter

        Returns:
            AggregatedMetric or None if no data
        """
        if end_time is None:
            end_time = time.time()

        start_time = end_time - time_window.value

        # Get points in window
        points = self.get_points_in_window(start_time, end_time)

        # Filter by tags if specified
        if tags:
            points = [
                point for point in points
                if all(point.tags.get(k) == v for k, v in tags.items())
            ]

        if not points:
            return None

        values = [point.value for point in points]
        aggregated_value = self._calculate_aggregation(values, method)

        return AggregatedMetric(
            metric_name=self.name,
            aggregation_method=method,
            time_window=time_window,
            start_time=start_time,
            end_time=end_time,
            value=aggregated_value,
            sample_count=len(values),
            tags=tags or {},
        )

    def _calculate_aggregation(self, values: List[Union[int, float]], method: AggregationMethod) -> Union[int, float]:
        """Calculate aggregated value using specified method."""
        if not values:
            return 0

        if method == AggregationMethod.SUM:
            return sum(values)
        elif method == AggregationMethod.AVERAGE:
            return statistics.mean(values)
        elif method == AggregationMethod.MIN:
            return min(values)
        elif method == AggregationMethod.MAX:
            return max(values)
        elif method == AggregationMethod.MEDIAN:
            return statistics.median(values)
        elif method == AggregationMethod.P95:
            return self._percentile(values, 95)
        elif method == AggregationMethod.P99:
            return self._percentile(values, 99)
        elif method == AggregationMethod.COUNT:
            return len(values)
        elif method == AggregationMethod.RATE:
            # Calculate rate per second
            if len(values) < 2:
                return 0
            time_span = max(1, len(values))  # Approximate time span
            return len(values) / time_span
        elif method == AggregationMethod.STDDEV:
            return statistics.stdev(values) if len(values) > 1 else 0
        else:
            return statistics.mean(values)

    def _percentile(self, values: List[Union[int, float]], percentile: float) -> Union[int, float]:
        """Calculate percentile of values."""
        if not values:
            return 0

        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            if upper_index >= len(sorted_values):
                return sorted_values[lower_index]

            weight = index - lower_index
            return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight

    def get_histogram_data(self) -> List[Dict[str, Any]]:
        """Get histogram bucket data."""
        if not self.is_histogram:
            return []

        return [
            {
                "upper_bound": bucket.upper_bound,
                "count": bucket.count,
                "percentage": bucket.count / self.total_points * 100 if self.total_points > 0 else 0
            }
            for bucket in self._histogram_buckets
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for this metric."""
        if not self._data_points:
            return {
                "count": 0,
                "latest_value": None,
                "statistics": {}
            }

        values = [point.value for point in self._data_points]

        stats = {
            "count": len(values),
            "latest_value": values[-1],
            "statistics": {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stddev": statistics.stdev(values) if len(values) > 1 else 0,
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99),
            }
        }

        if self.is_histogram:
            stats["histogram"] = self.get_histogram_data()

        return stats

    def get_time_series(self, time_window: TimeWindow, method: AggregationMethod = AggregationMethod.AVERAGE) -> List[AggregatedMetric]:
        """
        Get time series data with specified granularity.

        Args:
            time_window: Window size for each data point
            method: Aggregation method for each window

        Returns:
            List of aggregated metrics over time
        """
        if not self._data_points:
            return []

        end_time = time.time()
        start_time = end_time - self.retention_seconds
        window_size = time_window.value

        series = []
        current_time = start_time

        while current_time < end_time:
            window_end = min(current_time + window_size, end_time)
            aggregated = self.aggregate(method, TimeWindow(window_size), window_end)

            if aggregated:
                series.append(aggregated)

            current_time += window_size

        return series

    def reset(self) -> None:
        """Reset all metric data."""
        self._data_points.clear()
        self.total_points = 0
        self.updated_at = time.time()

        if self.is_histogram:
            for bucket in self._histogram_buckets:
                bucket.count = 0

    def merge(self, other: 'PerformanceMetric') -> None:
        """
        Merge data from another metric.

        Args:
            other: Another PerformanceMetric to merge from
        """
        if other.name != self.name or other.metric_type != self.metric_type:
            raise ValueError("Cannot merge metrics with different names or types")

        # Merge data points
        for point in other._data_points:
            self._data_points.append(point)

        # Update counters
        self.total_points += other.total_points

        # Merge histogram buckets
        if self.is_histogram and other.is_histogram:
            for i, other_bucket in enumerate(other._histogram_buckets):
                if i < len(self._histogram_buckets):
                    self._histogram_buckets[i].count += other_bucket.count

        self.updated_at = time.time()
        self._cleanup_old_data()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary representation."""
        return {
            "name": self.name,
            "metric_type": self.metric_type.value,
            "description": self.description,
            "unit": self.unit,
            "retention_seconds": self.retention_seconds,
            "max_points": self.max_points,
            "default_tags": self.default_tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_points": self.total_points,
            "current_size": self.current_size,
            "data_points": [point.to_dict() for point in list(self._data_points)[-100:]],  # Last 100 points
            "histogram_buckets": [
                {"upper_bound": bucket.upper_bound, "count": bucket.count}
                for bucket in self._histogram_buckets
            ] if self.is_histogram else [],
            "statistics": self.get_statistics(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetric':
        """Create metric from dictionary representation."""
        metric = cls(
            name=data["name"],
            metric_type=MetricType(data["metric_type"]),
            description=data.get("description", ""),
            unit=data.get("unit", ""),
            retention_seconds=data.get("retention_seconds", 86400),
            max_points=data.get("max_points", 10000),
            default_tags=data.get("default_tags", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            total_points=data.get("total_points", 0),
        )

        # Add data points
        for point_data in data.get("data_points", []):
            point = MetricPoint.from_dict(point_data)
            metric._data_points.append(point)

        # Add histogram buckets
        if metric.is_histogram:
            bucket_data = data.get("histogram_buckets", [])
            if bucket_data:
                metric._histogram_buckets = [
                    HistogramBucket(bucket["upper_bound"], bucket["count"])
                    for bucket in bucket_data
                ]

        return metric

    def to_json(self) -> str:
        """Convert metric to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'PerformanceMetric':
        """Create metric from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of the metric."""
        latest = self.get_latest_value()
        return f"PerformanceMetric({self.name}, {self.metric_type.value}, latest={latest})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"PerformanceMetric(name='{self.name}', type='{self.metric_type.value}', "
                f"points={self.current_size}, total={self.total_points})")

    def __eq__(self, other) -> bool:
        """Check equality based on name and type."""
        if not isinstance(other, PerformanceMetric):
            return False
        return self.name == other.name and self.metric_type == other.metric_type

    def __hash__(self) -> int:
        """Hash based on name and type."""
        return hash((self.name, self.metric_type.value))