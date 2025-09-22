"""
Cache Tier model for hierarchical cache management.

This module defines the CacheTier model for managing different levels
of cache storage in a hierarchical cache architecture.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import time


class TierType(Enum):
    """Types of cache tiers."""
    MEMORY = "memory"
    REDIS = "redis"
    FILESYSTEM = "filesystem"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"
    CUSTOM = "custom"


class TierStatus(Enum):
    """Status of a cache tier."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"


@dataclass
class TierMetrics:
    """Metrics for a cache tier."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    promotions: int = 0
    demotions: int = 0
    total_size_bytes: int = 0
    used_size_bytes: int = 0
    avg_latency_ms: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def hit_ratio(self) -> float:
        """Calculate hit ratio for this tier."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def utilization(self) -> float:
        """Calculate storage utilization percentage."""
        return (self.used_size_bytes / self.total_size_bytes * 100) if self.total_size_bytes > 0 else 0.0


@dataclass
class CacheTier:
    """
    Cache tier model for hierarchical cache management.

    Represents a single tier in a multi-level cache hierarchy,
    such as L1 (memory), L2 (Redis), L3 (cloud storage).
    """

    name: str
    tier_type: str
    capacity: int
    latency_ms: float = 1.0
    cost_per_gb: float = 0.0
    eviction_strategy: str = "lru"
    status: TierStatus = TierStatus.ACTIVE
    priority: int = 0  # Lower values = higher priority (closer to client)

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Metrics
    metrics: TierMetrics = field(default_factory=TierMetrics)

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.tier_type not in [t.value for t in TierType]:
            if self.tier_type != "custom":
                raise ValueError(f"Invalid tier_type: {self.tier_type}")

        if self.capacity <= 0:
            raise ValueError("Capacity must be positive")

        if self.latency_ms < 0:
            raise ValueError("Latency cannot be negative")

    @property
    def tier_type_enum(self) -> TierType:
        """Get tier type as enum."""
        try:
            return TierType(self.tier_type)
        except ValueError:
            return TierType.CUSTOM

    @property
    def is_active(self) -> bool:
        """Check if tier is active."""
        return self.status == TierStatus.ACTIVE

    @property
    def is_cloud_tier(self) -> bool:
        """Check if this is a cloud storage tier."""
        return self.tier_type in ["s3", "azure_blob", "gcs"]

    @property
    def is_local_tier(self) -> bool:
        """Check if this is a local storage tier."""
        return self.tier_type in ["memory", "filesystem"]

    def update_metrics(self, **kwargs) -> None:
        """Update tier metrics."""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)

        self.metrics.last_updated = time.time()
        self.updated_at = time.time()

    def record_hit(self) -> None:
        """Record a cache hit for this tier."""
        self.metrics.hits += 1
        self.metrics.last_updated = time.time()

    def record_miss(self) -> None:
        """Record a cache miss for this tier."""
        self.metrics.misses += 1
        self.metrics.last_updated = time.time()

    def record_eviction(self) -> None:
        """Record an eviction from this tier."""
        self.metrics.evictions += 1
        self.metrics.last_updated = time.time()

    def record_promotion(self) -> None:
        """Record a promotion to this tier."""
        self.metrics.promotions += 1
        self.metrics.last_updated = time.time()

    def record_demotion(self) -> None:
        """Record a demotion from this tier."""
        self.metrics.demotions += 1
        self.metrics.last_updated = time.time()

    def set_status(self, status: TierStatus) -> None:
        """Set tier status."""
        self.status = status
        self.updated_at = time.time()

    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to the tier."""
        self.tags[key] = value
        self.updated_at = time.time()

    def remove_tag(self, key: str) -> None:
        """Remove a tag from the tier."""
        self.tags.pop(key, None)
        self.updated_at = time.time()

    def get_cost_estimate(self, size_bytes: int) -> float:
        """Calculate cost estimate for storing data of given size."""
        size_gb = size_bytes / (1024 * 1024 * 1024)
        return size_gb * self.cost_per_gb

    def can_accommodate(self, size_bytes: int) -> bool:
        """Check if tier can accommodate additional data."""
        if not self.is_active:
            return False

        available_space = self.capacity - self.metrics.used_size_bytes
        return available_space >= size_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert tier to dictionary representation."""
        return {
            "name": self.name,
            "tier_type": self.tier_type,
            "capacity": self.capacity,
            "latency_ms": self.latency_ms,
            "cost_per_gb": self.cost_per_gb,
            "eviction_strategy": self.eviction_strategy,
            "status": self.status.value,
            "priority": self.priority,
            "config": self.config,
            "metrics": {
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "evictions": self.metrics.evictions,
                "promotions": self.metrics.promotions,
                "demotions": self.metrics.demotions,
                "total_size_bytes": self.metrics.total_size_bytes,
                "used_size_bytes": self.metrics.used_size_bytes,
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "hit_ratio": self.metrics.hit_ratio,
                "utilization": self.metrics.utilization,
                "last_updated": self.metrics.last_updated,
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheTier':
        """Create tier from dictionary representation."""
        metrics_data = data.get("metrics", {})
        metrics = TierMetrics(
            hits=metrics_data.get("hits", 0),
            misses=metrics_data.get("misses", 0),
            evictions=metrics_data.get("evictions", 0),
            promotions=metrics_data.get("promotions", 0),
            demotions=metrics_data.get("demotions", 0),
            total_size_bytes=metrics_data.get("total_size_bytes", 0),
            used_size_bytes=metrics_data.get("used_size_bytes", 0),
            avg_latency_ms=metrics_data.get("avg_latency_ms", 0.0),
            last_updated=metrics_data.get("last_updated", time.time()),
        )

        return cls(
            name=data["name"],
            tier_type=data["tier_type"],
            capacity=data["capacity"],
            latency_ms=data.get("latency_ms", 1.0),
            cost_per_gb=data.get("cost_per_gb", 0.0),
            eviction_strategy=data.get("eviction_strategy", "lru"),
            status=TierStatus(data.get("status", "active")),
            priority=data.get("priority", 0),
            config=data.get("config", {}),
            metrics=metrics,
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
        )

    def __str__(self) -> str:
        """String representation of the tier."""
        return f"CacheTier({self.name}, {self.tier_type}, {self.capacity} bytes, {self.latency_ms}ms)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"CacheTier(name='{self.name}', tier_type='{self.tier_type}', "
                f"capacity={self.capacity}, status={self.status.value})")

    def __eq__(self, other) -> bool:
        """Check equality based on name and tier_type."""
        if not isinstance(other, CacheTier):
            return False
        return self.name == other.name and self.tier_type == other.tier_type

    def __hash__(self) -> int:
        """Hash based on name and tier_type."""
        return hash((self.name, self.tier_type))

    def __lt__(self, other) -> bool:
        """Compare tiers by priority (lower priority = higher precedence)."""
        if not isinstance(other, CacheTier):
            return NotImplemented
        return self.priority < other.priority