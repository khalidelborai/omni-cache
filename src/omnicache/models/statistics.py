"""
Statistics entity model.
"""

from datetime import datetime
from typing import Dict, Any


class Statistics:
    """Cache statistics tracking."""

    def __init__(self, cache_name: str) -> None:
        self.cache_name = cache_name
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.error_count = 0
        self.total_size_bytes = 0
        self.entry_count = 0
        self.avg_access_time_ms = 0.0
        self.last_reset = datetime.now()
        self.collection_interval = 5.0

    async def initialize(self) -> None:
        """Initialize statistics."""
        pass

    async def shutdown(self) -> None:
        """Shutdown statistics."""
        pass

    async def get_current_stats(self) -> 'Statistics':
        """Get current statistics."""
        return self

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    @property
    def backend_status(self) -> str:
        """Get backend status."""
        return "connected"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cache_name": self.cache_name,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
            "eviction_count": self.eviction_count,
            "error_count": self.error_count,
            "total_size_bytes": self.total_size_bytes,
            "entry_count": self.entry_count,
            "avg_access_time_ms": self.avg_access_time_ms,
            "backend_status": self.backend_status,
            "last_reset": self.last_reset.isoformat()
        }