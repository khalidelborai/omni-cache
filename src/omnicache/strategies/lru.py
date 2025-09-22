"""
LRU (Least Recently Used) strategy implementation.
"""

from typing import Optional, Any


class LRUStrategy:
    """Least Recently Used eviction strategy."""

    def __init__(self, max_size: Optional[int] = None) -> None:
        self.max_size = max_size
        self.name = "lru"

    async def initialize(self, cache: Any) -> None:
        """Initialize strategy."""
        pass

    async def shutdown(self) -> None:
        """Shutdown strategy."""
        pass

    def __str__(self) -> str:
        return f"LRU(max_size={self.max_size})"