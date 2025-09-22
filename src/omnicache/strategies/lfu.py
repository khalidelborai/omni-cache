"""
LFU (Least Frequently Used) strategy implementation.
"""

from typing import Optional, Any


class LFUStrategy:
    """Least Frequently Used eviction strategy."""

    def __init__(self, max_size: Optional[int] = None) -> None:
        self.max_size = max_size
        self.name = "lfu"

    async def initialize(self, cache: Any) -> None:
        """Initialize strategy."""
        pass

    async def shutdown(self) -> None:
        """Shutdown strategy."""
        pass

    def __str__(self) -> str:
        return f"LFU(max_size={self.max_size})"