"""
Size-based strategy implementation.
"""

from typing import Optional, Any


class SizeStrategy:
    """Size-based eviction strategy."""

    def __init__(self, max_size_bytes: Optional[int] = None) -> None:
        self.max_size_bytes = max_size_bytes
        self.name = "size"

    async def initialize(self, cache: Any) -> None:
        """Initialize strategy."""
        pass

    async def shutdown(self) -> None:
        """Shutdown strategy."""
        pass

    def __str__(self) -> str:
        return f"Size(max_size_bytes={self.max_size_bytes})"