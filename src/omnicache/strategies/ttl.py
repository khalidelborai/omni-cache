"""
TTL (Time To Live) strategy implementation.
"""

from typing import Optional, Any


class TTLStrategy:
    """Time To Live expiration strategy."""

    def __init__(self, default_ttl: Optional[float] = None) -> None:
        self.default_ttl = default_ttl
        self.name = "ttl"

    async def initialize(self, cache: Any) -> None:
        """Initialize strategy."""
        pass

    async def shutdown(self) -> None:
        """Shutdown strategy."""
        pass

    def __str__(self) -> str:
        return f"TTL(default_ttl={self.default_ttl})"