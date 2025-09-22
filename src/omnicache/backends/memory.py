"""
Memory backend implementation.
"""

from typing import Any, Optional, List, Dict
from datetime import datetime


class MemoryBackend:
    """In-memory storage backend."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize backend."""
        pass

    async def shutdown(self) -> None:
        """Shutdown backend."""
        self._data.clear()

    async def set(self, key: str, value: Any, ttl: Optional[float] = None,
                 tags: Optional[List[str]] = None, priority: Optional[float] = None) -> None:
        """Set a value."""
        self._data[key] = value

    async def get(self, key: str) -> Any:
        """Get a value."""
        return self._data.get(key)

    async def delete(self, key: str) -> bool:
        """Delete a value."""
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def get_entry(self, key: str) -> Any:
        """Get entry with metadata."""
        if key in self._data:
            return MockCacheEntry(key, self._data[key])
        return None

    async def clear(self, pattern: Optional[str] = None, tags: Optional[List[str]] = None) -> Any:
        """Clear entries."""
        if pattern is None and tags is None:
            count = len(self._data)
            self._data.clear()
        else:
            count = 0  # Simplified for now

        return MockClearResult(count)

    def __str__(self) -> str:
        return "Memory"


class MockCacheEntry:
    """Mock cache entry for testing."""

    def __init__(self, key: str, value: Any) -> None:
        self.key = MockKey(key)
        self.value = MockValue(value)
        self.ttl = None
        self.access_count = 1
        self.last_accessed = datetime.now()
        self.created_at = datetime.now()
        self.priority = 0.5


class MockKey:
    """Mock key object."""

    def __init__(self, value: str) -> None:
        self.value = value
        self.tags = set()
        self.namespace = ""
        self.hash_value = hash(value)
        self.created_at = datetime.now()


class MockValue:
    """Mock value object."""

    def __init__(self, data: Any) -> None:
        self.data = data
        self.serialized_data = str(data).encode()
        self.serializer_type = "json"
        self.size_bytes = len(self.serialized_data)
        self.content_type = "application/json"
        self.checksum = ""
        self.version = 1


class MockClearResult:
    """Mock clear result."""

    def __init__(self, cleared_count: int) -> None:
        self.cleared_count = cleared_count