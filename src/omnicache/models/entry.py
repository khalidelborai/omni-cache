"""
CacheEntry entity model.

Defines the CacheEntry class that combines Key and Value with cache-specific
metadata like TTL, access patterns, and lifecycle information.
"""

from datetime import datetime, timedelta
from typing import Optional, Any, Dict
from enum import Enum

from omnicache.models.key import Key
from omnicache.models.value import Value


class EntryStatus(Enum):
    """Cache entry status enumeration."""
    ACTIVE = "active"
    EXPIRED = "expired"
    EVICTED = "evicted"
    DELETED = "deleted"


class CacheEntry:
    """
    Cache entry combining key, value, and metadata.

    Represents a complete cache entry with lifecycle management,
    access tracking, and expiration handling.
    """

    def __init__(
        self,
        key: Key,
        value: Value,
        ttl: Optional[float] = None,
        priority: float = 0.5
    ) -> None:
        """
        Initialize a cache entry.

        Args:
            key: Cache key object
            value: Cache value object
            ttl: Time-to-live in seconds (None = no expiration)
            priority: Entry priority for eviction (0.0-1.0, higher = more important)

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if not isinstance(key, Key):
            raise ValueError("Key must be a Key instance")
        if not isinstance(value, Value):
            raise ValueError("Value must be a Value instance")
        if ttl is not None and ttl <= 0:
            raise ValueError("TTL must be positive")
        if not (0.0 <= priority <= 1.0):
            raise ValueError("Priority must be between 0.0 and 1.0")

        # Core components
        self._key = key
        self._value = value
        self._priority = priority

        # Timing attributes
        self._created_at = datetime.now()
        self._last_accessed = self._created_at
        self._last_modified = self._created_at
        self._expires_at = self._created_at + timedelta(seconds=ttl) if ttl else None

        # Access tracking
        self._access_count = 0
        self._access_frequency = 0.0  # accesses per second
        self._hit_count = 0

        # Status and metadata
        self._status = EntryStatus.ACTIVE
        self._version = 1

    @property
    def key(self) -> Key:
        """Get the cache key."""
        return self._key

    @property
    def value(self) -> Value:
        """Get the cache value."""
        return self._value

    @property
    def priority(self) -> float:
        """Get the entry priority."""
        return self._priority

    @property
    def created_at(self) -> datetime:
        """Get the creation timestamp."""
        return self._created_at

    @property
    def last_accessed(self) -> datetime:
        """Get the last access timestamp."""
        return self._last_accessed

    @property
    def last_modified(self) -> datetime:
        """Get the last modification timestamp."""
        return self._last_modified

    @property
    def expires_at(self) -> Optional[datetime]:
        """Get the expiration timestamp."""
        return self._expires_at

    @property
    def access_count(self) -> int:
        """Get the total access count."""
        return self._access_count

    @property
    def access_frequency(self) -> float:
        """Get the access frequency (accesses per second)."""
        return self._access_frequency

    @property
    def hit_count(self) -> int:
        """Get the hit count."""
        return self._hit_count

    @property
    def status(self) -> EntryStatus:
        """Get the entry status."""
        return self._status

    @property
    def version(self) -> int:
        """Get the entry version."""
        return self._version

    @property
    def size_bytes(self) -> int:
        """Get the total size in bytes."""
        # Approximate size calculation
        key_size = len(str(self._key)) * 2  # Rough estimate for Unicode
        value_size = self._value.size_bytes
        metadata_size = 200  # Rough estimate for metadata overhead
        return key_size + value_size + metadata_size

    @property
    def age_seconds(self) -> float:
        """Get the age in seconds since creation."""
        return (datetime.now() - self._created_at).total_seconds()

    @property
    def time_since_access(self) -> float:
        """Get seconds since last access."""
        return (datetime.now() - self._last_accessed).total_seconds()

    @property
    def ttl_remaining(self) -> Optional[float]:
        """Get remaining TTL in seconds (None if no expiration)."""
        if self._expires_at is None:
            return None
        remaining = (self._expires_at - datetime.now()).total_seconds()
        return max(0.0, remaining)

    def is_expired(self) -> bool:
        """
        Check if the entry has expired.

        Returns:
            True if expired, False otherwise
        """
        if self._expires_at is None:
            return False
        return datetime.now() >= self._expires_at

    def is_active(self) -> bool:
        """
        Check if the entry is active (not expired/evicted/deleted).

        Returns:
            True if active, False otherwise
        """
        return self._status == EntryStatus.ACTIVE and not self.is_expired()

    def access(self) -> None:
        """Record an access to this entry."""
        now = datetime.now()
        self._last_accessed = now
        self._access_count += 1
        self._hit_count += 1

        # Update access frequency (simple moving average)
        if self._access_count > 1:
            time_diff = (now - self._created_at).total_seconds()
            self._access_frequency = self._access_count / max(time_diff, 1.0)

    def update_value(self, new_value: Value, ttl: Optional[float] = None) -> None:
        """
        Update the entry value and optionally TTL.

        Args:
            new_value: New value to store
            ttl: New TTL in seconds (None to keep current)

        Raises:
            ValueError: If value is invalid
        """
        if not isinstance(new_value, Value):
            raise ValueError("Value must be a Value instance")

        self._value = new_value
        self._last_modified = datetime.now()
        self._version += 1

        # Update TTL if provided
        if ttl is not None:
            if ttl <= 0:
                raise ValueError("TTL must be positive")
            self._expires_at = datetime.now() + timedelta(seconds=ttl)

    def update_priority(self, priority: float) -> None:
        """
        Update the entry priority.

        Args:
            priority: New priority value (0.0-1.0)

        Raises:
            ValueError: If priority is invalid
        """
        if not (0.0 <= priority <= 1.0):
            raise ValueError("Priority must be between 0.0 and 1.0")
        self._priority = priority

    def extend_ttl(self, seconds: float) -> None:
        """
        Extend the TTL by specified seconds.

        Args:
            seconds: Seconds to add to TTL

        Raises:
            ValueError: If seconds is invalid or entry has no TTL
        """
        if seconds <= 0:
            raise ValueError("Extension must be positive")
        if self._expires_at is None:
            raise ValueError("Cannot extend TTL on entry without expiration")

        self._expires_at += timedelta(seconds=seconds)

    def mark_expired(self) -> None:
        """Mark the entry as expired."""
        self._status = EntryStatus.EXPIRED

    def mark_evicted(self) -> None:
        """Mark the entry as evicted."""
        self._status = EntryStatus.EVICTED

    def mark_deleted(self) -> None:
        """Mark the entry as deleted."""
        self._status = EntryStatus.DELETED

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entry to dictionary representation.

        Returns:
            Dictionary containing all entry data
        """
        return {
            "key": self._key.to_dict(),
            "value": self._value.to_dict(),
            "priority": self._priority,
            "created_at": self._created_at.isoformat(),
            "last_accessed": self._last_accessed.isoformat(),
            "last_modified": self._last_modified.isoformat(),
            "expires_at": self._expires_at.isoformat() if self._expires_at else None,
            "access_count": self._access_count,
            "access_frequency": self._access_frequency,
            "hit_count": self._hit_count,
            "status": self._status.value,
            "version": self._version,
            "size_bytes": self.size_bytes,
            "age_seconds": self.age_seconds,
            "time_since_access": self.time_since_access,
            "ttl_remaining": self.ttl_remaining,
            "is_expired": self.is_expired(),
            "is_active": self.is_active()
        }

    @classmethod
    def create(
        cls,
        key_value: str,
        data: Any,
        namespace: str = "",
        ttl: Optional[float] = None,
        priority: float = 0.5,
        **kwargs: Any
    ) -> 'CacheEntry':
        """
        Create a cache entry from simple parameters.

        Args:
            key_value: The key string
            data: The data to cache
            namespace: Optional namespace
            ttl: Time-to-live in seconds
            priority: Entry priority
            **kwargs: Additional Key/Value parameters

        Returns:
            CacheEntry instance
        """
        key = Key(value=key_value, namespace=namespace, **kwargs)
        value = Value(data=data)
        return cls(key=key, value=value, ttl=ttl, priority=priority)

    def __str__(self) -> str:
        status_indicator = "✓" if self.is_active() else "✗"
        ttl_info = f", TTL={self.ttl_remaining:.1f}s" if self.ttl_remaining else ""
        return f"CacheEntry({status_indicator} {self._key.full_key}, {self._value}{ttl_info})"

    def __repr__(self) -> str:
        return (f"<CacheEntry(key='{self._key.full_key}', "
                f"status={self._status.value}, version={self._version})>")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CacheEntry):
            return False
        return (
            self._key == other._key and
            self._value == other._value and
            self._version == other._version
        )

    def __hash__(self) -> int:
        return hash((self._key, self._value.checksum, self._version))