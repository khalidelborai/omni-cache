"""
Access Pattern model for ML-powered cache prediction.

This module defines the AccessPattern model for tracking and analyzing
cache access patterns to enable machine learning predictions.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
import time
import hashlib
import json


@dataclass
class AccessPattern:
    """
    Access pattern model for ML-powered cache prediction.

    Represents a single cache access event with context information
    for machine learning analysis and prediction.
    """

    user_id: str
    key: str
    timestamp: float
    operation: str  # GET, SET, DELETE, etc.
    session_id: str

    # Optional context
    value_size: Optional[int] = None
    cache_hit: Optional[bool] = None
    latency_ms: Optional[float] = None
    backend_name: Optional[str] = None

    # Request context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    referer: Optional[str] = None
    request_id: Optional[str] = None

    # Application context
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation."""
        if not self.user_id:
            raise ValueError("user_id is required")
        if not self.key:
            raise ValueError("key is required")
        if self.timestamp <= 0:
            raise ValueError("timestamp must be positive")
        if not self.operation:
            raise ValueError("operation is required")
        if not self.session_id:
            raise ValueError("session_id is required")

    @property
    def datetime(self) -> datetime:
        """Get timestamp as datetime object."""
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

    @property
    def hour_of_day(self) -> int:
        """Get hour of day (0-23) from timestamp."""
        return self.datetime.hour

    @property
    def day_of_week(self) -> int:
        """Get day of week (0=Monday, 6=Sunday) from timestamp."""
        return self.datetime.weekday()

    @property
    def is_weekend(self) -> bool:
        """Check if access occurred on weekend."""
        return self.day_of_week >= 5  # Saturday=5, Sunday=6

    @property
    def key_pattern(self) -> str:
        """Extract pattern from cache key (e.g., 'user:123:profile' -> 'user:*:profile')."""
        parts = self.key.split(':')
        pattern_parts = []

        for part in parts:
            if part.isdigit() or len(part) > 10:  # Likely an ID
                pattern_parts.append('*')
            else:
                pattern_parts.append(part)

        return ':'.join(pattern_parts)

    @property
    def key_depth(self) -> int:
        """Get depth of cache key (number of ':' separators + 1)."""
        return len(self.key.split(':'))

    @property
    def is_read_operation(self) -> bool:
        """Check if this is a read operation."""
        return self.operation.upper() in ['GET', 'MGET', 'EXIST', 'TTL']

    @property
    def is_write_operation(self) -> bool:
        """Check if this is a write operation."""
        return self.operation.upper() in ['SET', 'MSET', 'DELETE', 'EXPIRE']

    def extract_features(self) -> Dict[str, Any]:
        """
        Extract features for machine learning models.

        Returns a dictionary of features that can be used for
        training ML models to predict cache access patterns.
        """
        return {
            # Temporal features
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "is_weekend": self.is_weekend,
            "timestamp_normalized": self.timestamp % 86400,  # Seconds since midnight

            # Key features
            "key_pattern": self.key_pattern,
            "key_depth": self.key_depth,
            "key_length": len(self.key),
            "key_hash": self._hash_key(),

            # Operation features
            "operation_type": self.operation.upper(),
            "is_read_operation": self.is_read_operation,
            "is_write_operation": self.is_write_operation,

            # Performance features
            "value_size": self.value_size or 0,
            "latency_ms": self.latency_ms or 0.0,
            "cache_hit": self.cache_hit,

            # Session features
            "session_hash": self._hash_session(),

            # Context features
            "has_user_agent": self.user_agent is not None,
            "has_referer": self.referer is not None,
            "backend_name": self.backend_name or "unknown",

            # Feature flags (one-hot encoded)
            **{f"flag_{k}": v for k, v in self.feature_flags.items()},

            # Metadata features
            "metadata_count": len(self.metadata),
        }

    def extract_sequence_features(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Extract features for sequence-based models.

        Args:
            window_size: Size of the sequence window

        Returns:
            Features suitable for RNN/LSTM models
        """
        base_features = self.extract_features()

        return {
            **base_features,
            "sequence_position": 0,  # Will be set by collector
            "window_size": window_size,
            "relative_timestamp": 0.0,  # Will be set by collector
        }

    def get_similarity_features(self) -> Tuple[str, ...]:
        """
        Get features for similarity calculations.

        Returns a tuple of features that can be used to calculate
        similarity between access patterns.
        """
        return (
            self.key_pattern,
            self.operation.upper(),
            str(self.hour_of_day),
            str(self.day_of_week),
            str(self.is_weekend),
            self.backend_name or "unknown",
        )

    def is_similar_to(self, other: 'AccessPattern', threshold: float = 0.8) -> bool:
        """
        Check if this pattern is similar to another pattern.

        Args:
            other: Another access pattern
            threshold: Similarity threshold (0.0 - 1.0)

        Returns:
            True if patterns are similar
        """
        if not isinstance(other, AccessPattern):
            return False

        my_features = self.get_similarity_features()
        other_features = other.get_similarity_features()

        matches = sum(1 for a, b in zip(my_features, other_features) if a == b)
        similarity = matches / len(my_features)

        return similarity >= threshold

    def _hash_key(self) -> str:
        """Generate hash of the cache key for feature extraction."""
        return hashlib.md5(self.key.encode()).hexdigest()[:8]

    def _hash_session(self) -> str:
        """Generate hash of the session ID for anonymization."""
        return hashlib.md5(self.session_id.encode()).hexdigest()[:8]

    def anonymize(self) -> 'AccessPattern':
        """
        Create an anonymized copy of this access pattern.

        Removes or hashes personally identifiable information
        while preserving pattern information for ML training.
        """
        return AccessPattern(
            user_id=self._hash_user_id(),
            key=self._anonymize_key(),
            timestamp=self.timestamp,
            operation=self.operation,
            session_id=self._hash_session(),
            value_size=self.value_size,
            cache_hit=self.cache_hit,
            latency_ms=self.latency_ms,
            backend_name=self.backend_name,
            ip_address=None,  # Remove IP
            user_agent=None,  # Remove user agent
            referer=None,     # Remove referer
            request_id=None,  # Remove request ID
            feature_flags=self.feature_flags.copy(),
            metadata={}       # Remove metadata
        )

    def _hash_user_id(self) -> str:
        """Generate anonymized user ID."""
        return f"user_{hashlib.md5(self.user_id.encode()).hexdigest()[:16]}"

    def _anonymize_key(self) -> str:
        """Anonymize cache key while preserving pattern."""
        parts = self.key.split(':')
        anonymized_parts = []

        for part in parts:
            if part.isdigit():
                # Hash numeric IDs
                anonymized_parts.append(hashlib.md5(part.encode()).hexdigest()[:8])
            elif len(part) > 10:
                # Hash long strings (likely IDs)
                anonymized_parts.append(hashlib.md5(part.encode()).hexdigest()[:8])
            else:
                # Keep short strings (likely patterns)
                anonymized_parts.append(part)

        return ':'.join(anonymized_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert access pattern to dictionary representation."""
        return {
            "user_id": self.user_id,
            "key": self.key,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "session_id": self.session_id,
            "value_size": self.value_size,
            "cache_hit": self.cache_hit,
            "latency_ms": self.latency_ms,
            "backend_name": self.backend_name,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "referer": self.referer,
            "request_id": self.request_id,
            "feature_flags": self.feature_flags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccessPattern':
        """Create access pattern from dictionary representation."""
        return cls(
            user_id=data["user_id"],
            key=data["key"],
            timestamp=data["timestamp"],
            operation=data["operation"],
            session_id=data["session_id"],
            value_size=data.get("value_size"),
            cache_hit=data.get("cache_hit"),
            latency_ms=data.get("latency_ms"),
            backend_name=data.get("backend_name"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            referer=data.get("referer"),
            request_id=data.get("request_id"),
            feature_flags=data.get("feature_flags", {}),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Convert access pattern to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'AccessPattern':
        """Create access pattern from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of the access pattern."""
        dt = self.datetime.strftime("%Y-%m-%d %H:%M:%S")
        return f"AccessPattern({self.user_id}, {self.key}, {self.operation}, {dt})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"AccessPattern(user_id='{self.user_id}', key='{self.key}', "
                f"operation='{self.operation}', timestamp={self.timestamp})")

    def __eq__(self, other) -> bool:
        """Check equality based on all key fields."""
        if not isinstance(other, AccessPattern):
            return False
        return (
            self.user_id == other.user_id and
            self.key == other.key and
            self.timestamp == other.timestamp and
            self.operation == other.operation and
            self.session_id == other.session_id
        )

    def __hash__(self) -> int:
        """Hash based on key identifying fields."""
        return hash((self.user_id, self.key, self.timestamp, self.operation, self.session_id))