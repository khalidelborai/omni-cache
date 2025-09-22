"""
Access Pattern Collectors for ML prediction system.

This module implements data collectors that gather and organize access patterns
for machine learning model training and real-time prediction.
"""

import time
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
import logging
from datetime import datetime, timedelta

from omnicache.models.access_pattern import AccessPattern


logger = logging.getLogger(__name__)


@dataclass
class Session:
    """User session for tracking related access patterns."""
    session_id: str
    user_id: str
    start_time: float
    last_activity: float
    patterns: List[AccessPattern] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        return self.last_activity - self.start_time

    def is_active(self, timeout: float = 3600) -> bool:
        """Check if session is still active."""
        return (time.time() - self.last_activity) < timeout


class AccessPatternCollector:
    """
    Collects and organizes access patterns for ML training and prediction.

    Tracks user sessions, aggregates patterns, and provides data for
    machine learning model training and real-time prediction systems.
    """

    def __init__(
        self,
        session_timeout: float = 3600,  # 1 hour
        max_patterns_per_user: int = 10000,
        max_sessions_per_user: int = 100,
        pattern_retention_days: int = 30,
        enable_real_time_features: bool = True,
    ):
        """
        Initialize access pattern collector.

        Args:
            session_timeout: Session timeout in seconds
            max_patterns_per_user: Maximum patterns to store per user
            max_sessions_per_user: Maximum sessions to track per user
            pattern_retention_days: Days to retain patterns
            enable_real_time_features: Enable real-time feature extraction
        """
        self.session_timeout = session_timeout
        self.max_patterns_per_user = max_patterns_per_user
        self.max_sessions_per_user = max_sessions_per_user
        self.pattern_retention_days = pattern_retention_days
        self.enable_real_time_features = enable_real_time_features

        # Storage
        self._patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_patterns_per_user))
        self._sessions: Dict[str, Dict[str, Session]] = defaultdict(dict)
        self._active_sessions: Dict[str, str] = {}  # user_id -> active_session_id

        # Caches for fast lookups
        self._key_sequences: Dict[str, List[str]] = defaultdict(list)
        self._transition_counts: Dict[str, Dict[Tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
        self._user_preferences: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Statistics
        self._total_patterns = 0
        self._total_sessions = 0

        # Thread safety
        self._lock = threading.RLock()

        logger.info("AccessPatternCollector initialized")

    def collect(self, pattern: AccessPattern) -> None:
        """
        Collect a single access pattern.

        Args:
            pattern: Access pattern to collect
        """
        self.record_access(
            user_id=pattern.user_id,
            key=pattern.key,
            operation=pattern.operation,
            timestamp=pattern.timestamp,
            session_id=pattern.session_id,
            context={
                'value_size': pattern.value_size,
                'cache_hit': pattern.cache_hit,
                'latency_ms': pattern.latency_ms,
                'backend_name': pattern.backend_name,
                'ip_address': pattern.ip_address,
                'user_agent': pattern.user_agent,
                'referer': pattern.referer,
                'request_id': pattern.request_id,
                'feature_flags': pattern.feature_flags,
                'metadata': pattern.metadata,
            }
        )

    def record_access(
        self,
        user_id: str,
        key: str,
        operation: str,
        timestamp: Optional[float] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a cache access event.

        Args:
            user_id: User identifier
            key: Cache key accessed
            operation: Operation type (GET, SET, DELETE, etc.)
            timestamp: Access timestamp (defaults to current time)
            session_id: Session identifier (auto-generated if not provided)
            context: Additional context information

        Returns:
            Session ID for this access
        """
        if timestamp is None:
            timestamp = time.time()

        if context is None:
            context = {}

        with self._lock:
            # Determine session
            if session_id is None:
                session_id = self._get_or_create_session(user_id, timestamp)
            else:
                self._update_session_activity(user_id, session_id, timestamp)

            # Create access pattern
            pattern = AccessPattern(
                user_id=user_id,
                key=key,
                timestamp=timestamp,
                operation=operation.upper(),
                session_id=session_id,
                value_size=context.get('value_size'),
                cache_hit=context.get('cache_hit'),
                latency_ms=context.get('latency_ms'),
                backend_name=context.get('backend_name'),
                ip_address=context.get('ip_address'),
                user_agent=context.get('user_agent'),
                referer=context.get('referer'),
                request_id=context.get('request_id'),
                feature_flags=context.get('feature_flags', {}),
                metadata=context.get('metadata', {}),
            )

            # Store pattern
            self._patterns[user_id].append(pattern)
            self._total_patterns += 1

            # Update caches for real-time features
            if self.enable_real_time_features:
                self._update_real_time_features(user_id, pattern)

            # Clean up old data
            self._cleanup_old_data()

            logger.debug(f"Recorded access: {user_id}, {key}, {operation}")

        return session_id

    def get_patterns(self, user_id: str, limit: Optional[int] = None) -> List[AccessPattern]:
        """
        Get access patterns for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of patterns to return

        Returns:
            List of access patterns
        """
        with self._lock:
            patterns = list(self._patterns[user_id])
            if limit:
                patterns = patterns[-limit:]
            return patterns

    def get_sessions(self, user_id: str) -> List[Session]:
        """
        Get sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            List of user sessions
        """
        with self._lock:
            return list(self._sessions[user_id].values())

    def get_active_session(self, user_id: str) -> Optional[Session]:
        """
        Get the active session for a user.

        Args:
            user_id: User identifier

        Returns:
            Active session or None
        """
        with self._lock:
            session_id = self._active_sessions.get(user_id)
            if session_id and session_id in self._sessions[user_id]:
                session = self._sessions[user_id][session_id]
                if session.is_active(self.session_timeout):
                    return session
                else:
                    # Session expired
                    del self._active_sessions[user_id]
            return None

    def get_key_sequence(self, user_id: str, session_id: Optional[str] = None) -> List[str]:
        """
        Get sequence of keys accessed by user.

        Args:
            user_id: User identifier
            session_id: Optional session identifier

        Returns:
            List of keys in access order
        """
        with self._lock:
            if session_id and session_id in self._sessions[user_id]:
                return [p.key for p in self._sessions[user_id][session_id].patterns]
            else:
                return self._key_sequences[user_id]

    def get_transition_probabilities(self, user_id: str) -> Dict[str, Dict[str, float]]:
        """
        Get key transition probabilities for a user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary mapping from_key -> {to_key: probability}
        """
        with self._lock:
            transitions = self._transition_counts[user_id]
            probabilities = {}

            for (from_key, to_key), count in transitions.items():
                if from_key not in probabilities:
                    probabilities[from_key] = {}
                probabilities[from_key][to_key] = count

            # Normalize to probabilities
            for from_key, to_keys in probabilities.items():
                total = sum(to_keys.values())
                if total > 0:
                    for to_key in to_keys:
                        to_keys[to_key] /= total

            return probabilities

    def get_user_preferences(self, user_id: str) -> Dict[str, float]:
        """
        Get user preferences based on access patterns.

        Args:
            user_id: User identifier

        Returns:
            Dictionary mapping key_pattern -> preference_score
        """
        with self._lock:
            return dict(self._user_preferences[user_id])

    def get_temporal_patterns(self, user_id: str) -> Dict[str, List[AccessPattern]]:
        """
        Get temporal access patterns grouped by time buckets.

        Args:
            user_id: User identifier

        Returns:
            Dictionary mapping time_bucket -> patterns
        """
        with self._lock:
            patterns = self._patterns[user_id]
            temporal_buckets = defaultdict(list)

            for pattern in patterns:
                hour_bucket = f"{pattern.hour_of_day:02d}:00"
                temporal_buckets[hour_bucket].append(pattern)

            return dict(temporal_buckets)

    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics."""
        with self._lock:
            active_sessions = sum(
                1 for user_sessions in self._sessions.values()
                for session in user_sessions.values()
                if session.is_active(self.session_timeout)
            )

            return {
                "total_patterns": self._total_patterns,
                "total_sessions": self._total_sessions,
                "active_sessions": active_sessions,
                "total_users": len(self._patterns),
                "patterns_by_user": {
                    user_id: len(patterns)
                    for user_id, patterns in self._patterns.items()
                },
                "avg_patterns_per_user": (
                    self._total_patterns / len(self._patterns)
                    if self._patterns else 0
                ),
                "session_timeout": self.session_timeout,
                "pattern_retention_days": self.pattern_retention_days,
            }

    def clear_user_data(self, user_id: str) -> None:
        """
        Clear all data for a specific user.

        Args:
            user_id: User identifier
        """
        with self._lock:
            if user_id in self._patterns:
                del self._patterns[user_id]
            if user_id in self._sessions:
                del self._sessions[user_id]
            if user_id in self._active_sessions:
                del self._active_sessions[user_id]
            if user_id in self._key_sequences:
                del self._key_sequences[user_id]
            if user_id in self._transition_counts:
                del self._transition_counts[user_id]
            if user_id in self._user_preferences:
                del self._user_preferences[user_id]

            logger.info(f"Cleared data for user: {user_id}")

    def clear_all_data(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self._patterns.clear()
            self._sessions.clear()
            self._active_sessions.clear()
            self._key_sequences.clear()
            self._transition_counts.clear()
            self._user_preferences.clear()
            self._total_patterns = 0
            self._total_sessions = 0

            logger.info("Cleared all collector data")

    def _get_or_create_session(self, user_id: str, timestamp: float) -> str:
        """Get or create a session for the user."""
        # Check if there's an active session
        active_session = self.get_active_session(user_id)
        if active_session:
            active_session.last_activity = timestamp
            return active_session.session_id

        # Create new session
        session_id = f"{user_id}_{int(timestamp)}_{self._total_sessions}"
        session = Session(
            session_id=session_id,
            user_id=user_id,
            start_time=timestamp,
            last_activity=timestamp,
        )

        self._sessions[user_id][session_id] = session
        self._active_sessions[user_id] = session_id
        self._total_sessions += 1

        # Limit sessions per user
        if len(self._sessions[user_id]) > self.max_sessions_per_user:
            oldest_session_id = min(
                self._sessions[user_id].keys(),
                key=lambda sid: self._sessions[user_id][sid].start_time
            )
            del self._sessions[user_id][oldest_session_id]

        return session_id

    def _update_session_activity(self, user_id: str, session_id: str, timestamp: float) -> None:
        """Update session activity timestamp."""
        if session_id in self._sessions[user_id]:
            session = self._sessions[user_id][session_id]
            session.last_activity = timestamp
            self._active_sessions[user_id] = session_id

    def _update_real_time_features(self, user_id: str, pattern: AccessPattern) -> None:
        """Update real-time feature caches."""
        # Update key sequences
        self._key_sequences[user_id].append(pattern.key)
        if len(self._key_sequences[user_id]) > 1000:  # Limit sequence length
            self._key_sequences[user_id] = self._key_sequences[user_id][-500:]

        # Update transition counts
        if len(self._key_sequences[user_id]) >= 2:
            from_key = self._key_sequences[user_id][-2]
            to_key = pattern.key
            self._transition_counts[user_id][(from_key, to_key)] += 1

        # Update user preferences
        key_pattern = pattern.key_pattern
        # Simple preference scoring based on access frequency and recency
        current_score = self._user_preferences[user_id][key_pattern]
        recency_factor = 1.0  # Recent accesses are weighted more
        frequency_factor = 0.1  # Each access adds to preference
        self._user_preferences[user_id][key_pattern] = current_score + recency_factor + frequency_factor

        # Add to session
        if pattern.session_id in self._sessions[user_id]:
            self._sessions[user_id][pattern.session_id].patterns.append(pattern)

    def _cleanup_old_data(self) -> None:
        """Clean up old patterns and sessions."""
        if self._total_patterns % 1000 != 0:  # Only cleanup periodically
            return

        cutoff_time = time.time() - (self.pattern_retention_days * 24 * 3600)

        with self._lock:
            for user_id in list(self._patterns.keys()):
                # Clean old patterns
                patterns = self._patterns[user_id]
                while patterns and patterns[0].timestamp < cutoff_time:
                    patterns.popleft()

                # Clean old sessions
                user_sessions = self._sessions[user_id]
                old_sessions = [
                    sid for sid, session in user_sessions.items()
                    if session.start_time < cutoff_time
                ]
                for session_id in old_sessions:
                    del user_sessions[session_id]

                # Remove empty user data
                if not patterns and not user_sessions:
                    del self._patterns[user_id]
                    del self._sessions[user_id]
                    self._active_sessions.pop(user_id, None)
                    self._key_sequences.pop(user_id, None)
                    self._transition_counts.pop(user_id, None)
                    self._user_preferences.pop(user_id, None)

    def __str__(self) -> str:
        return f"AccessPatternCollector(users={len(self._patterns)}, patterns={self._total_patterns})"

    def __repr__(self) -> str:
        return (f"AccessPatternCollector(session_timeout={self.session_timeout}, "
                f"users={len(self._patterns)}, patterns={self._total_patterns})")