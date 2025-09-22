"""
Prefetch Recommendation System for intelligent cache preloading.

This module implements a comprehensive prefetch recommendation system that uses
ML predictions, user behavior analysis, and cache state to generate intelligent
prefetch recommendations for optimal cache performance.
"""

import time
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

from omnicache.models.access_pattern import AccessPattern
from omnicache.ml.prediction import PredictionEngine, Prediction, PredictionContext


logger = logging.getLogger(__name__)


class PrefetchPriority(Enum):
    """Priority levels for prefetch recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PrefetchRecommendation:
    """Single prefetch recommendation."""
    key: str
    priority: PrefetchPriority
    confidence: float
    estimated_access_time: Optional[float] = None
    data_source: Optional[str] = None
    cache_tier: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def priority_score(self) -> float:
        """Get numerical priority score for sorting."""
        priority_scores = {
            PrefetchPriority.CRITICAL: 4.0,
            PrefetchPriority.HIGH: 3.0,
            PrefetchPriority.MEDIUM: 2.0,
            PrefetchPriority.LOW: 1.0,
        }
        return priority_scores[self.priority]


@dataclass
class PrefetchConfig:
    """Configuration for prefetch recommendation system."""
    max_recommendations: int = 10
    confidence_threshold: float = 0.3
    priority_threshold: PrefetchPriority = PrefetchPriority.LOW
    model_update_interval: float = 3600  # 1 hour
    enable_real_time_learning: bool = True
    cache_state_weight: float = 0.3
    temporal_weight: float = 0.2
    user_pattern_weight: float = 0.5
    prefetch_horizon_seconds: float = 300  # 5 minutes


class PrefetchRecommendationSystem:
    """
    Intelligent prefetch recommendation system.

    Combines ML predictions, user behavior analysis, cache state monitoring,
    and temporal patterns to generate optimal prefetch recommendations.
    """

    def __init__(
        self,
        prediction_engine: Optional[PredictionEngine] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize prefetch recommendation system.

        Args:
            prediction_engine: ML prediction engine
            config: System configuration
        """
        # Configuration
        if config is None:
            config = {}

        self.max_recommendations = config.get("max_recommendations", 10)
        self.confidence_threshold = config.get("confidence_threshold", 0.3)
        self.model_update_interval = config.get("model_update_interval", 3600)
        self.enable_real_time_learning = config.get("enable_real_time_learning", True)

        self.config = PrefetchConfig(
            max_recommendations=self.max_recommendations,
            confidence_threshold=self.confidence_threshold,
            model_update_interval=self.model_update_interval,
            enable_real_time_learning=self.enable_real_time_learning,
        )

        # ML prediction engine
        self.prediction_engine = prediction_engine or PredictionEngine()

        # User behavior tracking
        self._user_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._user_preferences: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._temporal_patterns: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Cache state monitoring
        self._cache_state: Dict[str, Set[str]] = {}  # tier -> cached_keys
        self._cache_hit_rates: Dict[str, float] = defaultdict(float)
        self._prefetch_success_rates: Dict[str, float] = defaultdict(float)

        # Recommendation tracking
        self._active_recommendations: Dict[str, List[PrefetchRecommendation]] = defaultdict(list)
        self._recommendation_history: deque = deque(maxlen=10000)

        # Statistics
        self._stats = {
            "total_recommendations": 0,
            "successful_prefetches": 0,
            "failed_prefetches": 0,
            "cache_hits_after_prefetch": 0,
        }

        logger.info("PrefetchRecommendationSystem initialized")

    def recommend(
        self,
        user_id: str,
        current_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        exclude_cached: Optional[List[str]] = None,
        max_recommendations: Optional[int] = None,
    ) -> List[PrefetchRecommendation]:
        """
        Generate prefetch recommendations for a user.

        Args:
            user_id: User identifier
            current_key: Current cache key being accessed
            context: Additional context information
            exclude_cached: Keys to exclude (already cached)
            max_recommendations: Maximum number of recommendations

        Returns:
            List of prefetch recommendations
        """
        if context is None:
            context = {}

        if exclude_cached is None:
            exclude_cached = []

        max_recs = max_recommendations or self.max_recommendations

        # Create prediction context
        prediction_context = PredictionContext(
            user_id=user_id,
            current_key=current_key,
            session_id=context.get("session_id"),
            timestamp=context.get("timestamp", time.time()),
            operation=context.get("operation", "GET"),
            recent_keys=self._get_recent_keys(user_id),
            user_features=self._get_user_features(user_id),
        )

        # Generate recommendations from multiple sources
        recommendations = []

        # ML-based predictions
        ml_recommendations = self._get_ml_recommendations(prediction_context, max_recs)
        recommendations.extend(ml_recommendations)

        # Pattern-based recommendations
        pattern_recommendations = self._get_pattern_recommendations(user_id, current_key, max_recs)
        recommendations.extend(pattern_recommendations)

        # Temporal recommendations
        temporal_recommendations = self._get_temporal_recommendations(user_id, max_recs)
        recommendations.extend(temporal_recommendations)

        # User preference recommendations
        preference_recommendations = self._get_preference_recommendations(user_id, max_recs)
        recommendations.extend(preference_recommendations)

        # Filter and rank recommendations
        filtered_recommendations = self._filter_recommendations(
            recommendations, exclude_cached, user_id
        )

        ranked_recommendations = self._rank_recommendations(
            filtered_recommendations, prediction_context
        )

        # Limit to max recommendations
        final_recommendations = ranked_recommendations[:max_recs]

        # Track recommendations
        self._active_recommendations[user_id] = final_recommendations
        self._stats["total_recommendations"] += len(final_recommendations)

        logger.debug(f"Generated {len(final_recommendations)} recommendations for user {user_id}")

        return final_recommendations

    def update_user_access(self, pattern: AccessPattern) -> None:
        """
        Update user access patterns for learning.

        Args:
            pattern: New access pattern
        """
        user_id = pattern.user_id

        # Update user patterns
        self._user_patterns[user_id].append(pattern)

        # Update user preferences
        key_pattern = pattern.key_pattern
        self._user_preferences[user_id][key_pattern] += 1.0

        # Update temporal patterns
        hour = pattern.hour_of_day
        self._temporal_patterns[user_id][hour].append(pattern.key)

        # Limit temporal pattern storage
        if len(self._temporal_patterns[user_id][hour]) > 100:
            self._temporal_patterns[user_id][hour] = self._temporal_patterns[user_id][hour][-50:]

        # Update ML model if real-time learning is enabled
        if self.enable_real_time_learning:
            self.prediction_engine.update_model(pattern)

        logger.debug(f"Updated access patterns for user {user_id}")

    def update_cache_state(self, tier: str, cached_keys: Set[str]) -> None:
        """
        Update cache state for better recommendations.

        Args:
            tier: Cache tier name
            cached_keys: Set of currently cached keys
        """
        self._cache_state[tier] = cached_keys.copy()

    def record_prefetch_success(self, user_id: str, key: str, successful: bool) -> None:
        """
        Record prefetch success/failure for learning.

        Args:
            user_id: User identifier
            key: Prefetched key
            successful: Whether prefetch was successful (used)
        """
        if successful:
            self._stats["successful_prefetches"] += 1
            self._prefetch_success_rates[key] = min(
                self._prefetch_success_rates[key] + 0.1, 1.0
            )
        else:
            self._stats["failed_prefetches"] += 1
            self._prefetch_success_rates[key] = max(
                self._prefetch_success_rates[key] - 0.05, 0.0
            )

        # Update user preferences based on success
        if successful:
            key_pattern = key.split(':')[0] if ':' in key else key
            self._user_preferences[user_id][key_pattern] += 0.5

    def get_statistics(self) -> Dict[str, Any]:
        """Get prefetch system statistics."""
        total_prefetches = self._stats["successful_prefetches"] + self._stats["failed_prefetches"]
        success_rate = (
            self._stats["successful_prefetches"] / total_prefetches
            if total_prefetches > 0 else 0.0
        )

        return {
            **self._stats,
            "prefetch_success_rate": success_rate,
            "active_users": len(self._user_patterns),
            "total_user_patterns": sum(len(patterns) for patterns in self._user_patterns.values()),
            "cached_tiers": len(self._cache_state),
            "prediction_engine_stats": self.prediction_engine.get_prediction_stats(),
        }

    def clear_user_data(self, user_id: str) -> None:
        """
        Clear all data for a specific user.

        Args:
            user_id: User identifier
        """
        if user_id in self._user_patterns:
            del self._user_patterns[user_id]
        if user_id in self._user_preferences:
            del self._user_preferences[user_id]
        if user_id in self._temporal_patterns:
            del self._temporal_patterns[user_id]
        if user_id in self._active_recommendations:
            del self._active_recommendations[user_id]

        logger.info(f"Cleared prefetch data for user: {user_id}")

    def _get_ml_recommendations(self, context: PredictionContext, max_recs: int) -> List[PrefetchRecommendation]:
        """Get recommendations from ML prediction engine."""
        try:
            predictions = self.prediction_engine.predict(context, max_recs * 2)
            recommendations = []

            for pred in predictions:
                if pred.confidence >= self.confidence_threshold:
                    priority = self._confidence_to_priority(pred.confidence)

                    recommendation = PrefetchRecommendation(
                        key=pred.key,
                        priority=priority,
                        confidence=pred.confidence,
                        metadata={
                            "source": "ml_prediction",
                            "prediction_type": pred.prediction_type,
                            **pred.metadata
                        }
                    )
                    recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error(f"Error getting ML recommendations: {e}")
            return []

    def _get_pattern_recommendations(self, user_id: str, current_key: Optional[str], max_recs: int) -> List[PrefetchRecommendation]:
        """Get recommendations based on user access patterns."""
        recommendations = []

        if not current_key or user_id not in self._user_patterns:
            return recommendations

        user_patterns = list(self._user_patterns[user_id])

        # Find sequences that contain current key
        sequences = []
        for i, pattern in enumerate(user_patterns):
            if pattern.key == current_key and i < len(user_patterns) - 1:
                # Look ahead for next few keys
                next_keys = [
                    user_patterns[j].key
                    for j in range(i + 1, min(i + 4, len(user_patterns)))
                    if user_patterns[j].user_id == user_id
                ]
                sequences.extend(next_keys)

        # Count frequency of next keys
        key_counts = defaultdict(int)
        for key in sequences:
            key_counts[key] += 1

        # Create recommendations from most frequent patterns
        total_occurrences = sum(key_counts.values())
        for key, count in sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:max_recs]:
            confidence = count / total_occurrences if total_occurrences > 0 else 0.1
            priority = self._confidence_to_priority(confidence)

            recommendation = PrefetchRecommendation(
                key=key,
                priority=priority,
                confidence=confidence,
                metadata={
                    "source": "access_pattern",
                    "frequency": count,
                    "total_sequences": len(sequences)
                }
            )
            recommendations.append(recommendation)

        return recommendations

    def _get_temporal_recommendations(self, user_id: str, max_recs: int) -> List[PrefetchRecommendation]:
        """Get recommendations based on temporal access patterns."""
        recommendations = []

        if user_id not in self._temporal_patterns:
            return recommendations

        current_hour = time.localtime().tm_hour
        temporal_patterns = self._temporal_patterns[user_id]

        # Get keys commonly accessed at this hour
        hourly_keys = temporal_patterns.get(current_hour, [])

        if hourly_keys:
            key_counts = defaultdict(int)
            for key in hourly_keys:
                key_counts[key] += 1

            total_accesses = len(hourly_keys)
            for key, count in sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:max_recs]:
                confidence = count / total_accesses if total_accesses > 0 else 0.1
                priority = self._confidence_to_priority(confidence * 0.7)  # Lower priority for temporal

                recommendation = PrefetchRecommendation(
                    key=key,
                    priority=priority,
                    confidence=confidence,
                    estimated_access_time=time.time() + 300,  # Estimate 5 minutes
                    metadata={
                        "source": "temporal_pattern",
                        "hour": current_hour,
                        "frequency": count,
                        "total_hourly_accesses": total_accesses
                    }
                )
                recommendations.append(recommendation)

        return recommendations

    def _get_preference_recommendations(self, user_id: str, max_recs: int) -> List[PrefetchRecommendation]:
        """Get recommendations based on user preferences."""
        recommendations = []

        if user_id not in self._user_preferences:
            return recommendations

        preferences = self._user_preferences[user_id]
        total_preference = sum(preferences.values())

        for key_pattern, preference_score in sorted(preferences.items(), key=lambda x: x[1], reverse=True)[:max_recs]:
            confidence = preference_score / total_preference if total_preference > 0 else 0.1
            priority = self._confidence_to_priority(confidence * 0.8)  # Lower priority for preferences

            # Generate specific key from pattern
            specific_key = f"{key_pattern}:prefetch_{int(time.time())}"

            recommendation = PrefetchRecommendation(
                key=specific_key,
                priority=priority,
                confidence=confidence,
                metadata={
                    "source": "user_preference",
                    "key_pattern": key_pattern,
                    "preference_score": preference_score
                }
            )
            recommendations.append(recommendation)

        return recommendations

    def _filter_recommendations(self, recommendations: List[PrefetchRecommendation], exclude_cached: List[str], user_id: str) -> List[PrefetchRecommendation]:
        """Filter recommendations based on cache state and exclusions."""
        filtered = []

        exclude_set = set(exclude_cached)

        # Add currently cached keys to exclusion set
        for tier_keys in self._cache_state.values():
            exclude_set.update(tier_keys)

        for rec in recommendations:
            # Skip if key is already cached or excluded
            if rec.key in exclude_set:
                continue

            # Skip if confidence is below threshold
            if rec.confidence < self.confidence_threshold:
                continue

            # Skip if we have too many active recommendations for this user
            if len(self._active_recommendations[user_id]) >= self.max_recommendations:
                continue

            filtered.append(rec)

        return filtered

    def _rank_recommendations(self, recommendations: List[PrefetchRecommendation], context: PredictionContext) -> List[PrefetchRecommendation]:
        """Rank recommendations by composite score."""
        for rec in recommendations:
            # Calculate composite score
            confidence_score = rec.confidence
            priority_score = rec.priority_score / 4.0  # Normalize to 0-1

            # Historical success rate for this key
            success_rate = self._prefetch_success_rates.get(rec.key, 0.5)

            # Composite score with weights
            composite_score = (
                confidence_score * self.config.user_pattern_weight +
                priority_score * self.config.cache_state_weight +
                success_rate * self.config.temporal_weight
            )

            rec.metadata["composite_score"] = composite_score

        # Sort by composite score
        return sorted(recommendations, key=lambda r: r.metadata["composite_score"], reverse=True)

    def _confidence_to_priority(self, confidence: float) -> PrefetchPriority:
        """Convert confidence score to priority level."""
        if confidence >= 0.8:
            return PrefetchPriority.CRITICAL
        elif confidence >= 0.6:
            return PrefetchPriority.HIGH
        elif confidence >= 0.4:
            return PrefetchPriority.MEDIUM
        else:
            return PrefetchPriority.LOW

    def _get_recent_keys(self, user_id: str, limit: int = 10) -> List[str]:
        """Get recent keys accessed by user."""
        if user_id not in self._user_patterns:
            return []

        recent_patterns = list(self._user_patterns[user_id])[-limit:]
        return [pattern.key for pattern in recent_patterns]

    def _get_user_features(self, user_id: str) -> Dict[str, Any]:
        """Get user-specific features for prediction."""
        if user_id not in self._user_patterns:
            return {}

        patterns = list(self._user_patterns[user_id])

        if not patterns:
            return {}

        # Calculate user features
        total_accesses = len(patterns)
        unique_keys = len(set(pattern.key for pattern in patterns))
        avg_session_length = total_accesses / len(set(pattern.session_id for pattern in patterns))

        recent_patterns = patterns[-10:] if len(patterns) >= 10 else patterns
        recent_operations = [pattern.operation for pattern in recent_patterns]
        read_ratio = sum(1 for op in recent_operations if op.upper() == 'GET') / len(recent_operations)

        return {
            "total_accesses": total_accesses,
            "unique_keys": unique_keys,
            "diversity_ratio": unique_keys / total_accesses,
            "avg_session_length": avg_session_length,
            "read_ratio": read_ratio,
            "active_hours": len(set(pattern.hour_of_day for pattern in patterns)),
        }

    def __str__(self) -> str:
        return f"PrefetchRecommendationSystem(users={len(self._user_patterns)}, max_recs={self.max_recommendations})"

    def __repr__(self) -> str:
        return (f"PrefetchRecommendationSystem(config={self.config}, "
                f"users={len(self._user_patterns)}, engine={self.prediction_engine})")