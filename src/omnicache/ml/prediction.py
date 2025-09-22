"""
ML Prediction Engine for cache prefetching.

This module implements prediction engines that use trained ML models
to predict likely cache accesses and generate prefetch recommendations.
"""

import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import logging
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from omnicache.models.access_pattern import AccessPattern


logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Single prediction result."""
    key: str
    confidence: float
    prediction_type: str = "next_key"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PredictionContext:
    """Context for making predictions."""
    user_id: str
    current_key: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[float] = None
    operation: Optional[str] = None
    recent_keys: Optional[List[str]] = None
    user_features: Optional[Dict[str, Any]] = None


class PredictionEngine:
    """
    ML-powered prediction engine for cache prefetching.

    Uses trained ML models to predict likely cache accesses and
    generate prefetch recommendations based on access patterns.
    """

    def __init__(
        self,
        model_cache_size: int = 10,
        prediction_cache_ttl: float = 300,  # 5 minutes
        enable_ensemble: bool = True,
        confidence_threshold: float = 0.1,
    ):
        """
        Initialize prediction engine.

        Args:
            model_cache_size: Number of models to keep in memory
            prediction_cache_ttl: TTL for prediction cache in seconds
            enable_ensemble: Enable ensemble predictions
            confidence_threshold: Minimum confidence for predictions
        """
        self.model_cache_size = model_cache_size
        self.prediction_cache_ttl = prediction_cache_ttl
        self.enable_ensemble = enable_ensemble
        self.confidence_threshold = confidence_threshold

        # Model storage
        self._models: Dict[str, Any] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}

        # Prediction caching
        self._prediction_cache: Dict[str, Tuple[List[Prediction], float]] = {}

        # Statistics
        self._prediction_stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "model_predictions": 0,
            "fallback_predictions": 0,
        }

        logger.info("PredictionEngine initialized")

    def predict(self, context: PredictionContext, top_k: int = 5) -> List[Prediction]:
        """
        Make predictions based on context.

        Args:
            context: Prediction context
            top_k: Number of top predictions to return

        Returns:
            List of predictions ordered by confidence
        """
        # Check prediction cache
        cache_key = self._get_cache_key(context)
        if cache_key in self._prediction_cache:
            predictions, timestamp = self._prediction_cache[cache_key]
            if time.time() - timestamp < self.prediction_cache_ttl:
                self._prediction_stats["cache_hits"] += 1
                return predictions[:top_k]

        # Generate new predictions
        predictions = self._generate_predictions(context, top_k)

        # Cache predictions
        self._prediction_cache[cache_key] = (predictions, time.time())
        self._prediction_stats["total_predictions"] += 1

        return predictions

    def predict_next_keys(self, pattern: AccessPattern, top_k: int = 5) -> List[Prediction]:
        """
        Predict next likely keys based on current access pattern.

        Args:
            pattern: Current access pattern
            top_k: Number of predictions to return

        Returns:
            List of key predictions
        """
        context = PredictionContext(
            user_id=pattern.user_id,
            current_key=pattern.key,
            session_id=pattern.session_id,
            timestamp=pattern.timestamp,
            operation=pattern.operation,
        )

        return self.predict(context, top_k)

    def predict_batch(self, patterns: List[AccessPattern], top_k: int = 3) -> Dict[str, List[Prediction]]:
        """
        Make batch predictions for multiple patterns.

        Args:
            patterns: List of access patterns
            top_k: Number of predictions per pattern

        Returns:
            Dictionary mapping pattern key to predictions
        """
        results = {}

        for pattern in patterns:
            predictions = self.predict_next_keys(pattern, top_k)
            results[pattern.key] = predictions

        return results

    def load_model(self, model: Any, model_name: str = "default") -> None:
        """
        Load a trained model into the engine.

        Args:
            model: Trained ML model
            model_name: Name to identify the model
        """
        self._models[model_name] = model
        self._model_metadata[model_name] = {
            "loaded_at": time.time(),
            "prediction_count": 0,
            "last_used": time.time(),
        }

        # Limit model cache size
        if len(self._models) > self.model_cache_size:
            self._evict_least_used_model()

        logger.info(f"Loaded model: {model_name}")

    def update_model(self, new_pattern: AccessPattern, model_name: str = "default") -> None:
        """
        Update model with new pattern (online learning).

        Args:
            new_pattern: New access pattern for learning
            model_name: Name of model to update
        """
        if model_name not in self._models:
            logger.warning(f"Model {model_name} not found for update")
            return

        model = self._models[model_name]

        # For mock implementation, just log
        logger.debug(f"Updated model {model_name} with pattern: {new_pattern.key}")

        # Clear prediction cache since model changed
        self._prediction_cache.clear()

    def online_learning(self, pattern: AccessPattern) -> None:
        """
        Perform online learning with new pattern.

        Args:
            pattern: New access pattern for learning
        """
        self.update_model(pattern)

    def get_model_info(self, model_name: str = "default") -> Optional[Dict[str, Any]]:
        """
        Get information about a loaded model.

        Args:
            model_name: Name of the model

        Returns:
            Model information dictionary
        """
        if model_name not in self._model_metadata:
            return None

        metadata = self._model_metadata[model_name].copy()
        metadata["model_type"] = type(self._models[model_name]).__name__

        return metadata

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction engine statistics."""
        cache_hit_ratio = (
            self._prediction_stats["cache_hits"] / self._prediction_stats["total_predictions"]
            if self._prediction_stats["total_predictions"] > 0 else 0.0
        )

        return {
            **self._prediction_stats,
            "cache_hit_ratio": cache_hit_ratio,
            "cached_predictions": len(self._prediction_cache),
            "loaded_models": len(self._models),
            "model_cache_size": self.model_cache_size,
            "confidence_threshold": self.confidence_threshold,
        }

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self._prediction_cache.clear()
        logger.info("Prediction cache cleared")

    def _generate_predictions(self, context: PredictionContext, top_k: int) -> List[Prediction]:
        """Generate predictions using available models."""
        predictions = []

        # Try ML model predictions
        if self._models:
            ml_predictions = self._predict_with_models(context, top_k)
            predictions.extend(ml_predictions)
            self._prediction_stats["model_predictions"] += 1
        else:
            # Fallback to heuristic predictions
            fallback_predictions = self._predict_with_heuristics(context, top_k)
            predictions.extend(fallback_predictions)
            self._prediction_stats["fallback_predictions"] += 1

        # Sort by confidence and limit results
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:top_k]

    def _predict_with_models(self, context: PredictionContext, top_k: int) -> List[Prediction]:
        """Make predictions using ML models."""
        all_predictions = []

        for model_name, model in self._models.items():
            try:
                model_predictions = self._predict_with_single_model(model, context, top_k)

                # Add model info to metadata
                for pred in model_predictions:
                    pred.metadata["model_name"] = model_name

                all_predictions.extend(model_predictions)

                # Update model usage stats
                self._model_metadata[model_name]["prediction_count"] += 1
                self._model_metadata[model_name]["last_used"] = time.time()

            except Exception as e:
                logger.error(f"Error predicting with model {model_name}: {e}")

        # Ensemble predictions if enabled
        if self.enable_ensemble and len(self._models) > 1:
            return self._ensemble_predictions(all_predictions, top_k)
        else:
            return all_predictions[:top_k]

    def _predict_with_single_model(self, model: Any, context: PredictionContext, top_k: int) -> List[Prediction]:
        """Make predictions using a single model."""
        predictions = []

        # Handle different model types
        if isinstance(model, dict) and model.get("type") == "ngram":
            predictions = self._predict_with_ngram_model(model, context, top_k)
        elif isinstance(model, dict) and model.get("type") == "mock":
            predictions = self._predict_with_mock_model(context, top_k)
        else:
            # Try scikit-learn model
            predictions = self._predict_with_sklearn_model(model, context, top_k)

        return predictions

    def _predict_with_ngram_model(self, model: Dict[str, Any], context: PredictionContext, top_k: int) -> List[Prediction]:
        """Make predictions using n-gram model."""
        predictions = []
        probabilities = model.get("probabilities", {})

        if context.recent_keys:
            # Use recent keys as context
            sequence_length = model.get("sequence_length", 3)
            ngram_context = tuple(context.recent_keys[-sequence_length:])

            # Try exact match first
            if ngram_context in probabilities:
                target_probs = probabilities[ngram_context]
                for key, prob in sorted(target_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                    if prob >= self.confidence_threshold:
                        predictions.append(Prediction(
                            key=key,
                            confidence=prob,
                            prediction_type="ngram",
                            metadata={"context": ngram_context}
                        ))

            # Try shorter contexts if no exact match
            for context_len in range(len(ngram_context) - 1, 0, -1):
                if predictions:
                    break

                short_context = ngram_context[-context_len:]
                for full_context, target_probs in probabilities.items():
                    if full_context[-context_len:] == short_context:
                        for key, prob in sorted(target_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                            if prob >= self.confidence_threshold:
                                predictions.append(Prediction(
                                    key=key,
                                    confidence=prob * 0.8,  # Penalty for shorter context
                                    prediction_type="ngram_partial",
                                    metadata={"context": short_context}
                                ))
                        break

        return predictions

    def _predict_with_sklearn_model(self, model: Any, context: PredictionContext, top_k: int) -> List[Prediction]:
        """Make predictions using scikit-learn model."""
        predictions = []

        try:
            # Create features from context
            features = self._create_features_from_context(context)

            if hasattr(model, 'predict_proba'):
                # Get prediction probabilities
                probabilities = model.predict_proba([features])[0]
                classes = model.classes_

                # Create predictions from probabilities
                for i, prob in enumerate(probabilities):
                    if prob >= self.confidence_threshold:
                        predictions.append(Prediction(
                            key=classes[i],
                            confidence=prob,
                            prediction_type="sklearn",
                            metadata={"features": len(features)}
                        ))

            elif hasattr(model, 'predict'):
                # Simple prediction without probabilities
                predicted_key = model.predict([features])[0]
                predictions.append(Prediction(
                    key=predicted_key,
                    confidence=0.7,  # Default confidence
                    prediction_type="sklearn",
                    metadata={"features": len(features)}
                ))

        except Exception as e:
            logger.error(f"Error with scikit-learn prediction: {e}")

        return predictions[:top_k]

    def _predict_with_mock_model(self, context: PredictionContext, top_k: int) -> List[Prediction]:
        """Make mock predictions for testing."""
        predictions = []

        # Generate mock predictions based on current key
        if context.current_key:
            base_key = context.current_key.split(':')[0] if ':' in context.current_key else context.current_key

            for i in range(min(top_k, 3)):
                mock_key = f"{base_key}:related_{i+1}"
                confidence = 0.9 - (i * 0.2)

                predictions.append(Prediction(
                    key=mock_key,
                    confidence=confidence,
                    prediction_type="mock",
                    metadata={"base_key": base_key}
                ))

        return predictions

    def _predict_with_heuristics(self, context: PredictionContext, top_k: int) -> List[Prediction]:
        """Make heuristic-based predictions as fallback."""
        predictions = []

        if context.current_key:
            # Pattern-based predictions
            key_parts = context.current_key.split(':')

            # Common patterns
            patterns = [
                f"{':'.join(key_parts[:-1])}:related",
                f"{':'.join(key_parts[:-1])}:details",
                f"{':'.join(key_parts[:-1])}:metadata",
            ]

            for i, pattern in enumerate(patterns[:top_k]):
                confidence = 0.6 - (i * 0.1)
                predictions.append(Prediction(
                    key=pattern,
                    confidence=confidence,
                    prediction_type="heuristic",
                    metadata={"pattern": "related_keys"}
                ))

        return predictions

    def _create_features_from_context(self, context: PredictionContext) -> List[float]:
        """Create feature vector from prediction context."""
        features = []

        # Mock feature creation
        if context.timestamp:
            dt = time.localtime(context.timestamp)
            features.extend([
                float(dt.tm_hour),  # hour of day
                float(dt.tm_wday),  # day of week
                float(dt.tm_wday >= 5),  # is weekend
            ])
        else:
            features.extend([12.0, 1.0, 0.0])  # Default values

        # Key features
        if context.current_key:
            features.extend([
                float(len(context.current_key)),
                float(context.current_key.count(':')),
            ])
        else:
            features.extend([10.0, 2.0])

        # Operation features
        if context.operation:
            op_encoding = {"GET": 1.0, "SET": 2.0, "DELETE": 3.0}.get(context.operation.upper(), 0.0)
            features.append(op_encoding)
        else:
            features.append(1.0)

        return features

    def _ensemble_predictions(self, all_predictions: List[Prediction], top_k: int) -> List[Prediction]:
        """Combine predictions from multiple models."""
        # Group predictions by key
        key_predictions = defaultdict(list)
        for pred in all_predictions:
            key_predictions[pred.key].append(pred)

        # Aggregate predictions for each key
        ensemble_predictions = []
        for key, preds in key_predictions.items():
            # Simple average of confidences
            avg_confidence = sum(p.confidence for p in preds) / len(preds)

            # Boost confidence for keys predicted by multiple models
            if len(preds) > 1:
                avg_confidence *= 1.2

            ensemble_predictions.append(Prediction(
                key=key,
                confidence=min(avg_confidence, 1.0),
                prediction_type="ensemble",
                metadata={
                    "model_count": len(preds),
                    "contributing_models": [p.metadata.get("model_name", "unknown") for p in preds]
                }
            ))

        # Sort and return top predictions
        ensemble_predictions.sort(key=lambda p: p.confidence, reverse=True)
        return ensemble_predictions[:top_k]

    def _get_cache_key(self, context: PredictionContext) -> str:
        """Generate cache key for prediction context."""
        return f"{context.user_id}:{context.current_key}:{context.session_id}"

    def _evict_least_used_model(self) -> None:
        """Evict the least recently used model."""
        if not self._model_metadata:
            return

        # Find least recently used model
        lru_model = min(
            self._model_metadata.keys(),
            key=lambda name: self._model_metadata[name]["last_used"]
        )

        # Remove model and metadata
        del self._models[lru_model]
        del self._model_metadata[lru_model]

        logger.info(f"Evicted model: {lru_model}")

    def __str__(self) -> str:
        return f"PredictionEngine(models={len(self._models)}, cached_predictions={len(self._prediction_cache)})"

    def __repr__(self) -> str:
        return (f"PredictionEngine(models={len(self._models)}, "
                f"cache_ttl={self.prediction_cache_ttl}, "
                f"confidence_threshold={self.confidence_threshold})")