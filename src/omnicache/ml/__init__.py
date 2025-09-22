"""
Machine Learning module for predictive caching and analytics.

This module provides ML-powered features for OmniCache including:
- Access pattern analysis and prediction
- Predictive prefetching
- Model training and management
- Performance optimization through machine learning
"""

from .collectors import AccessPatternCollector, Session
from .training import ModelTrainer, ModelMetrics, TrainingConfig
from .prediction import PredictionEngine, Prediction, PredictionContext
from .prefetch import PrefetchRecommendationSystem, PrefetchRecommendation, PrefetchPriority

__all__ = [
    "AccessPatternCollector",
    "Session",
    "ModelTrainer",
    "ModelMetrics",
    "TrainingConfig",
    "PredictionEngine",
    "Prediction",
    "PredictionContext",
    "PrefetchRecommendationSystem",
    "PrefetchRecommendation",
    "PrefetchPriority",
]

__version__ = "0.1.0"