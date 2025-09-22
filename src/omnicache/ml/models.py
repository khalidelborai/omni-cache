"""
ML prediction model classes for cache prefetching.

This module defines machine learning models for cache prediction,
including pattern recognition, prefetching, and optimization.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import time
import json
import numpy as np
import pickle
import base64


class ModelType(Enum):
    """Types of ML models."""
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CLUSTERING = "clustering"
    ASSOCIATION_RULES = "association_rules"


class PredictionType(Enum):
    """Types of predictions."""
    ACCESS_PROBABILITY = "access_probability"
    PREFETCH_KEYS = "prefetch_keys"
    TTL_OPTIMIZATION = "ttl_optimization"
    EVICTION_PRIORITY = "eviction_priority"
    HOTSPOT_DETECTION = "hotspot_detection"


class TrainingStatus(Enum):
    """Training status of models."""
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"
    UPDATING = "updating"


@dataclass
class ModelFeatures:
    """Feature vector for ML models."""
    numerical_features: Dict[str, float] = field(default_factory=dict)
    categorical_features: Dict[str, str] = field(default_factory=dict)
    temporal_features: Dict[str, float] = field(default_factory=dict)
    sequence_features: List[Dict[str, Any]] = field(default_factory=list)

    def to_vector(self, feature_map: Dict[str, int]) -> List[float]:
        """Convert features to numerical vector."""
        vector = [0.0] * len(feature_map)

        # Add numerical features
        for name, value in self.numerical_features.items():
            if name in feature_map:
                vector[feature_map[name]] = value

        # Add categorical features (one-hot encoded)
        for name, value in self.categorical_features.items():
            feature_name = f"{name}_{value}"
            if feature_name in feature_map:
                vector[feature_map[feature_name]] = 1.0

        # Add temporal features
        for name, value in self.temporal_features.items():
            if name in feature_map:
                vector[feature_map[name]] = value

        return vector

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "temporal_features": self.temporal_features,
            "sequence_features": self.sequence_features,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelFeatures':
        """Create from dictionary."""
        return cls(
            numerical_features=data.get("numerical_features", {}),
            categorical_features=data.get("categorical_features", {}),
            temporal_features=data.get("temporal_features", {}),
            sequence_features=data.get("sequence_features", []),
        )


@dataclass
class PredictionResult:
    """Result of a model prediction."""
    prediction_type: PredictionType
    confidence: float
    value: Any
    features_used: List[str] = field(default_factory=list)
    model_version: str = "1.0"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate prediction result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction_type": self.prediction_type.value,
            "confidence": self.confidence,
            "value": self.value,
            "features_used": self.features_used,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResult':
        """Create from dictionary."""
        return cls(
            prediction_type=PredictionType(data["prediction_type"]),
            confidence=data["confidence"],
            value=data["value"],
            features_used=data.get("features_used", []),
            model_version=data.get("model_version", "1.0"),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TrainingData:
    """Training data for ML models."""
    features: List[ModelFeatures] = field(default_factory=list)
    labels: List[Any] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Validate training data."""
        if len(self.features) != len(self.labels):
            raise ValueError("Features and labels must have same length")

        if self.weights and len(self.weights) != len(self.features):
            raise ValueError("Weights must have same length as features")

        if self.timestamps and len(self.timestamps) != len(self.features):
            raise ValueError("Timestamps must have same length as features")

    def add_sample(self, features: ModelFeatures, label: Any,
                  weight: float = 1.0, timestamp: Optional[float] = None) -> None:
        """Add a training sample."""
        self.features.append(features)
        self.labels.append(label)
        self.weights.append(weight)
        self.timestamps.append(timestamp or time.time())

    def get_batch(self, batch_size: int, start_index: int = 0) -> 'TrainingData':
        """Get a batch of training data."""
        end_index = min(start_index + batch_size, len(self.features))
        return TrainingData(
            features=self.features[start_index:end_index],
            labels=self.labels[start_index:end_index],
            weights=self.weights[start_index:end_index] if self.weights else [],
            timestamps=self.timestamps[start_index:end_index] if self.timestamps else [],
        )

    def shuffle(self) -> None:
        """Shuffle the training data."""
        import random
        indices = list(range(len(self.features)))
        random.shuffle(indices)

        self.features = [self.features[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        if self.weights:
            self.weights = [self.weights[i] for i in indices]
        if self.timestamps:
            self.timestamps = [self.timestamps[i] for i in indices]

    @property
    def size(self) -> int:
        """Get size of training data."""
        return len(self.features)


@dataclass
class ModelMetrics:
    """Performance metrics for ML models."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    mean_absolute_error: float = 0.0
    mean_squared_error: float = 0.0
    r2_score: float = 0.0
    training_time_seconds: float = 0.0
    prediction_time_ms: float = 0.0
    model_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_score": self.auc_score,
            "mean_absolute_error": self.mean_absolute_error,
            "mean_squared_error": self.mean_squared_error,
            "r2_score": self.r2_score,
            "training_time_seconds": self.training_time_seconds,
            "prediction_time_ms": self.prediction_time_ms,
            "model_size_bytes": self.model_size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        """Create from dictionary."""
        return cls(
            accuracy=data.get("accuracy", 0.0),
            precision=data.get("precision", 0.0),
            recall=data.get("recall", 0.0),
            f1_score=data.get("f1_score", 0.0),
            auc_score=data.get("auc_score", 0.0),
            mean_absolute_error=data.get("mean_absolute_error", 0.0),
            mean_squared_error=data.get("mean_squared_error", 0.0),
            r2_score=data.get("r2_score", 0.0),
            training_time_seconds=data.get("training_time_seconds", 0.0),
            prediction_time_ms=data.get("prediction_time_ms", 0.0),
            model_size_bytes=data.get("model_size_bytes", 0),
        )


@dataclass
class CachePredictionModel:
    """
    Base class for cache prediction models.

    Provides infrastructure for training and using ML models
    for cache optimization and prefetching.
    """

    name: str
    model_type: ModelType
    prediction_type: PredictionType
    description: str = ""

    # Model state
    status: TrainingStatus = TrainingStatus.UNTRAINED
    version: str = "1.0"
    feature_map: Dict[str, int] = field(default_factory=dict)
    _model_data: Optional[bytes] = None

    # Configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_selection: List[str] = field(default_factory=list)
    max_training_samples: int = 100000
    min_training_samples: int = 1000
    retrain_threshold: float = 0.1  # Retrain if accuracy drops below this

    # Performance tracking
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    training_history: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: float = field(default_factory=time.time)
    last_trained_at: Optional[float] = None
    last_prediction_at: Optional[float] = None
    total_predictions: int = 0

    def __post_init__(self):
        """Post-initialization setup."""
        if not self.name:
            raise ValueError("Model name is required")

        if self.min_training_samples <= 0:
            raise ValueError("Min training samples must be positive")

        if self.max_training_samples < self.min_training_samples:
            raise ValueError("Max training samples must be >= min training samples")

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.status == TrainingStatus.TRAINED

    @property
    def can_predict(self) -> bool:
        """Check if model can make predictions."""
        return self.is_trained and self._model_data is not None

    def extract_features(self, access_pattern: Any) -> ModelFeatures:
        """
        Extract features from access pattern.

        Args:
            access_pattern: AccessPattern object

        Returns:
            ModelFeatures object
        """
        features = ModelFeatures()

        # Extract basic features
        if hasattr(access_pattern, 'extract_features'):
            basic_features = access_pattern.extract_features()

            # Categorize features
            for name, value in basic_features.items():
                if isinstance(value, (int, float)):
                    features.numerical_features[name] = float(value)
                elif isinstance(value, str):
                    features.categorical_features[name] = value
                elif isinstance(value, bool):
                    features.numerical_features[name] = float(value)

        # Extract temporal features
        if hasattr(access_pattern, 'hour_of_day'):
            features.temporal_features.update({
                "hour_of_day": access_pattern.hour_of_day,
                "day_of_week": access_pattern.day_of_week,
                "is_weekend": float(access_pattern.is_weekend),
            })

        return features

    def build_feature_map(self, training_data: TrainingData) -> None:
        """Build feature map from training data."""
        feature_names = set()

        for features in training_data.features:
            # Add numerical features
            feature_names.update(features.numerical_features.keys())

            # Add categorical features (one-hot encoded)
            for name, value in features.categorical_features.items():
                feature_names.add(f"{name}_{value}")

            # Add temporal features
            feature_names.update(features.temporal_features.keys())

        # Create feature map
        self.feature_map = {name: i for i, name in enumerate(sorted(feature_names))}

    def prepare_training_data(self, training_data: TrainingData) -> Tuple[List[List[float]], List[Any]]:
        """Prepare training data for model."""
        if not self.feature_map:
            self.build_feature_map(training_data)

        X = []
        y = []

        for features, label in zip(training_data.features, training_data.labels):
            feature_vector = features.to_vector(self.feature_map)
            X.append(feature_vector)
            y.append(label)

        return X, y

    def train(self, training_data: TrainingData) -> bool:
        """
        Train the model.

        Args:
            training_data: Training data

        Returns:
            True if training succeeded
        """
        if training_data.size < self.min_training_samples:
            raise ValueError(f"Need at least {self.min_training_samples} samples for training")

        self.status = TrainingStatus.TRAINING
        start_time = time.time()

        try:
            # Prepare data
            X, y = self.prepare_training_data(training_data)

            # Train model based on type
            model = self._train_model(X, y)

            # Serialize model
            self._model_data = pickle.dumps(model)

            # Update metrics
            self.metrics.training_time_seconds = time.time() - start_time
            self.metrics.model_size_bytes = len(self._model_data)

            # Evaluate model
            self._evaluate_model(X, y, model)

            # Update status
            self.status = TrainingStatus.TRAINED
            self.last_trained_at = time.time()

            # Add to training history
            self.training_history.append({
                "timestamp": self.last_trained_at,
                "samples": training_data.size,
                "metrics": self.metrics.to_dict(),
                "version": self.version,
            })

            return True

        except Exception as e:
            self.status = TrainingStatus.FAILED
            raise e

    def _train_model(self, X: List[List[float]], y: List[Any]) -> Any:
        """Train the specific model type."""
        if self.model_type == ModelType.LINEAR_REGRESSION:
            return self._train_linear_regression(X, y)
        elif self.model_type == ModelType.DECISION_TREE:
            return self._train_decision_tree(X, y)
        elif self.model_type == ModelType.RANDOM_FOREST:
            return self._train_random_forest(X, y)
        elif self.model_type == ModelType.CLUSTERING:
            return self._train_clustering(X)
        else:
            # For now, use a simple linear model as fallback
            return self._train_linear_regression(X, y)

    def _train_linear_regression(self, X: List[List[float]], y: List[Any]) -> Dict[str, Any]:
        """Train a simple linear regression model."""
        # Simple implementation using numpy-like operations
        X_array = np.array(X) if X else np.array([[]])
        y_array = np.array(y) if y else np.array([])

        if X_array.size == 0 or y_array.size == 0:
            return {"weights": [], "intercept": 0.0}

        # Add bias term
        X_with_bias = np.column_stack([np.ones(len(X_array)), X_array])

        # Solve normal equation: (X^T * X)^-1 * X^T * y
        try:
            XtX = X_with_bias.T @ X_with_bias
            Xty = X_with_bias.T @ y_array
            params = np.linalg.solve(XtX, Xty)

            return {
                "weights": params[1:].tolist(),
                "intercept": params[0],
                "feature_names": list(self.feature_map.keys())
            }
        except np.linalg.LinAlgError:
            # Fallback to simple averages
            return {
                "weights": [0.0] * len(self.feature_map),
                "intercept": np.mean(y_array) if len(y_array) > 0 else 0.0,
                "feature_names": list(self.feature_map.keys())
            }

    def _train_decision_tree(self, X: List[List[float]], y: List[Any]) -> Dict[str, Any]:
        """Train a simple decision tree model."""
        # Simplified decision tree implementation
        return {
            "type": "decision_tree",
            "rules": [],  # Would contain actual decision rules
            "feature_names": list(self.feature_map.keys())
        }

    def _train_random_forest(self, X: List[List[float]], y: List[Any]) -> Dict[str, Any]:
        """Train a random forest model."""
        # Simplified random forest implementation
        return {
            "type": "random_forest",
            "trees": [],  # Would contain multiple decision trees
            "feature_names": list(self.feature_map.keys())
        }

    def _train_clustering(self, X: List[List[float]]) -> Dict[str, Any]:
        """Train a clustering model."""
        # Simplified clustering implementation
        return {
            "type": "clustering",
            "centroids": [],  # Would contain cluster centroids
            "feature_names": list(self.feature_map.keys())
        }

    def _evaluate_model(self, X: List[List[float]], y: List[Any], model: Any) -> None:
        """Evaluate model performance."""
        if not X or not y:
            return

        # Make predictions
        predictions = []
        for x in X:
            pred_result = self._predict_with_model(model, x)
            predictions.append(pred_result)

        # Calculate basic metrics
        if self.prediction_type == PredictionType.ACCESS_PROBABILITY:
            self._calculate_classification_metrics(y, predictions)
        else:
            self._calculate_regression_metrics(y, predictions)

    def _calculate_classification_metrics(self, y_true: List[Any], y_pred: List[Any]) -> None:
        """Calculate classification metrics."""
        if not y_true or not y_pred:
            return

        # Simple accuracy calculation
        correct = sum(1 for true, pred in zip(y_true, y_pred) if abs(true - pred) < 0.5)
        self.metrics.accuracy = correct / len(y_true)

    def _calculate_regression_metrics(self, y_true: List[Any], y_pred: List[Any]) -> None:
        """Calculate regression metrics."""
        if not y_true or not y_pred:
            return

        # Calculate MAE and MSE
        errors = [abs(true - pred) for true, pred in zip(y_true, y_pred)]
        squared_errors = [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]

        self.metrics.mean_absolute_error = sum(errors) / len(errors)
        self.metrics.mean_squared_error = sum(squared_errors) / len(squared_errors)

    def predict(self, features: ModelFeatures) -> PredictionResult:
        """
        Make a prediction using the trained model.

        Args:
            features: Input features

        Returns:
            PredictionResult
        """
        if not self.can_predict:
            raise ValueError("Model is not trained or ready for prediction")

        start_time = time.time()

        # Convert features to vector
        feature_vector = features.to_vector(self.feature_map)

        # Load model
        model = pickle.loads(self._model_data)

        # Make prediction
        prediction_value = self._predict_with_model(model, feature_vector)

        # Calculate confidence (simplified)
        confidence = min(1.0, max(0.0, prediction_value))

        # Update tracking
        self.total_predictions += 1
        self.last_prediction_at = time.time()
        self.metrics.prediction_time_ms = (time.time() - start_time) * 1000

        return PredictionResult(
            prediction_type=self.prediction_type,
            confidence=confidence,
            value=prediction_value,
            features_used=list(features.numerical_features.keys()),
            model_version=self.version,
        )

    def _predict_with_model(self, model: Any, feature_vector: List[float]) -> float:
        """Make prediction with specific model type."""
        if isinstance(model, dict):
            if model.get("weights") and model.get("intercept") is not None:
                # Linear regression prediction
                weights = model["weights"]
                intercept = model["intercept"]

                if len(weights) != len(feature_vector):
                    return 0.0

                prediction = intercept + sum(w * x for w, x in zip(weights, feature_vector))
                return prediction

        return 0.0

    def batch_predict(self, features_list: List[ModelFeatures]) -> List[PredictionResult]:
        """Make batch predictions."""
        return [self.predict(features) for features in features_list]

    def update_model(self, new_data: TrainingData) -> bool:
        """Update model with new data (online learning)."""
        if not self.is_trained:
            return self.train(new_data)

        # For now, retrain with new data
        # In a full implementation, this would do incremental learning
        self.status = TrainingStatus.UPDATING
        success = self.train(new_data)
        if success:
            self.version = f"{float(self.version) + 0.1:.1f}"
        return success

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            return {}

        model = pickle.loads(self._model_data)
        if isinstance(model, dict) and "weights" in model:
            weights = model["weights"]
            feature_names = list(self.feature_map.keys())

            importance = {}
            for i, name in enumerate(feature_names):
                if i < len(weights):
                    importance[name] = abs(weights[i])

            # Normalize
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}

            return importance

        return {}

    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        model_data = {
            "name": self.name,
            "model_type": self.model_type.value,
            "prediction_type": self.prediction_type.value,
            "version": self.version,
            "feature_map": self.feature_map,
            "model_data": base64.b64encode(self._model_data).decode() if self._model_data else None,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics.to_dict(),
            "training_history": self.training_history,
            "created_at": self.created_at,
            "last_trained_at": self.last_trained_at,
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load_model(cls, filepath: str) -> 'CachePredictionModel':
        """Load model from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        model = cls(
            name=data["name"],
            model_type=ModelType(data["model_type"]),
            prediction_type=PredictionType(data["prediction_type"]),
        )

        model.version = data.get("version", "1.0")
        model.feature_map = data.get("feature_map", {})
        model.hyperparameters = data.get("hyperparameters", {})
        model.training_history = data.get("training_history", [])
        model.created_at = data.get("created_at", time.time())
        model.last_trained_at = data.get("last_trained_at")

        if data.get("model_data"):
            model._model_data = base64.b64decode(data["model_data"])
            model.status = TrainingStatus.TRAINED

        if data.get("metrics"):
            model.metrics = ModelMetrics.from_dict(data["metrics"])

        return model

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "prediction_type": self.prediction_type.value,
            "description": self.description,
            "status": self.status.value,
            "version": self.version,
            "feature_map": self.feature_map,
            "hyperparameters": self.hyperparameters,
            "feature_selection": self.feature_selection,
            "max_training_samples": self.max_training_samples,
            "min_training_samples": self.min_training_samples,
            "retrain_threshold": self.retrain_threshold,
            "metrics": self.metrics.to_dict(),
            "training_history": self.training_history,
            "created_at": self.created_at,
            "last_trained_at": self.last_trained_at,
            "last_prediction_at": self.last_prediction_at,
            "total_predictions": self.total_predictions,
            "is_trained": self.is_trained,
            "can_predict": self.can_predict,
        }

    def __str__(self) -> str:
        """String representation of the model."""
        return f"CachePredictionModel({self.name}, {self.model_type.value}, {self.status.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"CachePredictionModel(name='{self.name}', type='{self.model_type.value}', "
                f"status='{self.status.value}', version='{self.version}')")

    def __eq__(self, other) -> bool:
        """Check equality based on name and version."""
        if not isinstance(other, CachePredictionModel):
            return False
        return self.name == other.name and self.version == other.version

    def __hash__(self) -> int:
        """Hash based on name and version."""
        return hash((self.name, self.version))