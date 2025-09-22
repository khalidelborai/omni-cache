"""
Integration test for ML-driven prefetching system.

This test validates the complete ML prefetching workflow including
pattern collection, model training, prediction generation, and cache preloading.
"""

import pytest
import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Tuple
from unittest.mock import AsyncMock, MagicMock

from omnicache.core.cache import Cache
from omnicache.strategies.lru import LRUStrategy
from omnicache.backends.memory import MemoryBackend
from omnicache.ml.prefetcher import MLPrefetcher
from omnicache.ml.pattern_collector import PatternCollector, AccessPattern
from omnicache.ml.models.lstm_predictor import LSTMPredictor
from omnicache.ml.models.transformer_predictor import TransformerPredictor
from omnicache.ml.feature_extractor import FeatureExtractor
from omnicache.ml.training_pipeline import TrainingPipeline


@pytest.mark.integration
class TestMLPrefetching:
    """Integration tests for ML-driven prefetching system."""

    @pytest.fixture
    async def cache(self):
        """Create cache instance for ML prefetching tests."""
        backend = MemoryBackend()
        strategy = LRUStrategy(capacity=1000)
        cache = Cache(backend=backend, strategy=strategy, name="ml_test_cache")
        return cache

    @pytest.fixture
    async def pattern_collector(self, cache):
        """Create pattern collector for access pattern analysis."""
        return PatternCollector(
            cache=cache,
            collection_window=60,  # 1 minute
            min_pattern_length=3,
            max_pattern_length=10
        )

    @pytest.fixture
    async def feature_extractor(self):
        """Create feature extractor for ML model input preparation."""
        return FeatureExtractor(
            time_window_sizes=[5, 15, 60],  # 5s, 15s, 1m windows
            key_embedding_dim=128,
            temporal_features=True,
            sequence_features=True
        )

    @pytest.fixture
    async def lstm_predictor(self, feature_extractor):
        """Create LSTM-based predictor model."""
        return LSTMPredictor(
            feature_extractor=feature_extractor,
            hidden_size=256,
            num_layers=2,
            sequence_length=20,
            prediction_horizon=5
        )

    @pytest.fixture
    async def transformer_predictor(self, feature_extractor):
        """Create Transformer-based predictor model."""
        return TransformerPredictor(
            feature_extractor=feature_extractor,
            d_model=512,
            num_heads=8,
            num_layers=6,
            sequence_length=50,
            prediction_horizon=10
        )

    @pytest.fixture
    async def training_pipeline(self, pattern_collector, lstm_predictor, transformer_predictor):
        """Create ML training pipeline."""
        return TrainingPipeline(
            pattern_collector=pattern_collector,
            models=[lstm_predictor, transformer_predictor],
            training_interval=300,  # 5 minutes
            min_training_samples=1000,
            validation_split=0.2
        )

    @pytest.fixture
    async def ml_prefetcher(self, cache, training_pipeline):
        """Create complete ML prefetcher system."""
        return MLPrefetcher(
            cache=cache,
            training_pipeline=training_pipeline,
            prefetch_batch_size=50,
            prefetch_threshold=0.7,
            max_prefetch_size_mb=100
        )

    async def test_pattern_collection_workflow(self, pattern_collector):
        """Test access pattern collection and analysis."""
        # Simulate realistic user access patterns
        user_sessions = {
            "user_123": [
                "profile", "dashboard", "notifications", "settings",
                "profile", "messages", "notifications"
            ],
            "user_456": [
                "login", "dashboard", "reports", "analytics",
                "dashboard", "reports", "logout"
            ],
            "user_789": [
                "api_key", "documentation", "examples", "tutorials",
                "documentation", "api_key", "examples"
            ]
        }

        # Generate access patterns with realistic timing
        for user_id, keys in user_sessions.items():
            for i, key in enumerate(keys):
                full_key = f"{user_id}:{key}"
                await pattern_collector.record_access(
                    key=full_key,
                    timestamp=time.time() + i * 2,  # 2-second intervals
                    metadata={"user_id": user_id, "session": "session_1"}
                )

        # Allow pattern collection to process
        await asyncio.sleep(1)

        # Analyze collected patterns
        patterns = await pattern_collector.extract_patterns()

        assert len(patterns) > 0
        assert any(p.confidence > 0.5 for p in patterns)

        # Verify pattern structure
        for pattern in patterns:
            assert isinstance(pattern, AccessPattern)
            assert pattern.sequence is not None
            assert pattern.frequency > 0
            assert 0 <= pattern.confidence <= 1

        # Test sequential patterns
        sequential_patterns = [p for p in patterns if p.pattern_type == "sequential"]
        assert len(sequential_patterns) > 0

        # Verify user-specific patterns
        user_patterns = await pattern_collector.get_user_patterns("user_123")
        assert len(user_patterns) > 0

    async def test_feature_extraction_pipeline(self, feature_extractor, pattern_collector):
        """Test feature extraction for ML model training."""
        # Generate training data
        access_history = []
        for i in range(100):
            access_data = {
                "key": f"key_{i % 20}",
                "timestamp": time.time() + i * 10,
                "user_id": f"user_{i % 5}",
                "access_type": "read" if i % 3 == 0 else "write",
                "size": np.random.randint(100, 10000)
            }
            access_history.append(access_data)

        # Extract features for different time windows
        features = await feature_extractor.extract_features(access_history)

        # Verify feature structure
        assert "temporal_features" in features
        assert "sequence_features" in features
        assert "key_embeddings" in features

        # Verify temporal features
        temporal = features["temporal_features"]
        assert temporal.shape[0] == len(access_history)
        assert temporal.shape[1] > 0  # Feature dimensions

        # Verify sequence features
        sequences = features["sequence_features"]
        assert len(sequences) > 0
        assert all(len(seq) >= 3 for seq in sequences)  # Min sequence length

        # Test real-time feature extraction
        current_context = access_history[-10:]  # Last 10 accesses
        real_time_features = await feature_extractor.extract_real_time_features(current_context)

        assert "context_vector" in real_time_features
        assert "predicted_next_keys" in real_time_features

    async def test_lstm_model_training_and_prediction(self, lstm_predictor, pattern_collector):
        """Test LSTM model training and prediction workflow."""
        # Generate synthetic training data
        training_data = await self._generate_training_data(pattern_collector, num_samples=1000)

        # Train LSTM model
        training_results = await lstm_predictor.train(
            training_data,
            epochs=5,
            batch_size=32,
            validation_split=0.2
        )

        # Verify training results
        assert "loss" in training_results
        assert "accuracy" in training_results
        assert training_results["loss"] < 1.0  # Should converge
        assert training_results["accuracy"] > 0.5  # Better than random

        # Test prediction
        test_sequence = training_data["sequences"][:5]  # Take first 5 sequences
        predictions = await lstm_predictor.predict(test_sequence)

        assert len(predictions) == len(test_sequence)
        for pred in predictions:
            assert "next_keys" in pred
            assert "probabilities" in pred
            assert "confidence" in pred
            assert 0 <= pred["confidence"] <= 1

        # Test real-time prediction
        current_sequence = ["user_1:profile", "user_1:dashboard", "user_1:settings"]
        real_time_pred = await lstm_predictor.predict_next(current_sequence)

        assert "predicted_keys" in real_time_pred
        assert "probabilities" in real_time_pred
        assert len(real_time_pred["predicted_keys"]) > 0

    async def test_transformer_model_advanced_patterns(self, transformer_predictor, pattern_collector):
        """Test Transformer model for complex pattern recognition."""
        # Generate complex access patterns with dependencies
        complex_patterns = await self._generate_complex_patterns(num_users=50, session_length=100)

        # Train transformer model
        training_results = await transformer_predictor.train(
            complex_patterns,
            epochs=10,
            batch_size=16,
            learning_rate=0.001
        )

        # Verify advanced capabilities
        assert training_results["attention_weights"] is not None
        assert training_results["perplexity"] < 50  # Good language modeling

        # Test attention mechanism
        test_sequence = complex_patterns["sequences"][0]
        attention_analysis = await transformer_predictor.analyze_attention(test_sequence)

        assert "attention_weights" in attention_analysis
        assert "key_importance" in attention_analysis
        assert "temporal_dependencies" in attention_analysis

        # Test long-range dependency prediction
        long_sequence = complex_patterns["long_sequences"][0]  # 50+ steps
        long_pred = await transformer_predictor.predict_long_range(long_sequence)

        assert len(long_pred["predictions"]) > 5  # Multiple future steps
        assert "uncertainty" in long_pred
        assert "dependency_graph" in long_pred

    async def test_ml_prefetcher_end_to_end(self, ml_prefetcher, cache):
        """Test complete ML prefetcher workflow."""
        # Phase 1: Cold start - collect initial patterns
        await ml_prefetcher.start_collection()

        # Simulate user activity
        user_activities = await self._simulate_realistic_user_activity(cache, duration=60)

        # Allow pattern collection
        await asyncio.sleep(2)

        # Phase 2: Initial training
        training_status = await ml_prefetcher.trigger_training()
        assert training_status["status"] == "completed"
        assert training_status["models_trained"] > 0

        # Phase 3: Start prefetching
        await ml_prefetcher.enable_prefetching()

        # Simulate more activity with prefetching enabled
        prefetch_activities = await self._simulate_prefetch_scenario(cache, ml_prefetcher)

        # Verify prefetching effectiveness
        prefetch_stats = await ml_prefetcher.get_prefetch_statistics()

        assert prefetch_stats["total_prefetches"] > 0
        assert prefetch_stats["hit_rate"] > 0.3  # At least 30% hit rate
        assert prefetch_stats["false_positive_rate"] < 0.4  # Less than 40% false positives

        # Verify cache performance improvement
        cache_stats = await cache.get_statistics()
        assert cache_stats.hit_ratio > 0.6  # Improved hit ratio with prefetching

    async def test_adaptive_learning_and_model_updates(self, ml_prefetcher):
        """Test adaptive learning and continuous model improvement."""
        # Initial training phase
        await ml_prefetcher.start_collection()
        await self._generate_phase_1_patterns(ml_prefetcher)
        await ml_prefetcher.trigger_training()

        initial_performance = await ml_prefetcher.get_model_performance()

        # Simulate pattern shift (new user behavior)
        await self._generate_pattern_shift(ml_prefetcher)

        # Trigger adaptive retraining
        adaptation_results = await ml_prefetcher.adapt_to_pattern_changes()

        assert adaptation_results["pattern_drift_detected"] is True
        assert adaptation_results["retraining_triggered"] is True

        # Verify improved performance after adaptation
        updated_performance = await ml_prefetcher.get_model_performance()

        # Model should adapt to new patterns
        assert updated_performance["accuracy"] >= initial_performance["accuracy"] * 0.9
        assert updated_performance["adaptability_score"] > 0.7

    async def test_multi_user_pattern_isolation(self, ml_prefetcher):
        """Test user-specific pattern learning and isolation."""
        # Generate distinct patterns for different user types
        user_types = {
            "developer": ["api_docs", "code_examples", "error_logs", "performance"],
            "analyst": ["reports", "dashboards", "data_export", "visualizations"],
            "admin": ["user_management", "system_config", "security_logs", "backups"]
        }

        # Train separate models for each user type
        for user_type, typical_keys in user_types.items():
            await self._generate_user_type_patterns(ml_prefetcher, user_type, typical_keys)

        # Verify user-specific model creation
        user_models = await ml_prefetcher.get_user_models()
        assert len(user_models) == len(user_types)

        # Test user-specific predictions
        for user_type in user_types:
            user_context = [f"{user_type}:session_start"]
            predictions = await ml_prefetcher.predict_for_user(user_type, user_context)

            # Predictions should match user type patterns
            predicted_keys = predictions["predicted_keys"]
            assert any(key in user_types[user_type] for key in predicted_keys)

    async def test_prefetch_resource_management(self, ml_prefetcher, cache):
        """Test prefetch resource management and optimization."""
        # Configure resource limits
        await ml_prefetcher.configure_resources(
            max_memory_mb=50,
            max_prefetch_batch=20,
            prefetch_priority_threshold=0.8
        )

        # Generate high-confidence predictions
        high_conf_predictions = await self._generate_high_confidence_scenarios(ml_prefetcher)

        # Verify resource constraints are respected
        prefetch_stats = await ml_prefetcher.get_resource_usage()

        assert prefetch_stats["memory_usage_mb"] <= 50
        assert prefetch_stats["active_prefetches"] <= 20

        # Test priority-based prefetching
        priority_results = await ml_prefetcher.get_prefetch_priorities()
        high_priority = [p for p in priority_results if p["priority"] > 0.8]

        assert len(high_priority) > 0
        assert all(p["scheduled"] for p in high_priority)  # High priority items scheduled

    async def test_prefetch_accuracy_monitoring(self, ml_prefetcher):
        """Test prefetch accuracy monitoring and feedback loop."""
        # Enable accuracy tracking
        await ml_prefetcher.enable_accuracy_tracking()

        # Generate prefetch scenarios with known outcomes
        test_scenarios = await self._generate_accuracy_test_scenarios(ml_prefetcher)

        # Allow prefetching and track results
        for scenario in test_scenarios:
            await ml_prefetcher.execute_prefetch_scenario(scenario)

        await asyncio.sleep(5)  # Allow tracking to process

        # Analyze accuracy metrics
        accuracy_report = await ml_prefetcher.get_accuracy_report()

        assert "overall_accuracy" in accuracy_report
        assert "precision" in accuracy_report
        assert "recall" in accuracy_report
        assert "f1_score" in accuracy_report

        # Verify feedback loop
        feedback_applied = await ml_prefetcher.apply_accuracy_feedback()
        assert feedback_applied["model_updates"] > 0
        assert feedback_applied["threshold_adjustments"] is True

    @pytest.mark.parametrize("model_type", ["lstm", "transformer", "ensemble"])
    async def test_model_comparison_and_selection(self, ml_prefetcher, model_type):
        """Test different ML models and automatic selection."""
        # Train multiple models
        model_results = await ml_prefetcher.train_all_models()

        assert len(model_results) >= 2  # At least LSTM and Transformer

        # Compare model performance
        comparison = await ml_prefetcher.compare_models()

        for model_name, metrics in comparison.items():
            assert "accuracy" in metrics
            assert "latency" in metrics
            assert "memory_usage" in metrics
            assert "prediction_quality" in metrics

        # Test model selection based on use case
        if model_type == "lstm":
            selected = await ml_prefetcher.select_model(criteria="low_latency")
            assert selected["model_type"] == "lstm"

        elif model_type == "transformer":
            selected = await ml_prefetcher.select_model(criteria="high_accuracy")
            assert selected["model_type"] == "transformer"

        elif model_type == "ensemble":
            selected = await ml_prefetcher.select_model(criteria="balanced")
            assert selected["model_type"] == "ensemble"

        # Verify model switching
        switch_result = await ml_prefetcher.switch_active_model(selected["model_id"])
        assert switch_result["success"] is True

    # Helper methods for test data generation

    async def _generate_training_data(self, pattern_collector, num_samples: int) -> Dict[str, Any]:
        """Generate synthetic training data for ML models."""
        sequences = []
        labels = []

        for i in range(num_samples):
            # Generate realistic access sequence
            sequence = [f"user_{i % 10}:key_{j}" for j in range(5, 15)]
            next_key = f"user_{i % 10}:key_{np.random.randint(0, 20)}"

            sequences.append(sequence)
            labels.append(next_key)

        return {
            "sequences": sequences,
            "labels": labels,
            "metadata": {"generation_time": time.time(), "num_samples": num_samples}
        }

    async def _generate_complex_patterns(self, num_users: int, session_length: int) -> Dict[str, Any]:
        """Generate complex access patterns with temporal dependencies."""
        patterns = {
            "sequences": [],
            "long_sequences": [],
            "temporal_features": [],
            "user_contexts": []
        }

        for user_id in range(num_users):
            # Generate user-specific patterns
            user_pattern = []
            for step in range(session_length):
                if step % 10 == 0:  # Periodic pattern
                    key = f"user_{user_id}:periodic_action"
                elif step % 7 == 0:  # Weekly pattern
                    key = f"user_{user_id}:weekly_report"
                else:
                    key = f"user_{user_id}:action_{np.random.randint(0, 20)}"

                user_pattern.append(key)

            patterns["sequences"].append(user_pattern[:20])  # Short sequences
            patterns["long_sequences"].append(user_pattern)  # Full sequences

        return patterns

    async def _simulate_realistic_user_activity(self, cache, duration: int) -> List[Dict]:
        """Simulate realistic user activity for testing."""
        activities = []
        start_time = time.time()

        while time.time() - start_time < duration:
            # Simulate different user behaviors
            user_id = f"user_{np.random.randint(0, 10)}"
            action_type = np.random.choice(["read", "write", "delete"], p=[0.7, 0.25, 0.05])

            if action_type == "read":
                key = f"{user_id}:data_{np.random.randint(0, 100)}"
                await cache.get(key)
            elif action_type == "write":
                key = f"{user_id}:data_{np.random.randint(0, 100)}"
                value = f"content_{time.time()}"
                await cache.set(key, value)

            activities.append({
                "user_id": user_id,
                "action": action_type,
                "timestamp": time.time()
            })

            await asyncio.sleep(0.1)  # 100ms between actions

        return activities

    async def _simulate_prefetch_scenario(self, cache, ml_prefetcher) -> Dict[str, Any]:
        """Simulate scenarios where prefetching should be effective."""
        # Create predictable access patterns
        predictable_sequences = [
            ["login", "dashboard", "notifications"],
            ["api_key", "documentation", "examples"],
            ["settings", "profile", "security"]
        ]

        results = {"predicted_correctly": 0, "total_predictions": 0}

        for sequence in predictable_sequences:
            # Access first part of sequence
            for key in sequence[:-1]:
                await cache.get(f"test:{key}")

            # Let ML predict next access
            predictions = await ml_prefetcher.get_current_predictions()
            results["total_predictions"] += 1

            # Check if last key was predicted
            if f"test:{sequence[-1]}" in predictions.get("predicted_keys", []):
                results["predicted_correctly"] += 1

            # Complete the sequence
            await cache.get(f"test:{sequence[-1]}")

        return results

    async def _generate_phase_1_patterns(self, ml_prefetcher):
        """Generate initial patterns for training."""
        # Morning work patterns
        morning_pattern = ["email", "calendar", "tasks", "reports"]
        for _ in range(50):
            for key in morning_pattern:
                await ml_prefetcher.record_access(f"morning:{key}")

    async def _generate_pattern_shift(self, ml_prefetcher):
        """Generate new patterns representing user behavior change."""
        # Afternoon patterns (different from morning)
        afternoon_pattern = ["analytics", "meetings", "documentation", "admin"]
        for _ in range(50):
            for key in afternoon_pattern:
                await ml_prefetcher.record_access(f"afternoon:{key}")

    async def _generate_user_type_patterns(self, ml_prefetcher, user_type: str, typical_keys: List[str]):
        """Generate patterns specific to a user type."""
        for _ in range(30):  # 30 sessions per user type
            for key in typical_keys:
                await ml_prefetcher.record_access(f"{user_type}:{key}")

    async def _generate_high_confidence_scenarios(self, ml_prefetcher) -> List[Dict]:
        """Generate scenarios that should produce high-confidence predictions."""
        return [
            {"pattern": ["login", "dashboard"], "expected_next": "notifications", "confidence": 0.9},
            {"pattern": ["search", "results"], "expected_next": "details", "confidence": 0.85},
            {"pattern": ["settings", "profile"], "expected_next": "security", "confidence": 0.8}
        ]

    async def _generate_accuracy_test_scenarios(self, ml_prefetcher) -> List[Dict]:
        """Generate test scenarios for accuracy measurement."""
        return [
            {
                "id": "scenario_1",
                "sequence": ["user_1:login", "user_1:dashboard"],
                "expected_next": "user_1:notifications",
                "should_prefetch": True
            },
            {
                "id": "scenario_2",
                "sequence": ["user_2:api", "user_2:docs"],
                "expected_next": "user_2:examples",
                "should_prefetch": True
            }
        ]