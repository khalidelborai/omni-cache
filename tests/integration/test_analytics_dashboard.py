"""
Integration test for analytics dashboard and monitoring.

This test validates Prometheus metrics, distributed tracing,
anomaly detection, and real-time analytics dashboards.
"""

import pytest
import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from omnicache.core.cache import Cache
from omnicache.strategies.lru import LRUStrategy
from omnicache.backends.memory import MemoryBackend
from omnicache.analytics.metrics_collector import MetricsCollector, MetricType
from omnicache.analytics.prometheus_exporter import PrometheusExporter, PrometheusConfig
from omnicache.analytics.tracer import DistributedTracer, TraceConfig, Span
from omnicache.analytics.anomaly_detector import AnomalyDetector, AnomalyConfig
from omnicache.analytics.dashboard import AnalyticsDashboard, DashboardConfig
from omnicache.analytics.alerting import AlertManager, AlertRule, AlertChannel


@pytest.mark.integration
class TestAnalyticsDashboard:
    """Integration tests for analytics dashboard and monitoring."""

    @pytest.fixture
    async def metrics_collector(self):
        """Create metrics collector for testing."""
        return MetricsCollector(
            collection_interval=1,  # 1 second for testing
            buffer_size=1000,
            metrics_retention_days=7
        )

    @pytest.fixture
    async def prometheus_config(self):
        """Create Prometheus configuration."""
        return PrometheusConfig(
            port=9090,
            endpoint="/metrics",
            namespace="omnicache",
            enable_histograms=True,
            enable_summaries=True,
            custom_labels={"environment": "test", "service": "cache"}
        )

    @pytest.fixture
    async def prometheus_exporter(self, prometheus_config, metrics_collector):
        """Create Prometheus metrics exporter."""
        exporter = PrometheusExporter(
            config=prometheus_config,
            metrics_collector=metrics_collector
        )
        await exporter.start()
        return exporter

    @pytest.fixture
    async def trace_config(self):
        """Create distributed tracing configuration."""
        return TraceConfig(
            service_name="omnicache-test",
            sampling_rate=1.0,  # 100% sampling for testing
            enable_jaeger=True,
            jaeger_endpoint="http://localhost:14268/api/traces",
            enable_zipkin=False
        )

    @pytest.fixture
    async def distributed_tracer(self, trace_config):
        """Create distributed tracer."""
        tracer = DistributedTracer(config=trace_config)
        await tracer.initialize()
        return tracer

    @pytest.fixture
    async def anomaly_config(self):
        """Create anomaly detection configuration."""
        return AnomalyConfig(
            detection_algorithms=["isolation_forest", "lstm_autoencoder", "statistical"],
            sensitivity=0.7,
            window_size=100,
            training_window=1000,
            alert_threshold=0.8
        )

    @pytest.fixture
    async def anomaly_detector(self, anomaly_config, metrics_collector):
        """Create anomaly detector."""
        detector = AnomalyDetector(
            config=anomaly_config,
            metrics_collector=metrics_collector
        )
        await detector.initialize()
        return detector

    @pytest.fixture
    async def dashboard_config(self):
        """Create analytics dashboard configuration."""
        return DashboardConfig(
            port=8080,
            update_interval=5,
            enable_real_time=True,
            enable_historical=True,
            max_data_points=10000,
            chart_types=["line", "bar", "heatmap", "gauge"]
        )

    @pytest.fixture
    async def analytics_dashboard(self, dashboard_config, metrics_collector, anomaly_detector):
        """Create analytics dashboard."""
        dashboard = AnalyticsDashboard(
            config=dashboard_config,
            metrics_collector=metrics_collector,
            anomaly_detector=anomaly_detector
        )
        await dashboard.start()
        return dashboard

    @pytest.fixture
    async def alert_manager(self):
        """Create alert manager for notifications."""
        return AlertManager(
            channels=[
                AlertChannel.EMAIL,
                AlertChannel.SLACK,
                AlertChannel.WEBHOOK
            ],
            escalation_policy={
                "warning": ["email"],
                "critical": ["email", "slack"],
                "emergency": ["email", "slack", "webhook"]
            }
        )

    @pytest.fixture
    async def monitored_cache(self, metrics_collector, distributed_tracer):
        """Create cache with full monitoring enabled."""
        backend = MemoryBackend()
        strategy = LRUStrategy(capacity=500)
        cache = Cache(
            backend=backend,
            strategy=strategy,
            name="monitored_cache",
            metrics_collector=metrics_collector,
            tracer=distributed_tracer
        )
        return cache

    async def test_metrics_collection_and_export(self, monitored_cache, prometheus_exporter):
        """Test comprehensive metrics collection and Prometheus export."""
        # Generate diverse cache operations
        operations = [
            ("set", "key_1", "value_1"),
            ("set", "key_2", "value_2"),
            ("get", "key_1", None),
            ("get", "key_3", None),  # Cache miss
            ("delete", "key_1", None),
            ("set", "key_4", "large_value_" * 1000),  # Large value
        ]

        # Execute operations and collect metrics
        for operation, key, value in operations:
            if operation == "set":
                await monitored_cache.set(key, value)
            elif operation == "get":
                await monitored_cache.get(key)
            elif operation == "delete":
                await monitored_cache.delete(key)

        # Allow metrics collection
        await asyncio.sleep(2)

        # Verify metrics are collected
        collected_metrics = await prometheus_exporter.get_collected_metrics()

        # Check core metrics
        assert "omnicache_operations_total" in collected_metrics
        assert "omnicache_hit_ratio" in collected_metrics
        assert "omnicache_response_time_seconds" in collected_metrics
        assert "omnicache_cache_size_bytes" in collected_metrics

        # Verify operation counts
        operations_metric = collected_metrics["omnicache_operations_total"]
        assert operations_metric["labels"]["operation"]["set"] >= 3
        assert operations_metric["labels"]["operation"]["get"] >= 2
        assert operations_metric["labels"]["operation"]["delete"] >= 1

        # Test metrics endpoint
        metrics_response = await prometheus_exporter.get_metrics_endpoint()
        assert "# HELP omnicache_operations_total" in metrics_response
        assert "# TYPE omnicache_operations_total counter" in metrics_response

        # Verify custom metrics
        custom_metrics = await prometheus_exporter.get_custom_metrics()
        assert len(custom_metrics) > 0

    async def test_distributed_tracing_workflow(self, monitored_cache, distributed_tracer):
        """Test distributed tracing across cache operations."""
        # Create parent trace for user session
        with distributed_tracer.start_span("user_session") as session_span:
            session_span.set_tag("user_id", "user_123")
            session_span.set_tag("session_id", "session_456")

            # Nested operations with tracing
            with distributed_tracer.start_span("cache_operations", child_of=session_span) as ops_span:
                # Set operation
                with distributed_tracer.start_span("cache_set", child_of=ops_span) as set_span:
                    set_span.set_tag("key", "trace_test_key")
                    set_span.set_tag("operation", "set")
                    await monitored_cache.set("trace_test_key", {"data": "traced_data"})
                    set_span.set_tag("success", True)

                # Get operation
                with distributed_tracer.start_span("cache_get", child_of=ops_span) as get_span:
                    get_span.set_tag("key", "trace_test_key")
                    get_span.set_tag("operation", "get")
                    result = await monitored_cache.get("trace_test_key")
                    get_span.set_tag("cache_hit", result is not None)
                    get_span.set_tag("success", True)

        # Verify trace collection
        traces = await distributed_tracer.get_traces(
            service_name="omnicache-test",
            operation_name="user_session"
        )

        assert len(traces) > 0

        # Verify trace structure
        session_trace = traces[0]
        assert session_trace.operation_name == "user_session"
        assert session_trace.tags["user_id"] == "user_123"
        assert len(session_trace.children) > 0

        # Verify span hierarchy
        cache_ops_span = session_trace.children[0]
        assert cache_ops_span.operation_name == "cache_operations"
        assert len(cache_ops_span.children) == 2  # set and get operations

        # Test trace analysis
        trace_analysis = await distributed_tracer.analyze_trace(session_trace.trace_id)
        assert trace_analysis["total_duration"] > 0
        assert trace_analysis["span_count"] >= 3
        assert trace_analysis["error_count"] == 0

    async def test_anomaly_detection_workflow(self, anomaly_detector, monitored_cache):
        """Test anomaly detection on cache metrics."""
        # Generate normal traffic pattern
        normal_pattern = await self._generate_normal_traffic(monitored_cache, duration=30)

        # Allow detector to learn normal patterns
        await anomaly_detector.train_baseline(normal_pattern)

        # Generate anomalous traffic
        anomalous_scenarios = [
            ("spike_in_requests", lambda: self._generate_request_spike(monitored_cache)),
            ("unusual_error_rate", lambda: self._generate_error_spike(monitored_cache)),
            ("memory_usage_anomaly", lambda: self._generate_memory_anomaly(monitored_cache)),
            ("response_time_spike", lambda: self._generate_latency_spike(monitored_cache))
        ]

        detected_anomalies = []

        for scenario_name, scenario_func in anomalous_scenarios:
            # Execute anomalous scenario
            await scenario_func()

            # Check for anomaly detection
            anomalies = await anomaly_detector.detect_anomalies(
                window_size=50,
                threshold=0.7
            )

            if anomalies:
                detected_anomalies.extend(anomalies)

        # Verify anomaly detection
        assert len(detected_anomalies) > 0

        for anomaly in detected_anomalies:
            assert anomaly.confidence > 0.7
            assert anomaly.anomaly_type in ["spike", "drift", "outlier"]
            assert anomaly.metrics_affected is not None

        # Test anomaly classification
        classification_results = await anomaly_detector.classify_anomalies(detected_anomalies)

        assert "performance_related" in classification_results
        assert "usage_related" in classification_results
        assert "error_related" in classification_results

    async def test_real_time_dashboard_updates(self, analytics_dashboard, monitored_cache):
        """Test real-time dashboard updates and data streaming."""
        # Start dashboard data streaming
        await analytics_dashboard.start_streaming()

        # Generate continuous activity
        activity_task = asyncio.create_task(
            self._generate_continuous_activity(monitored_cache, duration=20)
        )

        # Monitor dashboard updates
        dashboard_updates = []
        async def capture_updates():
            async for update in analytics_dashboard.stream_updates():
                dashboard_updates.append(update)
                if len(dashboard_updates) >= 10:  # Capture 10 updates
                    break

        update_task = asyncio.create_task(capture_updates())

        # Wait for both tasks
        await asyncio.gather(activity_task, update_task)

        # Verify dashboard updates
        assert len(dashboard_updates) >= 10

        for update in dashboard_updates:
            assert "timestamp" in update
            assert "metrics" in update
            assert "charts_data" in update

            # Verify real-time metrics
            metrics = update["metrics"]
            assert "current_operations_per_second" in metrics
            assert "current_hit_ratio" in metrics
            assert "current_response_time" in metrics

        # Test dashboard endpoints
        dashboard_data = await analytics_dashboard.get_dashboard_data()

        assert "overview" in dashboard_data
        assert "performance" in dashboard_data
        assert "usage_patterns" in dashboard_data
        assert "anomalies" in dashboard_data

    async def test_historical_analytics_and_reporting(self, analytics_dashboard, monitored_cache):
        """Test historical data analysis and report generation."""
        # Generate historical data over time periods
        time_periods = [
            ("1_hour_ago", timedelta(hours=1)),
            ("6_hours_ago", timedelta(hours=6)),
            ("1_day_ago", timedelta(days=1)),
            ("1_week_ago", timedelta(days=7))
        ]

        for period_name, time_delta in time_periods:
            # Simulate historical activity
            await self._simulate_historical_activity(
                monitored_cache,
                start_time=datetime.now() - time_delta,
                duration_minutes=60
            )

        # Generate comprehensive reports
        reports = {
            "hourly": await analytics_dashboard.generate_report(
                period="last_24_hours",
                granularity="hourly"
            ),
            "daily": await analytics_dashboard.generate_report(
                period="last_week",
                granularity="daily"
            ),
            "weekly": await analytics_dashboard.generate_report(
                period="last_month",
                granularity="weekly"
            )
        }

        # Verify report structure
        for report_type, report_data in reports.items():
            assert "summary" in report_data
            assert "trends" in report_data
            assert "top_keys" in report_data
            assert "performance_metrics" in report_data

            # Verify trend analysis
            trends = report_data["trends"]
            assert "hit_ratio_trend" in trends
            assert "operation_volume_trend" in trends
            assert "response_time_trend" in trends

        # Test custom report generation
        custom_report = await analytics_dashboard.generate_custom_report(
            metrics=["hit_ratio", "response_time", "memory_usage"],
            time_range={"start": datetime.now() - timedelta(hours=24), "end": datetime.now()},
            filters={"operation_type": ["get", "set"]},
            aggregations=["avg", "max", "95th_percentile"]
        )

        assert len(custom_report["data_points"]) > 0
        assert "aggregated_metrics" in custom_report

    async def test_alerting_and_notification_system(self, alert_manager, anomaly_detector, monitored_cache):
        """Test comprehensive alerting and notification system."""
        # Configure alert rules
        alert_rules = [
            AlertRule(
                name="high_error_rate",
                condition="error_rate > 0.05",
                severity="warning",
                description="Error rate exceeded 5%"
            ),
            AlertRule(
                name="low_hit_ratio",
                condition="hit_ratio < 0.7",
                severity="warning",
                description="Cache hit ratio below 70%"
            ),
            AlertRule(
                name="high_response_time",
                condition="response_time_p95 > 100",
                severity="critical",
                description="95th percentile response time over 100ms"
            ),
            AlertRule(
                name="memory_usage_critical",
                condition="memory_usage_percent > 90",
                severity="emergency",
                description="Memory usage critical"
            )
        ]

        for rule in alert_rules:
            await alert_manager.add_rule(rule)

        # Generate conditions that trigger alerts
        alert_scenarios = [
            ("high_error_rate", lambda: self._generate_high_error_rate(monitored_cache)),
            ("low_hit_ratio", lambda: self._generate_low_hit_ratio(monitored_cache)),
            ("high_response_time", lambda: self._generate_high_latency(monitored_cache))
        ]

        triggered_alerts = []

        for scenario_name, scenario_func in alert_scenarios:
            await scenario_func()

            # Check for triggered alerts
            alerts = await alert_manager.check_alert_conditions()
            triggered_alerts.extend(alerts)

        # Verify alerts were triggered
        assert len(triggered_alerts) > 0

        for alert in triggered_alerts:
            assert alert.rule_name in [rule.name for rule in alert_rules]
            assert alert.severity in ["warning", "critical", "emergency"]
            assert alert.timestamp is not None

        # Test alert notification delivery
        notification_results = await alert_manager.send_notifications(triggered_alerts)

        for result in notification_results:
            assert result["alert_id"] is not None
            assert result["channel"] in ["email", "slack", "webhook"]
            assert result["status"] in ["sent", "failed", "pending"]

        # Test alert escalation
        critical_alerts = [a for a in triggered_alerts if a.severity == "critical"]
        if critical_alerts:
            escalation_result = await alert_manager.escalate_alerts(critical_alerts)
            assert escalation_result["escalated_count"] > 0

    async def test_performance_benchmarking_and_optimization(self, analytics_dashboard, monitored_cache):
        """Test performance benchmarking and optimization recommendations."""
        # Run comprehensive performance benchmarks
        benchmark_scenarios = [
            {
                "name": "read_heavy_workload",
                "read_ratio": 0.9,
                "operations": 1000,
                "concurrency": 10
            },
            {
                "name": "write_heavy_workload",
                "read_ratio": 0.1,
                "operations": 1000,
                "concurrency": 10
            },
            {
                "name": "mixed_workload",
                "read_ratio": 0.7,
                "operations": 2000,
                "concurrency": 20
            }
        ]

        benchmark_results = {}

        for scenario in benchmark_scenarios:
            # Execute benchmark
            result = await self._execute_benchmark(monitored_cache, scenario)
            benchmark_results[scenario["name"]] = result

        # Analyze benchmark results
        performance_analysis = await analytics_dashboard.analyze_performance(benchmark_results)

        assert "bottlenecks_identified" in performance_analysis
        assert "optimization_recommendations" in performance_analysis
        assert "performance_score" in performance_analysis

        # Verify optimization recommendations
        recommendations = performance_analysis["optimization_recommendations"]
        assert len(recommendations) > 0

        for recommendation in recommendations:
            assert "category" in recommendation  # e.g., "memory", "strategy", "backend"
            assert "impact" in recommendation    # e.g., "high", "medium", "low"
            assert "description" in recommendation
            assert "implementation" in recommendation

    async def test_multi_cache_monitoring_and_comparison(self, analytics_dashboard):
        """Test monitoring and comparison of multiple cache instances."""
        # Create multiple cache instances with different configurations
        cache_configs = [
            {"name": "cache_lru", "strategy": "lru", "capacity": 100},
            {"name": "cache_lfu", "strategy": "lfu", "capacity": 100},
            {"name": "cache_arc", "strategy": "arc", "capacity": 100}
        ]

        caches = {}
        for config in cache_configs:
            backend = MemoryBackend()
            # Strategy creation would depend on actual implementation
            cache = Cache(
                backend=backend,
                strategy=LRUStrategy(capacity=config["capacity"]),  # Simplified
                name=config["name"]
            )
            caches[config["name"]] = cache

        # Generate identical workload for each cache
        test_workload = [
            ("set", f"key_{i}", f"value_{i}") for i in range(50)
        ] + [
            ("get", f"key_{i % 30}") for i in range(100)  # Some cache misses
        ]

        # Execute workload on all caches
        for cache_name, cache in caches.items():
            for operation, key, *args in test_workload:
                if operation == "set":
                    await cache.set(key, args[0])
                elif operation == "get":
                    await cache.get(key)

        # Compare cache performance
        comparison_report = await analytics_dashboard.compare_caches(list(caches.keys()))

        assert "performance_comparison" in comparison_report
        assert "hit_ratio_comparison" in comparison_report
        assert "response_time_comparison" in comparison_report

        # Verify comparison includes all caches
        performance_data = comparison_report["performance_comparison"]
        for cache_name in caches.keys():
            assert cache_name in performance_data

        # Test ranking and recommendations
        rankings = comparison_report["rankings"]
        assert "best_hit_ratio" in rankings
        assert "best_response_time" in rankings
        assert "most_efficient" in rankings

    # Helper methods for test scenarios

    async def _generate_normal_traffic(self, cache, duration: int):
        """Generate normal traffic pattern for baseline learning."""
        start_time = time.time()
        operations = []

        while time.time() - start_time < duration:
            # Normal operations distribution
            if np.random.random() < 0.7:  # 70% reads
                key = f"normal_key_{np.random.randint(0, 100)}"
                await cache.get(key)
                operations.append(("get", key, time.time()))
            else:  # 30% writes
                key = f"normal_key_{np.random.randint(0, 100)}"
                value = f"normal_value_{time.time()}"
                await cache.set(key, value)
                operations.append(("set", key, time.time()))

            await asyncio.sleep(0.1)  # Normal operation rate

        return operations

    async def _generate_request_spike(self, cache):
        """Generate sudden spike in requests."""
        # Rapid burst of operations
        tasks = []
        for i in range(200):  # High volume
            tasks.append(cache.get(f"spike_key_{i % 50}"))

        await asyncio.gather(*tasks)

    async def _generate_error_spike(self, cache):
        """Generate spike in error conditions."""
        # Attempt operations that will fail
        for i in range(50):
            try:
                await cache.get(f"nonexistent_key_{i}")
            except:
                pass

    async def _generate_memory_anomaly(self, cache):
        """Generate memory usage anomaly."""
        # Store large values
        large_value = "x" * 100000  # 100KB value
        for i in range(20):
            await cache.set(f"large_key_{i}", large_value)

    async def _generate_latency_spike(self, cache):
        """Generate response time spike."""
        # Simulate slow operations
        for i in range(10):
            await asyncio.sleep(0.2)  # Artificial delay
            await cache.get(f"slow_key_{i}")

    async def _generate_continuous_activity(self, cache, duration: int):
        """Generate continuous cache activity."""
        start_time = time.time()

        while time.time() - start_time < duration:
            # Mixed operations
            operation = np.random.choice(["get", "set", "delete"], p=[0.6, 0.3, 0.1])

            if operation == "get":
                await cache.get(f"activity_key_{np.random.randint(0, 50)}")
            elif operation == "set":
                await cache.set(
                    f"activity_key_{np.random.randint(0, 50)}",
                    f"value_{time.time()}"
                )
            elif operation == "delete":
                await cache.delete(f"activity_key_{np.random.randint(0, 50)}")

            await asyncio.sleep(0.05)  # 20 ops/second

    async def _simulate_historical_activity(self, cache, start_time: datetime, duration_minutes: int):
        """Simulate historical activity for reports."""
        # This would typically involve injecting backdated metrics
        # For testing, we'll generate current activity as a proxy
        operations_count = duration_minutes * 10  # 10 ops per minute

        for i in range(operations_count):
            await cache.set(f"hist_key_{i}", f"hist_value_{i}")
            if i % 3 == 0:
                await cache.get(f"hist_key_{i-1}")

    async def _generate_high_error_rate(self, cache):
        """Generate scenario with high error rate."""
        # Access non-existent keys to create cache misses/errors
        for i in range(20):
            await cache.get(f"error_key_{i}")  # These don't exist

    async def _generate_low_hit_ratio(self, cache):
        """Generate scenario with low hit ratio."""
        # Many cache misses
        for i in range(100):
            await cache.get(f"miss_key_{i}")  # Unique keys, guaranteed misses

    async def _generate_high_latency(self, cache):
        """Generate scenario with high latency."""
        # Operations with artificial delays
        for i in range(10):
            start = time.time()
            await cache.get(f"latency_key_{i}")
            # Simulate processing delay
            await asyncio.sleep(0.15)

    async def _execute_benchmark(self, cache, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance benchmark scenario."""
        name = scenario["name"]
        read_ratio = scenario["read_ratio"]
        operations = scenario["operations"]
        concurrency = scenario["concurrency"]

        start_time = time.time()

        async def worker(worker_id: int):
            """Benchmark worker."""
            ops_per_worker = operations // concurrency
            worker_start = time.time()

            for i in range(ops_per_worker):
                if np.random.random() < read_ratio:
                    await cache.get(f"bench_key_{i % 100}")
                else:
                    await cache.set(f"bench_key_{i % 100}", f"bench_value_{i}")

            return time.time() - worker_start

        # Run concurrent workers
        worker_times = await asyncio.gather(*[worker(i) for i in range(concurrency)])

        total_time = time.time() - start_time

        return {
            "scenario": name,
            "total_time": total_time,
            "operations_per_second": operations / total_time,
            "avg_worker_time": np.mean(worker_times),
            "max_worker_time": max(worker_times),
            "concurrency_achieved": concurrency
        }