#!/usr/bin/env python3
"""
Cache performance monitor for OmniCache Enterprise Demo.

Collects metrics from the demo application and exposes them
for Prometheus scraping.
"""

import asyncio
import time
import json
import os
from typing import Dict, Any, List
import httpx
import psutil
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
cache_hit_ratio = Gauge('omnicache_hit_ratio', 'Cache hit ratio', ['cache_name'])
cache_operations_total = Counter('omnicache_operations_total', 'Total cache operations', ['cache_name', 'operation'])
cache_current_size = Gauge('omnicache_current_size', 'Current cache size', ['cache_name'])
cache_max_size = Gauge('omnicache_max_size', 'Maximum cache size', ['cache_name'])
cache_memory_usage = Gauge('omnicache_memory_usage_bytes', 'Cache memory usage in bytes', ['cache_name'])

# ARC-specific metrics
arc_t1_hits = Gauge('omnicache_arc_t1_hits', 'ARC T1 list hits', ['cache_name'])
arc_t2_hits = Gauge('omnicache_arc_t2_hits', 'ARC T2 list hits', ['cache_name'])
arc_adaptations = Gauge('omnicache_arc_adaptations', 'ARC adaptations count', ['cache_name'])
arc_target_t1_size = Gauge('omnicache_arc_target_t1_size', 'ARC target T1 size', ['cache_name'])

# ML metrics
ml_prediction_accuracy = Gauge('omnicache_ml_prediction_accuracy', 'ML prediction accuracy', ['cache_name'])
ml_predictions_made = Counter('omnicache_ml_predictions_made_total', 'Total ML predictions made', ['cache_name'])
ml_prefetch_hits = Counter('omnicache_ml_prefetch_hits_total', 'ML prefetch hits', ['cache_name'])

# Security metrics
security_events_total = Counter('omnicache_security_events_total', 'Security events', ['cache_name', 'event_type'])
encryption_operations = Counter('omnicache_encryption_operations_total', 'Encryption operations', ['cache_name', 'operation_type'])
security_violations = Counter('omnicache_security_violations_total', 'Security violations', ['cache_name', 'violation_type'])

# Performance metrics
operation_duration = Histogram('omnicache_operation_duration_seconds', 'Operation duration', ['cache_name', 'operation'])
error_rate = Counter('omnicache_errors_total', 'Total errors', ['cache_name', 'error_type'])

# System metrics
system_cpu_usage = Gauge('omnicache_system_cpu_usage_percent', 'System CPU usage')
system_memory_usage = Gauge('omnicache_system_memory_usage_percent', 'System memory usage')

class CacheMonitor:
    """Monitors cache performance and exports metrics."""

    def __init__(self):
        """Initialize the monitor."""
        self.demo_api_url = os.getenv('DEMO_API_URL', 'http://localhost:8000')
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        self.monitor_interval = int(os.getenv('MONITOR_INTERVAL', '30'))
        self.client = httpx.AsyncClient(timeout=10.0)

    async def start(self):
        """Start the monitoring service."""
        logger.info("Starting OmniCache Performance Monitor")

        # Start Prometheus metrics server
        start_http_server(8080)
        logger.info("Prometheus metrics server started on port 8080")

        # Start monitoring loop
        while True:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Short retry delay

    async def collect_metrics(self):
        """Collect metrics from the demo application."""
        try:
            # Collect cache statistics
            await self.collect_cache_stats()

            # Collect system metrics
            self.collect_system_metrics()

            # Test various cache operations
            await self.test_cache_operations()

            logger.info("Metrics collection completed successfully")

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")

    async def collect_cache_stats(self):
        """Collect cache statistics from the demo API."""
        try:
            response = await self.client.get(f"{self.demo_api_url}/api/cache/stats")
            if response.status_code == 200:
                stats = response.json()

                for stat in stats:
                    cache_name = stat['cache_name']

                    # Basic metrics
                    cache_hit_ratio.labels(cache_name=cache_name).set(stat['hit_ratio'])
                    cache_current_size.labels(cache_name=cache_name).set(stat['current_size'])
                    cache_max_size.labels(cache_name=cache_name).set(stat['max_size'])

                    # Increment operation counter (simulated)
                    cache_operations_total.labels(cache_name=cache_name, operation='get').inc(
                        stat['total_operations'] * 0.7  # Assume 70% gets
                    )
                    cache_operations_total.labels(cache_name=cache_name, operation='set').inc(
                        stat['total_operations'] * 0.3  # Assume 30% sets
                    )

                logger.info(f"Collected stats for {len(stats)} caches")

        except Exception as e:
            logger.error(f"Failed to collect cache stats: {e}")

    def collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            system_memory_usage.set(memory.percent)

            # Memory usage per cache (estimated)
            memory_per_cache = memory.used / 4  # Rough estimate for 4 caches
            for cache_name in ['user_profiles', 'product_catalog', 'analytics_events', 'secure_data']:
                cache_memory_usage.labels(cache_name=cache_name).set(memory_per_cache)

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def test_cache_operations(self):
        """Test various cache operations to generate realistic metrics."""
        try:
            # Test ARC performance
            arc_response = await self.client.post(f"{self.demo_api_url}/api/test/arc")
            if arc_response.status_code == 200:
                arc_data = arc_response.json()
                cache_name = "user_profiles"  # ARC cache

                # Simulate ARC metrics
                arc_t1_hits.labels(cache_name=cache_name).set(100)  # Simulated
                arc_t2_hits.labels(cache_name=cache_name).set(150)  # Simulated
                arc_adaptations.labels(cache_name=cache_name).set(25)  # Simulated
                arc_target_t1_size.labels(cache_name=cache_name).set(250)  # Simulated

            # Test security features
            security_response = await self.client.post(f"{self.demo_api_url}/api/test/security")
            if security_response.status_code == 200:
                cache_name = "secure_data"
                encryption_operations.labels(cache_name=cache_name, operation_type='encrypt').inc(10)
                encryption_operations.labels(cache_name=cache_name, operation_type='decrypt').inc(8)
                security_events_total.labels(cache_name=cache_name, event_type='access_granted').inc(5)

            # Test ML features
            ml_response = await self.client.post(f"{self.demo_api_url}/api/test/ml")
            if ml_response.status_code == 200:
                cache_name = "product_catalog"
                ml_prediction_accuracy.labels(cache_name=cache_name).set(0.85)
                ml_predictions_made.labels(cache_name=cache_name).inc(50)
                ml_prefetch_hits.labels(cache_name=cache_name).inc(35)

            # Simulate operation durations
            import random
            for cache_name in ['user_profiles', 'product_catalog', 'analytics_events', 'secure_data']:
                # Simulate different operation times
                get_time = random.uniform(0.001, 0.020)  # 1-20ms
                set_time = random.uniform(0.002, 0.030)  # 2-30ms

                operation_duration.labels(cache_name=cache_name, operation='get').observe(get_time)
                operation_duration.labels(cache_name=cache_name, operation='set').observe(set_time)

        except Exception as e:
            logger.error(f"Failed to test cache operations: {e}")

    async def generate_alerts(self):
        """Generate test alerts for demonstration."""
        try:
            # Simulate various alert conditions
            cache_names = ['user_profiles', 'product_catalog', 'analytics_events', 'secure_data']

            for cache_name in cache_names:
                # Randomly trigger low hit ratio for demonstration
                if time.time() % 120 < 10:  # 10 seconds every 2 minutes
                    cache_hit_ratio.labels(cache_name=cache_name).set(0.3)  # Low hit ratio
                    logger.warning(f"Simulated low hit ratio alert for {cache_name}")

                # Simulate occasional security events
                if time.time() % 300 < 5:  # 5 seconds every 5 minutes
                    security_violations.labels(cache_name=cache_name, violation_type='suspicious_access').inc()
                    logger.warning(f"Simulated security violation for {cache_name}")

        except Exception as e:
            logger.error(f"Failed to generate alerts: {e}")

    async def cleanup(self):
        """Cleanup resources."""
        await self.client.aclose()

async def main():
    """Main monitoring function."""
    monitor = CacheMonitor()

    try:
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Shutting down monitor...")
    finally:
        await monitor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())