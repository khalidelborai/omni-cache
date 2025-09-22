#!/usr/bin/env python3
"""
Load generator for OmniCache Enterprise Demo.

Simulates realistic user behavior and cache access patterns
to demonstrate performance characteristics under load.
"""

import random
import time
from locust import HttpUser, task, between
from faker import Faker

fake = Faker()

class OmniCacheUser(HttpUser):
    """Simulates a user interacting with the OmniCache demo."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Initialize user session."""
        self.user_ids = [f"user_{i}" for i in range(1, 101)]
        self.product_ids = [f"prod_{i}" for i in range(1, 501)]
        self.categories = ["Electronics", "Books", "Clothing", "Home", "Sports"]

    @task(10)
    def get_user_profile(self):
        """Get user profile - most common operation."""
        user_id = random.choice(self.user_ids)
        with self.client.get(f"/api/users/{user_id}", name="/api/users/[id]") as response:
            if response.status_code == 200:
                pass  # Success

    @task(8)
    def get_product(self):
        """Get product details."""
        product_id = random.choice(self.product_ids)
        with self.client.get(f"/api/products/{product_id}", name="/api/products/[id]") as response:
            if response.status_code == 200:
                pass  # Success

    @task(5)
    def get_products_by_category(self):
        """Browse products by category."""
        category = random.choice(self.categories)
        limit = random.choice([10, 20, 50])
        with self.client.get(f"/api/products/category/{category}?limit={limit}",
                           name="/api/products/category/[category]") as response:
            if response.status_code == 200:
                pass  # Success

    @task(3)
    def get_random_users(self):
        """Get random users for dashboard."""
        count = random.choice([5, 10, 15])
        with self.client.get(f"/api/users/random?count={count}",
                           name="/api/users/random") as response:
            if response.status_code == 200:
                pass  # Success

    @task(2)
    def get_analytics_events(self):
        """Get recent analytics events."""
        hours = random.choice([6, 12, 24])
        limit = random.choice([20, 50, 100])
        with self.client.get(f"/api/analytics/events/recent?hours={hours}&limit={limit}",
                           name="/api/analytics/events/recent") as response:
            if response.status_code == 200:
                pass  # Success

    @task(2)
    def get_cache_stats(self):
        """Monitor cache performance."""
        with self.client.get("/api/cache/stats", name="/api/cache/stats") as response:
            if response.status_code == 200:
                pass  # Success

    @task(1)
    def test_arc_performance(self):
        """Test ARC strategy performance."""
        with self.client.post("/api/test/arc", name="/api/test/arc") as response:
            if response.status_code == 200:
                pass  # Success

    @task(1)
    def test_security(self):
        """Test security features."""
        with self.client.post("/api/test/security", name="/api/test/security") as response:
            if response.status_code == 200:
                pass  # Success

    @task(1)
    def test_ml_prefetch(self):
        """Test ML prefetching."""
        with self.client.post("/api/test/ml", name="/api/test/ml") as response:
            if response.status_code == 200:
                pass  # Success

    @task(1)
    def health_check(self):
        """Health check."""
        with self.client.get("/api/health", name="/api/health") as response:
            if response.status_code == 200:
                pass  # Success

class HighVolumeUser(HttpUser):
    """Simulates high-volume automated system interactions."""

    wait_time = between(0.1, 0.5)  # Very frequent requests

    def on_start(self):
        """Initialize automated system session."""
        self.user_ids = [f"user_{i}" for i in range(1, 21)]  # Focus on hot keys
        self.product_ids = [f"prod_{i}" for i in range(1, 51)]  # Hot products

    @task(20)
    def rapid_user_lookups(self):
        """Rapid user profile lookups - creates cache pressure."""
        user_id = random.choice(self.user_ids)
        with self.client.get(f"/api/users/{user_id}", name="/api/users/[id]-rapid") as response:
            pass

    @task(15)
    def rapid_product_lookups(self):
        """Rapid product lookups."""
        product_id = random.choice(self.product_ids)
        with self.client.get(f"/api/products/{product_id}", name="/api/products/[id]-rapid") as response:
            pass

class MLTrainingUser(HttpUser):
    """Simulates access patterns to train ML prefetching."""

    wait_time = between(0.5, 1.0)

    def on_start(self):
        """Initialize ML training session."""
        self.sequence_position = 0
        self.sequence_keys = [f"ml_user_{i}" for i in range(1, 31)]

    @task(1)
    def sequential_access_pattern(self):
        """Create predictable access patterns for ML training."""
        # Access keys in sequence to create pattern
        user_id = self.sequence_keys[self.sequence_position]
        self.sequence_position = (self.sequence_position + 1) % len(self.sequence_keys)

        with self.client.get(f"/api/users/{user_id}", name="/api/users/[id]-ml-pattern") as response:
            pass