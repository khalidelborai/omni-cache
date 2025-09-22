"""
Contract tests for FastAPI decorators.

Tests the FastAPI caching decorators according to the integration specification.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omnicache.integrations.fastapi import cache, cache_response


class TestFastAPIDecorators:
    """Test FastAPI decorator functionality according to specification."""

    @pytest.fixture
    def app(self):
        """Create a test FastAPI application."""
        app = FastAPI()

        @app.get("/users/{user_id}")
        @cache(cache_name="user_cache", ttl=300)
        async def get_user(user_id: int):
            return {"user_id": user_id, "name": f"User {user_id}"}

        @app.get("/products/{product_id}")
        @cache_response(ttl=600, vary_on=["product_id"])
        async def get_product(product_id: int):
            return {"product_id": product_id, "name": f"Product {product_id}"}

        return app

    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return TestClient(app)

    def test_cache_decorator_basic_functionality(self, client):
        """Test basic cache decorator functionality."""
        # First request should execute function
        response1 = client.get("/users/123")
        assert response1.status_code == 200
        assert response1.json() == {"user_id": 123, "name": "User 123"}

        # Second request should be served from cache
        response2 = client.get("/users/123")
        assert response2.status_code == 200
        assert response2.json() == {"user_id": 123, "name": "User 123"}

    def test_cache_response_decorator_with_headers(self, client):
        """Test cache_response decorator adds appropriate headers."""
        response = client.get("/products/456")

        assert response.status_code == 200
        assert response.json() == {"product_id": 456, "name": "Product 456"}

        # Should include caching headers
        assert "Cache-Control" in response.headers
        assert "ETag" in response.headers

    def test_cache_decorator_with_different_parameters(self, client):
        """Test that different parameters create separate cache entries."""
        # Request for user 1
        response1 = client.get("/users/1")
        assert response1.status_code == 200

        # Request for user 2 (different cache entry)
        response2 = client.get("/users/2")
        assert response2.status_code == 200
        assert response2.json()["user_id"] == 2

    def test_conditional_requests_with_etag(self, client):
        """Test conditional requests using ETag."""
        # First request
        response1 = client.get("/products/789")
        assert response1.status_code == 200
        etag = response1.headers["ETag"]

        # Conditional request with If-None-Match
        response2 = client.get("/products/789", headers={"If-None-Match": etag})
        assert response2.status_code == 304

    def test_cache_statistics_tracking(self, client):
        """Test that cache operations are tracked in statistics."""
        from omnicache.core.registry import CacheRegistry

        # Make requests to generate cache activity
        client.get("/users/999")  # Miss
        client.get("/users/999")  # Hit

        # Check cache statistics
        cache = CacheRegistry.get_cache("user_cache")
        stats = cache.get_statistics()

        assert stats.hit_count >= 1
        assert stats.miss_count >= 1