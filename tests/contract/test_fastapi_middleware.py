"""
Contract tests for FastAPI middleware.

Tests the FastAPI caching middleware according to the integration specification.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omnicache.integrations.fastapi import CacheMiddleware


class TestFastAPIMiddleware:
    """Test FastAPI middleware functionality according to specification."""

    @pytest.fixture
    def app_with_middleware(self):
        """Create a test FastAPI application with cache middleware."""
        app = FastAPI()

        app.add_middleware(
            CacheMiddleware,
            cache_name="middleware_cache",
            default_ttl=300,
            exclude_paths=["/health", "/metrics"]
        )

        @app.get("/api/data/{item_id}")
        async def get_data(item_id: int):
            return {"item_id": item_id, "data": f"Data for item {item_id}"}

        @app.get("/health")
        async def health_check():
            return {"status": "healthy"}

        @app.post("/api/data")
        async def create_data(data: dict):
            return {"message": "created", "data": data}

        return app

    @pytest.fixture
    def client_with_middleware(self, app_with_middleware):
        """Create a test client with middleware."""
        return TestClient(app_with_middleware)

    def test_middleware_caches_get_requests(self, client_with_middleware):
        """Test that middleware automatically caches GET requests."""
        # First request should miss cache
        response1 = client_with_middleware.get("/api/data/123")
        assert response1.status_code == 200

        # Second request should hit cache and include cache headers
        response2 = client_with_middleware.get("/api/data/123")
        assert response2.status_code == 200
        assert response2.json() == response1.json()
        assert "Cache-Control" in response2.headers

    def test_middleware_excludes_configured_paths(self, client_with_middleware):
        """Test that excluded paths are not cached."""
        # Health endpoint should not be cached
        response1 = client_with_middleware.get("/health")
        assert response1.status_code == 200
        assert "Cache-Control" not in response1.headers

    def test_middleware_ignores_non_get_requests(self, client_with_middleware):
        """Test that non-GET requests are not cached."""
        # POST request should not be cached
        response = client_with_middleware.post("/api/data", json={"test": "data"})
        assert response.status_code == 200
        assert "Cache-Control" not in response.headers

    def test_middleware_generates_etags(self, client_with_middleware):
        """Test that middleware generates ETags for responses."""
        response = client_with_middleware.get("/api/data/456")
        assert response.status_code == 200
        assert "ETag" in response.headers
        assert len(response.headers["ETag"]) > 0

    def test_middleware_handles_conditional_requests(self, client_with_middleware):
        """Test that middleware handles If-None-Match headers."""
        # Get initial response with ETag
        response1 = client_with_middleware.get("/api/data/789")
        etag = response1.headers["ETag"]

        # Make conditional request
        response2 = client_with_middleware.get(
            "/api/data/789",
            headers={"If-None-Match": etag}
        )
        assert response2.status_code == 304

    def test_middleware_cache_control_headers(self, client_with_middleware):
        """Test that appropriate Cache-Control headers are added."""
        response = client_with_middleware.get("/api/data/101")
        assert response.status_code == 200

        cache_control = response.headers.get("Cache-Control")
        assert cache_control is not None
        assert "max-age=" in cache_control