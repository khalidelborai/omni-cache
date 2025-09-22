"""
Integration test for FastAPI caching scenario.

Tests the complete FastAPI integration workflow from the quickstart guide.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omnicache.integrations.fastapi import cache, cache_response, CacheMiddleware


class TestFastAPIIntegration:
    """Test FastAPI integration according to quickstart scenario 2."""

    @pytest.mark.asyncio
    async def test_complete_fastapi_integration_workflow(self):
        """Test the complete FastAPI integration from quickstart guide."""
        app = FastAPI()

        # Function-level caching
        @app.get("/users/{user_id}")
        @cache(cache_name="integration_user_cache", ttl=1800)
        async def get_user(user_id: int):
            return {"user_id": user_id, "name": f"User {user_id}", "cached": False}

        # Response caching with headers
        @app.get("/products/{product_id}")
        @cache_response(ttl=600, vary_on=["product_id"])
        async def get_product(product_id: int):
            return {"product_id": product_id, "name": f"Product {product_id}"}

        # Middleware for automatic caching
        app.add_middleware(
            CacheMiddleware,
            cache_name="http_cache",
            default_ttl=300,
            exclude_paths=["/health", "/admin"]
        )

        @app.get("/api/data/{data_id}")
        async def get_data(data_id: int):
            return {"data_id": data_id, "content": f"Data {data_id}"}

        client = TestClient(app)

        # Test function-level caching
        response1 = client.get("/users/123")
        assert response1.status_code == 200
        
        response2 = client.get("/users/123")
        assert response2.status_code == 200
        assert response1.json() == response2.json()

        # Test response caching with headers
        response = client.get("/products/456")
        assert response.status_code == 200
        assert "Cache-Control" in response.headers
        assert "ETag" in response.headers

        # Test middleware automatic caching
        response = client.get("/api/data/789")
        assert response.status_code == 200
        assert "Cache-Control" in response.headers