"""
Contract tests for cache entry operations API.

Tests cache entry set, get, delete, and bulk operations according to the OpenAPI specification.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from omnicache import Cache
from omnicache.core.exceptions import CacheKeyError, CacheError


class TestCacheEntryOperationsAPI:
    """Test cache entry operations functionality according to API contract."""

    @pytest.fixture
    async def cache(self):
        """Create and initialize a test cache."""
        cache = Cache(name="entry_test_cache")
        await cache.initialize()
        return cache

    @pytest.mark.asyncio
    async def test_set_entry_basic(self, cache):
        """Test setting a basic cache entry."""
        await cache.set("test_key", "test_value")

        # Verify entry was stored
        value = await cache.get("test_key")
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_set_entry_with_ttl(self, cache):
        """Test setting an entry with TTL."""
        await cache.set("ttl_key", "ttl_value", ttl=1)

        # Should be available immediately
        value = await cache.get("ttl_key")
        assert value == "ttl_value"

        # Should expire after TTL
        await asyncio.sleep(1.1)
        value = await cache.get("ttl_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_set_entry_with_tags(self, cache):
        """Test setting an entry with tags."""
        await cache.set("tagged_key", "tagged_value", tags=["tag1", "tag2"])

        # Verify entry exists and has tags
        entry = await cache.get_entry("tagged_key")
        assert entry.value.data == "tagged_value"
        assert "tag1" in entry.key.tags
        assert "tag2" in entry.key.tags

    @pytest.mark.asyncio
    async def test_set_entry_with_priority(self, cache):
        """Test setting an entry with priority."""
        await cache.set("priority_key", "priority_value", priority=0.8)

        # Verify entry has correct priority
        entry = await cache.get_entry("priority_key")
        assert entry.priority == 0.8

    @pytest.mark.asyncio
    async def test_get_entry_not_found(self, cache):
        """Test getting a non-existent entry returns None."""
        value = await cache.get("nonexistent_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_get_entry_detailed_info(self, cache):
        """Test getting detailed entry information."""
        await cache.set("detailed_key", {"name": "test", "value": 42}, ttl=300)

        entry = await cache.get_entry("detailed_key")

        assert entry.key.value == "detailed_key"
        assert entry.value.data == {"name": "test", "value": 42}
        assert entry.ttl == 300
        assert entry.access_count >= 1
        assert entry.last_accessed is not None
        assert entry.created_at is not None

    @pytest.mark.asyncio
    async def test_delete_entry_success(self, cache):
        """Test deleting an existing entry."""
        await cache.set("delete_key", "delete_value")

        # Verify entry exists
        value = await cache.get("delete_key")
        assert value == "delete_value"

        # Delete the entry
        result = await cache.delete("delete_key")
        assert result is True

        # Verify entry is gone
        value = await cache.get("delete_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete_entry_not_found(self, cache):
        """Test deleting a non-existent entry."""
        result = await cache.delete("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_set_bulk_entries(self, cache):
        """Test setting multiple entries in bulk."""
        entries = {
            "bulk_key1": "bulk_value1",
            "bulk_key2": "bulk_value2",
            "bulk_key3": "bulk_value3"
        }

        result = await cache.set_bulk(entries)

        # Verify all entries were set
        assert result.success_count == 3
        assert result.failure_count == 0

        # Verify entries are retrievable
        for key, expected_value in entries.items():
            value = await cache.get(key)
            assert value == expected_value

    @pytest.mark.asyncio
    async def test_set_bulk_entries_with_default_ttl(self, cache):
        """Test setting bulk entries with default TTL."""
        entries = {
            "bulk_ttl_key1": "bulk_ttl_value1",
            "bulk_ttl_key2": "bulk_ttl_value2"
        }

        result = await cache.set_bulk(entries, default_ttl=300)

        # Verify entries have TTL set
        entry1 = await cache.get_entry("bulk_ttl_key1")
        entry2 = await cache.get_entry("bulk_ttl_key2")

        assert entry1.ttl == 300
        assert entry2.ttl == 300

    @pytest.mark.asyncio
    async def test_delete_bulk_entries_by_keys(self, cache):
        """Test deleting multiple entries by keys."""
        # Set up test data
        await cache.set("bulk_del_key1", "value1")
        await cache.set("bulk_del_key2", "value2")
        await cache.set("bulk_del_key3", "value3")

        keys_to_delete = ["bulk_del_key1", "bulk_del_key3", "nonexistent_key"]
        result = await cache.delete_bulk(keys=keys_to_delete)

        # Should delete 2 existing keys, 1 not found
        assert result.success_count == 2
        assert result.failure_count == 1

        # Verify correct entries were deleted
        assert await cache.get("bulk_del_key1") is None
        assert await cache.get("bulk_del_key2") == "value2"  # Should still exist
        assert await cache.get("bulk_del_key3") is None

    @pytest.mark.asyncio
    async def test_delete_bulk_entries_by_pattern(self, cache):
        """Test deleting entries by key pattern."""
        # Set up test data
        await cache.set("user:1:profile", "profile1")
        await cache.set("user:1:settings", "settings1")
        await cache.set("user:2:profile", "profile2")
        await cache.set("other:data", "other")

        # Delete all user:1 entries
        result = await cache.delete_bulk(pattern="user:1:*")

        # Should delete 2 entries
        assert result.success_count == 2

        # Verify correct entries were deleted
        assert await cache.get("user:1:profile") is None
        assert await cache.get("user:1:settings") is None
        assert await cache.get("user:2:profile") == "profile2"  # Should remain
        assert await cache.get("other:data") == "other"  # Should remain

    @pytest.mark.asyncio
    async def test_delete_bulk_entries_by_tags(self, cache):
        """Test deleting entries by tags."""
        # Set up test data with tags
        await cache.set("tagged_key1", "value1", tags=["session", "user:1"])
        await cache.set("tagged_key2", "value2", tags=["session", "user:2"])
        await cache.set("tagged_key3", "value3", tags=["permanent", "user:1"])
        await cache.set("untagged_key", "value4")

        # Delete all entries with "session" tag
        result = await cache.delete_bulk(tags=["session"])

        # Should delete 2 entries
        assert result.success_count == 2

        # Verify correct entries were deleted
        assert await cache.get("tagged_key1") is None
        assert await cache.get("tagged_key2") is None
        assert await cache.get("tagged_key3") == "value3"  # Should remain
        assert await cache.get("untagged_key") == "value4"  # Should remain

    @pytest.mark.asyncio
    async def test_clear_cache_all_entries(self, cache):
        """Test clearing all entries from cache."""
        # Set up test data
        await cache.set("clear_key1", "value1")
        await cache.set("clear_key2", "value2")
        await cache.set("clear_key3", "value3")

        # Clear all entries
        result = await cache.clear()

        assert result.cleared_count == 3

        # Verify all entries are gone
        assert await cache.get("clear_key1") is None
        assert await cache.get("clear_key2") is None
        assert await cache.get("clear_key3") is None

    @pytest.mark.asyncio
    async def test_clear_cache_by_pattern(self, cache):
        """Test clearing entries by pattern."""
        # Set up test data
        await cache.set("temp:session1", "session1")
        await cache.set("temp:session2", "session2")
        await cache.set("perm:data1", "data1")

        # Clear only temp entries
        result = await cache.clear(pattern="temp:*")

        assert result.cleared_count == 2

        # Verify correct entries were cleared
        assert await cache.get("temp:session1") is None
        assert await cache.get("temp:session2") is None
        assert await cache.get("perm:data1") == "data1"  # Should remain

    @pytest.mark.asyncio
    async def test_clear_cache_by_tags(self, cache):
        """Test clearing entries by tags."""
        # Set up test data with tags
        await cache.set("key1", "value1", tags=["expired", "session"])
        await cache.set("key2", "value2", tags=["expired", "temp"])
        await cache.set("key3", "value3", tags=["permanent"])

        # Clear entries with "expired" tag
        result = await cache.clear(tags=["expired"])

        assert result.cleared_count == 2

        # Verify correct entries were cleared
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") == "value3"  # Should remain

    @pytest.mark.asyncio
    async def test_entry_validation_empty_key(self, cache):
        """Test that empty keys are rejected."""
        with pytest.raises(CacheKeyError, match="Key cannot be empty"):
            await cache.set("", "value")

    @pytest.mark.asyncio
    async def test_entry_validation_invalid_ttl(self, cache):
        """Test that invalid TTL values are rejected."""
        with pytest.raises(ValueError, match="TTL must be positive"):
            await cache.set("key", "value", ttl=-1)

        with pytest.raises(ValueError, match="TTL must be positive"):
            await cache.set("key", "value", ttl=0)

    @pytest.mark.asyncio
    async def test_entry_validation_invalid_priority(self, cache):
        """Test that invalid priority values are rejected."""
        with pytest.raises(ValueError, match="Priority must be between 0.0 and 1.0"):
            await cache.set("key", "value", priority=-0.1)

        with pytest.raises(ValueError, match="Priority must be between 0.0 and 1.0"):
            await cache.set("key", "value", priority=1.1)