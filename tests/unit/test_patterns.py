"""
Unit tests for caching patterns.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class TestCacheEvents:
    """Tests for cache event system."""

    def test_event_types(self):
        """Test all event types exist."""
        from omnicache.core.patterns import CacheEventType

        assert CacheEventType.HIT.value == "hit"
        assert CacheEventType.MISS.value == "miss"
        assert CacheEventType.SET.value == "set"
        assert CacheEventType.DELETE.value == "delete"
        assert CacheEventType.EVICT.value == "evict"
        assert CacheEventType.REFRESH.value == "refresh"

    def test_cache_event_creation(self):
        """Test CacheEvent dataclass."""
        from omnicache.core.patterns import CacheEvent, CacheEventType

        event = CacheEvent(
            event_type=CacheEventType.HIT,
            cache_name="test",
            key="test_key",
            value="test_value",
            ttl=300
        )

        assert event.event_type == CacheEventType.HIT
        assert event.cache_name == "test"
        assert event.key == "test_key"
        assert event.value == "test_value"
        assert event.ttl == 300
        assert event.timestamp is not None

    def test_event_emitter_registration(self):
        """Test event listener registration."""
        from omnicache.core.patterns import CacheEventEmitter, CacheEventType

        emitter = CacheEventEmitter()
        callback = MagicMock()

        emitter.on(CacheEventType.HIT, callback)

        assert callback in emitter._listeners[CacheEventType.HIT]

    def test_event_emitter_on_hit(self):
        """Test on_hit convenience method."""
        from omnicache.core.patterns import CacheEventEmitter, CacheEventType

        emitter = CacheEventEmitter()
        callback = MagicMock()

        emitter.on_hit(callback)

        assert callback in emitter._listeners[CacheEventType.HIT]

    def test_event_emitter_on_any(self):
        """Test global listener registration."""
        from omnicache.core.patterns import CacheEventEmitter

        emitter = CacheEventEmitter()
        callback = MagicMock()

        emitter.on_any(callback)

        assert callback in emitter._global_listeners

    @pytest.mark.asyncio
    async def test_event_emitter_emit(self):
        """Test event emission."""
        from omnicache.core.patterns import CacheEventEmitter, CacheEvent, CacheEventType

        emitter = CacheEventEmitter()
        callback = MagicMock()

        emitter.on(CacheEventType.HIT, callback)

        event = CacheEvent(
            event_type=CacheEventType.HIT,
            cache_name="test",
            key="key"
        )

        await emitter.emit(event)

        callback.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_event_emitter_async_callback(self):
        """Test async callback support."""
        from omnicache.core.patterns import CacheEventEmitter, CacheEvent, CacheEventType

        emitter = CacheEventEmitter()
        callback = AsyncMock()

        emitter.on(CacheEventType.MISS, callback)

        event = CacheEvent(
            event_type=CacheEventType.MISS,
            cache_name="test",
            key="key"
        )

        await emitter.emit(event)

        callback.assert_called_once_with(event)


class TestTagManager:
    """Tests for tag-based invalidation."""

    @pytest.mark.asyncio
    async def test_add_tags(self):
        """Test adding tags to a key."""
        from omnicache.core.patterns import TagManager

        tm = TagManager()

        await tm.add_tags("cache", "key1", ["tag1", "tag2"])

        keys = await tm.get_keys_by_tag("tag1")
        assert "cache:key1" in keys

    @pytest.mark.asyncio
    async def test_get_tags_for_key(self):
        """Test getting tags for a key."""
        from omnicache.core.patterns import TagManager

        tm = TagManager()

        await tm.add_tags("cache", "key1", ["tag1", "tag2", "tag3"])

        tags = await tm.get_tags_for_key("cache", "key1")
        assert "tag1" in tags
        assert "tag2" in tags
        assert "tag3" in tags

    @pytest.mark.asyncio
    async def test_remove_key(self):
        """Test removing a key from tags."""
        from omnicache.core.patterns import TagManager

        tm = TagManager()

        await tm.add_tags("cache", "key1", ["tag1"])
        await tm.remove_key("cache", "key1")

        keys = await tm.get_keys_by_tag("tag1")
        assert "cache:key1" not in keys

    @pytest.mark.asyncio
    async def test_invalidate_tag(self):
        """Test tag invalidation."""
        from omnicache.core.patterns import TagManager

        tm = TagManager()

        await tm.add_tags("cache", "key1", ["products"])
        await tm.add_tags("cache", "key2", ["products"])
        await tm.add_tags("cache", "key3", ["users"])

        keys = await tm.invalidate_tag("products")

        assert len(keys) == 2
        assert "cache:key1" in keys
        assert "cache:key2" in keys

        # Tag should be cleaned up
        remaining = await tm.get_keys_by_tag("products")
        assert len(remaining) == 0

    def test_get_stats(self):
        """Test tag statistics."""
        from omnicache.core.patterns import TagManager

        tm = TagManager()
        stats = tm.get_stats()

        assert "total_tags" in stats
        assert "total_tagged_keys" in stats


class TestCacheLock:
    """Tests for cache stampede prevention."""

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        """Test basic lock acquire/release."""
        from omnicache.core.patterns import CacheLock

        lock = CacheLock()

        acquired = await lock.acquire("key1")
        assert acquired is True

        is_locked = await lock.is_locked("key1")
        assert is_locked is True

        lock.release("key1")

        is_locked = await lock.is_locked("key1")
        assert is_locked is False

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test lock acquisition timeout."""
        from omnicache.core.patterns import CacheLock

        lock = CacheLock()

        # First acquire succeeds
        await lock.acquire("key1")

        # Second acquire should timeout
        acquired = await lock.acquire("key1", timeout=0.1)
        assert acquired is False

        lock.release("key1")

    @pytest.mark.asyncio
    async def test_cleanup_stale_locks(self):
        """Test stale lock cleanup."""
        from omnicache.core.patterns import CacheLock

        lock = CacheLock()

        await lock.acquire("key1")
        lock._lock_times["key1"] = 0  # Make it stale

        cleaned = await lock.cleanup_stale_locks(max_age=1.0)

        assert cleaned == 1
        is_locked = await lock.is_locked("key1")
        assert is_locked is False


class TestStaleWhileRevalidate:
    """Tests for SWR pattern."""

    @pytest.mark.asyncio
    async def test_fresh_data(self):
        """Test returning fresh data."""
        from omnicache.core.patterns import StaleWhileRevalidate

        swr = StaleWhileRevalidate()
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        async def loader():
            return "fresh_value"

        result = await swr.get(mock_cache, "key", loader, ttl=60, stale_ttl=30)

        assert result == "fresh_value"
        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cached_data(self):
        """Test returning cached data."""
        from omnicache.core.patterns import StaleWhileRevalidate

        swr = StaleWhileRevalidate()
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value="cached_value")

        async def loader():
            return "fresh_value"

        result = await swr.get(mock_cache, "key", loader, ttl=60, stale_ttl=30)

        assert result == "cached_value"

    @pytest.mark.asyncio
    async def test_invalidate(self):
        """Test SWR invalidation."""
        from omnicache.core.patterns import StaleWhileRevalidate

        swr = StaleWhileRevalidate()

        # Add an entry
        swr._entries["key"] = MagicMock()

        await swr.invalidate("key")

        assert "key" not in swr._entries


class TestCachedWithTags:
    """Tests for cached_with_tags decorator."""

    @pytest.mark.asyncio
    async def test_decorator_caches_result(self):
        """Test that decorator caches function result."""
        from omnicache.core.patterns import cached_with_tags

        call_count = 0

        @cached_with_tags(cache_name="test", ttl=300, tags=["test_tag"])
        async def my_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        with patch('omnicache.core.patterns.manager') as mock_manager:
            mock_manager.get_cache = AsyncMock(return_value=mock_cache)

            result = await my_func(5)

            assert result == 10
            mock_cache.set.assert_called_once()


class TestCachedWithLock:
    """Tests for cached_with_lock decorator."""

    @pytest.mark.asyncio
    async def test_decorator_prevents_stampede(self):
        """Test that decorator prevents cache stampede."""
        from omnicache.core.patterns import cached_with_lock

        @cached_with_lock(cache_name="test", ttl=300)
        async def expensive_func():
            return "result"

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        with patch('omnicache.core.patterns.manager') as mock_manager:
            mock_manager.get_cache = AsyncMock(return_value=mock_cache)

            result = await expensive_func()

            assert result == "result"


class TestCachedSWR:
    """Tests for cached_swr decorator."""

    @pytest.mark.asyncio
    async def test_decorator_uses_swr(self):
        """Test that decorator uses SWR pattern."""
        from omnicache.core.patterns import cached_swr

        @cached_swr(cache_name="test", ttl=60, stale_ttl=30)
        async def api_call():
            return {"data": "value"}

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        with patch('omnicache.core.patterns.manager') as mock_manager:
            mock_manager.get_cache = AsyncMock(return_value=mock_cache)

            result = await api_call()

            assert result == {"data": "value"}


class TestCachedBatch:
    """Tests for cached_batch decorator."""

    @pytest.mark.asyncio
    async def test_batch_decorator(self):
        """Test batch caching decorator."""
        from omnicache.core.patterns import cached_batch

        @cached_batch(cache_name="test", ttl=300, key_prefix="item", id_param="ids")
        async def get_items(ids):
            return [{"id": i, "name": f"Item {i}"} for i in ids]

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        with patch('omnicache.core.patterns.manager') as mock_manager:
            mock_manager.get_cache = AsyncMock(return_value=mock_cache)

            result = await get_items([1, 2, 3])

            assert len(result) == 3


class TestModuleExports:
    """Tests for module exports."""

    def test_patterns_exports(self):
        """Test that all patterns are exported."""
        from omnicache.core.patterns import (
            CacheEventType,
            CacheEvent,
            CacheEventEmitter,
            cache_events,
            TagManager,
            tag_manager,
            CacheLock,
            cache_lock,
            StaleWhileRevalidate,
            swr_manager,
            cached_with_tags,
            cached_with_lock,
            cached_swr,
            cached_batch,
            invalidate_by_tag,
            invalidate_by_pattern,
        )

        assert CacheEventType is not None
        assert CacheEvent is not None
        assert CacheEventEmitter is not None
        assert cache_events is not None
        assert TagManager is not None
        assert tag_manager is not None
        assert CacheLock is not None
        assert cache_lock is not None
        assert StaleWhileRevalidate is not None
        assert swr_manager is not None
        assert cached_with_tags is not None
        assert cached_with_lock is not None
        assert cached_swr is not None
        assert cached_batch is not None
        assert invalidate_by_tag is not None
        assert invalidate_by_pattern is not None

    def test_integrations_exports(self):
        """Test that integrations exports work."""
        from omnicache.integrations import (
            # Events
            CacheEventType,
            CacheEvent,
            cache_events,
            # Tags
            tag_manager,
            invalidate_by_tag,
            invalidate_by_pattern,
            # Decorators
            cached_with_tags,
            cached_with_lock,
            cached_swr,
            cached_batch,
            # Locking
            cache_lock,
            swr_manager,
        )

        assert CacheEventType is not None
        assert cache_events is not None
        assert tag_manager is not None
        assert cached_with_tags is not None
        assert cached_with_lock is not None
        assert cached_swr is not None
        assert cached_batch is not None
