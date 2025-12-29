"""
Unit tests for power caching features.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import time


class TestCompression:
    """Tests for compression utilities."""

    def test_compression_algorithms(self):
        """Test compression algorithm enum."""
        from omnicache.core.compression import CompressionAlgorithm

        assert CompressionAlgorithm.NONE.value == "none"
        assert CompressionAlgorithm.ZLIB.value == "zlib"
        assert CompressionAlgorithm.GZIP.value == "gzip"
        assert CompressionAlgorithm.LZMA.value == "lzma"

    def test_json_serializer(self):
        """Test JSON serialization."""
        from omnicache.core.compression import JSONSerializer

        serializer = JSONSerializer()
        data = {"key": "value", "number": 42}

        serialized = serializer.serialize(data)
        assert isinstance(serialized, bytes)

        deserialized = serializer.deserialize(serialized)
        assert deserialized == data

    def test_pickle_serializer(self):
        """Test pickle serialization."""
        from omnicache.core.compression import PickleSerializer

        serializer = PickleSerializer()

        class CustomObject:
            def __init__(self, value):
                self.value = value

        obj = CustomObject(42)
        serialized = serializer.serialize(obj)
        deserialized = serializer.deserialize(serialized)

        assert deserialized.value == 42

    def test_zlib_compressor(self):
        """Test zlib compression."""
        from omnicache.core.compression import ZlibCompressor

        compressor = ZlibCompressor(level=6)
        data = b"hello world " * 100

        compressed = compressor.compress(data)
        assert len(compressed) < len(data)

        decompressed = compressor.decompress(compressed)
        assert decompressed == data

    def test_gzip_compressor(self):
        """Test gzip compression."""
        from omnicache.core.compression import GzipCompressor

        compressor = GzipCompressor(level=6)
        data = b"test data " * 50

        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data

    def test_compression_middleware(self):
        """Test compression middleware encode/decode."""
        from omnicache.core.compression import (
            CompressionMiddleware,
            CompressionConfig,
            CompressionAlgorithm
        )

        config = CompressionConfig(
            algorithm=CompressionAlgorithm.ZLIB,
            min_size=10
        )
        middleware = CompressionMiddleware(config)

        # Test with large data (should compress)
        large_data = {"items": list(range(1000))}
        encoded = middleware.encode(large_data)
        decoded = middleware.decode(encoded)

        assert decoded == large_data

    def test_compression_middleware_small_data(self):
        """Test that small data is not compressed."""
        from omnicache.core.compression import (
            CompressionMiddleware,
            CompressionConfig,
            CompressionAlgorithm
        )

        config = CompressionConfig(
            algorithm=CompressionAlgorithm.ZLIB,
            min_size=1000  # High threshold
        )
        middleware = CompressionMiddleware(config)

        small_data = {"a": 1}
        encoded = middleware.encode(small_data)

        # Should not have compression header
        assert not encoded.startswith(CompressionMiddleware.MAGIC_HEADER)

    def test_compression_stats(self):
        """Test compression statistics."""
        from omnicache.core.compression import (
            CompressionMiddleware,
            CompressionConfig,
            CompressionAlgorithm
        )

        config = CompressionConfig(algorithm=CompressionAlgorithm.ZLIB)
        middleware = CompressionMiddleware(config)

        data = {"items": list(range(1000))}
        stats = middleware.get_compression_stats(data)

        assert "original_size" in stats
        assert "compressed_size" in stats
        assert "ratio" in stats
        assert "savings_percent" in stats


class TestRequestCoalescing:
    """Tests for request coalescing."""

    @pytest.mark.asyncio
    async def test_single_request(self):
        """Test single request goes through."""
        from omnicache.core.power import RequestCoalescer

        coalescer = RequestCoalescer()
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return "result"

        result = await coalescer.get_or_compute("key1", compute)

        assert result == "result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_requests_coalesced(self):
        """Test that concurrent requests are coalesced."""
        from omnicache.core.power import RequestCoalescer

        coalescer = RequestCoalescer()
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return "result"

        # Launch multiple concurrent requests
        results = await asyncio.gather(
            coalescer.get_or_compute("key1", compute),
            coalescer.get_or_compute("key1", compute),
            coalescer.get_or_compute("key1", compute),
        )

        # All should get same result
        assert all(r == "result" for r in results)
        # But compute should only be called once
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_keys_not_coalesced(self):
        """Test that different keys are not coalesced."""
        from omnicache.core.power import RequestCoalescer

        coalescer = RequestCoalescer()
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            return f"result-{call_count}"

        result1 = await coalescer.get_or_compute("key1", compute)
        result2 = await coalescer.get_or_compute("key2", compute)

        assert result1 != result2
        assert call_count == 2


class TestMultiLevelCache:
    """Tests for multi-level caching."""

    @pytest.mark.asyncio
    async def test_l1_hit(self):
        """Test L1 cache hit."""
        from omnicache.core.power import MultiLevelCache, CacheLevel

        l1 = MagicMock()
        l1.get = AsyncMock(return_value="cached_value")
        l2 = MagicMock()
        l2.get = AsyncMock(return_value=None)

        ml_cache = MultiLevelCache([
            CacheLevel(name="L1", cache=l1, ttl=60),
            CacheLevel(name="L2", cache=l2, ttl=300),
        ])

        result = await ml_cache.get("key")

        assert result == "cached_value"
        l1.get.assert_called_once_with("key")
        l2.get.assert_not_called()  # Should not check L2

    @pytest.mark.asyncio
    async def test_l2_hit_with_promotion(self):
        """Test L2 hit with promotion to L1."""
        from omnicache.core.power import MultiLevelCache, CacheLevel

        l1 = MagicMock()
        l1.get = AsyncMock(return_value=None)
        l1.set = AsyncMock()
        l2 = MagicMock()
        l2.get = AsyncMock(return_value="l2_value")

        ml_cache = MultiLevelCache([
            CacheLevel(name="L1", cache=l1, ttl=60),
            CacheLevel(name="L2", cache=l2, ttl=300),
        ])

        result = await ml_cache.get("key")

        assert result == "l2_value"
        # Should promote to L1
        l1.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_writes_to_all_levels(self):
        """Test that set writes to all levels."""
        from omnicache.core.power import MultiLevelCache, CacheLevel

        l1 = MagicMock()
        l1.set = AsyncMock()
        l2 = MagicMock()
        l2.set = AsyncMock()

        ml_cache = MultiLevelCache([
            CacheLevel(name="L1", cache=l1, ttl=60),
            CacheLevel(name="L2", cache=l2, ttl=300),
        ])

        await ml_cache.set("key", "value")

        l1.set.assert_called_once()
        l2.set.assert_called_once()

    def test_stats(self):
        """Test statistics tracking."""
        from omnicache.core.power import MultiLevelCache, CacheLevel

        l1 = MagicMock()
        l2 = MagicMock()

        ml_cache = MultiLevelCache([
            CacheLevel(name="L1", cache=l1),
            CacheLevel(name="L2", cache=l2),
        ])

        stats = ml_cache.get_stats()

        assert "levels" in stats
        assert "summary" in stats
        assert "L1" in stats["levels"]
        assert "L2" in stats["levels"]


class TestVersionedCache:
    """Tests for cache versioning."""

    @pytest.mark.asyncio
    async def test_version_match(self):
        """Test that matching versions return data."""
        from omnicache.core.power import VersionedCache

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value={
            "__versioned__": True,
            "value": "data",
            "version": "1.0"
        })

        vcache = VersionedCache(mock_cache, version="1.0")
        result = await vcache.get("key")

        assert result == "data"

    @pytest.mark.asyncio
    async def test_version_mismatch(self):
        """Test that version mismatch returns None."""
        from omnicache.core.power import VersionedCache

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value={
            "__versioned__": True,
            "value": "old_data",
            "version": "1.0"
        })

        vcache = VersionedCache(mock_cache, version="2.0")
        result = await vcache.get("key")

        assert result is None  # Version mismatch

    @pytest.mark.asyncio
    async def test_set_with_version(self):
        """Test that set includes version."""
        from omnicache.core.power import VersionedCache

        mock_cache = MagicMock()
        mock_cache.set = AsyncMock()

        vcache = VersionedCache(mock_cache, version="1.0")
        await vcache.set("key", "value")

        call_args = mock_cache.set.call_args
        stored_data = call_args[0][1]

        assert stored_data["__versioned__"] is True
        assert stored_data["version"] == "1.0"
        assert stored_data["value"] == "value"


class TestNegativeCache:
    """Tests for negative caching."""

    @pytest.mark.asyncio
    async def test_positive_cache_hit(self):
        """Test positive cache hit."""
        from omnicache.core.power import NegativeCache

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value="value")

        ncache = NegativeCache(mock_cache)
        value, is_negative = await ncache.get("key")

        assert value == "value"
        assert is_negative is False

    @pytest.mark.asyncio
    async def test_negative_cache_hit(self):
        """Test negative cache hit."""
        from omnicache.core.power import NegativeCache

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value={
            "__negative__": True,
            "reason": "not_found"
        })

        ncache = NegativeCache(mock_cache)
        value, is_negative = await ncache.get("key")

        assert value is None
        assert is_negative is True

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss."""
        from omnicache.core.power import NegativeCache

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)

        ncache = NegativeCache(mock_cache)
        value, is_negative = await ncache.get("key")

        assert value is None
        assert is_negative is False

    @pytest.mark.asyncio
    async def test_set_negative(self):
        """Test setting negative cache entry."""
        from omnicache.core.power import NegativeCache

        mock_cache = MagicMock()
        mock_cache.set = AsyncMock()

        ncache = NegativeCache(mock_cache, negative_ttl=60)
        await ncache.set_negative("key", reason="not_found")

        call_args = mock_cache.set.call_args
        stored_data = call_args[0][1]

        assert stored_data["__negative__"] is True
        assert stored_data["reason"] == "not_found"


class TestMemoization:
    """Tests for memoization."""

    @pytest.mark.asyncio
    async def test_lru_memoizer(self):
        """Test LRU memoizer."""
        from omnicache.core.power import LRUMemoizer

        memoizer = LRUMemoizer(maxsize=3)

        await memoizer.set("key1", "value1")
        await memoizer.set("key2", "value2")
        await memoizer.set("key3", "value3")

        assert await memoizer.get("key1") == "value1"
        assert await memoizer.get("key2") == "value2"
        assert await memoizer.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        from omnicache.core.power import LRUMemoizer

        memoizer = LRUMemoizer(maxsize=2)

        await memoizer.set("key1", "value1")
        await memoizer.set("key2", "value2")
        await memoizer.set("key3", "value3")  # Should evict key1

        assert await memoizer.get("key1") is None  # Evicted
        assert await memoizer.get("key2") == "value2"
        assert await memoizer.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_memoize_decorator(self):
        """Test memoize decorator."""
        from omnicache.core.power import memoize

        call_count = 0

        @memoize(maxsize=10)
        async def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = await expensive_func(5)
        result2 = await expensive_func(5)  # Should be cached

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Only called once

    def test_memoizer_stats(self):
        """Test memoizer statistics."""
        from omnicache.core.power import LRUMemoizer

        memoizer = LRUMemoizer(maxsize=10)
        stats = memoizer.stats()

        assert "size" in stats
        assert "maxsize" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


class TestPrefetcher:
    """Tests for prefetching."""

    def test_record_access(self):
        """Test access recording."""
        from omnicache.core.power import Prefetcher

        mock_cache = MagicMock()

        async def loader(key):
            return f"value:{key}"

        prefetcher = Prefetcher(mock_cache, loader)

        prefetcher.record_access("key1")
        prefetcher.record_access("key2")
        prefetcher.record_access("key1")

        assert prefetcher._access_counts["key1"] == 2
        assert prefetcher._access_counts["key2"] == 1

    def test_predict_next(self):
        """Test next key prediction."""
        from omnicache.core.power import Prefetcher

        mock_cache = MagicMock()

        async def loader(key):
            return f"value:{key}"

        prefetcher = Prefetcher(mock_cache, loader, confidence_threshold=0.3)

        # Build access pattern: key1 -> key2 frequently
        for _ in range(10):
            prefetcher.record_access("key1")
            prefetcher.record_access("key2")

        predictions = prefetcher.predict_next("key1")

        assert "key2" in predictions


class TestModuleExports:
    """Tests for module exports."""

    def test_power_exports(self):
        """Test that all power features are exported."""
        from omnicache.core.power import (
            RequestCoalescer,
            coalesce_requests,
            MultiLevelCache,
            multilevel_cache,
            VersionedCache,
            versioned_cache,
            NegativeCache,
            negative_cache,
            LRUMemoizer,
            memoize,
            WriteBehindCache,
            Prefetcher,
        )

        assert RequestCoalescer is not None
        assert coalesce_requests is not None
        assert MultiLevelCache is not None
        assert multilevel_cache is not None
        assert VersionedCache is not None
        assert versioned_cache is not None
        assert NegativeCache is not None
        assert negative_cache is not None
        assert LRUMemoizer is not None
        assert memoize is not None
        assert WriteBehindCache is not None
        assert Prefetcher is not None

    def test_integrations_power_exports(self):
        """Test power features exported from integrations."""
        from omnicache.integrations import (
            compressed_cache,
            coalesce_requests,
            multilevel_cache,
            versioned_cache,
            negative_cache,
            memoize,
            CompressedCache,
            MultiLevelCache,
            Prefetcher,
        )

        assert compressed_cache is not None
        assert coalesce_requests is not None
        assert multilevel_cache is not None
        assert versioned_cache is not None
        assert negative_cache is not None
        assert memoize is not None
