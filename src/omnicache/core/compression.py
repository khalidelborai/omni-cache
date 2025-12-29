"""
Cache compression utilities.

Provides transparent compression/decompression for cached values
to reduce memory usage and network transfer.
"""

import zlib
import gzip
import lzma
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union
import json
import pickle
import logging

logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    LZMA = "lzma"


class Serializer(ABC):
    """Abstract base class for serializers."""

    @abstractmethod
    def serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        pass


class JSONSerializer(Serializer):
    """JSON serializer for cache values."""

    def serialize(self, value: Any) -> bytes:
        return json.dumps(value, default=str).encode('utf-8')

    def deserialize(self, data: bytes) -> Any:
        return json.loads(data.decode('utf-8'))


class PickleSerializer(Serializer):
    """Pickle serializer for complex Python objects."""

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol

    def serialize(self, value: Any) -> bytes:
        return pickle.dumps(value, protocol=self.protocol)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


class Compressor(ABC):
    """Abstract base class for compressors."""

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        pass


class NoCompressor(Compressor):
    """No-op compressor."""

    def compress(self, data: bytes) -> bytes:
        return data

    def decompress(self, data: bytes) -> bytes:
        return data


class ZlibCompressor(Compressor):
    """Zlib compression (fast, moderate ratio)."""

    def __init__(self, level: int = 6):
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data, level=self.level)

    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)


class GzipCompressor(Compressor):
    """Gzip compression (compatible, good ratio)."""

    def __init__(self, level: int = 6):
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=self.level)

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)


class LzmaCompressor(Compressor):
    """LZMA compression (slow, best ratio)."""

    def __init__(self, preset: int = 6):
        self.preset = preset

    def compress(self, data: bytes) -> bytes:
        return lzma.compress(data, preset=self.preset)

    def decompress(self, data: bytes) -> bytes:
        return lzma.decompress(data)


@dataclass
class CompressionConfig:
    """Configuration for cache compression."""
    algorithm: CompressionAlgorithm = CompressionAlgorithm.ZLIB
    level: int = 6
    min_size: int = 1024  # Only compress if larger than this (bytes)
    serializer: str = "json"  # "json" or "pickle"


class CompressionMiddleware:
    """
    Middleware for transparent compression of cached values.

    Automatically compresses values before storing and
    decompresses when retrieving.
    """

    # Magic bytes to identify compressed data
    MAGIC_HEADER = b'\x00OC\x01'  # OmniCache compression marker

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self._compressor = self._get_compressor()
        self._serializer = self._get_serializer()

    def _get_compressor(self) -> Compressor:
        """Get compressor based on config."""
        algo = self.config.algorithm

        if algo == CompressionAlgorithm.NONE:
            return NoCompressor()
        elif algo == CompressionAlgorithm.ZLIB:
            return ZlibCompressor(self.config.level)
        elif algo == CompressionAlgorithm.GZIP:
            return GzipCompressor(self.config.level)
        elif algo == CompressionAlgorithm.LZMA:
            return LzmaCompressor(self.config.level)
        else:
            return NoCompressor()

    def _get_serializer(self) -> Serializer:
        """Get serializer based on config."""
        if self.config.serializer == "pickle":
            return PickleSerializer()
        return JSONSerializer()

    def encode(self, value: Any) -> bytes:
        """
        Serialize and optionally compress a value.

        Returns bytes with a header indicating compression status.
        """
        # Serialize
        serialized = self._serializer.serialize(value)

        # Check if we should compress
        if len(serialized) < self.config.min_size:
            # Too small, don't compress
            return b'\x00' + serialized

        if self.config.algorithm == CompressionAlgorithm.NONE:
            return b'\x00' + serialized

        # Compress
        compressed = self._compressor.compress(serialized)

        # Only use compressed if it's smaller
        if len(compressed) < len(serialized):
            # Add header: magic + algorithm byte
            algo_byte = list(CompressionAlgorithm).index(self.config.algorithm).to_bytes(1, 'big')
            return self.MAGIC_HEADER + algo_byte + compressed
        else:
            return b'\x00' + serialized

    def decode(self, data: bytes) -> Any:
        """
        Decompress (if needed) and deserialize a value.
        """
        if not data:
            return None

        # Check for compression header
        if data.startswith(self.MAGIC_HEADER):
            # Extract algorithm and decompress
            algo_index = data[len(self.MAGIC_HEADER)]
            algo = list(CompressionAlgorithm)[algo_index]
            compressed_data = data[len(self.MAGIC_HEADER) + 1:]

            # Get appropriate decompressor
            if algo == CompressionAlgorithm.ZLIB:
                decompressor = ZlibCompressor()
            elif algo == CompressionAlgorithm.GZIP:
                decompressor = GzipCompressor()
            elif algo == CompressionAlgorithm.LZMA:
                decompressor = LzmaCompressor()
            else:
                decompressor = NoCompressor()

            serialized = decompressor.decompress(compressed_data)
        elif data[0:1] == b'\x00':
            # Uncompressed data
            serialized = data[1:]
        else:
            # Legacy data without header - assume uncompressed JSON
            serialized = data

        return self._serializer.deserialize(serialized)

    def get_compression_stats(self, original: Any) -> dict:
        """Get compression statistics for a value."""
        serialized = self._serializer.serialize(original)
        encoded = self.encode(original)

        return {
            "original_size": len(serialized),
            "compressed_size": len(encoded),
            "ratio": len(encoded) / len(serialized) if serialized else 0,
            "savings_percent": (1 - len(encoded) / len(serialized)) * 100 if serialized else 0,
            "is_compressed": encoded.startswith(self.MAGIC_HEADER)
        }


class CompressedCache:
    """
    Cache wrapper that adds transparent compression.

    Wraps any cache instance and handles compression automatically.
    """

    def __init__(
        self,
        cache: Any,
        config: Optional[CompressionConfig] = None
    ):
        self._cache = cache
        self._middleware = CompressionMiddleware(config)
        self._stats = {
            "total_sets": 0,
            "compressed_sets": 0,
            "bytes_saved": 0
        }

    async def get(self, key: str) -> Any:
        """Get and decompress a value."""
        data = await self._cache.get(key)
        if data is None:
            return None

        if isinstance(data, bytes):
            return self._middleware.decode(data)
        return data

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        **kwargs
    ) -> None:
        """Compress and set a value."""
        encoded = self._middleware.encode(value)

        # Track stats
        self._stats["total_sets"] += 1
        if encoded.startswith(CompressionMiddleware.MAGIC_HEADER):
            self._stats["compressed_sets"] += 1

        if ttl:
            await self._cache.set(key, encoded, ttl=ttl, **kwargs)
        else:
            await self._cache.set(key, encoded, **kwargs)

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        return await self._cache.delete(key)

    def get_stats(self) -> dict:
        """Get compression statistics."""
        return {
            **self._stats,
            "compression_rate": (
                self._stats["compressed_sets"] / self._stats["total_sets"]
                if self._stats["total_sets"] > 0 else 0
            )
        }

    def __getattr__(self, name: str) -> Any:
        """Proxy other methods to underlying cache."""
        return getattr(self._cache, name)


# =============================================================================
# Decorator for compressed caching
# =============================================================================

def compressed_cache(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    algorithm: CompressionAlgorithm = CompressionAlgorithm.ZLIB,
    level: int = 6,
    min_size: int = 1024,
    key_func: Optional[callable] = None
):
    """
    Caching decorator with automatic compression.

    Args:
        cache_name: Name of cache to use
        ttl: Time to live
        algorithm: Compression algorithm
        level: Compression level (1-9)
        min_size: Minimum size to compress (bytes)
        key_func: Custom key generation function

    Example:
        @compressed_cache(cache_name="large_data", algorithm=CompressionAlgorithm.LZMA)
        async def get_large_dataset():
            return await fetch_huge_json()
    """
    from functools import wraps

    config = CompressionConfig(
        algorithm=algorithm,
        level=level,
        min_size=min_size
    )
    middleware = CompressionMiddleware(config)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from omnicache.core.manager import manager

            cache = await manager.get_cache(cache_name, auto_create=True)

            # Generate key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")
                cache_key = ":".join(key_parts)

            # Try to get compressed value
            cached = await cache.get(cache_key)
            if cached is not None:
                if isinstance(cached, bytes):
                    return middleware.decode(cached)
                return cached

            # Execute and compress
            result = await func(*args, **kwargs)

            encoded = middleware.encode(result)

            if ttl:
                await cache.set(cache_key, encoded, ttl=ttl)
            else:
                await cache.set(cache_key, encoded)

            return result

        return wrapper
    return decorator


__all__ = [
    "CompressionAlgorithm",
    "CompressionConfig",
    "CompressionMiddleware",
    "CompressedCache",
    "Serializer",
    "JSONSerializer",
    "PickleSerializer",
    "Compressor",
    "ZlibCompressor",
    "GzipCompressor",
    "LzmaCompressor",
    "compressed_cache",
]
