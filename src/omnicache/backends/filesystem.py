"""
Filesystem backend implementation.

File-based storage backend with directory organization,
atomic operations, and disk space management.
"""

import os
import json
import shutil
import hashlib
import fnmatch
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from omnicache.models.backend import Backend
from omnicache.models.key import Key
from omnicache.models.value import Value, SerializerType
from omnicache.models.entry import CacheEntry
from omnicache.models.result import ClearResult
from omnicache.core.exceptions import CacheBackendError


class FileSystemBackend(Backend):
    """
    Filesystem storage backend.

    Provides persistent file-based cache storage with
    atomic operations, directory organization, and cleanup.
    """

    def __init__(
        self,
        path: str = ".cache",
        max_size_bytes: Optional[int] = None,
        create_dirs: bool = True,
        **config: Any
    ) -> None:
        """
        Initialize filesystem backend.

        Args:
            path: Root directory for cache storage
            max_size_bytes: Maximum total cache size in bytes
            create_dirs: Create directories if they don't exist
            **config: Additional configuration parameters
        """
        super().__init__(
            name="filesystem",
            path=path,
            max_size_bytes=max_size_bytes,
            create_dirs=create_dirs,
            **config
        )

        self.cache_path = Path(path).resolve()
        self.max_size_bytes = max_size_bytes
        self.create_dirs = create_dirs

        # Internal paths
        self.data_path = self.cache_path / "data"
        self.meta_path = self.cache_path / "meta"
        self.tags_path = self.cache_path / "tags"

        # Statistics
        self._total_gets = 0
        self._total_sets = 0
        self._total_deletes = 0
        self._total_hits = 0
        self._total_misses = 0

    async def initialize(self) -> None:
        """Initialize the filesystem backend."""
        try:
            if self.create_dirs:
                # Create cache directories
                self.cache_path.mkdir(parents=True, exist_ok=True)
                self.data_path.mkdir(exist_ok=True)
                self.meta_path.mkdir(exist_ok=True)
                self.tags_path.mkdir(exist_ok=True)

            # Verify directories exist and are writable
            if not self.cache_path.exists():
                raise CacheBackendError(f"Cache directory does not exist: {self.cache_path}")

            if not os.access(self.cache_path, os.W_OK):
                raise CacheBackendError(f"Cache directory not writable: {self.cache_path}")

            # Clean up expired entries on startup
            await self._cleanup_expired()

            await super().initialize()
            self._clear_error()

        except Exception as e:
            self._record_error(f"Failed to initialize filesystem backend: {str(e)}")
            raise CacheBackendError(f"Filesystem backend initialization failed: {str(e)}")

    async def shutdown(self) -> None:
        """Shutdown the filesystem backend."""
        try:
            # Optional: clean up expired entries on shutdown
            await self._cleanup_expired()

            await super().shutdown()
            self._clear_error()

        except Exception as e:
            self._record_error(f"Failed to shutdown filesystem backend: {str(e)}")
            raise CacheBackendError(f"Filesystem backend shutdown failed: {str(e)}")

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
        priority: Optional[float] = None
    ) -> None:
        """Store a cache entry to filesystem."""
        try:
            self._total_sets += 1

            # Check disk space if limit is set
            if self.max_size_bytes:
                current_size = await self.get_memory_usage()
                if current_size >= self.max_size_bytes:
                    raise CacheBackendError(f"Filesystem cache full (max_size={self.max_size_bytes})")

            # Create Key and Value objects
            cache_key = Key(value=key, tags=set(tags) if tags else None)
            cache_value = Value(data=value)

            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=cache_value,
                ttl=ttl,
                priority=priority or 0.5
            )

            # Get file paths
            data_file, meta_file = self._get_file_paths(key)

            # Write data and metadata atomically
            await self._write_entry_atomic(entry, data_file, meta_file)

            # Store tags if present
            if tags:
                await self._store_tags(key, tags)

            self._clear_error()

        except Exception as e:
            self._record_error(f"Failed to set key '{key}': {str(e)}")
            raise CacheBackendError(f"Filesystem backend set failed: {str(e)}")

    async def get(self, key: str) -> Any:
        """Retrieve a cache entry value from filesystem."""
        try:
            self._total_gets += 1

            entry = await self.get_entry(key)

            if entry is None:
                self._total_misses += 1
                return None

            self._total_hits += 1
            return entry.value.data

        except Exception as e:
            self._record_error(f"Failed to get key '{key}': {str(e)}")
            raise CacheBackendError(f"Filesystem backend get failed: {str(e)}")

    async def delete(self, key: str) -> bool:
        """Delete a cache entry from filesystem."""
        try:
            self._total_deletes += 1

            data_file, meta_file = self._get_file_paths(key)

            deleted = False

            # Remove data file
            if data_file.exists():
                data_file.unlink()
                deleted = True

            # Remove metadata file
            if meta_file.exists():
                meta_file.unlink()

            # Remove tags
            await self._remove_tags(key)

            self._clear_error()
            return deleted

        except Exception as e:
            self._record_error(f"Failed to delete key '{key}': {str(e)}")
            raise CacheBackendError(f"Filesystem backend delete failed: {str(e)}")

    async def get_entry(self, key: str) -> Optional[CacheEntry]:
        """Get complete cache entry with metadata from filesystem."""
        try:
            data_file, meta_file = self._get_file_paths(key)

            # Check if files exist
            if not data_file.exists() or not meta_file.exists():
                return None

            # Load metadata
            with meta_file.open('r') as f:
                meta_data = json.load(f)

            # Reconstruct entry
            entry = self._deserialize_entry(meta_data, data_file)

            # Check if expired
            if entry.is_expired():
                await self.delete(key)
                return None

            # Record access
            entry.access()

            # Update metadata with access info
            await self._update_metadata(meta_file, entry)

            self._clear_error()
            return entry

        except Exception as e:
            self._record_error(f"Failed to get entry '{key}': {str(e)}")
            raise CacheBackendError(f"Filesystem backend get_entry failed: {str(e)}")

    async def clear(
        self,
        pattern: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> ClearResult:
        """Clear cache entries from filesystem."""
        try:
            cleared_keys = []
            errors = []

            # Get all cache keys
            all_keys = await self.get_keys()

            for key in all_keys:
                should_clear = True

                # Check pattern match
                if pattern and not fnmatch.fnmatch(key, pattern):
                    should_clear = False

                # Check tag match
                if tags and should_clear:
                    entry_tags = await self._get_tags(key)
                    if not set(tags).issubset(entry_tags):
                        should_clear = False

                if should_clear:
                    try:
                        if await self.delete(key):
                            cleared_keys.append(key)
                    except Exception as e:
                        errors.append(f"Failed to clear key '{key}': {str(e)}")

            # Clean up empty tag files
            if tags is None:
                await self._clear_all_tags()

            self._clear_error()

            return ClearResult(
                cleared_count=len(cleared_keys),
                pattern=pattern,
                tags=set(tags) if tags else None,
                error_count=len(errors),
                errors=errors
            )

        except Exception as e:
            self._record_error(f"Failed to clear entries: {str(e)}")
            raise CacheBackendError(f"Filesystem backend clear failed: {str(e)}")

    async def exists(self, key: str) -> bool:
        """Check if a cache entry exists in filesystem."""
        try:
            data_file, meta_file = self._get_file_paths(key)
            return data_file.exists() and meta_file.exists()

        except Exception as e:
            self._record_error(f"Failed to check existence of key '{key}': {str(e)}")
            raise CacheBackendError(f"Filesystem backend exists failed: {str(e)}")

    async def get_size(self) -> int:
        """Get the number of cache entries in filesystem."""
        try:
            # Clean up expired entries first
            await self._cleanup_expired()

            # Count data files
            return len(list(self.data_path.glob("*.dat")))

        except Exception as e:
            self._record_error(f"Failed to get size: {str(e)}")
            raise CacheBackendError(f"Filesystem backend get_size failed: {str(e)}")

    async def get_memory_usage(self) -> int:
        """Get total disk usage in bytes."""
        try:
            total_bytes = 0

            # Calculate size of all cache files
            for file_path in self.cache_path.rglob("*"):
                if file_path.is_file():
                    total_bytes += file_path.stat().st_size

            return total_bytes

        except Exception as e:
            self._record_error(f"Failed to calculate disk usage: {str(e)}")
            raise CacheBackendError(f"Filesystem backend get_memory_usage failed: {str(e)}")

    async def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get list of cache keys matching optional pattern."""
        try:
            # Clean up expired entries first
            await self._cleanup_expired()

            keys = []

            # Get all data files
            for data_file in self.data_path.glob("*.dat"):
                key = self._filename_to_key(data_file.stem)

                if pattern is None or fnmatch.fnmatch(key, pattern):
                    keys.append(key)

            return keys

        except Exception as e:
            self._record_error(f"Failed to get keys: {str(e)}")
            raise CacheBackendError(f"Filesystem backend get_keys failed: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get backend statistics."""
        return {
            "total_gets": self._total_gets,
            "total_sets": self._total_sets,
            "total_deletes": self._total_deletes,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": self._total_hits / max(self._total_gets, 1),
            "cache_path": str(self.cache_path),
            "max_size_bytes": self.max_size_bytes
        }

    def _get_file_paths(self, key: str) -> tuple[Path, Path]:
        """Get data and metadata file paths for a key."""
        filename = self._key_to_filename(key)
        data_file = self.data_path / f"{filename}.dat"
        meta_file = self.meta_path / f"{filename}.meta"
        return data_file, meta_file

    def _key_to_filename(self, key: str) -> str:
        """Convert cache key to safe filename."""
        # Use SHA-256 hash for safe, consistent filenames
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    def _filename_to_key(self, filename: str) -> str:
        """Convert filename back to cache key."""
        # Load key from metadata file
        meta_file = self.meta_path / f"{filename}.meta"
        if meta_file.exists():
            with meta_file.open('r') as f:
                meta_data = json.load(f)
                return meta_data.get("key", {}).get("value", filename)
        return filename

    async def _write_entry_atomic(self, entry: CacheEntry, data_file: Path, meta_file: Path) -> None:
        """Write cache entry atomically using temporary files."""
        # Create temporary files
        temp_data = tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=self.data_path)
        temp_meta = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=self.meta_path)

        try:
            # Write data
            temp_data.write(entry.value.serialized_data.encode() if isinstance(entry.value.serialized_data, str) else entry.value.serialized_data)
            temp_data.close()

            # Write metadata
            meta_data = entry.to_dict()
            json.dump(meta_data, temp_meta, default=str, indent=2)
            temp_meta.close()

            # Atomic move to final locations
            shutil.move(temp_data.name, data_file)
            shutil.move(temp_meta.name, meta_file)

        except Exception:
            # Clean up temporary files on error
            try:
                os.unlink(temp_data.name)
            except:
                pass
            try:
                os.unlink(temp_meta.name)
            except:
                pass
            raise

    def _deserialize_entry(self, meta_data: Dict[str, Any], data_file: Path) -> CacheEntry:
        """Reconstruct cache entry from metadata and data file."""
        # Reconstruct Key object
        key_data = meta_data["key"]
        cache_key = Key.from_dict(key_data)

        # Load and reconstruct Value object
        with data_file.open('rb') as f:
            serialized_data = f.read()

        value_data = meta_data["value"]
        serializer_type = SerializerType(value_data["serializer_type"])

        cache_value = Value.from_serialized(
            serialized_data=serialized_data,
            serializer_type=serializer_type,
            content_type=value_data.get("content_type"),
            version=value_data.get("version", 1)
        )

        # Create CacheEntry
        entry = CacheEntry(
            key=cache_key,
            value=cache_value,
            ttl=meta_data.get("ttl_remaining"),
            priority=meta_data.get("priority", 0.5)
        )

        return entry

    async def _update_metadata(self, meta_file: Path, entry: CacheEntry) -> None:
        """Update metadata file with current entry state."""
        try:
            meta_data = entry.to_dict()
            with meta_file.open('w') as f:
                json.dump(meta_data, f, default=str, indent=2)
        except Exception:
            # Non-critical operation, don't fail the request
            pass

    async def _cleanup_expired(self) -> None:
        """Clean up expired cache entries."""
        try:
            expired_keys = []

            for meta_file in self.meta_path.glob("*.meta"):
                try:
                    with meta_file.open('r') as f:
                        meta_data = json.load(f)

                    # Check if expired
                    expires_at = meta_data.get("expires_at")
                    if expires_at:
                        expire_time = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                        if datetime.now() >= expire_time:
                            key = meta_data.get("key", {}).get("value", "")
                            if key:
                                expired_keys.append(key)

                except Exception:
                    # Skip corrupted files
                    continue

            # Delete expired entries
            for key in expired_keys:
                try:
                    await self.delete(key)
                except Exception:
                    # Continue with other deletions
                    continue

        except Exception:
            # Non-critical operation
            pass

    async def _store_tags(self, key: str, tags: List[str]) -> None:
        """Store tag associations for a key."""
        if not tags:
            return

        tag_file = self.tags_path / f"{self._key_to_filename(key)}.tags"
        with tag_file.open('w') as f:
            json.dump(list(tags), f)

    async def _get_tags(self, key: str) -> set:
        """Get tags for a key."""
        tag_file = self.tags_path / f"{self._key_to_filename(key)}.tags"

        if not tag_file.exists():
            return set()

        try:
            with tag_file.open('r') as f:
                tags = json.load(f)
                return set(tags)
        except Exception:
            return set()

    async def _remove_tags(self, key: str) -> None:
        """Remove tag associations for a key."""
        tag_file = self.tags_path / f"{self._key_to_filename(key)}.tags"

        if tag_file.exists():
            try:
                tag_file.unlink()
            except Exception:
                pass

    async def _clear_all_tags(self) -> None:
        """Clear all tag files."""
        try:
            for tag_file in self.tags_path.glob("*.tags"):
                tag_file.unlink()
        except Exception:
            pass

    def __str__(self) -> str:
        return f"FileSystem({self.cache_path})"