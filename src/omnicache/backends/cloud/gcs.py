"""
Google Cloud Storage Backend for cloud storage tier.

This module implements a GCS-based cache backend for the hierarchical
cache system, providing durable, cost-effective storage for cold data.
"""

import json
import pickle
import asyncio
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
import logging

try:
    from google.cloud import storage
    from google.cloud.exceptions import NotFound, GoogleCloudError
    from google.auth.exceptions import DefaultCredentialsError
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


logger = logging.getLogger(__name__)


class GCSBackend:
    """
    Google Cloud Storage cache backend for hierarchical storage.

    Provides persistent, cost-effective storage for cache data using
    Google Cloud Storage with configurable storage classes and lifecycle policies.
    """

    def __init__(
        self,
        bucket_name: str,
        project_id: Optional[str] = None,
        prefix: str = "omnicache/",
        storage_class: str = "STANDARD",
        credentials_path: Optional[str] = None,
        max_size: int = 1000000000,  # 1GB default
        serialization: str = "pickle",
        compression: bool = True,
        encryption: bool = False,
    ):
        """
        Initialize GCS backend.

        Args:
            bucket_name: GCS bucket name
            project_id: Google Cloud project ID
            prefix: Key prefix for cache objects
            storage_class: GCS storage class (STANDARD, NEARLINE, COLDLINE, ARCHIVE)
            credentials_path: Path to service account credentials JSON
            max_size: Maximum storage size in bytes
            serialization: Serialization method (pickle, json)
            compression: Enable compression
            encryption: Enable customer-managed encryption
        """
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage is required for GCS backend. Install with: pip install google-cloud-storage")

        self.bucket_name = bucket_name
        self.project_id = project_id
        self.prefix = prefix.rstrip("/") + "/"
        self.storage_class = storage_class
        self.max_size = max_size
        self.serialization = serialization
        self.compression = compression
        self.encryption = encryption

        # Initialize GCS client
        try:
            if credentials_path:
                self.client = storage.Client.from_service_account_json(
                    credentials_path, project=project_id
                )
            else:
                self.client = storage.Client(project=project_id)
        except DefaultCredentialsError:
            logger.error("No GCS credentials found. Set GOOGLE_APPLICATION_CREDENTIALS or provide credentials_path")
            raise

        # Get bucket reference
        self.bucket = self.client.bucket(bucket_name)

        # Statistics
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._current_size = 0

        # Ensure bucket exists
        asyncio.create_task(self._ensure_bucket_exists())

    async def _ensure_bucket_exists(self) -> None:
        """Ensure the GCS bucket exists."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.bucket.reload
            )
        except NotFound:
            # Bucket doesn't exist, create it
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.client.create_bucket,
                    self.bucket,
                    location="US"  # Default location
                )
                logger.info(f"Created GCS bucket: {self.bucket_name}")
            except GoogleCloudError as e:
                logger.error(f"Failed to create GCS bucket: {e}")
                raise
        except GoogleCloudError as e:
            logger.error(f"Failed to access GCS bucket: {e}")
            raise

    def _get_blob_name(self, cache_key: str) -> str:
        """Generate GCS blob name from cache key."""
        return f"{self.prefix}{cache_key}"

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.serialization == "pickle":
            data = pickle.dumps(value)
        elif self.serialization == "json":
            data = json.dumps(value).encode('utf-8')
        else:
            # Fallback to string representation
            data = str(value).encode('utf-8')

        if self.compression:
            import gzip
            data = gzip.compress(data)

        return data

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if self.compression:
            import gzip
            data = gzip.decompress(data)

        if self.serialization == "pickle":
            return pickle.loads(data)
        elif self.serialization == "json":
            return json.loads(data.decode('utf-8'))
        else:
            return data.decode('utf-8')

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from GCS.

        Args:
            key: Cache key

        Returns:
            Value if found, None otherwise
        """
        blob_name = self._get_blob_name(key)
        blob = self.bucket.blob(blob_name)

        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None, blob.download_as_bytes
            )

            value = self._deserialize_value(data)

            self._hits += 1
            logger.debug(f"GCS cache hit for key: {key}")
            return value

        except NotFound:
            self._misses += 1
            logger.debug(f"GCS cache miss for key: {key}")
            return None
        except GoogleCloudError as e:
            self._errors += 1
            logger.error(f"GCS error getting key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value in GCS.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds (used for object lifecycle)
        """
        blob_name = self._get_blob_name(key)
        data = self._serialize_value(value)

        blob = self.bucket.blob(blob_name)

        # Set storage class
        blob.storage_class = self.storage_class

        # Set metadata
        blob.metadata = {
            "cache-key": key,
            "created-at": datetime.utcnow().isoformat(),
            "serialization": self.serialization,
            "compressed": str(self.compression).lower(),
        }

        # Set content type
        blob.content_type = "application/octet-stream"

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, blob.upload_from_string, data
            )

            self._current_size += len(data)
            logger.debug(f"Stored key {key} in GCS ({len(data)} bytes)")

        except GoogleCloudError as e:
            self._errors += 1
            logger.error(f"GCS error setting key {key}: {e}")
            raise

    async def delete(self, key: str) -> bool:
        """
        Delete key from GCS.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        blob_name = self._get_blob_name(key)
        blob = self.bucket.blob(blob_name)

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, blob.delete
            )

            logger.debug(f"Deleted key {key} from GCS")
            return True

        except NotFound:
            logger.debug(f"Key {key} not found in GCS for deletion")
            return False
        except GoogleCloudError as e:
            self._errors += 1
            logger.error(f"GCS error deleting key {key}: {e}")
            return False

    async def clear(self) -> None:
        """Clear all cache objects with the configured prefix."""
        try:
            blobs_to_delete = []

            # List all blobs with prefix
            blobs = await asyncio.get_event_loop().run_in_executor(
                None, self.client.list_blobs, self.bucket, prefix=self.prefix
            )

            for blob in blobs:
                blobs_to_delete.append(blob)

            # Delete blobs
            for blob in blobs_to_delete:
                await asyncio.get_event_loop().run_in_executor(
                    None, blob.delete
                )

            self._current_size = 0
            logger.info(f"Cleared {len(blobs_to_delete)} objects from GCS")

        except GoogleCloudError as e:
            self._errors += 1
            logger.error(f"GCS error clearing cache: {e}")
            raise

    async def exists(self, key: str) -> bool:
        """Check if key exists in GCS."""
        blob_name = self._get_blob_name(key)
        blob = self.bucket.blob(blob_name)

        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, blob.exists
            )
        except GoogleCloudError as e:
            self._errors += 1
            logger.error(f"GCS error checking key {key}: {e}")
            return False

    async def keys(self) -> List[str]:
        """Get all cache keys."""
        cache_keys = []

        try:
            blobs = await asyncio.get_event_loop().run_in_executor(
                None, self.client.list_blobs, self.bucket, prefix=self.prefix
            )

            for blob in blobs:
                if blob.name.startswith(self.prefix):
                    cache_key = blob.name[len(self.prefix):]
                    cache_keys.append(cache_key)

        except GoogleCloudError as e:
            self._errors += 1
            logger.error(f"GCS error listing keys: {e}")

        return cache_keys

    async def size(self) -> int:
        """Get current storage size in bytes."""
        try:
            total_size = 0

            blobs = await asyncio.get_event_loop().run_in_executor(
                None, self.client.list_blobs, self.bucket, prefix=self.prefix
            )

            for blob in blobs:
                total_size += blob.size or 0

            self._current_size = total_size
            return total_size

        except GoogleCloudError as e:
            self._errors += 1
            logger.error(f"GCS error calculating size: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get GCS backend statistics."""
        return {
            "backend_type": "gcs",
            "bucket_name": self.bucket_name,
            "project_id": self.project_id,
            "storage_class": self.storage_class,
            "hits": self._hits,
            "misses": self._misses,
            "errors": self._errors,
            "hit_ratio": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0,
            "current_size_bytes": self._current_size,
            "max_size_bytes": self.max_size,
            "utilization": (self._current_size / self.max_size * 100) if self.max_size > 0 else 0.0,
            "compression": self.compression,
            "encryption": self.encryption,
            "serialization": self.serialization,
        }

    async def optimize(self) -> None:
        """Run optimization tasks like setting lifecycle policies."""
        try:
            # Set lifecycle policy for cost optimization
            lifecycle_rule = {
                "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
                "condition": {"age": 30}
            }

            lifecycle_rule_coldline = {
                "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
                "condition": {"age": 90}
            }

            lifecycle_rule_archive = {
                "action": {"type": "SetStorageClass", "storageClass": "ARCHIVE"},
                "condition": {"age": 365}
            }

            lifecycle_policy = [lifecycle_rule, lifecycle_rule_coldline, lifecycle_rule_archive]

            await asyncio.get_event_loop().run_in_executor(
                None, setattr, self.bucket, "lifecycle_rules", lifecycle_policy
            )

            await asyncio.get_event_loop().run_in_executor(
                None, self.bucket.patch
            )

            logger.info("Updated GCS lifecycle policy for cost optimization")

        except GoogleCloudError as e:
            logger.error(f"Failed to optimize GCS configuration: {e}")

    def __str__(self) -> str:
        return f"GCSBackend(bucket={self.bucket_name}, project={self.project_id})"

    def __repr__(self) -> str:
        return f"GCSBackend(bucket_name='{self.bucket_name}', project_id='{self.project_id}', prefix='{self.prefix}')"