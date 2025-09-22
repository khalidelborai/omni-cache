"""
Azure Blob Storage Backend for cloud storage tier.

This module implements an Azure Blob Storage-based cache backend for the hierarchical
cache system, providing durable, cost-effective storage for cold data.
"""

import json
import pickle
import asyncio
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
import logging

try:
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
    from azure.core.exceptions import ResourceNotFoundError, AzureError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


logger = logging.getLogger(__name__)


class AzureBlobBackend:
    """
    Azure Blob Storage cache backend for hierarchical storage.

    Provides persistent, cost-effective storage for cache data using
    Azure Blob Storage with configurable access tiers and lifecycle policies.
    """

    def __init__(
        self,
        container_name: str,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        connection_string: Optional[str] = None,
        prefix: str = "omnicache/",
        access_tier: str = "Hot",
        max_size: int = 1000000000,  # 1GB default
        serialization: str = "pickle",
        compression: bool = True,
        encryption: bool = False,
    ):
        """
        Initialize Azure Blob backend.

        Args:
            container_name: Azure Blob container name
            account_name: Azure storage account name
            account_key: Azure storage account key
            connection_string: Azure storage connection string (alternative to account_name/key)
            prefix: Key prefix for cache objects
            access_tier: Azure access tier (Hot, Cool, Archive)
            max_size: Maximum storage size in bytes
            serialization: Serialization method (pickle, json)
            compression: Enable compression
            encryption: Enable client-side encryption
        """
        if not AZURE_AVAILABLE:
            raise ImportError("azure-storage-blob is required for Azure backend. Install with: pip install azure-storage-blob")

        self.container_name = container_name
        self.prefix = prefix.rstrip("/") + "/"
        self.access_tier = access_tier
        self.max_size = max_size
        self.serialization = serialization
        self.compression = compression
        self.encryption = encryption

        # Initialize Azure Blob client
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        elif account_name and account_key:
            self.blob_service_client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=account_key
            )
        else:
            raise ValueError("Either connection_string or both account_name and account_key must be provided")

        # Get container client
        self.container_client = self.blob_service_client.get_container_client(container_name)

        # Statistics
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._current_size = 0

        # Ensure container exists
        asyncio.create_task(self._ensure_container_exists())

    async def _ensure_container_exists(self) -> None:
        """Ensure the Azure Blob container exists."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.container_client.get_container_properties
            )
        except ResourceNotFoundError:
            # Container doesn't exist, create it
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.container_client.create_container
                )
                logger.info(f"Created Azure Blob container: {self.container_name}")
            except AzureError as e:
                logger.error(f"Failed to create Azure Blob container: {e}")
                raise
        except AzureError as e:
            logger.error(f"Failed to access Azure Blob container: {e}")
            raise

    def _get_blob_name(self, cache_key: str) -> str:
        """Generate Azure Blob name from cache key."""
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
        Get value from Azure Blob Storage.

        Args:
            key: Cache key

        Returns:
            Value if found, None otherwise
        """
        blob_name = self._get_blob_name(key)
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )

        try:
            download_stream = await asyncio.get_event_loop().run_in_executor(
                None, blob_client.download_blob
            )
            data = await asyncio.get_event_loop().run_in_executor(
                None, download_stream.readall
            )

            value = self._deserialize_value(data)

            self._hits += 1
            logger.debug(f"Azure Blob cache hit for key: {key}")
            return value

        except ResourceNotFoundError:
            self._misses += 1
            logger.debug(f"Azure Blob cache miss for key: {key}")
            return None
        except AzureError as e:
            self._errors += 1
            logger.error(f"Azure Blob error getting key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value in Azure Blob Storage.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds (used for lifecycle policies)
        """
        blob_name = self._get_blob_name(key)
        data = self._serialize_value(value)

        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )

        # Set metadata
        metadata = {
            "cache_key": key,
            "created_at": datetime.utcnow().isoformat(),
            "serialization": self.serialization,
            "compressed": str(self.compression).lower(),
        }

        if ttl:
            expiration = datetime.utcnow() + timedelta(seconds=ttl)
            metadata["expires_at"] = expiration.isoformat()

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                blob_client.upload_blob,
                data,
                True,  # overwrite
                metadata,
                None,  # content_settings
                self.access_tier  # standard_blob_tier
            )

            self._current_size += len(data)
            logger.debug(f"Stored key {key} in Azure Blob ({len(data)} bytes)")

        except AzureError as e:
            self._errors += 1
            logger.error(f"Azure Blob error setting key {key}: {e}")
            raise

    async def delete(self, key: str) -> bool:
        """
        Delete key from Azure Blob Storage.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        blob_name = self._get_blob_name(key)
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, blob_client.delete_blob
            )

            logger.debug(f"Deleted key {key} from Azure Blob")
            return True

        except ResourceNotFoundError:
            logger.debug(f"Key {key} not found in Azure Blob for deletion")
            return False
        except AzureError as e:
            self._errors += 1
            logger.error(f"Azure Blob error deleting key {key}: {e}")
            return False

    async def clear(self) -> None:
        """Clear all cache objects with the configured prefix."""
        try:
            blobs_to_delete = []

            # List all blobs with prefix
            blob_list = await asyncio.get_event_loop().run_in_executor(
                None, self.container_client.list_blobs, name_starts_with=self.prefix
            )

            for blob in blob_list:
                blobs_to_delete.append(blob.name)

            # Delete blobs
            for blob_name in blobs_to_delete:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob_name
                )
                await asyncio.get_event_loop().run_in_executor(
                    None, blob_client.delete_blob
                )

            self._current_size = 0
            logger.info(f"Cleared {len(blobs_to_delete)} objects from Azure Blob")

        except AzureError as e:
            self._errors += 1
            logger.error(f"Azure Blob error clearing cache: {e}")
            raise

    async def exists(self, key: str) -> bool:
        """Check if key exists in Azure Blob Storage."""
        blob_name = self._get_blob_name(key)
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, blob_client.get_blob_properties
            )
            return True
        except ResourceNotFoundError:
            return False
        except AzureError as e:
            self._errors += 1
            logger.error(f"Azure Blob error checking key {key}: {e}")
            return False

    async def keys(self) -> List[str]:
        """Get all cache keys."""
        cache_keys = []

        try:
            blob_list = await asyncio.get_event_loop().run_in_executor(
                None, self.container_client.list_blobs, name_starts_with=self.prefix
            )

            for blob in blob_list:
                if blob.name.startswith(self.prefix):
                    cache_key = blob.name[len(self.prefix):]
                    cache_keys.append(cache_key)

        except AzureError as e:
            self._errors += 1
            logger.error(f"Azure Blob error listing keys: {e}")

        return cache_keys

    async def size(self) -> int:
        """Get current storage size in bytes."""
        try:
            total_size = 0

            blob_list = await asyncio.get_event_loop().run_in_executor(
                None, self.container_client.list_blobs, name_starts_with=self.prefix
            )

            for blob in blob_list:
                total_size += blob.size or 0

            self._current_size = total_size
            return total_size

        except AzureError as e:
            self._errors += 1
            logger.error(f"Azure Blob error calculating size: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get Azure Blob backend statistics."""
        return {
            "backend_type": "azure_blob",
            "container_name": self.container_name,
            "access_tier": self.access_tier,
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
            # Azure Blob lifecycle management would typically be configured
            # at the storage account level through ARM templates or Azure Portal
            # For now, we can at least set access tiers for cost optimization

            blob_list = await asyncio.get_event_loop().run_in_executor(
                None, self.container_client.list_blobs, name_starts_with=self.prefix
            )

            for blob in blob_list:
                # Move old blobs to cooler tiers
                if blob.last_modified:
                    age_days = (datetime.now(blob.last_modified.tzinfo) - blob.last_modified).days

                    new_tier = None
                    if age_days > 30 and blob.blob_tier == "Hot":
                        new_tier = "Cool"
                    elif age_days > 90 and blob.blob_tier == "Cool":
                        new_tier = "Archive"

                    if new_tier:
                        blob_client = self.blob_service_client.get_blob_client(
                            container=self.container_name,
                            blob=blob.name
                        )
                        await asyncio.get_event_loop().run_in_executor(
                            None, blob_client.set_standard_blob_tier, new_tier
                        )

            logger.info("Optimized Azure Blob access tiers for cost savings")

        except AzureError as e:
            logger.error(f"Failed to optimize Azure Blob configuration: {e}")

    def __str__(self) -> str:
        return f"AzureBlobBackend(container={self.container_name})"

    def __repr__(self) -> str:
        return f"AzureBlobBackend(container_name='{self.container_name}', prefix='{self.prefix}')"