"""
Amazon S3 Backend for cloud storage tier.

This module implements an S3-based cache backend for the hierarchical
cache system, providing durable, cost-effective storage for cold data.
"""

import json
import pickle
import asyncio
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
import logging

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


logger = logging.getLogger(__name__)


class S3Backend:
    """
    Amazon S3 cache backend for hierarchical storage.

    Provides persistent, cost-effective storage for cache data using
    Amazon S3 with configurable storage classes and lifecycle policies.
    """

    def __init__(
        self,
        bucket_name: str,
        region: str = "us-east-1",
        prefix: str = "omnicache/",
        storage_class: str = "STANDARD",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        max_size: int = 1000000000,  # 1GB default
        serialization: str = "pickle",
        compression: bool = True,
        encryption: bool = False,
    ):
        """
        Initialize S3 backend.

        Args:
            bucket_name: S3 bucket name
            region: AWS region
            prefix: Key prefix for cache objects
            storage_class: S3 storage class (STANDARD, IA, GLACIER, etc.)
            aws_access_key_id: AWS access key (optional, uses environment/IAM)
            aws_secret_access_key: AWS secret key (optional)
            endpoint_url: Custom S3 endpoint (for S3-compatible services)
            max_size: Maximum storage size in bytes
            serialization: Serialization method (pickle, json)
            compression: Enable compression
            encryption: Enable server-side encryption
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 backend. Install with: pip install boto3")

        self.bucket_name = bucket_name
        self.region = region
        self.prefix = prefix.rstrip("/") + "/"
        self.storage_class = storage_class
        self.max_size = max_size
        self.serialization = serialization
        self.compression = compression
        self.encryption = encryption

        # Initialize S3 client
        session_kwargs = {}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
            })

        self.session = boto3.Session(**session_kwargs)

        client_kwargs = {"region_name": region}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        self.s3_client = self.session.client("s3", **client_kwargs)

        # Statistics
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._current_size = 0

        # Ensure bucket exists
        asyncio.create_task(self._ensure_bucket_exists())

    async def _ensure_bucket_exists(self) -> None:
        """Ensure the S3 bucket exists."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.s3_client.head_bucket, {"Bucket": self.bucket_name}
            )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                # Bucket doesn't exist, create it
                try:
                    if self.region == "us-east-1":
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.s3_client.create_bucket, {"Bucket": self.bucket_name}
                        )
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            self.s3_client.create_bucket,
                            {
                                "Bucket": self.bucket_name,
                                "CreateBucketConfiguration": {"LocationConstraint": self.region}
                            }
                        )
                    logger.info(f"Created S3 bucket: {self.bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Failed to create S3 bucket: {create_error}")
                    raise
            else:
                logger.error(f"Failed to access S3 bucket: {e}")
                raise

    def _get_s3_key(self, cache_key: str) -> str:
        """Generate S3 object key from cache key."""
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
        Get value from S3.

        Args:
            key: Cache key

        Returns:
            Value if found, None otherwise
        """
        s3_key = self._get_s3_key(key)

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.s3_client.get_object,
                {"Bucket": self.bucket_name, "Key": s3_key}
            )

            data = response["Body"].read()
            value = self._deserialize_value(data)

            self._hits += 1
            logger.debug(f"S3 cache hit for key: {key}")
            return value

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                self._misses += 1
                logger.debug(f"S3 cache miss for key: {key}")
                return None
            else:
                self._errors += 1
                logger.error(f"S3 error getting key {key}: {e}")
                return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value in S3.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds (used for object expiration)
        """
        s3_key = self._get_s3_key(key)
        data = self._serialize_value(value)

        put_kwargs = {
            "Bucket": self.bucket_name,
            "Key": s3_key,
            "Body": data,
            "StorageClass": self.storage_class,
        }

        # Add metadata
        put_kwargs["Metadata"] = {
            "cache-key": key,
            "created-at": datetime.utcnow().isoformat(),
            "serialization": self.serialization,
            "compressed": str(self.compression).lower(),
        }

        # Set expiration if TTL provided
        if ttl:
            expiration = datetime.utcnow() + timedelta(seconds=ttl)
            put_kwargs["Expires"] = expiration

        # Enable encryption if configured
        if self.encryption:
            put_kwargs["ServerSideEncryption"] = "AES256"

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.s3_client.put_object, put_kwargs
            )

            self._current_size += len(data)
            logger.debug(f"Stored key {key} in S3 ({len(data)} bytes)")

        except ClientError as e:
            self._errors += 1
            logger.error(f"S3 error setting key {key}: {e}")
            raise

    async def delete(self, key: str) -> bool:
        """
        Delete key from S3.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        s3_key = self._get_s3_key(key)

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.s3_client.delete_object,
                {"Bucket": self.bucket_name, "Key": s3_key}
            )

            logger.debug(f"Deleted key {key} from S3")
            return True

        except ClientError as e:
            self._errors += 1
            logger.error(f"S3 error deleting key {key}: {e}")
            return False

    async def clear(self) -> None:
        """Clear all cache objects with the configured prefix."""
        try:
            # List all objects with prefix
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)

            objects_to_delete = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        objects_to_delete.append({"Key": obj["Key"]})

            # Delete objects in batches
            if objects_to_delete:
                # S3 allows up to 1000 objects per delete request
                for i in range(0, len(objects_to_delete), 1000):
                    batch = objects_to_delete[i:i+1000]
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.s3_client.delete_objects,
                        {
                            "Bucket": self.bucket_name,
                            "Delete": {"Objects": batch}
                        }
                    )

            self._current_size = 0
            logger.info(f"Cleared {len(objects_to_delete)} objects from S3")

        except ClientError as e:
            self._errors += 1
            logger.error(f"S3 error clearing cache: {e}")
            raise

    async def exists(self, key: str) -> bool:
        """Check if key exists in S3."""
        s3_key = self._get_s3_key(key)

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.s3_client.head_object,
                {"Bucket": self.bucket_name, "Key": s3_key}
            )
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                return False
            else:
                self._errors += 1
                logger.error(f"S3 error checking key {key}: {e}")
                return False

    async def keys(self) -> List[str]:
        """Get all cache keys."""
        cache_keys = []

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)

            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        s3_key = obj["Key"]
                        if s3_key.startswith(self.prefix):
                            cache_key = s3_key[len(self.prefix):]
                            cache_keys.append(cache_key)

        except ClientError as e:
            self._errors += 1
            logger.error(f"S3 error listing keys: {e}")

        return cache_keys

    async def size(self) -> int:
        """Get current storage size in bytes."""
        try:
            total_size = 0
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)

            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        total_size += obj["Size"]

            self._current_size = total_size
            return total_size

        except ClientError as e:
            self._errors += 1
            logger.error(f"S3 error calculating size: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get S3 backend statistics."""
        return {
            "backend_type": "s3",
            "bucket_name": self.bucket_name,
            "region": self.region,
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
        """Run optimization tasks like transitioning to cheaper storage classes."""
        try:
            # Set lifecycle policy for cost optimization
            lifecycle_config = {
                "Rules": [
                    {
                        "ID": "omnicache-transition",
                        "Status": "Enabled",
                        "Filter": {"Prefix": self.prefix},
                        "Transitions": [
                            {
                                "Days": 30,
                                "StorageClass": "STANDARD_IA"
                            },
                            {
                                "Days": 90,
                                "StorageClass": "GLACIER"
                            },
                            {
                                "Days": 365,
                                "StorageClass": "DEEP_ARCHIVE"
                            }
                        ]
                    }
                ]
            }

            await asyncio.get_event_loop().run_in_executor(
                None,
                self.s3_client.put_bucket_lifecycle_configuration,
                {
                    "Bucket": self.bucket_name,
                    "LifecycleConfiguration": lifecycle_config
                }
            )

            logger.info("Updated S3 lifecycle policy for cost optimization")

        except ClientError as e:
            logger.error(f"Failed to optimize S3 configuration: {e}")

    def __str__(self) -> str:
        return f"S3Backend(bucket={self.bucket_name}, region={self.region})"

    def __repr__(self) -> str:
        return f"S3Backend(bucket_name='{self.bucket_name}', region='{self.region}', prefix='{self.prefix}')"