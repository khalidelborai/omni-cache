"""
Cloud storage backends for L3 cache tier.

This module provides cloud storage backends for OmniCache including:
- AWS S3 backend
- Azure Blob Storage backend
- Google Cloud Storage backend
- Multi-cloud cost optimization
"""

from .s3 import S3Backend
from .gcs import GCSBackend
from .azure import AzureBlobBackend

__all__ = [
    "S3Backend",
    "GCSBackend",
    "AzureBlobBackend",
]

__version__ = "0.1.0"