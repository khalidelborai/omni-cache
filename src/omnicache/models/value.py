"""
Value entity model.

Defines the Value class that represents cache entry values with serialization
and metadata.
"""

import hashlib
import json
import pickle
from datetime import datetime
from typing import Any, Dict, Optional, Union
from enum import Enum


class SerializerType(Enum):
    """Serializer type enumeration."""
    NONE = "none"
    JSON = "json"
    PICKLE = "pickle"
    STRING = "string"
    BYTES = "bytes"


class Value:
    """
    Cache value with serialization and metadata.

    Represents a cache value with automatic serialization,
    size tracking, and integrity validation.
    """

    def __init__(
        self,
        data: Any,
        serializer: Optional[SerializerType] = None,
        content_type: Optional[str] = None,
        version: int = 1
    ) -> None:
        """
        Initialize a cache value.

        Args:
            data: The value data to store
            serializer: Serialization method to use
            content_type: MIME content type for the data
            version: Version number for the value

        Raises:
            ValueError: If validation fails
            TypeError: If data cannot be serialized
        """
        # Store original data
        self._data = data
        self._version = version

        # Auto-detect serializer if not provided
        self._serializer_type = serializer or self._detect_serializer(data)

        # Serialize the data
        self._serialized_data = self._serialize(data, self._serializer_type)

        # Set content type
        self._content_type = content_type or self._detect_content_type(data, self._serializer_type)

        # Calculate metadata
        self._size_bytes = len(self._serialized_data) if isinstance(self._serialized_data, (bytes, str)) else 0
        self._checksum = self._compute_checksum()
        self._created_at = datetime.now()

    @property
    def data(self) -> Any:
        """Get the original data."""
        return self._data

    @property
    def serialized_data(self) -> Union[str, bytes]:
        """Get the serialized data."""
        return self._serialized_data

    @property
    def serializer_type(self) -> SerializerType:
        """Get the serializer type used."""
        return self._serializer_type

    @property
    def content_type(self) -> str:
        """Get the content type."""
        return self._content_type

    @property
    def size_bytes(self) -> int:
        """Get the size in bytes."""
        return self._size_bytes

    @property
    def checksum(self) -> str:
        """Get the data checksum."""
        return self._checksum

    @property
    def version(self) -> int:
        """Get the value version."""
        return self._version

    @property
    def created_at(self) -> datetime:
        """Get the creation timestamp."""
        return self._created_at

    def verify_integrity(self) -> bool:
        """
        Verify data integrity using checksum.

        Returns:
            True if data integrity is valid, False otherwise
        """
        current_checksum = self._compute_checksum()
        return current_checksum == self._checksum

    def deserialize(self) -> Any:
        """
        Deserialize the stored data.

        Returns:
            The original data object

        Raises:
            ValueError: If deserialization fails
        """
        return self._deserialize(self._serialized_data, self._serializer_type)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert value to dictionary representation.

        Returns:
            Dictionary containing value metadata
        """
        return {
            "serializer_type": self._serializer_type.value,
            "content_type": self._content_type,
            "size_bytes": self._size_bytes,
            "checksum": self._checksum,
            "version": self._version,
            "created_at": self._created_at.isoformat()
        }

    @classmethod
    def from_serialized(
        cls,
        serialized_data: Union[str, bytes],
        serializer_type: SerializerType,
        content_type: Optional[str] = None,
        version: int = 1
    ) -> 'Value':
        """
        Create a Value instance from already serialized data.

        Args:
            serialized_data: The serialized data
            serializer_type: The serializer used
            content_type: Optional content type
            version: Value version

        Returns:
            Value instance
        """
        # Create a temporary instance to deserialize
        temp_value = cls.__new__(cls)
        temp_value._serialized_data = serialized_data
        temp_value._serializer_type = serializer_type

        # Deserialize to get original data
        original_data = temp_value._deserialize(serialized_data, serializer_type)

        # Create proper instance
        return cls(original_data, serializer_type, content_type, version)

    def _detect_serializer(self, data: Any) -> SerializerType:
        """Auto-detect appropriate serializer for data."""
        if data is None:
            return SerializerType.NONE
        elif isinstance(data, str):
            return SerializerType.STRING
        elif isinstance(data, bytes):
            return SerializerType.BYTES
        elif isinstance(data, (dict, list, int, float, bool)):
            return SerializerType.JSON
        else:
            return SerializerType.PICKLE

    def _detect_content_type(self, data: Any, serializer: SerializerType) -> str:
        """Auto-detect content type based on data and serializer."""
        if serializer == SerializerType.JSON:
            return "application/json"
        elif serializer == SerializerType.STRING:
            return "text/plain"
        elif serializer == SerializerType.BYTES:
            return "application/octet-stream"
        elif serializer == SerializerType.PICKLE:
            return "application/python-pickle"
        else:
            return "application/octet-stream"

    def _serialize(self, data: Any, serializer: SerializerType) -> Union[str, bytes]:
        """Serialize data using specified serializer."""
        try:
            if serializer == SerializerType.NONE:
                return b""
            elif serializer == SerializerType.STRING:
                return str(data) if not isinstance(data, str) else data
            elif serializer == SerializerType.BYTES:
                return data if isinstance(data, bytes) else str(data).encode('utf-8')
            elif serializer == SerializerType.JSON:
                return json.dumps(data, default=str, ensure_ascii=False)
            elif serializer == SerializerType.PICKLE:
                return pickle.dumps(data)
            else:
                raise ValueError(f"Unknown serializer type: {serializer}")

        except Exception as e:
            raise TypeError(f"Failed to serialize data with {serializer.value}: {str(e)}")

    def _deserialize(self, serialized_data: Union[str, bytes], serializer: SerializerType) -> Any:
        """Deserialize data using specified serializer."""
        try:
            if serializer == SerializerType.NONE:
                return None
            elif serializer == SerializerType.STRING:
                return serialized_data
            elif serializer == SerializerType.BYTES:
                return serialized_data
            elif serializer == SerializerType.JSON:
                if isinstance(serialized_data, bytes):
                    serialized_data = serialized_data.decode('utf-8')
                return json.loads(serialized_data)
            elif serializer == SerializerType.PICKLE:
                if isinstance(serialized_data, str):
                    serialized_data = serialized_data.encode('utf-8')
                return pickle.loads(serialized_data)
            else:
                raise ValueError(f"Unknown serializer type: {serializer}")

        except Exception as e:
            raise ValueError(f"Failed to deserialize data with {serializer.value}: {str(e)}")

    def _compute_checksum(self) -> str:
        """Compute SHA-256 checksum of serialized data."""
        if isinstance(self._serialized_data, str):
            data_bytes = self._serialized_data.encode('utf-8')
        else:
            data_bytes = self._serialized_data

        return hashlib.sha256(data_bytes).hexdigest()

    def __str__(self) -> str:
        size_kb = self._size_bytes / 1024 if self._size_bytes >= 1024 else self._size_bytes
        size_unit = "KB" if self._size_bytes >= 1024 else "B"
        return f"Value({self._serializer_type.value}, {size_kb:.1f}{size_unit})"

    def __repr__(self) -> str:
        return (f"<Value(serializer={self._serializer_type.value}, "
                f"size={self._size_bytes}, version={self._version})>")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Value):
            return False
        return (
            self._checksum == other._checksum and
            self._serializer_type == other._serializer_type
        )

    def __hash__(self) -> int:
        return hash((self._checksum, self._serializer_type.value))