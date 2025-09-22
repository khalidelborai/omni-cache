"""
Key entity model.

Defines the Key class that represents cache entry keys with metadata.
"""

import hashlib
import re
from datetime import datetime
from typing import Set, Optional, Any, Dict


class Key:
    """
    Cache key with metadata and validation.

    Represents a cache key with associated metadata like tags,
    namespace, and creation timestamp.
    """

    # Valid key pattern: non-empty string, no control characters
    _KEY_PATTERN = re.compile(r'^[^\x00-\x1f\x7f]+$')

    # Valid namespace pattern: alphanumeric, underscores, dots (empty string allowed)
    _NAMESPACE_PATTERN = re.compile(r'^[a-zA-Z0-9._]*$')

    # Valid tag pattern: alphanumeric, hyphens, underscores
    _TAG_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

    def __init__(
        self,
        value: str,
        namespace: str = "",
        tags: Optional[Set[str]] = None
    ) -> None:
        """
        Initialize a cache key.

        Args:
            value: The key string value
            namespace: Optional namespace for the key
            tags: Optional set of tags for bulk operations

        Raises:
            ValueError: If validation fails
        """
        # Validate inputs
        self._validate_value(value)
        self._validate_namespace(namespace)
        if tags:
            self._validate_tags(tags)

        # Core attributes
        self._value = value
        self._namespace = namespace
        self._tags = tags or set()

        # Metadata
        self._created_at = datetime.now()
        self._hash_value = self._compute_hash()

    @property
    def value(self) -> str:
        """Get the key value."""
        return self._value

    @property
    def namespace(self) -> str:
        """Get the key namespace."""
        return self._namespace

    @property
    def tags(self) -> Set[str]:
        """Get the key tags."""
        return self._tags.copy()

    @property
    def hash_value(self) -> str:
        """Get the computed hash value."""
        return self._hash_value

    @property
    def created_at(self) -> datetime:
        """Get the key creation timestamp."""
        return self._created_at

    @property
    def full_key(self) -> str:
        """Get the full namespaced key."""
        if self._namespace:
            return f"{self._namespace}:{self._value}"
        return self._value

    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the key.

        Args:
            tag: Tag to add

        Raises:
            ValueError: If tag is invalid
        """
        self._validate_tag(tag)
        self._tags.add(tag)

    def remove_tag(self, tag: str) -> bool:
        """
        Remove a tag from the key.

        Args:
            tag: Tag to remove

        Returns:
            True if tag was removed, False if not found
        """
        try:
            self._tags.remove(tag)
            return True
        except KeyError:
            return False

    def has_tag(self, tag: str) -> bool:
        """
        Check if key has a specific tag.

        Args:
            tag: Tag to check

        Returns:
            True if key has the tag, False otherwise
        """
        return tag in self._tags

    def matches_pattern(self, pattern: str) -> bool:
        """
        Check if key matches a glob-style pattern.

        Args:
            pattern: Glob pattern to match against

        Returns:
            True if key matches pattern, False otherwise
        """
        import fnmatch
        return fnmatch.fnmatch(self.full_key, pattern)

    def matches_tags(self, required_tags: Set[str]) -> bool:
        """
        Check if key has all required tags.

        Args:
            required_tags: Set of tags that must be present

        Returns:
            True if all required tags are present, False otherwise
        """
        return required_tags.issubset(self._tags)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert key to dictionary representation.

        Returns:
            Dictionary containing key data
        """
        return {
            "value": self._value,
            "namespace": self._namespace,
            "tags": list(self._tags),
            "full_key": self.full_key,
            "hash_value": self._hash_value,
            "created_at": self._created_at.isoformat()
        }

    @classmethod
    def from_string(cls, key_string: str, namespace: str = "") -> 'Key':
        """
        Create a Key instance from a string.

        Args:
            key_string: The key string
            namespace: Optional namespace

        Returns:
            Key instance
        """
        return cls(value=key_string, namespace=namespace)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Key':
        """
        Create a Key instance from a dictionary.

        Args:
            data: Dictionary containing key data

        Returns:
            Key instance
        """
        tags = set(data.get("tags", []))
        return cls(
            value=data["value"],
            namespace=data.get("namespace", ""),
            tags=tags if tags else None
        )

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of the full key."""
        hash_input = self.full_key.encode('utf-8')
        return hashlib.sha256(hash_input).hexdigest()

    @staticmethod
    def _validate_value(value: str) -> None:
        """Validate key value."""
        if not value:
            raise ValueError("Key value cannot be empty")

        if not isinstance(value, str):
            raise ValueError("Key value must be a string")

        if not Key._KEY_PATTERN.match(value):
            raise ValueError("Key value contains invalid characters")

        if len(value) > 1024:  # Reasonable key length limit
            raise ValueError("Key value too long (max 1024 characters)")

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        """Validate namespace."""
        if not isinstance(namespace, str):
            raise ValueError("Namespace must be a string")

        if not Key._NAMESPACE_PATTERN.match(namespace):
            raise ValueError("Namespace contains invalid characters")

        if len(namespace) > 256:  # Reasonable namespace length limit
            raise ValueError("Namespace too long (max 256 characters)")

    @staticmethod
    def _validate_tag(tag: str) -> None:
        """Validate a single tag."""
        if not isinstance(tag, str):
            raise ValueError("Tag must be a string")

        if not tag:
            raise ValueError("Tag cannot be empty")

        if not Key._TAG_PATTERN.match(tag):
            raise ValueError("Tag contains invalid characters")

        if len(tag) > 64:  # Reasonable tag length limit
            raise ValueError("Tag too long (max 64 characters)")

    @staticmethod
    def _validate_tags(tags: Set[str]) -> None:
        """Validate a set of tags."""
        if not isinstance(tags, set):
            raise ValueError("Tags must be a set")

        if len(tags) > 100:  # Reasonable tag count limit
            raise ValueError("Too many tags (max 100)")

        for tag in tags:
            Key._validate_tag(tag)

    def __str__(self) -> str:
        return self.full_key

    def __repr__(self) -> str:
        return f"<Key(value='{self._value}', namespace='{self._namespace}', tags={self._tags})>"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Key):
            return False
        return (
            self._value == other._value and
            self._namespace == other._namespace and
            self._tags == other._tags
        )

    def __hash__(self) -> int:
        return hash((self._value, self._namespace, tuple(sorted(self._tags))))

    def __lt__(self, other: 'Key') -> bool:
        """Compare keys for sorting."""
        if not isinstance(other, Key):
            return NotImplemented
        return self.full_key < other.full_key