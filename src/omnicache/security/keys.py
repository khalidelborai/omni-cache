"""
Encryption key model with versioning and rotation.

This module defines encryption key management with automatic rotation,
versioning, and secure storage for enterprise cache deployments.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import time
import secrets
import hashlib
import json
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os


class KeyType(Enum):
    """Types of encryption keys."""
    AES_256 = "aes-256"
    AES_192 = "aes-192"
    AES_128 = "aes-128"
    CHACHA20 = "chacha20"
    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"
    ECDSA_P256 = "ecdsa-p256"
    ECDSA_P384 = "ecdsa-p384"


class KeyStatus(Enum):
    """Status of encryption keys."""
    ACTIVE = "active"
    RETIRED = "retired"
    COMPROMISED = "compromised"
    PENDING = "pending"
    REVOKED = "revoked"


class RotationTrigger(Enum):
    """Triggers for key rotation."""
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    SIZE_BASED = "size_based"
    MANUAL = "manual"
    SECURITY_EVENT = "security_event"


@dataclass
class KeyMetadata:
    """Metadata for encryption keys."""
    created_by: str = ""
    purpose: str = "cache_encryption"
    environment: str = "production"
    compliance_tags: List[str] = field(default_factory=list)
    custom_attributes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "created_by": self.created_by,
            "purpose": self.purpose,
            "environment": self.environment,
            "compliance_tags": self.compliance_tags,
            "custom_attributes": self.custom_attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KeyMetadata':
        """Create from dictionary."""
        return cls(
            created_by=data.get("created_by", ""),
            purpose=data.get("purpose", "cache_encryption"),
            environment=data.get("environment", "production"),
            compliance_tags=data.get("compliance_tags", []),
            custom_attributes=data.get("custom_attributes", {}),
        )


@dataclass
class RotationPolicy:
    """Key rotation policy configuration."""
    max_age_seconds: int = 86400 * 30  # 30 days
    max_usage_count: int = 1000000     # 1 million operations
    max_data_size_bytes: int = 1024 * 1024 * 1024 * 10  # 10 GB
    triggers: List[RotationTrigger] = field(default_factory=lambda: [RotationTrigger.TIME_BASED])
    auto_rotate: bool = True
    retention_count: int = 3  # Keep 3 old keys
    notification_threshold: float = 0.8  # Notify when 80% of limits reached

    def __post_init__(self):
        """Validate rotation policy."""
        if self.max_age_seconds <= 0:
            raise ValueError("Max age must be positive")
        if self.max_usage_count <= 0:
            raise ValueError("Max usage count must be positive")
        if self.max_data_size_bytes <= 0:
            raise ValueError("Max data size must be positive")
        if not 0.0 <= self.notification_threshold <= 1.0:
            raise ValueError("Notification threshold must be between 0.0 and 1.0")

    def should_rotate(self, key: 'EncryptionKey') -> bool:
        """Check if key should be rotated based on policy."""
        current_time = time.time()

        # Check time-based rotation
        if RotationTrigger.TIME_BASED in self.triggers:
            age = current_time - key.created_at
            if age >= self.max_age_seconds:
                return True

        # Check usage-based rotation
        if RotationTrigger.USAGE_BASED in self.triggers:
            if key.usage_count >= self.max_usage_count:
                return True

        # Check size-based rotation
        if RotationTrigger.SIZE_BASED in self.triggers:
            if key.data_encrypted_bytes >= self.max_data_size_bytes:
                return True

        return False

    def needs_notification(self, key: 'EncryptionKey') -> bool:
        """Check if rotation notification should be sent."""
        current_time = time.time()

        # Check age threshold
        age_ratio = (current_time - key.created_at) / self.max_age_seconds
        if age_ratio >= self.notification_threshold:
            return True

        # Check usage threshold
        usage_ratio = key.usage_count / self.max_usage_count
        if usage_ratio >= self.notification_threshold:
            return True

        # Check size threshold
        size_ratio = key.data_encrypted_bytes / self.max_data_size_bytes
        if size_ratio >= self.notification_threshold:
            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_age_seconds": self.max_age_seconds,
            "max_usage_count": self.max_usage_count,
            "max_data_size_bytes": self.max_data_size_bytes,
            "triggers": [trigger.value for trigger in self.triggers],
            "auto_rotate": self.auto_rotate,
            "retention_count": self.retention_count,
            "notification_threshold": self.notification_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RotationPolicy':
        """Create from dictionary."""
        return cls(
            max_age_seconds=data.get("max_age_seconds", 86400 * 30),
            max_usage_count=data.get("max_usage_count", 1000000),
            max_data_size_bytes=data.get("max_data_size_bytes", 1024 * 1024 * 1024 * 10),
            triggers=[RotationTrigger(t) for t in data.get("triggers", ["time_based"])],
            auto_rotate=data.get("auto_rotate", True),
            retention_count=data.get("retention_count", 3),
            notification_threshold=data.get("notification_threshold", 0.8),
        )


@dataclass
class EncryptionKey:
    """
    Encryption key model with versioning and rotation.

    Manages encryption keys with automatic rotation, usage tracking,
    and secure storage for enterprise cache deployments.
    """

    key_id: str
    key_type: KeyType
    version: int = 1
    status: KeyStatus = KeyStatus.PENDING

    # Key material (stored securely)
    _key_material: Optional[bytes] = None
    _derived_keys: Dict[str, bytes] = field(default_factory=dict)

    # Usage tracking
    usage_count: int = 0
    data_encrypted_bytes: int = 0
    last_used_at: Optional[float] = None

    # Lifecycle
    created_at: float = field(default_factory=time.time)
    activated_at: Optional[float] = None
    retired_at: Optional[float] = None
    expires_at: Optional[float] = None

    # Metadata
    metadata: KeyMetadata = field(default_factory=KeyMetadata)
    rotation_policy: RotationPolicy = field(default_factory=RotationPolicy)

    # Security
    salt: Optional[bytes] = None
    checksum: Optional[str] = None

    def __post_init__(self):
        """Post-initialization setup."""
        if not self.key_id:
            raise ValueError("Key ID is required")

        if self.version <= 0:
            raise ValueError("Version must be positive")

        # Generate key material if not provided
        if self._key_material is None:
            self._generate_key_material()

        # Generate salt if not provided
        if self.salt is None:
            self.salt = secrets.token_bytes(32)

        # Calculate checksum
        self._update_checksum()

    def _generate_key_material(self) -> None:
        """Generate secure key material."""
        if self.key_type == KeyType.AES_256:
            self._key_material = secrets.token_bytes(32)
        elif self.key_type == KeyType.AES_192:
            self._key_material = secrets.token_bytes(24)
        elif self.key_type == KeyType.AES_128:
            self._key_material = secrets.token_bytes(16)
        elif self.key_type == KeyType.CHACHA20:
            self._key_material = secrets.token_bytes(32)
        else:
            # Default to AES-256
            self._key_material = secrets.token_bytes(32)

    def _update_checksum(self) -> None:
        """Update key checksum for integrity verification."""
        if self._key_material:
            hasher = hashlib.sha256()
            hasher.update(self._key_material)
            hasher.update(self.salt or b'')
            hasher.update(self.key_id.encode())
            self.checksum = hasher.hexdigest()

    @property
    def is_active(self) -> bool:
        """Check if key is active."""
        return self.status == KeyStatus.ACTIVE

    @property
    def is_retired(self) -> bool:
        """Check if key is retired."""
        return self.status == KeyStatus.RETIRED

    @property
    def is_compromised(self) -> bool:
        """Check if key is compromised."""
        return self.status == KeyStatus.COMPROMISED

    @property
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get key age in seconds."""
        return time.time() - self.created_at

    @property
    def size_bytes(self) -> int:
        """Get key size in bytes."""
        return len(self._key_material) if self._key_material else 0

    def activate(self) -> None:
        """Activate the key."""
        if self.status != KeyStatus.PENDING:
            raise ValueError(f"Cannot activate key in {self.status.value} status")

        self.status = KeyStatus.ACTIVE
        self.activated_at = time.time()

    def retire(self) -> None:
        """Retire the key."""
        if self.status not in [KeyStatus.ACTIVE, KeyStatus.PENDING]:
            raise ValueError(f"Cannot retire key in {self.status.value} status")

        self.status = KeyStatus.RETIRED
        self.retired_at = time.time()

    def revoke(self) -> None:
        """Revoke the key."""
        self.status = KeyStatus.REVOKED
        self.retired_at = time.time()

    def mark_compromised(self) -> None:
        """Mark key as compromised."""
        self.status = KeyStatus.COMPROMISED
        self.retired_at = time.time()

    def get_key_material(self) -> bytes:
        """Get the raw key material."""
        if not self._key_material:
            raise ValueError("Key material not available")

        if not self.is_active:
            raise ValueError("Key is not active")

        return self._key_material

    def derive_key(self, purpose: str, length: int = 32) -> bytes:
        """
        Derive a key for specific purpose.

        Args:
            purpose: Purpose of the derived key
            length: Length of derived key in bytes

        Returns:
            Derived key bytes
        """
        if not self._key_material:
            raise ValueError("Key material not available")

        cache_key = f"{purpose}:{length}"
        if cache_key in self._derived_keys:
            return self._derived_keys[cache_key]

        # Use PBKDF2 for key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=length,
            salt=self.salt + purpose.encode(),
            iterations=100000,
            backend=default_backend()
        )

        derived_key = kdf.derive(self._key_material)
        self._derived_keys[cache_key] = derived_key

        return derived_key

    def encrypt_data(self, plaintext: bytes) -> Dict[str, Any]:
        """
        Encrypt data with this key.

        Args:
            plaintext: Data to encrypt

        Returns:
            Dictionary with encrypted data and metadata
        """
        if not self.is_active:
            raise ValueError("Key is not active")

        # Generate IV
        iv = secrets.token_bytes(12)  # 96 bits for GCM

        # Get encryption key
        encryption_key = self.get_key_material()

        # Encrypt using AES-GCM
        cipher = Cipher(
            algorithms.AES(encryption_key),
            modes.GCM(iv),
            backend=default_backend()
        )

        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        # Update usage statistics
        self.usage_count += 1
        self.data_encrypted_bytes += len(plaintext)
        self.last_used_at = time.time()

        return {
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "iv": base64.b64encode(iv).decode(),
            "tag": base64.b64encode(encryptor.tag).decode(),
            "key_id": self.key_id,
            "key_version": self.version,
            "algorithm": self.key_type.value,
        }

    def decrypt_data(self, encrypted_data: Dict[str, Any]) -> bytes:
        """
        Decrypt data with this key.

        Args:
            encrypted_data: Dictionary with encrypted data and metadata

        Returns:
            Decrypted plaintext
        """
        if encrypted_data.get("key_id") != self.key_id:
            raise ValueError("Key ID mismatch")

        if encrypted_data.get("key_version") != self.version:
            raise ValueError("Key version mismatch")

        # Extract components
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        iv = base64.b64decode(encrypted_data["iv"])
        tag = base64.b64decode(encrypted_data["tag"])

        # Get decryption key
        decryption_key = self.get_key_material()

        # Decrypt using AES-GCM
        cipher = Cipher(
            algorithms.AES(decryption_key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )

        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        # Update usage statistics
        self.usage_count += 1
        self.last_used_at = time.time()

        return plaintext

    def verify_integrity(self) -> bool:
        """Verify key integrity using checksum."""
        if not self.checksum:
            return False

        expected_checksum = self.checksum
        self._update_checksum()
        actual_checksum = self.checksum

        # Restore original checksum
        self.checksum = expected_checksum

        return expected_checksum == actual_checksum

    def should_rotate(self) -> bool:
        """Check if key should be rotated."""
        if not self.rotation_policy:
            return False

        return self.rotation_policy.should_rotate(self)

    def needs_notification(self) -> bool:
        """Check if rotation notification is needed."""
        if not self.rotation_policy:
            return False

        return self.rotation_policy.needs_notification(self)

    def export_public_info(self) -> Dict[str, Any]:
        """Export non-sensitive key information."""
        return {
            "key_id": self.key_id,
            "key_type": self.key_type.value,
            "version": self.version,
            "status": self.status.value,
            "usage_count": self.usage_count,
            "data_encrypted_bytes": self.data_encrypted_bytes,
            "age_seconds": self.age_seconds,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "activated_at": self.activated_at,
            "retired_at": self.retired_at,
            "expires_at": self.expires_at,
            "last_used_at": self.last_used_at,
            "metadata": self.metadata.to_dict(),
            "is_active": self.is_active,
            "is_expired": self.is_expired,
            "should_rotate": self.should_rotate(),
            "needs_notification": self.needs_notification(),
        }

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert key to dictionary representation."""
        data = self.export_public_info()

        data.update({
            "rotation_policy": self.rotation_policy.to_dict(),
            "checksum": self.checksum,
        })

        if include_sensitive:
            data.update({
                "key_material": base64.b64encode(self._key_material).decode() if self._key_material else None,
                "salt": base64.b64encode(self.salt).decode() if self.salt else None,
                "derived_keys": {
                    k: base64.b64encode(v).decode()
                    for k, v in self._derived_keys.items()
                }
            })

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], include_sensitive: bool = False) -> 'EncryptionKey':
        """Create key from dictionary representation."""
        metadata = KeyMetadata.from_dict(data.get("metadata", {}))
        rotation_policy = RotationPolicy.from_dict(data.get("rotation_policy", {}))

        key = cls(
            key_id=data["key_id"],
            key_type=KeyType(data["key_type"]),
            version=data.get("version", 1),
            status=KeyStatus(data.get("status", "pending")),
            usage_count=data.get("usage_count", 0),
            data_encrypted_bytes=data.get("data_encrypted_bytes", 0),
            last_used_at=data.get("last_used_at"),
            created_at=data.get("created_at", time.time()),
            activated_at=data.get("activated_at"),
            retired_at=data.get("retired_at"),
            expires_at=data.get("expires_at"),
            metadata=metadata,
            rotation_policy=rotation_policy,
            checksum=data.get("checksum"),
        )

        if include_sensitive and data.get("key_material"):
            key._key_material = base64.b64decode(data["key_material"])

        if include_sensitive and data.get("salt"):
            key.salt = base64.b64decode(data["salt"])

        if include_sensitive and data.get("derived_keys"):
            key._derived_keys = {
                k: base64.b64decode(v)
                for k, v in data["derived_keys"].items()
            }

        return key

    def to_json(self, include_sensitive: bool = False) -> str:
        """Convert key to JSON string."""
        return json.dumps(self.to_dict(include_sensitive), default=str, indent=2)

    @classmethod
    def from_json(cls, json_str: str, include_sensitive: bool = False) -> 'EncryptionKey':
        """Create key from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data, include_sensitive)

    def __str__(self) -> str:
        """String representation of the key."""
        return f"EncryptionKey({self.key_id}, v{self.version}, {self.status.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"EncryptionKey(key_id='{self.key_id}', type='{self.key_type.value}', "
                f"version={self.version}, status='{self.status.value}')")

    def __eq__(self, other) -> bool:
        """Check equality based on key ID and version."""
        if not isinstance(other, EncryptionKey):
            return False
        return self.key_id == other.key_id and self.version == other.version

    def __hash__(self) -> int:
        """Hash based on key ID and version."""
        return hash((self.key_id, self.version))


@dataclass
class KeyManager:
    """Manager for encryption keys with automatic rotation."""

    name: str = "default"
    _keys: Dict[str, List[EncryptionKey]] = field(default_factory=dict)
    _active_keys: Dict[str, EncryptionKey] = field(default_factory=dict)

    default_key_type: KeyType = KeyType.AES_256
    default_rotation_policy: RotationPolicy = field(default_factory=RotationPolicy)

    def add_key(self, key: EncryptionKey) -> None:
        """Add a key to the manager."""
        if key.key_id not in self._keys:
            self._keys[key.key_id] = []

        self._keys[key.key_id].append(key)

        if key.is_active:
            self._active_keys[key.key_id] = key

    def get_active_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get the active key for a key ID."""
        return self._active_keys.get(key_id)

    def get_key(self, key_id: str, version: int) -> Optional[EncryptionKey]:
        """Get a specific key version."""
        for key in self._keys.get(key_id, []):
            if key.version == version:
                return key
        return None

    def create_key(self, key_id: str, key_type: Optional[KeyType] = None,
                  metadata: Optional[KeyMetadata] = None) -> EncryptionKey:
        """Create a new encryption key."""
        if key_type is None:
            key_type = self.default_key_type

        if metadata is None:
            metadata = KeyMetadata()

        # Determine version
        existing_keys = self._keys.get(key_id, [])
        version = max([k.version for k in existing_keys], default=0) + 1

        # Create key
        key = EncryptionKey(
            key_id=key_id,
            key_type=key_type,
            version=version,
            metadata=metadata,
            rotation_policy=self.default_rotation_policy,
        )

        # Retire old active key
        if key_id in self._active_keys:
            old_key = self._active_keys[key_id]
            old_key.retire()

        # Activate new key
        key.activate()
        self.add_key(key)

        return key

    def rotate_key(self, key_id: str) -> EncryptionKey:
        """Rotate a key (create new version)."""
        active_key = self.get_active_key(key_id)
        if not active_key:
            raise ValueError(f"No active key found for {key_id}")

        return self.create_key(
            key_id=key_id,
            key_type=active_key.key_type,
            metadata=active_key.metadata
        )

    def get_keys_needing_rotation(self) -> List[EncryptionKey]:
        """Get keys that need rotation."""
        keys_to_rotate = []
        for key in self._active_keys.values():
            if key.should_rotate():
                keys_to_rotate.append(key)
        return keys_to_rotate

    def auto_rotate_keys(self) -> List[EncryptionKey]:
        """Automatically rotate keys that need rotation."""
        rotated_keys = []
        for key in self.get_keys_needing_rotation():
            if key.rotation_policy.auto_rotate:
                new_key = self.rotate_key(key.key_id)
                rotated_keys.append(new_key)
        return rotated_keys

    def cleanup_old_keys(self) -> int:
        """Clean up old retired keys based on retention policy."""
        cleaned_count = 0

        for key_id, keys in self._keys.items():
            # Sort by version (newest first)
            sorted_keys = sorted(keys, key=lambda k: k.version, reverse=True)

            # Find active key to get retention policy
            active_key = self._active_keys.get(key_id)
            retention_count = active_key.rotation_policy.retention_count if active_key else 3

            # Keep active key + retention count
            keys_to_keep = []
            retired_count = 0

            for key in sorted_keys:
                if key.is_active or retired_count < retention_count:
                    keys_to_keep.append(key)
                    if key.is_retired:
                        retired_count += 1
                else:
                    cleaned_count += 1

            self._keys[key_id] = keys_to_keep

        return cleaned_count

    def export_summary(self) -> Dict[str, Any]:
        """Export key manager summary."""
        total_keys = sum(len(keys) for keys in self._keys.values())
        active_keys = len(self._active_keys)
        retired_keys = sum(1 for keys in self._keys.values() for key in keys if key.is_retired)

        return {
            "name": self.name,
            "total_keys": total_keys,
            "active_keys": active_keys,
            "retired_keys": retired_keys,
            "key_ids": list(self._keys.keys()),
            "keys_needing_rotation": len(self.get_keys_needing_rotation()),
        }