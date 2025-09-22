"""
Security Policy model for OmniCache enterprise features.

This module defines security policies with encryption settings, PII detection rules,
and GDPR compliance configurations for enterprise cache deployments.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set
from enum import Enum
import time
import re
import json


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    AES_128_GCM = "aes-128-gcm"


class KeyRotationStrategy(Enum):
    """Key rotation strategies."""
    TIME_BASED = "time_based"
    ACCESS_BASED = "access_based"
    SIZE_BASED = "size_based"
    MANUAL = "manual"


class PIIDetectionMode(Enum):
    """PII detection modes."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"
    DISABLED = "disabled"


class AccessControlLevel(Enum):
    """Access control levels."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class ComplianceFramework(Enum):
    """Compliance frameworks."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOX = "sox"
    CCPA = "ccpa"


@dataclass
class EncryptionConfig:
    """Encryption configuration."""
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    key_size: int = 256
    iv_size: int = 12
    tag_size: int = 16
    key_rotation_interval: int = 86400  # 24 hours in seconds
    key_rotation_strategy: KeyRotationStrategy = KeyRotationStrategy.TIME_BASED
    encrypt_keys: bool = False
    encrypt_values: bool = True
    encrypt_metadata: bool = False

    def __post_init__(self):
        """Validate encryption configuration."""
        if self.key_size not in [128, 192, 256]:
            raise ValueError("Key size must be 128, 192, or 256 bits")
        if self.key_rotation_interval <= 0:
            raise ValueError("Key rotation interval must be positive")


@dataclass
class PIIRule:
    """PII detection rule."""
    name: str
    pattern: str
    field_types: List[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high, critical
    enabled: bool = True

    def __post_init__(self):
        """Validate PII rule."""
        try:
            re.compile(self.pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

        if self.severity not in ["low", "medium", "high", "critical"]:
            raise ValueError("Severity must be low, medium, high, or critical")


@dataclass
class PIIDetectionConfig:
    """PII detection configuration."""
    mode: PIIDetectionMode = PIIDetectionMode.MODERATE
    rules: List[PIIRule] = field(default_factory=list)
    scan_keys: bool = True
    scan_values: bool = True
    scan_metadata: bool = False
    custom_patterns: Dict[str, str] = field(default_factory=dict)
    whitelist_patterns: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize with default PII rules if none provided."""
        if not self.rules:
            self.rules = self._get_default_rules()

    def _get_default_rules(self) -> List[PIIRule]:
        """Get default PII detection rules."""
        return [
            PIIRule(
                name="social_security_number",
                pattern=r"\b\d{3}-\d{2}-\d{4}\b",
                field_types=["ssn", "social_security"],
                severity="critical"
            ),
            PIIRule(
                name="credit_card",
                pattern=r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
                field_types=["credit_card", "card_number"],
                severity="critical"
            ),
            PIIRule(
                name="email_address",
                pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                field_types=["email", "email_address"],
                severity="high"
            ),
            PIIRule(
                name="phone_number",
                pattern=r"\b\d{3}[-.]\d{3}[-.]\d{4}\b",
                field_types=["phone", "phone_number"],
                severity="medium"
            ),
            PIIRule(
                name="ip_address",
                pattern=r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                field_types=["ip", "ip_address"],
                severity="low"
            ),
        ]


@dataclass
class ComplianceConfig:
    """Compliance configuration."""
    frameworks: Set[ComplianceFramework] = field(default_factory=set)
    data_retention_days: int = 365
    audit_logging: bool = True
    consent_required: bool = False
    right_to_erasure: bool = False
    data_portability: bool = False
    pseudonymization: bool = False
    anonymization: bool = False

    def __post_init__(self):
        """Apply framework-specific defaults."""
        if ComplianceFramework.GDPR in self.frameworks:
            self.right_to_erasure = True
            self.data_portability = True
            self.consent_required = True
            self.audit_logging = True

        if ComplianceFramework.HIPAA in self.frameworks:
            self.audit_logging = True
            self.pseudonymization = True


@dataclass
class AccessControl:
    """Access control configuration."""
    require_authentication: bool = True
    allowed_users: Set[str] = field(default_factory=set)
    allowed_roles: Set[str] = field(default_factory=set)
    allowed_ips: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)
    rate_limit_per_user: int = 1000  # requests per hour
    session_timeout: int = 3600  # seconds

    def is_user_allowed(self, user_id: str, role: str = None, ip: str = None) -> bool:
        """Check if user is allowed access."""
        if not self.require_authentication:
            return True

        # Check blocked IPs
        if ip and ip in self.blocked_ips:
            return False

        # Check allowed IPs
        if self.allowed_ips and ip and ip not in self.allowed_ips:
            return False

        # Check user
        if self.allowed_users and user_id not in self.allowed_users:
            return False

        # Check role
        if self.allowed_roles and role and role not in self.allowed_roles:
            return False

        return True


@dataclass
class SecurityPolicy:
    """
    Security policy model for OmniCache enterprise features.

    Defines comprehensive security policies including encryption settings,
    PII detection rules, GDPR compliance, and access controls.
    """

    name: str
    description: str = ""
    enabled: bool = True

    # Encryption configuration
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)

    # PII detection configuration
    pii_detection: PIIDetectionConfig = field(default_factory=PIIDetectionConfig)

    # Compliance configuration
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)

    # Access control
    access_control: AccessControl = field(default_factory=AccessControl)

    # Audit settings
    audit_all_operations: bool = False
    audit_failed_operations: bool = True
    audit_pii_access: bool = True
    audit_retention_days: int = 90

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    created_by: str = ""
    version: int = 1
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation."""
        if not self.name:
            raise ValueError("Security policy name is required")

        if self.audit_retention_days <= 0:
            raise ValueError("Audit retention days must be positive")

    @property
    def requires_encryption(self) -> bool:
        """Check if policy requires encryption."""
        return (self.encryption.encrypt_keys or
                self.encryption.encrypt_values or
                self.encryption.encrypt_metadata)

    @property
    def has_pii_detection(self) -> bool:
        """Check if PII detection is enabled."""
        return self.pii_detection.mode != PIIDetectionMode.DISABLED

    @property
    def is_gdpr_compliant(self) -> bool:
        """Check if policy is GDPR compliant."""
        return ComplianceFramework.GDPR in self.compliance.frameworks

    def validate_data(self, key: str, value: Any) -> Dict[str, Any]:
        """
        Validate data against PII detection rules.

        Args:
            key: Cache key
            value: Cache value

        Returns:
            Dictionary with validation results
        """
        violations = []

        if not self.has_pii_detection:
            return {"violations": violations, "safe": True}

        # Convert value to string for pattern matching
        value_str = str(value) if value is not None else ""

        # Check key patterns
        if self.pii_detection.scan_keys:
            violations.extend(self._check_patterns(key, "key"))

        # Check value patterns
        if self.pii_detection.scan_values:
            violations.extend(self._check_patterns(value_str, "value"))

        return {
            "violations": violations,
            "safe": len(violations) == 0,
            "severity": max([v["severity"] for v in violations], default="none")
        }

    def _check_patterns(self, text: str, field_type: str) -> List[Dict[str, Any]]:
        """Check text against PII patterns."""
        violations = []

        for rule in self.pii_detection.rules:
            if not rule.enabled:
                continue

            # Check whitelist patterns first
            if any(re.search(pattern, text, re.IGNORECASE)
                   for pattern in self.pii_detection.whitelist_patterns):
                continue

            if re.search(rule.pattern, text, re.IGNORECASE):
                violations.append({
                    "rule": rule.name,
                    "pattern": rule.pattern,
                    "field_type": field_type,
                    "severity": rule.severity,
                    "match_found": True
                })

        return violations

    def should_encrypt_key(self) -> bool:
        """Check if keys should be encrypted."""
        return self.encryption.encrypt_keys

    def should_encrypt_value(self) -> bool:
        """Check if values should be encrypted."""
        return self.encryption.encrypt_values

    def should_encrypt_metadata(self) -> bool:
        """Check if metadata should be encrypted."""
        return self.encryption.encrypt_metadata

    def should_audit_operation(self, operation: str, failed: bool = False, has_pii: bool = False) -> bool:
        """Determine if operation should be audited."""
        if self.audit_all_operations:
            return True

        if failed and self.audit_failed_operations:
            return True

        if has_pii and self.audit_pii_access:
            return True

        return False

    def add_pii_rule(self, rule: PIIRule) -> None:
        """Add a new PII detection rule."""
        self.pii_detection.rules.append(rule)
        self.updated_at = time.time()
        self.version += 1

    def remove_pii_rule(self, rule_name: str) -> bool:
        """Remove a PII detection rule."""
        original_count = len(self.pii_detection.rules)
        self.pii_detection.rules = [r for r in self.pii_detection.rules if r.name != rule_name]

        if len(self.pii_detection.rules) < original_count:
            self.updated_at = time.time()
            self.version += 1
            return True
        return False

    def add_compliance_framework(self, framework: ComplianceFramework) -> None:
        """Add a compliance framework."""
        self.compliance.frameworks.add(framework)
        self.updated_at = time.time()
        self.version += 1

    def remove_compliance_framework(self, framework: ComplianceFramework) -> None:
        """Remove a compliance framework."""
        self.compliance.frameworks.discard(framework)
        self.updated_at = time.time()
        self.version += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert security policy to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "encryption": {
                "algorithm": self.encryption.algorithm.value,
                "key_size": self.encryption.key_size,
                "iv_size": self.encryption.iv_size,
                "tag_size": self.encryption.tag_size,
                "key_rotation_interval": self.encryption.key_rotation_interval,
                "key_rotation_strategy": self.encryption.key_rotation_strategy.value,
                "encrypt_keys": self.encryption.encrypt_keys,
                "encrypt_values": self.encryption.encrypt_values,
                "encrypt_metadata": self.encryption.encrypt_metadata,
            },
            "pii_detection": {
                "mode": self.pii_detection.mode.value,
                "rules": [
                    {
                        "name": rule.name,
                        "pattern": rule.pattern,
                        "field_types": rule.field_types,
                        "severity": rule.severity,
                        "enabled": rule.enabled,
                    }
                    for rule in self.pii_detection.rules
                ],
                "scan_keys": self.pii_detection.scan_keys,
                "scan_values": self.pii_detection.scan_values,
                "scan_metadata": self.pii_detection.scan_metadata,
                "custom_patterns": self.pii_detection.custom_patterns,
                "whitelist_patterns": self.pii_detection.whitelist_patterns,
            },
            "compliance": {
                "frameworks": [f.value for f in self.compliance.frameworks],
                "data_retention_days": self.compliance.data_retention_days,
                "audit_logging": self.compliance.audit_logging,
                "consent_required": self.compliance.consent_required,
                "right_to_erasure": self.compliance.right_to_erasure,
                "data_portability": self.compliance.data_portability,
                "pseudonymization": self.compliance.pseudonymization,
                "anonymization": self.compliance.anonymization,
            },
            "access_control": {
                "require_authentication": self.access_control.require_authentication,
                "allowed_users": list(self.access_control.allowed_users),
                "allowed_roles": list(self.access_control.allowed_roles),
                "allowed_ips": self.access_control.allowed_ips,
                "blocked_ips": self.access_control.blocked_ips,
                "rate_limit_per_user": self.access_control.rate_limit_per_user,
                "session_timeout": self.access_control.session_timeout,
            },
            "audit_all_operations": self.audit_all_operations,
            "audit_failed_operations": self.audit_failed_operations,
            "audit_pii_access": self.audit_pii_access,
            "audit_retention_days": self.audit_retention_days,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "version": self.version,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityPolicy':
        """Create security policy from dictionary representation."""
        # Parse encryption config
        enc_data = data.get("encryption", {})
        encryption = EncryptionConfig(
            algorithm=EncryptionAlgorithm(enc_data.get("algorithm", "aes-256-gcm")),
            key_size=enc_data.get("key_size", 256),
            iv_size=enc_data.get("iv_size", 12),
            tag_size=enc_data.get("tag_size", 16),
            key_rotation_interval=enc_data.get("key_rotation_interval", 86400),
            key_rotation_strategy=KeyRotationStrategy(enc_data.get("key_rotation_strategy", "time_based")),
            encrypt_keys=enc_data.get("encrypt_keys", False),
            encrypt_values=enc_data.get("encrypt_values", True),
            encrypt_metadata=enc_data.get("encrypt_metadata", False),
        )

        # Parse PII detection config
        pii_data = data.get("pii_detection", {})
        pii_rules = [
            PIIRule(
                name=rule["name"],
                pattern=rule["pattern"],
                field_types=rule.get("field_types", []),
                severity=rule.get("severity", "medium"),
                enabled=rule.get("enabled", True),
            )
            for rule in pii_data.get("rules", [])
        ]

        pii_detection = PIIDetectionConfig(
            mode=PIIDetectionMode(pii_data.get("mode", "moderate")),
            rules=pii_rules,
            scan_keys=pii_data.get("scan_keys", True),
            scan_values=pii_data.get("scan_values", True),
            scan_metadata=pii_data.get("scan_metadata", False),
            custom_patterns=pii_data.get("custom_patterns", {}),
            whitelist_patterns=pii_data.get("whitelist_patterns", []),
        )

        # Parse compliance config
        comp_data = data.get("compliance", {})
        compliance = ComplianceConfig(
            frameworks=set(ComplianceFramework(f) for f in comp_data.get("frameworks", [])),
            data_retention_days=comp_data.get("data_retention_days", 365),
            audit_logging=comp_data.get("audit_logging", True),
            consent_required=comp_data.get("consent_required", False),
            right_to_erasure=comp_data.get("right_to_erasure", False),
            data_portability=comp_data.get("data_portability", False),
            pseudonymization=comp_data.get("pseudonymization", False),
            anonymization=comp_data.get("anonymization", False),
        )

        # Parse access control config
        ac_data = data.get("access_control", {})
        access_control = AccessControl(
            require_authentication=ac_data.get("require_authentication", True),
            allowed_users=set(ac_data.get("allowed_users", [])),
            allowed_roles=set(ac_data.get("allowed_roles", [])),
            allowed_ips=ac_data.get("allowed_ips", []),
            blocked_ips=ac_data.get("blocked_ips", []),
            rate_limit_per_user=ac_data.get("rate_limit_per_user", 1000),
            session_timeout=ac_data.get("session_timeout", 3600),
        )

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            encryption=encryption,
            pii_detection=pii_detection,
            compliance=compliance,
            access_control=access_control,
            audit_all_operations=data.get("audit_all_operations", False),
            audit_failed_operations=data.get("audit_failed_operations", True),
            audit_pii_access=data.get("audit_pii_access", True),
            audit_retention_days=data.get("audit_retention_days", 90),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            created_by=data.get("created_by", ""),
            version=data.get("version", 1),
            tags=data.get("tags", {}),
        )

    def to_json(self) -> str:
        """Convert security policy to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'SecurityPolicy':
        """Create security policy from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of the security policy."""
        status = "enabled" if self.enabled else "disabled"
        frameworks = ", ".join(f.value for f in self.compliance.frameworks)
        return f"SecurityPolicy({self.name}, {status}, frameworks: {frameworks})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"SecurityPolicy(name='{self.name}', enabled={self.enabled}, "
                f"version={self.version})")

    def __eq__(self, other) -> bool:
        """Check equality based on name and version."""
        if not isinstance(other, SecurityPolicy):
            return False
        return self.name == other.name and self.version == other.version

    def __hash__(self) -> int:
        """Hash based on name."""
        return hash(self.name)