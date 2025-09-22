"""
Integration test for security and GDPR compliance features.

This test validates encryption, PII detection, data anonymization,
audit logging, and compliance with privacy regulations.
"""

import pytest
import asyncio
import time
import hashlib
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from cryptography.fernet import Fernet

from omnicache.core.cache import Cache
from omnicache.strategies.lru import LRUStrategy
from omnicache.backends.memory import MemoryBackend
from omnicache.security.encryption import EncryptionManager, EncryptionConfig
from omnicache.security.pii_detector import PIIDetector, PIICategory
from omnicache.security.anonymizer import DataAnonymizer, AnonymizationStrategy
from omnicache.security.audit_logger import AuditLogger, AuditEvent
from omnicache.security.compliance import GDPRCompliance, ComplianceConfig
from omnicache.security.access_control import AccessController, Permission


@pytest.mark.integration
class TestSecurityCompliance:
    """Integration tests for security and compliance features."""

    @pytest.fixture
    async def encryption_config(self):
        """Create encryption configuration."""
        return EncryptionConfig(
            algorithm="AES-256-GCM",
            key_rotation_interval=3600,  # 1 hour
            at_rest_encryption=True,
            in_transit_encryption=True,
            key_derivation="PBKDF2"
        )

    @pytest.fixture
    async def encryption_manager(self, encryption_config):
        """Create encryption manager for testing."""
        manager = EncryptionManager(config=encryption_config)
        await manager.initialize()
        return manager

    @pytest.fixture
    async def pii_detector(self):
        """Create PII detector with enterprise patterns."""
        detector = PIIDetector()
        await detector.load_patterns([
            PIICategory.SSN,
            PIICategory.CREDIT_CARD,
            PIICategory.EMAIL,
            PIICategory.PHONE,
            PIICategory.IP_ADDRESS,
            PIICategory.PASSPORT,
            PIICategory.MEDICAL_ID
        ])
        return detector

    @pytest.fixture
    async def data_anonymizer(self):
        """Create data anonymizer with multiple strategies."""
        return DataAnonymizer(
            strategies={
                "mask": AnonymizationStrategy.MASKING,
                "hash": AnonymizationStrategy.HASHING,
                "tokenize": AnonymizationStrategy.TOKENIZATION,
                "generalize": AnonymizationStrategy.GENERALIZATION,
                "suppress": AnonymizationStrategy.SUPPRESSION
            }
        )

    @pytest.fixture
    async def audit_logger(self):
        """Create audit logger for compliance tracking."""
        return AuditLogger(
            log_level="INFO",
            retention_days=2555,  # 7 years for compliance
            encryption_enabled=True,
            immutable_storage=True
        )

    @pytest.fixture
    async def compliance_config(self):
        """Create GDPR compliance configuration."""
        return ComplianceConfig(
            data_retention_days=90,
            right_to_be_forgotten=True,
            consent_tracking=True,
            data_portability=True,
            breach_notification_hours=72,
            privacy_by_design=True
        )

    @pytest.fixture
    async def gdpr_compliance(self, compliance_config, audit_logger, pii_detector):
        """Create GDPR compliance manager."""
        return GDPRCompliance(
            config=compliance_config,
            audit_logger=audit_logger,
            pii_detector=pii_detector
        )

    @pytest.fixture
    async def access_controller(self):
        """Create access controller for permission management."""
        return AccessController(
            rbac_enabled=True,
            mfa_required=True,
            session_timeout=1800,  # 30 minutes
            max_failed_attempts=3
        )

    @pytest.fixture
    async def secure_cache(self, encryption_manager, pii_detector, audit_logger):
        """Create security-enabled cache instance."""
        backend = MemoryBackend()
        strategy = LRUStrategy(capacity=500)
        cache = Cache(
            backend=backend,
            strategy=strategy,
            name="secure_cache",
            encryption_manager=encryption_manager,
            pii_detector=pii_detector,
            audit_logger=audit_logger
        )
        return cache

    async def test_end_to_end_encryption_workflow(self, secure_cache, encryption_manager):
        """Test complete encryption workflow from storage to retrieval."""
        # Test data with varying sensitivity levels
        test_data = {
            "public_config": {"theme": "dark", "language": "en"},
            "user_profile": {"name": "John Doe", "email": "john@example.com"},
            "sensitive_data": {"ssn": "123-45-6789", "credit_card": "4111-1111-1111-1111"},
            "medical_record": {"patient_id": "P123456", "diagnosis": "Type 2 Diabetes"}
        }

        # Store data with automatic encryption based on sensitivity
        for key, value in test_data.items():
            result = await secure_cache.set(key, value)
            assert result.success is True

        # Verify data is encrypted at rest
        raw_storage = await secure_cache.backend.get_raw(list(test_data.keys()))
        for key, raw_value in raw_storage.items():
            if key != "public_config":  # Public data might not be encrypted
                # Should be encrypted (not readable as original JSON)
                assert raw_value != json.dumps(test_data[key])

        # Verify decryption on retrieval
        for key, expected_value in test_data.items():
            retrieved = await secure_cache.get(key)
            assert retrieved == expected_value

        # Test key rotation
        await encryption_manager.rotate_keys()

        # Verify data still accessible after key rotation
        for key, expected_value in test_data.items():
            retrieved = await secure_cache.get(key)
            assert retrieved == expected_value

        # Verify encryption metrics
        encryption_stats = await encryption_manager.get_statistics()
        assert encryption_stats["total_encrypted"] >= len(test_data)
        assert encryption_stats["key_rotations"] >= 1

    async def test_pii_detection_and_handling(self, pii_detector, data_anonymizer, secure_cache):
        """Test PII detection, classification, and anonymization."""
        # Test data with various PII types
        pii_test_cases = [
            {
                "key": "user_data_1",
                "data": {
                    "name": "Alice Johnson",
                    "ssn": "987-65-4321",
                    "email": "alice.johnson@company.com",
                    "phone": "+1-555-123-4567",
                    "address": "123 Main St, Anytown, CA 90210"
                },
                "expected_pii": [PIICategory.SSN, PIICategory.EMAIL, PIICategory.PHONE]
            },
            {
                "key": "payment_info",
                "data": {
                    "card_number": "5555-5555-5555-4444",
                    "cvv": "123",
                    "expiry": "12/25",
                    "billing_zip": "90210"
                },
                "expected_pii": [PIICategory.CREDIT_CARD]
            },
            {
                "key": "medical_data",
                "data": {
                    "patient_id": "MRN-789456",
                    "dob": "1985-03-15",
                    "condition": "Hypertension",
                    "doctor": "Dr. Smith"
                },
                "expected_pii": [PIICategory.MEDICAL_ID]
            }
        ]

        for test_case in pii_test_cases:
            # Detect PII in data
            pii_analysis = await pii_detector.analyze_data(test_case["data"])

            # Verify PII detection
            detected_categories = [item.category for item in pii_analysis.pii_items]
            for expected_category in test_case["expected_pii"]:
                assert expected_category in detected_categories

            # Test different anonymization strategies
            for strategy_name in ["mask", "hash", "tokenize"]:
                anonymized = await data_anonymizer.anonymize(
                    test_case["data"],
                    pii_analysis,
                    strategy=strategy_name
                )

                # Verify PII is anonymized
                anonymized_analysis = await pii_detector.analyze_data(anonymized)
                assert len(anonymized_analysis.pii_items) < len(pii_analysis.pii_items)

            # Store with automatic PII handling
            await secure_cache.set(test_case["key"], test_case["data"])

            # Verify PII handling was applied
            cache_entry = await secure_cache.get_entry_metadata(test_case["key"])
            assert cache_entry["pii_detected"] is True
            assert cache_entry["anonymization_applied"] is True

    async def test_gdpr_compliance_workflow(self, gdpr_compliance, secure_cache):
        """Test complete GDPR compliance workflow."""
        # Subject data for GDPR testing
        subject_id = "gdpr_subject_123"
        subject_data = {
            f"{subject_id}:profile": {
                "name": "Maria Garcia",
                "email": "maria@example.com",
                "preferences": {"newsletter": True, "analytics": False}
            },
            f"{subject_id}:activity": {
                "last_login": "2024-01-15T10:30:00Z",
                "page_views": 157,
                "purchases": [{"item": "Widget", "date": "2024-01-10"}]
            },
            f"{subject_id}:consent": {
                "marketing": {"granted": True, "date": "2024-01-01T00:00:00Z"},
                "analytics": {"granted": False, "date": "2024-01-01T00:00:00Z"}
            }
        }

        # Store subject data
        for key, value in subject_data.items():
            await secure_cache.set(key, value)
            await gdpr_compliance.register_personal_data(subject_id, key, value)

        # Test Right to Access (Article 15)
        access_request = await gdpr_compliance.handle_access_request(subject_id)

        assert access_request["status"] == "completed"
        assert len(access_request["data"]) == len(subject_data)
        assert access_request["data_categories"] is not None

        # Test Data Portability (Article 20)
        portability_export = await gdpr_compliance.export_subject_data(
            subject_id,
            format="json"
        )

        assert portability_export["format"] == "json"
        assert len(portability_export["data"]) > 0
        assert portability_export["checksum"] is not None

        # Test Right to Rectification (Article 16)
        rectification_data = {
            f"{subject_id}:profile": {
                "name": "Maria Garcia-Smith",  # Updated name
                "email": "maria.smith@example.com"
            }
        }

        rectification_result = await gdpr_compliance.handle_rectification_request(
            subject_id,
            rectification_data
        )

        assert rectification_result["status"] == "completed"
        assert rectification_result["updated_records"] == 1

        # Verify rectification applied
        updated_profile = await secure_cache.get(f"{subject_id}:profile")
        assert updated_profile["name"] == "Maria Garcia-Smith"

        # Test Right to Erasure/Forget (Article 17)
        erasure_result = await gdpr_compliance.handle_erasure_request(
            subject_id,
            reason="withdrawal_of_consent"
        )

        assert erasure_result["status"] == "completed"
        assert erasure_result["deleted_records"] >= len(subject_data)

        # Verify data is erased
        for key in subject_data.keys():
            result = await secure_cache.get(key)
            assert result is None

        # Test Right to Restrict Processing (Article 18)
        # Add new data first
        await secure_cache.set(f"{subject_id}:new_data", {"test": "data"})

        restriction_result = await gdpr_compliance.restrict_processing(
            subject_id,
            reason="accuracy_dispute"
        )

        assert restriction_result["status"] == "completed"
        assert restriction_result["restricted_keys"] >= 1

    async def test_audit_logging_and_compliance_tracking(self, audit_logger, secure_cache):
        """Test comprehensive audit logging for compliance."""
        # Enable detailed audit logging
        await audit_logger.set_audit_level("DETAILED")

        # Perform various cache operations
        operations = [
            ("set", "audit_test_1", {"data": "test1"}),
            ("get", "audit_test_1", None),
            ("delete", "audit_test_1", None),
            ("set", "audit_test_2", {"sensitive": "ssn:123-45-6789"}),
        ]

        for operation, key, value in operations:
            if operation == "set":
                await secure_cache.set(key, value)
            elif operation == "get":
                await secure_cache.get(key)
            elif operation == "delete":
                await secure_cache.delete(key)

        # Retrieve audit trail
        audit_trail = await audit_logger.get_audit_trail(
            start_time=datetime.now() - timedelta(minutes=5),
            end_time=datetime.now()
        )

        assert len(audit_trail) >= len(operations)

        # Verify audit event structure
        for event in audit_trail:
            assert isinstance(event, AuditEvent)
            assert event.timestamp is not None
            assert event.operation in ["set", "get", "delete"]
            assert event.key is not None
            assert event.user_context is not None

        # Test audit trail integrity
        integrity_check = await audit_logger.verify_audit_integrity()
        assert integrity_check["tamper_detected"] is False
        assert integrity_check["missing_events"] == 0

        # Test compliance reporting
        compliance_report = await audit_logger.generate_compliance_report(
            period_days=1
        )

        assert compliance_report["total_operations"] >= len(operations)
        assert compliance_report["pii_operations"] >= 1  # SSN data operation
        assert compliance_report["encryption_events"] >= 0

    async def test_access_control_and_permissions(self, access_controller, secure_cache):
        """Test role-based access control and permissions."""
        # Define roles and permissions
        roles = {
            "admin": [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN],
            "user": [Permission.READ, Permission.WRITE],
            "viewer": [Permission.READ],
            "system": [Permission.SYSTEM]
        }

        # Create test users with different roles
        users = {
            "admin_user": "admin",
            "regular_user": "user",
            "readonly_user": "viewer",
            "system_process": "system"
        }

        # Setup users and roles
        for user_id, role in users.items():
            await access_controller.create_user(user_id, role, permissions=roles[role])

        # Test permission enforcement
        test_key = "protected_data"
        test_value = {"sensitive": True, "level": "confidential"}

        # Admin should have full access
        admin_context = await access_controller.authenticate("admin_user", "admin_password")
        assert admin_context["authenticated"] is True

        result = await secure_cache.set(test_key, test_value, user_context=admin_context)
        assert result.success is True

        # Regular user should be able to read and write
        user_context = await access_controller.authenticate("regular_user", "user_password")
        read_result = await secure_cache.get(test_key, user_context=user_context)
        assert read_result == test_value

        # Viewer should only be able to read
        viewer_context = await access_controller.authenticate("readonly_user", "viewer_password")
        read_result = await secure_cache.get(test_key, user_context=viewer_context)
        assert read_result == test_value

        # Viewer should not be able to write
        with pytest.raises(PermissionError):
            await secure_cache.set("new_key", "new_value", user_context=viewer_context)

        # Test MFA requirement for sensitive operations
        sensitive_key = "highly_classified"
        mfa_required_result = await secure_cache.set(
            sensitive_key,
            {"classification": "top_secret"},
            user_context=admin_context,
            require_mfa=True
        )

        # Should fail without MFA
        assert mfa_required_result.success is False
        assert "mfa_required" in mfa_required_result.error

        # Complete MFA and retry
        mfa_token = await access_controller.generate_mfa_token("admin_user")
        admin_context_with_mfa = await access_controller.verify_mfa(
            admin_context,
            mfa_token
        )

        mfa_result = await secure_cache.set(
            sensitive_key,
            {"classification": "top_secret"},
            user_context=admin_context_with_mfa
        )
        assert mfa_result.success is True

    async def test_data_breach_detection_and_notification(self, gdpr_compliance, audit_logger):
        """Test data breach detection and GDPR notification requirements."""
        # Simulate potential breach scenarios
        breach_scenarios = [
            {
                "type": "unauthorized_access",
                "affected_records": 150,
                "data_types": ["personal", "financial"],
                "severity": "high"
            },
            {
                "type": "data_exfiltration",
                "affected_records": 50,
                "data_types": ["personal"],
                "severity": "medium"
            },
            {
                "type": "encryption_failure",
                "affected_records": 300,
                "data_types": ["personal", "health"],
                "severity": "critical"
            }
        ]

        for scenario in breach_scenarios:
            # Simulate breach detection
            breach_event = await gdpr_compliance.detect_potential_breach(scenario)

            # Verify breach assessment
            assert breach_event["detected"] is True
            assert breach_event["severity"] == scenario["severity"]
            assert breach_event["gdpr_notification_required"] is True

            # Test notification workflow
            if scenario["severity"] in ["high", "critical"]:
                notification_result = await gdpr_compliance.initiate_breach_notification(
                    breach_event["breach_id"]
                )

                assert notification_result["supervisory_authority_notified"] is True
                assert notification_result["notification_time"] <= 72  # Hours

            # Test affected subject notification
            subject_notification = await gdpr_compliance.notify_affected_subjects(
                breach_event["breach_id"]
            )

            assert subject_notification["notifications_sent"] >= 0
            assert subject_notification["notification_method"] in ["email", "portal", "mail"]

        # Verify breach logs
        breach_logs = await audit_logger.get_breach_logs()
        assert len(breach_logs) == len(breach_scenarios)

    async def test_data_retention_and_automated_deletion(self, gdpr_compliance, secure_cache):
        """Test automated data retention and deletion policies."""
        # Create test data with different retention requirements
        retention_test_data = [
            {
                "key": "short_term_data",
                "value": {"type": "session", "created": time.time()},
                "retention_days": 1
            },
            {
                "key": "medium_term_data",
                "value": {"type": "preference", "created": time.time()},
                "retention_days": 30
            },
            {
                "key": "long_term_data",
                "value": {"type": "contract", "created": time.time()},
                "retention_days": 2555  # 7 years
            }
        ]

        # Store data with retention policies
        for item in retention_test_data:
            await secure_cache.set(item["key"], item["value"])
            await gdpr_compliance.set_retention_policy(
                item["key"],
                retention_days=item["retention_days"]
            )

        # Verify retention policies are set
        for item in retention_test_data:
            policy = await gdpr_compliance.get_retention_policy(item["key"])
            assert policy["retention_days"] == item["retention_days"]
            assert policy["auto_delete"] is True

        # Simulate time passage and test automated deletion
        await gdpr_compliance.simulate_time_passage(days=2)
        await gdpr_compliance.process_retention_policies()

        # Short-term data should be deleted
        result = await secure_cache.get("short_term_data")
        assert result is None

        # Medium and long-term data should still exist
        medium_result = await secure_cache.get("medium_term_data")
        assert medium_result is not None

        long_result = await secure_cache.get("long_term_data")
        assert long_result is not None

        # Verify deletion audit trail
        deletion_logs = await gdpr_compliance.get_deletion_logs()
        assert any(log["key"] == "short_term_data" for log in deletion_logs)

    async def test_cross_border_data_transfer_compliance(self, gdpr_compliance):
        """Test compliance for international data transfers."""
        # Test data localization requirements
        eu_data = {
            "key": "eu_citizen_data",
            "value": {"name": "Hans Mueller", "country": "DE"},
            "jurisdiction": "EU"
        }

        us_data = {
            "key": "us_citizen_data",
            "value": {"name": "John Smith", "country": "US"},
            "jurisdiction": "US"
        }

        # Store data with jurisdiction requirements
        await gdpr_compliance.store_with_jurisdiction(
            eu_data["key"],
            eu_data["value"],
            jurisdiction=eu_data["jurisdiction"]
        )

        await gdpr_compliance.store_with_jurisdiction(
            us_data["key"],
            us_data["value"],
            jurisdiction=us_data["jurisdiction"]
        )

        # Test adequacy decision validation
        transfer_request = await gdpr_compliance.validate_transfer_request(
            from_jurisdiction="EU",
            to_jurisdiction="US",
            data_category="personal"
        )

        assert transfer_request["adequacy_decision"] is not None
        assert transfer_request["safeguards_required"] is True

        # Test Standard Contractual Clauses (SCCs)
        scc_validation = await gdpr_compliance.validate_scc_compliance(
            transfer_request["transfer_id"]
        )

        assert scc_validation["scc_in_place"] is True
        assert scc_validation["compliant"] is True

    @pytest.mark.parametrize("encryption_algorithm", ["AES-256-GCM", "ChaCha20-Poly1305", "AES-256-CBC"])
    async def test_encryption_algorithm_compliance(self, encryption_algorithm):
        """Test different encryption algorithms for compliance requirements."""
        # Create encryption manager with specific algorithm
        config = EncryptionConfig(
            algorithm=encryption_algorithm,
            key_size=256,
            at_rest_encryption=True,
            fips_140_2_compliance=True
        )

        encryption_manager = EncryptionManager(config=config)
        await encryption_manager.initialize()

        # Test encryption strength
        test_data = "sensitive_test_data_" * 100  # Large data
        encrypted = await encryption_manager.encrypt(test_data)

        assert encrypted != test_data
        assert len(encrypted) > len(test_data)  # Includes IV, auth tag

        # Test decryption
        decrypted = await encryption_manager.decrypt(encrypted)
        assert decrypted == test_data

        # Test FIPS compliance
        compliance_status = await encryption_manager.verify_fips_compliance()
        assert compliance_status["compliant"] is True
        assert compliance_status["algorithm"] == encryption_algorithm

    async def test_security_monitoring_and_alerting(self, secure_cache, audit_logger):
        """Test security monitoring and real-time alerting."""
        # Enable security monitoring
        await secure_cache.enable_security_monitoring(
            alert_thresholds={
                "failed_access_attempts": 5,
                "bulk_data_access": 100,
                "unusual_access_patterns": True,
                "potential_data_exfiltration": True
            }
        )

        # Simulate suspicious activities
        suspicious_activities = [
            # Multiple failed access attempts
            *[("failed_access", f"protected_key_{i}") for i in range(10)],
            # Bulk data access
            *[("bulk_access", f"data_key_{i}") for i in range(150)],
            # Unusual timing (off-hours access)
            ("unusual_timing", "sensitive_data")
        ]

        alerts_generated = []

        for activity_type, key in suspicious_activities:
            try:
                if activity_type == "failed_access":
                    # This should fail and generate alert
                    await secure_cache.get(key)
                elif activity_type == "bulk_access":
                    await secure_cache.set(key, f"bulk_data_{key}")
                elif activity_type == "unusual_timing":
                    # Simulate off-hours access
                    await secure_cache.get_with_context(
                        key,
                        access_time=datetime.now().replace(hour=3)  # 3 AM
                    )
            except Exception:
                pass  # Expected for some scenarios

        # Check generated alerts
        security_alerts = await secure_cache.get_security_alerts()

        assert len(security_alerts) > 0
        assert any(alert["type"] == "bulk_access_detected" for alert in security_alerts)
        assert any(alert["type"] == "failed_access_threshold" for alert in security_alerts)

        # Verify alert details
        for alert in security_alerts:
            assert alert["timestamp"] is not None
            assert alert["severity"] in ["low", "medium", "high", "critical"]
            assert alert["description"] is not None
            assert alert["recommended_actions"] is not None