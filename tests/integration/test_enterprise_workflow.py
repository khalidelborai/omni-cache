"""
Integration test for complete enterprise workflow.

This test validates the integration of all enterprise features:
tier management, ML prefetching, security compliance, analytics,
and event-driven invalidation working together as a cohesive system.
"""

import pytest
import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from omnicache.core.cache import Cache
from omnicache.strategies.lru import LRUStrategy
from omnicache.backends.memory import MemoryBackend
from omnicache.backends.redis import RedisBackend

# Enterprise feature imports
from omnicache.enterprise.tier_manager import TierManager, TierConfig
from omnicache.enterprise.hierarchy import HierarchicalCache
from omnicache.ml.prefetcher import MLPrefetcher
from omnicache.ml.training_pipeline import TrainingPipeline
from omnicache.security.encryption import EncryptionManager, EncryptionConfig
from omnicache.security.compliance import GDPRCompliance, ComplianceConfig
from omnicache.security.access_control import AccessController
from omnicache.analytics.dashboard import AnalyticsDashboard, DashboardConfig
from omnicache.analytics.metrics_collector import MetricsCollector
from omnicache.events.invalidation_engine import InvalidationEngine
from omnicache.events.event_bus import EventBus, Event, EventType


@dataclass
class EnterpriseConfig:
    """Configuration for enterprise features."""
    tier_config: TierConfig
    encryption_config: EncryptionConfig
    compliance_config: ComplianceConfig
    dashboard_config: DashboardConfig
    enable_ml_prefetching: bool = True
    enable_real_time_analytics: bool = True
    enable_event_invalidation: bool = True


@pytest.mark.integration
class TestEnterpriseWorkflow:
    """Integration tests for complete enterprise workflow scenarios."""

    @pytest.fixture
    async def enterprise_config(self):
        """Create comprehensive enterprise configuration."""
        return EnterpriseConfig(
            tier_config=TierConfig(
                l1_capacity=100,
                l1_promotion_threshold=5,
                l2_capacity=500,
                l2_promotion_threshold=15,
                l3_capacity=2000,
                demotion_period=300,  # 5 minutes
                promotion_cooldown=30
            ),
            encryption_config=EncryptionConfig(
                algorithm="AES-256-GCM",
                key_rotation_interval=3600,
                at_rest_encryption=True,
                in_transit_encryption=True
            ),
            compliance_config=ComplianceConfig(
                data_retention_days=90,
                right_to_be_forgotten=True,
                consent_tracking=True,
                data_portability=True,
                breach_notification_hours=72
            ),
            dashboard_config=DashboardConfig(
                port=8080,
                update_interval=5,
                enable_real_time=True,
                enable_historical=True
            )
        )

    @pytest.fixture
    async def enterprise_cache_system(self, enterprise_config):
        """Create complete enterprise cache system."""
        # Core components
        event_bus = EventBus(
            max_events=50000,
            retention_hours=24,
            enable_persistence=True
        )

        metrics_collector = MetricsCollector(
            collection_interval=1,
            buffer_size=5000
        )

        # Security components
        encryption_manager = EncryptionManager(config=enterprise_config.encryption_config)
        await encryption_manager.initialize()

        access_controller = AccessController(
            rbac_enabled=True,
            mfa_required=True,
            session_timeout=1800
        )

        gdpr_compliance = GDPRCompliance(
            config=enterprise_config.compliance_config,
            audit_logger=None,  # Will be set up separately
            pii_detector=None   # Will be set up separately
        )

        # Tier management
        l1_backend = MemoryBackend()
        l1_cache = Cache(
            backend=l1_backend,
            strategy=LRUStrategy(capacity=enterprise_config.tier_config.l1_capacity),
            name="l1_enterprise_cache"
        )

        l2_backend = MemoryBackend()
        l2_cache = Cache(
            backend=l2_backend,
            strategy=LRUStrategy(capacity=enterprise_config.tier_config.l2_capacity),
            name="l2_enterprise_cache"
        )

        l3_backend = RedisBackend(host="localhost", port=6379, db=2)
        l3_cache = Cache(
            backend=l3_backend,
            strategy=LRUStrategy(capacity=enterprise_config.tier_config.l3_capacity),
            name="l3_enterprise_cache"
        )

        tier_manager = TierManager(
            l1_cache=l1_cache,
            l2_cache=l2_cache,
            l3_cache=l3_cache,
            config=enterprise_config.tier_config
        )

        # Event-driven invalidation
        invalidation_engine = InvalidationEngine(
            event_bus=event_bus,
            dependency_graph=None,  # Will be created
            batch_size=100
        )

        # ML Prefetching
        training_pipeline = TrainingPipeline(
            pattern_collector=None,  # Will be created
            models=[],  # Will be populated
            training_interval=300
        )

        ml_prefetcher = MLPrefetcher(
            cache=l1_cache,  # Use L1 for ML prefetching
            training_pipeline=training_pipeline,
            prefetch_batch_size=50,
            prefetch_threshold=0.7
        )

        # Analytics dashboard
        analytics_dashboard = AnalyticsDashboard(
            config=enterprise_config.dashboard_config,
            metrics_collector=metrics_collector,
            anomaly_detector=None  # Will be created
        )

        # Hierarchical cache with all features
        hierarchical_cache = HierarchicalCache(
            tier_manager=tier_manager,
            encryption_manager=encryption_manager,
            access_controller=access_controller,
            invalidation_engine=invalidation_engine,
            ml_prefetcher=ml_prefetcher if enterprise_config.enable_ml_prefetching else None,
            metrics_collector=metrics_collector,
            event_bus=event_bus
        )

        # Initialize all components
        await tier_manager.initialize()
        await invalidation_engine.initialize()
        await ml_prefetcher.initialize()
        await analytics_dashboard.start()
        await hierarchical_cache.initialize()

        return {
            "hierarchical_cache": hierarchical_cache,
            "tier_manager": tier_manager,
            "ml_prefetcher": ml_prefetcher,
            "analytics_dashboard": analytics_dashboard,
            "encryption_manager": encryption_manager,
            "access_controller": access_controller,
            "gdpr_compliance": gdpr_compliance,
            "event_bus": event_bus,
            "metrics_collector": metrics_collector
        }

    async def test_complete_enterprise_user_workflow(self, enterprise_cache_system):
        """Test complete enterprise user workflow from login to logout."""
        system = enterprise_cache_system
        cache = system["hierarchical_cache"]
        access_controller = system["access_controller"]
        analytics = system["analytics_dashboard"]

        # Phase 1: User Authentication and Authorization
        # Create enterprise user with specific permissions
        await access_controller.create_user(
            "enterprise_user_001",
            role="senior_analyst",
            permissions=["read", "write", "analytics_access"],
            department="finance",
            security_clearance="level_2"
        )

        # User login with MFA
        auth_context = await access_controller.authenticate(
            "enterprise_user_001",
            "secure_password_123",
            require_mfa=True
        )

        assert auth_context["authenticated"] is True
        assert auth_context["mfa_verified"] is True

        # Phase 2: Initial Data Access and Tier Placement
        # User accesses various data types
        user_data_requests = [
            # Personal dashboard data (frequently accessed)
            ("user:enterprise_user_001:dashboard", {
                "widgets": ["revenue_chart", "kpi_summary", "alerts"],
                "preferences": {"theme": "dark", "refresh_rate": 30}
            }),

            # Department reports (moderately accessed)
            ("department:finance:monthly_report", {
                "revenue": 1250000,
                "expenses": 890000,
                "profit_margin": 0.288,
                "generated": datetime.now().isoformat()
            }),

            # Company-wide analytics (occasionally accessed)
            ("company:analytics:yearly_trends", {
                "growth_rate": 0.15,
                "market_share": 0.23,
                "employee_count": 5000,
                "updated": datetime.now().isoformat()
            }),

            # Sensitive financial data (access controlled)
            ("finance:sensitive:quarterly_projections", {
                "q1_projection": 2100000,
                "q2_projection": 2300000,
                "confidence_level": 0.87,
                "classification": "confidential"
            })
        ]

        # Store and access data with automatic tier placement
        for key, value in user_data_requests:
            # Set data with user context for access control
            result = await cache.set(key, value, user_context=auth_context)
            assert result.success is True

            # Immediate read-back to establish access pattern
            retrieved = await cache.get(key, user_context=auth_context)
            assert retrieved == value

        # Phase 3: ML Pattern Learning and Prefetching
        # Simulate repeated access patterns for ML learning
        access_patterns = [
            # Morning routine pattern
            ["user:enterprise_user_001:dashboard", "department:finance:monthly_report"],
            # Analysis pattern
            ["department:finance:monthly_report", "company:analytics:yearly_trends"],
            # Executive review pattern
            ["company:analytics:yearly_trends", "finance:sensitive:quarterly_projections"]
        ]

        # Generate access patterns multiple times for ML training
        for pattern_cycle in range(10):
            for pattern in access_patterns:
                for key in pattern:
                    await cache.get(key, user_context=auth_context)
                    await asyncio.sleep(0.1)  # Realistic timing

        # Allow ML system to learn patterns
        await asyncio.sleep(5)

        # Trigger ML training
        training_result = await system["ml_prefetcher"].trigger_training()
        assert training_result["status"] == "completed"

        # Phase 4: Real-time Analytics and Monitoring
        # Check that analytics are being collected
        analytics_data = await analytics.get_dashboard_data()

        assert "user_activity" in analytics_data
        assert "access_patterns" in analytics_data
        assert "performance_metrics" in analytics_data

        # Verify user-specific analytics
        user_analytics = await analytics.get_user_analytics("enterprise_user_001")
        assert user_analytics["total_requests"] > 0
        assert user_analytics["data_categories_accessed"] > 0

        # Phase 5: Event-driven Updates and Invalidation
        # Simulate external data update that should trigger invalidation
        await system["event_bus"].emit(Event(
            type=EventType.DATA_CHANGED,
            source="finance_system",
            data={
                "entity_type": "financial_report",
                "entity_id": "monthly_report",
                "action": "update",
                "affected_keys": ["department:finance:monthly_report"],
                "user_context": auth_context
            }
        ))

        await asyncio.sleep(2)

        # Verify invalidation occurred and dependent data was updated
        report_result = await cache.get(
            "department:finance:monthly_report",
            user_context=auth_context
        )
        # Should be invalidated due to event
        assert report_result is None

        # Phase 6: Security and Compliance Validation
        # Verify data encryption for sensitive information
        encryption_status = await system["encryption_manager"].get_key_encryption_status(
            "finance:sensitive:quarterly_projections"
        )
        assert encryption_status["encrypted"] is True

        # Test GDPR compliance - data access request
        gdpr_request = await system["gdpr_compliance"].handle_access_request(
            subject_id="enterprise_user_001"
        )
        assert gdpr_request["status"] == "completed"
        assert len(gdpr_request["data"]) > 0

        # Phase 7: Prefetching Effectiveness Test
        # Start new session to test prefetching predictions
        new_session_context = await access_controller.create_session(
            "enterprise_user_001",
            session_type="morning_routine"
        )

        # Access first item in known pattern
        await cache.get(
            "user:enterprise_user_001:dashboard",
            user_context=new_session_context
        )

        # Check if next items in pattern were prefetched
        prefetch_stats = await system["ml_prefetcher"].get_prefetch_statistics()
        assert prefetch_stats["successful_predictions"] > 0

        # Phase 8: Performance and Tier Optimization
        # Check tier distribution after all operations
        tier_stats = await system["tier_manager"].get_tier_stats()

        # Frequently accessed data should be in L1
        l1_keys = await system["tier_manager"].get_tier_contents("l1")
        assert "user:enterprise_user_001:dashboard" in l1_keys

        # Performance metrics should show good cache efficiency
        performance_metrics = await analytics.get_performance_metrics()
        assert performance_metrics["cache_hit_ratio"] > 0.6
        assert performance_metrics["avg_response_time"] < 0.1  # < 100ms

        # Phase 9: User Logout and Session Cleanup
        logout_result = await access_controller.logout(auth_context["session_id"])
        assert logout_result["success"] is True

        # Verify session cleanup
        session_valid = await access_controller.validate_session(auth_context["session_id"])
        assert session_valid is False

    async def test_multi_tenant_enterprise_scenario(self, enterprise_cache_system):
        """Test enterprise features with multiple tenants and complex access patterns."""
        system = enterprise_cache_system
        cache = system["hierarchical_cache"]
        access_controller = system["access_controller"]

        # Set up multiple tenants with different characteristics
        tenants = {
            "enterprise_corp": {
                "users": ["ceo", "cfo", "analyst_1", "analyst_2"],
                "data_sensitivity": "high",
                "compliance_requirements": ["gdpr", "sox", "hipaa"],
                "tier_preferences": {"hot_data_retention": 24, "cold_data_archive": 90}
            },
            "startup_inc": {
                "users": ["founder", "engineer_1", "marketer_1"],
                "data_sensitivity": "medium",
                "compliance_requirements": ["gdpr"],
                "tier_preferences": {"hot_data_retention": 6, "cold_data_archive": 30}
            },
            "small_business": {
                "users": ["owner", "assistant"],
                "data_sensitivity": "low",
                "compliance_requirements": [],
                "tier_preferences": {"hot_data_retention": 2, "cold_data_archive": 7}
            }
        }

        # Create tenant users and data
        tenant_data = {}
        for tenant_id, tenant_info in tenants.items():
            tenant_data[tenant_id] = {}

            # Create users for each tenant
            for user in tenant_info["users"]:
                user_id = f"{tenant_id}:{user}"
                await access_controller.create_user(
                    user_id,
                    role=user,
                    tenant_id=tenant_id,
                    permissions=["read", "write", "tenant_data_access"]
                )

                # Create tenant-specific data
                tenant_data[tenant_id][f"{user_id}:workspace"] = {
                    "tenant": tenant_id,
                    "user": user,
                    "created": datetime.now().isoformat(),
                    "data_classification": tenant_info["data_sensitivity"]
                }

                tenant_data[tenant_id][f"{tenant_id}:shared:reports"] = {
                    "monthly_summary": {"revenue": 100000 * len(tenant_info["users"])},
                    "user_activity": {"active_users": len(tenant_info["users"])},
                    "tenant_id": tenant_id
                }

        # Store tenant data with appropriate security and tier settings
        for tenant_id, data_dict in tenant_data.items():
            for key, value in data_dict.items():
                # Authenticate as tenant user
                primary_user = f"{tenant_id}:{tenants[tenant_id]['users'][0]}"
                auth_context = await access_controller.authenticate(
                    primary_user,
                    "password_123"
                )

                # Store with tenant context
                await cache.set(
                    key,
                    value,
                    user_context=auth_context,
                    tenant_isolation=True,
                    encryption_level=tenants[tenant_id]["data_sensitivity"]
                )

        # Test cross-tenant access isolation
        # User from enterprise_corp should not access startup_inc data
        enterprise_user_context = await access_controller.authenticate(
            "enterprise_corp:ceo",
            "password_123"
        )

        startup_user_context = await access_controller.authenticate(
            "startup_inc:founder",
            "password_123"
        )

        # Enterprise user trying to access startup data should fail
        with pytest.raises(PermissionError):
            await cache.get(
                "startup_inc:founder:workspace",
                user_context=enterprise_user_context
            )

        # Startup user should access their own data successfully
        startup_data = await cache.get(
            "startup_inc:founder:workspace",
            user_context=startup_user_context
        )
        assert startup_data is not None
        assert startup_data["tenant"] == "startup_inc"

        # Test tenant-specific analytics
        for tenant_id in tenants.keys():
            tenant_analytics = await system["analytics_dashboard"].get_tenant_analytics(tenant_id)
            assert tenant_analytics["tenant_id"] == tenant_id
            assert tenant_analytics["user_count"] == len(tenants[tenant_id]["users"])
            assert tenant_analytics["data_access_count"] > 0

        # Test tenant-specific compliance
        # Enterprise tenant should have stricter compliance
        enterprise_compliance = await system["gdpr_compliance"].get_tenant_compliance_status("enterprise_corp")
        assert "sox" in enterprise_compliance["required_compliance"]
        assert "hipaa" in enterprise_compliance["required_compliance"]

        # Small business should have minimal compliance requirements
        small_biz_compliance = await system["gdpr_compliance"].get_tenant_compliance_status("small_business")
        assert len(small_biz_compliance["required_compliance"]) == 0

    async def test_enterprise_disaster_recovery_workflow(self, enterprise_cache_system):
        """Test enterprise disaster recovery and failover scenarios."""
        system = enterprise_cache_system
        cache = system["hierarchical_cache"]

        # Set up critical enterprise data
        critical_data = {
            "system:primary:config": {"database_url": "primary-db.company.com", "api_key": "critical_key_123"},
            "system:primary:sessions": {"active_count": 1500, "peak_today": 2100},
            "business:primary:revenue": {"today": 45000, "month_to_date": 1200000},
            "operations:primary:alerts": {"critical": 0, "warning": 3, "info": 12}
        }

        # Store critical data with high availability requirements
        for key, value in critical_data.items():
            await cache.set(
                key,
                value,
                replication_level="high",
                backup_to_all_tiers=True,
                priority="critical"
            )

        # Verify data is stored across all tiers
        for key in critical_data.keys():
            for tier in ["l1", "l2", "l3"]:
                tier_result = await system["tier_manager"].get(key, tier=tier)
                assert tier_result is not None

        # Simulate L1 cache failure
        await system["tier_manager"].simulate_tier_failure("l1")

        # Verify data still accessible from L2/L3
        for key, expected_value in critical_data.items():
            result = await cache.get(key)
            assert result == expected_value

        # Test failover metrics
        failover_stats = await system["tier_manager"].get_failover_statistics()
        assert failover_stats["active_failovers"] > 0
        assert failover_stats["l1_status"] == "failed"
        assert failover_stats["l2_status"] == "active"

        # Simulate L1 recovery
        await system["tier_manager"].recover_tier("l1")

        # Test data synchronization after recovery
        sync_result = await system["tier_manager"].synchronize_tiers()
        assert sync_result["synchronization_complete"] is True
        assert sync_result["data_consistency_verified"] is True

        # Verify L1 has been repopulated with critical data
        l1_contents = await system["tier_manager"].get_tier_contents("l1")
        critical_keys_in_l1 = [key for key in critical_data.keys() if key in l1_contents]
        assert len(critical_keys_in_l1) > 0

    async def test_enterprise_compliance_audit_workflow(self, enterprise_cache_system):
        """Test comprehensive compliance audit workflow."""
        system = enterprise_cache_system
        cache = system["hierarchical_cache"]
        compliance = system["gdpr_compliance"]

        # Set up audit scenario with PII and sensitive data
        audit_data = {
            "customer:12345:profile": {
                "name": "John Smith",
                "email": "john.smith@email.com",
                "ssn": "123-45-6789",
                "credit_score": 750,
                "account_created": "2023-01-15"
            },
            "customer:12345:transactions": [
                {"amount": 1500.00, "date": "2024-01-15", "merchant": "Electronics Store"},
                {"amount": 85.50, "date": "2024-01-14", "merchant": "Gas Station"}
            ],
            "customer:12345:marketing_consent": {
                "email_marketing": {"consent": True, "date": "2023-01-15"},
                "phone_marketing": {"consent": False, "date": "2023-01-15"}
            },
            "system:audit:access_log": [
                {"user": "analyst_1", "action": "view_profile", "timestamp": "2024-01-15T10:30:00Z"},
                {"user": "support_agent", "action": "update_profile", "timestamp": "2024-01-15T14:20:00Z"}
            ]
        }

        # Store data with compliance tracking
        for key, value in audit_data.items():
            await cache.set(
                key,
                value,
                compliance_tracking=True,
                data_classification="personal",
                retention_policy="customer_data"
            )

            # Register with GDPR compliance system
            await compliance.register_personal_data(
                subject_id="12345",
                data_key=key,
                data_value=value
            )

        # Simulate compliance audit process
        audit_report = await compliance.generate_audit_report(
            audit_type="comprehensive",
            subject_id="12345",
            include_access_logs=True,
            include_data_lineage=True
        )

        # Verify audit report completeness
        assert audit_report["subject_id"] == "12345"
        assert audit_report["total_data_points"] >= len(audit_data)
        assert "pii_detected" in audit_report
        assert "consent_status" in audit_report
        assert "retention_compliance" in audit_report

        # Test data subject rights
        # Right to Access
        access_response = await compliance.handle_access_request("12345")
        assert access_response["status"] == "completed"
        assert len(access_response["data"]) > 0

        # Right to Rectification
        rectification_data = {
            "customer:12345:profile": {
                "name": "John Q. Smith",  # Updated name
                "email": "john.q.smith@newemail.com"  # Updated email
            }
        }

        rectification_result = await compliance.handle_rectification_request(
            "12345",
            rectification_data
        )
        assert rectification_result["status"] == "completed"

        # Right to Erasure (Right to be Forgotten)
        erasure_result = await compliance.handle_erasure_request(
            "12345",
            reason="customer_request"
        )
        assert erasure_result["status"] == "completed"
        assert erasure_result["deleted_records"] >= len(audit_data)

        # Verify data has been erased
        for key in audit_data.keys():
            result = await cache.get(key)
            assert result is None

        # Verify audit trail of erasure
        erasure_audit = await compliance.get_erasure_audit_trail("12345")
        assert len(erasure_audit) > 0
        assert erasure_audit[0]["action"] == "data_erasure"

    async def test_enterprise_performance_optimization_workflow(self, enterprise_cache_system):
        """Test enterprise performance optimization and auto-tuning."""
        system = enterprise_cache_system
        cache = system["hierarchical_cache"]
        analytics = system["analytics_dashboard"]

        # Generate realistic enterprise workload
        workload_scenarios = [
            {
                "name": "morning_rush",
                "duration": 30,  # seconds
                "operations_per_second": 100,
                "read_write_ratio": 0.8,
                "data_pattern": "user_dashboards"
            },
            {
                "name": "business_hours",
                "duration": 60,
                "operations_per_second": 200,
                "read_write_ratio": 0.7,
                "data_pattern": "mixed_enterprise"
            },
            {
                "name": "batch_processing",
                "duration": 20,
                "operations_per_second": 500,
                "read_write_ratio": 0.3,
                "data_pattern": "bulk_updates"
            }
        ]

        baseline_performance = {}

        for scenario in workload_scenarios:
            # Execute workload scenario
            scenario_start = time.time()

            workload_task = asyncio.create_task(
                self._execute_enterprise_workload(cache, scenario)
            )

            await workload_task

            scenario_duration = time.time() - scenario_start

            # Collect performance metrics
            performance_metrics = await analytics.get_performance_metrics()
            baseline_performance[scenario["name"]] = {
                "duration": scenario_duration,
                "ops_per_second": performance_metrics["operations_per_second"],
                "hit_ratio": performance_metrics["cache_hit_ratio"],
                "avg_response_time": performance_metrics["avg_response_time"]
            }

        # Trigger auto-optimization
        optimization_result = await system["tier_manager"].optimize_performance(
            baseline_metrics=baseline_performance,
            optimization_goals=["throughput", "latency", "hit_ratio"]
        )

        assert optimization_result["optimizations_applied"] > 0
        assert optimization_result["estimated_improvement"] > 0

        # Re-run scenarios to measure improvement
        optimized_performance = {}

        for scenario in workload_scenarios:
            workload_task = asyncio.create_task(
                self._execute_enterprise_workload(cache, scenario)
            )
            await workload_task

            performance_metrics = await analytics.get_performance_metrics()
            optimized_performance[scenario["name"]] = {
                "ops_per_second": performance_metrics["operations_per_second"],
                "hit_ratio": performance_metrics["cache_hit_ratio"],
                "avg_response_time": performance_metrics["avg_response_time"]
            }

        # Verify performance improvements
        for scenario_name in workload_scenarios:
            baseline = baseline_performance[scenario_name]
            optimized = optimized_performance[scenario_name]

            # Should show improvement in at least one key metric
            improvement_found = (
                optimized["ops_per_second"] > baseline["ops_per_second"] * 1.05 or
                optimized["hit_ratio"] > baseline["hit_ratio"] * 1.02 or
                optimized["avg_response_time"] < baseline["avg_response_time"] * 0.95
            )

            assert improvement_found, f"No improvement found for scenario {scenario_name}"

    async def test_enterprise_scaling_and_load_balancing(self, enterprise_cache_system):
        """Test enterprise scaling under high load with load balancing."""
        system = enterprise_cache_system
        cache = system["hierarchical_cache"]

        # Configure auto-scaling parameters
        await system["tier_manager"].configure_auto_scaling(
            enable_auto_scaling=True,
            scale_up_threshold=0.8,  # 80% capacity
            scale_down_threshold=0.3,  # 30% capacity
            max_scale_factor=3,
            min_instances=1
        )

        # Generate high load to trigger scaling
        high_load_tasks = []

        for worker_id in range(20):  # 20 concurrent workers
            task = asyncio.create_task(
                self._generate_high_load_worker(cache, worker_id, operations=500)
            )
            high_load_tasks.append(task)

        # Monitor scaling during load
        scaling_events = []

        async def monitor_scaling():
            for _ in range(30):  # Monitor for 30 seconds
                scaling_status = await system["tier_manager"].get_scaling_status()
                if scaling_status["scaling_event"]:
                    scaling_events.append({
                        "timestamp": time.time(),
                        "event": scaling_status["scaling_event"],
                        "current_instances": scaling_status["active_instances"]
                    })
                await asyncio.sleep(1)

        monitor_task = asyncio.create_task(monitor_scaling())

        # Wait for all load and monitoring
        await asyncio.gather(*high_load_tasks, monitor_task)

        # Verify scaling occurred
        assert len(scaling_events) > 0
        scale_up_events = [e for e in scaling_events if e["event"] == "scale_up"]
        assert len(scale_up_events) > 0

        # Verify system handled load effectively
        final_performance = await system["analytics_dashboard"].get_performance_metrics()
        assert final_performance["error_rate"] < 0.01  # < 1% errors
        assert final_performance["operations_per_second"] > 1000  # High throughput maintained

        # Test scale-down after load reduction
        await asyncio.sleep(10)  # Allow system to detect reduced load

        scale_down_status = await system["tier_manager"].get_scaling_status()
        # Should start scaling down after load reduction
        assert scale_down_status["scaling_direction"] in ["down", "stable"]

    # Helper methods for complex workflow testing

    async def _execute_enterprise_workload(self, cache, scenario: Dict[str, Any]):
        """Execute a realistic enterprise workload scenario."""
        duration = scenario["duration"]
        ops_per_second = scenario["operations_per_second"]
        read_write_ratio = scenario["read_write_ratio"]
        pattern = scenario["data_pattern"]

        start_time = time.time()
        operation_count = 0

        while time.time() - start_time < duration:
            # Determine operation type
            if np.random.random() < read_write_ratio:
                operation = "read"
            else:
                operation = "write"

            # Generate key based on pattern
            if pattern == "user_dashboards":
                key = f"user:{np.random.randint(1, 1000)}:dashboard"
                value = {"widgets": ["chart1", "chart2"], "last_updated": time.time()}
            elif pattern == "mixed_enterprise":
                categories = ["user", "department", "company", "system"]
                category = np.random.choice(categories)
                key = f"{category}:{np.random.randint(1, 100)}:data"
                value = {"category": category, "data": f"value_{operation_count}"}
            elif pattern == "bulk_updates":
                key = f"bulk:batch_{operation_count // 10}:item_{operation_count % 10}"
                value = {"batch_id": operation_count // 10, "item_data": f"bulk_value_{operation_count}"}

            # Execute operation
            if operation == "read":
                await cache.get(key)
            else:
                await cache.set(key, value)

            operation_count += 1

            # Rate limiting
            target_interval = 1.0 / ops_per_second
            await asyncio.sleep(target_interval)

    async def _generate_high_load_worker(self, cache, worker_id: int, operations: int):
        """Generate high load from a single worker."""
        for i in range(operations):
            key = f"load_test:worker_{worker_id}:op_{i}"
            value = f"load_data_{worker_id}_{i}"

            # Mix of operations
            if i % 3 == 0:
                await cache.set(key, value)
            else:
                await cache.get(key)

            # High frequency operations
            await asyncio.sleep(0.01)  # 100 ops/second per worker