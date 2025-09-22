"""
Integration test for event-driven invalidation system.

This test validates event-driven cache invalidation with dependency graphs,
real-time updates, and complex invalidation patterns.
"""

import pytest
import asyncio
import time
import json
from typing import Dict, List, Any, Set, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from omnicache.core.cache import Cache
from omnicache.strategies.lru import LRUStrategy
from omnicache.backends.memory import MemoryBackend
from omnicache.events.event_bus import EventBus, Event, EventType
from omnicache.events.invalidation_engine import InvalidationEngine, InvalidationStrategy
from omnicache.events.dependency_graph import DependencyGraph, DependencyType, DependencyNode
from omnicache.events.event_listeners import EventListener, InvalidationListener
from omnicache.events.event_patterns import EventPattern, PatternMatcher


class EventCategory(Enum):
    """Event categories for testing."""
    DATA_UPDATE = "data_update"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    EXTERNAL_API = "external_api"
    SCHEDULED_TASK = "scheduled_task"


@dataclass
class TestEvent:
    """Test event structure."""
    id: str
    category: EventCategory
    entity_type: str
    entity_id: str
    action: str
    timestamp: datetime
    metadata: Dict[str, Any]


@pytest.mark.integration
class TestEventInvalidation:
    """Integration tests for event-driven invalidation system."""

    @pytest.fixture
    async def event_bus(self):
        """Create event bus for testing."""
        return EventBus(
            max_events=10000,
            retention_hours=24,
            enable_persistence=True,
            enable_replay=True
        )

    @pytest.fixture
    async def dependency_graph(self):
        """Create dependency graph for cache relationships."""
        graph = DependencyGraph(
            enable_cycle_detection=True,
            max_depth=10,
            enable_transitive_dependencies=True
        )
        await graph.initialize()
        return graph

    @pytest.fixture
    async def invalidation_engine(self, event_bus, dependency_graph):
        """Create invalidation engine."""
        engine = InvalidationEngine(
            event_bus=event_bus,
            dependency_graph=dependency_graph,
            batch_size=50,
            batch_timeout=1.0,  # 1 second for testing
            enable_smart_invalidation=True
        )
        await engine.initialize()
        return engine

    @pytest.fixture
    async def pattern_matcher(self):
        """Create event pattern matcher."""
        return PatternMatcher(
            enable_wildcards=True,
            enable_regex=True,
            case_sensitive=False
        )

    @pytest.fixture
    async def cache_with_events(self, invalidation_engine):
        """Create cache with event-driven invalidation."""
        backend = MemoryBackend()
        strategy = LRUStrategy(capacity=1000)
        cache = Cache(
            backend=backend,
            strategy=strategy,
            name="event_cache",
            invalidation_engine=invalidation_engine
        )
        return cache

    async def test_basic_event_driven_invalidation(self, cache_with_events, event_bus, dependency_graph):
        """Test basic event-driven invalidation workflow."""
        # Set up data and dependencies
        test_data = {
            "user:123:profile": {"name": "John Doe", "email": "john@example.com"},
            "user:123:preferences": {"theme": "dark", "language": "en"},
            "user:123:sessions": [{"id": "session_1", "created": "2024-01-01"}],
            "user:123:cache_summary": {"total_items": 3, "last_update": time.time()}
        }

        # Store data
        for key, value in test_data.items():
            await cache_with_events.set(key, value)

        # Set up dependencies - summary depends on other user data
        await dependency_graph.add_dependency(
            "user:123:cache_summary",
            ["user:123:profile", "user:123:preferences", "user:123:sessions"],
            dependency_type=DependencyType.COMPUTED
        )

        # Emit user profile update event
        profile_update_event = TestEvent(
            id="event_001",
            category=EventCategory.DATA_UPDATE,
            entity_type="user",
            entity_id="123",
            action="profile_update",
            timestamp=datetime.now(),
            metadata={"fields_changed": ["email"]}
        )

        await event_bus.emit(Event(
            type=EventType.DATA_CHANGED,
            source="user_service",
            data={
                "entity_type": profile_update_event.entity_type,
                "entity_id": profile_update_event.entity_id,
                "action": profile_update_event.action,
                "affected_keys": ["user:123:profile"]
            }
        ))

        # Allow invalidation to process
        await asyncio.sleep(1)

        # Verify profile was invalidated
        profile_result = await cache_with_events.get("user:123:profile")
        assert profile_result is None

        # Verify dependent summary was also invalidated
        summary_result = await cache_with_events.get("user:123:cache_summary")
        assert summary_result is None

        # Verify non-dependent data remains
        preferences_result = await cache_with_events.get("user:123:preferences")
        assert preferences_result is not None

    async def test_complex_dependency_graph_invalidation(self, cache_with_events, dependency_graph, event_bus):
        """Test complex dependency relationships and transitive invalidation."""
        # Create hierarchical data structure
        hierarchy_data = {
            # Raw data
            "product:1:details": {"name": "Widget", "price": 29.99},
            "product:1:inventory": {"quantity": 100, "location": "warehouse_a"},
            "product:1:reviews": [{"rating": 5, "comment": "Great!"}],

            # Computed data (level 1)
            "product:1:summary": {"avg_rating": 5.0, "in_stock": True},
            "product:1:display": {"formatted_price": "$29.99", "availability": "In Stock"},

            # Computed data (level 2)
            "category:electronics:summary": {"total_products": 1, "avg_price": 29.99},
            "search:index:products": {"product:1": {"name": "Widget", "searchable": True}},

            # Computed data (level 3)
            "dashboard:admin:overview": {"categories": 1, "total_inventory": 100}
        }

        # Store all data
        for key, value in hierarchy_data.items():
            await cache_with_events.set(key, value)

        # Set up dependency relationships
        dependencies = [
            # Level 1 dependencies
            ("product:1:summary", ["product:1:details", "product:1:reviews"], DependencyType.COMPUTED),
            ("product:1:display", ["product:1:details", "product:1:inventory"], DependencyType.COMPUTED),

            # Level 2 dependencies
            ("category:electronics:summary", ["product:1:summary"], DependencyType.AGGREGATION),
            ("search:index:products", ["product:1:details"], DependencyType.INDEX),

            # Level 3 dependencies
            ("dashboard:admin:overview", ["category:electronics:summary", "product:1:inventory"], DependencyType.AGGREGATION)
        ]

        for dependent_key, dependency_keys, dep_type in dependencies:
            await dependency_graph.add_dependency(dependent_key, dependency_keys, dep_type)

        # Emit product details update
        await event_bus.emit(Event(
            type=EventType.DATA_CHANGED,
            source="product_service",
            data={
                "entity_type": "product",
                "entity_id": "1",
                "action": "details_update",
                "affected_keys": ["product:1:details"]
            }
        ))

        await asyncio.sleep(2)  # Allow cascading invalidation

        # Verify cascading invalidation
        invalidated_keys = [
            "product:1:details",      # Direct
            "product:1:summary",      # Level 1
            "product:1:display",      # Level 1
            "category:electronics:summary",  # Level 2
            "search:index:products",  # Level 2
            "dashboard:admin:overview"  # Level 3
        ]

        for key in invalidated_keys:
            result = await cache_with_events.get(key)
            assert result is None, f"Key {key} should have been invalidated"

        # Verify non-dependent data remains
        reviews_result = await cache_with_events.get("product:1:reviews")
        inventory_result = await cache_with_events.get("product:1:inventory")
        assert reviews_result is not None
        assert inventory_result is not None

    async def test_event_pattern_matching_and_selective_invalidation(self, cache_with_events, event_bus, pattern_matcher):
        """Test event pattern matching for selective invalidation."""
        # Set up data for different tenants/users
        multi_tenant_data = {
            "tenant:A:user:1:profile": {"name": "Alice"},
            "tenant:A:user:2:profile": {"name": "Bob"},
            "tenant:B:user:1:profile": {"name": "Charlie"},
            "tenant:B:user:2:profile": {"name": "Diana"},
            "tenant:A:config:settings": {"theme": "light"},
            "tenant:B:config:settings": {"theme": "dark"}
        }

        for key, value in multi_tenant_data.items():
            await cache_with_events.set(key, value)

        # Set up pattern-based invalidation rules
        invalidation_patterns = [
            {
                "pattern": "tenant:A:*",
                "event_pattern": {"entity_type": "tenant", "entity_id": "A"},
                "description": "Invalidate all Tenant A data"
            },
            {
                "pattern": "*:user:*:profile",
                "event_pattern": {"entity_type": "user", "action": "profile_update"},
                "description": "Invalidate user profiles on updates"
            },
            {
                "pattern": "tenant:*:config:*",
                "event_pattern": {"entity_type": "config"},
                "description": "Invalidate configuration data"
            }
        ]

        # Register patterns with invalidation engine
        for pattern_def in invalidation_patterns:
            await cache_with_events.invalidation_engine.register_pattern(
                pattern_def["pattern"],
                pattern_def["event_pattern"]
            )

        # Test tenant-specific invalidation
        await event_bus.emit(Event(
            type=EventType.TENANT_UPDATE,
            source="tenant_service",
            data={
                "entity_type": "tenant",
                "entity_id": "A",
                "action": "settings_changed"
            }
        ))

        await asyncio.sleep(1)

        # Verify only Tenant A data was invalidated
        tenant_a_keys = ["tenant:A:user:1:profile", "tenant:A:user:2:profile", "tenant:A:config:settings"]
        tenant_b_keys = ["tenant:B:user:1:profile", "tenant:B:user:2:profile", "tenant:B:config:settings"]

        for key in tenant_a_keys:
            result = await cache_with_events.get(key)
            assert result is None, f"Tenant A key {key} should be invalidated"

        for key in tenant_b_keys:
            result = await cache_with_events.get(key)
            assert result is not None, f"Tenant B key {key} should remain valid"

    async def test_real_time_event_streaming_invalidation(self, cache_with_events, event_bus):
        """Test real-time event streaming and immediate invalidation."""
        # Set up real-time data that changes frequently
        real_time_data = {
            "stock:AAPL:price": {"current": 150.00, "timestamp": time.time()},
            "stock:GOOGL:price": {"current": 2500.00, "timestamp": time.time()},
            "portfolio:user123:value": {"total": 10000.00, "updated": time.time()},
            "market:summary:stats": {"total_volume": 1000000, "updated": time.time()}
        }

        for key, value in real_time_data.items():
            await cache_with_events.set(key, value)

        # Simulate real-time price updates
        price_updates = [
            {"symbol": "AAPL", "price": 151.50, "timestamp": time.time() + 1},
            {"symbol": "AAPL", "price": 152.25, "timestamp": time.time() + 2},
            {"symbol": "GOOGL", "price": 2510.00, "timestamp": time.time() + 1.5},
        ]

        invalidation_events = []

        # Set up event listener to track invalidations
        async def track_invalidations(event):
            invalidation_events.append({
                "timestamp": time.time(),
                "invalidated_keys": event.data.get("invalidated_keys", [])
            })

        await event_bus.subscribe("cache.invalidation", track_invalidations)

        # Stream price updates
        for update in price_updates:
            await event_bus.emit(Event(
                type=EventType.REAL_TIME_UPDATE,
                source="market_data",
                data={
                    "entity_type": "stock",
                    "entity_id": update["symbol"],
                    "action": "price_update",
                    "new_price": update["price"],
                    "affected_keys": [f"stock:{update['symbol']}:price"]
                }
            ))

            # Very short delay to simulate real-time
            await asyncio.sleep(0.1)

        await asyncio.sleep(1)

        # Verify immediate invalidations occurred
        assert len(invalidation_events) >= len(price_updates)

        # Verify specific stocks were invalidated
        aapl_result = await cache_with_events.get("stock:AAPL:price")
        googl_result = await cache_with_events.get("stock:GOOGL:price")

        assert aapl_result is None
        assert googl_result is None

    async def test_batch_invalidation_optimization(self, cache_with_events, event_bus, invalidation_engine):
        """Test batch invalidation for performance optimization."""
        # Create large dataset
        batch_data = {}
        for i in range(200):
            batch_data[f"batch:item:{i}:data"] = {"id": i, "value": f"data_{i}"}
            batch_data[f"batch:item:{i}:metadata"] = {"created": time.time(), "index": i}

        for key, value in batch_data.items():
            await cache_with_events.set(key, value)

        # Configure batch invalidation
        await invalidation_engine.configure_batching(
            batch_size=50,
            batch_timeout=0.5,  # 500ms
            enable_optimization=True
        )

        # Emit rapid sequence of events
        start_time = time.time()

        for i in range(100):
            await event_bus.emit(Event(
                type=EventType.BULK_UPDATE,
                source="batch_processor",
                data={
                    "entity_type": "batch_item",
                    "entity_id": str(i),
                    "action": "update",
                    "affected_keys": [f"batch:item:{i}:data", f"batch:item:{i}:metadata"]
                }
            ))

        # Allow batch processing
        await asyncio.sleep(2)

        processing_time = time.time() - start_time

        # Verify batch optimization occurred
        batch_stats = await invalidation_engine.get_batch_statistics()

        assert batch_stats["total_batches"] > 0
        assert batch_stats["total_batches"] < 100  # Should be fewer than individual events
        assert batch_stats["average_batch_size"] > 1
        assert processing_time < 5  # Should complete quickly due to batching

        # Verify invalidations were effective
        invalidated_count = 0
        for i in range(100):
            data_result = await cache_with_events.get(f"batch:item:{i}:data")
            if data_result is None:
                invalidated_count += 1

        assert invalidated_count >= 100  # All items should be invalidated

    async def test_conditional_invalidation_rules(self, cache_with_events, event_bus):
        """Test conditional invalidation based on event content."""
        # Set up conditional data
        conditional_data = {
            "user:vip:123:profile": {"tier": "platinum", "credit": 10000},
            "user:regular:456:profile": {"tier": "bronze", "credit": 100},
            "user:vip:789:profile": {"tier": "gold", "credit": 5000},
            "campaign:vip:current": {"discount": 20, "expires": "2024-12-31"},
            "campaign:regular:current": {"discount": 5, "expires": "2024-12-31"}
        }

        for key, value in conditional_data.items():
            await cache_with_events.set(key, value)

        # Set up conditional invalidation rules
        await cache_with_events.invalidation_engine.add_conditional_rule(
            condition="event.data.tier in ['platinum', 'gold']",
            action="invalidate_pattern",
            pattern="campaign:vip:*",
            description="Invalidate VIP campaigns when VIP users change"
        )

        await cache_with_events.invalidation_engine.add_conditional_rule(
            condition="event.data.credit_change > 1000",
            action="invalidate_key",
            key_template="user:{entity_id}:recommendations",
            description="Invalidate recommendations for large credit changes"
        )

        # Test VIP user change - should trigger VIP campaign invalidation
        await event_bus.emit(Event(
            type=EventType.USER_UPDATE,
            source="user_service",
            data={
                "entity_type": "user",
                "entity_id": "123",
                "action": "profile_update",
                "tier": "platinum",
                "credit_change": 500,
                "affected_keys": ["user:vip:123:profile"]
            }
        ))

        await asyncio.sleep(1)

        # VIP campaign should be invalidated
        vip_campaign = await cache_with_events.get("campaign:vip:current")
        assert vip_campaign is None

        # Regular campaign should remain
        regular_campaign = await cache_with_events.get("campaign:regular:current")
        assert regular_campaign is not None

        # Test large credit change
        await cache_with_events.set("user:456:recommendations", {"items": ["item1", "item2"]})

        await event_bus.emit(Event(
            type=EventType.USER_UPDATE,
            source="user_service",
            data={
                "entity_type": "user",
                "entity_id": "456",
                "action": "credit_update",
                "tier": "bronze",
                "credit_change": 2000,  # Large change
                "affected_keys": ["user:regular:456:profile"]
            }
        ))

        await asyncio.sleep(1)

        # Recommendations should be invalidated due to large credit change
        recommendations = await cache_with_events.get("user:456:recommendations")
        assert recommendations is None

    async def test_event_replay_and_recovery(self, event_bus, invalidation_engine, cache_with_events):
        """Test event replay for recovery scenarios."""
        # Set up initial data
        recovery_data = {
            "system:config:database": {"host": "db1", "port": 5432},
            "system:config:cache": {"ttl": 3600, "size": "1GB"},
            "system:status:health": {"status": "healthy", "timestamp": time.time()}
        }

        for key, value in recovery_data.items():
            await cache_with_events.set(key, value)

        # Record events for replay
        events_to_replay = [
            Event(
                type=EventType.CONFIG_CHANGE,
                source="admin_panel",
                data={
                    "entity_type": "system",
                    "entity_id": "database",
                    "action": "config_update",
                    "affected_keys": ["system:config:database"]
                }
            ),
            Event(
                type=EventType.CONFIG_CHANGE,
                source="admin_panel",
                data={
                    "entity_type": "system",
                    "entity_id": "cache",
                    "action": "config_update",
                    "affected_keys": ["system:config:cache"]
                }
            )
        ]

        # Emit and record events
        event_ids = []
        for event in events_to_replay:
            event_id = await event_bus.emit(event)
            event_ids.append(event_id)

        await asyncio.sleep(1)

        # Verify initial invalidation
        assert await cache_with_events.get("system:config:database") is None
        assert await cache_with_events.get("system:config:cache") is None

        # Restore data (simulate cache rebuild)
        for key, value in recovery_data.items():
            await cache_with_events.set(key, value)

        # Simulate system restart - replay events
        replay_result = await event_bus.replay_events(
            from_timestamp=datetime.now() - timedelta(minutes=5),
            to_timestamp=datetime.now(),
            event_filter={"type": EventType.CONFIG_CHANGE}
        )

        await asyncio.sleep(1)

        # Verify replay caused re-invalidation
        assert replay_result["events_replayed"] == len(events_to_replay)
        assert await cache_with_events.get("system:config:database") is None
        assert await cache_with_events.get("system:config:cache") is None

        # Health status should remain (not affected by config changes)
        health_status = await cache_with_events.get("system:status:health")
        assert health_status is not None

    async def test_cross_cache_invalidation_coordination(self, event_bus):
        """Test invalidation coordination across multiple cache instances."""
        # Create multiple cache instances
        caches = {}
        for cache_name in ["cache_primary", "cache_secondary", "cache_regional"]:
            backend = MemoryBackend()
            strategy = LRUStrategy(capacity=500)

            # Each cache shares the same event bus for coordination
            invalidation_engine = InvalidationEngine(
                event_bus=event_bus,
                dependency_graph=DependencyGraph(),
                instance_id=cache_name
            )
            await invalidation_engine.initialize()

            cache = Cache(
                backend=backend,
                strategy=strategy,
                name=cache_name,
                invalidation_engine=invalidation_engine
            )
            caches[cache_name] = cache

        # Store same data in all caches
        shared_data = {
            "global:config:api_key": {"key": "abc123", "expires": "2024-12-31"},
            "global:user:session_timeout": {"value": 1800, "unit": "seconds"}
        }

        for cache in caches.values():
            for key, value in shared_data.items():
                await cache.set(key, value)

        # Emit global invalidation event
        await event_bus.emit(Event(
            type=EventType.GLOBAL_INVALIDATION,
            source="config_service",
            data={
                "entity_type": "global_config",
                "action": "update",
                "scope": "all_instances",
                "affected_keys": list(shared_data.keys())
            }
        ))

        await asyncio.sleep(2)

        # Verify all caches invalidated the same keys
        for cache_name, cache in caches.items():
            for key in shared_data.keys():
                result = await cache.get(key)
                assert result is None, f"Key {key} should be invalidated in {cache_name}"

    @pytest.mark.parametrize("invalidation_strategy", [
        InvalidationStrategy.IMMEDIATE,
        InvalidationStrategy.LAZY,
        InvalidationStrategy.TIME_BASED,
        InvalidationStrategy.SMART
    ])
    async def test_invalidation_strategies(self, cache_with_events, event_bus, invalidation_strategy):
        """Test different invalidation strategies."""
        # Configure strategy
        await cache_with_events.invalidation_engine.set_strategy(invalidation_strategy)

        # Set up test data
        strategy_data = {
            f"strategy_test:{invalidation_strategy.value}:key1": {"data": "value1"},
            f"strategy_test:{invalidation_strategy.value}:key2": {"data": "value2"}
        }

        for key, value in strategy_data.items():
            await cache_with_events.set(key, value)

        # Emit invalidation event
        await event_bus.emit(Event(
            type=EventType.DATA_CHANGED,
            source="strategy_test",
            data={
                "entity_type": "test",
                "action": "update",
                "affected_keys": list(strategy_data.keys())
            }
        ))

        # Behavior depends on strategy
        if invalidation_strategy == InvalidationStrategy.IMMEDIATE:
            await asyncio.sleep(0.1)
            # Should be invalidated immediately
            for key in strategy_data.keys():
                result = await cache_with_events.get(key)
                assert result is None

        elif invalidation_strategy == InvalidationStrategy.LAZY:
            # Should remain until next access
            for key in strategy_data.keys():
                result = await cache_with_events.get(key)
                # Lazy invalidation happens on access
                assert result is None  # Invalidated during get()

        elif invalidation_strategy == InvalidationStrategy.TIME_BASED:
            # Should be marked for future invalidation
            await asyncio.sleep(2)  # Wait for time-based invalidation
            for key in strategy_data.keys():
                result = await cache_with_events.get(key)
                assert result is None

        elif invalidation_strategy == InvalidationStrategy.SMART:
            # Should use intelligent decision making
            await asyncio.sleep(1)
            invalidation_stats = await cache_with_events.invalidation_engine.get_strategy_stats()
            assert invalidation_stats["strategy"] == "smart"
            assert invalidation_stats["decisions_made"] > 0

    async def test_invalidation_performance_and_monitoring(self, cache_with_events, event_bus):
        """Test invalidation performance and monitoring metrics."""
        # Set up performance test data
        perf_data = {f"perf:key:{i}": f"value_{i}" for i in range(1000)}

        for key, value in perf_data.items():
            await cache_with_events.set(key, value)

        # Measure invalidation performance
        start_time = time.time()

        # Emit bulk invalidation event
        await event_bus.emit(Event(
            type=EventType.BULK_INVALIDATION,
            source="performance_test",
            data={
                "entity_type": "performance",
                "action": "bulk_update",
                "affected_keys": list(perf_data.keys())
            }
        ))

        await asyncio.sleep(3)  # Allow invalidation to complete

        invalidation_time = time.time() - start_time

        # Verify performance metrics
        performance_stats = await cache_with_events.invalidation_engine.get_performance_stats()

        assert performance_stats["total_invalidations"] >= 1000
        assert performance_stats["average_invalidation_time"] < 0.01  # < 10ms per key
        assert performance_stats["throughput_per_second"] > 100  # > 100 invalidations/second

        # Verify monitoring data
        monitoring_data = await cache_with_events.invalidation_engine.get_monitoring_data()

        assert "invalidation_events" in monitoring_data
        assert "dependency_traversals" in monitoring_data
        assert "error_rate" in monitoring_data
        assert monitoring_data["error_rate"] < 0.01  # < 1% error rate

        # Test error handling and recovery
        # Emit malformed event
        try:
            await event_bus.emit(Event(
                type="INVALID_TYPE",
                source="error_test",
                data={"malformed": "data"}
            ))
        except:
            pass

        # Verify system stability after error
        post_error_stats = await cache_with_events.invalidation_engine.get_performance_stats()
        assert post_error_stats["error_count"] > 0
        assert post_error_stats["system_healthy"] is True