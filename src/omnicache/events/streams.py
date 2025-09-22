"""
Event Stream model for reactive cache invalidation.

This module defines event streaming for real-time cache invalidation,
reactive patterns, and event-driven cache management.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, AsyncIterator, Union
from enum import Enum
import time
import json
import asyncio
from collections import defaultdict, deque
import uuid
import weakref


class EventType(Enum):
    """Types of cache events."""
    CACHE_SET = "cache_set"
    CACHE_GET = "cache_get"
    CACHE_DELETE = "cache_delete"
    CACHE_EXPIRE = "cache_expire"
    CACHE_EVICT = "cache_evict"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_CLEAR = "cache_clear"
    DATA_CHANGED = "data_changed"
    INVALIDATION_REQUEST = "invalidation_request"
    INVALIDATION_COMPLETE = "invalidation_complete"
    SYSTEM_EVENT = "system_event"


class EventPriority(Enum):
    """Priority levels for events."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class StreamStatus(Enum):
    """Status of event streams."""
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class EventMetadata:
    """Metadata for cache events."""
    source: str = ""
    correlation_id: str = ""
    trace_id: str = ""
    user_id: str = ""
    session_id: str = ""
    request_id: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventMetadata':
        """Create from dictionary."""
        return cls(
            source=data.get("source", ""),
            correlation_id=data.get("correlation_id", ""),
            trace_id=data.get("trace_id", ""),
            user_id=data.get("user_id", ""),
            session_id=data.get("session_id", ""),
            request_id=data.get("request_id", ""),
            tags=data.get("tags", {}),
        )


@dataclass
class CacheEvent:
    """
    Cache event for reactive invalidation.

    Represents a single event in the cache system that can trigger
    reactive invalidation and other cache management operations.
    """

    event_id: str
    event_type: EventType
    timestamp: float
    cache_name: str
    key: str

    # Event data
    value: Any = None
    old_value: Any = None
    ttl: Optional[float] = None
    tags: List[str] = field(default_factory=list)

    # Event context
    priority: EventPriority = EventPriority.NORMAL
    metadata: EventMetadata = field(default_factory=EventMetadata)

    # Processing
    processed: bool = False
    processing_started_at: Optional[float] = None
    processing_completed_at: Optional[float] = None
    processing_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization validation."""
        if not self.event_id:
            self.event_id = str(uuid.uuid4())

        if self.timestamp <= 0:
            self.timestamp = time.time()

        if not self.cache_name:
            raise ValueError("Cache name is required")

        if not self.key:
            raise ValueError("Key is required")

    @property
    def age_seconds(self) -> float:
        """Get event age in seconds."""
        return time.time() - self.timestamp

    @property
    def processing_duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.processing_started_at and self.processing_completed_at:
            return self.processing_completed_at - self.processing_started_at
        return None

    @property
    def is_read_event(self) -> bool:
        """Check if this is a read event."""
        return self.event_type in [EventType.CACHE_GET, EventType.CACHE_HIT, EventType.CACHE_MISS]

    @property
    def is_write_event(self) -> bool:
        """Check if this is a write event."""
        return self.event_type in [EventType.CACHE_SET, EventType.CACHE_DELETE, EventType.CACHE_CLEAR]

    @property
    def is_invalidation_event(self) -> bool:
        """Check if this is an invalidation event."""
        return self.event_type in [
            EventType.CACHE_DELETE, EventType.CACHE_EXPIRE, EventType.CACHE_EVICT,
            EventType.INVALIDATION_REQUEST, EventType.DATA_CHANGED
        ]

    def start_processing(self) -> None:
        """Mark event as starting processing."""
        self.processing_started_at = time.time()

    def complete_processing(self, error: Optional[str] = None) -> None:
        """Mark event as completed processing."""
        self.processing_completed_at = time.time()
        self.processed = True

        if error:
            self.processing_errors.append(error)

    def add_error(self, error: str) -> None:
        """Add a processing error."""
        self.processing_errors.append(error)

    def matches_pattern(self, pattern: str) -> bool:
        """Check if event matches a key pattern."""
        import fnmatch
        return fnmatch.fnmatch(self.key, pattern)

    def matches_filter(self, event_filter: Dict[str, Any]) -> bool:
        """Check if event matches filter criteria."""
        # Filter by event type
        if "event_types" in event_filter:
            if self.event_type not in event_filter["event_types"]:
                return False

        # Filter by cache name
        if "cache_names" in event_filter:
            if self.cache_name not in event_filter["cache_names"]:
                return False

        # Filter by key pattern
        if "key_patterns" in event_filter:
            if not any(self.matches_pattern(pattern) for pattern in event_filter["key_patterns"]):
                return False

        # Filter by tags
        if "tags" in event_filter:
            if not any(tag in self.tags for tag in event_filter["tags"]):
                return False

        # Filter by priority
        if "min_priority" in event_filter:
            if self.priority.value < event_filter["min_priority"]:
                return False

        # Filter by age
        if "max_age_seconds" in event_filter:
            if self.age_seconds > event_filter["max_age_seconds"]:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "cache_name": self.cache_name,
            "key": self.key,
            "value": self.value,
            "old_value": self.old_value,
            "ttl": self.ttl,
            "tags": self.tags,
            "priority": self.priority.value,
            "metadata": self.metadata.to_dict(),
            "processed": self.processed,
            "processing_started_at": self.processing_started_at,
            "processing_completed_at": self.processing_completed_at,
            "processing_errors": self.processing_errors,
            "age_seconds": self.age_seconds,
            "processing_duration": self.processing_duration,
            "is_read_event": self.is_read_event,
            "is_write_event": self.is_write_event,
            "is_invalidation_event": self.is_invalidation_event,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEvent':
        """Create event from dictionary representation."""
        metadata = EventMetadata.from_dict(data.get("metadata", {}))

        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=EventType(data["event_type"]),
            timestamp=data["timestamp"],
            cache_name=data["cache_name"],
            key=data["key"],
            value=data.get("value"),
            old_value=data.get("old_value"),
            ttl=data.get("ttl"),
            tags=data.get("tags", []),
            priority=EventPriority(data.get("priority", 2)),
            metadata=metadata,
            processed=data.get("processed", False),
            processing_started_at=data.get("processing_started_at"),
            processing_completed_at=data.get("processing_completed_at"),
            processing_errors=data.get("processing_errors", []),
        )

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'CacheEvent':
        """Create event from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of the event."""
        return f"CacheEvent({self.event_type.value}, {self.cache_name}:{self.key})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"CacheEvent(id='{self.event_id}', type='{self.event_type.value}', "
                f"cache='{self.cache_name}', key='{self.key}')")


# Type alias for event handlers
EventHandler = Callable[[CacheEvent], None]
AsyncEventHandler = Callable[[CacheEvent], asyncio.Task]


@dataclass
class EventSubscription:
    """Subscription to an event stream."""

    subscription_id: str
    handler: Union[EventHandler, AsyncEventHandler]
    event_filter: Dict[str, Any] = field(default_factory=dict)
    is_async: bool = False
    created_at: float = field(default_factory=time.time)
    processed_count: int = 0
    error_count: int = 0
    last_processed_at: Optional[float] = None

    def __post_init__(self):
        """Post-initialization setup."""
        if not self.subscription_id:
            self.subscription_id = str(uuid.uuid4())

    async def handle_event(self, event: CacheEvent) -> bool:
        """
        Handle an event with this subscription.

        Returns:
            True if event was handled successfully
        """
        if not event.matches_filter(self.event_filter):
            return False

        try:
            if self.is_async:
                await self.handler(event)
            else:
                self.handler(event)

            self.processed_count += 1
            self.last_processed_at = time.time()
            return True

        except Exception as e:
            self.error_count += 1
            event.add_error(f"Subscription {self.subscription_id} error: {str(e)}")
            return False


@dataclass
class EventStream:
    """
    Event stream for reactive cache invalidation.

    Manages event publishing, subscription, and processing for
    real-time cache invalidation and reactive patterns.
    """

    name: str = "default"
    description: str = ""

    # Stream configuration
    max_buffer_size: int = 10000
    max_event_age_seconds: int = 3600  # 1 hour
    batch_size: int = 100
    processing_interval: float = 0.1  # 100ms

    # Stream state
    status: StreamStatus = StreamStatus.STARTING
    _event_buffer: deque = field(default_factory=lambda: deque(maxlen=10000))
    _subscriptions: Dict[str, EventSubscription] = field(default_factory=dict)
    _event_handlers: Dict[EventType, List[EventSubscription]] = field(default_factory=lambda: defaultdict(list))

    # Processing
    _processing_task: Optional[asyncio.Task] = None
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Statistics
    total_events_published: int = 0
    total_events_processed: int = 0
    total_processing_errors: int = 0
    created_at: float = field(default_factory=time.time)
    last_event_at: Optional[float] = None

    def __post_init__(self):
        """Post-initialization setup."""
        if not self.name:
            raise ValueError("Stream name is required")

        if self.max_buffer_size <= 0:
            raise ValueError("Max buffer size must be positive")

    @property
    def is_running(self) -> bool:
        """Check if stream is running."""
        return self.status == StreamStatus.ACTIVE

    @property
    def buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self._event_buffer)

    @property
    def subscription_count(self) -> int:
        """Get number of active subscriptions."""
        return len(self._subscriptions)

    async def start(self) -> None:
        """Start the event stream."""
        if self.status == StreamStatus.ACTIVE:
            return

        self.status = StreamStatus.ACTIVE
        self._stop_event.clear()

        # Start processing task
        self._processing_task = asyncio.create_task(self._process_events())

    async def stop(self) -> None:
        """Stop the event stream."""
        if self.status != StreamStatus.ACTIVE:
            return

        self.status = StreamStatus.STOPPED
        self._stop_event.set()

        # Wait for processing task to complete
        if self._processing_task:
            await self._processing_task
            self._processing_task = None

    async def pause(self) -> None:
        """Pause the event stream."""
        if self.status == StreamStatus.ACTIVE:
            self.status = StreamStatus.PAUSED

    async def resume(self) -> None:
        """Resume the event stream."""
        if self.status == StreamStatus.PAUSED:
            self.status = StreamStatus.ACTIVE

    def publish(self, event: CacheEvent) -> None:
        """Publish an event to the stream."""
        if self.status == StreamStatus.STOPPED:
            raise ValueError("Cannot publish to stopped stream")

        # Add to buffer
        self._event_buffer.append(event)

        # Update statistics
        self.total_events_published += 1
        self.last_event_at = time.time()

        # Clean old events
        self._cleanup_old_events()

    def publish_event(self, event_type: EventType, cache_name: str, key: str, **kwargs) -> CacheEvent:
        """Create and publish an event."""
        event = CacheEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            cache_name=cache_name,
            key=key,
            **kwargs
        )

        self.publish(event)
        return event

    def subscribe(self, handler: Union[EventHandler, AsyncEventHandler],
                 event_filter: Optional[Dict[str, Any]] = None,
                 subscription_id: Optional[str] = None) -> str:
        """
        Subscribe to events.

        Args:
            handler: Event handler function
            event_filter: Filter criteria for events
            subscription_id: Optional custom subscription ID

        Returns:
            Subscription ID
        """
        if subscription_id is None:
            subscription_id = str(uuid.uuid4())

        is_async = asyncio.iscoroutinefunction(handler)

        subscription = EventSubscription(
            subscription_id=subscription_id,
            handler=handler,
            event_filter=event_filter or {},
            is_async=is_async,
        )

        self._subscriptions[subscription_id] = subscription

        # Add to event type handlers if filter specifies event types
        if "event_types" in subscription.event_filter:
            for event_type in subscription.event_filter["event_types"]:
                self._event_handlers[event_type].append(subscription)
        else:
            # Subscribe to all event types
            for event_type in EventType:
                self._event_handlers[event_type].append(subscription)

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if subscription was removed
        """
        if subscription_id not in self._subscriptions:
            return False

        subscription = self._subscriptions.pop(subscription_id)

        # Remove from event type handlers
        for handlers in self._event_handlers.values():
            if subscription in handlers:
                handlers.remove(subscription)

        return True

    async def _process_events(self) -> None:
        """Process events from the buffer."""
        while not self._stop_event.is_set():
            try:
                if self.status != StreamStatus.ACTIVE:
                    await asyncio.sleep(self.processing_interval)
                    continue

                # Process a batch of events
                processed_count = 0

                while self._event_buffer and processed_count < self.batch_size:
                    event = self._event_buffer.popleft()
                    await self._process_single_event(event)
                    processed_count += 1

                if processed_count == 0:
                    await asyncio.sleep(self.processing_interval)

            except Exception as e:
                self.total_processing_errors += 1
                self.status = StreamStatus.ERROR
                await asyncio.sleep(self.processing_interval)

    async def _process_single_event(self, event: CacheEvent) -> None:
        """Process a single event."""
        event.start_processing()

        try:
            # Get handlers for this event type
            handlers = self._event_handlers.get(event.event_type, [])

            # Process with all matching handlers
            for subscription in handlers:
                try:
                    await subscription.handle_event(event)
                except Exception as e:
                    event.add_error(f"Handler error: {str(e)}")

            # Mark as completed
            event.complete_processing()
            self.total_events_processed += 1

        except Exception as e:
            event.complete_processing(f"Processing error: {str(e)}")
            self.total_processing_errors += 1

    def _cleanup_old_events(self) -> None:
        """Remove old events from buffer."""
        cutoff_time = time.time() - self.max_event_age_seconds

        while self._event_buffer and self._event_buffer[0].timestamp < cutoff_time:
            self._event_buffer.popleft()

    def get_events(self, limit: int = 100, event_filter: Optional[Dict[str, Any]] = None) -> List[CacheEvent]:
        """Get events from buffer."""
        events = []
        count = 0

        for event in reversed(self._event_buffer):
            if count >= limit:
                break

            if event_filter is None or event.matches_filter(event_filter):
                events.append(event)
                count += 1

        return events

    async def wait_for_event(self, event_filter: Dict[str, Any], timeout: float = 10.0) -> Optional[CacheEvent]:
        """Wait for a specific event."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check existing events
            for event in reversed(self._event_buffer):
                if event.matches_filter(event_filter):
                    return event

            await asyncio.sleep(0.1)

        return None

    def get_subscription_stats(self) -> Dict[str, Any]:
        """Get subscription statistics."""
        stats = {}

        for sub_id, subscription in self._subscriptions.items():
            stats[sub_id] = {
                "processed_count": subscription.processed_count,
                "error_count": subscription.error_count,
                "last_processed_at": subscription.last_processed_at,
                "created_at": subscription.created_at,
                "event_filter": subscription.event_filter,
                "is_async": subscription.is_async,
            }

        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get stream statistics."""
        return {
            "name": self.name,
            "status": self.status.value,
            "buffer_size": self.buffer_size,
            "max_buffer_size": self.max_buffer_size,
            "subscription_count": self.subscription_count,
            "total_events_published": self.total_events_published,
            "total_events_processed": self.total_events_processed,
            "total_processing_errors": self.total_processing_errors,
            "created_at": self.created_at,
            "last_event_at": self.last_event_at,
            "processing_interval": self.processing_interval,
            "batch_size": self.batch_size,
            "is_running": self.is_running,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert stream to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "max_buffer_size": self.max_buffer_size,
            "max_event_age_seconds": self.max_event_age_seconds,
            "batch_size": self.batch_size,
            "processing_interval": self.processing_interval,
            "statistics": self.get_statistics(),
            "subscription_stats": self.get_subscription_stats(),
        }

    def __str__(self) -> str:
        """String representation of the stream."""
        return f"EventStream({self.name}, {self.status.value}, {self.buffer_size} events)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"EventStream(name='{self.name}', status='{self.status.value}', "
                f"buffer_size={self.buffer_size}, subscriptions={self.subscription_count})")


@dataclass
class ReactiveInvalidator:
    """Reactive cache invalidator using event streams."""

    stream: EventStream
    dependency_graph: Any = None  # DependencyGraph from dependency_graph.py

    def __init__(self, stream: EventStream, dependency_graph: Any = None):
        """Initialize reactive invalidator."""
        self.stream = stream
        self.dependency_graph = dependency_graph

        # Subscribe to invalidation events
        self.stream.subscribe(
            self._handle_invalidation_event,
            event_filter={"event_types": [
                EventType.CACHE_DELETE,
                EventType.CACHE_SET,
                EventType.DATA_CHANGED,
                EventType.INVALIDATION_REQUEST,
            ]}
        )

    async def _handle_invalidation_event(self, event: CacheEvent) -> None:
        """Handle invalidation events."""
        if not self.dependency_graph:
            return

        # Create invalidation plan
        root_keys = {event.key}
        plan = self.dependency_graph.create_invalidation_plan(root_keys)

        # Execute invalidation
        for level in plan.levels:
            for key in level:
                # Publish invalidation event for dependent keys
                self.stream.publish_event(
                    event_type=EventType.INVALIDATION_REQUEST,
                    cache_name=event.cache_name,
                    key=key,
                    metadata=EventMetadata(
                        source="reactive_invalidator",
                        correlation_id=event.metadata.correlation_id,
                        trace_id=event.metadata.trace_id,
                    )
                )

        # Publish completion event
        self.stream.publish_event(
            event_type=EventType.INVALIDATION_COMPLETE,
            cache_name=event.cache_name,
            key=event.key,
            value={"invalidated_keys": list(plan.get_all_keys())},
            metadata=EventMetadata(
                source="reactive_invalidator",
                correlation_id=event.metadata.correlation_id,
                trace_id=event.metadata.trace_id,
            )
        )