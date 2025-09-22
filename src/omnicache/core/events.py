"""
Event System implementation.

Provides event-driven architecture for cache operations with
publishers, subscribers, and event filtering capabilities.
"""

import asyncio
from typing import Dict, List, Set, Optional, Any, Callable, Union, TYPE_CHECKING
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import weakref

if TYPE_CHECKING:
    from omnicache.models.cache import Cache


class EventType(Enum):
    """Cache event types."""
    CACHE_CREATED = "cache_created"
    CACHE_INITIALIZED = "cache_initialized"
    CACHE_SHUTDOWN = "cache_shutdown"
    CACHE_CLEARED = "cache_cleared"

    ENTRY_SET = "entry_set"
    ENTRY_GET = "entry_get"
    ENTRY_DELETE = "entry_delete"
    ENTRY_UPDATE = "entry_update"
    ENTRY_EXPIRED = "entry_expired"
    ENTRY_EVICTED = "entry_evicted"

    HIT = "cache_hit"
    MISS = "cache_miss"

    STRATEGY_CHANGED = "strategy_changed"
    BACKEND_ERROR = "backend_error"
    CONFIGURATION_CHANGED = "configuration_changed"

    CUSTOM = "custom"


@dataclass
class CacheEvent:
    """
    Cache event data structure.

    Contains all relevant information about a cache event.
    """
    event_type: EventType
    cache_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    key: Optional[str] = None
    value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "cache_name": self.cache_name,
            "timestamp": self.timestamp.isoformat(),
            "key": self.key,
            "value": self.value,
            "metadata": self.metadata,
            "source": self.source
        }


class EventFilter:
    """
    Event filtering capabilities.

    Allows subscribers to filter events based on various criteria.
    """

    def __init__(
        self,
        event_types: Optional[Set[EventType]] = None,
        cache_names: Optional[Set[str]] = None,
        key_patterns: Optional[List[str]] = None,
        custom_filter: Optional[Callable[[CacheEvent], bool]] = None
    ) -> None:
        """
        Initialize event filter.

        Args:
            event_types: Set of event types to include
            cache_names: Set of cache names to include
            key_patterns: List of key patterns to match
            custom_filter: Custom filter function
        """
        self.event_types = event_types
        self.cache_names = cache_names
        self.key_patterns = key_patterns or []
        self.custom_filter = custom_filter

    def matches(self, event: CacheEvent) -> bool:
        """
        Check if event matches filter criteria.

        Args:
            event: Event to check

        Returns:
            True if event matches, False otherwise
        """
        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Check cache name
        if self.cache_names and event.cache_name not in self.cache_names:
            return False

        # Check key patterns
        if self.key_patterns and event.key:
            import fnmatch
            if not any(fnmatch.fnmatch(event.key, pattern) for pattern in self.key_patterns):
                return False

        # Check custom filter
        if self.custom_filter and not self.custom_filter(event):
            return False

        return True


class EventSubscriber:
    """Event subscriber wrapper."""

    def __init__(
        self,
        callback: Callable[[CacheEvent], Union[None, Any]],
        event_filter: Optional[EventFilter] = None,
        async_callback: bool = False
    ) -> None:
        """
        Initialize event subscriber.

        Args:
            callback: Function to call when event occurs
            event_filter: Optional filter for events
            async_callback: Whether callback is async
        """
        self.callback = callback
        self.event_filter = event_filter
        self.async_callback = async_callback
        self.subscription_time = datetime.now()
        self.events_received = 0
        self.last_event_time: Optional[datetime] = None


class EventBus:
    """
    Global event bus for cache events.

    Provides publish/subscribe pattern for cache events with
    filtering, async support, and error handling.
    """

    def __init__(self) -> None:
        """Initialize event bus."""
        self._subscribers: Dict[str, EventSubscriber] = {}
        self._event_history: List[CacheEvent] = []
        self._max_history = 1000
        self._stats = {
            "events_published": 0,
            "events_delivered": 0,
            "subscription_count": 0,
            "error_count": 0
        }
        self._lock = asyncio.Lock()

    async def subscribe(
        self,
        callback: Callable[[CacheEvent], Union[None, Any]],
        subscriber_id: Optional[str] = None,
        event_filter: Optional[EventFilter] = None,
        async_callback: bool = False
    ) -> str:
        """
        Subscribe to cache events.

        Args:
            callback: Function to call when event occurs
            subscriber_id: Optional subscriber ID (auto-generated if None)
            event_filter: Optional filter for events
            async_callback: Whether callback is async

        Returns:
            Subscriber ID
        """
        if subscriber_id is None:
            subscriber_id = f"sub_{len(self._subscribers)}_{datetime.now().timestamp()}"

        async with self._lock:
            subscriber = EventSubscriber(callback, event_filter, async_callback)
            self._subscribers[subscriber_id] = subscriber
            self._stats["subscription_count"] += 1

        return subscriber_id

    async def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Unsubscribe from cache events.

        Args:
            subscriber_id: Subscriber ID to remove

        Returns:
            True if unsubscribed, False if not found
        """
        async with self._lock:
            if subscriber_id in self._subscribers:
                del self._subscribers[subscriber_id]
                return True
            return False

    async def publish(self, event: CacheEvent) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        async with self._lock:
            self._stats["events_published"] += 1

            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

            # Get current subscribers (copy to avoid modification during iteration)
            current_subscribers = dict(self._subscribers)

        # Deliver to subscribers outside of lock
        for subscriber_id, subscriber in current_subscribers.items():
            try:
                # Check filter
                if subscriber.event_filter and not subscriber.event_filter.matches(event):
                    continue

                # Update subscriber stats
                subscriber.events_received += 1
                subscriber.last_event_time = datetime.now()

                # Call subscriber
                if subscriber.async_callback:
                    if asyncio.iscoroutinefunction(subscriber.callback):
                        await subscriber.callback(event)
                    else:
                        await asyncio.get_event_loop().run_in_executor(None, subscriber.callback, event)
                else:
                    subscriber.callback(event)

                self._stats["events_delivered"] += 1

            except Exception as e:
                self._stats["error_count"] += 1
                print(f"Error delivering event to subscriber {subscriber_id}: {e}")

    def get_history(self, limit: Optional[int] = None, event_filter: Optional[EventFilter] = None) -> List[CacheEvent]:
        """
        Get event history.

        Args:
            limit: Maximum number of events to return
            event_filter: Optional filter for events

        Returns:
            List of historical events
        """
        events = self._event_history

        # Apply filter
        if event_filter:
            events = [event for event in events if event_filter.matches(event)]

        # Apply limit
        if limit:
            events = events[-limit:]

        return events

    def get_stats(self) -> Dict[str, Any]:
        """
        Get event bus statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self._stats,
            "active_subscribers": len(self._subscribers),
            "history_size": len(self._event_history),
            "max_history": self._max_history
        }

    def get_subscriber_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all subscribers.

        Returns:
            List of subscriber information
        """
        info = []
        for sub_id, subscriber in self._subscribers.items():
            info.append({
                "id": sub_id,
                "subscription_time": subscriber.subscription_time.isoformat(),
                "events_received": subscriber.events_received,
                "last_event_time": subscriber.last_event_time.isoformat() if subscriber.last_event_time else None,
                "async_callback": subscriber.async_callback,
                "has_filter": subscriber.event_filter is not None
            })
        return info

    async def clear_history(self) -> None:
        """Clear event history."""
        async with self._lock:
            self._event_history.clear()

    async def clear_subscribers(self) -> None:
        """Clear all subscribers."""
        async with self._lock:
            self._subscribers.clear()


class CacheEventEmitter:
    """
    Event emitter for cache instances.

    Allows caches to emit events to the global event bus.
    """

    def __init__(self, cache_name: str, event_bus: Optional[EventBus] = None) -> None:
        """
        Initialize event emitter.

        Args:
            cache_name: Name of the cache
            event_bus: Event bus to use (uses global if None)
        """
        self.cache_name = cache_name
        self.event_bus = event_bus or global_event_bus
        self._enabled = True

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if event emission is enabled."""
        return self._enabled

    async def emit(
        self,
        event_type: EventType,
        key: Optional[str] = None,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> None:
        """
        Emit a cache event.

        Args:
            event_type: Type of event
            key: Optional cache key
            value: Optional value
            metadata: Optional metadata
            source: Optional source identifier
        """
        if not self._enabled:
            return

        event = CacheEvent(
            event_type=event_type,
            cache_name=self.cache_name,
            key=key,
            value=value,
            metadata=metadata or {},
            source=source
        )

        await self.event_bus.publish(event)

    # Convenience methods for common events
    async def emit_cache_created(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit cache created event."""
        await self.emit(EventType.CACHE_CREATED, metadata=metadata)

    async def emit_cache_initialized(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit cache initialized event."""
        await self.emit(EventType.CACHE_INITIALIZED, metadata=metadata)

    async def emit_entry_set(self, key: str, value: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit entry set event."""
        await self.emit(EventType.ENTRY_SET, key=key, value=value, metadata=metadata)

    async def emit_entry_get(self, key: str, value: Any = None, hit: bool = True) -> None:
        """Emit entry get event."""
        event_type = EventType.HIT if hit else EventType.MISS
        await self.emit(event_type, key=key, value=value)

    async def emit_entry_delete(self, key: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit entry delete event."""
        await self.emit(EventType.ENTRY_DELETE, key=key, metadata=metadata)

    async def emit_entry_expired(self, key: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit entry expired event."""
        await self.emit(EventType.ENTRY_EXPIRED, key=key, metadata=metadata)

    async def emit_entry_evicted(self, key: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit entry evicted event."""
        await self.emit(EventType.ENTRY_EVICTED, key=key, metadata=metadata)


# Global event bus instance
global_event_bus = EventBus()


# Convenience functions
async def subscribe_to_events(
    callback: Callable[[CacheEvent], Union[None, Any]],
    event_types: Optional[Set[EventType]] = None,
    cache_names: Optional[Set[str]] = None,
    async_callback: bool = False
) -> str:
    """Subscribe to cache events with optional filtering."""
    event_filter = EventFilter(event_types=event_types, cache_names=cache_names) if (event_types or cache_names) else None
    return await global_event_bus.subscribe(callback, event_filter=event_filter, async_callback=async_callback)


async def unsubscribe_from_events(subscriber_id: str) -> bool:
    """Unsubscribe from cache events."""
    return await global_event_bus.unsubscribe(subscriber_id)


def get_event_history(limit: Optional[int] = 100) -> List[CacheEvent]:
    """Get recent event history."""
    return global_event_bus.get_history(limit=limit)