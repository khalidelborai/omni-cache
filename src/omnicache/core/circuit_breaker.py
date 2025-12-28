"""
Circuit Breaker pattern implementation for backend resilience.

Provides protection against cascading failures by tracking backend
health and temporarily disabling failing backends.
"""

import asyncio
from enum import Enum
from typing import Callable, Any, Optional, TypeVar, Awaitable
from datetime import datetime
from dataclasses import dataclass, field
from functools import wraps


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation - requests flow through
    OPEN = "open"          # Circuit tripped - requests fail fast
    HALF_OPEN = "half_open"  # Testing recovery - limited requests allowed


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5           # Number of failures before opening
    success_threshold: int = 2           # Successes needed to close from half-open
    timeout: float = 30.0                # Seconds before attempting half-open
    half_open_max_calls: int = 3         # Max concurrent calls in half-open state
    excluded_exceptions: tuple = ()      # Exceptions that don't count as failures


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "success_rate": self.successful_calls / max(self.total_calls, 1),
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "last_state_change": self.last_state_change.isoformat() if self.last_state_change else None,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and rejecting calls."""

    def __init__(self, message: str, retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Circuit tripped after failures, requests fail fast
    - HALF_OPEN: Testing recovery, limited requests allowed

    Usage:
        breaker = CircuitBreaker()

        try:
            result = await breaker.call(some_async_function, arg1, arg2)
        except CircuitBreakerOpenError:
            # Handle circuit open (use fallback, return cached data, etc.)
            pass
    """

    def __init__(
        self,
        name: str = "default",
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name for this circuit breaker (for logging/monitoring)
            config: Configuration options
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[datetime] = None

        self._lock = asyncio.Lock()
        self._stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result from the function

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception from the wrapped function
        """
        async with self._lock:
            self._stats.total_calls += 1

            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self._stats.rejected_calls += 1
                    retry_after = self._get_retry_after()
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is open",
                        retry_after=retry_after
                    )

            # Check half-open call limit
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._stats.rejected_calls += 1
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' half-open call limit reached",
                        retry_after=1.0
                    )
                self._half_open_calls += 1

        # Execute the call outside the lock
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            if not isinstance(e, self.config.excluded_exceptions):
                await self._on_failure()
            raise

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self._stats.successful_calls += 1
            self._stats.last_success_time = datetime.now()
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._reset()

    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self._stats.failed_calls += 1
            self._stats.last_failure_time = datetime.now()
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens the circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.config.timeout

    def _get_retry_after(self) -> float:
        """Get seconds until retry should be attempted."""
        if self._last_failure_time is None:
            return 0.0
        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return max(0.0, self.config.timeout - elapsed)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if self._state != new_state:
            self._state = new_state
            self._stats.state_changes += 1
            self._stats.last_state_change = datetime.now()

            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
                self._success_count = 0

    def _reset(self) -> None:
        """Reset circuit to closed state."""
        self._transition_to(CircuitState.CLOSED)
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

    async def force_open(self) -> None:
        """Force circuit to open state (for manual intervention)."""
        async with self._lock:
            self._transition_to(CircuitState.OPEN)
            self._last_failure_time = datetime.now()

    async def force_close(self) -> None:
        """Force circuit to closed state (for manual intervention)."""
        async with self._lock:
            self._reset()

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            },
            **self._stats.to_dict()
        }


T = TypeVar('T')


def circuit_protected(
    circuit_breaker: CircuitBreaker,
    fallback: Optional[Callable[..., Awaitable[T]]] = None
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator to protect async functions with circuit breaker.

    Args:
        circuit_breaker: Circuit breaker instance to use
        fallback: Optional fallback function to call when circuit is open

    Usage:
        breaker = CircuitBreaker()

        @circuit_protected(breaker)
        async def fetch_data():
            ...

        @circuit_protected(breaker, fallback=get_cached_data)
        async def fetch_with_fallback():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await circuit_breaker.call(func, *args, **kwargs)
            except CircuitBreakerOpenError:
                if fallback is not None:
                    return await fallback(*args, **kwargs)
                raise
        return wrapper
    return decorator


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name, config=config)
            return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_stats(self) -> dict:
        """Get stats for all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }

    async def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.force_close()


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()
