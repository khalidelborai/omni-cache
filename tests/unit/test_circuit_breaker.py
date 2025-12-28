"""
Unit tests for Circuit Breaker implementation.
"""

import pytest
import asyncio
from omnicache.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpenError,
    circuit_protected,
    CircuitBreakerRegistry,
)


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self):
        """Test circuit starts in closed state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_successful_call_increments_stats(self):
        """Test successful calls are tracked."""
        breaker = CircuitBreaker()

        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"

        stats = breaker.get_stats()
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 0

    @pytest.mark.asyncio
    async def test_failed_calls_increment_failure_count(self):
        """Test failed calls are tracked and increment failure count."""
        breaker = CircuitBreaker()

        async def fail_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await breaker.call(fail_func)

        stats = breaker.get_stats()
        assert stats["failed_calls"] == 1
        assert stats["failure_count"] == 1

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self):
        """Test circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config=config)

        async def fail_func():
            raise ValueError("test error")

        # Fail 3 times
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_rejects_calls_when_open(self):
        """Test circuit rejects calls when open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=60.0)
        breaker = CircuitBreaker(config=config)

        async def fail_func():
            raise ValueError("test error")

        # Trigger open
        with pytest.raises(ValueError):
            await breaker.call(fail_func)

        assert breaker.is_open

        # Now calls should be rejected
        async def success_func():
            return "success"

        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(success_func)

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        breaker = CircuitBreaker(config=config)

        async def fail_func():
            raise ValueError("test error")

        # Trigger open
        with pytest.raises(ValueError):
            await breaker.call(fail_func)

        assert breaker.is_open

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Next call should transition to half-open and be allowed
        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_closes_after_success_threshold_in_half_open(self):
        """Test circuit closes after success threshold in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout=0.1
        )
        breaker = CircuitBreaker(config=config)

        async def fail_func():
            raise ValueError("test error")

        async def success_func():
            return "success"

        # Trigger open
        with pytest.raises(ValueError):
            await breaker.call(fail_func)

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Two successful calls should close the circuit
        await breaker.call(success_func)
        await breaker.call(success_func)

        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_force_open(self):
        """Test force opening the circuit."""
        breaker = CircuitBreaker()
        assert breaker.is_closed

        await breaker.force_open()
        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_force_close(self):
        """Test force closing the circuit."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config=config)

        async def fail_func():
            raise ValueError("test error")

        # Trigger open
        with pytest.raises(ValueError):
            await breaker.call(fail_func)

        assert breaker.is_open

        await breaker.force_close()
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_excluded_exceptions_dont_trigger_failure(self):
        """Test excluded exceptions don't count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            excluded_exceptions=(KeyError,)
        )
        breaker = CircuitBreaker(config=config)

        async def excluded_error():
            raise KeyError("excluded")

        with pytest.raises(KeyError):
            await breaker.call(excluded_error)

        # Circuit should still be closed
        assert breaker.is_closed
        assert breaker.get_stats()["failure_count"] == 0


class TestCircuitBreakerDecorator:
    """Test cases for circuit_protected decorator."""

    @pytest.mark.asyncio
    async def test_decorator_passes_through_result(self):
        """Test decorator passes through successful result."""
        breaker = CircuitBreaker()

        @circuit_protected(breaker)
        async def my_func(x, y):
            return x + y

        result = await my_func(1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_decorator_with_fallback(self):
        """Test decorator uses fallback when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=60.0)
        breaker = CircuitBreaker(config=config)

        async def fallback_func(*args, **kwargs):
            return "fallback"

        @circuit_protected(breaker, fallback=fallback_func)
        async def my_func():
            raise ValueError("error")

        # First call fails and opens circuit
        with pytest.raises(ValueError):
            await my_func()

        # Second call should use fallback
        result = await my_func()
        assert result == "fallback"


class TestCircuitBreakerRegistry:
    """Test cases for CircuitBreakerRegistry."""

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new_breaker(self):
        """Test registry creates new breaker."""
        registry = CircuitBreakerRegistry()
        breaker = await registry.get_or_create("test")

        assert breaker is not None
        assert breaker.name == "test"

    @pytest.mark.asyncio
    async def test_get_or_create_returns_existing(self):
        """Test registry returns existing breaker."""
        registry = CircuitBreakerRegistry()
        breaker1 = await registry.get_or_create("test")
        breaker2 = await registry.get_or_create("test")

        assert breaker1 is breaker2

    @pytest.mark.asyncio
    async def test_get_all_stats(self):
        """Test getting stats for all breakers."""
        registry = CircuitBreakerRegistry()
        await registry.get_or_create("test1")
        await registry.get_or_create("test2")

        stats = registry.get_all_stats()
        assert "test1" in stats
        assert "test2" in stats

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """Test resetting all breakers."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(failure_threshold=1)

        breaker = await registry.get_or_create("test", config)

        # Open the breaker
        await breaker.force_open()
        assert breaker.is_open

        # Reset all
        await registry.reset_all()
        assert breaker.is_closed
