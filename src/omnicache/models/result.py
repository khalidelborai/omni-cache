"""
Result entity models.

Defines result classes for cache operations like clear, bulk operations,
and administrative actions.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from enum import Enum


class OperationStatus(Enum):
    """Operation status enumeration."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ClearResult:
    """
    Result of a cache clear operation.

    Contains information about entries cleared, patterns matched,
    and operation performance metrics.
    """

    def __init__(
        self,
        cleared_count: int,
        pattern: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        error_count: int = 0,
        errors: Optional[List[str]] = None
    ) -> None:
        """
        Initialize clear operation result.

        Args:
            cleared_count: Number of entries successfully cleared
            pattern: Key pattern used for selective clearing
            tags: Tags used for selective clearing
            error_count: Number of errors encountered
            errors: List of error messages
        """
        self.cleared_count = cleared_count
        self.pattern = pattern
        self.tags = tags or set()
        self.error_count = error_count
        self.errors = errors or []
        self.timestamp = datetime.now()
        self.status = self._determine_status()

    @property
    def success(self) -> bool:
        """Check if operation was successful."""
        return self.status == OperationStatus.SUCCESS

    @property
    def partial_success(self) -> bool:
        """Check if operation had partial success."""
        return self.status == OperationStatus.PARTIAL

    @property
    def failed(self) -> bool:
        """Check if operation failed."""
        return self.status == OperationStatus.FAILED

    def _determine_status(self) -> OperationStatus:
        """Determine operation status based on results."""
        if self.error_count == 0:
            return OperationStatus.SUCCESS
        elif self.cleared_count > 0:
            return OperationStatus.PARTIAL
        else:
            return OperationStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cleared_count": self.cleared_count,
            "pattern": self.pattern,
            "tags": list(self.tags),
            "error_count": self.error_count,
            "errors": self.errors,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success
        }

    def __str__(self) -> str:
        status_emoji = "✓" if self.success else "⚠" if self.partial_success else "✗"
        return f"ClearResult({status_emoji} cleared={self.cleared_count}, errors={self.error_count})"

    def __repr__(self) -> str:
        return (f"<ClearResult(cleared={self.cleared_count}, "
                f"errors={self.error_count}, status={self.status.value})>")


class BulkResult:
    """
    Result of a bulk operation (set, get, delete).

    Contains information about successful and failed operations
    with detailed error tracking.
    """

    def __init__(
        self,
        operation: str,
        total_requested: int,
        successful_count: int = 0,
        failed_count: int = 0,
        errors: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initialize bulk operation result.

        Args:
            operation: Name of the bulk operation
            total_requested: Total number of operations requested
            successful_count: Number of successful operations
            failed_count: Number of failed operations
            errors: Dictionary mapping keys to error messages
        """
        self.operation = operation
        self.total_requested = total_requested
        self.successful_count = successful_count
        self.failed_count = failed_count
        self.errors = errors or {}
        self.timestamp = datetime.now()
        self.status = self._determine_status()

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_requested == 0:
            return 100.0
        return (self.successful_count / self.total_requested) * 100.0

    @property
    def success(self) -> bool:
        """Check if all operations were successful."""
        return self.status == OperationStatus.SUCCESS

    @property
    def partial_success(self) -> bool:
        """Check if some operations were successful."""
        return self.status == OperationStatus.PARTIAL

    @property
    def failed(self) -> bool:
        """Check if all operations failed."""
        return self.status == OperationStatus.FAILED

    def _determine_status(self) -> OperationStatus:
        """Determine operation status based on results."""
        if self.failed_count == 0:
            return OperationStatus.SUCCESS
        elif self.successful_count > 0:
            return OperationStatus.PARTIAL
        else:
            return OperationStatus.FAILED

    def add_success(self, key: str) -> None:
        """Record a successful operation."""
        self.successful_count += 1
        # Remove from errors if it was there
        self.errors.pop(key, None)
        self.status = self._determine_status()

    def add_error(self, key: str, error: str) -> None:
        """Record a failed operation."""
        self.failed_count += 1
        self.errors[key] = error
        self.status = self._determine_status()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operation": self.operation,
            "total_requested": self.total_requested,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "success_rate": self.success_rate,
            "errors": self.errors,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success
        }

    def __str__(self) -> str:
        status_emoji = "✓" if self.success else "⚠" if self.partial_success else "✗"
        return (f"BulkResult({status_emoji} {self.operation}: "
                f"{self.successful_count}/{self.total_requested} successful)")

    def __repr__(self) -> str:
        return (f"<BulkResult(operation='{self.operation}', "
                f"success={self.successful_count}/{self.total_requested})>")


class EvictionResult:
    """
    Result of a cache eviction operation.

    Contains information about entries evicted and the reason
    for eviction.
    """

    def __init__(
        self,
        evicted_keys: List[str],
        reason: str,
        strategy_name: str,
        freed_bytes: int = 0
    ) -> None:
        """
        Initialize eviction result.

        Args:
            evicted_keys: List of keys that were evicted
            reason: Reason for eviction
            strategy_name: Name of strategy that performed eviction
            freed_bytes: Bytes freed by eviction
        """
        self.evicted_keys = evicted_keys
        self.evicted_count = len(evicted_keys)
        self.reason = reason
        self.strategy_name = strategy_name
        self.freed_bytes = freed_bytes
        self.timestamp = datetime.now()

    @property
    def success(self) -> bool:
        """Check if eviction was successful."""
        return self.evicted_count > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "evicted_keys": self.evicted_keys,
            "evicted_count": self.evicted_count,
            "reason": self.reason,
            "strategy_name": self.strategy_name,
            "freed_bytes": self.freed_bytes,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success
        }

    def __str__(self) -> str:
        return (f"EvictionResult(evicted={self.evicted_count}, "
                f"freed={self.freed_bytes}B, reason='{self.reason}')")

    def __repr__(self) -> str:
        return (f"<EvictionResult(count={self.evicted_count}, "
                f"strategy='{self.strategy_name}')>")


class ValidationResult:
    """
    Result of a cache validation operation.

    Contains information about integrity checks, corrupt entries,
    and repair actions taken.
    """

    def __init__(
        self,
        total_checked: int,
        valid_count: int = 0,
        invalid_count: int = 0,
        repaired_count: int = 0,
        removed_count: int = 0,
        errors: Optional[List[str]] = None
    ) -> None:
        """
        Initialize validation result.

        Args:
            total_checked: Total number of entries checked
            valid_count: Number of valid entries
            invalid_count: Number of invalid entries found
            repaired_count: Number of entries repaired
            removed_count: Number of entries removed
            errors: List of validation errors
        """
        self.total_checked = total_checked
        self.valid_count = valid_count
        self.invalid_count = invalid_count
        self.repaired_count = repaired_count
        self.removed_count = removed_count
        self.errors = errors or []
        self.timestamp = datetime.now()

    @property
    def integrity_rate(self) -> float:
        """Get integrity rate as percentage."""
        if self.total_checked == 0:
            return 100.0
        return (self.valid_count / self.total_checked) * 100.0

    @property
    def has_issues(self) -> bool:
        """Check if validation found issues."""
        return self.invalid_count > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_checked": self.total_checked,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "repaired_count": self.repaired_count,
            "removed_count": self.removed_count,
            "integrity_rate": self.integrity_rate,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
            "has_issues": self.has_issues
        }

    def __str__(self) -> str:
        status_emoji = "✓" if not self.has_issues else "⚠"
        return (f"ValidationResult({status_emoji} {self.integrity_rate:.1f}% integrity, "
                f"issues={self.invalid_count})")

    def __repr__(self) -> str:
        return (f"<ValidationResult(checked={self.total_checked}, "
                f"valid={self.valid_count}, invalid={self.invalid_count})>")