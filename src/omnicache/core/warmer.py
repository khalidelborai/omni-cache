"""
Cache warming and preloading capabilities.

Provides mechanisms to preload cache data on startup or on-demand,
supporting various data sources and warming strategies.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional, AsyncIterator, Union
from enum import Enum
import logging


class WarmupStrategy(Enum):
    """Cache warmup strategies."""
    EAGER = "eager"           # Load all data at once
    PROGRESSIVE = "progressive"  # Load in batches with delays
    LAZY = "lazy"             # Load on first access (marker only)
    PRIORITY = "priority"     # Load high-priority items first


@dataclass
class WarmupConfig:
    """Cache warmup configuration."""
    strategy: WarmupStrategy = WarmupStrategy.PROGRESSIVE
    batch_size: int = 100
    delay_between_batches: float = 0.1
    max_concurrent: int = 10
    timeout: float = 300.0  # Overall timeout in seconds
    on_error: str = "continue"  # "continue", "stop", "retry"
    retry_count: int = 3
    retry_delay: float = 1.0
    log_progress: bool = True


@dataclass
class WarmupResult:
    """Result of cache warmup operation."""
    total_items: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "duration_seconds": self.duration_seconds,
            "success_rate": self.successful / max(self.total_items, 1),
            "errors_count": len(self.errors),
            "errors": self.errors[:10],  # Limit errors in response
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @property
    def is_complete(self) -> bool:
        """Check if warmup is complete."""
        return self.completed_at is not None


@dataclass
class WarmupItem:
    """Item to be warmed in cache."""
    key: str
    value: Any
    ttl: Optional[float] = None
    priority: float = 0.5
    tags: Optional[List[str]] = None


class CacheWarmer:
    """
    Cache warmer for preloading data.

    Supports multiple data sources and warming strategies for
    efficient cache population.
    """

    def __init__(self, config: Optional[WarmupConfig] = None):
        """
        Initialize cache warmer.

        Args:
            config: Warmup configuration
        """
        self.config = config or WarmupConfig()
        self._warmup_tasks: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, WarmupResult] = {}
        self._logger = logging.getLogger(__name__)

    async def warm_from_iterator(
        self,
        cache: Any,
        data_iterator: AsyncIterator[Union[tuple, WarmupItem]],
        cache_name: str = "default"
    ) -> WarmupResult:
        """
        Warm cache from async iterator.

        Args:
            cache: Cache instance
            data_iterator: Async iterator yielding (key, value, ttl) tuples or WarmupItem objects
            cache_name: Name for tracking

        Returns:
            WarmupResult with statistics
        """
        result = WarmupResult(started_at=datetime.now())

        if self.config.strategy == WarmupStrategy.EAGER:
            await self._warm_eager(cache, data_iterator, result)
        elif self.config.strategy == WarmupStrategy.PROGRESSIVE:
            await self._warm_progressive(cache, data_iterator, result)
        elif self.config.strategy == WarmupStrategy.PRIORITY:
            await self._warm_priority(cache, data_iterator, result)
        else:
            # Lazy - just mark as ready
            result.skipped = result.total_items

        result.completed_at = datetime.now()
        result.duration_seconds = (result.completed_at - result.started_at).total_seconds()
        self._results[cache_name] = result

        if self.config.log_progress:
            self._logger.info(
                f"Cache warmup complete for '{cache_name}': "
                f"{result.successful}/{result.total_items} items loaded in {result.duration_seconds:.2f}s"
            )

        return result

    async def _warm_eager(
        self,
        cache: Any,
        data_iterator: AsyncIterator,
        result: WarmupResult
    ) -> None:
        """Warm cache eagerly (all at once)."""
        items = []
        async for item in data_iterator:
            items.append(self._normalize_item(item))
            result.total_items += 1

        # Process all items concurrently
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def set_item(item: WarmupItem):
            async with semaphore:
                await self._set_with_retry(cache, item, result)

        await asyncio.gather(*[set_item(item) for item in items], return_exceptions=True)

    async def _warm_progressive(
        self,
        cache: Any,
        data_iterator: AsyncIterator,
        result: WarmupResult
    ) -> None:
        """Warm cache progressively (in batches)."""
        batch = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def process_batch(batch_items: List[WarmupItem]):
            async def set_item(item: WarmupItem):
                async with semaphore:
                    await self._set_with_retry(cache, item, result)

            await asyncio.gather(*[set_item(item) for item in batch_items], return_exceptions=True)

        async for item in data_iterator:
            batch.append(self._normalize_item(item))
            result.total_items += 1

            if len(batch) >= self.config.batch_size:
                await process_batch(batch)
                batch = []

                if self.config.log_progress and result.total_items % 1000 == 0:
                    self._logger.info(f"Warmup progress: {result.successful}/{result.total_items} items")

                await asyncio.sleep(self.config.delay_between_batches)

        # Process remaining items
        if batch:
            await process_batch(batch)

    async def _warm_priority(
        self,
        cache: Any,
        data_iterator: AsyncIterator,
        result: WarmupResult
    ) -> None:
        """Warm cache by priority (high priority first)."""
        items = []
        async for item in data_iterator:
            items.append(self._normalize_item(item))
            result.total_items += 1

        # Sort by priority (highest first)
        items.sort(key=lambda x: x.priority, reverse=True)

        # Process in priority order using progressive strategy
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        for i in range(0, len(items), self.config.batch_size):
            batch = items[i:i + self.config.batch_size]

            async def set_item(item: WarmupItem):
                async with semaphore:
                    await self._set_with_retry(cache, item, result)

            await asyncio.gather(*[set_item(item) for item in batch], return_exceptions=True)
            await asyncio.sleep(self.config.delay_between_batches)

    def _normalize_item(self, item: Union[tuple, WarmupItem]) -> WarmupItem:
        """Normalize item to WarmupItem."""
        if isinstance(item, WarmupItem):
            return item

        if isinstance(item, tuple):
            if len(item) >= 3:
                return WarmupItem(key=item[0], value=item[1], ttl=item[2])
            elif len(item) >= 2:
                return WarmupItem(key=item[0], value=item[1])
            else:
                raise ValueError(f"Invalid tuple format: {item}")

        raise ValueError(f"Unsupported item type: {type(item)}")

    async def _set_with_retry(
        self,
        cache: Any,
        item: WarmupItem,
        result: WarmupResult
    ) -> None:
        """Set item with retry logic."""
        for attempt in range(self.config.retry_count):
            try:
                if item.ttl:
                    await cache.set(item.key, item.value, ttl=item.ttl)
                else:
                    await cache.set(item.key, item.value)
                result.successful += 1
                return
            except Exception as e:
                if attempt < self.config.retry_count - 1:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    result.failed += 1
                    result.errors.append(f"{item.key}: {str(e)}")

                    if self.config.on_error == "stop":
                        raise

    async def warm_from_dict(
        self,
        cache: Any,
        data: Dict[str, Any],
        default_ttl: Optional[float] = None,
        cache_name: str = "default"
    ) -> WarmupResult:
        """
        Warm cache from dictionary.

        Args:
            cache: Cache instance
            data: Dictionary of key-value pairs
            default_ttl: Default TTL for all entries
            cache_name: Name for tracking

        Returns:
            WarmupResult with statistics
        """
        async def dict_iterator():
            for key, value in data.items():
                yield (key, value, default_ttl)

        return await self.warm_from_iterator(cache, dict_iterator(), cache_name)

    async def warm_from_function(
        self,
        cache: Any,
        keys: List[str],
        loader_func: Callable[[str], Any],
        default_ttl: Optional[float] = None,
        cache_name: str = "default"
    ) -> WarmupResult:
        """
        Warm cache by calling loader function for each key.

        Args:
            cache: Cache instance
            keys: List of keys to warm
            loader_func: Async function that loads data for a key
            default_ttl: Default TTL for entries
            cache_name: Name for tracking

        Returns:
            WarmupResult with statistics
        """
        async def loader_iterator():
            for key in keys:
                try:
                    if asyncio.iscoroutinefunction(loader_func):
                        value = await loader_func(key)
                    else:
                        value = loader_func(key)
                    yield (key, value, default_ttl)
                except Exception as e:
                    self._logger.warning(f"Failed to load key '{key}': {e}")
                    continue

        return await self.warm_from_iterator(cache, loader_iterator(), cache_name)

    async def warm_from_file(
        self,
        cache: Any,
        file_path: str,
        parser: Callable[[str], AsyncIterator[tuple]],
        cache_name: str = "default"
    ) -> WarmupResult:
        """
        Warm cache from file.

        Args:
            cache: Cache instance
            file_path: Path to data file
            parser: Async function that yields (key, value, ttl) tuples
            cache_name: Name for tracking

        Returns:
            WarmupResult with statistics
        """
        return await self.warm_from_iterator(cache, parser(file_path), cache_name)

    async def warm_background(
        self,
        cache: Any,
        data_source: Any,
        cache_name: str = "default"
    ) -> None:
        """
        Start warmup in background.

        Args:
            cache: Cache instance
            data_source: Iterator, dict, or other data source
            cache_name: Name for tracking
        """
        if cache_name in self._warmup_tasks:
            if not self._warmup_tasks[cache_name].done():
                return  # Already warming

        async def warmup_task():
            try:
                if isinstance(data_source, dict):
                    await self.warm_from_dict(cache, data_source, cache_name=cache_name)
                else:
                    await self.warm_from_iterator(cache, data_source, cache_name=cache_name)
            except Exception as e:
                self._logger.error(f"Background warmup failed for '{cache_name}': {e}")

        self._warmup_tasks[cache_name] = asyncio.create_task(warmup_task())

    async def wait_for_warmup(
        self,
        cache_name: str,
        timeout: Optional[float] = None
    ) -> WarmupResult:
        """
        Wait for background warmup to complete.

        Args:
            cache_name: Cache name
            timeout: Maximum wait time in seconds

        Returns:
            WarmupResult (may be incomplete if timed out)
        """
        task = self._warmup_tasks.get(cache_name)
        if not task:
            return WarmupResult()

        try:
            await asyncio.wait_for(task, timeout=timeout or self.config.timeout)
        except asyncio.TimeoutError:
            self._logger.warning(f"Warmup timeout for '{cache_name}'")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        return self._results.get(cache_name, WarmupResult())

    def get_warmup_status(self, cache_name: str) -> Dict[str, Any]:
        """
        Get warmup status for a cache.

        Args:
            cache_name: Cache name

        Returns:
            Status dictionary
        """
        task = self._warmup_tasks.get(cache_name)
        result = self._results.get(cache_name)

        in_progress = task is not None and not task.done()
        completed = task is not None and task.done()

        return {
            "cache_name": cache_name,
            "in_progress": in_progress,
            "completed": completed,
            "result": result.to_dict() if result else None
        }

    def get_all_status(self) -> Dict[str, Any]:
        """Get warmup status for all caches."""
        return {
            name: self.get_warmup_status(name)
            for name in set(list(self._warmup_tasks.keys()) + list(self._results.keys()))
        }

    async def cancel_warmup(self, cache_name: str) -> bool:
        """
        Cancel a running warmup.

        Args:
            cache_name: Cache name

        Returns:
            True if cancelled, False if not running
        """
        task = self._warmup_tasks.get(cache_name)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True
        return False

    async def cancel_all(self) -> int:
        """
        Cancel all running warmups.

        Returns:
            Number of warmups cancelled
        """
        cancelled = 0
        for name in list(self._warmup_tasks.keys()):
            if await self.cancel_warmup(name):
                cancelled += 1
        return cancelled
