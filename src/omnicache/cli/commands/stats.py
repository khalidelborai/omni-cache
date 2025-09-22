"""
Statistics and monitoring CLI commands.

Implements commands for viewing cache statistics, performance metrics, and monitoring.
"""

import asyncio
import click
import sys
from typing import Optional

from omnicache.core.manager import manager
from omnicache.core.exceptions import CacheNotFoundError
from omnicache.cli.formatters import format_output


@click.command()
@click.argument('cache_name')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), help='Output format')
@click.option('--watch', is_flag=True, help='Watch statistics in real-time')
@click.option('--interval', type=float, default=2.0, help='Watch interval in seconds')
@click.pass_context
def stats_group(
    ctx: click.Context,
    cache_name: str,
    format: Optional[str],
    watch: bool,
    interval: float
) -> None:
    """Show cache statistics."""

    async def _get_stats():
        try:
            # Get cache statistics
            stats = await manager.get_cache_stats(cache_name)

            # Output result
            output_format = format or ctx.obj['format']

            if output_format == 'json':
                click.echo(format_output(stats, 'json'))
            elif output_format == 'yaml':
                click.echo(format_output(stats, 'yaml'))
            else:
                click.echo(format_output(stats, 'table', title=f"Statistics for cache '{cache_name}'"))

        except CacheNotFoundError:
            click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error getting statistics: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    async def _watch_stats():
        try:
            while True:
                # Clear screen for better readability
                if not ctx.obj['format'] == 'json':
                    click.clear()

                # Get and display stats
                await _get_stats()

                # Wait for interval
                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            click.echo("\nStopped watching statistics.", err=True)
        except Exception as e:
            click.echo(f"Error in watch mode: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        if watch:
            asyncio.run(_watch_stats())
        else:
            asyncio.run(_get_stats())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)


@click.command()
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), help='Output format')
@click.pass_context
def stats_all(ctx: click.Context, format: Optional[str]) -> None:
    """Show statistics for all caches."""

    async def _get_all_stats():
        try:
            # Get all cache names
            cache_list = manager.list_caches()

            if not cache_list:
                if ctx.obj['format'] == 'json' or format == 'json':
                    click.echo(format_output({}, 'json'))
                else:
                    click.echo("No caches found")
                return

            # Collect statistics for all caches
            all_stats = {}
            for cache_info in cache_list:
                try:
                    cache_name = cache_info['name']
                    stats = await manager.get_cache_stats(cache_name)
                    all_stats[cache_name] = stats
                except Exception as e:
                    # Include error information for failed stats
                    all_stats[cache_info['name']] = {"error": str(e)}

            # Output result
            output_format = format or ctx.obj['format']

            if output_format == 'json':
                click.echo(format_output(all_stats, 'json'))
            elif output_format == 'yaml':
                click.echo(format_output(all_stats, 'yaml'))
            else:
                click.echo(format_output(all_stats, 'table', title="Statistics for all caches"))

        except Exception as e:
            click.echo(f"Error getting statistics: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_get_all_stats())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)


@click.command()
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), help='Output format')
@click.pass_context
def manager_stats(ctx: click.Context, format: Optional[str]) -> None:
    """Show cache manager statistics."""

    async def _get_manager_stats():
        try:
            # Get manager statistics
            stats = manager.get_manager_stats()

            # Output result
            output_format = format or ctx.obj['format']

            if output_format == 'json':
                click.echo(format_output(stats, 'json'))
            elif output_format == 'yaml':
                click.echo(format_output(stats, 'yaml'))
            else:
                click.echo(format_output(stats, 'table', title="Cache Manager Statistics"))

        except Exception as e:
            click.echo(f"Error getting manager statistics: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_get_manager_stats())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)


@click.command()
@click.option('--limit', type=int, default=10, help='Number of recent events to show')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), help='Output format')
@click.option('--event-type', help='Filter by event type')
@click.option('--cache-name', help='Filter by cache name')
@click.pass_context
def event_stats(
    ctx: click.Context,
    limit: int,
    format: Optional[str],
    event_type: Optional[str],
    cache_name: Optional[str]
) -> None:
    """Show recent cache events."""

    async def _get_event_stats():
        try:
            from omnicache.core.events import global_event_bus, EventType, EventFilter

            # Create event filter if needed
            event_filter = None
            if event_type or cache_name:
                filter_kwargs = {}
                if event_type:
                    try:
                        filter_kwargs['event_types'] = {EventType(event_type)}
                    except ValueError:
                        click.echo(f"Error: Invalid event type '{event_type}'", err=True)
                        sys.exit(1)
                if cache_name:
                    filter_kwargs['cache_names'] = {cache_name}
                event_filter = EventFilter(**filter_kwargs)

            # Get event history
            events = global_event_bus.get_history(limit=limit, event_filter=event_filter)

            # Convert events to serializable format
            event_data = []
            for event in events:
                event_dict = event.to_dict()
                event_data.append(event_dict)

            # Output result
            output_format = format or ctx.obj['format']

            if output_format == 'json':
                click.echo(format_output(event_data, 'json'))
            elif output_format == 'yaml':
                click.echo(format_output(event_data, 'yaml'))
            else:
                if not event_data:
                    click.echo("No events found")
                else:
                    click.echo(format_output(event_data, 'table', title="Recent Cache Events"))

        except Exception as e:
            click.echo(f"Error getting event statistics: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_get_event_stats())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)


@click.command()
@click.argument('cache_name')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), help='Output format')
@click.pass_context
def performance_stats(ctx: click.Context, cache_name: str, format: Optional[str]) -> None:
    """Show detailed performance statistics for a cache."""

    async def _get_performance_stats():
        try:
            # Get cache and its components
            cache = await manager.get_cache(cache_name)
            if not cache:
                click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
                sys.exit(1)

            # Collect performance data
            stats = await manager.get_cache_stats(cache_name)

            # Enhanced performance metrics
            performance_data = {
                "cache_name": cache_name,
                "basic_stats": {
                    "hit_rate": stats.get("hit_rate", 0),
                    "miss_rate": 1 - stats.get("hit_rate", 0),
                    "total_operations": stats.get("total_hits", 0) + stats.get("total_misses", 0),
                    "average_access_time": stats.get("average_access_time", 0)
                },
                "strategy_performance": {},
                "backend_performance": {},
                "memory_usage": {
                    "current_entries": stats.get("entry_count", 0),
                    "memory_bytes": stats.get("memory_usage", 0),
                    "max_size": cache.max_size if cache.max_size else "unlimited"
                }
            }

            # Get strategy-specific performance data
            if hasattr(cache.strategy, 'get_priority_stats'):
                performance_data["strategy_performance"] = cache.strategy.get_priority_stats()
            elif hasattr(cache.strategy, 'get_size_stats'):
                performance_data["strategy_performance"] = await cache.strategy.get_size_stats()
            elif hasattr(cache.strategy, 'get_statistics'):
                performance_data["strategy_performance"] = cache.strategy.get_statistics()

            # Get backend-specific performance data
            if hasattr(cache.backend, 'get_performance_stats'):
                performance_data["backend_performance"] = await cache.backend.get_performance_stats()

            # Output result
            output_format = format or ctx.obj['format']

            if output_format == 'json':
                click.echo(format_output(performance_data, 'json'))
            elif output_format == 'yaml':
                click.echo(format_output(performance_data, 'yaml'))
            else:
                click.echo(format_output(performance_data, 'table', title=f"Performance Statistics for '{cache_name}'"))

        except CacheNotFoundError:
            click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error getting performance statistics: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_get_performance_stats())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)