"""
Cache management CLI commands.

Implements commands for creating, listing, configuring, and managing cache instances.
"""

import asyncio
import json
import click
import sys
from typing import Optional, Dict, Any

from omnicache.core.manager import manager
from omnicache.core.exceptions import CacheError, CacheNotFoundError
from omnicache.cli.formatters import format_output, format_cache_list, format_cache_info
from omnicache.cli.config import get_cache_defaults


@click.group()
def cache_group():
    """Cache management commands."""
    pass


@cache_group.command('create')
@click.argument('name')
@click.option('--strategy', type=click.Choice(['memory', 'lru', 'lfu', 'ttl', 'size', 'priority']),
              help='Eviction strategy')
@click.option('--backend', type=click.Choice(['memory', 'redis', 'filesystem']),
              help='Storage backend')
@click.option('--max-size', type=int, help='Maximum number of entries')
@click.option('--default-ttl', type=float, help='Default TTL in seconds')
@click.option('--namespace', help='Cache namespace')
@click.option('--config', help='Backend configuration as JSON string')
@click.pass_context
def create_cache(
    ctx: click.Context,
    name: str,
    strategy: Optional[str],
    backend: Optional[str],
    max_size: Optional[int],
    default_ttl: Optional[float],
    namespace: Optional[str],
    config: Optional[str]
) -> None:
    """Create a new cache instance."""

    async def _create_cache():
        try:
            # Get defaults from CLI config
            cli_config = ctx.obj['config']
            cache_config = get_cache_defaults(cli_config)

            # Override with command-line options
            if strategy:
                cache_config['strategy'] = strategy
            if backend:
                cache_config['backend'] = backend
            if max_size is not None:
                cache_config['max_size'] = max_size
            if default_ttl is not None:
                cache_config['default_ttl'] = default_ttl
            if namespace:
                cache_config['namespace'] = namespace

            # Parse backend configuration
            if config:
                try:
                    backend_config = json.loads(config)
                    cache_config['backend_config'] = backend_config
                except json.JSONDecodeError as e:
                    click.echo(f"Error: Invalid JSON in --config: {e}", err=True)
                    sys.exit(1)

            # Create the cache
            cache = await manager.create_cache(name, cache_config)

            # Output result
            if ctx.obj['quiet']:
                return

            if ctx.obj['format'] == 'json':
                result = {
                    "status": "success",
                    "message": f"Cache \"{name}\" created successfully",
                    "cache_info": cache.get_info()
                }
                click.echo(format_output(result, 'json'))
            else:
                click.echo(f"✓ Cache \"{name}\" created successfully")
                if ctx.obj['verbose']:
                    click.echo(f"  Strategy: {cache.strategy}")
                    click.echo(f"  Backend: {cache.backend}")
                    if cache.max_size:
                        click.echo(f"  Max Size: {cache.max_size}")
                    if cache.default_ttl:
                        click.echo(f"  Default TTL: {cache.default_ttl}s")

        except CacheError as e:
            if 'already exists' in str(e):
                click.echo(f"Error: Cache \"{name}\" already exists", err=True)
            else:
                click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error creating cache: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_create_cache())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)


@cache_group.command('list')
@click.option('--namespace', help='Filter by namespace')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), help='Output format')
@click.option('--clear-all', is_flag=True, hidden=True, help='Clear all caches (for testing)')
@click.pass_context
def list_caches(
    ctx: click.Context,
    namespace: Optional[str],
    format: Optional[str],
    clear_all: bool
) -> None:
    """List all cache instances."""

    async def _list_caches():
        try:
            # Handle clear-all flag (for testing)
            if clear_all:
                cache_list = manager.list_caches()
                for cache_info in cache_list:
                    try:
                        await manager.delete_cache(cache_info['name'])
                    except:
                        pass
                if not ctx.obj['quiet']:
                    click.echo("All caches cleared")
                return

            # Get cache list
            cache_list = manager.list_caches()

            # Filter by namespace if specified
            if namespace:
                cache_list = [
                    cache for cache in cache_list
                    if cache.get('namespace') == namespace
                ]

            # Handle empty result
            if not cache_list:
                if ctx.obj['format'] == 'json' or format == 'json':
                    click.echo(format_output([], 'json'))
                else:
                    if not ctx.obj['quiet']:
                        click.echo("No caches found")
                return

            # Output result
            output_format = format or ctx.obj['format']

            if output_format == 'json':
                click.echo(format_output(cache_list, 'json'))
            elif output_format == 'yaml':
                click.echo(format_output(cache_list, 'yaml'))
            else:
                click.echo(format_cache_list(cache_list))

        except Exception as e:
            click.echo(f"Error listing caches: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_list_caches())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)


@cache_group.command('info')
@click.argument('name')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), help='Output format')
@click.pass_context
def cache_info(ctx: click.Context, name: str, format: Optional[str]) -> None:
    """Get detailed information about a cache."""

    async def _cache_info():
        try:
            # Get cache information
            cache_info = await manager.get_cache_stats(name)

            # Output result
            output_format = format or ctx.obj['format']

            if output_format == 'json':
                click.echo(format_output(cache_info, 'json'))
            elif output_format == 'yaml':
                click.echo(format_output(cache_info, 'yaml'))
            else:
                click.echo(format_cache_info(cache_info))

        except CacheNotFoundError:
            click.echo(f"Error: Cache \"{name}\" not found", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error getting cache info: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_cache_info())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)


@cache_group.command('delete')
@click.argument('name')
@click.option('--force', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def delete_cache(ctx: click.Context, name: str, force: bool) -> None:
    """Delete a cache instance."""

    async def _delete_cache():
        try:
            # Check if cache exists
            cache = await manager.get_cache(name)
            if not cache:
                click.echo(f"Error: Cache \"{name}\" not found", err=True)
                sys.exit(1)

            # Confirmation prompt
            if not force:
                if click.confirm(f"Are you sure you want to delete cache \"{name}\"?"):
                    confirmed = True
                else:
                    click.echo("Operation cancelled")
                    return
            else:
                confirmed = True

            if confirmed:
                # Delete the cache
                success = await manager.delete_cache(name)

                if success:
                    if not ctx.obj['quiet']:
                        click.echo(f"✓ Cache \"{name}\" deleted successfully")
                else:
                    click.echo(f"Error: Failed to delete cache \"{name}\"", err=True)
                    sys.exit(1)

        except CacheNotFoundError:
            click.echo(f"Error: Cache \"{name}\" not found", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error deleting cache: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_delete_cache())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)


@cache_group.command('clear')
@click.argument('name')
@click.option('--pattern', help='Clear entries matching pattern')
@click.option('--force', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def clear_cache(ctx: click.Context, name: str, pattern: Optional[str], force: bool) -> None:
    """Clear entries from a cache."""

    async def _clear_cache():
        try:
            # Confirmation prompt
            if not force:
                if pattern:
                    message = f"Are you sure you want to clear entries matching \"{pattern}\" from cache \"{name}\"?"
                else:
                    message = f"Are you sure you want to clear all entries from cache \"{name}\"?"

                if click.confirm(message):
                    confirmed = True
                else:
                    click.echo("Operation cancelled")
                    return
            else:
                confirmed = True

            if confirmed:
                # Clear the cache
                cleared_count = await manager.clear_cache(name, pattern=pattern)

                if not ctx.obj['quiet']:
                    if pattern:
                        click.echo(f"✓ Cleared {cleared_count} entries matching \"{pattern}\" from cache \"{name}\"")
                    else:
                        click.echo(f"✓ Cleared {cleared_count} entries from cache \"{name}\"")

        except CacheNotFoundError:
            click.echo(f"Error: Cache \"{name}\" not found", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error clearing cache: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_clear_cache())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)