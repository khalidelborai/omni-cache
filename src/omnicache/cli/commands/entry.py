"""
Cache entry management CLI commands.

Implements commands for setting, getting, deleting, and listing cache entries.
"""

import asyncio
import json
import click
import sys
from typing import Optional, List, Any

from omnicache.core.manager import manager
from omnicache.core.exceptions import CacheNotFoundError, CacheError
from omnicache.cli.formatters import format_output, format_entry_list


@click.group()
def entry_group():
    """Cache entry management commands."""
    pass


@entry_group.command('set')
@click.argument('cache_name')
@click.argument('key')
@click.argument('value', required=False)
@click.option('--ttl', type=float, help='Time to live in seconds')
@click.option('--tags', help='Comma-separated tags')
@click.option('--priority', type=float, help='Entry priority (0.0-1.0)')
@click.pass_context
def set_entry(
    ctx: click.Context,
    cache_name: str,
    key: str,
    value: Optional[str],
    ttl: Optional[float],
    tags: Optional[str],
    priority: Optional[float]
) -> None:
    """Set a cache entry value."""

    async def _set_entry():
        nonlocal value
        try:
            # Validate key
            if not key or key.strip() == '':
                click.echo("Error: Key cannot be empty", err=True)
                sys.exit(1)

            # Validate TTL
            if ttl is not None and ttl < 0:
                click.echo("Error: TTL must be positive", err=True)
                sys.exit(1)

            # Validate priority
            if priority is not None and not (0.0 <= priority <= 1.0):
                click.echo("Error: Priority must be between 0.0 and 1.0", err=True)
                sys.exit(1)

            # Get value from stdin if not provided
            if value is None:
                if not sys.stdin.isatty():
                    value = sys.stdin.read().strip()
                else:
                    click.echo("Error: Value must be provided as argument or via stdin", err=True)
                    sys.exit(1)

            # Get cache
            cache = await manager.get_cache(cache_name)
            if not cache:
                click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
                sys.exit(1)

            # Prepare entry options
            options = {}
            if ttl is not None:
                options['ttl'] = ttl

            # Parse tags if provided
            tag_list = None
            if tags:
                tag_list = [tag.strip() for tag in tags.split(',')]
                options['tags'] = tag_list

            if priority is not None:
                options['priority'] = priority

            # Set the entry
            if options:
                await cache.set(key, value, **options)
            else:
                await cache.set(key, value)

            # Output result
            if ctx.obj['quiet']:
                return

            if ctx.obj['format'] == 'json':
                result = {
                    "status": "success",
                    "message": "Entry set successfully",
                    "cache": cache_name,
                    "key": key,
                    "ttl": ttl,
                    "tags": tag_list,
                    "priority": priority
                }
                click.echo(format_output(result, 'json'))
            else:
                click.echo("✓ Entry set successfully")
                if ctx.obj['verbose']:
                    if ttl is not None:
                        click.echo(f"  TTL: {ttl}s")
                    if tags:
                        click.echo(f"  Tags: {tags}")
                    if priority is not None:
                        click.echo(f"  Priority: {priority}")

        except CacheNotFoundError:
            click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error setting entry: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_set_entry())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)


@entry_group.command('get')
@click.argument('cache_name')
@click.argument('key')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), help='Output format')
@click.pass_context
def get_entry(ctx: click.Context, cache_name: str, key: str, format: Optional[str]) -> None:
    """Get a cache entry value."""

    async def _get_entry():
        try:
            # Get cache
            cache = await manager.get_cache(cache_name)
            if not cache:
                click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
                sys.exit(1)

            # Get the entry
            value = await cache.get(key)

            if value is None:
                click.echo(f"Error: Entry \"{key}\" not found", err=True)
                sys.exit(1)

            # Output result
            output_format = format or ctx.obj['format']

            if output_format == 'json':
                # Try to get additional entry metadata
                try:
                    entry = await cache.backend.get_entry(key)
                    entry_data = {
                        "key": key,
                        "value": value,
                        "ttl": getattr(entry, 'ttl', None),
                        "created_at": getattr(entry, 'created_at', None),
                        "access_count": getattr(entry, 'access_count', None),
                        "tags": getattr(entry, 'tags', None),
                        "priority": getattr(entry, 'priority', None)
                    }
                except:
                    entry_data = {"key": key, "value": value}

                click.echo(format_output(entry_data, 'json'))
            elif output_format == 'yaml':
                entry_data = {"key": key, "value": value}
                click.echo(format_output(entry_data, 'yaml'))
            else:
                # Simple text output for table format
                click.echo(str(value))

        except CacheNotFoundError:
            click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error getting entry: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_get_entry())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)


@entry_group.command('delete')
@click.argument('cache_name')
@click.argument('key')
@click.option('--force', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def delete_entry(ctx: click.Context, cache_name: str, key: str, force: bool) -> None:
    """Delete a cache entry."""

    async def _delete_entry():
        try:
            # Get cache
            cache = await manager.get_cache(cache_name)
            if not cache:
                click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
                sys.exit(1)

            # Check if entry exists
            if not await cache.exists(key):
                click.echo(f"Error: Entry \"{key}\" not found", err=True)
                sys.exit(1)

            # Confirmation prompt
            if not force:
                if click.confirm(f"Are you sure you want to delete entry \"{key}\" from cache \"{cache_name}\"?"):
                    confirmed = True
                else:
                    click.echo("Operation cancelled")
                    return
            else:
                confirmed = True

            if confirmed:
                # Delete the entry
                success = await cache.delete(key)

                if success:
                    if not ctx.obj['quiet']:
                        click.echo(f"✓ Entry \"{key}\" deleted successfully")
                else:
                    click.echo(f"Error: Failed to delete entry \"{key}\"", err=True)
                    sys.exit(1)

        except CacheNotFoundError:
            click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error deleting entry: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_delete_entry())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)


@entry_group.command('list')
@click.argument('cache_name')
@click.option('--pattern', help='Filter entries by key pattern')
@click.option('--limit', type=int, help='Maximum number of entries to show')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml', 'keys-only']), help='Output format')
@click.pass_context
def list_entries(
    ctx: click.Context,
    cache_name: str,
    pattern: Optional[str],
    limit: Optional[int],
    format: Optional[str]
) -> None:
    """List cache entries."""

    async def _list_entries():
        try:
            # Get cache
            cache = await manager.get_cache(cache_name)
            if not cache:
                click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
                sys.exit(1)

            # Get all keys
            all_keys = await cache.keys(pattern=pattern)

            # Apply limit
            if limit and len(all_keys) > limit:
                all_keys = all_keys[:limit]

            if not all_keys:
                if ctx.obj['format'] == 'json' or format == 'json':
                    click.echo(format_output([], 'json'))
                else:
                    if not ctx.obj['quiet']:
                        click.echo("No entries found")
                return

            # Build entry list with metadata
            entries = []
            for key in all_keys:
                try:
                    # Get entry details
                    value = await cache.get(key)
                    entry_info = {"key": key, "value": value}

                    # Try to get additional metadata
                    try:
                        entry = await cache.backend.get_entry(key)
                        if entry:
                            entry_info.update({
                                "ttl": getattr(entry, 'ttl', None),
                                "created_at": getattr(entry, 'created_at', None),
                                "access_count": getattr(entry, 'access_count', None),
                                "tags": getattr(entry, 'tags', None),
                                "priority": getattr(entry, 'priority', None)
                            })
                    except:
                        # Fallback to basic info
                        pass

                    entries.append(entry_info)

                except Exception:
                    # Skip entries that can't be read
                    continue

            # Output result
            output_format = format or ctx.obj['format']

            if output_format == 'json':
                click.echo(format_output(entries, 'json'))
            elif output_format == 'yaml':
                click.echo(format_output(entries, 'yaml'))
            elif output_format == 'keys-only':
                click.echo(format_entry_list(entries, keys_only=True))
            else:
                click.echo(format_entry_list(entries))

        except CacheNotFoundError:
            click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error listing entries: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_list_entries())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)


@entry_group.command('exists')
@click.argument('cache_name')
@click.argument('key')
@click.pass_context
def entry_exists(ctx: click.Context, cache_name: str, key: str) -> None:
    """Check if a cache entry exists."""

    async def _entry_exists():
        try:
            # Get cache
            cache = await manager.get_cache(cache_name)
            if not cache:
                click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
                sys.exit(1)

            # Check if entry exists
            exists = await cache.exists(key)

            # Output result
            if ctx.obj['format'] == 'json':
                result = {
                    "cache": cache_name,
                    "key": key,
                    "exists": exists
                }
                click.echo(format_output(result, 'json'))
            else:
                if exists:
                    if not ctx.obj['quiet']:
                        click.echo(f"✓ Entry \"{key}\" exists in cache \"{cache_name}\"")
                else:
                    if not ctx.obj['quiet']:
                        click.echo(f"✗ Entry \"{key}\" does not exist in cache \"{cache_name}\"")
                    sys.exit(1)

        except CacheNotFoundError:
            click.echo(f"Error: Cache \"{cache_name}\" not found", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error checking entry: {e}", err=True)
            if ctx.obj['verbose']:
                raise
            sys.exit(1)

    try:
        asyncio.run(_entry_exists())
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(130)