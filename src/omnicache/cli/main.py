"""
Main CLI interface for OmniCache.

Provides the primary command-line interface for cache management and operations.
"""

import asyncio
import click
import json
import sys
from typing import Optional, Dict, Any

from omnicache.core.manager import manager
from omnicache.cli.commands.cache import cache_group
from omnicache.cli.commands.entry import entry_group
from omnicache.cli.commands.stats import stats_group
from omnicache.cli.commands.arc import arc_group
from omnicache.cli.commands.tiers import tiers_group
from omnicache.cli.commands.ml import ml_group
from omnicache.cli.commands.security import security_group
from omnicache.cli.commands.analytics import analytics_group
from omnicache.cli.formatters import format_output
from omnicache.cli.config import load_config, CLIConfig


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Enable quiet mode')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), default='table', help='Output format')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool, config: Optional[str], format: str) -> None:
    """
    OmniCache - High-performance caching library CLI.

    Manage cache instances, entries, configurations, and enterprise features
    through the command line.

    Enterprise Features:
      arc         - ARC (Adaptive Replacement Cache) strategy management
      tiers       - Hierarchical cache tier management
      ml          - Machine Learning-based optimization
      security    - Enterprise security and access control
      analytics   - Advanced analytics and reporting

    Core Features:
      cache       - Basic cache management
      entry       - Cache entry operations
      stats       - Performance statistics
    """
    # Initialize context object
    ctx.ensure_object(dict)

    # Load configuration
    cli_config = load_config(config)

    # Store configuration in context
    ctx.obj['config'] = cli_config
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    ctx.obj['format'] = format

    # Configure output verbosity
    if verbose and not quiet:
        click.echo(f"OmniCache CLI - Verbose mode enabled", err=True)
        if config:
            click.echo(f"Using config file: {config}", err=True)

    # Initialize manager
    async def init_manager():
        try:
            manager.config_path = config
            await manager.initialize()
        except Exception as e:
            if not quiet:
                click.echo(f"Warning: Failed to initialize manager: {e}", err=True)

    # Run initialization - handle both sync and async contexts
    try:
        try:
            # Try to get existing event loop
            loop = asyncio.get_running_loop()
            # If we have a running loop, create a task
            task = loop.create_task(init_manager())
            # Store for potential cleanup
            ctx.obj['init_task'] = task
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            asyncio.run(init_manager())
    except Exception as e:
        if not quiet:
            click.echo(f"Warning: Manager initialization failed: {e}", err=True)
        # Don't exit on init failure, continue with limited functionality


# Add core command groups
cli.add_command(cache_group, name='cache')
cli.add_command(entry_group, name='entry')
cli.add_command(stats_group, name='stats')

# Add enterprise command groups
cli.add_command(arc_group, name='arc')
cli.add_command(tiers_group, name='tiers')
cli.add_command(ml_group, name='ml')
cli.add_command(security_group, name='security')
cli.add_command(analytics_group, name='analytics')


@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show version information."""
    try:
        from omnicache import __version__
        version_info = __version__
    except ImportError:
        version_info = "development"

    if ctx.obj['format'] == 'json':
        output = {"version": version_info}
        click.echo(json.dumps(output, indent=2))
    else:
        click.echo(f"OmniCache version {version_info}")


@cli.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """Check system health and connectivity."""
    async def check_health():
        try:
            # Check manager status
            manager_status = manager.get_manager_stats()

            # Test creating a temporary cache
            test_cache = await manager.create_cache("__health_check__")
            await test_cache.set("test", "value")
            value = await test_cache.get("test")
            await manager.delete_cache("__health_check__")

            health_data = {
                "status": "healthy",
                "manager_initialized": manager_status.get("manager_initialized", False),
                "test_cache_operations": value == "value",
                "registered_caches": manager_status.get("registry_stats", {}).get("total_caches", 0)
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(health_data, indent=2))
            else:
                click.echo("✓ OmniCache is healthy")
                if ctx.obj['verbose']:
                    click.echo(f"  Manager initialized: {health_data['manager_initialized']}")
                    click.echo(f"  Cache operations: {'✓' if health_data['test_cache_operations'] else '✗'}")
                    click.echo(f"  Registered caches: {health_data['registered_caches']}")

        except Exception as e:
            health_data = {
                "status": "unhealthy",
                "error": str(e)
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(health_data, indent=2))
            else:
                click.echo("✗ OmniCache health check failed")
                if not ctx.obj['quiet']:
                    click.echo(f"  Error: {e}")

            sys.exit(1)

    try:
        asyncio.run(check_health())
    except Exception as e:
        if not ctx.obj['quiet']:
            click.echo(f"Health check failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('cache_name')
@click.option('--strategy', type=click.Choice(['arc', 'lru', 'lfu']), default='arc',
              help='Caching strategy')
@click.option('--enable-tiers', is_flag=True, help='Enable hierarchical tiers')
@click.option('--enable-security', is_flag=True, help='Enable security features')
@click.option('--enable-ml', is_flag=True, help='Enable ML optimization')
@click.option('--enable-analytics', is_flag=True, help='Enable analytics tracking')
@click.option('--max-size', type=int, default=10000, help='Maximum cache size')
@click.pass_context
def enterprise(
    ctx: click.Context,
    cache_name: str,
    strategy: str,
    enable_tiers: bool,
    enable_security: bool,
    enable_ml: bool,
    enable_analytics: bool,
    max_size: int
):
    """
    Create an enterprise cache with all advanced features.

    This command creates a cache with enterprise-grade features including
    advanced caching strategies, hierarchical tiers, security, ML optimization,
    and comprehensive analytics.

    Examples:
        omnicache enterprise my_cache --strategy arc --enable-tiers --enable-security
        omnicache enterprise api_cache --enable-ml --enable-analytics --max-size 50000
    """
    async def _create_enterprise():
        try:
            # Build enterprise configuration
            tiers = None
            if enable_tiers:
                tiers = [
                    {
                        "name": "L1",
                        "tier_type": "memory",
                        "capacity": max_size // 10,
                        "default_ttl": 300,
                        "backend_config": {"type": "memory"},
                        "priority": 1
                    },
                    {
                        "name": "L2",
                        "tier_type": "filesystem",
                        "capacity": max_size,
                        "default_ttl": 3600,
                        "backend_config": {"type": "file", "directory": f"/tmp/{cache_name}_l2"},
                        "priority": 2
                    }
                ]

            security_policy = None
            if enable_security:
                security_policy = {
                    'name': f'{cache_name}_security_policy',
                    'description': f'Security policy for {cache_name}',
                    'access_control_level': 'authenticated',
                    'encryption_algorithm': 'aes_256',
                    'require_explicit_permissions': True,
                    'enable_audit_logging': True
                }

            # Create enterprise cache
            cache = await manager.create_enterprise_cache(
                cache_name,
                strategy=strategy,
                tiers=tiers,
                security_policy=security_policy,
                enable_analytics=enable_analytics,
                enable_ml_prefetch=enable_ml,
                max_size=max_size
            )

            # Get initial stats
            stats = await cache.get_statistics()
            cache_info = cache.get_info()

            result = {
                "cache_name": cache_name,
                "type": "enterprise",
                "strategy": strategy,
                "features_enabled": {
                    "hierarchical_tiers": enable_tiers,
                    "security": enable_security,
                    "ml_optimization": enable_ml,
                    "analytics": enable_analytics
                },
                "max_size": max_size,
                "status": "created",
                "initial_stats": stats.to_dict(),
                "cache_info": cache_info
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2, default=str))
            else:
                click.echo(f"✓ Created enterprise cache '{cache_name}'")
                click.echo(f"  Strategy: {strategy.upper()}")
                click.echo(f"  Max Size: {max_size:,}")

                enabled_features = []
                if enable_tiers:
                    enabled_features.append("Hierarchical Tiers")
                if enable_security:
                    enabled_features.append("Security & Encryption")
                if enable_ml:
                    enabled_features.append("ML Optimization")
                if enable_analytics:
                    enabled_features.append("Advanced Analytics")

                if enabled_features:
                    click.echo(f"  Enterprise Features: {', '.join(enabled_features)}")
                else:
                    click.echo(f"  Enterprise Features: Basic configuration")

                if ctx.obj['verbose']:
                    click.echo(f"\nNext steps:")
                    if enable_tiers:
                        click.echo(f"  • Monitor tiers: omnicache tiers stats {cache_name}")
                    if enable_security:
                        click.echo(f"  • Check security: omnicache security status {cache_name}")
                    if enable_ml:
                        click.echo(f"  • Train ML model: omnicache ml train {cache_name}")
                    if enable_analytics:
                        click.echo(f"  • View analytics: omnicache analytics report {cache_name}")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to create enterprise cache '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_create_enterprise())


def run_cli():
    """Entry point for CLI execution."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user.", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    run_cli()