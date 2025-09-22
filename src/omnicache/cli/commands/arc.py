"""
ARC (Adaptive Replacement Cache) strategy CLI commands.

Provides comprehensive management of ARC caching strategy including
configuration, monitoring, and optimization.
"""

import asyncio
import click
import json
from typing import Optional, Dict, Any

from omnicache.core.manager import manager
from omnicache.cli.formatters import format_output
from omnicache.core.exceptions import CacheError


@click.group()
def arc_group():
    """
    ARC (Adaptive Replacement Cache) strategy management.

    ARC is an adaptive caching algorithm that dynamically balances between
    recency and frequency of access patterns for optimal performance.
    """
    pass


@arc_group.command()
@click.argument('cache_name')
@click.option('--max-size', type=int, default=1000, help='Maximum cache size')
@click.option('--c-factor', type=float, default=1.0, help='ARC adaptation factor (0.1-2.0)')
@click.option('--namespace', help='Cache namespace')
@click.option('--ttl', type=float, help='Default TTL in seconds')
@click.option('--backend', default='memory', help='Backend type (memory, redis, file)')
@click.pass_context
def create(
    ctx: click.Context,
    cache_name: str,
    max_size: int,
    c_factor: float,
    namespace: Optional[str],
    ttl: Optional[float],
    backend: str
):
    """
    Create a new ARC strategy cache.

    Examples:
        omnicache arc create my_cache --max-size 2000 --c-factor 1.2
        omnicache arc create web_cache --ttl 300 --backend redis
    """
    async def _create():
        try:
            # Validate c_factor
            if not 0.1 <= c_factor <= 2.0:
                raise ValueError("C-factor must be between 0.1 and 2.0")

            kwargs = {
                'backend': backend
            }

            if namespace:
                kwargs['namespace'] = namespace
            if ttl:
                kwargs['default_ttl'] = ttl

            cache = await manager.create_arc_cache(
                cache_name,
                max_size=max_size,
                c_factor=c_factor,
                **kwargs
            )

            # Get initial stats
            stats = await cache.get_statistics()
            cache_info = cache.get_info()

            result = {
                "cache_name": cache_name,
                "strategy": "arc",
                "max_size": max_size,
                "c_factor": c_factor,
                "backend": backend,
                "namespace": namespace,
                "default_ttl": ttl,
                "status": "created",
                "initial_stats": stats.to_dict(),
                "cache_info": cache_info
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2, default=str))
            else:
                click.echo(f"✓ Created ARC cache '{cache_name}'")
                if ctx.obj['verbose']:
                    click.echo(f"  Strategy: ARC (Adaptive Replacement Cache)")
                    click.echo(f"  Max size: {max_size}")
                    click.echo(f"  C-factor: {c_factor}")
                    click.echo(f"  Backend: {backend}")
                    if namespace:
                        click.echo(f"  Namespace: {namespace}")
                    if ttl:
                        click.echo(f"  Default TTL: {ttl}s")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to create ARC cache '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_create())


@arc_group.command()
@click.argument('cache_name')
@click.option('--c-factor', type=float, help='New ARC adaptation factor')
@click.option('--max-size', type=int, help='New maximum cache size')
@click.pass_context
def configure(
    ctx: click.Context,
    cache_name: str,
    c_factor: Optional[float],
    max_size: Optional[int]
):
    """
    Configure ARC strategy parameters for an existing cache.

    Examples:
        omnicache arc configure my_cache --c-factor 1.5
        omnicache arc configure my_cache --max-size 5000 --c-factor 0.8
    """
    async def _configure():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            config_updates = {}

            if c_factor is not None:
                if not 0.1 <= c_factor <= 2.0:
                    raise ValueError("C-factor must be between 0.1 and 2.0")
                config_updates['arc_c_factor'] = c_factor

            if max_size is not None:
                if max_size <= 0:
                    raise ValueError("Max size must be positive")
                config_updates['max_size'] = max_size

            if not config_updates:
                raise ValueError("No configuration parameters provided")

            # Update configuration
            await cache.update_config(config_updates)

            # Get updated stats
            stats = await cache.get_statistics()
            cache_info = cache.get_info()

            result = {
                "cache_name": cache_name,
                "configuration_updated": config_updates,
                "current_stats": stats.to_dict(),
                "cache_info": cache_info
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2, default=str))
            else:
                click.echo(f"✓ Updated ARC configuration for '{cache_name}'")
                if ctx.obj['verbose']:
                    for key, value in config_updates.items():
                        click.echo(f"  {key}: {value}")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to configure ARC cache '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_configure())


@arc_group.command()
@click.argument('cache_name')
@click.option('--watch', '-w', is_flag=True, help='Watch ARC statistics in real-time')
@click.option('--interval', type=float, default=2.0, help='Watch interval in seconds')
@click.pass_context
def stats(
    ctx: click.Context,
    cache_name: str,
    watch: bool,
    interval: float
):
    """
    Show detailed ARC strategy statistics and performance metrics.

    Examples:
        omnicache arc stats my_cache
        omnicache arc stats my_cache --watch --interval 1.0
    """
    async def _stats():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            def display_stats(stats_data):
                if ctx.obj['format'] == 'json':
                    click.echo(json.dumps(stats_data, indent=2, default=str))
                else:
                    strategy_info = stats_data.get('strategy_info', {})

                    click.echo(f"\nARC Statistics for '{cache_name}':")
                    click.echo("=" * 50)

                    # Basic metrics
                    click.echo(f"Hit Ratio: {stats_data.get('hit_ratio', 0):.2%}")
                    click.echo(f"Total Operations: {stats_data.get('total_operations', 0)}")
                    click.echo(f"Current Size: {stats_data.get('current_size', 0)}")
                    click.echo(f"Max Size: {stats_data.get('max_size', 0)}")

                    # ARC-specific metrics
                    if strategy_info:
                        click.echo("\nARC Strategy Details:")
                        click.echo(f"  T1 Size (Recent): {strategy_info.get('t1_size', 0)}")
                        click.echo(f"  T2 Size (Frequent): {strategy_info.get('t2_size', 0)}")
                        click.echo(f"  B1 Size (Recent Ghost): {strategy_info.get('b1_size', 0)}")
                        click.echo(f"  B2 Size (Frequent Ghost): {strategy_info.get('b2_size', 0)}")
                        click.echo(f"  P Value (Adaptation): {strategy_info.get('p_value', 0):.2f}")
                        click.echo(f"  C Factor: {strategy_info.get('c_factor', 1.0):.2f}")

                        # Performance indicators
                        recency_ratio = strategy_info.get('t1_size', 0) / max(stats_data.get('current_size', 1), 1)
                        frequency_ratio = strategy_info.get('t2_size', 0) / max(stats_data.get('current_size', 1), 1)

                        click.echo(f"\nAccess Pattern Analysis:")
                        click.echo(f"  Recency Preference: {recency_ratio:.2%}")
                        click.echo(f"  Frequency Preference: {frequency_ratio:.2%}")

                        if recency_ratio > 0.6:
                            click.echo("  → Pattern: Recent access heavy")
                        elif frequency_ratio > 0.6:
                            click.echo("  → Pattern: Frequent access heavy")
                        else:
                            click.echo("  → Pattern: Balanced access")

            if watch:
                click.echo("Watching ARC statistics (Press Ctrl+C to stop)...")
                try:
                    while True:
                        stats = await cache.get_statistics()
                        stats_data = stats.to_dict()

                        # Clear screen for better display
                        click.clear()
                        display_stats(stats_data)

                        await asyncio.sleep(interval)

                except KeyboardInterrupt:
                    click.echo("\nStopped watching.")
            else:
                stats = await cache.get_statistics()
                stats_data = stats.to_dict()
                display_stats(stats_data)

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to get ARC stats for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_stats())


@arc_group.command()
@click.argument('cache_name')
@click.option('--auto-tune', is_flag=True, help='Enable automatic parameter tuning')
@click.option('--target-hit-ratio', type=float, help='Target hit ratio for optimization')
@click.pass_context
def optimize(
    ctx: click.Context,
    cache_name: str,
    auto_tune: bool,
    target_hit_ratio: Optional[float]
):
    """
    Optimize ARC strategy parameters based on access patterns.

    Examples:
        omnicache arc optimize my_cache --auto-tune
        omnicache arc optimize my_cache --target-hit-ratio 0.85
    """
    async def _optimize():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Get current stats
            current_stats = await cache.get_statistics()
            current_hit_ratio = current_stats.hit_ratio

            optimization_results = {
                "cache_name": cache_name,
                "optimization_type": "arc_strategy",
                "before_optimization": {
                    "hit_ratio": current_hit_ratio,
                    "stats": current_stats.to_dict()
                },
                "changes_made": [],
                "recommendations": []
            }

            if auto_tune:
                # Automatic tuning based on access patterns
                if hasattr(cache.strategy, 'get_adaptation_info'):
                    adaptation_info = cache.strategy.get_adaptation_info()

                    # Analyze access patterns
                    if adaptation_info.get('recent_dominance', 0) > 0.7:
                        # Recent access dominance - adjust c_factor down
                        new_c_factor = max(0.5, adaptation_info.get('c_factor', 1.0) * 0.8)
                        await cache.update_config({'arc_c_factor': new_c_factor})
                        optimization_results["changes_made"].append(f"Reduced c_factor to {new_c_factor:.2f}")

                    elif adaptation_info.get('frequent_dominance', 0) > 0.7:
                        # Frequent access dominance - adjust c_factor up
                        new_c_factor = min(2.0, adaptation_info.get('c_factor', 1.0) * 1.2)
                        await cache.update_config({'arc_c_factor': new_c_factor})
                        optimization_results["changes_made"].append(f"Increased c_factor to {new_c_factor:.2f}")

            if target_hit_ratio:
                # Optimize for target hit ratio
                if target_hit_ratio > current_hit_ratio:
                    # Need to improve hit ratio
                    if current_hit_ratio < 0.5:
                        # Very low hit ratio - increase cache size
                        current_size = cache.get_info().get('max_size', 1000)
                        new_size = int(current_size * 1.5)
                        optimization_results["recommendations"].append(
                            f"Consider increasing cache size to {new_size} for better hit ratio"
                        )
                    else:
                        # Moderate hit ratio - tune strategy
                        optimization_results["recommendations"].append(
                            "Consider adjusting c_factor or analyzing access patterns"
                        )
                else:
                    optimization_results["recommendations"].append(
                        "Current hit ratio meets or exceeds target"
                    )

            # Get final stats
            final_stats = await cache.get_statistics()
            optimization_results["after_optimization"] = {
                "hit_ratio": final_stats.hit_ratio,
                "stats": final_stats.to_dict()
            }

            improvement = final_stats.hit_ratio - current_hit_ratio
            optimization_results["improvement"] = improvement

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(optimization_results, indent=2, default=str))
            else:
                click.echo(f"✓ ARC optimization completed for '{cache_name}'")

                if optimization_results["changes_made"]:
                    click.echo("\nChanges made:")
                    for change in optimization_results["changes_made"]:
                        click.echo(f"  • {change}")

                if optimization_results["recommendations"]:
                    click.echo("\nRecommendations:")
                    for rec in optimization_results["recommendations"]:
                        click.echo(f"  • {rec}")

                if improvement > 0:
                    click.echo(f"\n📈 Hit ratio improved by {improvement:.2%}")
                elif improvement < 0:
                    click.echo(f"\n📉 Hit ratio decreased by {abs(improvement):.2%}")
                else:
                    click.echo("\n📊 Hit ratio unchanged")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to optimize ARC cache '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_optimize())


@arc_group.command()
@click.argument('cache_name')
@click.pass_context
def analyze(ctx: click.Context, cache_name: str):
    """
    Analyze ARC strategy effectiveness and provide insights.

    Examples:
        omnicache arc analyze my_cache
    """
    async def _analyze():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            stats = await cache.get_statistics()
            stats_data = stats.to_dict()
            strategy_info = stats_data.get('strategy_info', {})

            analysis = {
                "cache_name": cache_name,
                "analysis_timestamp": stats_data.get('timestamp'),
                "performance_metrics": {
                    "hit_ratio": stats.hit_ratio,
                    "miss_ratio": 1 - stats.hit_ratio,
                    "total_operations": stats.total_operations,
                    "avg_access_time": stats.average_access_time
                },
                "arc_effectiveness": {},
                "recommendations": [],
                "strengths": [],
                "weaknesses": []
            }

            # Analyze ARC effectiveness
            if strategy_info:
                t1_ratio = strategy_info.get('t1_size', 0) / max(stats_data.get('current_size', 1), 1)
                t2_ratio = strategy_info.get('t2_size', 0) / max(stats_data.get('current_size', 1), 1)

                analysis["arc_effectiveness"] = {
                    "recent_cache_utilization": t1_ratio,
                    "frequent_cache_utilization": t2_ratio,
                    "adaptation_effectiveness": strategy_info.get('p_value', 0),
                    "ghost_list_efficiency": {
                        "b1_ratio": strategy_info.get('b1_size', 0) / max(stats_data.get('max_size', 1), 1),
                        "b2_ratio": strategy_info.get('b2_size', 0) / max(stats_data.get('max_size', 1), 1)
                    }
                }

                # Performance analysis
                if stats.hit_ratio > 0.8:
                    analysis["strengths"].append("Excellent hit ratio performance")
                elif stats.hit_ratio > 0.6:
                    analysis["strengths"].append("Good hit ratio performance")
                else:
                    analysis["weaknesses"].append("Low hit ratio - may need optimization")

                # Pattern analysis
                if t1_ratio > 0.7:
                    analysis["strengths"].append("Effectively caching recent access patterns")
                    analysis["recommendations"].append("Consider increasing T2 capacity for better frequency handling")
                elif t2_ratio > 0.7:
                    analysis["strengths"].append("Effectively caching frequent access patterns")
                    analysis["recommendations"].append("Consider optimizing for recent access patterns")
                else:
                    analysis["strengths"].append("Balanced recent/frequent access handling")

                # Adaptation analysis
                p_value = strategy_info.get('p_value', 0)
                if p_value > 0.8:
                    analysis["recommendations"].append("High P value - consider reducing c_factor")
                elif p_value < 0.2:
                    analysis["recommendations"].append("Low P value - consider increasing c_factor")

            # General recommendations
            if stats.total_operations < 1000:
                analysis["recommendations"].append("Insufficient data for comprehensive analysis - more operations needed")

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(analysis, indent=2, default=str))
            else:
                click.echo(f"\nARC Strategy Analysis for '{cache_name}':")
                click.echo("=" * 60)

                # Performance metrics
                click.echo(f"Hit Ratio: {analysis['performance_metrics']['hit_ratio']:.2%}")
                click.echo(f"Total Operations: {analysis['performance_metrics']['total_operations']}")

                # Strengths
                if analysis["strengths"]:
                    click.echo(f"\n✓ Strengths:")
                    for strength in analysis["strengths"]:
                        click.echo(f"  • {strength}")

                # Weaknesses
                if analysis["weaknesses"]:
                    click.echo(f"\n✗ Areas for Improvement:")
                    for weakness in analysis["weaknesses"]:
                        click.echo(f"  • {weakness}")

                # Recommendations
                if analysis["recommendations"]:
                    click.echo(f"\n💡 Recommendations:")
                    for rec in analysis["recommendations"]:
                        click.echo(f"  • {rec}")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to analyze ARC cache '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_analyze())


# Export the command group
arc = arc_group