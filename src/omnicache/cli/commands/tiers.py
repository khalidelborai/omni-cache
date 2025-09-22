"""
Hierarchical cache tiers CLI commands.

Provides management of multi-tier caching architectures with automatic
promotion and demotion between cache tiers based on access patterns.
"""

import asyncio
import click
import json
from typing import List, Dict, Any, Optional

from omnicache.core.manager import manager
from omnicache.models.tier import CacheTier, TierType
from omnicache.cli.formatters import format_output
from omnicache.core.exceptions import CacheError


@click.group()
def tiers_group():
    """
    Hierarchical cache tier management.

    Manage multi-tier cache architectures with L1 (memory), L2 (SSD),
    and L3 (disk/network) tiers for optimal performance and cost efficiency.
    """
    pass


@tiers_group.command()
@click.argument('cache_name')
@click.option('--config-file', type=click.Path(exists=True), help='JSON configuration file for tiers')
@click.option('--l1-size', type=int, default=1000, help='L1 (memory) cache size')
@click.option('--l2-size', type=int, default=10000, help='L2 (SSD) cache size')
@click.option('--l3-size', type=int, help='L3 (disk/network) cache size')
@click.option('--l1-ttl', type=float, default=300, help='L1 default TTL in seconds')
@click.option('--l2-ttl', type=float, default=3600, help='L2 default TTL in seconds')
@click.option('--l3-ttl', type=float, help='L3 default TTL in seconds')
@click.option('--promotion-threshold', type=float, default=0.8, help='Hit ratio threshold for promotion')
@click.option('--demotion-threshold', type=float, default=0.2, help='Hit ratio threshold for demotion')
@click.pass_context
def create(
    ctx: click.Context,
    cache_name: str,
    config_file: Optional[str],
    l1_size: int,
    l2_size: int,
    l3_size: Optional[int],
    l1_ttl: float,
    l2_ttl: float,
    l3_ttl: Optional[float],
    promotion_threshold: float,
    demotion_threshold: float
):
    """
    Create a hierarchical cache with multiple tiers.

    Examples:
        omnicache tiers create web_cache --l1-size 1000 --l2-size 10000
        omnicache tiers create api_cache --config-file tiers.json
        omnicache tiers create data_cache --l1-size 500 --l2-size 5000 --l3-size 50000
    """
    async def _create():
        try:
            if config_file:
                # Load configuration from file
                with open(config_file, 'r') as f:
                    tier_configs = json.load(f)
            else:
                # Build configuration from command line options
                tier_configs = [
                    {
                        "name": "L1",
                        "tier_type": "memory",
                        "max_size": l1_size,
                        "default_ttl": l1_ttl,
                        "backend_config": {"type": "memory"},
                        "priority": 1
                    },
                    {
                        "name": "L2",
                        "tier_type": "ssd",
                        "max_size": l2_size,
                        "default_ttl": l2_ttl,
                        "backend_config": {"type": "file", "directory": f"/tmp/{cache_name}_l2"},
                        "priority": 2
                    }
                ]

                if l3_size:
                    tier_configs.append({
                        "name": "L3",
                        "tier_type": "disk",
                        "max_size": l3_size,
                        "default_ttl": l3_ttl or 86400,  # Default 24 hours
                        "backend_config": {"type": "file", "directory": f"/tmp/{cache_name}_l3"},
                        "priority": 3
                    })

            # Create hierarchical cache
            cache = await manager.create_hierarchical_cache(
                cache_name,
                tiers=tier_configs,
                promotion_threshold=promotion_threshold,
                demotion_threshold=demotion_threshold
            )

            # Get initial stats
            stats = await cache.get_statistics()
            cache_info = cache.get_info()

            result = {
                "cache_name": cache_name,
                "type": "hierarchical",
                "tiers": tier_configs,
                "promotion_threshold": promotion_threshold,
                "demotion_threshold": demotion_threshold,
                "status": "created",
                "initial_stats": stats.to_dict(),
                "cache_info": cache_info
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2, default=str))
            else:
                click.echo(f"✓ Created hierarchical cache '{cache_name}'")
                if ctx.obj['verbose']:
                    click.echo(f"  Tiers: {len(tier_configs)}")
                    for i, tier in enumerate(tier_configs):
                        click.echo(f"    {tier['name']} ({tier['tier_type']}): {tier['max_size']} items")
                    click.echo(f"  Promotion threshold: {promotion_threshold}")
                    click.echo(f"  Demotion threshold: {demotion_threshold}")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to create hierarchical cache '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_create())


@tiers_group.command()
@click.argument('cache_name')
@click.option('--tier-name', help='Specific tier to show stats for')
@click.option('--watch', '-w', is_flag=True, help='Watch tier statistics in real-time')
@click.option('--interval', type=float, default=2.0, help='Watch interval in seconds')
@click.pass_context
def stats(
    ctx: click.Context,
    cache_name: str,
    tier_name: Optional[str],
    watch: bool,
    interval: float
):
    """
    Show hierarchical cache tier statistics and distribution.

    Examples:
        omnicache tiers stats web_cache
        omnicache tiers stats web_cache --tier-name L1
        omnicache tiers stats web_cache --watch --interval 1.0
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
                    hierarchy_info = stats_data.get('hierarchy_info', {})

                    if tier_name:
                        # Show specific tier stats
                        tier_stats = hierarchy_info.get('tiers', {}).get(tier_name)
                        if not tier_stats:
                            click.echo(f"✗ Tier '{tier_name}' not found")
                            return

                        click.echo(f"\nTier '{tier_name}' Statistics:")
                        click.echo("=" * 40)
                        click.echo(f"Hit Ratio: {tier_stats.get('hit_ratio', 0):.2%}")
                        click.echo(f"Current Size: {tier_stats.get('current_size', 0)}")
                        click.echo(f"Max Size: {tier_stats.get('max_size', 0)}")
                        click.echo(f"Operations: {tier_stats.get('operations', 0)}")
                        click.echo(f"Evictions: {tier_stats.get('evictions', 0)}")
                        click.echo(f"Promotions: {tier_stats.get('promotions', 0)}")
                        click.echo(f"Demotions: {tier_stats.get('demotions', 0)}")
                    else:
                        # Show all tiers overview
                        click.echo(f"\nHierarchical Cache Statistics for '{cache_name}':")
                        click.echo("=" * 60)

                        # Overall metrics
                        click.echo(f"Overall Hit Ratio: {stats_data.get('hit_ratio', 0):.2%}")
                        click.echo(f"Total Operations: {stats_data.get('total_operations', 0)}")
                        click.echo(f"Average Access Time: {stats_data.get('average_access_time', 0):.3f}ms")

                        # Tier breakdown
                        tiers = hierarchy_info.get('tiers', {})
                        if tiers:
                            click.echo(f"\nTier Breakdown:")
                            click.echo("-" * 60)
                            click.echo(f"{'Tier':<8} {'Hit%':<8} {'Size':<12} {'Ops':<12} {'Prom':<8} {'Demo':<8}")
                            click.echo("-" * 60)

                            for tier_name, tier_data in tiers.items():
                                hit_ratio = tier_data.get('hit_ratio', 0)
                                size_info = f"{tier_data.get('current_size', 0)}/{tier_data.get('max_size', 0)}"
                                ops = tier_data.get('operations', 0)
                                promotions = tier_data.get('promotions', 0)
                                demotions = tier_data.get('demotions', 0)

                                click.echo(f"{tier_name:<8} {hit_ratio:<8.1%} {size_info:<12} {ops:<12} {promotions:<8} {demotions:<8}")

                        # Distribution analysis
                        if len(tiers) > 1:
                            l1_ratio = tiers.get('L1', {}).get('hit_ratio', 0)
                            l2_ratio = tiers.get('L2', {}).get('hit_ratio', 0)

                            click.echo(f"\nAccess Pattern Analysis:")
                            if l1_ratio > 0.8:
                                click.echo("  → Excellent L1 performance - hot data well cached")
                            elif l1_ratio > 0.5:
                                click.echo("  → Good L1 performance")
                            else:
                                click.echo("  → Consider increasing L1 size or adjusting promotion")

                            if len(tiers) > 2:
                                l3_ratio = tiers.get('L3', {}).get('hit_ratio', 0)
                                total_l2_l3 = l2_ratio + l3_ratio
                                if total_l2_l3 > 0.9:
                                    click.echo("  → Excellent overall cache efficiency")

            if watch:
                click.echo("Watching hierarchical cache statistics (Press Ctrl+C to stop)...")
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
                click.echo(f"✗ Failed to get tier stats for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_stats())


@tiers_group.command()
@click.argument('cache_name')
@click.argument('tier_name')
@click.option('--max-size', type=int, help='New maximum size for the tier')
@click.option('--ttl', type=float, help='New default TTL for the tier')
@click.option('--backend-config', help='New backend configuration (JSON string)')
@click.pass_context
def configure_tier(
    ctx: click.Context,
    cache_name: str,
    tier_name: str,
    max_size: Optional[int],
    ttl: Optional[float],
    backend_config: Optional[str]
):
    """
    Configure a specific tier in a hierarchical cache.

    Examples:
        omnicache tiers configure-tier web_cache L1 --max-size 2000
        omnicache tiers configure-tier web_cache L2 --ttl 7200
    """
    async def _configure_tier():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            config_updates = {}

            if max_size is not None:
                config_updates['max_size'] = max_size

            if ttl is not None:
                config_updates['default_ttl'] = ttl

            if backend_config:
                try:
                    config_updates['backend_config'] = json.loads(backend_config)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON for backend-config")

            if not config_updates:
                raise ValueError("No configuration parameters provided")

            # Update tier configuration
            if hasattr(cache, 'update_tier_config'):
                await cache.update_tier_config(tier_name, config_updates)
            else:
                raise ValueError("Cache does not support hierarchical tier configuration")

            # Get updated stats
            stats = await cache.get_statistics()
            hierarchy_info = stats.to_dict().get('hierarchy_info', {})
            tier_info = hierarchy_info.get('tiers', {}).get(tier_name, {})

            result = {
                "cache_name": cache_name,
                "tier_name": tier_name,
                "configuration_updated": config_updates,
                "tier_stats": tier_info
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2, default=str))
            else:
                click.echo(f"✓ Updated tier '{tier_name}' configuration for cache '{cache_name}'")
                if ctx.obj['verbose']:
                    for key, value in config_updates.items():
                        click.echo(f"  {key}: {value}")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name, "tier_name": tier_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to configure tier '{tier_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_configure_tier())


@tiers_group.command()
@click.argument('cache_name')
@click.option('--promotion-threshold', type=float, help='New promotion threshold')
@click.option('--demotion-threshold', type=float, help='New demotion threshold')
@click.option('--auto-balance', is_flag=True, help='Enable automatic tier balancing')
@click.pass_context
def optimize(
    ctx: click.Context,
    cache_name: str,
    promotion_threshold: Optional[float],
    demotion_threshold: Optional[float],
    auto_balance: bool
):
    """
    Optimize hierarchical cache tier thresholds and balancing.

    Examples:
        omnicache tiers optimize web_cache --promotion-threshold 0.9
        omnicache tiers optimize web_cache --auto-balance
    """
    async def _optimize():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Get current stats for analysis
            current_stats = await cache.get_statistics()
            hierarchy_info = current_stats.to_dict().get('hierarchy_info', {})

            optimization_results = {
                "cache_name": cache_name,
                "optimization_type": "hierarchical_tiers",
                "before_optimization": {
                    "overall_hit_ratio": current_stats.hit_ratio,
                    "tier_stats": hierarchy_info.get('tiers', {})
                },
                "changes_made": [],
                "recommendations": []
            }

            # Apply manual threshold updates
            config_updates = {}
            if promotion_threshold is not None:
                if not 0.0 <= promotion_threshold <= 1.0:
                    raise ValueError("Promotion threshold must be between 0.0 and 1.0")
                config_updates['promotion_threshold'] = promotion_threshold
                optimization_results["changes_made"].append(f"Updated promotion threshold to {promotion_threshold}")

            if demotion_threshold is not None:
                if not 0.0 <= demotion_threshold <= 1.0:
                    raise ValueError("Demotion threshold must be between 0.0 and 1.0")
                config_updates['demotion_threshold'] = demotion_threshold
                optimization_results["changes_made"].append(f"Updated demotion threshold to {demotion_threshold}")

            if config_updates:
                await cache.update_config(config_updates)

            # Auto-balancing analysis
            if auto_balance:
                tiers = hierarchy_info.get('tiers', {})

                # Analyze tier performance
                if 'L1' in tiers and 'L2' in tiers:
                    l1_hit_ratio = tiers['L1'].get('hit_ratio', 0)
                    l2_hit_ratio = tiers['L2'].get('hit_ratio', 0)
                    l1_utilization = tiers['L1'].get('current_size', 0) / max(tiers['L1'].get('max_size', 1), 1)

                    # L1 optimization recommendations
                    if l1_hit_ratio < 0.5 and l1_utilization > 0.9:
                        optimization_results["recommendations"].append(
                            "Consider increasing L1 cache size - high utilization but low hit ratio"
                        )

                    if l1_hit_ratio > 0.9 and l1_utilization < 0.5:
                        optimization_results["recommendations"].append(
                            "L1 cache may be oversized - consider reducing size"
                        )

                    # Promotion/demotion threshold recommendations
                    if l1_hit_ratio < 0.3:
                        optimization_results["recommendations"].append(
                            "Consider lowering promotion threshold to move hot data to L1 faster"
                        )

                    if l2_hit_ratio < 0.3:
                        optimization_results["recommendations"].append(
                            "Poor L2 performance - consider increasing L2 size or adjusting demotion threshold"
                        )

            # Get final stats
            final_stats = await cache.get_statistics()
            final_hierarchy = final_stats.to_dict().get('hierarchy_info', {})

            optimization_results["after_optimization"] = {
                "overall_hit_ratio": final_stats.hit_ratio,
                "tier_stats": final_hierarchy.get('tiers', {})
            }

            improvement = final_stats.hit_ratio - current_stats.hit_ratio
            optimization_results["improvement"] = improvement

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(optimization_results, indent=2, default=str))
            else:
                click.echo(f"✓ Hierarchical cache optimization completed for '{cache_name}'")

                if optimization_results["changes_made"]:
                    click.echo("\nChanges made:")
                    for change in optimization_results["changes_made"]:
                        click.echo(f"  • {change}")

                if optimization_results["recommendations"]:
                    click.echo("\nRecommendations:")
                    for rec in optimization_results["recommendations"]:
                        click.echo(f"  • {rec}")

                if improvement > 0:
                    click.echo(f"\n📈 Overall hit ratio improved by {improvement:.2%}")
                elif improvement < 0:
                    click.echo(f"\n📉 Overall hit ratio decreased by {abs(improvement):.2%}")
                else:
                    click.echo("\n📊 Overall hit ratio unchanged")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to optimize hierarchical cache '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_optimize())


@tiers_group.command()
@click.argument('cache_name')
@click.argument('source_tier')
@click.argument('target_tier')
@click.argument('key')
@click.pass_context
def promote(
    ctx: click.Context,
    cache_name: str,
    source_tier: str,
    target_tier: str,
    key: str
):
    """
    Manually promote a cache entry between tiers.

    Examples:
        omnicache tiers promote web_cache L2 L1 user:123
        omnicache tiers promote web_cache L3 L2 config:app
    """
    async def _promote():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            if hasattr(cache, 'promote_entry'):
                success = await cache.promote_entry(key, source_tier, target_tier)
            else:
                raise ValueError("Cache does not support manual tier promotion")

            result = {
                "cache_name": cache_name,
                "operation": "promote",
                "key": key,
                "source_tier": source_tier,
                "target_tier": target_tier,
                "success": success
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2))
            else:
                if success:
                    click.echo(f"✓ Promoted key '{key}' from {source_tier} to {target_tier}")
                else:
                    click.echo(f"✗ Failed to promote key '{key}'")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to promote entry: {e}", err=True)
            ctx.exit(1)

    asyncio.run(_promote())


@tiers_group.command()
@click.argument('cache_name')
@click.pass_context
def analyze(ctx: click.Context, cache_name: str):
    """
    Analyze hierarchical cache effectiveness and tier distribution.

    Examples:
        omnicache tiers analyze web_cache
    """
    async def _analyze():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            stats = await cache.get_statistics()
            stats_data = stats.to_dict()
            hierarchy_info = stats_data.get('hierarchy_info', {})

            analysis = {
                "cache_name": cache_name,
                "analysis_timestamp": stats_data.get('timestamp'),
                "overall_performance": {
                    "hit_ratio": stats.hit_ratio,
                    "total_operations": stats.total_operations,
                    "avg_access_time": stats.average_access_time
                },
                "tier_analysis": {},
                "hierarchy_effectiveness": {},
                "recommendations": []
            }

            # Analyze each tier
            tiers = hierarchy_info.get('tiers', {})
            for tier_name, tier_data in tiers.items():
                tier_analysis = {
                    "hit_ratio": tier_data.get('hit_ratio', 0),
                    "utilization": tier_data.get('current_size', 0) / max(tier_data.get('max_size', 1), 1),
                    "operations": tier_data.get('operations', 0),
                    "promotions": tier_data.get('promotions', 0),
                    "demotions": tier_data.get('demotions', 0),
                    "performance_score": 0
                }

                # Calculate performance score
                hit_ratio_score = tier_data.get('hit_ratio', 0) * 0.6
                utilization_score = min(tier_analysis["utilization"] * 0.3, 0.3)
                movement_efficiency = 0.1 if (tier_data.get('promotions', 0) + tier_data.get('demotions', 0)) > 0 else 0

                tier_analysis["performance_score"] = hit_ratio_score + utilization_score + movement_efficiency
                analysis["tier_analysis"][tier_name] = tier_analysis

            # Hierarchy effectiveness analysis
            if len(tiers) >= 2:
                l1_hit = tiers.get('L1', {}).get('hit_ratio', 0)
                total_ops = sum(tier.get('operations', 0) for tier in tiers.values())

                analysis["hierarchy_effectiveness"] = {
                    "l1_efficiency": l1_hit,
                    "tier_distribution": {tier: tier_data.get('operations', 0) / max(total_ops, 1)
                                       for tier, tier_data in tiers.items()},
                    "promotion_activity": sum(tier.get('promotions', 0) for tier in tiers.values()),
                    "demotion_activity": sum(tier.get('demotions', 0) for tier in tiers.values())
                }

                # Generate recommendations
                if l1_hit < 0.4:
                    analysis["recommendations"].append("Low L1 hit ratio - consider increasing L1 size or lowering promotion threshold")

                if l1_hit > 0.95:
                    analysis["recommendations"].append("Very high L1 hit ratio - L2/L3 may be underutilized")

                promotion_rate = analysis["hierarchy_effectiveness"]["promotion_activity"] / max(total_ops, 1)
                if promotion_rate < 0.01:
                    analysis["recommendations"].append("Low promotion activity - consider lowering promotion threshold")
                elif promotion_rate > 0.1:
                    analysis["recommendations"].append("High promotion activity - may indicate thrashing")

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(analysis, indent=2, default=str))
            else:
                click.echo(f"\nHierarchical Cache Analysis for '{cache_name}':")
                click.echo("=" * 60)

                # Overall performance
                click.echo(f"Overall Hit Ratio: {analysis['overall_performance']['hit_ratio']:.2%}")
                click.echo(f"Total Operations: {analysis['overall_performance']['total_operations']}")

                # Tier analysis
                click.echo(f"\nTier Performance Analysis:")
                click.echo("-" * 60)
                for tier_name, tier_analysis in analysis["tier_analysis"].items():
                    click.echo(f"\n{tier_name} Tier:")
                    click.echo(f"  Hit Ratio: {tier_analysis['hit_ratio']:.2%}")
                    click.echo(f"  Utilization: {tier_analysis['utilization']:.1%}")
                    click.echo(f"  Performance Score: {tier_analysis['performance_score']:.2f}/1.0")

                # Recommendations
                if analysis["recommendations"]:
                    click.echo(f"\n💡 Optimization Recommendations:")
                    for rec in analysis["recommendations"]:
                        click.echo(f"  • {rec}")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to analyze hierarchical cache '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_analyze())


# Export the command group
tiers = tiers_group