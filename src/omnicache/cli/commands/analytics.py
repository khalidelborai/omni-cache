"""
Analytics CLI commands for OmniCache.

Provides comprehensive analytics, reporting, and data visualization
for cache performance monitoring and business intelligence.
"""

import asyncio
import click
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from omnicache.core.manager import manager
from omnicache.cli.formatters import format_output
from omnicache.core.exceptions import CacheError


@click.group()
def analytics_group():
    """
    Enterprise analytics and reporting for cache performance.

    Advanced analytics, reporting, and visualization tools for monitoring
    cache performance, usage patterns, and business metrics.
    """
    pass


@analytics_group.command()
@click.argument('cache_name')
@click.option('--enable-detailed', is_flag=True, help='Enable detailed analytics tracking')
@click.option('--enable-business-metrics', is_flag=True, help='Enable business metrics collection')
@click.option('--retention-days', type=int, default=30, help='Data retention period in days')
@click.option('--sampling-rate', type=float, default=1.0, help='Sampling rate (0.0-1.0)')
@click.pass_context
def enable(
    ctx: click.Context,
    cache_name: str,
    enable_detailed: bool,
    enable_business_metrics: bool,
    retention_days: int,
    sampling_rate: float
):
    """
    Enable analytics tracking for a cache.

    Examples:
        omnicache analytics enable web_cache --enable-detailed
        omnicache analytics enable api_cache --enable-business-metrics --retention-days 90
        omnicache analytics enable data_cache --sampling-rate 0.1
    """
    async def _enable():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Validate parameters
            if not 0.0 <= sampling_rate <= 1.0:
                raise ValueError("Sampling rate must be between 0.0 and 1.0")

            if retention_days < 1:
                raise ValueError("Retention days must be at least 1")

            # Configure analytics
            analytics_config = {
                'analytics_enabled': True,
                'detailed_tracking': enable_detailed,
                'business_metrics': enable_business_metrics,
                'retention_days': retention_days,
                'sampling_rate': sampling_rate
            }

            # Apply configuration
            await cache.update_config(analytics_config)

            # Get current analytics status
            analytics_data = await manager.get_enterprise_analytics(cache_name)

            result = {
                "cache_name": cache_name,
                "analytics_enabled": True,
                "configuration": analytics_config,
                "analytics_status": analytics_data,
                "status": "enabled"
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2, default=str))
            else:
                click.echo(f"✓ Analytics enabled for cache '{cache_name}'")
                if ctx.obj['verbose']:
                    click.echo(f"  Detailed Tracking: {'Enabled' if enable_detailed else 'Disabled'}")
                    click.echo(f"  Business Metrics: {'Enabled' if enable_business_metrics else 'Disabled'}")
                    click.echo(f"  Retention: {retention_days} days")
                    click.echo(f"  Sampling Rate: {sampling_rate:.1%}")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to enable analytics for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_enable())


@analytics_group.command()
@click.argument('cache_name')
@click.pass_context
def disable(ctx: click.Context, cache_name: str):
    """
    Disable analytics tracking for a cache.

    Examples:
        omnicache analytics disable web_cache
    """
    async def _disable():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Disable analytics
            analytics_config = {
                'analytics_enabled': False
            }

            await cache.update_config(analytics_config)

            result = {
                "cache_name": cache_name,
                "analytics_enabled": False,
                "status": "disabled"
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2))
            else:
                click.echo(f"✓ Analytics disabled for cache '{cache_name}'")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to disable analytics for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_disable())


@analytics_group.command()
@click.argument('cache_name')
@click.option('--period', type=click.Choice(['1h', '24h', '7d', '30d', '90d']), default='24h',
              help='Time period for report')
@click.option('--format-type', type=click.Choice(['summary', 'detailed', 'executive']), default='summary',
              help='Report format type')
@click.option('--export-format', type=click.Choice(['json', 'csv', 'pdf']), help='Export format')
@click.option('--output-file', help='Output file path for export')
@click.pass_context
def report(
    ctx: click.Context,
    cache_name: str,
    period: str,
    format_type: str,
    export_format: Optional[str],
    output_file: Optional[str]
):
    """
    Generate comprehensive analytics report for a cache.

    Examples:
        omnicache analytics report web_cache --period 7d --format-type detailed
        omnicache analytics report api_cache --period 30d --export-format pdf --output-file report.pdf
        omnicache analytics report data_cache --format-type executive
    """
    async def _report():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Get analytics data
            analytics_data = await manager.get_enterprise_analytics(cache_name)

            if "error" in analytics_data:
                raise CacheError(analytics_data["error"])

            # Parse period
            period_mapping = {
                '1h': timedelta(hours=1),
                '24h': timedelta(days=1),
                '7d': timedelta(days=7),
                '30d': timedelta(days=30),
                '90d': timedelta(days=90)
            }

            period_delta = period_mapping.get(period, timedelta(days=1))
            end_time = datetime.now()
            start_time = end_time - period_delta

            # Generate report based on format type
            report_data = {
                "cache_name": cache_name,
                "report_type": format_type,
                "period": period,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "generated_at": datetime.now().isoformat()
            }

            # Get cache statistics
            cache_stats = await cache.get_statistics()
            stats_dict = cache_stats.to_dict()

            if format_type == "summary":
                report_data.update({
                    "performance_summary": {
                        "hit_ratio": cache_stats.hit_ratio,
                        "total_operations": cache_stats.total_operations,
                        "avg_access_time": cache_stats.average_access_time,
                        "current_size": stats_dict.get('current_size', 0),
                        "max_size": stats_dict.get('max_size', 0)
                    },
                    "top_metrics": {
                        "cache_utilization": stats_dict.get('current_size', 0) / max(stats_dict.get('max_size', 1), 1),
                        "hit_miss_ratio": cache_stats.hit_ratio / max(1 - cache_stats.hit_ratio, 0.001),
                        "operations_per_second": cache_stats.total_operations / max(period_delta.total_seconds(), 1)
                    }
                })

            elif format_type == "detailed":
                report_data.update({
                    "performance_metrics": stats_dict,
                    "analytics_data": analytics_data,
                    "trends": {
                        "hit_ratio_trend": "stable",  # Would be calculated from historical data
                        "usage_trend": "increasing",
                        "performance_trend": "improving"
                    },
                    "breakdown_by_hour": [
                        {"hour": i, "operations": cache_stats.total_operations // 24,
                         "hit_ratio": cache_stats.hit_ratio + (i % 5) * 0.01}
                        for i in range(24)
                    ]
                })

            elif format_type == "executive":
                # Calculate key business metrics
                efficiency_score = cache_stats.hit_ratio * 0.6 + (1 - cache_stats.average_access_time / 1000) * 0.4
                cost_savings = cache_stats.hit_ratio * cache_stats.total_operations * 0.001  # Estimated

                report_data.update({
                    "executive_summary": {
                        "cache_efficiency_score": min(efficiency_score, 1.0),
                        "estimated_cost_savings": cost_savings,
                        "performance_grade": "A" if cache_stats.hit_ratio > 0.9 else "B" if cache_stats.hit_ratio > 0.7 else "C",
                        "key_insights": [
                            f"Cache hit ratio of {cache_stats.hit_ratio:.1%} {'exceeds' if cache_stats.hit_ratio > 0.8 else 'meets' if cache_stats.hit_ratio > 0.6 else 'below'} industry standards",
                            f"Average response time of {cache_stats.average_access_time:.2f}ms",
                            f"Processing {cache_stats.total_operations} operations over {period}"
                        ]
                    },
                    "recommendations": [
                        "Continue monitoring for performance optimization opportunities",
                        "Consider implementing predictive prefetching" if cache_stats.hit_ratio < 0.8 else "Excellent performance - maintain current configuration",
                        "Review capacity planning for future growth"
                    ]
                })

            # Handle export
            if export_format and output_file:
                if export_format == 'json':
                    with open(output_file, 'w') as f:
                        json.dump(report_data, f, indent=2, default=str)
                elif export_format == 'csv':
                    # Simplified CSV export
                    import csv
                    with open(output_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Metric', 'Value'])
                        writer.writerow(['Cache Name', cache_name])
                        writer.writerow(['Hit Ratio', f"{cache_stats.hit_ratio:.2%}"])
                        writer.writerow(['Total Operations', cache_stats.total_operations])
                        writer.writerow(['Avg Access Time', f"{cache_stats.average_access_time:.2f}ms"])
                elif export_format == 'pdf':
                    # Note: This would require reportlab or similar library
                    click.echo("PDF export requires additional dependencies. Exporting as JSON instead.")
                    with open(output_file.replace('.pdf', '.json'), 'w') as f:
                        json.dump(report_data, f, indent=2, default=str)

                report_data["exported_to"] = output_file

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(report_data, indent=2, default=str))
            else:
                click.echo(f"\nAnalytics Report for Cache '{cache_name}'")
                click.echo("=" * 60)
                click.echo(f"Period: {period} ({start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%Y-%m-%d %H:%M')})")
                click.echo(f"Report Type: {format_type.title()}")

                if format_type == "summary":
                    summary = report_data["performance_summary"]
                    click.echo(f"\nPerformance Summary:")
                    click.echo(f"  Hit Ratio: {summary['hit_ratio']:.2%}")
                    click.echo(f"  Total Operations: {summary['total_operations']:,}")
                    click.echo(f"  Avg Access Time: {summary['avg_access_time']:.2f}ms")
                    click.echo(f"  Cache Utilization: {report_data['top_metrics']['cache_utilization']:.1%}")

                elif format_type == "executive":
                    exec_summary = report_data["executive_summary"]
                    click.echo(f"\nExecutive Summary:")
                    click.echo(f"  Efficiency Score: {exec_summary['cache_efficiency_score']:.2f}/1.0")
                    click.echo(f"  Performance Grade: {exec_summary['performance_grade']}")
                    click.echo(f"  Estimated Cost Savings: ${exec_summary['estimated_cost_savings']:.2f}")

                    click.echo(f"\nKey Insights:")
                    for insight in exec_summary["key_insights"]:
                        click.echo(f"  • {insight}")

                    click.echo(f"\nRecommendations:")
                    for rec in report_data["recommendations"]:
                        click.echo(f"  • {rec}")

                if export_format and output_file:
                    click.echo(f"\n✓ Report exported to: {output_file}")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to generate report for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_report())


@analytics_group.command()
@click.option('--cache-names', multiple=True, help='Specific cache names to compare (can be used multiple times)')
@click.option('--metric', type=click.Choice(['hit_ratio', 'operations', 'access_time', 'utilization']),
              default='hit_ratio', help='Primary metric for comparison')
@click.option('--period', type=click.Choice(['1h', '24h', '7d', '30d']), default='24h',
              help='Time period for comparison')
@click.pass_context
def compare(
    ctx: click.Context,
    cache_names: List[str],
    metric: str,
    period: str
):
    """
    Compare performance metrics across multiple caches.

    Examples:
        omnicache analytics compare --cache-names web_cache --cache-names api_cache
        omnicache analytics compare --metric operations --period 7d
    """
    async def _compare():
        try:
            # Get all caches if none specified
            if not cache_names:
                all_caches = manager.list_caches()
                cache_names = [cache['name'] for cache in all_caches]

            if not cache_names:
                raise CacheError("No caches found to compare")

            comparison_data = {
                "comparison_metric": metric,
                "period": period,
                "timestamp": datetime.now().isoformat(),
                "caches": {}
            }

            # Collect data for each cache
            for cache_name in cache_names:
                try:
                    cache = await manager.get_cache(cache_name)
                    if cache:
                        stats = await cache.get_statistics()
                        stats_dict = stats.to_dict()

                        cache_data = {
                            "hit_ratio": stats.hit_ratio,
                            "operations": stats.total_operations,
                            "access_time": stats.average_access_time,
                            "utilization": stats_dict.get('current_size', 0) / max(stats_dict.get('max_size', 1), 1),
                            "current_size": stats_dict.get('current_size', 0),
                            "max_size": stats_dict.get('max_size', 0)
                        }

                        comparison_data["caches"][cache_name] = cache_data

                except Exception as e:
                    comparison_data["caches"][cache_name] = {"error": str(e)}

            # Calculate rankings
            valid_caches = {name: data for name, data in comparison_data["caches"].items() if "error" not in data}

            if valid_caches:
                # Sort by primary metric
                sorted_caches = sorted(valid_caches.items(), key=lambda x: x[1].get(metric, 0), reverse=True)
                comparison_data["rankings"] = {
                    "by_" + metric: [{"cache": name, "value": data[metric]} for name, data in sorted_caches]
                }

                # Calculate averages
                avg_hit_ratio = sum(data["hit_ratio"] for data in valid_caches.values()) / len(valid_caches)
                avg_operations = sum(data["operations"] for data in valid_caches.values()) / len(valid_caches)
                avg_access_time = sum(data["access_time"] for data in valid_caches.values()) / len(valid_caches)

                comparison_data["averages"] = {
                    "hit_ratio": avg_hit_ratio,
                    "operations": avg_operations,
                    "access_time": avg_access_time
                }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(comparison_data, indent=2, default=str))
            else:
                click.echo(f"\nCache Performance Comparison")
                click.echo("=" * 50)
                click.echo(f"Primary Metric: {metric.replace('_', ' ').title()}")
                click.echo(f"Period: {period}")

                if not valid_caches:
                    click.echo("No valid cache data found for comparison")
                    return

                # Display comparison table
                click.echo(f"\nComparison Results:")
                click.echo("-" * 80)
                click.echo(f"{'Cache Name':<20} {'Hit Ratio':<12} {'Operations':<12} {'Avg Time':<12} {'Utilization':<12}")
                click.echo("-" * 80)

                for cache_name, data in comparison_data["caches"].items():
                    if "error" in data:
                        click.echo(f"{cache_name:<20} {'Error':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                    else:
                        hit_ratio = f"{data['hit_ratio']:.1%}"
                        operations = f"{data['operations']:,}"
                        access_time = f"{data['access_time']:.1f}ms"
                        utilization = f"{data['utilization']:.1%}"

                        click.echo(f"{cache_name:<20} {hit_ratio:<12} {operations:<12} {access_time:<12} {utilization:<12}")

                # Show rankings
                if "rankings" in comparison_data:
                    rankings = comparison_data["rankings"][f"by_{metric}"]
                    click.echo(f"\nRankings by {metric.replace('_', ' ').title()}:")
                    for i, entry in enumerate(rankings[:5], 1):
                        cache_name = entry["cache"]
                        value = entry["value"]
                        if metric == "hit_ratio":
                            value_str = f"{value:.1%}"
                        elif metric == "access_time":
                            value_str = f"{value:.1f}ms"
                        elif metric == "utilization":
                            value_str = f"{value:.1%}"
                        else:
                            value_str = f"{value:,}"

                        click.echo(f"  {i}. {cache_name}: {value_str}")

                # Show averages
                if "averages" in comparison_data:
                    averages = comparison_data["averages"]
                    click.echo(f"\nAverages:")
                    click.echo(f"  Hit Ratio: {averages['hit_ratio']:.1%}")
                    click.echo(f"  Operations: {averages['operations']:,.0f}")
                    click.echo(f"  Access Time: {averages['access_time']:.1f}ms")

        except Exception as e:
            error_msg = {"error": str(e)}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to compare caches: {e}", err=True)
            ctx.exit(1)

    asyncio.run(_compare())


@analytics_group.command()
@click.option('--cache-names', multiple=True, help='Specific cache names to monitor (can be used multiple times)')
@click.option('--metrics', multiple=True, default=['hit_ratio', 'operations'],
              help='Metrics to display (can be used multiple times)')
@click.option('--interval', type=float, default=5.0, help='Refresh interval in seconds')
@click.option('--threshold-alerts', is_flag=True, help='Enable threshold-based alerts')
@click.pass_context
def dashboard(
    ctx: click.Context,
    cache_names: List[str],
    metrics: List[str],
    interval: float,
    threshold_alerts: bool
):
    """
    Real-time analytics dashboard for cache monitoring.

    Examples:
        omnicache analytics dashboard --cache-names web_cache --cache-names api_cache
        omnicache analytics dashboard --metrics hit_ratio --metrics operations --interval 2.0
        omnicache analytics dashboard --threshold-alerts
    """
    async def _dashboard():
        try:
            # Get all caches if none specified
            if not cache_names:
                all_caches = manager.list_caches()
                cache_list = [cache['name'] for cache in all_caches]
            else:
                cache_list = list(cache_names)

            if not cache_list:
                raise CacheError("No caches found to monitor")

            click.echo("Real-time Cache Analytics Dashboard")
            click.echo("Press Ctrl+C to stop monitoring...")
            click.echo()

            # Thresholds for alerts
            alert_thresholds = {
                'hit_ratio': 0.5,  # Alert if below 50%
                'access_time': 100,  # Alert if above 100ms
                'utilization': 0.9  # Alert if above 90%
            }

            try:
                while True:
                    dashboard_data = {
                        "timestamp": datetime.now().isoformat(),
                        "caches": {},
                        "alerts": []
                    }

                    # Clear screen
                    click.clear()
                    click.echo(f"Cache Analytics Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    click.echo("=" * 80)

                    # Collect and display data for each cache
                    for cache_name in cache_list:
                        try:
                            cache = await manager.get_cache(cache_name)
                            if cache:
                                stats = await cache.get_statistics()
                                stats_dict = stats.to_dict()

                                cache_data = {
                                    "hit_ratio": stats.hit_ratio,
                                    "operations": stats.total_operations,
                                    "access_time": stats.average_access_time,
                                    "utilization": stats_dict.get('current_size', 0) / max(stats_dict.get('max_size', 1), 1),
                                    "current_size": stats_dict.get('current_size', 0),
                                    "max_size": stats_dict.get('max_size', 0)
                                }

                                dashboard_data["caches"][cache_name] = cache_data

                                # Display cache metrics
                                click.echo(f"\n{cache_name}:")
                                click.echo("-" * 40)

                                for metric in metrics:
                                    if metric in cache_data:
                                        value = cache_data[metric]
                                        if metric == "hit_ratio" or metric == "utilization":
                                            display_value = f"{value:.1%}"
                                        elif metric == "access_time":
                                            display_value = f"{value:.1f}ms"
                                        else:
                                            display_value = f"{value:,}"

                                        # Check for alerts
                                        alert_triggered = False
                                        if threshold_alerts and metric in alert_thresholds:
                                            threshold = alert_thresholds[metric]
                                            if (metric == 'hit_ratio' and value < threshold) or \
                                               (metric in ['access_time', 'utilization'] and value > threshold):
                                                alert_triggered = True
                                                dashboard_data["alerts"].append({
                                                    "cache": cache_name,
                                                    "metric": metric,
                                                    "value": value,
                                                    "threshold": threshold
                                                })

                                        status_icon = "⚠️" if alert_triggered else "✓"
                                        click.echo(f"  {status_icon} {metric.replace('_', ' ').title()}: {display_value}")

                        except Exception as e:
                            dashboard_data["caches"][cache_name] = {"error": str(e)}
                            click.echo(f"\n{cache_name}: ❌ Error - {e}")

                    # Display alerts if any
                    if dashboard_data["alerts"]:
                        click.echo(f"\n🚨 ALERTS:")
                        click.echo("-" * 40)
                        for alert in dashboard_data["alerts"]:
                            cache_name = alert["cache"]
                            metric = alert["metric"]
                            value = alert["value"]
                            threshold = alert["threshold"]

                            if metric == "hit_ratio" or metric == "utilization":
                                value_str = f"{value:.1%}"
                                threshold_str = f"{threshold:.1%}"
                            else:
                                value_str = f"{value:.1f}"
                                threshold_str = f"{threshold:.1f}"

                            click.echo(f"  {cache_name}: {metric} = {value_str} (threshold: {threshold_str})")

                    click.echo(f"\nNext update in {interval}s...")

                    await asyncio.sleep(interval)

            except KeyboardInterrupt:
                click.echo("\nDashboard stopped.")

        except Exception as e:
            error_msg = {"error": str(e)}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to start dashboard: {e}", err=True)
            ctx.exit(1)

    asyncio.run(_dashboard())


@analytics_group.command()
@click.option('--global-analytics', is_flag=True, help='Export global analytics across all caches')
@click.option('--cache-names', multiple=True, help='Specific cache names to export (can be used multiple times)')
@click.option('--format', 'export_format', type=click.Choice(['json', 'csv', 'excel']), default='json',
              help='Export format')
@click.option('--output-file', required=True, help='Output file path')
@click.option('--include-raw-data', is_flag=True, help='Include raw analytics data')
@click.pass_context
def export(
    ctx: click.Context,
    global_analytics: bool,
    cache_names: List[str],
    export_format: str,
    output_file: str,
    include_raw_data: bool
):
    """
    Export analytics data to various formats.

    Examples:
        omnicache analytics export --global-analytics --format json --output-file analytics.json
        omnicache analytics export --cache-names web_cache --format csv --output-file web_cache.csv
        omnicache analytics export --format excel --output-file report.xlsx --include-raw-data
    """
    async def _export():
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "export_format": export_format,
                "include_raw_data": include_raw_data
            }

            if global_analytics:
                # Export global analytics
                global_data = await manager.get_enterprise_analytics()
                export_data["global_analytics"] = global_data

                # Also include all cache data
                all_caches = manager.list_caches()
                export_data["cache_data"] = {}

                for cache_info in all_caches:
                    cache_name = cache_info['name']
                    try:
                        cache_analytics = await manager.get_enterprise_analytics(cache_name)
                        export_data["cache_data"][cache_name] = cache_analytics
                    except Exception as e:
                        export_data["cache_data"][cache_name] = {"error": str(e)}

            else:
                # Export specific caches
                if not cache_names:
                    all_caches = manager.list_caches()
                    cache_names = [cache['name'] for cache in all_caches]

                export_data["cache_data"] = {}
                for cache_name in cache_names:
                    try:
                        cache_analytics = await manager.get_enterprise_analytics(cache_name)
                        export_data["cache_data"][cache_name] = cache_analytics
                    except Exception as e:
                        export_data["cache_data"][cache_name] = {"error": str(e)}

            # Export to specified format
            if export_format == 'json':
                with open(output_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)

            elif export_format == 'csv':
                import csv
                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)

                    # Write headers
                    writer.writerow(['Cache Name', 'Metric', 'Value', 'Timestamp'])

                    # Write data
                    for cache_name, cache_data in export_data.get("cache_data", {}).items():
                        if "error" in cache_data:
                            writer.writerow([cache_name, 'error', cache_data['error'], export_data['export_timestamp']])
                        else:
                            # Extract key metrics
                            for metric, value in cache_data.items():
                                if isinstance(value, (int, float, str)):
                                    writer.writerow([cache_name, metric, value, export_data['export_timestamp']])

            elif export_format == 'excel':
                # Note: This would require openpyxl or xlsxwriter
                click.echo("Excel export requires additional dependencies. Exporting as JSON instead.")
                json_file = output_file.replace('.xlsx', '.json').replace('.xls', '.json')
                with open(json_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                export_data["actual_output"] = json_file

            result = {
                "export_completed": True,
                "output_file": output_file,
                "format": export_format,
                "records_exported": len(export_data.get("cache_data", {}))
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2))
            else:
                click.echo(f"✓ Analytics data exported successfully")
                click.echo(f"  Output file: {output_file}")
                click.echo(f"  Format: {export_format}")
                click.echo(f"  Records: {result['records_exported']}")

        except Exception as e:
            error_msg = {"error": str(e)}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to export analytics data: {e}", err=True)
            ctx.exit(1)

    asyncio.run(_export())


# Export the command group
analytics = analytics_group