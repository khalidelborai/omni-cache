"""
Machine Learning prefetching CLI commands.

Provides ML-based cache optimization including access pattern prediction,
prefetching, and intelligent cache management.
"""

import asyncio
import click
import json
from typing import Optional, Dict, Any, List

from omnicache.core.manager import manager
from omnicache.cli.formatters import format_output
from omnicache.core.exceptions import CacheError


@click.group()
def ml_group():
    """
    Machine Learning-based cache optimization.

    Leverage ML algorithms for access pattern prediction, intelligent prefetching,
    and automated cache optimization based on historical usage data.
    """
    pass


@ml_group.command()
@click.argument('cache_name')
@click.option('--model-type', default='lstm', help='ML model type (lstm, transformer, decision_tree)')
@click.option('--training-window', type=int, default=10000, help='Training data window size')
@click.option('--prediction-horizon', type=int, default=100, help='Prediction horizon in operations')
@click.option('--prefetch-threshold', type=float, default=0.7, help='Prefetch confidence threshold')
@click.option('--enable-auto-tune', is_flag=True, help='Enable automatic model tuning')
@click.pass_context
def enable(
    ctx: click.Context,
    cache_name: str,
    model_type: str,
    training_window: int,
    prediction_horizon: int,
    prefetch_threshold: float,
    enable_auto_tune: bool
):
    """
    Enable ML prefetching for a cache.

    Examples:
        omnicache ml enable web_cache --model-type lstm
        omnicache ml enable api_cache --prefetch-threshold 0.8 --enable-auto-tune
    """
    async def _enable():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Validate parameters
            if not 0.0 <= prefetch_threshold <= 1.0:
                raise ValueError("Prefetch threshold must be between 0.0 and 1.0")

            if training_window < 100:
                raise ValueError("Training window must be at least 100 operations")

            # Configure ML features
            ml_config = {
                'ml_prefetch_enabled': True,
                'model_type': model_type,
                'training_window': training_window,
                'prediction_horizon': prediction_horizon,
                'prefetch_threshold': prefetch_threshold,
                'auto_tune_enabled': enable_auto_tune
            }

            # Apply configuration
            await cache.update_config(ml_config)

            # Initialize ML predictor if available through manager
            if hasattr(manager, '_access_predictor') and manager._access_predictor:
                await manager._access_predictor.configure_cache(cache_name, ml_config)

            # Get current stats
            stats = await cache.get_statistics()
            ml_info = stats.to_dict().get('ml_info', {})

            result = {
                "cache_name": cache_name,
                "ml_enabled": True,
                "configuration": ml_config,
                "ml_info": ml_info,
                "status": "enabled"
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2, default=str))
            else:
                click.echo(f"✓ Enabled ML prefetching for cache '{cache_name}'")
                if ctx.obj['verbose']:
                    click.echo(f"  Model Type: {model_type}")
                    click.echo(f"  Training Window: {training_window}")
                    click.echo(f"  Prediction Horizon: {prediction_horizon}")
                    click.echo(f"  Prefetch Threshold: {prefetch_threshold}")
                    click.echo(f"  Auto-tune: {'Enabled' if enable_auto_tune else 'Disabled'}")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to enable ML prefetching for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_enable())


@ml_group.command()
@click.argument('cache_name')
@click.pass_context
def disable(ctx: click.Context, cache_name: str):
    """
    Disable ML prefetching for a cache.

    Examples:
        omnicache ml disable web_cache
    """
    async def _disable():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Disable ML features
            ml_config = {
                'ml_prefetch_enabled': False
            }

            await cache.update_config(ml_config)

            result = {
                "cache_name": cache_name,
                "ml_enabled": False,
                "status": "disabled"
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2))
            else:
                click.echo(f"✓ Disabled ML prefetching for cache '{cache_name}'")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to disable ML prefetching for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_disable())


@ml_group.command()
@click.argument('cache_name')
@click.option('--watch', '-w', is_flag=True, help='Watch ML statistics in real-time')
@click.option('--interval', type=float, default=2.0, help='Watch interval in seconds')
@click.pass_context
def stats(
    ctx: click.Context,
    cache_name: str,
    watch: bool,
    interval: float
):
    """
    Show ML prefetching statistics and performance metrics.

    Examples:
        omnicache ml stats web_cache
        omnicache ml stats web_cache --watch --interval 1.0
    """
    async def _stats():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            def display_stats(stats_data):
                ml_info = stats_data.get('ml_info', {})

                if ctx.obj['format'] == 'json':
                    click.echo(json.dumps({
                        "cache_name": cache_name,
                        "ml_stats": ml_info,
                        "overall_stats": stats_data
                    }, indent=2, default=str))
                else:
                    click.echo(f"\nML Prefetching Statistics for '{cache_name}':")
                    click.echo("=" * 60)

                    # Check if ML is enabled
                    if not ml_info.get('enabled', False):
                        click.echo("❌ ML prefetching is not enabled for this cache")
                        click.echo("Use 'omnicache ml enable' to enable ML features")
                        return

                    # ML Performance metrics
                    click.echo(f"Model Status: {'✓ Active' if ml_info.get('model_active', False) else '❌ Inactive'}")
                    click.echo(f"Model Type: {ml_info.get('model_type', 'N/A')}")
                    click.echo(f"Training Status: {ml_info.get('training_status', 'N/A')}")

                    # Prediction metrics
                    predictions = ml_info.get('predictions', {})
                    if predictions:
                        click.echo(f"\nPrediction Performance:")
                        click.echo(f"  Accuracy: {predictions.get('accuracy', 0):.2%}")
                        click.echo(f"  Precision: {predictions.get('precision', 0):.2%}")
                        click.echo(f"  Recall: {predictions.get('recall', 0):.2%}")
                        click.echo(f"  F1 Score: {predictions.get('f1_score', 0):.3f}")

                    # Prefetching metrics
                    prefetch = ml_info.get('prefetch', {})
                    if prefetch:
                        click.echo(f"\nPrefetching Performance:")
                        click.echo(f"  Items Prefetched: {prefetch.get('items_prefetched', 0)}")
                        click.echo(f"  Prefetch Hit Rate: {prefetch.get('hit_rate', 0):.2%}")
                        click.echo(f"  Prefetch Efficiency: {prefetch.get('efficiency', 0):.2%}")
                        click.echo(f"  Wasted Prefetches: {prefetch.get('wasted', 0)}")

                    # Model training info
                    training = ml_info.get('training', {})
                    if training:
                        click.echo(f"\nModel Training:")
                        click.echo(f"  Training Samples: {training.get('samples', 0)}")
                        click.echo(f"  Last Training: {training.get('last_training', 'Never')}")
                        click.echo(f"  Training Loss: {training.get('loss', 0):.4f}")
                        click.echo(f"  Epochs Completed: {training.get('epochs', 0)}")

                    # Access pattern insights
                    patterns = ml_info.get('access_patterns', {})
                    if patterns:
                        click.echo(f"\nAccess Pattern Analysis:")
                        click.echo(f"  Temporal Patterns: {patterns.get('temporal', {}).get('strength', 0):.2%}")
                        click.echo(f"  Sequential Patterns: {patterns.get('sequential', {}).get('strength', 0):.2%}")
                        click.echo(f"  User-based Patterns: {patterns.get('user_based', {}).get('strength', 0):.2%}")

                        # Pattern recommendations
                        dominant_pattern = max(patterns.items(), key=lambda x: x[1].get('strength', 0) if isinstance(x[1], dict) else 0)
                        if dominant_pattern[1].get('strength', 0) > 0.7:
                            click.echo(f"  → Dominant Pattern: {dominant_pattern[0].replace('_', ' ').title()}")

            if watch:
                click.echo("Watching ML statistics (Press Ctrl+C to stop)...")
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
                click.echo(f"✗ Failed to get ML stats for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_stats())


@ml_group.command()
@click.argument('cache_name')
@click.option('--force', is_flag=True, help='Force retraining even if model is performing well')
@click.pass_context
def train(ctx: click.Context, cache_name: str, force: bool):
    """
    Train or retrain the ML model for a cache.

    Examples:
        omnicache ml train web_cache
        omnicache ml train web_cache --force
    """
    async def _train():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Get ML insights from manager
            ml_insights = await manager.get_ml_insights(cache_name)

            if 'error' in ml_insights:
                if ml_insights['error'] == "ML predictor not enabled":
                    click.echo("❌ ML predictor not enabled. Use 'omnicache ml enable' first.")
                    ctx.exit(1)
                else:
                    raise CacheError(ml_insights['error'])

            # Check if training is needed
            current_accuracy = ml_insights.get('model_performance', {}).get('accuracy', 0)
            training_needed = force or current_accuracy < 0.7

            if not training_needed and not force:
                result = {
                    "cache_name": cache_name,
                    "training_status": "skipped",
                    "reason": f"Model already performing well (accuracy: {current_accuracy:.2%})",
                    "suggestion": "Use --force to retrain anyway"
                }

                if ctx.obj['format'] == 'json':
                    click.echo(json.dumps(result, indent=2))
                else:
                    click.echo(f"ℹ️ Training skipped for '{cache_name}'")
                    click.echo(f"   Model accuracy: {current_accuracy:.2%} (good enough)")
                    click.echo("   Use --force to retrain anyway")
                return

            # Initiate training
            if hasattr(manager, '_access_predictor') and manager._access_predictor:
                training_result = await manager._access_predictor.train_model(cache)
            else:
                training_result = {"error": "ML predictor not available"}

            result = {
                "cache_name": cache_name,
                "training_status": "completed" if "error" not in training_result else "failed",
                "training_result": training_result
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2, default=str))
            else:
                if "error" in training_result:
                    click.echo(f"✗ Training failed for '{cache_name}': {training_result['error']}")
                    ctx.exit(1)
                else:
                    click.echo(f"✓ Training completed for cache '{cache_name}'")
                    if ctx.obj['verbose'] and 'metrics' in training_result:
                        metrics = training_result['metrics']
                        click.echo(f"  Final Accuracy: {metrics.get('accuracy', 0):.2%}")
                        click.echo(f"  Training Loss: {metrics.get('loss', 0):.4f}")
                        click.echo(f"  Training Time: {metrics.get('training_time', 0):.2f}s")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to train ML model for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_train())


@ml_group.command()
@click.argument('cache_name')
@click.option('--keys', '-k', multiple=True, help='Specific keys to predict (can be used multiple times)')
@click.option('--pattern', help='Key pattern to predict')
@click.option('--horizon', type=int, default=10, help='Prediction horizon')
@click.pass_context
def predict(
    ctx: click.Context,
    cache_name: str,
    keys: List[str],
    pattern: Optional[str],
    horizon: int
):
    """
    Get ML predictions for cache access patterns.

    Examples:
        omnicache ml predict web_cache --keys user:123 --keys config:app
        omnicache ml predict web_cache --pattern "user:*" --horizon 20
    """
    async def _predict():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            predictions = {"cache_name": cache_name, "predictions": []}

            # Get predictions from manager
            if hasattr(manager, '_access_predictor') and manager._access_predictor:
                if keys:
                    for key in keys:
                        pred = await manager._access_predictor.predict_access(cache, key, horizon)
                        predictions["predictions"].append({
                            "key": key,
                            "probability": pred.get('probability', 0),
                            "confidence": pred.get('confidence', 0),
                            "predicted_time": pred.get('predicted_time'),
                            "factors": pred.get('factors', [])
                        })

                elif pattern:
                    pattern_predictions = await manager._access_predictor.predict_pattern(cache, pattern, horizon)
                    predictions["predictions"] = pattern_predictions

                else:
                    # Get general predictions
                    general_predictions = await manager._access_predictor.get_general_predictions(cache, horizon)
                    predictions["predictions"] = general_predictions

            else:
                predictions["error"] = "ML predictor not available"

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(predictions, indent=2, default=str))
            else:
                if "error" in predictions:
                    click.echo(f"✗ {predictions['error']}")
                    ctx.exit(1)

                click.echo(f"\nML Predictions for '{cache_name}':")
                click.echo("=" * 50)

                if not predictions["predictions"]:
                    click.echo("No predictions available")
                    return

                for pred in predictions["predictions"]:
                    key = pred.get("key", "Unknown")
                    probability = pred.get("probability", 0)
                    confidence = pred.get("confidence", 0)

                    click.echo(f"\nKey: {key}")
                    click.echo(f"  Access Probability: {probability:.2%}")
                    click.echo(f"  Confidence: {confidence:.2%}")

                    if pred.get("predicted_time"):
                        click.echo(f"  Predicted Access Time: {pred['predicted_time']}")

                    factors = pred.get("factors", [])
                    if factors:
                        click.echo("  Contributing Factors:")
                        for factor in factors[:3]:  # Show top 3 factors
                            click.echo(f"    • {factor}")

                    # Recommendation
                    if probability > 0.7 and confidence > 0.6:
                        click.echo("  💡 Recommendation: High probability - consider prefetching")
                    elif probability > 0.4:
                        click.echo("  💡 Recommendation: Moderate probability - monitor")
                    else:
                        click.echo("  💡 Recommendation: Low probability - no action needed")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to get predictions for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_predict())


@ml_group.command()
@click.argument('cache_name')
@click.option('--auto-optimize', is_flag=True, help='Apply ML optimization recommendations automatically')
@click.option('--target-accuracy', type=float, help='Target prediction accuracy')
@click.pass_context
def optimize(
    ctx: click.Context,
    cache_name: str,
    auto_optimize: bool,
    target_accuracy: Optional[float]
):
    """
    Optimize cache using ML recommendations.

    Examples:
        omnicache ml optimize web_cache --auto-optimize
        omnicache ml optimize web_cache --target-accuracy 0.85
    """
    async def _optimize():
        try:
            # Use manager's ML optimization
            optimization_result = await manager.optimize_with_ml(cache_name)

            if "error" in optimization_result:
                if ctx.obj['format'] == 'json':
                    click.echo(json.dumps(optimization_result, indent=2))
                else:
                    click.echo(f"✗ {optimization_result['error']}")
                ctx.exit(1)

            # Apply additional optimizations if requested
            if auto_optimize:
                additional_optimizations = []

                # Apply recommendations automatically
                for rec in optimization_result.get("recommendations_applied", []):
                    if rec['type'] == 'prefetch_tuning':
                        additional_optimizations.append("Adjusted prefetch threshold")
                    elif rec['type'] == 'model_retraining':
                        additional_optimizations.append("Retrained ML model")

                optimization_result["auto_optimizations"] = additional_optimizations

            if target_accuracy:
                current_accuracy = optimization_result.get("performance_improvement", {}).get("prediction_accuracy", 0)
                if current_accuracy < target_accuracy:
                    optimization_result["target_accuracy_met"] = False
                    optimization_result["accuracy_gap"] = target_accuracy - current_accuracy
                else:
                    optimization_result["target_accuracy_met"] = True

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(optimization_result, indent=2, default=str))
            else:
                click.echo(f"✓ ML optimization completed for '{cache_name}'")

                applied = optimization_result.get("recommendations_applied", [])
                if applied:
                    click.echo("\nOptimizations Applied:")
                    for opt in applied:
                        click.echo(f"  • {opt.get('description', opt.get('type', 'Unknown'))}")

                improvements = optimization_result.get("performance_improvement", {})
                if improvements:
                    click.echo("\nPerformance Improvements:")
                    for metric, value in improvements.items():
                        if isinstance(value, float):
                            click.echo(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
                        else:
                            click.echo(f"  {metric.replace('_', ' ').title()}: {value}")

                if target_accuracy:
                    if optimization_result.get("target_accuracy_met", False):
                        click.echo(f"\n✓ Target accuracy ({target_accuracy:.1%}) achieved")
                    else:
                        gap = optimization_result.get("accuracy_gap", 0)
                        click.echo(f"\n⚠️ Target accuracy not met (gap: {gap:.1%})")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to optimize cache '{cache_name}' with ML: {e}", err=True)
            ctx.exit(1)

    asyncio.run(_optimize())


@ml_group.command()
@click.argument('cache_name')
@click.pass_context
def insights(ctx: click.Context, cache_name: str):
    """
    Get ML-based insights and analysis for cache performance.

    Examples:
        omnicache ml insights web_cache
    """
    async def _insights():
        try:
            # Get insights from manager
            insights = await manager.get_ml_insights(cache_name)

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(insights, indent=2, default=str))
            else:
                if "error" in insights:
                    click.echo(f"✗ {insights['error']}")
                    ctx.exit(1)

                click.echo(f"\nML Insights for Cache '{cache_name}':")
                click.echo("=" * 50)

                # Model performance
                performance = insights.get('model_performance', {})
                if performance:
                    click.echo("Model Performance:")
                    click.echo(f"  Accuracy: {performance.get('accuracy', 0):.2%}")
                    click.echo(f"  Precision: {performance.get('precision', 0):.2%}")
                    click.echo(f"  Recall: {performance.get('recall', 0):.2%}")

                # Access patterns
                patterns = insights.get('access_patterns', {})
                if patterns:
                    click.echo("\nDetected Access Patterns:")
                    for pattern_name, pattern_data in patterns.items():
                        strength = pattern_data.get('strength', 0)
                        click.echo(f"  {pattern_name.replace('_', ' ').title()}: {strength:.1%}")

                # Predictions
                predictions = insights.get('predictions', {})
                if predictions:
                    click.echo("\nUpcoming Predictions:")
                    for pred in predictions.get('top_predictions', [])[:5]:
                        key = pred.get('key', 'Unknown')
                        probability = pred.get('probability', 0)
                        click.echo(f"  {key}: {probability:.1%} likely to be accessed")

                # Recommendations
                recommendations = insights.get('recommendations', [])
                if recommendations:
                    click.echo("\nML Recommendations:")
                    for rec in recommendations:
                        click.echo(f"  • {rec}")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to get ML insights for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_insights())


# Export the command group
ml = ml_group