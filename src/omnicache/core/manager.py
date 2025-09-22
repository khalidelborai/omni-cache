"""
Cache Manager implementation.

High-level cache management with configuration-based cache creation,
lifecycle management, and advanced features like monitoring and optimization.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, TYPE_CHECKING
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

from omnicache.core.registry import registry
from omnicache.core.exceptions import (
    CacheError,
    CacheNotFoundError,
    CacheConfigurationError
)
from omnicache.models.security_policy import SecurityPolicy, EncryptionAlgorithm
from omnicache.models.tier import CacheTier

if TYPE_CHECKING:
    from omnicache.models.cache import Cache


class CacheManager:
    """
    High-level cache management interface.

    Provides configuration-based cache creation, lifecycle management,
    monitoring, and optimization features.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize cache manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._configurations: Dict[str, Dict[str, Any]] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._is_initialized = False
        self._shutdown_hooks: List[Callable] = []

        # Enterprise features
        self._analytics_tracker: Optional[Any] = None
        self._security_monitor: Optional[Any] = None
        self._access_predictor: Optional[Any] = None
        self._hierarchical_configs: Dict[str, List[CacheTier]] = {}
        self._security_policies: Dict[str, SecurityPolicy] = {}
        self._logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize the cache manager."""
        if self._is_initialized:
            return

        # Load configurations if path provided
        if self.config_path:
            await self.load_configurations()

        # Initialize registry
        await registry.initialize_all()

        # Initialize enterprise features
        await self._initialize_enterprise_features()

        self._is_initialized = True

    async def _initialize_enterprise_features(self) -> None:
        """Initialize enterprise features."""
        try:
            # Initialize analytics tracker (when available)
            # self._analytics_tracker = AnalyticsTracker()
            # await self._analytics_tracker.initialize()

            # Initialize security monitor (when available)
            # self._security_monitor = SecurityMonitor()
            # await self._security_monitor.initialize()

            # Initialize ML access predictor (when available)
            # self._access_predictor = AccessPredictor()
            # await self._access_predictor.initialize()

            self._logger.info("Enterprise features initialized successfully")
        except Exception as e:
            self._logger.warning(f"Failed to initialize some enterprise features: {e}")
            # Continue without enterprise features if initialization fails

    async def shutdown(self) -> None:
        """Shutdown the cache manager and all caches."""
        if not self._is_initialized:
            return

        # Stop monitoring tasks
        for task in self._monitoring_tasks.values():
            if not task.done():
                task.cancel()

        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)

        self._monitoring_tasks.clear()

        # Execute shutdown hooks
        for hook in self._shutdown_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except Exception as e:
                print(f"Error in shutdown hook: {e}")

        # Shutdown all caches
        await registry.shutdown_all()

        self._is_initialized = False

    async def create_cache(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> 'Cache':
        """
        Create and register a new cache.

        Args:
            name: Cache name
            config: Cache configuration dictionary
            **kwargs: Additional cache parameters

        Returns:
            Created cache instance

        Raises:
            CacheError: If cache creation fails
            CacheConfigurationError: If configuration is invalid
        """
        try:
            # Use provided config or look up stored configuration
            cache_config = config or self._configurations.get(name, {})
            cache_config.update(kwargs)

            # Import Cache class dynamically to avoid circular imports
            from omnicache.models.cache import Cache

            # Create cache instance
            if config:
                cache = Cache.from_config(name, cache_config)
            else:
                cache = Cache(name, **cache_config)

            # Register in registry
            registry.register(cache)

            # Initialize cache
            await cache.initialize()

            return cache

        except Exception as e:
            raise CacheError(f"Failed to create cache '{name}': {str(e)}")

    async def create_enterprise_cache(
        self,
        name: str,
        strategy: str = "arc",
        tiers: Optional[List[Dict[str, Any]]] = None,
        security_policy: Optional[Dict[str, Any]] = None,
        enable_analytics: bool = True,
        enable_ml_prefetch: bool = True,
        **kwargs: Any
    ) -> 'Cache':
        """
        Create an enterprise cache with advanced features.

        Args:
            name: Cache name
            strategy: Caching strategy (arc, lru, lfu)
            tiers: Hierarchical tier configuration
            security_policy: Security policy configuration
            enable_analytics: Enable analytics tracking
            enable_ml_prefetch: Enable ML-based prefetching
            **kwargs: Additional cache parameters

        Returns:
            Created enterprise cache instance
        """
        try:
            # Build enterprise configuration
            config = kwargs.copy()
            config['strategy'] = strategy
            config['enterprise'] = True

            # Ensure strategy gets the required parameters
            if 'max_size' in config and strategy == 'arc':
                # Move max_size to strategy parameters for ARC
                strategy_params = config.get('parameters', {})
                strategy_params['capacity'] = config['max_size']
                config['parameters'] = strategy_params

            # Configure tiers if provided
            if tiers:
                tier_objects = []
                for tier_config in tiers:
                    tier = CacheTier.from_dict(tier_config)
                    tier_objects.append(tier)
                self._hierarchical_configs[name] = tier_objects
                config['hierarchical_tiers'] = tier_objects

            # Configure security policy if provided
            if security_policy:
                policy = SecurityPolicy.from_dict(security_policy)
                self._security_policies[name] = policy
                config['security_policy'] = policy

            # Enable enterprise features
            if enable_analytics and self._analytics_tracker:
                config['analytics_enabled'] = True
                config['analytics_tracker'] = self._analytics_tracker

            if enable_ml_prefetch and self._access_predictor:
                config['ml_prefetch_enabled'] = True
                config['access_predictor'] = self._access_predictor

            # Create cache
            cache = await self.create_cache(name, config)

            # Setup enterprise monitoring
            if self._security_monitor:
                await self._security_monitor.register_cache(name, cache)

            return cache

        except Exception as e:
            raise CacheError(f"Failed to create enterprise cache '{name}': {str(e)}")

    async def create_arc_cache(
        self,
        name: str,
        max_size: int = 1000,
        c_factor: float = 1.0,
        **kwargs: Any
    ) -> 'Cache':
        """
        Create an ARC (Adaptive Replacement Cache) strategy cache.

        Args:
            name: Cache name
            max_size: Maximum cache size
            c_factor: ARC adaptation factor
            **kwargs: Additional parameters

        Returns:
            ARC cache instance
        """
        config = {
            'strategy': 'arc',
            'max_size': max_size,
            'arc_c_factor': c_factor,
            **kwargs
        }
        return await self.create_cache(name, config)

    async def create_hierarchical_cache(
        self,
        name: str,
        tiers: List[Dict[str, Any]],
        promotion_threshold: float = 0.8,
        demotion_threshold: float = 0.2,
        **kwargs: Any
    ) -> 'Cache':
        """
        Create a hierarchical cache with multiple tiers.

        Args:
            name: Cache name
            tiers: List of tier configurations
            promotion_threshold: Hit ratio threshold for promotion
            demotion_threshold: Hit ratio threshold for demotion
            **kwargs: Additional parameters

        Returns:
            Hierarchical cache instance
        """
        tier_objects = []
        for tier_config in tiers:
            tier = CacheTier.from_dict(tier_config)
            tier_objects.append(tier)

        self._hierarchical_configs[name] = tier_objects

        config = {
            'hierarchical_tiers': tier_objects,
            'promotion_threshold': promotion_threshold,
            'demotion_threshold': demotion_threshold,
            **kwargs
        }
        return await self.create_cache(name, config)

    async def create_secure_cache(
        self,
        name: str,
        access_level: str = "authenticated",
        encryption: str = "aes_256",
        require_permissions: bool = True,
        audit_enabled: bool = True,
        **kwargs: Any
    ) -> 'Cache':
        """
        Create a secure cache with encryption and access control.

        Args:
            name: Cache name
            access_level: Access control level
            encryption: Encryption algorithm
            require_permissions: Require explicit permissions
            audit_enabled: Enable audit logging
            **kwargs: Additional parameters

        Returns:
            Secure cache instance
        """
        # Create security policy
        policy = SecurityPolicy(
            access_control_level=AccessControlLevel(access_level),
            encryption_algorithm=EncryptionAlgorithm(encryption),
            require_explicit_permissions=require_permissions,
            enable_audit_logging=audit_enabled
        )

        self._security_policies[name] = policy

        config = {
            'security_policy': policy,
            'enterprise': True,
            **kwargs
        }
        return await self.create_cache(name, config)

    async def get_cache(self, name: str, auto_create: bool = False, **kwargs: Any) -> Optional['Cache']:
        """
        Get a cache instance, optionally creating if not found.

        Args:
            name: Cache name
            auto_create: Create cache if not found
            **kwargs: Creation parameters if auto_create is True

        Returns:
            Cache instance or None
        """
        cache = registry.get(name)

        if cache is None and auto_create:
            cache = await self.create_cache(name, **kwargs)

        return cache

    async def get_or_create_cache(self, name: str, **kwargs: Any) -> 'Cache':
        """
        Get cache or create if not exists.

        Args:
            name: Cache name
            **kwargs: Creation parameters

        Returns:
            Cache instance
        """
        cache = await self.get_cache(name, auto_create=True, **kwargs)
        if cache is None:
            raise CacheError(f"Failed to get or create cache '{name}'")
        return cache

    async def delete_cache(self, name: str) -> bool:
        """
        Delete a cache and unregister it.

        Args:
            name: Cache name

        Returns:
            True if deleted, False if not found
        """
        cache = registry.get(name)
        if cache:
            try:
                await cache.shutdown()
                return registry.unregister(name)
            except Exception as e:
                raise CacheError(f"Failed to delete cache '{name}': {str(e)}")

        return False

    def list_caches(self) -> List[Dict[str, Any]]:
        """
        List all managed caches.

        Returns:
            List of cache information dictionaries
        """
        return registry.list_caches()

    async def configure_cache(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Update cache configuration.

        Args:
            name: Cache name
            config: New configuration

        Returns:
            True if configured, False if cache not found
        """
        cache = registry.get(name)
        if cache:
            try:
                await cache.update_config(config)
                self._configurations[name] = config
                return True
            except Exception as e:
                raise CacheConfigurationError(f"Failed to configure cache '{name}': {str(e)}")

        return False

    async def load_configurations(self, path: Optional[str] = None) -> None:
        """
        Load cache configurations from file.

        Args:
            path: Configuration file path (uses instance path if None)
        """
        config_file = Path(path or self.config_path)

        if not config_file.exists():
            return

        try:
            with config_file.open('r') as f:
                configs = json.load(f)

            self._configurations.update(configs)

            # Auto-create configured caches
            for name, config in configs.items():
                if config.get('auto_create', False):
                    try:
                        await self.create_cache(name, config)
                    except Exception as e:
                        print(f"Failed to auto-create cache '{name}': {e}")

        except Exception as e:
            raise CacheConfigurationError(f"Failed to load configurations: {str(e)}")

    async def save_configurations(self, path: Optional[str] = None) -> None:
        """
        Save current configurations to file.

        Args:
            path: Configuration file path (uses instance path if None)
        """
        config_file = Path(path or self.config_path)

        try:
            # Ensure directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)

            with config_file.open('w') as f:
                json.dump(self._configurations, f, indent=2, default=str)

        except Exception as e:
            raise CacheConfigurationError(f"Failed to save configurations: {str(e)}")

    async def clear_cache(self, name: str, pattern: Optional[str] = None) -> int:
        """
        Clear entries from a cache.

        Args:
            name: Cache name
            pattern: Optional key pattern

        Returns:
            Number of entries cleared

        Raises:
            CacheNotFoundError: If cache not found
        """
        cache = registry.get_or_raise(name)
        result = await cache.clear(pattern=pattern)
        return result.cleared_count

    async def get_cache_stats(self, name: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a cache.

        Args:
            name: Cache name

        Returns:
            Statistics dictionary

        Raises:
            CacheNotFoundError: If cache not found
        """
        cache = registry.get_or_raise(name)
        stats = await cache.get_statistics()

        # Convert Statistics object to dictionary
        stats_dict = stats.to_dict()

        # Add cache info details
        cache_info = cache.get_info()
        stats_dict.update({
            "name": cache_info.get("name", name),
            "status": cache_info.get("status", ""),
            "strategy": str(cache.strategy),
            "backend": str(cache.backend),
            "namespace": cache_info.get("namespace", ""),
            "max_size": cache_info.get("max_size"),
            "default_ttl": cache_info.get("default_ttl"),
            "created_at": cache_info.get("created_at"),
            "last_accessed": cache_info.get("last_accessed")
        })

        # Add registry stats
        registry_info = registry.get_cache_info(name)
        if registry_info:
            stats_dict.update({
                "registry_info": {
                    "created_at": registry_info.get("registry_created_at"),
                    "access_count": registry_info.get("registry_access_count"),
                    "last_accessed": registry_info.get("registry_last_accessed")
                }
            })
            # Override with registry details if available
            if "registry_created_at" in registry_info:
                stats_dict["created_at"] = registry_info["registry_created_at"]
            if "registry_last_accessed" in registry_info:
                stats_dict["last_accessed"] = registry_info["registry_last_accessed"]

        return stats_dict

    async def monitor_cache(
        self,
        name: str,
        interval: float = 60.0,
        callback: Optional[Callable] = None
    ) -> None:
        """
        Start monitoring a cache with periodic statistics collection.

        Args:
            name: Cache name
            interval: Monitoring interval in seconds
            callback: Optional callback for statistics
        """
        if name in self._monitoring_tasks:
            return  # Already monitoring

        async def monitor_loop():
            while True:
                try:
                    await asyncio.sleep(interval)

                    if not registry.exists(name):
                        break  # Cache no longer exists

                    stats = await self.get_cache_stats(name)

                    if callback:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(name, stats)
                            else:
                                callback(name, stats)
                        except Exception as e:
                            print(f"Error in monitoring callback for '{name}': {e}")

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Error monitoring cache '{name}': {e}")

        self._monitoring_tasks[name] = asyncio.create_task(monitor_loop())

    async def stop_monitoring(self, name: str) -> None:
        """
        Stop monitoring a cache.

        Args:
            name: Cache name
        """
        if name in self._monitoring_tasks:
            task = self._monitoring_tasks.pop(name)
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def optimize_cache(self, name: str) -> Dict[str, Any]:
        """
        Run optimization on a cache (cleanup, defragmentation, etc.).

        Args:
            name: Cache name

        Returns:
            Optimization results

        Raises:
            CacheNotFoundError: If cache not found
        """
        cache = registry.get_or_raise(name)
        optimization_results = {
            "cache_name": name,
            "optimized_at": datetime.now().isoformat(),
            "actions_performed": []
        }

        try:
            # Perform backend-specific cleanup
            if hasattr(cache.backend, '_cleanup_expired'):
                await cache.backend._cleanup_expired()
                optimization_results["actions_performed"].append("expired_cleanup")

            # Strategy-specific optimization
            if hasattr(cache.strategy, 'cleanup_expired'):
                cleaned = await cache.strategy.cleanup_expired(cache)
                optimization_results["actions_performed"].append(f"strategy_cleanup_{cleaned}")

            optimization_results["success"] = True

        except Exception as e:
            optimization_results["success"] = False
            optimization_results["error"] = str(e)

        return optimization_results

    async def bulk_operation(
        self,
        operation: str,
        cache_names: List[str],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Perform bulk operations on multiple caches.

        Args:
            operation: Operation name (clear, delete, optimize, etc.)
            cache_names: List of cache names
            **kwargs: Operation parameters

        Returns:
            Results dictionary with per-cache results
        """
        results = {"operation": operation, "results": {}, "summary": {}}

        for name in cache_names:
            try:
                if operation == "clear":
                    result = await self.clear_cache(name, **kwargs)
                elif operation == "delete":
                    result = await self.delete_cache(name)
                elif operation == "optimize":
                    result = await self.optimize_cache(name)
                else:
                    result = {"error": f"Unknown operation: {operation}"}

                results["results"][name] = {"success": True, "result": result}

            except Exception as e:
                results["results"][name] = {"success": False, "error": str(e)}

        # Generate summary
        success_count = sum(1 for r in results["results"].values() if r["success"])
        results["summary"] = {
            "total": len(cache_names),
            "successful": success_count,
            "failed": len(cache_names) - success_count
        }

        return results

    def add_shutdown_hook(self, hook: Callable) -> None:
        """
        Add a shutdown hook to be called during manager shutdown.

        Args:
            hook: Callable to execute on shutdown
        """
        self._shutdown_hooks.append(hook)

    async def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get manager-wide statistics.

        Returns:
            Manager statistics dictionary
        """
        stats = {
            "manager_initialized": self._is_initialized,
            "config_path": str(self.config_path) if self.config_path else None,
            "configurations_loaded": len(self._configurations),
            "monitoring_tasks": len(self._monitoring_tasks),
            "shutdown_hooks": len(self._shutdown_hooks),
            "registry_stats": registry.get_statistics(),
            "enterprise_features": {
                "analytics_enabled": self._analytics_tracker is not None,
                "security_monitor_enabled": self._security_monitor is not None,
                "ml_predictor_enabled": self._access_predictor is not None,
                "hierarchical_caches": len(self._hierarchical_configs),
                "secured_caches": len(self._security_policies)
            }
        }

        # Add enterprise analytics if available
        if self._analytics_tracker:
            try:
                analytics_stats = await self._analytics_tracker.get_global_stats()
                stats["analytics"] = analytics_stats
            except Exception:
                pass

        return stats

    async def get_enterprise_analytics(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get enterprise analytics data.

        Args:
            cache_name: Specific cache name or None for global analytics

        Returns:
            Analytics data dictionary
        """
        if not self._analytics_tracker:
            return {"error": "Analytics not enabled"}

        try:
            if cache_name:
                return await self._analytics_tracker.get_cache_analytics(cache_name)
            else:
                return await self._analytics_tracker.get_global_analytics()
        except Exception as e:
            return {"error": str(e)}

    async def get_security_report(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get security monitoring report.

        Args:
            cache_name: Specific cache name or None for global report

        Returns:
            Security report dictionary
        """
        if not self._security_monitor:
            return {"error": "Security monitoring not enabled"}

        try:
            if cache_name:
                return await self._security_monitor.get_cache_report(cache_name)
            else:
                return await self._security_monitor.get_global_report()
        except Exception as e:
            return {"error": str(e)}

    async def get_ml_insights(self, cache_name: str) -> Dict[str, Any]:
        """
        Get ML-based cache insights and predictions.

        Args:
            cache_name: Cache name

        Returns:
            ML insights dictionary
        """
        if not self._access_predictor:
            return {"error": "ML predictor not enabled"}

        try:
            cache = registry.get(cache_name)
            if not cache:
                return {"error": f"Cache '{cache_name}' not found"}

            insights = await self._access_predictor.get_insights(cache)
            return insights
        except Exception as e:
            return {"error": str(e)}

    async def configure_security_policy(
        self,
        cache_name: str,
        policy_config: Dict[str, Any]
    ) -> bool:
        """
        Configure security policy for a cache.

        Args:
            cache_name: Cache name
            policy_config: Security policy configuration

        Returns:
            True if configured successfully
        """
        try:
            policy = SecurityPolicy.from_dict(policy_config)
            self._security_policies[cache_name] = policy

            # Apply to existing cache if found
            cache = registry.get(cache_name)
            if cache and hasattr(cache, 'update_security_policy'):
                await cache.update_security_policy(policy)

            return True
        except Exception as e:
            self._logger.error(f"Failed to configure security policy for '{cache_name}': {e}")
            return False

    async def optimize_with_ml(self, cache_name: str) -> Dict[str, Any]:
        """
        Optimize cache configuration using ML recommendations.

        Args:
            cache_name: Cache name

        Returns:
            Optimization results
        """
        if not self._access_predictor:
            return {"error": "ML predictor not enabled"}

        try:
            cache = registry.get_or_raise(cache_name)
            recommendations = await self._access_predictor.get_optimization_recommendations(cache)

            optimization_results = {
                "cache_name": cache_name,
                "optimized_at": datetime.now().isoformat(),
                "recommendations_applied": [],
                "performance_improvement": {}
            }

            # Apply recommendations
            for recommendation in recommendations:
                try:
                    if recommendation['type'] == 'strategy_change':
                        # Strategy optimization handled by cache
                        await cache.optimize_strategy(recommendation['params'])
                        optimization_results["recommendations_applied"].append(recommendation)

                    elif recommendation['type'] == 'size_adjustment':
                        # Size optimization
                        await cache.update_config({'max_size': recommendation['new_size']})
                        optimization_results["recommendations_applied"].append(recommendation)

                    elif recommendation['type'] == 'ttl_optimization':
                        # TTL optimization
                        await cache.update_config({'default_ttl': recommendation['new_ttl']})
                        optimization_results["recommendations_applied"].append(recommendation)

                except Exception as e:
                    self._logger.warning(f"Failed to apply recommendation {recommendation}: {e}")

            # Measure performance improvement
            if optimization_results["recommendations_applied"]:
                stats_after = await cache.get_statistics()
                optimization_results["performance_improvement"] = {
                    "hit_ratio_change": stats_after.hit_ratio,
                    "avg_access_time_change": stats_after.average_access_time
                }

            return optimization_results

        except Exception as e:
            return {"error": str(e), "cache_name": cache_name}

    async def bulk_enterprise_operation(
        self,
        operation: str,
        cache_names: List[str],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Perform bulk enterprise operations on multiple caches.

        Args:
            operation: Operation name (security_scan, ml_optimize, analytics_export)
            cache_names: List of cache names
            **kwargs: Operation parameters

        Returns:
            Results dictionary with per-cache results
        """
        results = {"operation": operation, "results": {}, "summary": {}}

        for name in cache_names:
            try:
                if operation == "security_scan":
                    result = await self.get_security_report(name)
                elif operation == "ml_optimize":
                    result = await self.optimize_with_ml(name)
                elif operation == "analytics_export":
                    result = await self.get_enterprise_analytics(name)
                elif operation == "performance_analyze":
                    cache = registry.get_or_raise(name)
                    stats = await cache.get_statistics()
                    result = stats.to_dict()
                else:
                    result = {"error": f"Unknown enterprise operation: {operation}"}

                results["results"][name] = {"success": True, "result": result}

            except Exception as e:
                results["results"][name] = {"success": False, "error": str(e)}

        # Generate summary
        success_count = sum(1 for r in results["results"].values() if r["success"])
        results["summary"] = {
            "total": len(cache_names),
            "successful": success_count,
            "failed": len(cache_names) - success_count,
            "operation_type": "enterprise"
        }

        return results

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._is_initialized:
            # Run shutdown in event loop if available
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.shutdown())
            except RuntimeError:
                # No event loop running
                asyncio.run(self.shutdown())


# Global manager instance
manager = CacheManager()


# Convenience functions for global access
async def create_cache(name: str, **kwargs: Any) -> 'Cache':
    """Create cache using global manager."""
    return await manager.create_cache(name, **kwargs)


async def get_cache(name: str, auto_create: bool = False, **kwargs: Any) -> Optional['Cache']:
    """Get cache using global manager."""
    return await manager.get_cache(name, auto_create, **kwargs)


async def delete_cache(name: str) -> bool:
    """Delete cache using global manager."""
    return await manager.delete_cache(name)