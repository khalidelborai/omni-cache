"""
CLI configuration management.

Handles loading and managing CLI configuration from files and environment.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CLIConfig:
    """CLI configuration data structure."""
    default_strategy: str = "lru"
    default_max_size: Optional[int] = None
    default_backend: str = "memory"
    default_ttl: Optional[float] = None
    default_namespace: Optional[str] = None
    output_format: str = "table"
    verbose: bool = False
    quiet: bool = False
    config_file: Optional[str] = None
    redis_config: Dict[str, Any] = field(default_factory=dict)
    filesystem_config: Dict[str, Any] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CLIConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "default_strategy": self.default_strategy,
            "default_max_size": self.default_max_size,
            "default_backend": self.default_backend,
            "default_ttl": self.default_ttl,
            "default_namespace": self.default_namespace,
            "output_format": self.output_format,
            "verbose": self.verbose,
            "quiet": self.quiet,
            "redis_config": self.redis_config,
            "filesystem_config": self.filesystem_config,
            "custom_settings": self.custom_settings
        }


def load_config(config_path: Optional[str] = None) -> CLIConfig:
    """
    Load CLI configuration from file and environment.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded CLI configuration
    """
    config = CLIConfig()

    # Load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            config = CLIConfig.from_dict(file_config)
            config.config_file = config_path
        except Exception as e:
            # Fallback to default config if file loading fails
            pass

    # Override with environment variables
    config = _load_from_environment(config)

    # Load from default config locations
    if not config_path:
        config = _load_from_default_locations(config)

    return config


def _load_from_environment(config: CLIConfig) -> CLIConfig:
    """Load configuration overrides from environment variables."""
    env_mappings = {
        'OMNICACHE_DEFAULT_STRATEGY': 'default_strategy',
        'OMNICACHE_DEFAULT_MAX_SIZE': 'default_max_size',
        'OMNICACHE_DEFAULT_BACKEND': 'default_backend',
        'OMNICACHE_DEFAULT_TTL': 'default_ttl',
        'OMNICACHE_DEFAULT_NAMESPACE': 'default_namespace',
        'OMNICACHE_OUTPUT_FORMAT': 'output_format',
        'OMNICACHE_VERBOSE': 'verbose',
        'OMNICACHE_QUIET': 'quiet'
    }

    for env_var, config_attr in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                # Convert types appropriately
                if config_attr in ['default_max_size']:
                    env_value = int(env_value) if env_value != 'null' else None
                elif config_attr in ['default_ttl']:
                    env_value = float(env_value) if env_value != 'null' else None
                elif config_attr in ['verbose', 'quiet']:
                    env_value = env_value.lower() in ('true', '1', 'yes', 'on')

                setattr(config, config_attr, env_value)
            except (ValueError, TypeError):
                # Skip invalid environment values
                continue

    # Load Redis configuration from environment
    redis_env_mappings = {
        'OMNICACHE_REDIS_HOST': 'host',
        'OMNICACHE_REDIS_PORT': 'port',
        'OMNICACHE_REDIS_DB': 'db',
        'OMNICACHE_REDIS_PASSWORD': 'password',
        'OMNICACHE_REDIS_USERNAME': 'username'
    }

    for env_var, redis_key in redis_env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                if redis_key in ['port', 'db']:
                    env_value = int(env_value)
                config.redis_config[redis_key] = env_value
            except ValueError:
                continue

    return config


def _load_from_default_locations(config: CLIConfig) -> CLIConfig:
    """Load configuration from default locations."""
    default_paths = [
        Path.home() / '.omnicache' / 'config.json',
        Path.cwd() / '.omnicache.json',
        Path.cwd() / 'omnicache.config.json'
    ]

    for path in default_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    file_config = json.load(f)
                config = CLIConfig.from_dict({**config.to_dict(), **file_config})
                config.config_file = str(path)
                break
            except Exception:
                continue

    return config


def save_config(config: CLIConfig, config_path: Optional[str] = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        config_path: Path to save configuration to
    """
    if not config_path:
        config_path = config.config_file or str(Path.home() / '.omnicache' / 'config.json')

    # Ensure directory exists
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def get_cache_defaults(config: CLIConfig) -> Dict[str, Any]:
    """
    Get default cache configuration from CLI config.

    Args:
        config: CLI configuration

    Returns:
        Dictionary of cache defaults
    """
    defaults = {}

    if config.default_strategy:
        defaults['strategy'] = config.default_strategy

    if config.default_max_size is not None:
        defaults['max_size'] = config.default_max_size

    if config.default_backend:
        defaults['backend'] = config.default_backend

    if config.default_ttl is not None:
        defaults['default_ttl'] = config.default_ttl

    if config.default_namespace:
        defaults['namespace'] = config.default_namespace

    # Add backend-specific configuration
    if config.default_backend == 'redis' and config.redis_config:
        defaults['backend_config'] = config.redis_config

    if config.default_backend == 'filesystem' and config.filesystem_config:
        defaults['backend_config'] = config.filesystem_config

    return defaults