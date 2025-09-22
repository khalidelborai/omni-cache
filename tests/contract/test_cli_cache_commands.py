"""
Contract tests for CLI cache management commands.

Tests the CLI cache commands according to the CLI specification.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
import subprocess
import json
import tempfile
import os
from click.testing import CliRunner
from omnicache.cli.main import cli


class TestCLICacheCommands:
    """Test CLI cache management commands according to specification."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_cli_cache_create_basic(self, runner):
        """Test creating a cache with basic parameters."""
        result = runner.invoke(cli, [
            'cache', 'create', 'test-cache'
        ])

        assert result.exit_code == 0
        assert 'Cache "test-cache" created successfully' in result.output

    def test_cli_cache_create_with_strategy(self, runner):
        """Test creating a cache with specific strategy."""
        result = runner.invoke(cli, [
            'cache', 'create', 'lru-cache',
            '--strategy', 'lru',
            '--max-size', '1000'
        ])

        assert result.exit_code == 0
        assert 'Cache "lru-cache" created successfully' in result.output
        assert 'Strategy: LRU' in result.output or 'lru' in result.output.lower()

    def test_cli_cache_create_with_ttl(self, runner):
        """Test creating a cache with default TTL."""
        result = runner.invoke(cli, [
            'cache', 'create', 'ttl-cache',
            '--default-ttl', '300'
        ])

        assert result.exit_code == 0
        assert 'Cache "ttl-cache" created successfully' in result.output

    def test_cli_cache_create_with_namespace(self, runner):
        """Test creating a cache with namespace."""
        result = runner.invoke(cli, [
            'cache', 'create', 'namespaced-cache',
            '--namespace', 'tenant-a'
        ])

        assert result.exit_code == 0
        assert 'Cache "namespaced-cache" created successfully' in result.output

    def test_cli_cache_create_with_redis_backend(self, runner):
        """Test creating a cache with Redis backend."""
        result = runner.invoke(cli, [
            'cache', 'create', 'redis-cache',
            '--backend', 'redis',
            '--config', '{"host":"localhost","port":6379}'
        ])

        assert result.exit_code == 0
        assert 'Cache "redis-cache" created successfully' in result.output

    def test_cli_cache_create_duplicate_name_error(self, runner):
        """Test that creating duplicate cache names fails."""
        # Create first cache
        result1 = runner.invoke(cli, [
            'cache', 'create', 'duplicate-cache'
        ])
        assert result1.exit_code == 0

        # Attempt to create duplicate
        result2 = runner.invoke(cli, [
            'cache', 'create', 'duplicate-cache'
        ])
        assert result2.exit_code != 0
        assert 'already exists' in result2.output

    def test_cli_cache_create_invalid_strategy_error(self, runner):
        """Test that invalid strategy names are rejected."""
        result = runner.invoke(cli, [
            'cache', 'create', 'invalid-strategy-cache',
            '--strategy', 'invalid-strategy'
        ])

        assert result.exit_code != 0
        assert 'Invalid strategy' in result.output or 'invalid-strategy' in result.output

    def test_cli_cache_list_empty(self, runner):
        """Test listing caches when none exist."""
        # Clear any existing caches first
        runner.invoke(cli, ['cache', 'list', '--clear-all'])

        result = runner.invoke(cli, ['cache', 'list'])

        assert result.exit_code == 0
        assert 'No caches found' in result.output or result.output.strip() == ''

    def test_cli_cache_list_table_format(self, runner):
        """Test listing caches in table format."""
        # Create test caches
        runner.invoke(cli, ['cache', 'create', 'list-test-1'])
        runner.invoke(cli, ['cache', 'create', 'list-test-2', '--strategy', 'lru'])

        result = runner.invoke(cli, ['cache', 'list', '--format', 'table'])

        assert result.exit_code == 0
        assert 'list-test-1' in result.output
        assert 'list-test-2' in result.output
        # Should have table headers
        assert 'Name' in result.output or 'Cache' in result.output

    def test_cli_cache_list_json_format(self, runner):
        """Test listing caches in JSON format."""
        # Create test cache
        runner.invoke(cli, ['cache', 'create', 'json-test-cache'])

        result = runner.invoke(cli, ['cache', 'list', '--format', 'json'])

        assert result.exit_code == 0

        # Parse JSON output
        try:
            cache_list = json.loads(result.output)
            assert isinstance(cache_list, list)
            assert any(cache['name'] == 'json-test-cache' for cache in cache_list)
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output: {result.output}")

    def test_cli_cache_list_namespace_filter(self, runner):
        """Test listing caches filtered by namespace."""
        # Create caches in different namespaces
        runner.invoke(cli, ['cache', 'create', 'ns-cache-1', '--namespace', 'tenant-a'])
        runner.invoke(cli, ['cache', 'create', 'ns-cache-2', '--namespace', 'tenant-b'])
        runner.invoke(cli, ['cache', 'create', 'ns-cache-3'])  # Default namespace

        # Filter by tenant-a
        result = runner.invoke(cli, ['cache', 'list', '--namespace', 'tenant-a'])

        assert result.exit_code == 0
        assert 'ns-cache-1' in result.output
        assert 'ns-cache-2' not in result.output

    def test_cli_cache_info_existing_cache(self, runner):
        """Test getting info for an existing cache."""
        # Create test cache
        runner.invoke(cli, [
            'cache', 'create', 'info-test-cache',
            '--strategy', 'lru',
            '--max-size', '500',
            '--default-ttl', '300'
        ])

        result = runner.invoke(cli, ['cache', 'info', 'info-test-cache'])

        assert result.exit_code == 0
        assert 'info-test-cache' in result.output
        assert 'lru' in result.output.lower()
        assert '500' in result.output
        assert '300' in result.output

    def test_cli_cache_info_nonexistent_cache(self, runner):
        """Test getting info for a non-existent cache."""
        result = runner.invoke(cli, ['cache', 'info', 'nonexistent-cache'])

        assert result.exit_code != 0
        assert 'not found' in result.output.lower()

    def test_cli_cache_info_json_format(self, runner):
        """Test getting cache info in JSON format."""
        runner.invoke(cli, ['cache', 'create', 'json-info-cache'])

        result = runner.invoke(cli, ['cache', 'info', 'json-info-cache', '--format', 'json'])

        assert result.exit_code == 0

        # Parse JSON output
        try:
            cache_info = json.loads(result.output)
            assert cache_info['name'] == 'json-info-cache'
            assert 'status' in cache_info
            assert 'created_at' in cache_info
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output: {result.output}")

    def test_cli_cache_delete_existing_cache(self, runner):
        """Test deleting an existing cache."""
        # Create cache to delete
        runner.invoke(cli, ['cache', 'create', 'delete-test-cache'])

        result = runner.invoke(cli, ['cache', 'delete', 'delete-test-cache', '--force'])

        assert result.exit_code == 0
        assert 'deleted successfully' in result.output

        # Verify cache is gone
        list_result = runner.invoke(cli, ['cache', 'list'])
        assert 'delete-test-cache' not in list_result.output

    def test_cli_cache_delete_nonexistent_cache(self, runner):
        """Test deleting a non-existent cache."""
        result = runner.invoke(cli, ['cache', 'delete', 'nonexistent-cache', '--force'])

        assert result.exit_code != 0
        assert 'not found' in result.output.lower()

    def test_cli_cache_delete_confirmation_prompt(self, runner):
        """Test that delete command prompts for confirmation when --force not used."""
        runner.invoke(cli, ['cache', 'create', 'prompt-test-cache'])

        # Should prompt for confirmation
        result = runner.invoke(cli, ['cache', 'delete', 'prompt-test-cache'], input='n\n')

        assert result.exit_code == 0
        assert 'cancelled' in result.output.lower() or 'aborted' in result.output.lower()

        # Cache should still exist
        list_result = runner.invoke(cli, ['cache', 'list'])
        assert 'prompt-test-cache' in list_result.output

    def test_cli_cache_delete_confirmation_yes(self, runner):
        """Test cache deletion when user confirms."""
        runner.invoke(cli, ['cache', 'create', 'confirm-test-cache'])

        # Confirm deletion
        result = runner.invoke(cli, ['cache', 'delete', 'confirm-test-cache'], input='y\n')

        assert result.exit_code == 0
        assert 'deleted successfully' in result.output

    def test_cli_global_options_verbose(self, runner):
        """Test global --verbose option."""
        result = runner.invoke(cli, ['--verbose', 'cache', 'create', 'verbose-test'])

        assert result.exit_code == 0
        # Verbose output should contain more details
        assert len(result.output) > 50  # More verbose than standard output

    def test_cli_global_options_quiet(self, runner):
        """Test global --quiet option."""
        result = runner.invoke(cli, ['--quiet', 'cache', 'create', 'quiet-test'])

        # Should succeed with minimal output
        assert result.exit_code == 0
        # Quiet output should be minimal
        assert len(result.output) < 50 or result.output.strip() == ''

    def test_cli_help_messages(self, runner):
        """Test that help messages are available."""
        # Main help
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'cache' in result.output

        # Cache command help
        result = runner.invoke(cli, ['cache', '--help'])
        assert result.exit_code == 0
        assert 'create' in result.output
        assert 'list' in result.output
        assert 'delete' in result.output

        # Create subcommand help
        result = runner.invoke(cli, ['cache', 'create', '--help'])
        assert result.exit_code == 0
        assert '--strategy' in result.output

    def test_cli_exit_codes(self, runner):
        """Test that appropriate exit codes are returned."""
        # Success case
        result = runner.invoke(cli, ['cache', 'create', 'exit-code-test'])
        assert result.exit_code == 0

        # Error case - invalid command
        result = runner.invoke(cli, ['cache', 'invalid-command'])
        assert result.exit_code != 0

        # Error case - missing required argument
        result = runner.invoke(cli, ['cache', 'create'])
        assert result.exit_code != 0

    def test_cli_config_file_support(self, runner):
        """Test that CLI supports configuration files."""
        # Create temporary config file
        config_data = {
            "default_strategy": "lru",
            "default_max_size": 1000,
            "default_backend": "memory"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            result = runner.invoke(cli, [
                '--config', config_file,
                'cache', 'create', 'config-test-cache'
            ])

            assert result.exit_code == 0
            assert 'config-test-cache' in result.output

        finally:
            os.unlink(config_file)