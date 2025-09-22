"""
Integration test for CLI management scenario.

Tests the complete CLI workflow from the quickstart guide.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
from click.testing import CliRunner
from omnicache.cli.main import cli


class TestCLIIntegration:
    """Test CLI integration according to quickstart scenario 3."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_complete_cli_workflow(self, runner):
        """Test the complete CLI workflow from quickstart guide."""
        # Create new cache
        result = runner.invoke(cli, [
            'cache', 'create', 'web-cache', 
            '--strategy', 'lru', 
            '--max-size', '10000'
        ])
        assert result.exit_code == 0

        # List all caches
        result = runner.invoke(cli, ['cache', 'list'])
        assert result.exit_code == 0
        assert 'web-cache' in result.output

        # Get cache information  
        result = runner.invoke(cli, ['cache', 'info', 'web-cache'])
        assert result.exit_code == 0
        assert 'web-cache' in result.output
        assert 'LRU' in result.output or 'lru' in result.output

        # Store cache entry
        result = runner.invoke(cli, [
            'entry', 'set', 'web-cache', 'user:456',
            '{"name":"Alice","role":"admin"}',
            '--ttl', '3600'
        ])
        assert result.exit_code == 0

        # Retrieve entry
        result = runner.invoke(cli, ['entry', 'get', 'web-cache', 'user:456'])
        assert result.exit_code == 0
        assert 'Alice' in result.output

        # View real-time statistics
        result = runner.invoke(cli, ['stats', 'web-cache'])
        assert result.exit_code == 0
        assert 'web-cache' in result.output