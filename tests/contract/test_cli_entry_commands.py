"""
Contract tests for CLI entry management commands.

Tests the CLI entry commands according to the CLI specification.
These tests MUST FAIL until the implementation is complete.
"""

import pytest
import json
from click.testing import CliRunner
from omnicache.cli.main import cli


class TestCLIEntryCommands:
    """Test CLI entry management commands according to specification."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cache(self, runner):
        """Set up a test cache for entry operations."""
        runner.invoke(cli, ['cache', 'create', 'entry-test-cache'])
        return 'entry-test-cache'

    def test_cli_entry_set_basic(self, runner, setup_cache):
        """Test setting a basic cache entry."""
        result = runner.invoke(cli, [
            'entry', 'set', setup_cache, 'test-key', 'test-value'
        ])

        assert result.exit_code == 0
        assert 'Entry set successfully' in result.output

    def test_cli_entry_set_with_ttl(self, runner, setup_cache):
        """Test setting an entry with TTL."""
        result = runner.invoke(cli, [
            'entry', 'set', setup_cache, 'ttl-key', 'ttl-value',
            '--ttl', '300'
        ])

        assert result.exit_code == 0
        assert 'TTL: 300' in result.output or 'ttl' in result.output.lower()

    def test_cli_entry_set_with_tags(self, runner, setup_cache):
        """Test setting an entry with tags."""
        result = runner.invoke(cli, [
            'entry', 'set', setup_cache, 'tagged-key', 'tagged-value',
            '--tags', 'tag1,tag2,session'
        ])

        assert result.exit_code == 0
        assert 'Entry set successfully' in result.output

    def test_cli_entry_set_json_value(self, runner, setup_cache):
        """Test setting an entry with JSON value."""
        json_value = '{"name":"John","age":30,"active":true}'
        result = runner.invoke(cli, [
            'entry', 'set', setup_cache, 'json-key', json_value
        ])

        assert result.exit_code == 0
        assert 'Entry set successfully' in result.output

    def test_cli_entry_get_existing(self, runner, setup_cache):
        """Test getting an existing entry."""
        # Set up test data
        runner.invoke(cli, ['entry', 'set', setup_cache, 'get-test-key', 'get-test-value'])

        result = runner.invoke(cli, ['entry', 'get', setup_cache, 'get-test-key'])

        assert result.exit_code == 0
        assert 'get-test-value' in result.output

    def test_cli_entry_get_nonexistent(self, runner, setup_cache):
        """Test getting a non-existent entry."""
        result = runner.invoke(cli, ['entry', 'get', setup_cache, 'nonexistent-key'])

        assert result.exit_code != 0
        assert 'not found' in result.output.lower()

    def test_cli_entry_get_json_format(self, runner, setup_cache):
        """Test getting entry in JSON format."""
        # Set up test data
        runner.invoke(cli, ['entry', 'set', setup_cache, 'json-get-key', 'json-get-value'])

        result = runner.invoke(cli, [
            'entry', 'get', setup_cache, 'json-get-key', '--format', 'json'
        ])

        assert result.exit_code == 0

        # Parse JSON output
        try:
            entry_data = json.loads(result.output)
            assert entry_data['key'] == 'json-get-key'
            assert entry_data['value'] == 'json-get-value'
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output: {result.output}")

    def test_cli_entry_delete_existing(self, runner, setup_cache):
        """Test deleting an existing entry."""
        # Set up test data
        runner.invoke(cli, ['entry', 'set', setup_cache, 'delete-key', 'delete-value'])

        result = runner.invoke(cli, ['entry', 'delete', setup_cache, 'delete-key'])

        assert result.exit_code == 0
        assert 'deleted successfully' in result.output

        # Verify entry is gone
        get_result = runner.invoke(cli, ['entry', 'get', setup_cache, 'delete-key'])
        assert get_result.exit_code != 0

    def test_cli_entry_list_all(self, runner, setup_cache):
        """Test listing all entries in a cache."""
        # Set up test data
        runner.invoke(cli, ['entry', 'set', setup_cache, 'list-key1', 'value1'])
        runner.invoke(cli, ['entry', 'set', setup_cache, 'list-key2', 'value2'])

        result = runner.invoke(cli, ['entry', 'list', setup_cache])

        assert result.exit_code == 0
        assert 'list-key1' in result.output
        assert 'list-key2' in result.output

    def test_cli_entry_list_with_pattern(self, runner, setup_cache):
        """Test listing entries with pattern filter."""
        # Set up test data
        runner.invoke(cli, ['entry', 'set', setup_cache, 'user:1:profile', 'profile1'])
        runner.invoke(cli, ['entry', 'set', setup_cache, 'user:1:settings', 'settings1'])
        runner.invoke(cli, ['entry', 'set', setup_cache, 'admin:config', 'config1'])

        result = runner.invoke(cli, [
            'entry', 'list', setup_cache, '--pattern', 'user:1:*'
        ])

        assert result.exit_code == 0
        assert 'user:1:profile' in result.output
        assert 'user:1:settings' in result.output
        assert 'admin:config' not in result.output

    def test_cli_entry_list_with_limit(self, runner, setup_cache):
        """Test listing entries with limit."""
        # Set up test data
        for i in range(10):
            runner.invoke(cli, ['entry', 'set', setup_cache, f'limit-key{i}', f'value{i}'])

        result = runner.invoke(cli, [
            'entry', 'list', setup_cache, '--limit', '5'
        ])

        assert result.exit_code == 0
        # Should only show 5 entries
        output_lines = [line for line in result.output.split('\n') if 'limit-key' in line]
        assert len(output_lines) <= 5

    def test_cli_entry_list_keys_only(self, runner, setup_cache):
        """Test listing only keys without values."""
        runner.invoke(cli, ['entry', 'set', setup_cache, 'keys-only-test', 'some-value'])

        result = runner.invoke(cli, [
            'entry', 'list', setup_cache, '--format', 'keys-only'
        ])

        assert result.exit_code == 0
        assert 'keys-only-test' in result.output
        assert 'some-value' not in result.output

    def test_cli_stdin_support_for_set(self, runner, setup_cache):
        """Test setting entry value from stdin."""
        stdin_value = '{"from": "stdin", "data": "test"}'

        result = runner.invoke(cli, [
            'entry', 'set', setup_cache, 'stdin-key'
        ], input=stdin_value)

        assert result.exit_code == 0
        assert 'Entry set successfully' in result.output

        # Verify value was set correctly
        get_result = runner.invoke(cli, ['entry', 'get', setup_cache, 'stdin-key'])
        assert stdin_value in get_result.output

    def test_cli_error_handling_invalid_cache(self, runner):
        """Test error handling for operations on non-existent cache."""
        result = runner.invoke(cli, [
            'entry', 'set', 'nonexistent-cache', 'key', 'value'
        ])

        assert result.exit_code != 0
        assert 'not found' in result.output.lower()

    def test_cli_error_handling_empty_key(self, runner, setup_cache):
        """Test error handling for empty key."""
        result = runner.invoke(cli, [
            'entry', 'set', setup_cache, '', 'value'
        ])

        assert result.exit_code != 0
        assert 'empty' in result.output.lower() or 'invalid' in result.output.lower()

    def test_cli_error_handling_invalid_ttl(self, runner, setup_cache):
        """Test error handling for invalid TTL values."""
        result = runner.invoke(cli, [
            'entry', 'set', setup_cache, 'key', 'value', '--ttl', '-1'
        ])

        assert result.exit_code != 0
        assert 'invalid' in result.output.lower() or 'positive' in result.output.lower()

    def test_cli_entry_operations_with_special_characters(self, runner, setup_cache):
        """Test entry operations with special characters in keys/values."""
        special_key = 'key:with/special@chars#and-spaces'
        special_value = 'value with "quotes" and \n newlines'

        # Set entry with special characters
        result = runner.invoke(cli, [
            'entry', 'set', setup_cache, special_key, special_value
        ])

        assert result.exit_code == 0

        # Get entry with special characters
        result = runner.invoke(cli, [
            'entry', 'get', setup_cache, special_key
        ])

        assert result.exit_code == 0
        assert 'quotes' in result.output

    def test_cli_entry_priority_setting(self, runner, setup_cache):
        """Test setting entry with priority."""
        result = runner.invoke(cli, [
            'entry', 'set', setup_cache, 'priority-key', 'priority-value',
            '--priority', '0.8'
        ])

        assert result.exit_code == 0
        assert 'Entry set successfully' in result.output

    def test_cli_entry_confirmation_for_delete(self, runner, setup_cache):
        """Test confirmation prompt for entry deletion."""
        # Set up test data
        runner.invoke(cli, ['entry', 'set', setup_cache, 'confirm-delete-key', 'value'])

        # Should prompt for confirmation without --force
        result = runner.invoke(cli, [
            'entry', 'delete', setup_cache, 'confirm-delete-key'
        ], input='n\n')

        assert result.exit_code == 0
        assert 'cancelled' in result.output.lower() or 'aborted' in result.output.lower()

        # Entry should still exist
        get_result = runner.invoke(cli, ['entry', 'get', setup_cache, 'confirm-delete-key'])
        assert get_result.exit_code == 0