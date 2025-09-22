"""
CLI output formatting utilities.

Provides formatters for different output formats (table, JSON, YAML).
"""

import json
import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def format_output(
    data: Any,
    output_format: str = "table",
    headers: Optional[List[str]] = None,
    title: Optional[str] = None
) -> str:
    """
    Format data for output in specified format.

    Args:
        data: Data to format
        output_format: Output format (table, json, yaml)
        headers: Column headers for table format
        title: Optional title for output

    Returns:
        Formatted string
    """
    if output_format == "json":
        return format_json(data)
    elif output_format == "yaml":
        return format_yaml(data)
    elif output_format == "table":
        return format_table(data, headers, title)
    else:
        # Fallback to table format
        return format_table(data, headers, title)


def format_json(data: Any) -> str:
    """Format data as JSON."""
    def json_serializer(obj):
        """Custom JSON serializer for non-serializable objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, indent=2, default=json_serializer, ensure_ascii=False)


def format_yaml(data: Any) -> str:
    """Format data as YAML."""
    if not HAS_YAML:
        return "YAML output requires PyYAML package. Install with: pip install PyYAML"

    def yaml_representer(dumper, data):
        """Custom YAML representer for datetime objects."""
        if isinstance(data, datetime):
            return dumper.represent_scalar('tag:yaml.org,2002:timestamp', data.isoformat())
        return dumper.represent_data(data)

    yaml.add_representer(datetime, yaml_representer)
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def format_table(
    data: Any,
    headers: Optional[List[str]] = None,
    title: Optional[str] = None
) -> str:
    """
    Format data as a table.

    Args:
        data: Data to format (list of dicts, single dict, or simple values)
        headers: Column headers
        title: Table title

    Returns:
        Formatted table string
    """
    if not data:
        return "No data available"

    output_lines = []

    # Add title if provided
    if title:
        output_lines.append(title)
        output_lines.append("=" * len(title))
        output_lines.append("")

    # Handle different data types
    if isinstance(data, list):
        if not data:
            return "No items found"

        # Check if it's a list of dictionaries
        if isinstance(data[0], dict):
            return _format_table_from_dicts(data, headers)
        else:
            # Simple list
            for item in data:
                output_lines.append(str(item))
    elif isinstance(data, dict):
        return _format_table_from_dict(data, headers)
    else:
        # Simple value
        output_lines.append(str(data))

    return "\n".join(output_lines)


def _format_table_from_dicts(data: List[Dict[str, Any]], headers: Optional[List[str]] = None) -> str:
    """Format list of dictionaries as table."""
    if not data:
        return "No data available"

    # Determine headers
    if headers is None:
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        headers = sorted(all_keys)

    # Calculate column widths
    col_widths = {}
    for header in headers:
        col_widths[header] = max(
            len(header),
            max(len(str(item.get(header, ""))) for item in data) if data else 0
        )

    # Build table
    lines = []

    # Header row
    header_row = " | ".join(header.ljust(col_widths[header]) for header in headers)
    lines.append(header_row)

    # Separator row
    separator = " | ".join("-" * col_widths[header] for header in headers)
    lines.append(separator)

    # Data rows
    for item in data:
        row = " | ".join(
            str(item.get(header, "")).ljust(col_widths[header])
            for header in headers
        )
        lines.append(row)

    return "\n".join(lines)


def _format_table_from_dict(data: Dict[str, Any], headers: Optional[List[str]] = None) -> str:
    """Format single dictionary as table."""
    if not data:
        return "No data available"

    # Create two-column table: Key | Value
    lines = []

    # Calculate column widths
    key_width = max(len("Key"), max(len(str(k)) for k in data.keys()) if data else 0)
    value_width = max(len("Value"), max(len(str(v)) for v in data.values()) if data else 0)

    # Header
    header = f"{'Key'.ljust(key_width)} | {'Value'.ljust(value_width)}"
    lines.append(header)

    # Separator
    separator = f"{'-' * key_width} | {'-' * value_width}"
    lines.append(separator)

    # Data rows
    for key, value in data.items():
        # Handle complex values
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, default=str)
        elif isinstance(value, datetime):
            value_str = value.isoformat()
        else:
            value_str = str(value)

        # Truncate very long values
        if len(value_str) > 60:
            value_str = value_str[:57] + "..."

        row = f"{str(key).ljust(key_width)} | {value_str.ljust(value_width)}"
        lines.append(row)

    return "\n".join(lines)


def format_cache_list(caches: List[Dict[str, Any]]) -> str:
    """Format cache list for display."""
    if not caches:
        return "No caches found"

    # Define display columns and their order
    display_headers = [
        "name", "status", "strategy", "backend", "entries",
        "namespace", "created_at", "last_accessed"
    ]

    # Transform data for display
    display_data = []
    for cache in caches:
        display_item = {}
        for header in display_headers:
            value = cache.get(header, "")

            # Format specific fields
            if header in ["created_at", "last_accessed"] and value:
                try:
                    if isinstance(value, str):
                        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        value = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            elif header == "entries":
                # Show entry count if available in cache stats
                value = cache.get("entry_count", cache.get("size", ""))

            display_item[header] = value

        display_data.append(display_item)

    return _format_table_from_dicts(display_data, display_headers)


def format_cache_info(cache_info: Dict[str, Any]) -> str:
    """Format detailed cache information."""
    if not cache_info:
        return "No cache information available"

    # Organize information into sections
    basic_info = {
        "Name": cache_info.get("name", ""),
        "Status": cache_info.get("status", ""),
        "Strategy": cache_info.get("strategy", ""),
        "Backend": cache_info.get("backend", ""),
        "Namespace": cache_info.get("namespace", ""),
        "Max Size": cache_info.get("max_size", "unlimited"),
        "Default TTL": cache_info.get("default_ttl", "none")
    }

    # Statistics section
    stats = cache_info.get("statistics", {})
    stats_info = {
        "Total Entries": stats.get("entry_count", 0),
        "Hit Rate": f"{stats.get('hit_rate', 0):.2%}",
        "Total Hits": stats.get("total_hits", 0),
        "Total Misses": stats.get("total_misses", 0),
        "Memory Usage": _format_bytes(stats.get("memory_usage", 0))
    }

    # Lifecycle information
    lifecycle_info = {
        "Created At": cache_info.get("created_at", ""),
        "Last Accessed": cache_info.get("last_accessed", ""),
        "Access Count": cache_info.get("access_count", 0)
    }

    sections = [
        ("Basic Information", basic_info),
        ("Statistics", stats_info),
        ("Lifecycle", lifecycle_info)
    ]

    output_lines = []
    for section_title, section_data in sections:
        output_lines.append(f"\n{section_title}:")
        output_lines.append("-" * len(section_title))
        output_lines.append(_format_table_from_dict(section_data))

    return "\n".join(output_lines).strip()


def format_entry_list(entries: List[Dict[str, Any]], keys_only: bool = False) -> str:
    """Format cache entry list."""
    if not entries:
        return "No entries found"

    if keys_only:
        return "\n".join(entry.get("key", "") for entry in entries)

    # Define display columns
    display_headers = ["key", "value", "ttl", "created_at", "access_count"]

    # Transform data for display
    display_data = []
    for entry in entries:
        display_item = {}
        for header in display_headers:
            value = entry.get(header, "")

            # Format specific fields
            if header == "value":
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                value = value_str
            elif header == "ttl":
                if value is None:
                    value = "none"
                elif isinstance(value, (int, float)):
                    value = f"{value}s"
            elif header == "created_at" and value:
                try:
                    if isinstance(value, str):
                        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        value = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass

            display_item[header] = value

        display_data.append(display_item)

    return _format_table_from_dicts(display_data, display_headers)


def _format_bytes(bytes_value: Union[int, float]) -> str:
    """Format bytes value in human-readable format."""
    if bytes_value == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(bytes_value)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.1f} {units[unit_index]}"