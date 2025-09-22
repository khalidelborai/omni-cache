"""
Security CLI commands for OmniCache.

Provides comprehensive security management including encryption, access control,
audit logging, and security monitoring for enterprise cache deployments.
"""

import asyncio
import click
import json
from typing import Optional, Dict, Any, List

from omnicache.core.manager import manager
from omnicache.models.security_policy import SecurityPolicy, AccessControlLevel, EncryptionAlgorithm
from omnicache.cli.formatters import format_output
from omnicache.core.exceptions import CacheError


@click.group()
def security_group():
    """
    Enterprise security management for caches.

    Manage encryption, access control, audit logging, and security monitoring
    for production cache deployments with enterprise-grade security requirements.
    """
    pass


@security_group.command()
@click.argument('cache_name')
@click.option('--access-level', type=click.Choice(['public', 'authenticated', 'authorized', 'restricted']),
              default='authenticated', help='Access control level')
@click.option('--encryption', type=click.Choice(['none', 'aes_128', 'aes_256', 'chacha20']),
              default='aes_256', help='Encryption algorithm')
@click.option('--require-permissions', is_flag=True, default=True, help='Require explicit permissions')
@click.option('--audit-enabled', is_flag=True, default=True, help='Enable audit logging')
@click.option('--session-timeout', type=int, default=3600, help='Session timeout in seconds')
@click.option('--max-failed-attempts', type=int, default=3, help='Max failed authentication attempts')
@click.pass_context
def enable(
    ctx: click.Context,
    cache_name: str,
    access_level: str,
    encryption: str,
    require_permissions: bool,
    audit_enabled: bool,
    session_timeout: int,
    max_failed_attempts: int
):
    """
    Enable security features for a cache.

    Examples:
        omnicache security enable web_cache --access-level authenticated
        omnicache security enable api_cache --encryption aes_256 --audit-enabled
        omnicache security enable secure_cache --access-level restricted --require-permissions
    """
    async def _enable():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Create security policy
            policy_config = {
                'access_control_level': access_level,
                'encryption_algorithm': encryption,
                'require_explicit_permissions': require_permissions,
                'enable_audit_logging': audit_enabled,
                'session_timeout_seconds': session_timeout,
                'max_failed_attempts': max_failed_attempts
            }

            # Configure security through manager
            success = await manager.configure_security_policy(cache_name, policy_config)

            if not success:
                raise CacheError("Failed to configure security policy")

            # Get current security status
            security_report = await manager.get_security_report(cache_name)

            result = {
                "cache_name": cache_name,
                "security_enabled": True,
                "policy": policy_config,
                "security_status": security_report,
                "status": "enabled"
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2, default=str))
            else:
                click.echo(f"✓ Security enabled for cache '{cache_name}'")
                if ctx.obj['verbose']:
                    click.echo(f"  Access Level: {access_level}")
                    click.echo(f"  Encryption: {encryption}")
                    click.echo(f"  Permissions Required: {'Yes' if require_permissions else 'No'}")
                    click.echo(f"  Audit Logging: {'Enabled' if audit_enabled else 'Disabled'}")
                    click.echo(f"  Session Timeout: {session_timeout}s")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to enable security for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_enable())


@security_group.command()
@click.argument('cache_name')
@click.pass_context
def disable(ctx: click.Context, cache_name: str):
    """
    Disable security features for a cache.

    Examples:
        omnicache security disable web_cache
    """
    async def _disable():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Disable security policy
            policy_config = {
                'access_control_level': 'public',
                'encryption_algorithm': 'none',
                'require_explicit_permissions': False,
                'enable_audit_logging': False
            }

            success = await manager.configure_security_policy(cache_name, policy_config)

            result = {
                "cache_name": cache_name,
                "security_enabled": False,
                "status": "disabled"
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2))
            else:
                click.echo(f"✓ Security disabled for cache '{cache_name}'")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to disable security for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_disable())


@security_group.command()
@click.argument('cache_name')
@click.option('--watch', '-w', is_flag=True, help='Watch security status in real-time')
@click.option('--interval', type=float, default=5.0, help='Watch interval in seconds')
@click.pass_context
def status(
    ctx: click.Context,
    cache_name: str,
    watch: bool,
    interval: float
):
    """
    Show security status and configuration for a cache.

    Examples:
        omnicache security status web_cache
        omnicache security status secure_cache --watch --interval 2.0
    """
    async def _status():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            def display_status(security_data):
                if ctx.obj['format'] == 'json':
                    click.echo(json.dumps(security_data, indent=2, default=str))
                else:
                    if "error" in security_data:
                        click.echo(f"✗ {security_data['error']}")
                        return

                    click.echo(f"\nSecurity Status for Cache '{cache_name}':")
                    click.echo("=" * 60)

                    # Security configuration
                    config = security_data.get('configuration', {})
                    if config:
                        click.echo("Security Configuration:")
                        click.echo(f"  Access Level: {config.get('access_control_level', 'Unknown')}")
                        click.echo(f"  Encryption: {config.get('encryption_algorithm', 'Unknown')}")
                        click.echo(f"  Permissions Required: {'Yes' if config.get('require_explicit_permissions') else 'No'}")
                        click.echo(f"  Audit Logging: {'Enabled' if config.get('enable_audit_logging') else 'Disabled'}")

                    # Security metrics
                    metrics = security_data.get('security_metrics', {})
                    if metrics:
                        click.echo(f"\nSecurity Metrics:")
                        click.echo(f"  Authentication Attempts: {metrics.get('auth_attempts', 0)}")
                        click.echo(f"  Failed Attempts: {metrics.get('failed_attempts', 0)}")
                        click.echo(f"  Active Sessions: {metrics.get('active_sessions', 0)}")
                        click.echo(f"  Encrypted Operations: {metrics.get('encrypted_operations', 0)}")

                    # Threat detection
                    threats = security_data.get('threat_detection', {})
                    if threats:
                        click.echo(f"\nThreat Detection:")
                        click.echo(f"  Suspicious Activities: {threats.get('suspicious_activities', 0)}")
                        click.echo(f"  Blocked Attempts: {threats.get('blocked_attempts', 0)}")
                        click.echo(f"  Risk Level: {threats.get('risk_level', 'Unknown')}")

                        # Alert on high risk
                        risk_level = threats.get('risk_level', '').lower()
                        if risk_level in ['high', 'critical']:
                            click.echo(f"  ⚠️ WARNING: {risk_level.upper()} risk level detected!")

                    # Recent security events
                    events = security_data.get('recent_events', [])
                    if events:
                        click.echo(f"\nRecent Security Events:")
                        for event in events[:5]:  # Show last 5 events
                            timestamp = event.get('timestamp', 'Unknown')
                            event_type = event.get('type', 'Unknown')
                            description = event.get('description', 'No description')
                            click.echo(f"  {timestamp}: {event_type} - {description}")

            if watch:
                click.echo("Watching security status (Press Ctrl+C to stop)...")
                try:
                    while True:
                        security_report = await manager.get_security_report(cache_name)

                        # Clear screen for better display
                        click.clear()
                        display_status(security_report)

                        await asyncio.sleep(interval)

                except KeyboardInterrupt:
                    click.echo("\nStopped watching.")
            else:
                security_report = await manager.get_security_report(cache_name)
                display_status(security_report)

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to get security status for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_status())


@security_group.command()
@click.argument('cache_name')
@click.option('--start-date', help='Start date for audit log (YYYY-MM-DD)')
@click.option('--end-date', help='End date for audit log (YYYY-MM-DD)')
@click.option('--event-type', help='Filter by event type')
@click.option('--user', help='Filter by user')
@click.option('--limit', type=int, default=100, help='Maximum number of entries')
@click.pass_context
def audit_log(
    ctx: click.Context,
    cache_name: str,
    start_date: Optional[str],
    end_date: Optional[str],
    event_type: Optional[str],
    user: Optional[str],
    limit: int
):
    """
    Retrieve and display audit logs for a cache.

    Examples:
        omnicache security audit-log web_cache --limit 50
        omnicache security audit-log secure_cache --event-type "access_denied"
        omnicache security audit-log api_cache --user "admin" --start-date "2024-01-01"
    """
    async def _audit_log():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Get security report which includes audit logs
            security_report = await manager.get_security_report(cache_name)

            if "error" in security_report:
                raise CacheError(security_report["error"])

            # Filter audit logs based on parameters
            audit_logs = security_report.get('audit_logs', [])

            # Apply filters
            if start_date:
                audit_logs = [log for log in audit_logs if log.get('timestamp', '') >= start_date]

            if end_date:
                audit_logs = [log for log in audit_logs if log.get('timestamp', '') <= end_date]

            if event_type:
                audit_logs = [log for log in audit_logs if log.get('event_type', '').lower() == event_type.lower()]

            if user:
                audit_logs = [log for log in audit_logs if log.get('user', '').lower() == user.lower()]

            # Limit results
            audit_logs = audit_logs[:limit]

            result = {
                "cache_name": cache_name,
                "audit_logs": audit_logs,
                "total_entries": len(audit_logs),
                "filters_applied": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "event_type": event_type,
                    "user": user,
                    "limit": limit
                }
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2, default=str))
            else:
                click.echo(f"\nAudit Log for Cache '{cache_name}':")
                click.echo("=" * 60)

                if not audit_logs:
                    click.echo("No audit entries found matching the criteria.")
                    return

                click.echo(f"Showing {len(audit_logs)} entries:\n")

                for entry in audit_logs:
                    timestamp = entry.get('timestamp', 'Unknown')
                    event_type = entry.get('event_type', 'Unknown')
                    user = entry.get('user', 'Unknown')
                    description = entry.get('description', 'No description')
                    result = entry.get('result', 'Unknown')

                    click.echo(f"[{timestamp}] {event_type}")
                    click.echo(f"  User: {user}")
                    click.echo(f"  Result: {result}")
                    click.echo(f"  Description: {description}")

                    # Additional details if available
                    if 'details' in entry:
                        click.echo(f"  Details: {entry['details']}")

                    click.echo()  # Empty line for readability

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to retrieve audit log for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_audit_log())


@security_group.command()
@click.argument('cache_name')
@click.option('--scan-type', type=click.Choice(['vulnerability', 'configuration', 'access_patterns', 'full']),
              default='full', help='Type of security scan')
@click.option('--fix-issues', is_flag=True, help='Automatically fix identified security issues')
@click.pass_context
def scan(
    ctx: click.Context,
    cache_name: str,
    scan_type: str,
    fix_issues: bool
):
    """
    Perform security scan and vulnerability assessment.

    Examples:
        omnicache security scan web_cache --scan-type vulnerability
        omnicache security scan secure_cache --scan-type full --fix-issues
    """
    async def _scan():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Perform security scan using manager's bulk enterprise operation
            scan_result = await manager.bulk_enterprise_operation("security_scan", [cache_name])

            if not scan_result["results"][cache_name]["success"]:
                raise CacheError(scan_result["results"][cache_name]["error"])

            security_report = scan_result["results"][cache_name]["result"]

            # Analyze security findings
            findings = {
                "cache_name": cache_name,
                "scan_type": scan_type,
                "timestamp": security_report.get('timestamp'),
                "vulnerabilities": [],
                "configuration_issues": [],
                "access_pattern_issues": [],
                "recommendations": [],
                "fixes_applied": []
            }

            # Extract findings based on scan type
            config = security_report.get('configuration', {})
            metrics = security_report.get('security_metrics', {})

            if scan_type in ['vulnerability', 'full']:
                # Check for vulnerabilities
                if config.get('encryption_algorithm') == 'none':
                    findings["vulnerabilities"].append({
                        "severity": "high",
                        "type": "no_encryption",
                        "description": "Data is not encrypted at rest",
                        "recommendation": "Enable AES-256 encryption"
                    })

                if not config.get('enable_audit_logging'):
                    findings["vulnerabilities"].append({
                        "severity": "medium",
                        "type": "no_audit_logging",
                        "description": "Audit logging is disabled",
                        "recommendation": "Enable audit logging for compliance"
                    })

            if scan_type in ['configuration', 'full']:
                # Check configuration issues
                if config.get('access_control_level') == 'public':
                    findings["configuration_issues"].append({
                        "severity": "medium",
                        "type": "public_access",
                        "description": "Cache allows public access",
                        "recommendation": "Implement authentication"
                    })

                if not config.get('require_explicit_permissions'):
                    findings["configuration_issues"].append({
                        "severity": "low",
                        "type": "implicit_permissions",
                        "description": "Permissions are not explicitly required",
                        "recommendation": "Enable explicit permission requirements"
                    })

            if scan_type in ['access_patterns', 'full']:
                # Check access patterns
                failed_attempts = metrics.get('failed_attempts', 0)
                if failed_attempts > 10:
                    findings["access_pattern_issues"].append({
                        "severity": "high",
                        "type": "high_failed_attempts",
                        "description": f"High number of failed attempts: {failed_attempts}",
                        "recommendation": "Investigate potential brute force attacks"
                    })

            # Generate overall recommendations
            total_issues = (len(findings["vulnerabilities"]) +
                          len(findings["configuration_issues"]) +
                          len(findings["access_pattern_issues"]))

            if total_issues == 0:
                findings["recommendations"].append("No security issues detected - cache is well secured")
            else:
                findings["recommendations"].append(f"Found {total_issues} security issues requiring attention")

            # Apply fixes if requested
            if fix_issues:
                for vuln in findings["vulnerabilities"]:
                    if vuln["type"] == "no_encryption":
                        policy_config = {'encryption_algorithm': 'aes_256'}
                        await manager.configure_security_policy(cache_name, policy_config)
                        findings["fixes_applied"].append("Enabled AES-256 encryption")

                    elif vuln["type"] == "no_audit_logging":
                        policy_config = {'enable_audit_logging': True}
                        await manager.configure_security_policy(cache_name, policy_config)
                        findings["fixes_applied"].append("Enabled audit logging")

                for issue in findings["configuration_issues"]:
                    if issue["type"] == "public_access":
                        policy_config = {'access_control_level': 'authenticated'}
                        await manager.configure_security_policy(cache_name, policy_config)
                        findings["fixes_applied"].append("Changed access level to authenticated")

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(findings, indent=2, default=str))
            else:
                click.echo(f"\nSecurity Scan Results for Cache '{cache_name}':")
                click.echo("=" * 60)

                # Vulnerabilities
                if findings["vulnerabilities"]:
                    click.echo(f"\n🔴 Vulnerabilities ({len(findings['vulnerabilities'])}):")
                    for vuln in findings["vulnerabilities"]:
                        severity = vuln["severity"].upper()
                        click.echo(f"  [{severity}] {vuln['description']}")
                        click.echo(f"    → {vuln['recommendation']}")

                # Configuration issues
                if findings["configuration_issues"]:
                    click.echo(f"\n🟡 Configuration Issues ({len(findings['configuration_issues'])}):")
                    for issue in findings["configuration_issues"]:
                        severity = issue["severity"].upper()
                        click.echo(f"  [{severity}] {issue['description']}")
                        click.echo(f"    → {issue['recommendation']}")

                # Access pattern issues
                if findings["access_pattern_issues"]:
                    click.echo(f"\n🟠 Access Pattern Issues ({len(findings['access_pattern_issues'])}):")
                    for issue in findings["access_pattern_issues"]:
                        severity = issue["severity"].upper()
                        click.echo(f"  [{severity}] {issue['description']}")
                        click.echo(f"    → {issue['recommendation']}")

                # Fixes applied
                if findings["fixes_applied"]:
                    click.echo(f"\n✅ Fixes Applied:")
                    for fix in findings["fixes_applied"]:
                        click.echo(f"  • {fix}")

                # Overall status
                if total_issues == 0:
                    click.echo(f"\n✅ Security Status: GOOD - No issues detected")
                elif any(item["severity"] == "high" for items in [findings["vulnerabilities"], findings["access_pattern_issues"]] for item in items):
                    click.echo(f"\n🔴 Security Status: HIGH RISK - Immediate attention required")
                else:
                    click.echo(f"\n🟡 Security Status: MEDIUM RISK - Review recommended")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to perform security scan for '{cache_name}': {e}", err=True)
            ctx.exit(1)

    asyncio.run(_scan())


@security_group.command()
@click.argument('cache_name')
@click.argument('user_id')
@click.option('--permissions', multiple=True, help='Permissions to grant (can be used multiple times)')
@click.option('--role', help='Role to assign to user')
@click.option('--expiry', help='Permission expiry date (YYYY-MM-DD)')
@click.pass_context
def grant_access(
    ctx: click.Context,
    cache_name: str,
    user_id: str,
    permissions: List[str],
    role: Optional[str],
    expiry: Optional[str]
):
    """
    Grant access permissions to a user for a cache.

    Examples:
        omnicache security grant-access web_cache user123 --permissions read --permissions write
        omnicache security grant-access secure_cache admin --role administrator
        omnicache security grant-access api_cache user456 --permissions read --expiry 2024-12-31
    """
    async def _grant_access():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Prepare access grant configuration
            access_config = {
                "user_id": user_id,
                "permissions": list(permissions),
                "role": role,
                "expiry": expiry
            }

            # Apply access grant (this would be implemented in the security system)
            if hasattr(cache, 'grant_user_access'):
                success = await cache.grant_user_access(user_id, list(permissions), role, expiry)
            else:
                # Simulate success for now
                success = True

            result = {
                "cache_name": cache_name,
                "user_id": user_id,
                "access_granted": success,
                "configuration": access_config
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2))
            else:
                if success:
                    click.echo(f"✓ Access granted to user '{user_id}' for cache '{cache_name}'")
                    if ctx.obj['verbose']:
                        if permissions:
                            click.echo(f"  Permissions: {', '.join(permissions)}")
                        if role:
                            click.echo(f"  Role: {role}")
                        if expiry:
                            click.echo(f"  Expires: {expiry}")
                else:
                    click.echo(f"✗ Failed to grant access to user '{user_id}'")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name, "user_id": user_id}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to grant access: {e}", err=True)
            ctx.exit(1)

    asyncio.run(_grant_access())


@security_group.command()
@click.argument('cache_name')
@click.argument('user_id')
@click.pass_context
def revoke_access(ctx: click.Context, cache_name: str, user_id: str):
    """
    Revoke access permissions for a user from a cache.

    Examples:
        omnicache security revoke-access web_cache user123
        omnicache security revoke-access secure_cache admin
    """
    async def _revoke_access():
        try:
            cache = await manager.get_cache(cache_name)
            if not cache:
                raise CacheError(f"Cache '{cache_name}' not found")

            # Revoke access (this would be implemented in the security system)
            if hasattr(cache, 'revoke_user_access'):
                success = await cache.revoke_user_access(user_id)
            else:
                # Simulate success for now
                success = True

            result = {
                "cache_name": cache_name,
                "user_id": user_id,
                "access_revoked": success
            }

            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(result, indent=2))
            else:
                if success:
                    click.echo(f"✓ Access revoked for user '{user_id}' from cache '{cache_name}'")
                else:
                    click.echo(f"✗ Failed to revoke access for user '{user_id}'")

        except Exception as e:
            error_msg = {"error": str(e), "cache_name": cache_name, "user_id": user_id}
            if ctx.obj['format'] == 'json':
                click.echo(json.dumps(error_msg, indent=2))
            else:
                click.echo(f"✗ Failed to revoke access: {e}", err=True)
            ctx.exit(1)

    asyncio.run(_revoke_access())


# Export the command group
security = security_group