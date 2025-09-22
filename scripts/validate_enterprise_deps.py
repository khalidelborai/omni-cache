#!/usr/bin/env python3
"""
Enterprise dependencies validation script.

This script validates that all required enterprise dependencies are installed
and functioning correctly for OmniCache advanced features.
"""

import sys
from typing import List, Tuple


def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True, f"✓ {package_name or module_name}"
    except ImportError as e:
        return False, f"✗ {package_name or module_name}: {str(e)}"


def validate_ml_dependencies() -> List[Tuple[bool, str]]:
    """Validate ML and analytics dependencies."""
    return [
        check_import("sklearn", "scikit-learn"),
        check_import("numpy", "numpy"),
        check_import("torch", "torch"),
        check_import("prometheus_client", "prometheus-client"),
        check_import("opentelemetry.api", "opentelemetry-api"),
        check_import("opentelemetry.sdk", "opentelemetry-sdk"),
        check_import("opentelemetry.instrumentation", "opentelemetry-instrumentation"),
    ]


def validate_security_dependencies() -> List[Tuple[bool, str]]:
    """Validate security and encryption dependencies."""
    return [
        check_import("cryptography", "cryptography"),
        check_import("spacy", "spacy"),
    ]


def validate_cloud_dependencies() -> List[Tuple[bool, str]]:
    """Validate cloud storage dependencies."""
    return [
        check_import("boto3", "boto3"),
        check_import("azure.storage.blob", "azure-storage-blob"),
        check_import("google.cloud.storage", "google-cloud-storage"),
    ]


def validate_event_dependencies() -> List[Tuple[bool, str]]:
    """Validate event processing dependencies."""
    return [
        check_import("kafka", "kafka-python"),
        check_import("asyncio_mqtt", "asyncio-mqtt"),
    ]


def validate_analytics_dependencies() -> List[Tuple[bool, str]]:
    """Validate analytics and monitoring dependencies."""
    return [
        check_import("grafana_api", "grafana-api"),
        check_import("pydantic", "pydantic"),
    ]


def main() -> int:
    """Main validation function."""
    print("🔍 Validating OmniCache Enterprise Dependencies...")
    print("=" * 50)

    all_checks = []

    print("\n📊 ML and Analytics Dependencies:")
    ml_checks = validate_ml_dependencies()
    all_checks.extend(ml_checks)
    for success, message in ml_checks:
        print(f"  {message}")

    print("\n🔒 Security Dependencies:")
    security_checks = validate_security_dependencies()
    all_checks.extend(security_checks)
    for success, message in security_checks:
        print(f"  {message}")

    print("\n☁️  Cloud Storage Dependencies:")
    cloud_checks = validate_cloud_dependencies()
    all_checks.extend(cloud_checks)
    for success, message in cloud_checks:
        print(f"  {message}")

    print("\n⚡ Event Processing Dependencies:")
    event_checks = validate_event_dependencies()
    all_checks.extend(event_checks)
    for success, message in event_checks:
        print(f"  {message}")

    print("\n📈 Analytics Dependencies:")
    analytics_checks = validate_analytics_dependencies()
    all_checks.extend(analytics_checks)
    for success, message in analytics_checks:
        print(f"  {message}")

    # Summary
    successful = sum(1 for success, _ in all_checks if success)
    total = len(all_checks)

    print("\n" + "=" * 50)
    print(f"📋 Summary: {successful}/{total} dependencies available")

    if successful == total:
        print("🎉 All enterprise dependencies are correctly installed!")
        return 0
    else:
        print("⚠️  Some enterprise dependencies are missing.")
        print("💡 Install missing dependencies with: pip install omnicache[enterprise]")
        return 1


if __name__ == "__main__":
    sys.exit(main())