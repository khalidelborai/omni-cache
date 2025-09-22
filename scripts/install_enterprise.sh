#!/bin/bash
"""
Enterprise dependencies installation script.

This script installs all enterprise dependencies required for OmniCache advanced features.
"""

set -e

echo "🚀 Installing OmniCache Enterprise Dependencies..."
echo "=================================================="

echo "📦 Installing enterprise package group..."
pip install -e ".[enterprise]"

echo ""
echo "🔍 Validating installation..."
python scripts/validate_enterprise_deps.py

echo ""
echo "✅ Enterprise installation complete!"
echo "🎯 You can now use advanced OmniCache features including:"
echo "   • ARC (Adaptive Replacement Cache) strategy"
echo "   • Hierarchical multi-tier caching"
echo "   • ML-powered predictive prefetching"
echo "   • Zero-trust security with encryption"
echo "   • Real-time analytics and monitoring"
echo "   • Event-driven cache invalidation"