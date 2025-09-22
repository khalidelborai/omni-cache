# omni-cache Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-09-22

## Active Technologies
- Python 3.11+ (based on pyproject.toml detection and FastAPI integration requirement) + FastAPI, AsyncIO, Click (CLI), Redis (optional backend), sqlite3 (fallback), typing-extensions (001-base-caching-librabry)
- Python 3.11+ (existing OmniCache codebase) + FastAPI (existing), Redis, scikit-learn/pytorch (ML), cryptography (encryption), prometheus-client, opentelemetry (002-implement-adaptive-replacement)
- Multi-tier: Memory (L1), Redis (L2), Cloud Storage APIs (L3) (002-implement-adaptive-replacement)

## Project Structure
```
src/
tests/
```

## Commands
cd src; pytest; ruff check .

## Code Style
Python 3.11+ (based on pyproject.toml detection and FastAPI integration requirement): Follow standard conventions

## Recent Changes
- 002-implement-adaptive-replacement: Added Python 3.11+ (existing OmniCache codebase) + FastAPI (existing), Redis, scikit-learn/pytorch (ML), cryptography (encryption), prometheus-client, opentelemetry
- 001-base-caching-librabry: Added Python 3.11+ (based on pyproject.toml detection and FastAPI integration requirement) + FastAPI, AsyncIO, Click (CLI), Redis (optional backend), sqlite3 (fallback), typing-extensions

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
