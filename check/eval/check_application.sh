#!/usr/bin/env bash
# Compatibility entry point for application DAGs generated before the
# summarizer was renamed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$REPO_ROOT/check/eval/application_summary.sh" "$@"
