#!/usr/bin/env bash
# Screen one source with the same implementation used by the serial command.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: preprocess_pglib.sh SOURCE RESULT_NAME" >&2
    exit 64
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source "${PIG_SETTINGS_FILE:-$REPO_ROOT/check/eval/ogdf_600_3600.env}"
exec "${PIG_PYTHON:-$REPO_ROOT/.venv/bin/python}" check/scripts/preprocess_pglib.py "$1" \
    --result-name "$2" --max-buses 2000 --time-limit 300 --threads 1
