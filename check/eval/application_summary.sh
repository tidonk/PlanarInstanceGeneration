#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: application_summary.sh DATASET EXPERIMENT [SND_METHOD]" >&2
    exit 64
fi

SND_METHOD_ARGS=()
if [[ $# -eq 3 ]]; then
    [[ "$1" == "snd" ]] || {
        echo "SND_METHOD applies only to the snd dataset" >&2
        exit 64
    }
    SND_METHOD_ARGS+=(--snd-method "$3")
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi
exec "$PYTHON" check/eval/summarize_applications.py \
    "$1" --experiment "$2" "${SND_METHOD_ARGS[@]}"
