#!/usr/bin/env bash
#
# Materialize the .jobs tables and Condor log directories from settings.env.
# Optional arguments restrict preparation to one or more configured datasets.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SETTINGS_FILE="${PIG_SETTINGS_FILE:-$REPO_ROOT/check/eval/settings.env}"
if [[ ! -r "$SETTINGS_FILE" ]]; then
    echo "settings file is not readable: $SETTINGS_FILE" >&2
    exit 66
fi
# shellcheck source=settings.env
source "$SETTINGS_FILE"

if [[ ! "$PIG_EXPERIMENT" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "PIG_EXPERIMENT must be a safe filename component" >&2
    exit 64
fi
if [[ ! "$PIG_INSTANCES" =~ ^[1-9][0-9]*$ ]]; then
    echo "PIG_INSTANCES must be a positive integer" >&2
    exit 64
fi
if [[ ! "$PIG_BASE_SEED" =~ ^[0-9]+$ ]]; then
    echo "PIG_BASE_SEED must be a nonnegative integer" >&2
    exit 64
fi

read -r -a CONFIGURED_DATASETS <<< "$PIG_DATASETS"
read -r -a METHODS <<< "$PIG_METHODS"
if (( ${#CONFIGURED_DATASETS[@]} == 0 || ${#METHODS[@]} == 0 )); then
    echo "PIG_DATASETS and PIG_METHODS must not be empty" >&2
    exit 64
fi
if (( $# )); then
    DATASETS=("$@")
else
    DATASETS=("${CONFIGURED_DATASETS[@]}")
fi

if [[ -n "${PIG_PYTHON:-}" ]]; then
    PYTHON="$PIG_PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi

exec "$PYTHON" check/eval/make_jobs.py \
    --experiment "$PIG_EXPERIMENT" \
    --instances "$PIG_INSTANCES" \
    --base-seed "$PIG_BASE_SEED" \
    --datasets "${DATASETS[@]}" \
    --methods "${METHODS[@]}"
