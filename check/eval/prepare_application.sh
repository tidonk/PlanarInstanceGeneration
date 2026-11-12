#!/usr/bin/env bash
# Build one application-evaluation row per accepted final composite.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: prepare_application.sh DATASET EXPERIMENT [INSTANCE_INDEX|SND_METHOD]" >&2
    exit 64
fi

DATASET="$1"
EXPERIMENT="$2"
INSTANCE_ARGS=()
SND_METHOD_ARGS=()
if [[ $# -eq 3 ]]; then
    case "$3" in
        standard|adaptive_ml)
            [[ "$DATASET" == "snd" ]] || {
                echo "SND_METHOD applies only to the snd dataset" >&2
                exit 64
            }
            SND_METHOD_ARGS+=(--snd-method "$3")
            ;;
        *)
            INSTANCE_ARGS+=(--instance-index "$3")
            ;;
    esac
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SETTINGS_FILE="${PIG_SETTINGS_FILE:-$REPO_ROOT/check/eval/settings.env}"
# shellcheck source=settings.env
source "$SETTINGS_FILE"
read -r -a METHODS <<< "$PIG_METHODS"

if [[ -n "${PIG_PYTHON:-}" ]]; then
    PYTHON="$PIG_PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi

exec "$PYTHON" check/eval/make_application_jobs.py "$DATASET" \
    --experiment "$EXPERIMENT" \
    --instances "$PIG_INSTANCES" \
    --base-seed "$PIG_BASE_SEED" \
    --methods "${METHODS[@]}" \
    "${INSTANCE_ARGS[@]}" \
    "${SND_METHOD_ARGS[@]}"
