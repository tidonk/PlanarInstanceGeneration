#!/usr/bin/env bash
#
# Consolidate one dataset's completed generation jobs and build the review
# report. The DAG waits for every method-level exact-skewness checkpoint
# produced for the selected data set.
#
# Arguments: DATASET [EXPERIMENT]
#
# Environment knobs:
#   PIG_PYTHON            interpreter with the package + Gurobi installed
#   PIG_POSTPROCESS_ARGS  extra flags forwarded to postprocess.py
#                         (e.g. --threads, --skewness-time-limit, --no-compile)
#
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: report.sh DATASET [EXPERIMENT] [DATA_ONLY=0|1]" >&2
    exit 64
fi

DATASET="$1"
EXPERIMENT="${2:-main}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SETTINGS_FILE="${PIG_SETTINGS_FILE:-$REPO_ROOT/check/eval/settings.env}"
if [[ -r "$SETTINGS_FILE" ]]; then
    # shellcheck source=settings.env
    source "$SETTINGS_FILE"
fi

if [[ -n "${PIG_PYTHON:-}" ]]; then
    PYTHON="$PIG_PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi

REPORT_DATASET_ARGS=()
case "${3:-0}" in
    1) REPORT_DATASET_ARGS+=(--no-compile) ;;
    0) ;;
    *) echo "data-only flag must be 0 or 1" >&2; exit 64 ;;
esac
if [[ "$DATASET" == "pglib" ]]; then
    CHECKPOINT_DIRECTORY="$REPO_ROOT/check/results/${EXPERIMENT}_pglib_exact/evaluation/cohort_checkpoints"
    CHECKPOINT_ARGUMENT="--pglib-cohort-checkpoints"
elif [[ "$DATASET" == "snd" ]]; then
    CHECKPOINT_DIRECTORY="$REPO_ROOT/check/results/${EXPERIMENT}_snd_exact/evaluation/cohort_checkpoints"
    CHECKPOINT_ARGUMENT="--snd-cohort-checkpoints"
else
    echo "unsupported dataset: $DATASET" >&2
    exit 64
fi
if [[ ! -d "$CHECKPOINT_DIRECTORY" ]]; then
    echo "distributed $DATASET checkpoints are missing: $CHECKPOINT_DIRECTORY" >&2
    exit 66
fi
REPORT_DATASET_ARGS+=("$CHECKPOINT_ARGUMENT" "$CHECKPOINT_DIRECTORY")

# shellcheck disable=SC2086
exec "$PYTHON" check/eval/postprocess.py "$DATASET" \
    --experiment "$EXPERIMENT" \
    --python "$PYTHON" \
    --report \
    --overwrite \
    "${REPORT_DATASET_ARGS[@]}" \
    ${PIG_POSTPROCESS_ARGS:-}
