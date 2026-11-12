#!/usr/bin/env bash
# Run one post-generation SCOPF or survivable-SND evaluation.
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "usage: application_job.sh DATASET INSTANCE RUN SEED STRATEGY INDEX [SND_METHOD]" >&2
    exit 64
fi

DATASET="$1"
INSTANCE_FILE="$2"
RUN_NAME="$3"
SEED="$4"
STRATEGY="$5"
INSTANCE_INDEX="$6"
SND_METHOD="${7:-standard}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SETTINGS_FILE="${PIG_SETTINGS_FILE:-$REPO_ROOT/check/eval/settings.env}"
# shellcheck source=settings.env
source "$SETTINGS_FILE"

INSTANCE_BASENAME="${INSTANCE_FILE##*/}"
INSTANCE_STEM="${INSTANCE_BASENAME%.*}"
APPLICATION_SUMMARY="$REPO_ROOT/check/results/$RUN_NAME/logs/$INSTANCE_STEM/summary.json"
APPLICATION_LOG_DIRECTORY="${APPLICATION_SUMMARY%/summary.json}"
if [[ -f "$APPLICATION_SUMMARY" ]] \
    && grep -Eq '"state"[[:space:]]*:[[:space:]]*"(complete|wall_time_limit)"' "$APPLICATION_SUMMARY"; then
    if grep -Eq '"state"[[:space:]]*:[[:space:]]*"complete"' "$APPLICATION_SUMMARY" \
        && [[ "$DATASET" == "pglib" ]] && {
        ! grep -q '"scopf_feasible"' "$APPLICATION_SUMMARY" ||
            ! grep -q '"positive_flow_branch_count"' "$APPLICATION_SUMMARY"
    }; then
        printf 'archiving complete result with obsolete SCOPF schema: %s\n' \
            "$APPLICATION_SUMMARY"
    else
        printf 'evaluation already complete; reusing %s\n' "$APPLICATION_SUMMARY"
        exit 0
    fi
fi
if [[ -d "$APPLICATION_LOG_DIRECTORY" ]]; then
    FAILED_ROOT="$REPO_ROOT/check/results/$RUN_NAME/failed_attempts"
    mkdir -p "$FAILED_ROOT"
    FAILED_DIRECTORY="$FAILED_ROOT/${INSTANCE_STEM}.failed.$(date -u +%Y%m%dT%H%M%SZ).$$"
    printf 'archiving incomplete evaluation %s as %s\n' \
        "$APPLICATION_LOG_DIRECTORY" "$FAILED_DIRECTORY"
    mv "$APPLICATION_LOG_DIRECTORY" "$FAILED_DIRECTORY"
fi

if [[ -n "${PIG_PYTHON:-}" ]]; then
    PYTHON="$PIG_PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi

source "$REPO_ROOT/check/eval/local_julia_depot.sh"

printf 'evaluation dataset=%s instance=i%03d strategy=%s input=%s\n' \
    "$DATASET" "$INSTANCE_INDEX" "$STRATEGY" "$INSTANCE_FILE"
case "$DATASET" in
    pglib)
        pig_local_julia_depot "$PIG_SCOPF_REPOSITORY"
        SCOTS_ARGS=(--ignore-manifest-julia)
        if [[ "${PIG_SCOPF_WARMSTART:-0}" == 1 ]]; then
            SCOTS_ARGS+=(--scopf-warmstart --warmstart-time-limit "${PIG_SCOPF_WARMSTART_TIME_LIMIT:-10800}")
        fi
        if [[ -n "${PIG_SCOTS_BENDERS_SETTINGS:-}" ]]; then
            SCOTS_ARGS+=(--benders-settings "$PIG_SCOTS_BENDERS_SETTINGS")
        fi
        exec "$PYTHON" check/scripts/run_scopf_benders.py "$INSTANCE_FILE" \
            --run "$RUN_NAME" \
            --topology-repository "$PIG_SCOPF_REPOSITORY" \
            --julia "$PIG_JULIA" \
            --time-limit "$PIG_APPLICATION_TIME_LIMIT" \
            --threads "$PIG_APPLICATION_THREADS" \
            --seed "$SEED" \
            --topological-action-limit "$PIG_TOPOLOGICAL_ACTION_LIMIT" "${SCOTS_ARGS[@]}"
        ;;
    snd)
        pig_local_julia_depot "$PIG_SND_BENDERS_REPOSITORY"
        SND_MANIFEST_ARGS=()
        case "${PIG_SND_IGNORE_MANIFEST_JULIA:-0}" in
            0) ;;
            1) SND_MANIFEST_ARGS=(--ignore-manifest-julia) ;;
            *)
                echo "PIG_SND_IGNORE_MANIFEST_JULIA must be 0 or 1" >&2
                exit 64
                ;;
        esac
        exec "$PYTHON" check/scripts/run_snd_benders.py "$INSTANCE_FILE" \
            --run "$RUN_NAME" \
            --benders-repository "$PIG_SND_BENDERS_REPOSITORY" \
            --julia "$PIG_JULIA" \
            "${SND_MANIFEST_ARGS[@]}" \
            --time-limit "$PIG_APPLICATION_TIME_LIMIT" \
            --wall-clock-grace "$PIG_SND_WALL_CLOCK_GRACE" \
            --threads "$PIG_APPLICATION_THREADS" \
            --method "$SND_METHOD" \
            --seed "$SEED"
        ;;
    *)
        echo "unknown dataset: $DATASET" >&2
        exit 64
        ;;
esac
