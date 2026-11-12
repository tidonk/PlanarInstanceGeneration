#!/usr/bin/env bash
#
# One HTCondor generation job: generate a single accepted composite for one
# instance and every configured merging method into an isolated result directory.
#
# Arguments (supplied by the submit file): DATASET INSTANCE SEED EXPERIMENT
#
# Environment knobs:
#   PIG_PYTHON          interpreter to run the generate script (default: python3)
#   PIG_GENERATE_ARGS   extra flags forwarded verbatim to the generate script
#                       (e.g. solver time limits, coupling-rule, sizes)
#
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "usage: job.sh DATASET INSTANCE SEED EXPERIMENT [OUTER_FACE_SAMPLING] [PREFER_NON_LEAF]" >&2
    exit 64
fi

DATASET="$1"
INSTANCE="$2"
SEED="$3"
EXPERIMENT="$4"
OUTER_FACE_SAMPLING="${5:-}"
PREFER_NON_LEAF="${6:-}"

if [[ -n "$OUTER_FACE_SAMPLING" ]]; then
    if [[ "$OUTER_FACE_SAMPLING" != "equidistant" && "$OUTER_FACE_SAMPLING" != "consecutive" && "$OUTER_FACE_SAMPLING" != "adaptive_skip_one" ]]; then
        echo "unknown outer-face sampling policy: $OUTER_FACE_SAMPLING" >&2
        exit 64
    fi
    export PIG_OUTER_FACE_SAMPLING="$OUTER_FACE_SAMPLING"
fi
if [[ -n "$PREFER_NON_LEAF" ]]; then
    if [[ "$PREFER_NON_LEAF" != "0" && "$PREFER_NON_LEAF" != "1" ]]; then
        echo "PREFER_NON_LEAF must be 0 or 1" >&2
        exit 64
    fi
    export PIG_OUTER_FACE_PREFER_NON_LEAF="$PREFER_NON_LEAF"
fi


REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SETTINGS_FILE="${PIG_SETTINGS_FILE:-$REPO_ROOT/check/eval/settings.env}"
if [[ -r "$SETTINGS_FILE" ]]; then
    # shellcheck source=settings.env
    source "$SETTINGS_FILE"
fi

# Keep the worker wrapper usable with a missing or deliberately minimal override.
: "${PIG_METHODS:=random high_degree kamada_kawai heuristic_outer_face exact_outer_face}"

# Resolve the interpreter: an explicit PIG_PYTHON wins; otherwise prefer the
# repository's own virtualenv (built on shared storage so execute nodes can
# import networkx and gurobipy on top of the node's system Python 3.11);
# fall back to python3 only as a last resort.
if [[ -n "${PIG_PYTHON:-}" ]]; then
    PYTHON="$PIG_PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi
TAG="$(printf 'i%03d' "$INSTANCE")"
RESULT_NAME="${EXPERIMENT}_${DATASET}_job_${TAG}"
ACTIVE_RESULT_DIRECTORY="$REPO_ROOT/check/results/$RESULT_NAME"
ARCHIVED_RESULT_DIRECTORY="$REPO_ROOT/check/results/_cohorts/${EXPERIMENT}_${DATASET}/$RESULT_NAME"
if [[ -d "$ACTIVE_RESULT_DIRECTORY" && -d "$ARCHIVED_RESULT_DIRECTORY" ]]; then
    echo "cohort exists in both active and archived result locations: $RESULT_NAME" >&2
    exit 75
fi
if [[ -d "$ARCHIVED_RESULT_DIRECTORY" ]]; then
    RESULT_DIRECTORY="$ARCHIVED_RESULT_DIRECTORY"
else
    RESULT_DIRECTORY="$ACTIVE_RESULT_DIRECTORY"
fi

# DAG restarts resubmit every node. Reuse a fully materialized cohort, but do
# not mistake an interrupted output directory for success: all configured
# method rows, final instances, and worker-compiled PDFs must be present.
if [[ -f "$RESULT_DIRECTORY/manifest.json" ]]; then
    read -r -a EXPECTED_METHODS <<< "$PIG_METHODS"
    ACCEPTED="$RESULT_DIRECTORY/tables/accepted_instances.csv"
    ACCEPTED_COUNT=0
    INSTANCE_COUNT=0
    PDF_COUNT=0
    if [[ -f "$ACCEPTED" ]]; then
        ACCEPTED_COUNT=$(awk 'NR > 1 { count += 1 } END { print count + 0 }' "$ACCEPTED")
    fi
    INSTANCE_COUNT=$(find "$RESULT_DIRECTORY/instances" -maxdepth 1 -type f | wc -l)
    PDF_COUNT=$(find "$RESULT_DIRECTORY/plots/composite" -mindepth 2 -maxdepth 2 -type f -name full.pdf | wc -l)
    if (( ACCEPTED_COUNT == ${#EXPECTED_METHODS[@]} && INSTANCE_COUNT == ${#EXPECTED_METHODS[@]} && PDF_COUNT == ${#EXPECTED_METHODS[@]} )); then
        echo "Reusing complete cohort: $RESULT_NAME"
        exit 0
    fi
    echo "existing manifest is incomplete for $RESULT_NAME" >&2
    exit 75
elif [[ -d "$RESULT_DIRECTORY" ]]; then
    echo "interrupted result directory must be archived before retry: $RESULT_DIRECTORY" >&2
    exit 75
fi

case "$DATASET" in
    pglib)
        SCRIPT="check/scripts/generate_pglib.py"
        DATASET_GENERATE_ARGS="${PIG_PGLIB_GENERATE_ARGS:-}"
        ;;
    snd)
        SCRIPT="check/scripts/generate_sndlib.py"
        DATASET_GENERATE_ARGS="${PIG_SND_GENERATE_ARGS:-}"
        ;;
    *)     echo "unknown dataset: $DATASET" >&2; exit 64 ;;
esac
EXTRA_GENERATE_ARGS="${PIG_GENERATE_ARGS:-$DATASET_GENERATE_ARGS}"

OUTER_FACE_PREFERENCE_ARGS=()
case "${PIG_OUTER_FACE_PREFER_NON_LEAF:-0}" in
    0) ;;
    1) OUTER_FACE_PREFERENCE_ARGS=(--outer-face-prefer-non-leaf) ;;
    *)
        echo "PIG_OUTER_FACE_PREFER_NON_LEAF must be 0 or 1" >&2
        exit 64
        ;;
esac
if [[ "$DATASET" == pglib && -n "${PIG_OPF_REPOSITORY:-}" ]]; then
    source "$REPO_ROOT/check/eval/local_julia_depot.sh"
    pig_local_julia_depot "$PIG_OPF_REPOSITORY"
    export PIG_OPF_JULIA_DEPOT="$JULIA_DEPOT_PATH"
fi

# Word-split PIG_GENERATE_ARGS intentionally so callers can pass several flags.
# shellcheck disable=SC2086
exec "$PYTHON" "$SCRIPT" \
    --result-name "$RESULT_NAME" \
    --suffix "$TAG" \
    --seed "$SEED" \
    --count 1 \
    --strategies ${PIG_METHODS} \
    --no-original-plots \
    "${OUTER_FACE_PREFERENCE_ARGS[@]}" \
    ${EXTRA_GENERATE_ARGS:-}
