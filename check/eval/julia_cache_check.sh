#!/usr/bin/env bash
# Existing DAG entry point: check local Julia with one solve, without precompiling.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PIG_SETTINGS_FILE="$REPO_ROOT/check/eval/ogdf_scopf_20261106.env"
export PIG_APPLICATION_TIME_LIMIT=300
check/eval/application_job.sh pglib \
    check/results/ogdf_20261104_pglib/instances/249_i087_2.m \
    ogdf_local_julia_20261106_smoke 87 high_degree 87
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('check/results/ogdf_local_julia_20261106_smoke/logs/249_i087_2/summary.json')
s = json.loads(p.read_text())
assert s['state'] == 'complete', s.get('error', s['state'])
assert s['configuration']['topological_action_limit'] == 0
assert s['result']['termination_status'] in ('OPTIMAL', 'INFEASIBLE', 'TIME_LIMIT')
assert s['result']['julia_version'] == '1.12.6'
assert s['command'][0].startswith('/local/tmp/')
print('LOCAL_JULIA_SOLVE_VERIFIED', s['result']['termination_status'], s['wall_seconds'])
PY
