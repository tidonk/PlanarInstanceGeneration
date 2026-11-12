#!/usr/bin/env python3
"""Check the production OPF gate against nodal balance and branch equations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from planar_instance_generation.gates import InnoptemDcOpfGate
from planar_instance_generation.io import read_pglib
from tests.test_power_generation import _case


def residuals(case, solution):
    balance = -case.bus[:, 2] - case.bus[:, 4]
    positions = {int(row[0]): i for i, row in enumerate(case.bus)}
    for row, generation in zip(case.gen, solution.generator_dispatch, strict=True):
        balance[positions[int(row[0])]] += generation
    physics = []
    thermal = []
    for row, flow in zip(case.branch, solution.branch_from_flows, strict=True):
        u, v = positions[int(row[0])], positions[int(row[1])]
        balance[u] -= flow
        balance[v] += flow
        # PowerModels DCP uses the imaginary part of the series admittance.
        expected = (case.base_mva * row[3] / (row[2] ** 2 + row[3] ** 2)
                    * np.radians(solution.bus_angles[u] - solution.bus_angles[v])) if row[10] else 0.0
        physics.append(abs(flow - expected))
        if row[5] > 0:
            thermal.append(max(0.0, abs(flow) - row[5]))
    generation_violation = max((max(0.0, row[9] - output, output - row[8])
                               if row[7] else abs(output)
                               for row, output in zip(case.gen, solution.generator_dispatch, strict=True)), default=0.0)
    return dict(balance_mw=float(np.max(np.abs(balance))),
                branch_equation_mw=float(max(physics, default=0.0)),
                thermal_mw=float(max(thermal, default=0.0)),
                generation_mw=float(generation_violation))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gate = InnoptemDcOpfGate(time_limit=60, log_directory=args.output / 'logs')
    cases = [_case('three_bus')]
    for name in ('pglib_opf_case73_ieee_rts', 'pglib_opf_case89_pegase', 'pglib_opf_case118_ieee'):
        cases.append(read_pglib(ROOT / 'check/data/pglib' / (name + '.m')))
    reordered = read_pglib(ROOT / 'check/data/pglib/pglib_opf_case14_ieee.m')
    reordered.name = 'reordered_case14'
    reordered.bus = reordered.bus[::-1].copy()
    reordered.branch = reordered.branch[::-1].copy()
    reordered.gen = reordered.gen[::-1].copy()
    reordered.gencost = reordered.gencost[::-1].copy()
    reordered.branch[0, 10] = 0
    cases.append(reordered)
    records = []
    for case in cases:
        result = gate.solve(case)
        assert result.feasible, (case.name, result.status)
        errors = residuals(case, result.solution)
        assert max(errors.values()) <= 1e-4, (case.name, errors)
        records.append(dict(case=case.name, status=result.status, objective=result.objective,
                            runtime_seconds=result.runtime_seconds, residuals=errors))
        print(case.name, result.status, errors, flush=True)
    impossible = _case('infeasible_three_bus')
    impossible.gen[0, 8] = 1.0
    result = gate.solve(impossible)
    assert not result.feasible and result.solution is None, result
    records.append(dict(case=impossible.name, status=result.status, feasible=result.feasible))
    print(impossible.name, result.status, flush=True)
    (args.output / 'verification.json').write_text(json.dumps(dict(
        backend=gate.identity(), passed=len(records), records=records), indent=2) + '\n')

if __name__ == '__main__':
    main()
