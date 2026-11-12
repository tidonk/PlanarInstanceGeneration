#!/usr/bin/env python3
"""Collect complete SND flow results after checking their prepared inputs."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

METHODS = ("random", "high_degree", "kamada_kawai", "heuristic_outer_face", "exact_outer_face")


def collect(root, output, expected_inputs=100):
    results = [json.loads(p.read_text()) for p in sorted((root / 'results').glob('*/summary.json'))]
    expected = expected_inputs * len(METHODS)
    if len(results) != expected or len({r['stem'] for r in results}) != expected:
        raise ValueError(f'Expected {expected} unique completed runs')
    counts = Counter(r['strategy'] for r in results)
    if set(counts) != set(METHODS) or set(counts.values()) != {expected_inputs}:
        raise ValueError('Expected equal complete inputs for all five methods')
    for method in METHODS:
        indices = {r['input_index'] for r in results if r['strategy'] == method}
        if indices != set(range(1, expected_inputs + 1)):
            raise ValueError(f'Missing or duplicate input indices for {method}')
    for row in results:
        if not row.get('success') or row.get('primary_status') != 2 or row.get('secondary_status') != 2:
            raise ValueError(f"Both optimization steps must be optimal: {row['stem']}")
        source = root / 'inputs' / (row['stem'] + '.json.gz')
        if hashlib.sha256(source.read_bytes()).hexdigest() != row['input_sha256']:
            raise ValueError(f"Prepared input has changed: {row['stem']}")
        numerator, denominator = row['coupling']['mean'], row['inherited']['mean']
        ratio = row['ratios']['mean']
        if (not all(math.isfinite(x) for x in (numerator, denominator, ratio))
                or numerator < 0 or denominator <= 0
                or not math.isclose(ratio, numerator / denominator, rel_tol=1e-12)):
            raise ValueError(f"Invalid flow ratio: {row['stem']}")
    rows = []
    for result in results:
        row = {key: result.get(key) for key in (
            'stem', 'strategy', 'input_index', 'success', 'total_seconds',
            'primary_seconds', 'secondary_seconds', 'primary_objective',
            'baseline_objective', 'relative_cost_change', 'positive_coupling_edges')}
        for group in ('coupling', 'inherited', 'ratios'):
            row.update({group + '_' + key: value for key, value in result[group].items()})
        rows.append(row)
    report = dict(completed=len(results), successful=len(results), methods={})
    for method in METHODS:
        selected = [row for row in results if row['strategy'] == method]
        report['methods'][method] = dict(
            n=len(selected),
            median_mean_ratio=statistics.median(row['ratios']['mean'] for row in selected),
            zero_coupling_cases=sum(row['positive_coupling_edges'] == 0 for row in selected))
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'results.json').write_text(json.dumps({'results': results}, indent=2, allow_nan=False) + '\n')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Measurement directory containing inputs/ and results/.')
    parser.add_argument('--output', type=Path, required=True, help='New directory for collected results.')
    parser.add_argument('--expected-inputs', type=int, default=100)
    args = parser.parse_args()
    if args.expected_inputs <= 0:
        parser.error('--expected-inputs must be positive')
    if args.output.exists():
        parser.error('--output must be a new directory')
    print(json.dumps(collect(args.run, args.output, args.expected_inputs), indent=2))


if __name__ == '__main__':
    main()
