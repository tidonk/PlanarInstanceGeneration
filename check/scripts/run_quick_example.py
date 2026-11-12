#!/usr/bin/env python3
"""Regenerate five illustrative PGLib comparisons with short solver limits."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from check.eval.make_full_dag import STRATEGIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='quick_example')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 3, 4, 13, 15])
    parser.add_argument('--generation-limit', type=float, default=10)
    parser.add_argument('--evaluation-limit', type=float, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if not args.name.replace('_', '').isalnum() or not 5 <= len(args.seeds) <= 10:
        parser.error('use a simple run name and between five and ten seeds')
    if min(args.generation_limit, args.evaluation_limit, args.workers, args.threads) <= 0:
        parser.error('limits and worker/thread counts must be positive')
    root = ROOT / 'check/results'
    output = root / args.name
    collisions = [output, root / (args.name + '_sources'), root / (args.name + '_pglib_exact'),
                  *[root / f'{args.name}_pglib_job_i{i:03d}' for i in range(1,len(args.seeds)+1)]]
    if any(p.exists() for p in collisions): parser.error('choose a fresh run name')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (output / 'settings.json').write_text(json.dumps(vars(args) | {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'illustrative PGLib subset; short limits; not the full evaluation',
    }, indent=2) + '\n')
    # Use the same accepted source set as the paper. The full screening remains
    # available separately; these source names carry no benchmark contents.
    names = json.loads((ROOT / 'reproduction/quick-sources.json').read_text())
    for name in names:
        if not (ROOT / 'check/data/pglib' / (name + '.m')).is_file():
            raise SystemExit('Missing benchmark data. Run check/scripts/fetch_inputs.py --dataset pglib')
    from check.scripts.fetch_inputs import verify
    with (ROOT / 'reproduction/input-files.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['dataset'] == 'pglib' and Path(row['filename']).stem in names:
                verify(ROOT / 'check/data/pglib' / row['filename'], row['sha256'])
    os.environ.setdefault('PIG_OPF_JULIA_DEPOT', str(ROOT / 'build/julia-depot'))

    def run_module(module, argv):
        previous = sys.argv
        try:
            sys.argv = [module, *map(str, argv)]
            importlib.import_module('check.scripts.' + module).main()
        finally:
            sys.argv = previous

    # Calling the production entry points in one process reuses Julia after its
    # first initialization. No saved dispatches or generated graphs are bundled.
    pool_name = args.name + '_sources'
    run_module('preprocess_pglib', [*(name + '.m' for name in names),
        '--result-name', pool_name, '--time-limit', 60, '--threads', 1])
    for index, seed in enumerate(args.seeds, 1):
        print(f'Generating set {index}/{len(args.seeds)}, seed {seed}', flush=True)
        run_module('generate_pglib', ['--source-pool', root / pool_name / 'manifest.json',
            '--result-name', f'{args.name}_pglib_job_i{index:03d}',
            '--suffix', f'i{index:03d}', '--count', 1, '--seed', seed,
            '--min-source-nodes', 73, '--exact-time-limit', args.generation_limit,
            '--threads', args.threads, '--opf-threads', 1, '--opf-time-limit', 60,
            '--no-compile', '--no-original-plots'])

    def evaluate(task):
        index, strategy = task
        log = output / f'evaluation-{index}-{strategy}.log'
        with log.open('w') as stream:
            subprocess.run([sys.executable, str(ROOT / 'check/scripts/evaluate_pglib_cohort_skewness.py'),
                args.name, str(index), strategy, '--time-limit', str(args.evaluation_limit),
                '--threads', str(args.threads)], cwd=ROOT, stdout=stream,
                stderr=subprocess.STDOUT, check=True)
        print(f'Evaluated set {index}: {strategy}', flush=True)
    tasks = [(i, method) for i in range(1, len(args.seeds) + 1) for method in STRATEGIES]
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        list(workers.map(evaluate, tasks))
    run_module('summarize_ogdf_progress', [args.name, '--instances', len(args.seeds)])
    subprocess.run([sys.executable, str(ROOT / 'check/scripts/export_quick_example.py'),
        args.name], cwd=ROOT, check=True)
    seconds = time.monotonic() - started
    (output / 'timing.json').write_text(json.dumps({'wall_seconds': seconds}, indent=2) + '\n')
    print(f'Finished in {seconds/60:.1f} minutes. Plots: {output / "figures"}', flush=True)

if __name__ == '__main__':
    main()
