#!/usr/bin/env python3
"""Evaluate only the retained SND source graphs for the source-skewness table."""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from planar_instance_generation.evaluation.snd_source_pool_io import load_snd_source_pool
from planar_instance_generation.evaluation.planarization_batch import PlanarizationTask, solve_planarization_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pool', type=Path, default=ROOT/'check/results/sndlib_source_pool/manifest.json')
    parser.add_argument('--path-root', type=Path, default=ROOT, help='base directory for paths saved in the source pool')
    parser.add_argument('--output', type=Path, default=ROOT/'check/results/ogdf_snd_source_skewness')
    parser.add_argument('--time-limit', type=float, default=300)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--merge-directory', type=Path, help='collect per-source CSVs without solving')
    parser.add_argument('--source', help='solve only this retained source')
    args = parser.parse_args()
    if args.merge_directory:
        sources = load_snd_source_pool(args.source_pool, path_root=args.path_root)
        records = []
        for source in sources:
            with (args.merge_directory/source.source_name/'skewness.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1 or rows[0]['instance'] != source.source_name:
                raise ValueError('Unexpected source result')
            records.extend(rows)
        target = args.merge_directory/'skewness.csv'
        temporary = target.with_suffix('.csv.tmp')
        with temporary.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        temporary.replace(target)
        return
    sources=load_snd_source_pool(args.source_pool,path_root=args.path_root)
    if args.source:
        sources = tuple(s for s in sources if s.source_name == args.source)
        if len(sources) != 1:
            raise ValueError(f'Unknown or duplicate retained source: {args.source}')
    args.output.mkdir(parents=True,exist_ok=True)
    tasks=[PlanarizationTask(s.network.topology(),0,args.output/'logs'/(s.source_name+'.log'),s.source_name) for s in sources]
    results=solve_planarization_batch(tasks,time_limit_seconds=args.time_limit,solver_threads=args.threads,
                                      workers=args.workers,checkpoint_directory=args.output/'checkpoints')
    records=[dict(group='Original sources',instance=s.source_name,node_count=r.original_node_count,
                  edge_count=r.original_edge_count,skewness_lower_bound=r.omitted_lower_bound,
                  skewness_upper_bound=r.omitted_upper_bound,skewness_exact=r.exact,
                  runtime_seconds=r.total_runtime_seconds)
             for s,r in zip(sources,results,strict=True)]
    target=args.output/'skewness.csv';temporary=target.with_suffix('.csv.tmp')
    with temporary.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    temporary.replace(target)
    print(target)

if __name__=='__main__':main()
