#!/usr/bin/env python3
"""Render the paper's runtime and added-skewness comparisons for a short run."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from check.scripts.export_paper_runtime_plot import METHODS, shifted_geometric_mean as sgm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    progress = ROOT / 'check/results' / (args.experiment + '_progress')
    summary = json.loads((progress / 'summary.json').read_text())
    if summary['errors']:
        raise ValueError(summary['errors'])
    ids = summary['datasets']['pglib']['complete_inputs']
    expected = summary['expected_inputs_per_dataset']
    if len(ids) != expected:
        raise ValueError('Refusing to plot an incomplete short run')
    with (progress / 'results.csv').open() as f:
        results = [r for r in csv.DictReader(f) if r['dataset'] == 'pglib']
    out = ROOT / 'check/results' / args.experiment / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
    labels = ['R', 'HD', 'KK', 'OF-H', 'OF-B']
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    groups = [[r for r in results if r['strategy'] == method] for method, _ in METHODS]
    generated = []
    for i in ids:
        p = ROOT / f'check/results/{args.experiment}_pglib_job_i{i:03d}/tables/accepted_instances.csv'
        with p.open() as f: generated.extend(csv.DictReader(f))
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
        fig.savefig(out / (name + '.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
    # Identical shift and component definitions to the paper runtime exporter.
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, ((method, _), color) in enumerate(zip(METHODS, colors)):
        rows = [r for r in generated if r['strategy'] == method]
        selection = sgm([float(r['merge_node_selection_seconds']) for r in rows])
        gate = sgm([float(r['gate_seconds']) for r in rows])
        overhead = sgm([max(0, float(r['generation_and_gate_seconds']) -
            float(r['merge_node_selection_seconds']) - float(r['gate_seconds'])) for r in rows])
        ax.barh(i, selection, color=color, label='Merge vertex selection' if i == 0 else None)
        ax.barh(i, gate, left=selection, color='white', edgecolor=color, hatch='////',
                label='Feasibility check' if i == 0 else None)
        ax.barh(i, overhead, left=selection+gate, color=color, alpha=.25,
                label='Extraction and assembly' if i == 0 else None)
    ax.set_yticks(range(5), labels);ax.invert_yaxis();ax.set_xlabel('Time [s], shifted geometric mean')
    ax.legend(fontsize=8);save(fig, 'figure-3-generation')
    # Bounds are visible. A lower/upper CDF pair brackets the true distribution.
    fig, ax = plt.subplots(figsize=(6, 3.7))
    for rows, label, color in zip(groups, labels, colors):
        lo = sorted(float(r['normalized_inflation_lower_bound']) for r in rows)
        hi = sorted(float(r['normalized_inflation_upper_bound']) for r in rows)
        for values, style in ((lo, '--'), (hi, '-')):
            ax.step([0, *values, 1], [0, *[100*(i+1)/len(ids) for i in range(len(ids))], 100],
                where='post', color=color, linestyle=style, label=label if style=='-' else None)
    ax.set(xlabel='Normalized added skewness', ylabel='Inputs [%]', xlim=(0,1), ylim=(0,102))
    ax.legend(ncol=5, fontsize=8, loc='lower center', bbox_to_anchor=(.5,1.02),
              title='Solid: upper bounds; dashed: lower bounds', title_fontsize=8, frameon=False)
    save(fig, 'figure-4a')
    fig, ax = plt.subplots(figsize=(6, 3.7))
    for i, (rows, label, color) in enumerate(zip(groups, labels, colors)):
        lo = [int(float(r['inflation_lower_bound'])) for r in rows]
        hi = [int(float(r['inflation_upper_bound'])) for r in rows]
        guaranteed = [sum(x==0 for x in hi)] + [sum(x>=k for x in lo) for k in range(1,6)]
        possible = [sum(x==0 for x in lo)] + [sum(x>=k for x in hi) for k in range(1,6)]
        xs=np.arange(6)+(i-2)*.12;low=np.array(guaranteed)*100/len(ids);high=np.array(possible)*100/len(ids)
        ax.vlines(xs,low,high,color=color)
        ax.scatter(xs,high,facecolors='white',edgecolors=color,s=20)
        ax.scatter(xs,low,color=color,s=14,label=label)
    ax.set_xticks(range(6), ['0', '≥1', '≥2', '≥3', '≥4', '≥5'])
    ax.set(xlabel='Added skewness', ylabel='Inputs [%]', ylim=(-3,103))
    ax.legend(ncol=5, fontsize=8, loc='lower center', bbox_to_anchor=(.5,1.02),
              title='Filled: guaranteed fraction; open: possible fraction', title_fontsize=8, frameon=False)
    save(fig, 'figure-5a')
    metrics=summary['datasets']['pglib']
    table=['Method,Completed,Exact,Zero added,Strictly better than R/HD/KK']
    for (method,_),label in zip(METHODS,labels):
        m=metrics['methods'][method]
        table.append(f'{label},{m["matched"]},{m["exact"]},{m["zero_added"]},{len(m["strictly_better_than_all_baselines"])}')
    (out/'summary.csv').write_text('\n'.join(table)+'\n')
    (out/'README.txt').write_text(
        f'Illustrative subset: {len(ids)} inputs, all five methods. Short limits, not full-paper results.\n'
        'Figure 3 excludes separately recorded Julia initialization.\n'
        'Figure 4: solid curves use upper bounds; dashed curves use lower bounds.\n'
        'Figure 5: filled markers are guaranteed counts; open markers are possible counts.\n'
        'The short example does not run SCOTS, SND, or supplementary sensitivity studies.\n')
    print('\n'.join(table), flush=True)

if __name__ == '__main__':
    main()
