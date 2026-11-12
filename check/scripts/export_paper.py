#!/usr/bin/env python3
"""Generate the evaluation figures and tables consumed by the manuscript."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import importlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from check.scripts import export_paper_skewness_tables as tables
from check.scripts.export_paper_structure_tikz import METHODS, LABELS, load_domain, rows


def call(module, arguments):
    previous = sys.argv
    try:
        sys.argv = [module, *map(str, arguments)]
        importlib.import_module('check.scripts.' + module).main()
    finally:
        sys.argv = previous


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def summary_table(out, inputs):
    lines = [r'\begin{tabular}{llrrrrrrr}', r'\toprule',
             r'Domain & method & $\bar{s}_C$ & $\underline a$ & $\bar a$ & $\underline\eta$ & $\bar\eta$ & '
             r'\makecell{No added\\skewness\\certified} & \makecell{Exact skewness\\certified}\\', r'\midrule']
    fields = ['skewness_upper_bound', 'skewness_inflation_lower_bound', 'skewness_inflation_upper_bound',
              'skewness_normalized_inflation_lower_bound', 'skewness_normalized_inflation_upper_bound']
    for domain, data in inputs.items():
        for method, label in zip(METHODS, LABELS):
            subset = [r for r in data['paired'] if r['strategy'] == method]
            if len(subset) != 100: raise ValueError('Expected 100 inputs per method')
            values = [tables._shifted_geometric_mean([float(r[field]) for r in subset]) for field in fields]
            zero = sum(float(r[fields[2]]) == 0 for r in subset)
            exact = sum(float(r['skewness_lower_bound']) == float(r['skewness_upper_bound']) for r in subset)
            title = {'pglib':'PGLib','snd':'SNDlib'}[domain] if method == METHODS[0] else ''
            lines.append(f'{title} & {label} & ' + ' & '.join(f'{x:.3f}' for x in values) + f' & {zero} & {exact} '+r'\\')
        lines.append(r'\midrule' if domain == 'pglib' else r'\bottomrule')
    (out / 'tab-eval-skewness.tex').write_text('\n'.join(lines+[r'\end{tabular}','']))


def funnel(out, roots, application):
    from check.scripts.export_paper_application_outcomes import outcome
    for domain, root in roots.items():
        attempts = {}
        for row in rows(root / 'tables/attempts.csv'):
            key = (row['output_index'], row['attempt_index'])
            attempts.setdefault(key, []).append(row)
        counts = Counter()
        for group in attempts.values():
            statuses = {r['status'] for r in group}
            if len(group) != 5 or {r['strategy'] for r in group} != set(METHODS):
                raise ValueError('Expected one outcome per method per generation attempt')
            if statuses == {'accepted'}:
                counts['accepted'] += 1
            else:
                reasons = {r.get('retry_trigger_status') or r['status'] for r in group}
                reasons.discard('paired_retry')
                if 'accepted' in reasons or len(reasons) != 1:
                    raise ValueError(f'Inconsistent retry reasons: {reasons}')
                counts[reasons.pop()] += 1
        if counts['accepted'] != 100: raise ValueError('Expected 100 accepted input sets')
        outcomes = Counter()
        app_domain = 'scots' if domain == 'pglib' else 'snd'
        for row in application[domain]:
            category = outcome(row, app_domain)
            if category == 'feasible' and row['termination_status'] == 'OPTIMAL': category = 'optimal'
            outcomes[category] += 1
        if sum(outcomes.values()) != 500: raise ValueError('Expected 500 application outcomes')
        labels = {'size_limit':'Size mismatch', 'gate_infeasible':'Feasibility check',
                  'selection_failed':'Selection failure', 'accepted':'Accepted'}
        total = sum(counts.values())
        text = [r'\begin{tikzpicture}[font=\scriptsize]',
                r'\path[use as bounding box] (-.30,7.4) rectangle (16.80,12.1);',
                rf'\node[anchor=east,align=right] at (1.55,10.15) {{{total} proposals\\(100\%)}};']
        pt = 28.45274
        top = 10.15 + total*.11/2/pt
        stages = [('accepted',counts['accepted'],10.45,'black!72')]
        rejected = [(k,v) for k,v in sorted(counts.items()) if k != 'accepted' and v]
        for i,(status,n) in enumerate(rejected):
            stages.append((status,n,9.45-.7*i,f'black!{18+16*i}'))
        for status,n,y,color in stages:
            width = n*.11
            center = top-width/2/pt
            top -= width/pt
            drawn = max(.6,width) if n <= 2 else width
            text += [rf'\funnelflow{{1.80}}{{{center:.3f}}}{{4.00}}{{{y:.2f}}}{{{drawn:g}pt}}{{{color}}}']
            if status == 'accepted':
                text += [rf'\node[anchor=south] at (4.80,11.10) {{{n} accepted ({100*n/total:.1f}\%)}};']
            else:
                label=labels.get(status,status.replace('_',' ').capitalize())
                text += [rf'\node[anchor=west] at (4.20,{y:.2f}) {{Rejected: {n} ({100*n/total:.1f}\%)}};',
                         rf'\node[anchor=west,font=\tiny,text=black!65] at (4.20,{y-.27:.2f}) {{{label}}};']
        text += [r'\funnelexpand{10.45}{8.70}',
                 r'\node[font=\tiny,text=white] at (7.35,10.45) {500 (100\%)};']
        application_label = r'SCOTS, \(\mathrm{TAL}=1\)' if domain == 'pglib' else 'SND'
        text += [rf'\node[anchor=west,font=\bfseries] at (8.58,11.60) {{{application_label} (\(n=500\))}};']
        top = 10.45+15/pt
        for key,label,y,color in [('optimal','Optimal',11,'black!82'),('feasible','Feasible',10.35,'black!58'),
                                  ('infeasible','Infeasible',9.65,'black!34'),('unknown','No solution',8.95,None)]:
            n=outcomes[key];width=n*.06;center=top-width/2/pt;top-=width/pt
            if n:
                drawn = max(.6,width) if n <= 2 else width
                if color: text += [rf'\funnelflow{{8.70}}{{{center:.3f}}}{{12.00}}{{{y:.2f}}}{{{drawn:g}pt}}{{{color}}}']
                else: text += [rf'\funnelnosolution{{8.70}}{{{center:.3f}}}{{12.00}}{{{y:.2f}}}{{{drawn:g}pt}}']
            text += [rf'\node[anchor=west] at (12.25,{y:.2f}) {{{label}: {n} ({n/5:.1f}\%)}};']
        text += [r'\end{tikzpicture}']
        (out/f'fig-app-evaluation-funnel-{domain}.tikz').write_text('\n'.join(text)+'\n')
    shutil.copyfile(ROOT/'reproduction/templates/fig-app-evaluation-funnel.tikz',out/'fig-app-evaluation-funnel.tikz')



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'reproduction/paper.json')
    parser.add_argument('--output', type=Path, required=True, help='empty directory for manuscript build inputs')
    parser.add_argument('--parts', nargs='+', choices=['generation','applications','sources','example','supplement','flows'],
                        default=['generation','applications','sources','example','supplement'])
    parser.add_argument("--application-domain", choices=("all", "scots", "snd"), default="all")
    parser.add_argument("--skip-flows", action="store_true", help="Leave the separate SCOPF flow evaluation for a later export")
    parser.add_argument("--omit-scots-sources", action="store_true", help="Export composites without an outdated source reference")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    paper = args.output.resolve()
    if paper.exists() and any(paper.iterdir()):
        parser.error('choose an empty output directory; existing paper files are never overwritten')
    out = paper/'resources/evaluation';out.mkdir(parents=True, exist_ok=True)
    roots = {domain:path(config['generation'][domain]) for domain in ('pglib','snd')}
    paired = {domain:path(config['paired'][domain]) for domain in roots}
    if 'generation' in args.parts:
        from check.scripts import export_paper_structure_tikz as structure
        from check.scripts.export_paper_skewness_cdf import export_skewness_pgf
        from check.scripts.export_paper_extraction import export as extraction
        from check.scripts.analyze_merge_node_degrees import summarize, write_plot_data
        from check.scripts.export_paper_merge_vertex_degrees import export as degrees
        inputs = {domain:load_domain(root) for domain, root in roots.items()}
        for domain in roots:
            # Reject incomplete or duplicate paired data before producing publication results.
            data = tables._read(paired[domain])
            inputs[domain]['paired'] = rows(paired[domain])
            title = 'PGLib' if domain == 'pglib' else 'SNDlib'
            table = tables._table(title, data)
            # The manuscript supplies the longtable, caption, and label.
            body = table[table.index(r'\toprule'):table.index(r'\end{longtable}')]
            (out/f'tab-app-full-skewness-{title.lower()}.tex').write_text(body)
        summary_table(out, inputs)
        export_skewness_pgf(out, paired)
        structure.export_bounded(out, inputs);structure.export_global(out, inputs)
        extraction(paper, {'PGLib':roots['pglib'],'SNDlib':roots['snd']})
        call('export_paper_runtime_plot',[paper,'--pglib',paired['pglib'],'--snd',paired['snd'],
            '--pglib-root',roots['pglib'],'--snd-root',roots['snd']])
        call('export_paper_skewness_distribution',['--paper-dir',paper])
        write_plot_data(out/'plot-data',summarize(list(roots.values()))['distribution']);degrees(paper)
    if 'applications' in args.parts:
        from check.scripts.export_paper_application_outcomes import render, read_rows
        from check.scripts.export_paper_exact_tables import select_source_rows
        settings = config['applications'];data = {}
        arguments = [paper,'--pglib-root',roots['pglib'],'--snd-root',roots['snd'], '--domain',args.application_domain]
        for domain, app in [('pglib','scots'),('snd','snd')]:
            if args.application_domain not in ('all', app): continue
            data[domain] = read_rows(path(settings[app]))
            sources = (None if app == 'scots' and args.omit_scots_sources else
                       select_source_rows(read_rows(path(settings[app+'_sources'])), roots[domain]))
            tex, _ = render(data[domain],sources,app)
            (out/f'fig-eval-application-outcomes-{app}.tikz').write_text(tex)
            arguments += ['--'+app+'-summary',path(settings[app]),'--'+app+'-source-summary',path(settings[app+'_sources'])]
        if settings.get('include_scots_warmstart',True): arguments += ['--scots-include-warmstart']
        if args.omit_scots_sources: arguments += ['--omit-scots-sources']
        call('export_paper_exact_tables',arguments)
        shutil.copyfile(ROOT/'reproduction/templates/fig-eval-application-outcomes.tikz',out/'fig-eval-application-outcomes.tikz')
        funnel(out,{domain:roots[domain] for domain in data},data)
        if not args.skip_flows:
            call('export_paper_scopf_flows',[paper,'--source-summary',path(settings['flow_baseline']),
                                           '--flow-summary',path(settings['flows'])])
    if 'sources' in args.parts:
        from check.scripts.export_paper_source_skewness import export
        s=config['sources']
        export(paper,path(s['pglib']),path(s['voltage']),path(s['snd']),
               json.loads((ROOT/'reproduction/quick-sources.json').read_text()))
    if 'example' in args.parts:
        call('export_paper_last_join',[paper,'--run',path(config['example']['run']),
                                      '--input',config['example']['input']])
        # Source legend is derived from this example, never copied from a different graph.
        example_rows = [r for r in rows(path(config['example']['run'])/'tables/accepted_instances.csv')
                        if int(r['output_index']) == config['example']['input']]
        metadata = path(config['example']['run'])/'plots/composite'/example_rows[0]['artifact_stem']/'metadata.json'
        names = json.loads(metadata.read_text())['source_names']
        from check.scripts.export_paper_last_join import _SUBTLE_SOURCE_COLORS
        palette = list(_SUBTLE_SOURCE_COLORS.values())
        legend = [r'\newcommand{\CompositeSourceLegend}{\normalsize\begin{tabular}{ll}']
        for i,name in enumerate(names,1):
            name=name.removeprefix('pglib_opf_').replace('_',r'\_')
            legend += [rf'\textcolor[HTML]{{{palette[i-1]}}}{{$\bullet$}} $G_{i}$ & \texttt{{{name}}}\\']
        legend += [r'\end{tabular}}',r'\newcommand{\CompositeGraphPanel}[1]{\includegraphics[width=\linewidth,height=6.95cm,keepaspectratio]{#1}}',
                   r'\newcommand{\CompositePanel}[1]{\CompositeGraphPanel{#1}\par\CompositeSourceLegend}']
        (out/'fig-app-example-composites-legend.tex').write_text('\n'.join(legend)+'\n')
    if 'supplement' in args.parts:
        s=config['supplement']
        call('export_paper_sampling',[paper,'--baseline',s['sampling_baseline'],'--experiment',s['sampling'],
                                     '--results',path(s['results']),
                                     '--baseline-application-summary',path(s['sampling_baseline_application'])])
        call('compare_scopf_normalization',['--baseline',path(s['normalization_baseline']),
            '--normalized',path(s['normalization']),'--audit',path(s['normalization_audit']),
            '--output',ROOT/'check/results/publication_checks/normalization',
            '--paper-table',out/'tab-eval-normalization.tex'])
        call('summarize_snd_piece_comparison',['--composite-run',path(s['snd_composites']),
            '--piece-run',path(s['snd_pieces']),'--output',ROOT/'check/results/publication_checks/snd_pieces.csv',
            '--paper-table',out/'tab-app-snd-piece-objectives.tex'])
    if 'flows' in args.parts:
        settings = config['flow_measurements']
        call('export_paper_scopf_flow_boxplot', ['--paper-dir', paper,
             '--summary', path(settings['scopf_summary'])])
        call('export_paper_snd_flow_boxplot', ['--paper-dir', paper,
             '--results', path(settings['snd_results'])])
    print('Publication inputs:',out)

if __name__ == '__main__': main()
