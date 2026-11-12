#!/usr/bin/env python3
"""Prepare SND flow measurements from a generation run or a recorded archive."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

METHODS = ("random", "high_degree", "kamada_kawai", "heuristic_outer_face", "exact_outer_face")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(read_bytes, names, base, pool_prefix, output, source_info, expected_inputs=100):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'inputs').mkdir()
    pool = {}
    pool_hashes = {}
    for name in names:
        if name.startswith(pool_prefix) and name.endswith('.json'):
            raw = read_bytes(name)
            d = json.loads(raw)
            pool[d['source_name']] = d['exact_mcmcf_solution']['installed_capacities']
            pool_hashes[d['source_name']] = digest(raw)
    rows = list(csv.DictReader(io.StringIO(read_bytes(base + 'tables/accepted_instances.csv').decode())))
    records = []
    for row in rows:
        stem = row['artifact_stem']
        xml = read_bytes(base + 'instances/' + stem + '.xml')
        metadata = read_bytes(base + 'plots/composite/' + stem + '/metadata.json')
        meta = json.loads(metadata)
        tree = ET.fromstring(xml)
        for el in tree.iter():
            el.tag = el.tag.rsplit('}', 1)[-1]
        nodes = [n.attrib['id'] for n in tree.findall('.//nodes/node')]
        capacities = {}
        for source, mapping in zip(meta['source_names'], meta['source_mappings'], strict=True):
            for source_link, composite_link in mapping['link_ids'].items():
                assert composite_link not in capacities
                capacities[composite_link] = pool[source][source_link]
        coupling = {x['id']: x for x in meta['coupling_links']}
        edges = []
        for el in tree.findall('.//links/link'):
            name = el.attrib['id']
            cost_text = el.findtext('routingCost')
            cost = 1.0 if cost_text is None else float(cost_text)
            if name in coupling:
                data = coupling[name]
                assert data['installed_reference_capacity'] == 0
                assert len(data['additional_modules']) == 1
                module = data['additional_modules'][0]
                assert module['cost'] == 0 and data['routing_cost'] == 0
                capacities[name] = float(module['capacity'])
            edges.append(dict(id=name, source=el.findtext('source'), target=el.findtext('target'),
                              capacity=capacities[name], cost=cost, coupling=name in coupling))
        assert set(capacities) == {e['id'] for e in edges}
        demands = [dict(id=el.attrib['id'], source=el.findtext('source'), target=el.findtext('target'),
                        value=float(el.findtext('demandValue')))
                   for el in tree.findall('.//demands/demand')]
        assert all(el.find('maxPathLength') is None and el.find('admissiblePaths') is None
                   for el in tree.findall('.//demands/demand'))
        edge_by_id = {e['id']: e for e in edges}
        demand_by_id = {d['id']: d for d in demands}
        loads = dict.fromkeys(edge_by_id, 0.0)
        seen = set()
        reference_cost = 0.0
        for path in meta['serialized_demand_provenance']:
            assert not path['carries_circulation']
            d = demand_by_id[path['serialized_demand_id']]
            assert d['id'] not in seen
            seen.add(d['id'])
            assert d['source'] == path['node_ids'][0] and d['target'] == path['node_ids'][-1]
            assert abs(d['value'] - path['value']) <= 1e-7 * max(1, d['value'])
            for a, b, link in zip(path['node_ids'], path['node_ids'][1:], path['link_ids']):
                e = edge_by_id[link]
                assert {a, b} == {e['source'], e['target']} and not e['coupling']
                loads[link] += path['value']
                reference_cost += e['cost'] * path['value']
        assert seen == set(demand_by_id)
        for e in edges:
            assert loads[e['id']] <= e['capacity'] + 1e-7 * max(1, e['capacity'])
            e['reference_load'] = loads[e['id']]
        baseline_objective = float(row['objective'])
        assert baseline_objective <= reference_cost + 1e-7 * max(1, reference_cost)
        data = dict(stem=stem, strategy=row['strategy'], input_index=int(row['output_index']),
                    nodes=nodes, edges=edges, demands=demands, baseline_objective=baseline_objective,
                    reference_cost=reference_cost, expected_seconds=float(row['gate_seconds']),
                    xml_sha256=digest(xml), metadata_sha256=digest(metadata),
                    source_solution_sha256={s: pool_hashes[s] for s in meta['source_names']})
        with gzip.open(output / 'inputs' / (stem + '.json.gz'), 'wt') as f:
            json.dump(data, f, separators=(',', ':'))
        records.append(dict(stem=stem, strategy=row['strategy'], seconds=float(row['gate_seconds']),
                            variables=int(row['gate_estimate_total_variable_count'])))
    expected = expected_inputs * 5
    if len(records) != expected or len({r['stem'] for r in records}) != expected:
        raise ValueError(f'Expected {expected} unique composites')
    counts = Counter(r['strategy'] for r in records)
    if set(counts) != set(METHODS) or set(counts.values()) != {expected_inputs}:
        raise ValueError('Expected equal complete inputs for all five methods')
    records.sort(key=lambda r: r['seconds'], reverse=True)
    smallest = min(records, key=lambda r: r['variables'])
    largest = max(records, key=lambda r: r['variables'])
    pilot = {smallest['stem'], largest['stem']}
    for filename, selection in [('pilot.jobs', [r for r in records if r['stem'] in pilot]),
                                ('remaining.jobs', [r for r in records if r['stem'] not in pilot]),
                                ('all.jobs', records)]:
        (output / filename).write_text(''.join(f"{r['stem']} {max(1, round(r['seconds']))}\n" for r in selection))
    info = dict(dataset=source_info, count=len(records),
                inherited_capacities='Exact capacities from the saved source MCMCF solutions, mapped through composition.',
                coupling_capacities='One existing zero-cost module on each coupling edge; the original generation check used zero.',
                primary_objective='Minimum routing cost, no failures and no capacity decisions.',
                secondary_objective='Minimum sum of edge flows after fixing routing cost to its optimum within numerical tolerance.',
                aggregation='For each edge, sum both nonnegative directed flows over all commodities.',
                baseline_validation='On pilot cases solve also with coupling capacities zero and compare to the saved generation objective.',
                pilot=list(pilot), historical_total_seconds=sum(r['seconds'] for r in records))
    (output / 'experiment.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(dict(count=len(records), pilot=sorted(pilot), input_bytes=sum(p.stat().st_size for p in (output/'inputs').iterdir()),
                          historical_total_seconds=info['historical_total_seconds']), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--run', type=Path, help='SND generation directory containing tables, instances, and plots.')
    inputs.add_argument('--archive', type=Path, help='Optional private evaluation archive.')
    parser.add_argument('--source-pool', type=Path, help='Source-pool directory containing solutions; required with --run.')
    parser.add_argument('--output', type=Path, required=True, help='New measurement directory.')
    parser.add_argument('--expected-inputs', type=int, default=100, help='Number of input sets per method.')
    parser.add_argument('--archive-prefix', default='planar-evaluation-20261110/code/check/results/')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must be a new directory')
    if args.expected_inputs <= 0:
        parser.error('--expected-inputs must be positive')
    if args.run:
        if args.source_pool is None:
            parser.error('--source-pool is required with --run')
        locations = {'generation': args.run, 'pool': args.source_pool}
        def read_bytes(name):
            group, relative = name.split('/', 1)
            return (locations[group] / relative).read_bytes()
        names = [f'pool/solutions/{p.name}' for p in sorted((args.source_pool / 'solutions').glob('*.json'))]
        prepare(read_bytes, names, 'generation/', 'pool/solutions/', args.output,
                {'run': str(args.run.resolve()), 'source_pool': str(args.source_pool.resolve())}, args.expected_inputs)
    else:
        prefix = args.archive_prefix.rstrip('/') + '/'
        with zipfile.ZipFile(args.archive) as archive:
            prepare(archive.read, archive.namelist(), prefix + 'ogdf_20261104_snd/',
                    prefix + 'sndlib_source_pool/solutions/', args.output,
                    {'archive': str(args.archive.resolve()), 'prefix': prefix}, args.expected_inputs)


if __name__ == '__main__':
    main()
