#!/usr/bin/env python3
"""Check complete feasible output after removing preliminary OGDF initialization."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

from ogdf_validation import digest, validate, write_json
from ogdf_validation import ENV, invoke
from ogdf_validation import check_blocks, read_graph


def check(job):
    driver, graph, cert, target, budget, mode, cpu, expected = job
    data = list(map(int, graph.read_text().split()))
    edges = list(zip(data[2::2], data[3::2]))
    item = dict(n=data[0], m=data[1], edges=edges, components=1,
                historical_bounds=expected)
    if target.exists():
        result = json.loads(target.read_text())
        assert result['driver_sha256'] == digest(driver)
        assert result['input_sha256'] == digest(graph)
        assert result['certificate_sha256'] == digest(cert)
        assert result['time_limit_seconds'] == budget and result['reuse_mode'] == mode
        validate(item, result)
    else:
        result = invoke(driver, graph, cert, target, budget, 20261104, mode, cpu, item)
    check_blocks(read_graph(graph), result)
    if mode & 1:
        assert result['deletion_upper_bound'] <= int(cert.read_text().split()[2])
    if budget:
        assert result['deletion_lower_bound'] == result['deletion_upper_bound'] == expected[0]
    else:
        assert all(b['linear_programs'] == b['subproblems'] == 0 for b in result['blocks'])
    return target.stem, result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('driver', type=Path)
    ap.add_argument('saved', type=Path)
    args = ap.parse_args()
    out, driver, saved = args.output.resolve(), args.driver.resolve(), args.saved.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((saved / 'summary.json').read_text())
    groups = json.loads((saved / 'settings.json').read_text())['cpu_groups']
    jobs = []
    for i, (name, record) in enumerate(summary.items()):
        folder = saved / name
        graph, cert = out / f'{name}.graph', out / f'{name}.cert'
        shutil.copy2(folder / 'final.graph', graph)
        shutil.copy2(folder / 'final.cert', cert)
        cpu = ','.join(map(str, groups[i % len(groups)]))
        for budget in (0, 60):
            jobs.append((driver, graph, cert, out / f'{name}.{budget}s.json',
                         budget, 3, cpu, record['interval']))
    known = out.parent / 'known'
    for name, value in [('K44', 4), ('two_K33', 2)]:
        cert = known / f'{name}.cert'
        key = name
        if name == 'two_K33':
            # The older verifier deliberately joins both nonplanar blocks.
            # Here preserve decomposition and omit its spanning bound row.
            data = list(map(int, cert.read_text().split()))
            _, lower, nwarm, _ = data[:4]
            cert = out / 'two_K33_blocks.cert'
            cert.write_text(f'0 {lower} {nwarm} 0\n' +
                            ' '.join(map(str, data[4:4 + nwarm])) + '\n')
            key = 'two_K33_blocks'
        for mode in range(4):
            jobs.append((driver, known / f'{name}.graph', cert,
                         out / f'{key}.zero_mode{mode}.json', 0, mode,
                         str(groups[mode % len(groups)][0]), [value, value]))
    # One list per CPU allocation avoids concurrent jobs sharing the same cores.
    records = {}
    for offset in range(0, len(jobs), 2 * len(groups)):
        batch = jobs[offset:offset + 2 * len(groups)]
        for parity in (0, 1):
            with ThreadPoolExecutor(max_workers=len(groups)) as pool:
                for key, result in pool.map(check, batch[parity::2]):
                    records[key] = result
    graph = known / 'K44.graph'
    m = int(graph.read_text().split()[1])
    invalid = out / 'disconnected.cert'
    invalid.write_text(f'0 {m} 0\n' + ' '.join(map(str, range(m))) + '\n')
    proc = subprocess.run([str(driver), str(graph), str(out / 'invalid.json'),
                           '0', '7', str(invalid), '1'], env=ENV,
                          capture_output=True, text=True, timeout=10)
    assert proc.returncode == 2 and 'planar and connected' in proc.stderr
    result = dict(passed=len(records), disconnected_start_rejected=True,
                  records=records, driver_sha256=digest(driver),
                  script_sha256=digest(Path(__file__)),
                  note='Saved graphs were solved again; their composition paths were not regenerated.')
    write_json(out / 'verification.json', result)
    print(f'All {len(records)} completion checks passed; disconnected start rejected.')


if __name__ == '__main__':
    main()
