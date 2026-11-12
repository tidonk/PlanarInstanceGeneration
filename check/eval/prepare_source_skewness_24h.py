#!/usr/bin/env python3
"""Prepare (do not submit) normal-user source solves for Table 2."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'ogdf_sources_20261106'

def main():
    logs = ROOT/'check/eval/logs'/PREFIX
    logs.mkdir(parents=True, exist_ok=True)
    pglib = json.loads((ROOT/'check/results/pglib_source_pool_innoptem/manifest.json').read_text())
    snd = json.loads((ROOT/'check/results/sndlib_source_pool/manifest.json').read_text())
    names = lambda pool: [r['source_name'] for r in pool['records'] if r.get('retained') and r.get('bus_count', 73) >= 73]
    pn, sn = names(pglib), names(snd)
    assert len(pn) == len(sn) == 16, (pn, sn)
    common = f'''universe = vanilla
executable = .venv/bin/python
should_transfer_files = NO
getenv = True
nice_user = False
priority = -100
batch_name = pig-ogdf_20261106-source-skewness
+FileSystemDomain = "clustor"
request_cpus = 16
request_memory = 14336
request_disk = 4194304
+MaxRuntime = 93600
periodic_hold = (JobStatus == 2) && ((time() - JobCurrentStartExecutingDate) > 97200)
on_exit_hold = (ExitCode =!= 0)
output = check/eval/logs/{PREFIX}/$(kind)_$(source).out
error = check/eval/logs/{PREFIX}/$(kind)_$(source).err
log = check/eval/logs/{PREFIX}/cluster.log
'''
    (ROOT/f'check/eval/{PREFIX}_pglib.sub').write_text(common + f'''arguments = check/scripts/evaluate_pglib_source_skewness.py $(source) --time-limit 86400 --threads 16 --seed 0 --output-directory check/results/{PREFIX}_$(kind) $(extra)
queue
''')
    (ROOT/f'check/eval/{PREFIX}_snd.sub').write_text(common + f'''arguments = check/scripts/evaluate_snd_source_skewness.py --source $(source) --time-limit 86400 --threads 16 --output check/results/{PREFIX}_snd/$(source)
queue
''')
    lines = ['# Prepared only. Each graph gets a 24-hour OGDF solve limit.']
    for name in pn:
        for kind, extra in [('voltage','--voltage-separated'), ('pglib',f'--voltage-warm-start --voltage-certificate-directory {ROOT}/check/results/{PREFIX}_voltage')]:
            node=f'{kind}_{name}'
            lines += [f'JOB {node} check/eval/{PREFIX}_pglib.sub', f'VARS {node} source="{name}" kind="{kind}" extra="{extra}"']
        lines += [f'PARENT voltage_{name} CHILD pglib_{name}']
    for name in sn:
        lines += [f'JOB snd_{name} check/eval/{PREFIX}_snd.sub',f'VARS snd_{name} source="{name}" kind="snd"']
    collect = common.replace('request_cpus = 16', 'request_cpus = 1')
    (ROOT/f'check/eval/{PREFIX}_collect.sub').write_text(collect + f'arguments = check/scripts/evaluate_snd_source_skewness.py --merge-directory check/results/{PREFIX}_snd\nqueue\n')
    lines += [f'JOB collect_snd check/eval/{PREFIX}_collect.sub', 'VARS collect_snd kind="collect" source="snd"', 'PARENT ' + ' '.join('snd_'+n for n in sn) + ' CHILD collect_snd']
    path=ROOT/f'check/eval/{PREFIX}.dag'
    if path.exists(): raise FileExistsError(path)
    path.write_text('\n'.join(lines)+'\n')
    (logs/'sources.json').write_text(json.dumps(dict(pglib=pn,snd=sn),indent=2)+'\n')
    print(f'Prepared {path}: {len(pn)*2+len(sn)} solves; not submitted.')

if __name__ == '__main__': main()
