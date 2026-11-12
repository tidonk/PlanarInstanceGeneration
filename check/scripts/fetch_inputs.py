#!/usr/bin/env python3
"""Download benchmark inputs from their owners and verify the recorded hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
PGLIB_REVISION = 'dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3'


def verify(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Benchmark differs from the paper input: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['pglib', 'sndlib', 'all'], default='all')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--directory', type=Path, default=ROOT / 'check/data')
    args = parser.parse_args()
    with (ROOT / 'reproduction/input-files.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if args.dataset in ('all', r['dataset'])]
    def fetch(row):
        domain, name = row['dataset'], row['filename']
        target = args.directory / domain / name
        if target.is_file():
            verify(target, row['sha256']);return
        if args.verify_only:
            raise FileNotFoundError(target)
        if domain == 'pglib':
            url = f'https://raw.githubusercontent.com/power-grid-lib/pglib-opf/{PGLIB_REVISION}/{name}'
        else:
            url = 'https://sndlib.zib.de/coredata.download.action?' + urlencode({
                'objectName': Path(name).stem, 'format': 'xml', 'objectType': 'network'})
        print('Downloading', url, flush=True)
        try:
            with urlopen(url, timeout=60) as stream: data=stream.read()
        except Exception as error:
            raise RuntimeError(f'Cannot download {url}. Obtain {name} from the source site, '
                               f'place it at {target}, and rerun --verify-only.') from error
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Downloaded data differ from the recorded input: {url}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.tmp');temporary.write_bytes(data);temporary.replace(target)
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(fetch, rows))
    print(f'Verified {len(rows)} original input files.')

if __name__=='__main__': main()
