#!/usr/bin/env python3
"""Download the pinned public protein release without Kaggle credentials."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
from urllib.request import Request, urlopen

API = 'https://api.kaggle.com/v1/models.ModelApiService/'
RELEASE = dict(ownerSlug='nikitakulya', modelSlug='phyloai-protein',
               framework='tensorFlow2', instanceSlug='topology-aware-length-aware', versionNumber=3)
PREFIX = 'PhyloAI-Fisher-Kaggle/PhyloAI-Fisher-Kaggle'
MANIFEST_SHA256 = '6166b5297a46e37e002c8bdbeb6c309368901212ce64bbc782a99021f9dd4569'


def request(method, **fields):
    # No SDK, cookie jar, Authorization header, or credential-file access.
    return urlopen(Request(API + method, data=json.dumps(dict(RELEASE, **fields)).encode(),
                           headers={'Content-Type': 'application/json'}), timeout=120)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe release path: {name}')
    return Path(*path.parts)


def download(output, family='fisher'):
    six_state = family == '6-state-la'
    manifest = Path(__file__).with_name('six_state_la_models.SHA256SUMS' if six_state else 'protein_models.SHA256SUMS')
    manifest_sha256 = 'edd16cfb8abf3f12026718dfb5afcee40032dde5053d9f2e92cc39b3bd9dcbb1' if six_state else MANIFEST_SHA256
    prefix = 'PhyloAI-6-state-LA-Kaggle/PhyloAI-6-state-LA-Kaggle' if six_state else PREFIX
    if digest(manifest) != manifest_sha256:
        raise ValueError('Bundled release checksum manifest has changed')
    expected = {}
    for line in manifest.read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        relative(name)
        expected[prefix + '/' + name] = checksum
    file_count = 19 if six_state else 105
    if len(expected) != file_count:
        raise ValueError(f'Expected checksums for {file_count} release files')
    output.mkdir(parents=True, exist_ok=True)

    def fetch(name, expected=None):
        local_name = name.removeprefix(prefix + '/')
        if six_state and local_name == 'README.md':
            local_name = 'README-6-state-LA.md'
        target = output / relative(local_name)
        if target.exists():
            if digest(target) == expected:
                return target
            raise ValueError(f'Existing file differs from release; refusing to overwrite: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.download')
        for attempt in range(3):
            try:
                with request('DownloadModelInstanceVersion', path=name) as response, partial.open('wb') as handle:
                    shutil.copyfileobj(response, handle)
                if digest(partial) != expected:
                    raise ValueError(f'Download verification failed: {name}')
                partial.replace(target)
                print(f'Verified {target.relative_to(output)}', flush=True)
                return target
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda name: fetch(name, expected[name]), sorted(expected)))
    models = list(output.glob('PhyloAI-*-6-LA/saved_model.pb' if six_state else 'PhyloAI-Fisher-*/saved_model.pb'))
    model_count = 3 if six_state else 15
    if len(models) != model_count:
        raise ValueError(f'Expected {model_count} networks; found {len(models)}')
    print(f'All {len(expected)} files verified; {model_count} {family} models installed in {output.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'PhyloAI-Fisher-Kaggle')
    parser.add_argument('--family', choices=['all', 'fisher', '6-state-la'], default='all')
    args = parser.parse_args()
    for family in (['fisher', '6-state-la'] if args.family == 'all' else [args.family]):
        download(args.output, family)
