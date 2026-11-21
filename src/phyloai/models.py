"""Download and verify the pretrained networks and ModelDetector (`phyloai download-models`).

The protein networks come from the pinned Kaggle release (no Kaggle credentials needed) and
ModelDetector from Figshare. Every file is checked against a SHA-256 checksum; existing files
that differ are never overwritten. By default everything goes into the PhyloAI data folder
(the PHYLOAI_DATA environment variable, otherwise ~/.local/share/phyloai on Linux and
~/Library/Application Support/phyloai on macOS).
"""
import argparse
import hashlib
import io
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZipFile

from phyloai import paths

API = 'https://api.kaggle.com/v1/models.ModelApiService/'
RELEASE = dict(ownerSlug='nikitakulya', modelSlug='phyloai-protein',
               framework='tensorFlow2', instanceSlug='topology-aware-length-aware', versionNumber=3)
PREFIX = 'PhyloAI-Fisher-Kaggle/PhyloAI-Fisher-Kaggle'
MANIFEST_SHA256 = '6166b5297a46e37e002c8bdbeb6c309368901212ce64bbc782a99021f9dd4569'
FAMILIES = ('fisher', '6-state-la')

# ModelDetector: only the network from Tinh Nguyen's Figshare deposit (version 22, CC BY 4.0).
DETECTOR_URL = 'https://ndownloader.figshare.com/files/49575015'
DETECTOR_ARCHIVE_SHA256 = '0d43f627fec89b1834125647ce82eeec79cdc8702db48375f15a7d94b6655669'
DETECTOR_MEMBER = 'test_scripts/ModelDetector.h5'
DETECTOR_SHA256 = '3b6679c2e3fe9f6ae68296d48e3fcce350dde96776e02bb31240fc1db2ca6ea7'


def manifest_file(name):
    return resources.files('phyloai') / 'data' / name


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
    manifest = manifest_file('six_state_la_models.SHA256SUMS' if six_state else 'protein_models.SHA256SUMS')
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


def download_detector(target):
    """Download ModelDetector.h5 from Figshare and check it; an existing verified file is kept."""
    target = Path(target)
    if target.exists():
        if digest(target) != DETECTOR_SHA256:
            raise ValueError(f'Existing {target} differs from the Figshare network; refusing to overwrite it.')
        print(f'ModelDetector already installed and verified: {target}')
        return
    with urlopen(DETECTOR_URL, timeout=120) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != DETECTOR_ARCHIVE_SHA256:
        raise ValueError('ModelDetector archive checksum mismatch')
    with ZipFile(io.BytesIO(archive)) as package:
        network = package.read(DETECTOR_MEMBER)
    if hashlib.sha256(network).hexdigest() != DETECTOR_SHA256:
        raise ValueError('ModelDetector network checksum mismatch')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.download')
    partial.write_bytes(network)
    partial.replace(target)
    print(f'Installed ModelDetector: {target}')


def build_parser(prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', type=Path, default=paths.models_dir(),
                        help='Folder for the networks (default: %(default)s)')
    parser.add_argument('--detector', type=Path, default=paths.detector_path(),
                        help='File for ModelDetector (default: %(default)s)')
    parser.add_argument('--family', choices=['all', *FAMILIES, 'none'], default='all',
                        help='Network families to download; none downloads only ModelDetector (default: all)')
    parser.add_argument('--no-detector', action='store_true', help='Do not download ModelDetector')
    return parser


def main(argv=None, prog=None):
    args = build_parser(prog).parse_args(argv)
    families = list(FAMILIES) if args.family == 'all' else [] if args.family == 'none' else [args.family]
    for family in families:
        download(args.models, family)
    if not args.no_detector:
        download_detector(args.detector)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
