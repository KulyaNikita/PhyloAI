#!/usr/bin/env bash
# Install QPC into the selected Python environment without changing shell files.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    bash "${BASH_SOURCE[0]}" "$@"
    return $?
fi
set -euo pipefail

REPO_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${PHYLOAI_ENV_NAME:-phyloAI}"
USE_EXISTING=0
SKIP_POLYMOSIM=0
usage() {
    echo "Usage: bash install.sh [--existing-env] [--skip-polymosim]"
    echo "  --existing-env    Use PHYLOAI_PYTHON (default: python3) from an activated environment."
    echo "  --skip-polymosim  Install QPC without downloading/building the optional simulator."
}
for arg in "$@"; do
    case "$arg" in
        --existing-env) USE_EXISTING=1 ;;
        --skip-polymosim) SKIP_POLYMOSIM=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [[ "$USE_EXISTING" == 1 ]]; then
    PYTHON_CMD=("${PHYLOAI_PYTHON:-python3}")
else
    CONDA_BIN="${CONDA_EXE:-conda}"
    if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
        echo "Conda is required; alternatively activate an environment and use --existing-env." >&2
        exit 1
    fi
    if ! "$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys' >/dev/null 2>&1; then
        "$CONDA_BIN" env create --name "$ENV_NAME" --file "$REPO_PATH/environment.yml"
    else
        echo "Using existing conda environment: $ENV_NAME"
        if ! "$CONDA_BIN" run -n "$ENV_NAME" python -c \
            'import shutil; assert shutil.which("astral") and shutil.which("java")' >/dev/null 2>&1; then
            "$CONDA_BIN" install --yes --name "$ENV_NAME" --override-channels -c conda-forge -c bioconda \
                astral-tree=5.7.8 openjdk=17
        fi
    fi
    PYTHON_CMD=("$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python)
fi

"${PYTHON_CMD[@]}" -c 'import sys, setuptools, wheel, pybind11; print("Installing with", sys.executable)'
"${PYTHON_CMD[@]}" -c '
import shutil
if not shutil.which("astral") or not shutil.which("java"):
    raise SystemExit("ASTRAL and Java are required. Update your conda environment using environment.yml.")
'
# Compile for the selected interpreter and install in that environment.
"${PYTHON_CMD[@]}" -m pip install --no-deps --no-build-isolation \
    "$REPO_PATH/quartet-pattern-counter-v.1.6"
"${PYTHON_CMD[@]}" -c '
import QuartetPatternCounterModule as qpc
if getattr(qpc, "__version__", None) != "1.6.1":
    raise SystemExit("Another QPC shadows this installation: " + str(qpc.__file__) +
                     ". Remove old QPC directories from PYTHONPATH or the working directory.")
assert qpc.AA_PATTERN_ORDER == "ACDEFGHIKLMNPQRSTVWY"
print("Installed QPC", qpc.__version__, "from", qpc.__file__)
'

# Fetch only the pretrained network from Figshare version 22, not its datasets/scripts.
"${PYTHON_CMD[@]}" - "$REPO_PATH/ModelDetector.h5" <<'MODELDETECTOR'
import hashlib
import io
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile

target = Path(sys.argv[1])
model_sha256 = '3b6679c2e3fe9f6ae68296d48e3fcce350dde96776e02bb31240fc1db2ca6ea7'
if target.exists():
    if hashlib.sha256(target.read_bytes()).hexdigest() != model_sha256:
        raise SystemExit(f'Existing {target} differs from the Figshare network; refusing to overwrite it.')
    print(f'ModelDetector already installed and verified: {target}')
else:
    with urlopen('https://ndownloader.figshare.com/files/49575015', timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != '0d43f627fec89b1834125647ce82eeec79cdc8702db48375f15a7d94b6655669':
        raise SystemExit('ModelDetector archive checksum mismatch')
    with ZipFile(io.BytesIO(archive)) as package:
        network = package.read('test_scripts/ModelDetector.h5')
    if hashlib.sha256(network).hexdigest() != model_sha256:
        raise SystemExit('ModelDetector network checksum mismatch')
    target.write_bytes(network)
    print(f'Installed ModelDetector: {target}')
MODELDETECTOR

# Install the published QFM-FI application and compile the local ASTRAQ source.
"${PYTHON_CMD[@]}" - "$REPO_PATH/ASTRAQ" <<'AMALGAMATION'
import hashlib
import io
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
from urllib.request import urlopen
from zipfile import ZipFile

bin_dir = Path(sysconfig.get_path('scripts'))
qfm_dir = Path(sysconfig.get_path('data')) / 'share' / 'phyloai' / 'QFM-FI'
members = ('QFM-FI.jar', 'QFM-FI_lib/main.jar')
if not all((qfm_dir / name).is_file() for name in members):
    url = ('https://raw.githubusercontent.com/sharmin-mim/qfm_java/'
           '990507a4a6c82d49faff2922b6a753b10f98adba/QFM-FI.zip')
    with urlopen(url, timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != '450bec175d42c8eb821cce95028fae66ec28f8726026fd65bdf37addc1b92317':
        raise SystemExit('QFM-FI archive checksum mismatch')
    with ZipFile(io.BytesIO(archive)) as package:
        for name in members:
            target = qfm_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(package.read(name))

java = shutil.which('java')
if java is None:
    raise SystemExit('Java is required to run QFM-FI')
launcher = bin_dir / 'QFM-FI'
launcher.write_text('#!/bin/sh\nexec ' + shlex.quote(java) + ' -jar ' +
                    shlex.quote(str(qfm_dir / 'QFM-FI.jar')) + ' "$@"\n')
launcher.chmod(0o755)
source = Path(sys.argv[1]) / 'ASTRAQ.cpp'
compiler = shlex.split(os.environ.get('CXX', 'c++'))
subprocess.run(compiler + ['-std=c++17', '-O3', str(source), '-o', str(bin_dir / 'ASTRAQ')], check=True)
print(f'Installed QFM-FI and ASTRAQ in {bin_dir}')
AMALGAMATION

if [[ "$SKIP_POLYMOSIM" == 0 ]]; then
    POLYMOSIM_DIR="$REPO_PATH/PolyMoSim"
    POLYMOSIM_BRANCH="devel-5-2026"
    if [[ ! -d "$POLYMOSIM_DIR" ]]; then
        git clone --single-branch --branch "$POLYMOSIM_BRANCH" \
            https://github.com/cmayer/PolyMoSim.git "$POLYMOSIM_DIR"
    else
        if [[ -n "$(git -C "$POLYMOSIM_DIR" status --porcelain --untracked-files=no)" ]]; then
            echo "PolyMoSim has local changes; refusing to switch or update its source." >&2
            exit 1
        fi
        git -C "$POLYMOSIM_DIR" fetch origin \
            "$POLYMOSIM_BRANCH:refs/remotes/origin/$POLYMOSIM_BRANCH"
        if git -C "$POLYMOSIM_DIR" show-ref --verify --quiet "refs/heads/$POLYMOSIM_BRANCH"; then
            git -C "$POLYMOSIM_DIR" switch "$POLYMOSIM_BRANCH"
        else
            git -C "$POLYMOSIM_DIR" switch --track -c "$POLYMOSIM_BRANCH" "origin/$POLYMOSIM_BRANCH"
        fi
        git -C "$POLYMOSIM_DIR" merge --ff-only "origin/$POLYMOSIM_BRANCH"
    fi
    "${PYTHON_CMD[@]}" - "$POLYMOSIM_DIR" <<'BUILD_POLYMOSIM'
import os
import subprocess
import sys
subprocess.run(['make', '-C', sys.argv[1], 'CPP=' + os.environ.get('CXX', 'c++')], check=True)
BUILD_POLYMOSIM
    "${PYTHON_CMD[@]}" - "$POLYMOSIM_DIR" <<'PY'
import pathlib
import re
import shutil
import sys
import sysconfig
source_dir = pathlib.Path(sys.argv[1])
binary = re.search(r'^BIN\s*=\s*(\S+)', (source_dir / 'Makefile').read_text(), re.MULTILINE)
if binary is None:
    raise SystemExit('PolyMoSim Makefile does not declare its BIN target')
source = source_dir / binary.group(1)
if not source.is_file():
    raise SystemExit(f"Expected simulator binary not found: {source}")
target = pathlib.Path(sysconfig.get_path("scripts")) / "PolyMoSim"
shutil.copy2(source, target)
print(f"Installed simulator: {target}")
PY
fi

# Download Fisher and six-state-LA models from pinned Kaggle release 3.
"${PYTHON_CMD[@]}" "$REPO_PATH/download_protein_models.py" \
    --output "$REPO_PATH/PhyloAI-Fisher-Kaggle" --family all

echo "QPC, ModelDetector, PhyloAI protein models, ASTRAL, QFM-FI, and ASTRAQ installation complete."
if [[ "$USE_EXISTING" == 0 ]]; then
    echo "Activate the environment with: conda activate $ENV_NAME"
fi
echo "Run repository scripts with python and their paths; no shell startup files were changed."
