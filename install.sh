#!/usr/bin/env bash
# Set up PhyloAI from this source folder (for development).
#
#   bash install.sh                 Create or update the conda environment from environment.yml
#                                   (tools from bioconda), install PhyloAI into it with
#                                   `pip install -e .` and download the models.
#   bash install.sh --existing-env  Install into the active environment (PHYLOAI_PYTHON, default
#                                   python3) instead. It must already have the dependencies.
#   bash install.sh --no-models     Do not download the models (run `phyloai download-models` later).
#
# The environment name is phyloAI; set PHYLOAI_ENV_NAME to change it. Models go into the PhyloAI
# data folder (PHYLOAI_DATA, default ~/.local/share/phyloai). No shell startup files are changed.
#
# To only use PhyloAI, installing the conda package is simpler:
#   conda create -n phyloai -c conda-forge -c bioconda phyloai
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    bash "${BASH_SOURCE[0]}" "$@"
    return $?
fi
set -euo pipefail

REPO_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${PHYLOAI_ENV_NAME:-phyloAI}"
USE_EXISTING=0
DOWNLOAD_MODELS=1
usage() {
    echo "Usage: bash install.sh [--existing-env] [--no-models]"
    echo "  --existing-env  Install into the active environment (PHYLOAI_PYTHON, default python3)."
    echo "  --no-models     Do not download the models."
}
for arg in "$@"; do
    case "$arg" in
        --existing-env) USE_EXISTING=1 ;;
        --no-models) DOWNLOAD_MODELS=0 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [[ "$USE_EXISTING" == 1 ]]; then
    PYTHON_CMD=("${PHYLOAI_PYTHON:-python3}")
else
    CONDA_BIN="${CONDA_EXE:-conda}"
    if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
        echo "Conda is required (Miniforge recommended); or activate an environment and use --existing-env." >&2
        exit 1
    fi
    if "$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys' >/dev/null 2>&1; then
        echo "Updating conda environment $ENV_NAME from environment.yml"
        "$CONDA_BIN" env update --name "$ENV_NAME" --file "$REPO_PATH/environment.yml"
    else
        echo "Creating conda environment $ENV_NAME from environment.yml"
        "$CONDA_BIN" env create --name "$ENV_NAME" --file "$REPO_PATH/environment.yml"
    fi
    PYTHON_CMD=("$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python)
fi

# The dependencies come from conda, so pip installs only PhyloAI itself.
"${PYTHON_CMD[@]}" -m pip install --no-deps --no-build-isolation -e "$REPO_PATH"

if [[ "$DOWNLOAD_MODELS" == 1 ]]; then
    "${PYTHON_CMD[@]}" -m phyloai download-models
fi
CHECK_ARGS=()
[[ "$DOWNLOAD_MODELS" == 1 ]] || CHECK_ARGS=(--no-models)
# (The ${...+...} form keeps an empty array working with `set -u` in macOS's bash 3.2.)
"${PYTHON_CMD[@]}" -m phyloai check ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}

if [[ "$USE_EXISTING" == 0 ]]; then
    echo "Activate the environment with: conda activate $ENV_NAME"
fi
echo "Then run: phyloai --help"
