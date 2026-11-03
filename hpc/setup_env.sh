#!/bin/bash
# Create the Python environment (.venv in the repository root) with torch + dependencies.
# On a cluster run it on a LOGIN node (compute nodes usually have no internet) via ./hpc/bridge.sh setup.
#   TORCH_INDEX  wheel index matching the GPU driver, e.g. https://download.pytorch.org/whl/cu126
#                (old drivers, CUDA 11.x: .../whl/cu118 with TORCH_SPEC=torch==2.7.1); empty = PyPI default
#   MODULES      environment modules to load first (space separated), e.g. "python/3.11"
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"
if [ -n "${MODULES:-}" ] && command -v module >/dev/null 2>&1; then module purge || true; module load $MODULES; fi
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
echo "using $PY ($($PY --version))"
$PY -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
if [ -n "${TORCH_INDEX:-}" ]; then pip install --upgrade "${TORCH_SPEC:-torch}" --index-url "$TORCH_INDEX"
else pip install --upgrade "${TORCH_SPEC:-torch}"; fi
pip install "numpy<2.3" scipy matplotlib pyyaml   # pyyaml: refractiveindex.info database files
pip install --no-deps torcwa          # RCWA solver (LGPL); --no-deps keeps the torch build chosen above
echo "${MODULES:-}" > "$VENV/.modules"
python -c "import torch, torcwa; print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| cuda available', torch.cuda.is_available())"
echo "OK: environment in $VENV"
