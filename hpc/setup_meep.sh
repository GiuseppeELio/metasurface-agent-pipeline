#!/bin/bash
# Install MIT Meep (MPI build, conda-forge) in .meep-env using a private micromamba. Needs internet.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) PLAT=linux-64 ;; Linux-aarch64) PLAT=linux-aarch64 ;;
  Darwin-arm64) PLAT=osx-arm64 ;; Darwin-x86_64) PLAT=osx-64 ;; *) echo "unsupported platform"; exit 1 ;;
esac
[ -x bin/micromamba ] || curl -Ls "https://micro.mamba.pm/api/micromamba/$PLAT/latest" | tar -xvj bin/micromamba
export MAMBA_ROOT_PREFIX="$ROOT/.mamba"
./bin/micromamba create -y -p "$ROOT/.meep-env" -c conda-forge python=3.11 "pymeep=*=mpi_mpich_*" numpy scipy h5py
"$ROOT/.meep-env/bin/mpirun" -np 2 "$ROOT/.meep-env/bin/python" -c "import meep as mp; print('meep', mp.__version__)"
echo "OK: Meep environment in $ROOT/.meep-env"
