#!/bin/bash
# One optimization on one GPU. Resources (partition, gres, time, account) come from the sbatch command line
# built by hpc/bridge.sh from hpc/site.env (GPU_SBATCH), so this file contains no site-specific settings.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
MODS=$(cat .venv/.modules 2>/dev/null || true); [ -n "$MODS" ] && { module purge; module load $MODS; }
unset PYTHONPATH   # site python modules may shadow the venv (Leonardo)
source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -u hpc/gpu_check.py
if [ "${1:-}" = "--check-only" ]; then exit 0; fi
CONFIG="$1"; shift
srun python -u run_topopt.py --config "$CONFIG" --device cuda "$@"
