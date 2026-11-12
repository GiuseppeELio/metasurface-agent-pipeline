#!/bin/bash
# Job array: task i runs line i of the list file. A line starting with <script>.py runs that script
# (e.g. select_atoms.py, supercell.py, evaluate.py), otherwise run_topopt.py with that line as arguments.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
MODS=$(cat .venv/.modules 2>/dev/null || true); [ -n "$MODS" ] && { module purge; module load $MODS; }
unset PYTHONPATH   # site python modules may shadow the venv (Leonardo)
source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "task $SLURM_ARRAY_TASK_ID: $ARGS"
set -f
FIRST=${ARGS%% *}
if [[ "$FIRST" == *.py ]]; then srun python -u $ARGS
else srun python -u run_topopt.py --device cuda $ARGS; fi
