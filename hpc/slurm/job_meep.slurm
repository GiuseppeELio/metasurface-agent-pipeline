#!/bin/bash
# Meep FDTD jobs on a CPU node, one MPI run per line of the list file:
#   a line holds validation/meep_validate.py arguments, or starts with <script>.py (e.g. validation/meep_lens.py ...).
# A failing line is reported and the loop continues with the next one.
# Resources come from CPU_SBATCH in hpc/site.env (e.g. --partition=<cpu> --ntasks=48 --mem=300G --time=04:00:00).
#SBATCH --nodes=1
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PATH="$PWD/.meep-env/bin:$PATH" OMP_NUM_THREADS=1
NP=${SLURM_NTASKS:-1}
LIST="$1"
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  set -f; set -- $line; set +f
  case "$1" in *.py) script="$1"; shift ;; *) script=validation/meep_validate.py ;; esac
  echo "=== $(date +%H:%M:%S) $script $*"
  mpirun -np "$NP" python -u "$script" "$@" < /dev/null || echo "!!! FAILED: $script $*"
done < "$LIST"
