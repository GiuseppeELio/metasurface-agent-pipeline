#!/bin/bash
# Execution bridge between the AI agent and the compute resource (local machine or SLURM cluster).
#
# Run from the repository root:
#   ./hpc/bridge.sh help
#   ./hpc/bridge.sh watch          # agent mode: executes requests the agent writes in queue/*.req
#
# Settings live in hpc/site.env (never committed). The agent creates it with the user at first use;
# see AGENTS.md ("First-run onboarding") and hpc/site.env.example.
#
# Modes (EXEC_MODE in site.env):
#   local  : jobs run on this machine (CUDA GPU if available, else CPU)
#   slurm  : jobs are submitted over SSH to a SLURM cluster; code is synced with rsync
#
# Compatible with bash 3.2 (macOS default) and Linux bash.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ENVF="$HERE/site.env"
[ -f "$ENVF" ] || { echo "Missing hpc/site.env: ask your agent to run the first-run onboarding (see AGENTS.md)."; exit 1; }
# shellcheck disable=SC1090
source "$ENVF"
EXEC_MODE="${EXEC_MODE:-local}"
Q="$REPO/queue"
mkdir -p "$REPO/logs" "$REPO/runs" "$Q/done" "$Q/failed"

# ---------------------------------------------------------------- local helpers
LOCAL_PY="${LOCAL_PYTHON:-python3}"
LOCAL_DEV="${LOCAL_DEVICE:-auto}"
LOCAL_MPI_NP="${LOCAL_MPI_NP:-4}"
local_run() {  # $1=log name, rest=command; returns exit code
  local name="$1"; shift
  ( cd "$REPO" && "$@" ) < /dev/null > "$REPO/logs/$name.out" 2>&1
}

# ---------------------------------------------------------------- slurm helpers
if [ "$EXEC_MODE" = "slurm" ]; then
  : "${SSH_TARGET:?set SSH_TARGET=user@login.host in hpc/site.env}"
  REMOTE_DIR="${REMOTE_DIR:-metasurface-agent-pipeline}"   # relative to the remote $HOME
  CTRL="$HOME/.ssh/cm-bridge-%r@%h:%p"
  mkdir -p "$HOME/.ssh"
  SSH=(ssh -o BatchMode=yes -o ServerAliveInterval=60 -o ControlMaster=auto -o ControlPersist=4h -o ControlPath="$CTRL" "$SSH_TARGET")
  RSYNC_E="ssh -o ControlMaster=auto -o ControlPersist=4h -o ControlPath=$CTRL"
  ACC=""; [ -n "${SLURM_ACCOUNT:-}" ] && ACC="--account=$SLURM_ACCOUNT"
  remote() { "${SSH[@]}" "cd ~/$REMOTE_DIR && $*"; }
fi

usage() { sed -n '/^# Commands:/,/^# end/p' "$0" | sed 's/^# \{0,1\}//'; }
# Commands:
#   login                      run SSH_LOGIN_CMD (e.g. a 2FA certificate helper); slurm only
#   push | pull                sync code to / results from the cluster; slurm only
#   setup | setup-meep         create the Python env (torch + deps) / the Meep env on the target
#   check                      short GPU sanity job (CUDA visible? eig benchmark)
#   submit <config> [args]     one optimization (run_topopt.py)
#   array <list.txt>           one task per line (run_topopt.py args, or a line starting with <script>.py)
#   meep <list.txt>            Meep FDTD validations, one meep_validate.py call per line (CPU)
#   status | log <id> | cancel <id> | usage [YYYY-MM-DD]
#   watch                      agent mode: execute queue/*.req, report state in queue/.watcher_state
# end

cmd="${1:-help}"; shift || true
case "$cmd" in
  help|-h|--help) usage ;;

  login)
    [ -n "${SSH_LOGIN_CMD:-}" ] && eval "$SSH_LOGIN_CMD" || echo "No SSH_LOGIN_CMD configured (nothing to do)." ;;

  push)
    [ "$EXEC_MODE" = "slurm" ] || { echo "push: local mode, nothing to do"; exit 0; }
    "${SSH[@]}" "mkdir -p ~/$REMOTE_DIR"
    rsync -az --delete -e "$RSYNC_E" --exclude '.git' --exclude '.DS_Store' --exclude 'runs/' --exclude 'logs/' \
      --exclude 'queue/' --exclude 'hpc/site.env' --exclude '.venv' --exclude '.meep-env' --exclude '.mamba' \
      "$REPO/" "$SSH_TARGET:~/$REMOTE_DIR/" ;;

  pull)
    [ "$EXEC_MODE" = "slurm" ] || { echo "pull: local mode, nothing to do"; exit 0; }
    "${SSH[@]}" "mkdir -p ~/$REMOTE_DIR/runs ~/$REMOTE_DIR/logs"
    rsync -az -e "$RSYNC_E" --exclude 'checkpoint.pt' "$SSH_TARGET:~/$REMOTE_DIR/runs/" "$REPO/runs/"
    rsync -az -e "$RSYNC_E" "$SSH_TARGET:~/$REMOTE_DIR/logs/" "$REPO/logs/" ;;

  setup)
    if [ "$EXEC_MODE" = "slurm" ]; then
      "${SSH[@]}" "cd ~/$REMOTE_DIR && MODULES='${ENV_MODULES:-}' TORCH_INDEX='${TORCH_INDEX:-}' TORCH_SPEC='${TORCH_SPEC:-torch}' bash hpc/setup_env.sh"
    else
      ( cd "$REPO" && TORCH_INDEX="${TORCH_INDEX:-}" TORCH_SPEC="${TORCH_SPEC:-torch}" bash hpc/setup_env.sh )
    fi ;;

  setup-meep)
    if [ "$EXEC_MODE" = "slurm" ]; then "${SSH[@]}" "cd ~/$REMOTE_DIR && bash hpc/setup_meep.sh"
    else ( cd "$REPO" && bash hpc/setup_meep.sh ); fi ;;

  check)
    if [ "$EXEC_MODE" = "slurm" ]; then
      remote "sbatch $ACC ${GPU_SBATCH:-} ${GPU_SHORT_SBATCH:---time=00:30:00} --job-name=gpucheck hpc/slurm/job_gpu.slurm --check-only"
    else
      local_run gpucheck "$REPO/.venv/bin/python" -u hpc/gpu_check.py && echo "Submitted local job gpucheck" || echo "local gpucheck failed (see logs/gpucheck.out)"
    fi ;;

  submit)
    if [ "$EXEC_MODE" = "slurm" ]; then
      remote "sbatch $ACC ${GPU_SBATCH:-} --job-name=topopt hpc/slurm/job_gpu.slurm $*"
    else
      id="local$(date +%s)"
      local_run "topopt_$id" "$REPO/.venv/bin/python" -u run_topopt.py --device "$LOCAL_DEV" --config "$@" \
        && echo "Submitted local job $id COMPLETED" || echo "Submitted local job $id FAILED"
    fi ;;

  array)
    LIST="$1"
    if [ "$EXEC_MODE" = "slurm" ]; then
      n=$(grep -c . "$REPO/$LIST" | tr -d ' ')
      remote "sbatch $ACC ${GPU_SBATCH:-} --job-name=topopt_arr --array=1-$n%${MAX_PARALLEL_GPU:-1} hpc/slurm/job_array.slurm $LIST"
    else
      id="local$(date +%s)"; i=0; st=COMPLETED
      while IFS= read -r line || [ -n "$line" ]; do
        [ -z "$line" ] && continue; i=$((i+1)); set -f
        first=${line%% *}
        if [ "${first##*.}" = "py" ]; then local_run "arr_${id}_$i" "$REPO/.venv/bin/python" -u $line || st=FAILED
        else local_run "arr_${id}_$i" "$REPO/.venv/bin/python" -u run_topopt.py --device "$LOCAL_DEV" $line || st=FAILED; fi
        set +f
      done < "$REPO/$LIST"
      echo "Submitted local job $id $st"
    fi ;;

  meep)
    LIST="$1"
    if [ "$EXEC_MODE" = "slurm" ]; then
      remote "sbatch $ACC ${CPU_SBATCH:-} --job-name=meep hpc/slurm/job_meep.slurm $LIST"
    else
      id="local$(date +%s)"; st=COMPLETED; i=0
      while IFS= read -r line || [ -n "$line" ]; do
        [ -z "$line" ] && continue; i=$((i+1)); set -f
        case "$line" in *.py\ *|*.py) mline="$line" ;; *) mline="validation/meep_validate.py $line" ;; esac
        local_run "meep_${id}_$i" "$REPO/.meep-env/bin/mpirun" -np "$LOCAL_MPI_NP" "$REPO/.meep-env/bin/python" -u $mline || st=FAILED
        set +f
      done < "$REPO/$LIST"
      echo "Submitted local job $id $st"
    fi ;;

  status) [ "$EXEC_MODE" = "slurm" ] && "${SSH[@]}" "squeue --me -o '%.12i %.14j %.8T %.10M %.10l %R'" || echo "local mode: jobs run synchronously in the watcher" ;;
  log)    if [ "$EXEC_MODE" = "slurm" ]; then remote "tail -n ${2:-40} logs/*_$1*.out"; else tail -n "${2:-40}" "$REPO"/logs/*"$1"*.out; fi ;;
  cancel) [ "$EXEC_MODE" = "slurm" ] && "${SSH[@]}" "scancel $1" ;;
  usage)  # consumption since a date from sacct: billing units x elapsed hours
    [ "$EXEC_MODE" = "slurm" ] || { echo "local mode: no accounting"; exit 0; }
    since="${1:-$(date +%Y-%m-%d)}"
    "${SSH[@]}" "sacct -X -n -P -S $since -o JobID,JobName,State,ElapsedRaw,AllocCPUS,AllocTRES" | awk -F'|' '
      { b=$5; n=split($6,t,","); for(i=1;i<=n;i++) if (t[i] ~ /^billing=/) { sub("billing=","",t[i]); b=t[i] }
        h=$4/3600; tot+=h*b; printf "%-14s %-14s %-10s %7.2f h x %4s = %8.2f\n", $1, $2, $3, h, b, h*b }
      END { printf "TOTAL (billing x elapsed) since '"$since"': %.1f core-hours\n", tot }' ;;

  watch)
    # Agent mode. A request is a file queue/<name>.req containing ONE line with an allowed subcommand, e.g.
    #   submit configs/metagrating_si1550_robust.json --iters 300
    # The watcher validates it, (slurm) pushes code and submits, follows the job, pulls results, then moves
    # the request to queue/done/ (with .state) or queue/failed/ (with .err). Heartbeat: queue/.heartbeat
    echo "[watch] mode=$EXEC_MODE queue=$Q (Ctrl-C to stop)"
    while true; do
      date '+%Y-%m-%d %H:%M:%S' > "$Q/.heartbeat"
      if [ "$EXEC_MODE" = "slurm" ]; then
        if ! "${SSH[@]}" true 2>/dev/null; then
          echo "ssh_down $(date '+%H:%M')" > "$Q/.watcher_state"
          echo "[watch] $(date '+%H:%M') SSH unavailable (expired certificate/key?) -> ./hpc/bridge.sh login"; sleep 60; continue
        fi
        "${SSH[@]}" "squeue --me -o '%.12i %.14j %.8T %.10M %.10l %R'" > "$Q/.squeue" 2>&1 || true
      fi
      for req in "$Q"/*.req; do
        [ -e "$req" ] || continue
        [ -e "$req.jobid" ] && continue
        line=$(head -n 1 "$req" | tr -d '\r')
        sub=${line%% *}
        case "$sub" in submit|array|meep|check) ;; *) echo "subcommand not allowed: $sub" > "$req.err"; mv "$req" "$req.err" "$Q/failed/"; continue ;; esac
        case "$line" in *[\;\|\&\$\`\>\<]*) echo "forbidden characters" > "$req.err"; mv "$req" "$req.err" "$Q/failed/"; continue ;; esac
        echo "[watch] $(date '+%H:%M') new request $(basename "$req"): $line"
        echo "running $(basename "$req")" > "$Q/.watcher_state"
        [ "$EXEC_MODE" = "slurm" ] && { "$0" push >/dev/null 2>&1 || { echo "[watch] push failed"; continue; }; }
        set -f; outp=$("$0" $line 2>&1) || true; set +f
        jid=$(echo "$outp" | sed -nE 's/.*Submitted (batch|local) job ([0-9a-z]+).*/\2/p' | head -n 1)
        if [ -z "$jid" ]; then
          echo "$outp" > "$req.err"; mv "$req" "$req.err" "$Q/failed/"; echo "[watch] submission failed"; continue
        fi
        echo "$jid" > "$req.jobid"; echo "[watch] job $jid"
        if [ "$EXEC_MODE" = "local" ]; then   # local jobs are synchronous: already finished
          echo "$outp" > "$req.state"; mv "$req" "$req.jobid" "$req.state" "$Q/done/"
          echo "[watch] $(date '+%H:%M') local job $jid finished"
        fi
      done
      if [ "$EXEC_MODE" = "slurm" ]; then
        for jf in "$Q"/*.req.jobid; do
          [ -e "$jf" ] || continue
          jid=$(cat "$jf"); req=${jf%.jobid}
          if [ -z "$("${SSH[@]}" "squeue -h -j $jid" 2>/dev/null)" ]; then
            "$0" pull >/dev/null 2>&1 || true
            "${SSH[@]}" "sacct -X -n -j $jid -o JobID,State,Elapsed,ExitCode" > "$req.state" 2>&1 || true
            mv "$req" "$jf" "$req.state" "$Q/done/"
            echo "[watch] $(date '+%H:%M') job $jid finished, results pulled"
          fi
        done
      fi
      echo "ok $(date '+%H:%M')" > "$Q/.watcher_state"
      sleep "${WATCH_INTERVAL:-60}"
    done ;;

  *) usage; exit 1 ;;
esac
