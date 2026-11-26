#!/bin/bash
# metasurf - compile and run the COMSOL models written by comsol_export.py (macOS / Linux, local COMSOL only).
#
#   bash validation/comsol/run_comsol.sh --build-only runs/<dir>        compile every runs/<dir>/comsol/MS_*.java to .class
#                                                                      (open the .class in COMSOL: File > Open)
#   bash validation/comsol/run_comsol.sh runs/<dir>                     compile + solve every mesh level in batch
#   bash validation/comsol/run_comsol.sh runs/<dir> MS_<case>_L1        only this class
#   bash validation/comsol/run_comsol.sh --cores 3 runs/<dir>           leave cores free to keep using the computer
#   bash validation/comsol/run_comsol.sh --mesh-only runs/<dir>         compile + build the mesh only (element count in the log)
#   bash validation/comsol/run_comsol.sh --no-save runs/<dir>           do not write the solved .mph (130-170 MB each; batch -nosave)
#
# Then:  python validation/comsol/comsol_compare.py --run runs/<dir>
#
# COMSOL executable: COMSOL=/path/to/comsol if it is not in the default location.
# COMSOL needs Preferences > Security > File system access: All files (the classes write their log and results).

C="${COMSOL:-/Applications/COMSOL64/Multiphysics/bin/comsol}"
NP=""; MODE="solve"; SAVE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --cores) NP="-np $2"; shift 2 ;;
    --build-only) MODE="build"; shift ;;
    --mesh-only) MODE="mesh"; shift ;;
    --no-save) SAVE=0; shift ;;
    *) break ;;
  esac
done
RUN="${1%/}"; shift
if [ -z "$RUN" ] || [ ! -d "$RUN/comsol" ]; then
  echo "usage: bash validation/comsol/run_comsol.sh [--build-only|--mesh-only] [--cores N] [--no-save] runs/<dir> [class ...]"
  echo "(no $RUN/comsol: run  python validation/comsol/comsol_export.py --run $RUN  first)"
  exit 1
fi
if [ ! -x "$C" ]; then
  echo "!!! COMSOL not found at $C : set COMSOL=/path/to/comsol"
  exit 1
fi
D="$(cd "$RUN/comsol" && pwd)"
if [ $# -gt 0 ]; then CLASSES="$*"
else CLASSES=$(ls "$D"/MS_*.java 2>/dev/null | xargs -n1 basename | sed 's/\.java$//')
fi
[ -n "$CLASSES" ] || { echo "!!! no MS_*.java in $D"; exit 1; }

KEEP=""; command -v caffeinate >/dev/null && KEEP="caffeinate -i"
for K in $CLASSES; do
  J="$D/$K.java"
  [ -f "$J" ] || { echo "!!! missing $J"; continue; }
  perl -pi -e "s|static String DIR = \".*\";|static String DIR = \"$D\";|" "$J"
  echo "=== $K: compile  $(date)"
  "$C" compile "$J" || { echo "!!! compile failed for $K"; continue; }
  sleep 3
  if [ "$MODE" = "build" ]; then
    echo "build" > "$D/$K.mode"
    echo "    $D/$K.class  (COMSOL: File > Open; builds the model without meshing or solving)"
    continue
  fi
  echo "$MODE" > "$D/$K.mode"
  OUT=(-nosave)
  [ "$SAVE" = "1" ] && OUT=(-outputfile "$D/${K}_$MODE.mph")
  echo "=== $K: $MODE  $(date)  ${NP:-all cores}"
  $KEEP "$C" batch $NP -inputfile "$D/$K.class" "${OUT[@]}" -batchlog "$D/${K}_batch.log" \
    || echo "!!! $K failed: see $D/${K}_batch.log and $D/${K}_log.txt"
  echo "build" > "$D/$K.mode"
  sleep 3
done
[ "$MODE" = "solve" ] && echo "Done. Next: python validation/comsol/comsol_compare.py --run $RUN"
[ "$MODE" = "build" ] && echo "Done. Open the .class files in COMSOL, or run without --build-only to solve in batch."
exit 0
