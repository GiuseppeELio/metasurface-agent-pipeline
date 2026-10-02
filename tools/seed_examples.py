#!/usr/bin/env python3
"""Copy the worked-example runs (examples/si1550_runs) into runs/ without overwriting existing directories.

runs/ is never pushed to a cluster: run this once on the execution target, e.g. as a line of a GPU list
(`tools/seed_examples.py`) or locally (`python tools/seed_examples.py`).
"""
import os
import shutil
import sys
from pathlib import Path

if int(os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0"))) != 0:
    sys.exit(0)
ROOT = Path(__file__).resolve().parent.parent
src, dst = ROOT / "examples" / "si1550_runs", ROOT / "runs"
dst.mkdir(exist_ok=True)
n = 0
for p in sorted(src.iterdir()):
    q = dst / p.name
    if q.exists():
        continue
    (shutil.copytree if p.is_dir() else shutil.copy2)(p, q)
    n += 1
print(f"[seed_examples] copied {n} entries into {dst}")
