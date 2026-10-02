#!/usr/bin/env python3
"""Generate a job-array list for a meta-atom phase library (LUT) from a disk sweep (disk_sweep.py output).

For every target phase, the initial disk radius is taken from the sweep (best phase match with T > tmin);
optional extra starts (clover, cross, square, ring) are scaled to the same Si area. The list ends with
select_atoms.py and, optionally, supercell.py lines.

  python tools/make_lut_sweep.py --sweep runs/<disk_sweep>/sweep.json --height 800 --step 15 \
      --name atomLUT --shapes disk clover cross --supercells 8 12 24 > sweeps/lut.txt
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rt", HERE / "run_topopt.py")
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", required=True)
ap.add_argument("--height", type=float, required=True, help="layer thickness (nm), must exist in the sweep")
ap.add_argument("--step", type=float, default=15.0, help="phase step (deg)")
ap.add_argument("--name", default="atomLUT")
ap.add_argument("--config", default="configs/metaatom_template.json")
ap.add_argument("--shapes", nargs="+", default=["disk", "clover", "cross"])
ap.add_argument("--tmin", type=float, default=0.85)
ap.add_argument("--edge-shift", type=float, default=10.0)
ap.add_argument("--supercells", type=int, nargs="*", default=[])
ap.add_argument("--prefixes", nargs="*", default=[], help="extra atom-run prefixes to include in the selection")
a = ap.parse_args()

sw = json.loads(Path(a.sweep).read_text())
d = sw["heights"][f"{a.height:.1f}" if f"{a.height:.1f}" in sw["heights"] else str(a.height)]
P = sw["args"]["period"]; N = sw["args"]["grid"]
R, T, ph = np.array(d["R"]), np.array(d["T"]), np.array(d["phase_deg"]) % 360
x = (torch.arange(N) + 0.5) * P / N - P / 2
X, Y = torch.meshgrid(x, x, indexing="ij")
SCALE = {"clover": lambda s: {"lobe_nm": 0.5 * s, "offset_nm": 0.55 * s},
         "cross": lambda s: {"arm_nm": 0.45 * s, "length_nm": 1.6 * s},
         "square": lambda s: {"side_nm": 1.25 * s, "angle_deg": 45},
         "ring": lambda s: {"radius_nm": 1.0 * s, "inner_nm": 0.45 * s}}


def fit(shape, area_frac):
    lo, hi = 5.0, P
    for _ in range(40):
        m = (lo + hi) / 2
        f = float(rt.INIT_SHAPES[shape](X, Y, SCALE[shape](m)).float().mean())
        lo, hi = (m, hi) if f < area_frac else (lo, m)
    return {k: round(v, 1) for k, v in SCALE[shape](m).items()}


common = (f"--set name={a.name} --set layer.thickness_nm={a.height} --set optimizer.robust_mode=geometric "
          f"--set optimizer.edge_shift_nm={a.edge_shift} --set optimizer.robust_blur_nm={2.5 * a.edge_shift} "
          f"--set optimizer.softmin_k=10")
lines = []
t = 0.0
while t < 360 - 1e-9:
    err = np.abs((ph - t + 180) % 360 - 180)
    i = int(np.argmin(np.where(T > a.tmin, err, 1e9)))
    Rt = float(R[i])
    for sh in a.shapes:
        init = {"type": "disk", "radius_nm": Rt} if sh == "disk" else {"type": sh, **fit(sh, math.pi * Rt * Rt / P ** 2)}
        init["noise"] = 0.1
        lines.append(f"--config {a.config} {common} --set objective.target_phase_deg={t:g} "
                     f"--set init_shape={json.dumps(init, separators=(',', ':'))} --tag p{int(round(t)):03d}_{sh}")
    t += a.step
lib = f"lut_{a.name}"
lines.append(f"select_atoms.py --prefixes {' '.join([a.name] + a.prefixes)} --out {lib}")
for n in a.supercells:
    lines.append(f"supercell.py --name sc{n}_{a.name} --library {lib} --levels {n} --order {max(40, 7 * n)} 6 --shifts -10 0 10")
print("\n".join(lines))
