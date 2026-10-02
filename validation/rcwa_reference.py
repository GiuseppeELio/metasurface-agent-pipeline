#!/usr/bin/env python3
"""RCWA reference for cross-solver validation: per-order T/R of a binary design at the verification truncation,
nominal and with exact edge shifts. This is the TORCWA side of the TORCWA <-> Meep comparison.

  python validation/rcwa_reference.py --run runs/<dir> [--order Nx Ny] [--shifts 0 -10 10] [--device auto]

Writes runs/<dir>/rcwa_ref.json:
  {"order": [Nx, Ny], "shifts": {"+0": {"x": {"T": {"m,n": eff}, "R": {...}, "T_total", "R_total"}, "y": {...}}, ...}}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "third_party" / "torcwa"))
from ptopt.device import pick_device  # noqa: E402
from ptopt.morph import shift_edges  # noqa: E402
from ptopt.problem import Problem  # noqa: E402
from ptopt.rundir import load_run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--order", type=int, nargs=2, help="Fourier truncation (default: verify_order, else order)")
ap.add_argument("--shifts", type=float, nargs="+", default=[0.0], help="edge shifts in nm (+ dilates solid)")
ap.add_argument("--out", default="rcwa_ref.json", help="file name inside the run directory (validate.py reads rcwa_ref.json)")
ap.add_argument("--device", default="auto")
a = ap.parse_args()

run = Path(a.run)
cfg, rho, kind = load_run(run)
cfg["order"] = list(a.order) if a.order else (cfg.get("verify_order") or cfg["order"])
dev = pick_device(a.device)
if dev.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
prob = Problem(cfg, dev)
dx, dy = cfg["period_nm"][0] / rho.shape[0], cfg["period_nm"][1] / rho.shape[1]
out_file = run / a.out
res = json.loads(out_file.read_text()) if out_file.exists() else {}
if res.get("order") != cfg["order"]:
    res = {"order": cfg["order"], "shifts": {}}
res["kind"] = kind
t0 = time.time()
for d in a.shifts:
    b = shift_edges(rho > 0.5, d, dx, dy)
    oa = prob.order_analysis(torch.as_tensor(b, dtype=prob.geo_dtype, device=dev))[0]["pols"]
    entry = {}
    for p, v in oa.items():
        entry[p] = {"T": {k: o["eff"] for k, o in v["T"].items()}, "R": {k: o["eff"] for k, o in v["R"].items()},
                    "T_total": v["T_total"], "R_total": v["R_total"]}
    res["shifts"][f"{d:+.0f}"] = entry
    print(f"[rcwa_reference] shift {d:+.0f} nm, order {cfg['order']}: " +
          "  ".join(f"{p}: T={e['T_total']:.4f} R={e['R_total']:.4f}" for p, e in entry.items()), flush=True)
res["wall_time_s"] = round(time.time() - t0, 1)
out_file.write_text(json.dumps(res, indent=2))
print("[rcwa_reference] wrote", out_file)
