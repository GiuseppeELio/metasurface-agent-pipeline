#!/usr/bin/env python3
"""Post-hoc fabrication-robustness evaluation of finished runs (binary designs).

For each run dir whose name starts with one of the given prefixes: apply exact edge shifts
(erosion/dilation, nm) to rho_binary.npy, recompute FoM (at verify_order if set), write robustness_eval.json.
  python evaluate.py si1550_defl_v2 si1550_defl_v3 --shifts -15 -10 -5 0 5 10 15
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "third_party" / "torcwa"))
from ptopt.device import pick_device  # noqa: E402
from ptopt.morph import feature_report, shift_edges  # noqa: E402
from ptopt.problem import Problem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prefixes", nargs="+")
ap.add_argument("--shifts", type=float, nargs="+", default=[-15, -10, -5, 0, 5, 10, 15])
ap.add_argument("--device", default="auto")
ap.add_argument("--rundir", default=str(HERE / "runs"))
a = ap.parse_args()
dev = pick_device(a.device)
if dev.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False

runs = sorted(d for d in Path(a.rundir).iterdir() if d.is_dir() and any(d.name.startswith(p) for p in a.prefixes)
              and (d / "rho_binary.npy").exists())
table = []
for d in runs:
    cfg = json.loads((d / "config.json").read_text())
    prob = Problem(cfg, dev)
    order = cfg.get("verify_order") or cfg["order"]
    b = np.load(d / "rho_binary.npy").astype(bool)
    dx, dy = prob.L[0] / prob.nx, prob.L[1] / prob.ny
    res = {"run": d.name, "order": order, "features": feature_report(b, dx, dy), "shifts": {}}
    for s in a.shifts:
        bs = shift_edges(b, s, dx, dy)
        rho = torch.as_tensor(bs, dtype=prob.geo_dtype, device=dev)
        with torch.no_grad():
            f = np.mean([float(prob.objective(prob.build_sim(rho, i, order=order), prob.ocfg(i))[0])
                         for i in range(len(prob.wavelengths))])
        res["shifts"][f"{s:+.0f}"] = {"fom": round(f, 4), "fill": round(float(bs.mean()), 4)}
    foms = [v["fom"] for v in res["shifts"].values()]
    res["worst"] = min(foms)
    (d / "robustness_eval.json").write_text(json.dumps(res, indent=2))
    print(d.name, " ".join(f"{k}:{v['fom']:.3f}" for k, v in res["shifts"].items()), f"worst {res['worst']:.3f}", flush=True)
    table.append(res)
out = Path(a.rundir) / ("robustness_" + "_".join(a.prefixes) + ".json")
out.write_text(json.dumps(table, indent=2))
print("[evaluate] wrote", out)
