#!/usr/bin/env python3
"""Pick the best meta-atom for each target phase among several runs (different initial shapes / seeds).

Score = worst-case FoM over the stored edge shifts (robustness_binary), FoM = mean_pol Re(t e^{-i phi_target})
(= |t| cos(phase error), x and y together). Writes runs/<out>.json = ordered list of run dirs for supercell.py --library.
  python select_atoms.py --prefixes atomB_h800 atomB2_h800 --out lib_h800_best [--shift-window 10]
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--prefixes", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shift-window", type=float, default=10.0, help="only shifts with |d| <= this enter the worst case")
a = ap.parse_args()
best = {}
rows = []
for d in sorted((HERE / "runs").iterdir()):
    if not (d.is_dir() and any(d.name.startswith(p) for p in a.prefixes) and (d / "summary.json").exists()):
        continue
    cfg = json.loads((d / "config.json").read_text()); s = json.loads((d / "summary.json").read_text())
    tgt = float(cfg["objective"]["target_phase_deg"])
    rb = s.get("robustness_binary", {})
    vals = [v["fom_binary"] for k, v in rb.items() if abs(float(k.replace("nm", ""))) <= a.shift_window] or [s["fom_binary"]]
    det = s["details_binary"][0]
    r = {"dir": d.name, "target": tgt, "shape": cfg.get("init_shape", {}).get("type", "random"),
         "nominal": s["fom_binary"], "worst": min(vals), "T": det["T_x"], "phase_err": det["phase_err_deg_x"],
         "pol_diff_deg": abs(det["phase_err_deg_x"] - det.get("phase_err_deg_y", det["phase_err_deg_x"]))}
    rows.append(r)
    if tgt not in best or r["worst"] > best[tgt]["worst"]:
        best[tgt] = r
for r in sorted(rows, key=lambda z: (z["target"], -z["worst"])):
    mark = "*" if best[r["target"]]["dir"] == r["dir"] else " "
    print(f"{mark} {r['target']:5.0f}  {r['shape']:8s} nom {r['nominal']:.3f} worst±{a.shift_window:.0f} {r['worst']:.3f}  T {r['T']:.3f}  dphi {r['phase_err']:+.1f}  {r['dir']}")
lib = [best[t]["dir"] for t in sorted(best)]
(HERE / "runs" / f"{a.out}.json").write_text(json.dumps({"atoms": lib, "selection": [best[t] for t in sorted(best)]}, indent=2))
print("[select_atoms] wrote", HERE / "runs" / f"{a.out}.json")
