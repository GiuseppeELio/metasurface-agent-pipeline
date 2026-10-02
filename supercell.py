#!/usr/bin/env python3
"""Stage C: assemble meta-atoms (library) into a supercell and simulate it rigorously with RCWA.

  python supercell.py --name sc8_v1 --atoms <prefix_of_atom_runs> [--order 40 6] [--shifts -10 0 10]
Atom runs are sorted by objective.target_phase_deg (0,45,...). For each: rho_binary.npy + t00 from summary.
Outputs runs/<name>/supercell.json: +1-order efficiency (x, y pol, all propagating T/R orders),
library (phase-sampling) prediction, and robustness to exact edge shifts.
"""
import argparse, cmath, json, math, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "third_party" / "torcwa"))
from ptopt.device import describe, pick_device  # noqa: E402
from ptopt.morph import shift_edges  # noqa: E402
from ptopt.problem import Problem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--atoms", help="prefix of the atom run directories in runs/")
ap.add_argument("--library", help="runs/<name>.json written by select_atoms.py (explicit list of atom runs)")
ap.add_argument("--order", type=int, nargs=2, default=[40, 6])
ap.add_argument("--shifts", type=float, nargs="+", default=[-10, 0, 10])
ap.add_argument("--levels", type=int, help="build an N-atom supercell picking, for each k*360/N, the library atom with the nearest phase")
ap.add_argument("--reverse", action="store_true", help="phase decreasing along x (deflect to -1 instead of +1)")
ap.add_argument("--device", default="auto")
a = ap.parse_args()
dev = pick_device(a.device)
if dev.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
if a.library:
    lp = Path(a.library) if Path(a.library).exists() else HERE / "runs" / f"{a.library}.json"
    runs = [HERE / "runs" / n for n in json.loads(lp.read_text())["atoms"]]
else:
    runs = [d for d in (HERE / "runs").iterdir() if d.is_dir() and d.name.startswith(a.atoms) and (d / "summary.json").exists()]
atoms = []
for d in runs:
    cfg = json.loads((d / "config.json").read_text()); s = json.loads((d / "summary.json").read_text())
    det = s["details_binary"][0]
    atoms.append({"dir": d.name, "target": cfg["objective"]["target_phase_deg"], "cfg": cfg,
                  "rho": np.load(d / "rho_binary.npy").astype(np.float32),
                  "T": det["T_x"], "phase": det["phase_rel_deg_x"]})
atoms.sort(key=lambda z: z["target"])
if a.levels:
    def dist(p, q):
        return abs((p - q + 180) % 360 - 180)
    pick, base = [], atoms[0]["phase"]
    for k in range(a.levels):
        want = (base + (-1 if a.reverse else 1) * k * 360.0 / a.levels) % 360
        pick.append(min(atoms, key=lambda z: dist(z["phase"], want)))
    atoms = pick
    print("[supercell] levels", a.levels, "->", [(round(z['phase']), z['dir'][:28]) for z in atoms], flush=True)
N = len(atoms)
base = atoms[0]["cfg"]
P = base["period_nm"]
print(f"[supercell] {describe(dev)} {N} atoms, P={P}, targets={[z['target'] for z in atoms]}", flush=True)

# library prediction (independent-atom / local periodic approximation): +1 order amplitude = mean t_k e^{-i 2pi k/N}
tk = [math.sqrt(z["T"]) * cmath.exp(1j * math.radians(z["phase"])) for z in atoms]
sgn = -1 if a.reverse else 1
amp = sum(t * cmath.exp(-1j * 2 * math.pi * sgn * k / N) for k, t in enumerate(tk)) / N
lib_eta = abs(amp) ** 2

cfg = json.loads(json.dumps(base))
cfg["name"] = a.name
cfg["period_nm"] = [P[0] * N, P[1]]
cfg["grid"] = [base["grid"][0] * N, base["grid"][1]]
cfg["order"] = list(a.order)
cfg["objective"] = {"type": "diffraction_efficiency", "port": "transmission", "orders": [[-1 if a.reverse else 1, 0]],
                    "input_pols": ["x", "y"]}
prob = Problem(cfg, dev)
rho_sc = np.concatenate([z["rho"] for z in atoms], axis=0)
out = HERE / "runs" / a.name; out.mkdir(parents=True, exist_ok=True)
np.save(out / "rho_supercell.npy", rho_sc.astype(np.uint8))
dx, dy = cfg["period_nm"][0] / cfg["grid"][0], cfg["period_nm"][1] / cfg["grid"][1]
po = prob.propagating_orders(0, "out")
res = {"atoms": [{k: z[k] for k in ("dir", "target", "T", "phase")} for z in atoms],
       "library_prediction_eta_plus1": lib_eta,
       "deflection_angle_deg": [ang for (o, ang) in po if o == (1, 0)],
       "propagating_T_orders": [list(o) for o, _ in po], "order_sign": -1 if a.reverse else 1, "shifts": {}}
for sh in a.shifts:
    b = shift_edges(rho_sc.astype(bool), sh, dx, dy)
    rho = torch.as_tensor(b, dtype=prob.geo_dtype, device=dev)
    with torch.no_grad():
        f, det = prob.fom(rho, return_details=True)
        oa = prob.order_analysis(rho) if sh == 0 else None
    res["shifts"][f"{sh:+.0f}nm"] = {"eta_plus1_mean": float(f), "per_pol": det[0]}
    if oa is not None:
        res["order_analysis"] = oa
    print(f"shift {sh:+.0f} nm: eta(+1) = {float(f):.3f}  {det[0]}", flush=True)
(out / "supercell.json").write_text(json.dumps(res, indent=2))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.imshow(rho_sc.T, origin="lower", extent=[0, cfg["period_nm"][0], 0, P[1]], cmap="gray_r")
    ax.set(xlabel="x (nm)", ylabel="y (nm)", title=f"{a.name}: eta(+1)={res['shifts'].get('+0nm',{}).get('eta_plus1_mean',float('nan')):.3f} (library {lib_eta:.3f})")
    fig.tight_layout(); fig.savefig(out / "supercell.png", dpi=150)
except ImportError:
    pass
print(f"[supercell] library prediction {lib_eta:.3f}; DONE {out}", flush=True)
