#!/usr/bin/env python3
"""Near fields of a binary design computed with TORCWA (RCWA), xz cross-section at y = Ly/2.

Writes runs/<dir>/fields_rcwa.npz with, for each input polarization p in {x, y}:
  x_um, z_um, E{x,y,z}_<p>  (complex, unit-amplitude plane wave incident from the input medium)
z = 0 is the bottom of the patterned layer (input side), z = h its top. Same convention as fields_meep.npz.

  python validation/torcwa_fields.py --run runs/<dir> [--order Nx Ny] [--nx 400] [--nz 300] [--zin 1.0 --zout 3.0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "third_party" / "torcwa"))
from ptopt.device import pick_device  # noqa: E402
from ptopt.problem import Problem  # noqa: E402
from ptopt.rundir import load_run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--order", type=int, nargs=2, help="Fourier truncation (default: verify_order or order of the run)")
ap.add_argument("--nx", type=int, default=400)
ap.add_argument("--nz", type=int, default=300)
ap.add_argument("--zin", type=float, default=1.0, help="um shown below the layer")
ap.add_argument("--zout", type=float, default=3.0, help="um shown above the layer")
ap.add_argument("--device", default="auto")
a = ap.parse_args()

run = Path(a.run)
cfg, rho, _ = load_run(run)
cfg["order"] = list(a.order) if a.order else (cfg.get("verify_order") or cfg["order"])
dev = pick_device(a.device)
if dev.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
prob = Problem(cfg, dev)
Lx, Ly = prob.L
h = prob.thickness
sim = prob.build_sim(torch.as_tensor(rho > 0.5, dtype=prob.geo_dtype, device=dev), 0)
x = torch.linspace(0, Lx, a.nx, dtype=prob.geo_dtype, device=dev)
z = torch.linspace(-a.zin * 1000, h + a.zout * 1000, a.nz, dtype=prob.geo_dtype, device=dev)
out = {"x_um": x.cpu().numpy() / 1000, "z_um": z.cpu().numpy() / 1000, "h_um": h / 1000,
       "period_um": np.array([Lx, Ly]) / 1000, "wavelength_um": prob.wavelengths[0] / 1000,
       "n_in": complex(prob.eps_in[0]) ** 0.5, "n_out": complex(prob.eps_out[0]) ** 0.5, "order": np.array(cfg["order"])}
for p, amp in (("x", [1.0, 0.0]), ("y", [0.0, 1.0])):
    sim.source_planewave(amplitude=amp, direction="forward")
    with torch.no_grad():
        (Ex, Ey, Ez), _ = sim.field_xz(x, z, Ly / 2)
    for c, f in (("Ex", Ex), ("Ey", Ey), ("Ez", Ez)):
        out[f"{c}_{p}"] = f.cpu().numpy()
np.savez_compressed(run / "fields_rcwa.npz", **out)
print("[torcwa_fields] wrote", run / "fields_rcwa.npz", "order", cfg["order"])
