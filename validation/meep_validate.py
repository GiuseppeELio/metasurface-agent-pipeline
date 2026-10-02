#!/usr/bin/env python3
"""Independent FDTD validation (MIT Meep) of a binary RCWA design.

Computes, at a single wavelength and normal incidence from the input medium, the diffraction efficiency
of every propagating transmitted order (x- and y-polarized input), plus total transmitted flux, and
compares with the RCWA (TORCWA) result stored in the run directory.

  mpirun -np 48 python validation/meep_validate.py --run runs/<run_dir> --resolution 50
  mpirun -np 48 python validation/meep_validate.py --run runs/sc12_lut --resolution 50   # library supercell

Design: runs/<dir>/rho_binary.npy (or rho_supercell.npy), indexed [x, y]. Geometry/materials are read from
config.json (for supercell dirs without config.json, from the first atom's config, period = N_atoms * P).
Output: runs/<dir>/meep_res<R>.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import meep as mp

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--resolution", type=float, default=50.0, help="pixels per micron")
ap.add_argument("--pols", nargs="+", default=["x", "y"])
ap.add_argument("--dpml", type=float, default=1.0, help="PML thickness (um)")
ap.add_argument("--dsub", type=float, default=1.0, help="substrate thickness between PML and layer (um)")
ap.add_argument("--dair", type=float, default=1.5, help="air gap above layer (um)")
ap.add_argument("--decay", type=float, default=1e-7, help="DFT convergence tolerance")
ap.add_argument("--courant", type=float, default=0.35,
                help="Courant factor (Meep default 0.5 can go unstable with anisotropic subpixel averaging of high-index grids)")
ap.add_argument("--no-smoothing", action="store_true", help="switch off Meep's subpixel averaging (eps_averaging)")
ap.add_argument("--grid-averaging", action="store_true",
                help="MaterialGrid do_averaging=True (gradient-based smoothing; produced NaN fields for binary designs at some "
                     "resolutions, hence off by default: the bilinear interpolation of the binary grid is used)")
ap.add_argument("--fields", action="store_true", help="also save xz-plane (y = 0) DFT fields to fields_meep_res<R>.npz")
ap.add_argument("--shift-nm", type=float, default=0.0,
                help="exact edge shift of the binary design before simulating (+ dilates solid); output meep_s<+d>_res<R>.json")
a = ap.parse_args()

run = Path(a.run)
if (run / "config.json").exists():
    cfg = json.loads((run / "config.json").read_text())
    nat = 1
else:  # supercell assembled by supercell.py
    sc = json.loads((run / "supercell.json").read_text())
    cfg = json.loads((run.parent / sc["atoms"][0]["dir"] / "config.json").read_text())
    nat = len(sc["atoms"])
rho_file = run / ("rho_binary.npy" if (run / "rho_binary.npy").exists() else "rho_supercell.npy")
rho = np.load(rho_file).astype(np.float64)
if a.shift_nm:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ptopt.morph import shift_edges
    rho = shift_edges(rho > 0.5, a.shift_nm, cfg["period_nm"][0] * nat / rho.shape[0],
                      cfg["period_nm"][1] / rho.shape[1]).astype(np.float64)
    assert not a.fields, "--fields is only supported for the nominal design"

lam = float(cfg.get("wavelengths_nm", [cfg.get("wavelength_nm")])[0]) / 1000.0
sx, sy = cfg["period_nm"][0] * nat / 1000.0, cfg["period_nm"][1] / 1000.0
h = cfg["layer"]["thickness_nm"] / 1000.0
n_core = complex(cfg["layer"]["core"].get("n", 1.0)).real
n_bg = complex(cfg["layer"].get("background", {"n": 1.0}).get("n", 1.0)).real
n_sub = cfg.get("substrate", {"n": 1.0})["n"]
n_sup = cfg.get("superstrate", {"n": 1.0})["n"]
if cfg.get("illumination", "from_substrate") == "from_superstrate":
    n_sub, n_sup = n_sup, n_sub          # input medium is always "below" (z < 0) here
assert rho.shape[0] > 1 and rho.shape[1] > 1

fcen = 1.0 / lam
sz = 2 * a.dpml + a.dsub + h + a.dair
z0 = -0.5 * sz
z_layer_bot = z0 + a.dpml + a.dsub
z_src = z0 + a.dpml + 0.3 * a.dsub
z_mon = z_layer_bot + h + 0.6 * a.dair
cell = mp.Vector3(sx, sy, sz)
pml = [mp.PML(a.dpml, direction=mp.Z)]
k0 = mp.Vector3()  # normal incidence, periodic (Bloch k = 0)

sub_med = mp.Medium(index=n_sub)
sup_med = mp.Medium(index=n_sup)
core_med = mp.Medium(index=n_core)
bg_med = mp.Medium(index=n_bg)


def geometry(with_layer=True, homogeneous_input=False):
    if homogeneous_input:  # reference: whole cell filled with the input medium
        return [mp.Block(size=mp.Vector3(mp.inf, mp.inf, mp.inf), material=sub_med)]
    g = [mp.Block(center=mp.Vector3(0, 0, 0.5 * (z0 + z_layer_bot)), size=mp.Vector3(mp.inf, mp.inf, z_layer_bot - z0),
                  material=sub_med),
         mp.Block(center=mp.Vector3(0, 0, 0.5 * (z_layer_bot + -z0)), size=mp.Vector3(mp.inf, mp.inf, -z0 - z_layer_bot),
                  material=sup_med)]
    if with_layer:
        grid = mp.MaterialGrid(mp.Vector3(rho.shape[0], rho.shape[1], 1), bg_med, core_med,
                               weights=rho.reshape(rho.shape[0], rho.shape[1], 1), do_averaging=a.grid_averaging, beta=0)
        g.append(mp.Block(center=mp.Vector3(0, 0, z_layer_bot + 0.5 * h), size=mp.Vector3(sx, sy, h), material=grid))
    return g


def simulate(pol, kind):
    comp = mp.Ex if pol == "x" else mp.Ey
    src = [mp.Source(mp.GaussianSource(fcen, fwidth=0.1 * fcen), component=comp,
                     center=mp.Vector3(0, 0, z_src), size=mp.Vector3(sx, sy, 0))]
    sim = mp.Simulation(resolution=a.resolution, cell_size=cell, boundary_layers=pml, k_point=k0,
                        sources=src, geometry=geometry(kind == "device", kind == "reference"),
                        eps_averaging=not a.no_smoothing, default_material=mp.air, Courant=a.courant)
    z_m = z_mon if kind != "reference" else z_mon
    fl = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, z_m), size=mp.Vector3(sx, sy, 0)))
    dft = None
    if a.fields:
        vol = mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(sx, 0, sz - 2 * a.dpml))
        dft = (sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], fcen, 0, 1, where=vol), vol)
    sim.run(until_after_sources=mp.stop_when_dft_decayed(a.decay))
    return sim, fl, dft


def propagating_orders(n_out):
    out = []
    mmax, nmax = int(sx * n_out / lam) + 1, int(sy * n_out / lam) + 1
    for m in range(-mmax, mmax + 1):
        for n in range(-nmax, nmax + 1):
            kt = math.hypot(m / sx, n / sy) * lam
            if kt < n_out:
                out.append((m, n, math.degrees(math.asin(kt / n_out))))
    return out


t0 = time.time()
res = {"run": str(run), "design": rho_file.name, "shift_nm": a.shift_nm, "resolution": a.resolution, "wavelength_um": lam,
       "period_um": [sx, sy], "thickness_um": h, "n_in": n_sub, "n_out": n_sup, "pols": {}}
orders = propagating_orders(n_sup)
fields = {"h_um": h, "period_um": np.array([sx, sy]), "wavelength_um": lam, "n_in": n_sub, "n_out": n_sup,
          "resolution": a.resolution}
for pol in a.pols:
    sim_ref, fl_ref, dft_ref = simulate(pol, "reference")      # incident power (homogeneous input medium)
    P_inc = mp.get_fluxes(fl_ref)[0]
    if a.fields:  # incident complex amplitude at the input interface (z = layer bottom), for normalization
        c_in = mp.Ex if pol == "x" else mp.Ey
        ref_line = np.squeeze(sim_ref.get_dft_array(dft_ref[0], c_in, 0))
        zs = np.linspace(-0.5 * (sz - 2 * a.dpml), 0.5 * (sz - 2 * a.dpml), ref_line.shape[-1])
        E0 = complex(np.mean(ref_line[:, int(np.argmin(np.abs(zs - z_layer_bot)))]))
    sim_ref.reset_meep()
    sim, fl, dft = simulate(pol, "device")
    if a.fields:
        for c, nm in ((mp.Ex, "Ex"), (mp.Ey, "Ey"), (mp.Ez, "Ez")):
            arr = np.squeeze(sim.get_dft_array(dft[0], c, 0)) / E0
            fields[f"{nm}_{pol}"] = arr
        nxp, nzp = fields[f"Ex_{pol}"].shape
        fields["x_um"] = np.linspace(0, sx, nxp)                       # shifted: 0 .. Lx as in RCWA
        fields["z_um"] = np.linspace(z0 + a.dpml, -z0 - a.dpml, nzp) - z_layer_bot  # 0 = layer bottom
    T_flux = mp.get_fluxes(fl)[0] / P_inc
    eff = {}
    for (m, n, ang) in orders:
        p = 0.0
        for s_amp, p_amp in ((1, 0), (0, 1)):
            dp = mp.DiffractedPlanewave((m, n, 0), mp.Vector3(0, 1, 0) if n == 0 else mp.Vector3(1, 0, 0), s_amp, p_amp)
            c = sim.get_eigenmode_coefficients(fl, dp).alpha
            p += abs(c[0, 0, 0]) ** 2
        eff[f"{m},{n}"] = {"eff": p / P_inc, "angle_deg": round(ang, 2)}
    res["pols"][pol] = {"T_total_flux": T_flux, "T_orders_sum": sum(v["eff"] for v in eff.values()), "orders": eff}
    if mp.am_master():
        top = sorted(eff.items(), key=lambda kv: -kv[1]["eff"])[:4]
        print(f"[meep] pol {pol}: T_flux={T_flux:.4f}  " + "  ".join(f"({k}) {v['eff']:.4f}" for k, v in top), flush=True)
    sim.reset_meep()

# RCWA reference from the same directory, if available
ref = {}
try:
    if (run / "summary.json").exists():
        s = json.loads((run / "summary.json").read_text())
        oa = s["order_analysis_binary"][0]["pols"]
    else:
        oa = json.loads((run / "supercell.json").read_text())["order_analysis"][0]["pols"]
    for pol in a.pols:
        ref[pol] = {k: v["eff"] for k, v in oa[pol]["T"].items()}
        ref[pol]["T_total"] = oa[pol]["T_total"]
except Exception as e:  # noqa: BLE001
    ref = {"error": str(e)}
res["rcwa"] = ref
res["wall_time_s"] = round(time.time() - t0, 1)
if mp.am_master():
    stem = f"meep_s{a.shift_nm:+.0f}_" if a.shift_nm else "meep_"
    out = run / f"{stem}res{int(a.resolution)}.json"
    out.write_text(json.dumps(res, indent=2))
    if a.fields:
        np.savez_compressed(run / f"fields_meep_res{int(a.resolution)}.npz", **fields)
    print("[meep] DONE", out, flush=True)
