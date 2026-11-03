#!/usr/bin/env python3
"""Full-wave 3-D FDTD check (MIT Meep) of a metalens built by metalens.py.

The whole aperture is simulated: substrate, patterned layer and a short output region. The focal fields come from
Meep's near-to-far transformation on a plane just above the layer, so the domain stays thin even for focal
lengths of tens of microns. x-polarized illumination from the input medium, uniform over the circular aperture
(a d4 library on a square lens grid makes the y-polarized result the 90-degree rotation of this one).

  mpirun -np 48 python validation/meep_lens.py --run runs/<lens> --resolution 30
  mpirun -np 48 python validation/meep_lens.py --run runs/<lens> --resolution 15 --crop-cells 9   # smoke test

Normalization (--ref full, default): the incident power is the flux of the same aperture source in the same cell
filled with the input medium (a second, shorter run without resonances).
Outputs runs/<lens>/meep_lens_res<R>.json: total transmission, focal position, FWHM (x, y), focusing efficiency
(power through a disk of radius 3 x FWHM in the focal plane / incident power), and the LPA prediction from
lens.json for comparison; meep_lens_res<R>.npz: axial and focal-plane |E|^2; meep_lens_plane_res<R>.npz: the
transverse field just above the lens, which validation/lens_post.py propagates with the same vector angular-spectrum
code as the model (longitudinal sections, depth of focus, FWHM).
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
ap.add_argument("--resolution", type=float, default=30.0, help="pixels per micron")
ap.add_argument("--crop-cells", type=int, help="simulate only the central N x N cells (odd N), e.g. for a smoke test")
ap.add_argument("--margin", type=float, default=1.0, help="lateral gap between aperture and PML (um)")
ap.add_argument("--dpml", type=float, default=1.0)
ap.add_argument("--dsub", type=float, default=0.6, help="input medium between PML and layer (um)")
ap.add_argument("--dair", type=float, default=0.8, help="output medium between layer and PML (um)")
ap.add_argument("--decay", type=float, default=1e-6)
ap.add_argument("--courant", type=float, default=0.35)
ap.add_argument("--grid-px-um", type=float, default=0, help="MaterialGrid sampling (default 2 x resolution)")
ap.add_argument("--far-res", type=float, default=12.0, help="pixels per micron of the focal-plane map")
ap.add_argument("--ref", choices=["full", "periodic"], default="full",
                help="incident power: 'full' = same aperture source in a cell filled with the input medium (exact, "
                     "costs a second, shorter run); 'periodic' = power density of a small periodic cell x aperture area "
                     "(cheap; the pixelized aperture edge biases it by ~2/(R*resolution))")
a = ap.parse_args()

run = Path(a.run)
L = json.loads((run / "lens.json").read_text())
z_ = np.load(run / "rho_lens.npz")
lay = np.unpackbits(z_["packed"])[:int(np.prod(z_["shape"]))].reshape(z_["shape"]).astype(np.float32)
P, Nc, npx = L["period_um"], L["cells"], L["atom_grid_px"]
if a.crop_cells:
    nc = a.crop_cells + (a.crop_cells + 1) % 2
    c0 = (Nc - nc) // 2
    lay = lay[c0 * npx:(c0 + nc) * npx, c0 * npx:(c0 + nc) * npx]
    Nc = nc
D = Nc * P; R = min(D, L["diameter_um"]) / 2

def _index(spec, lam_nm):
    """Real index of a material spec at the design wavelength: constant n, a materials/*.txt table or a
    refractiveindex.info entry ("rii:..."), resolved with the same code as the optimizer (k is ignored here)."""
    spec = spec or {"n": 1.0}
    if "material" in spec:
        import sys as _s
        _s.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from ptopt.materials import refractive_index
        return float(refractive_index(spec, lam_nm).real)
    return complex(spec.get("n", 1.0)).real

lam = L["wavelength_um"]; f = L["focal_um"]
h = L["layer"]["thickness_nm"] / 1000.0
n_core = _index(L["layer"]["core"], lam * 1000.0)
n_bg = _index(L["layer"].get("background", {"n": 1.0}), lam * 1000.0)
n_in = _index(L["substrate"], lam * 1000.0)
n_out = _index(L["superstrate"], lam * 1000.0)
if L.get("illumination") == "from_superstrate":
    n_in, n_out = n_out, n_in
# downsample the 5-nm layout to the MaterialGrid sampling (area average -> gray fill at the edges)
gpx = a.grid_px_um or 2 * a.resolution
fac = max(1, int(round(npx / P / gpx)))
nn = (lay.shape[0] // fac) * fac
w = lay[:nn, :nn].reshape(nn // fac, fac, nn // fac, fac).mean(axis=(1, 3))

fcen = 1.0 / lam
sxy = D + 2 * a.margin + 2 * a.dpml
sz = 2 * a.dpml + a.dsub + h + a.dair
z0 = -0.5 * sz
z_bot = z0 + a.dpml + a.dsub
z_top = z_bot + h
z_src = z0 + a.dpml + 0.3 * a.dsub
z_mon = z_top + 0.5 * a.dair
sub, sup = mp.Medium(index=n_in), mp.Medium(index=n_out)
core, bg = mp.Medium(index=n_core), mp.Medium(index=n_bg)


def aperture(p):
    return 1.0 if math.hypot(p.x, p.y) <= R else 0.0


# ---- incident power ---------------------------------------------------------------------------------------------
t0 = time.time()
src = [mp.Source(mp.GaussianSource(fcen, fwidth=0.2 * fcen), component=mp.Ex, center=mp.Vector3(0, 0, z_src),
                 size=mp.Vector3(2 * R, 2 * R, 0), amp_func=aperture)]
inner = sxy - 2 * a.dpml
if a.ref == "full":
    ref = mp.Simulation(resolution=a.resolution, cell_size=mp.Vector3(sxy, sxy, sz), boundary_layers=[mp.PML(a.dpml)],
                        sources=src, default_material=sub, Courant=a.courant)
    fr = ref.add_flux(fcen, 0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, z_bot), size=mp.Vector3(inner, inner, 0)))
    ref.run(until_after_sources=mp.stop_when_dft_decayed(a.decay))
    P_inc = mp.get_fluxes(fr)[0]
else:
    cs = 0.5
    ref = mp.Simulation(resolution=a.resolution, cell_size=mp.Vector3(cs, cs, sz), boundary_layers=[mp.PML(a.dpml, direction=mp.Z)],
                        k_point=mp.Vector3(), default_material=sub, Courant=a.courant,
                        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=0.2 * fcen), component=mp.Ex,
                                           center=mp.Vector3(0, 0, z_src), size=mp.Vector3(cs, cs, 0))])
    fr = ref.add_flux(fcen, 0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, z_bot), size=mp.Vector3(cs, cs, 0)))
    ref.run(until_after_sources=mp.stop_when_dft_decayed(a.decay))
    g = np.arange(-R, R + 1e-9, 1.0 / a.resolution)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    P_inc = mp.get_fluxes(fr)[0] / (cs * cs) * float((np.hypot(GX, GY) <= R).sum()) / a.resolution ** 2
ref.reset_meep()
t_ref = time.time() - t0

# ---- device ---------------------------------------------------------------------------------------------------
grid = mp.MaterialGrid(mp.Vector3(w.shape[0], w.shape[1], 1), bg, core, weights=w[:, :, None], do_averaging=False, beta=0)
geom = [mp.Block(center=mp.Vector3(0, 0, 0.5 * (z0 + z_bot)), size=mp.Vector3(mp.inf, mp.inf, z_bot - z0), material=sub),
        mp.Block(center=mp.Vector3(0, 0, z_bot + 0.5 * h), size=mp.Vector3(mp.inf, mp.inf, h), material=bg),
        mp.Block(center=mp.Vector3(0, 0, z_bot + 0.5 * h), size=mp.Vector3(D, D, h), material=grid)]
sim = mp.Simulation(resolution=a.resolution, cell_size=mp.Vector3(sxy, sxy, sz), boundary_layers=[mp.PML(a.dpml)],
                    geometry=geom, sources=src, default_material=sup, Courant=a.courant, eps_averaging=True)
ft = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, z_mon), size=mp.Vector3(inner, inner, 0)))
n2f = sim.add_near2far(fcen, 0, 1, mp.Near2FarRegion(center=mp.Vector3(0, 0, z_mon), size=mp.Vector3(inner, inner, 0)))
pl = sim.add_dft_fields([mp.Ex, mp.Ey], fcen, 0, 1, center=mp.Vector3(0, 0, z_mon), size=mp.Vector3(inner, inner, 0))
sim.run(until_after_sources=mp.stop_when_dft_decayed(a.decay))
T_total = mp.get_fluxes(ft)[0] / P_inc
plane = {c: np.squeeze(sim.get_dft_array(pl, comp, 0)) for c, comp in (("Ex", mp.Ex), ("Ey", mp.Ey))}
t_fdtd = time.time() - t0

# axial profile (distances measured from the top of the layer)
zs = np.linspace(0.3 * f, 1.7 * f, 141)
axI = []
for z in zs:
    ff = sim.get_farfield(n2f, mp.Vector3(0, 0, z_top + z))
    axI.append(sum(abs(ff[i]) ** 2 for i in range(3)))
axI = np.array(axI)
zf = float(zs[int(np.argmax(axI))])
fw0 = 0.514 * lam / (n_out * R / math.hypot(R, zf))
wnd = 8 * fw0
ff = sim.get_farfields(n2f, a.far_res, center=mp.Vector3(0, 0, z_top + zf), size=mp.Vector3(wnd, wnd, 0))
Ex, Ey, Ez = (np.squeeze(ff[c]) for c in ("Ex", "Ey", "Ez"))
Hx, Hy = np.squeeze(ff["Hx"]), np.squeeze(ff["Hy"])
I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
Sz = np.real(np.conj(Ex) * Hy - np.conj(Ey) * Hx)        # same convention as Meep's flux
xs = np.linspace(-wnd / 2, wnd / 2, I.shape[0])
c0 = I.shape[0] // 2


def fwhm(line, dx):
    m = line.max(); idx = np.where(line >= m / 2)[0]
    return (idx[-1] - idx[0] + 1) * dx if idx.size else float("nan")


dxf = xs[1] - xs[0]
fx, fy = fwhm(I[:, c0], dxf), fwhm(I[c0, :], dxf)
XX, YY = np.meshgrid(xs, xs, indexing="ij")
disk = np.hypot(XX, YY) <= 3 * 0.5 * (fx + fy)
eff = float(Sz[disk].sum() * dxf * dxf / P_inc)

res = {"run": run.name, "resolution": a.resolution, "crop_cells": a.crop_cells, "cells": Nc, "aperture_diameter_um": 2 * R,
       "T_total": T_total, "z_focus_um": zf, "fwhm_x_um": fx, "fwhm_y_um": fy, "focusing_efficiency": eff,
       "diffraction_limit_fwhm_at_focus_um": fw0, "materialgrid_px": list(w.shape),
       "wall_time_s": round(time.time() - t0, 1), "fdtd_time_s": round(t_fdtd, 1), "ref": a.ref, "ref_time_s": round(t_ref, 1),
       "lpa_model": None if a.crop_cells else L["lpa_model"]["lens"]}
if mp.am_master():
    tag = f"_crop{Nc}" if a.crop_cells else ""
    (run / f"meep_lens{tag}_res{int(a.resolution)}.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(run / f"meep_lens{tag}_res{int(a.resolution)}.npz", z_um=zs, axial_I=axI, x_um=xs, focal_I=I, focal_Sz=Sz)
    # transverse field on the monitor plane, for validation/lens_post.py (same propagator as the LPA model)
    np.savez_compressed(run / f"meep_lens{tag}_plane_res{int(a.resolution)}.npz", Ex=plane["Ex"], Ey=plane["Ey"],
                        width_um=inner, z_above_layer_um=z_mon - z_top, T_total=T_total, wavelength_um=lam, n_out=n_out)
    print(f"[meep_lens] {run.name} res {a.resolution:g}: T={T_total:.3f} focus z={zf:.2f} um FWHM x/y={fx:.2f}/{fy:.2f} um "
          f"focusing eff={eff:.3f}  ({res['wall_time_s']} s)", flush=True)
    if res["lpa_model"]:
        m = res["lpa_model"]
        print(f"[meep_lens] LPA model: T={m['transmitted_fraction']:.3f} focus z={m['z_focus_um']:.2f} um "
              f"FWHM x/y={m['fwhm_x_um']:.2f}/{m['fwhm_y_um']:.2f} um focusing eff={m['focusing_efficiency']:.3f}", flush=True)
