#!/usr/bin/env python3
"""Full-diameter 3-D Meep check of a large, rotationally symmetric metalens built by tools/achro_tio2.py.

A full 3-D run of a 400 um lens is not affordable, so the simulated object is a strip through the optical axis:
the real pillars of the row y = 0 (x from -R to R, one cell per period P), with periodic boundaries in y (period P).
This is exact for the coupling between neighbouring different atoms along the radius (which the locally periodic model
neglects) and uses the true 3-D pillars, dispersive materials and the hard aperture edge. Neighbours along y are
replaced by copies of the same atom: in the real lens they differ by r -> sqrt(r^2 + P^2), negligible except within a
few cells of the axis.

Two runs per lens: x-polarized source (E along the radius: radial response t_rad) and y-polarized source (E along the
rows: azimuthal response t_az). validation/achro_strip_post.py rebuilds the 2-D aperture field of the circular lens,
    Ex = t_rad cos^2(phi) + t_az sin^2(phi),   Ey = (t_rad - t_az) cos(phi) sin(phi)   (x-polarized illumination),
and focuses it with the same vector angular-spectrum code as the design model.

Broadband: one Gaussian pulse (700-1000 nm), Lorentz fits of the tabulated n,k (materials/*.txt) for TiO2 and SiO2,
DFT monitors at --lams. Normalization: a periodic reference run filled with the input medium gives the incident field;
t(x) = E(x) / (sqrt(n_in) E_inc), so |t|^2 is a power fraction (same convention as the LUT).

  mpirun -np 48 python validation/meep_achro_strip.py --run runs/<lens> --pol x --resolution 40
  mpirun -np 48 python validation/meep_achro_strip.py --run runs/<lens> --pol x --resolution 40 --x0-um 100 --width-um 30   # smoke
Output runs/<lens>/meep_strip_<pol>_res<R>[_tag].npz and .json (timing, fits, flux).
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import meep as mp

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from ptopt.materials import refractive_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--pol", choices=["x", "y"], default="x")
ap.add_argument("--resolution", type=float, default=40.0)
ap.add_argument("--lams", type=float, nargs="+", default=[700, 750, 800, 850, 900, 950, 1000])
ap.add_argument("--x0-um", type=float, default=0.0, help="centre of a cropped strip (smoke tests)")
ap.add_argument("--width-um", type=float, default=0.0, help="cropped strip width (0 = full diameter)")
ap.add_argument("--margin", type=float, default=2.0, help="gap between aperture/crop edge and PML (um)")
ap.add_argument("--dpml", type=float, default=1.0)
ap.add_argument("--dsub", type=float, default=0.6)
ap.add_argument("--dair", type=float, default=0.8)
ap.add_argument("--zmon", type=float, default=0.4, help="monitor height above the pillars (um)")
ap.add_argument("--dft-tol", type=float, default=1e-4)
ap.add_argument("--tmax", type=float, default=400.0, help="cap on the run time after the source (Meep units, um/c)")
ap.add_argument("--courant", type=float, default=0.35, help="0.5 went unstable (NaN) once with the dispersive media + PML")
ap.add_argument("--tcheck", type=float, default=0.0, help="if > 0 and the DFT has not decayed by then, record the DFT, continue to --tmax and report the relative change")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="", help="output folder (default: the lens run folder)")
A = ap.parse_args()
mp.verbosity(1)
run = Path(A.run); L = json.load(open(run / "lens.json")); lut = np.load(run / "lut_used.npz", allow_pickle=True)
largs = json.loads(str(lut["args"]))
P = L["P_nm"] / 1000.0; H = float(largs["height"]) / 1000.0; R = L["R_um"]
spec_p, spec_s = json.loads(largs["pillar"]), json.loads(largs["substrate"])
lams = np.array(A.lams, float); freqs = 1000.0 / lams


def lorentz_fit(spec, lo=660.0, hi=1040.0):
    """eps(f) = eps_inf + sigma f0^2 / (f0^2 - f^2) (lossless, eps_inf >= 1 for stability), least squares on n."""
    lam = np.linspace(lo, hi, 60); f = 1000.0 / lam
    eps = np.array([refractive_index(spec, l).real for l in lam]) ** 2
    best = None
    for f0 in np.linspace(1.6, 30.0, 1500):
        g = f0 ** 2 / (f0 ** 2 - f ** 2); M = np.c_[np.ones_like(f), g]
        c, *_ = np.linalg.lstsq(M, eps, rcond=None)
        if c[0] < 1.0: c = np.array([1.0, np.dot(g, eps - 1) / np.dot(g, g)])
        if c[1] <= 0: continue
        err = np.max(np.abs(np.sqrt(M @ c) - np.sqrt(eps)))
        if best is None or err < best[0]: best = (err, f0, c[0], c[1])
    err, f0, einf, sig = best
    mat = mp.Medium(epsilon=einf, E_susceptibilities=[mp.LorentzianSusceptibility(frequency=f0, gamma=0.0, sigma=sig)])
    return mat, dict(eps_inf=float(einf), f0=float(f0), sigma=float(sig), max_dn=float(err))


TiO2, fit_p = lorentz_fit(spec_p); SiO2, fit_s = lorentz_fit(spec_s)
n_in = {float(l): refractive_index(spec_s, l).real for l in lams}

# ---------------- strip geometry ----------------
ncell = int(round(2 * L["R_um"] * 1000 / L["P_nm"])) | 1
xc = (np.arange(ncell) - ncell // 2) * P
r_grid = np.array(L["radial_assign"]["r_nm"]) / 1000.0; assign = np.array(L["radial_assign"]["atom"])
inside = np.abs(xc) <= R
if A.width_um > 0:
    inside &= np.abs(xc - A.x0_um) <= A.width_um / 2
cells = np.nonzero(inside)[0]
x_lo, x_hi = xc[cells[0]] - P / 2, xc[cells[-1]] + P / 2
xmid = 0.5 * (x_lo + x_hi); wsrc = x_hi - x_lo
sx = wsrc + 2 * (A.margin + A.dpml); sy = P; sz = 2 * A.dpml + A.dsub + H + A.dair
z0 = -sz / 2 + A.dpml                       # bottom of the substrate region (above the PML)
z_layer = z0 + A.dsub                        # substrate/pillar interface
z_mon = z_layer + H + A.zmon
z_src = z0 + 0.3

geom = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, (z_layer - (-sz / 2))), center=mp.Vector3(0, 0, (-sz / 2 + z_layer) / 2),
                 material=SiO2)]
atoms = lut["atoms"]; fam_names = [str(f) for f in lut["families"]]
zc = z_layer + H / 2; used = {}
for j in cells:
    x = xc[j] - xmid
    a = atoms[assign[min(np.searchsorted(r_grid, abs(xc[j])), len(r_grid) - 1)]]
    fam, p1, p2 = int(a[0]), a[1] / 1000.0, a[2] / 1000.0
    used[fam_names[fam]] = used.get(fam_names[fam], 0) + 1
    c = mp.Vector3(x, 0, zc)
    if fam == 0:
        geom.append(mp.Cylinder(radius=p1 / 2, height=H, center=c, material=TiO2))
    elif fam == 1:
        geom.append(mp.Block(size=mp.Vector3(p1, p1, H), center=c, material=TiO2))
    elif fam == 2:
        geom += [mp.Cylinder(radius=p1 / 2, height=H, center=c, material=TiO2),
                 mp.Cylinder(radius=p2 / 2, height=H, center=c, material=mp.air)]
    elif fam == 3:
        geom += [mp.Block(size=mp.Vector3(p1, p1, H), center=c, material=TiO2),
                 mp.Block(size=mp.Vector3(p2, p2, H), center=c, material=mp.air)]
    elif fam == 4:
        geom += [mp.Block(size=mp.Vector3(p1, p2, H), center=c, material=TiO2),
                 mp.Block(size=mp.Vector3(p2, p1, H), center=c, material=TiO2)]

# Inside the PML layers the substrate is non-dispersive (n at the centre wavelength): a Lorentzian medium inside a
# PML made the y-polarized runs grow slowly and end in NaN after t ~ 270 (seen at res 60 and 100).
SiO2_nd = mp.Medium(index=float(refractive_index(spec_s, float(np.mean(lams))).real))
zs_top = z_layer
geom += [mp.Block(size=mp.Vector3(A.dpml, mp.inf, zs_top + sz / 2), center=mp.Vector3(sgn * (sx / 2 - A.dpml / 2), 0, (-sz / 2 + zs_top) / 2),
                  material=SiO2_nd) for sgn in (-1, 1)]
geom += [mp.Block(size=mp.Vector3(mp.inf, mp.inf, A.dpml), center=mp.Vector3(0, 0, -sz / 2 + A.dpml / 2), material=SiO2_nd)]
fmin, fmax = freqs.min(), freqs.max(); fcen = 0.5 * (fmin + fmax); df = 1.6 * (fmax - fmin)
comp = mp.Ex if A.pol == "x" else mp.Ey
src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=comp, center=mp.Vector3(0, 0, z_src),
                 size=mp.Vector3(wsrc, sy, 0))]
pml = [mp.PML(A.dpml, direction=mp.X), mp.PML(A.dpml, direction=mp.Z)]
t0 = time.time()
sim = mp.Simulation(cell_size=mp.Vector3(sx, sy, sz), resolution=A.resolution, geometry=geom, sources=src,
                    boundary_layers=pml, k_point=mp.Vector3(), default_material=mp.air, Courant=A.courant)
mon = mp.Volume(center=mp.Vector3(0, 0, z_mon), size=mp.Vector3(sx - 2 * A.dpml, sy, 0))
dft = sim.add_dft_fields([mp.Ex, mp.Ey], freqs, where=mon)
flux = sim.add_flux(freqs, mp.FluxRegion(center=mp.Vector3(0, 0, z_mon), size=mp.Vector3(sx - 2 * A.dpml, sy, 0)))
t_first = A.tcheck if 0 < A.tcheck < A.tmax else A.tmax
sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=A.dft_tol, minimum_run_time=40, maximum_run_time=t_first))
Ex = np.array([sim.get_dft_array(dft, mp.Ex, k) for k in range(len(freqs))])
Ey = np.array([sim.get_dft_array(dft, mp.Ey, k) for k in range(len(freqs))])
dft_change = None
if t_first < A.tmax and sim.meep_time() >= t_first:     # not decayed: continue and measure how much the DFT still moves
    sim.run(until=A.tmax - t_first)
    co = mp.Ex if A.pol == "x" else mp.Ey; E1 = Ex if A.pol == "x" else Ey
    Ex2 = np.array([sim.get_dft_array(dft, mp.Ex, k) for k in range(len(freqs))])
    Ey2 = np.array([sim.get_dft_array(dft, mp.Ey, k) for k in range(len(freqs))])
    E2 = Ex2 if A.pol == "x" else Ey2
    dft_change = [float(np.linalg.norm(E2[k] - E1[k]) / max(np.linalg.norm(E2[k]), 1e-30)) for k in range(len(freqs))]
    Ex, Ey = Ex2, Ey2
t_run = time.time() - t0; t_meep = sim.meep_time()
xg, yg, zg, wg = sim.get_array_metadata(dft_cell=dft)
F_lens = np.array(mp.get_fluxes(flux))
sim.reset_meep()

# ---------------- reference: input medium everywhere, periodic plane wave ----------------
t1 = time.time()
ref = mp.Simulation(cell_size=mp.Vector3(P, P, sz), resolution=A.resolution, sources=[
    mp.Source(mp.GaussianSource(fcen, fwidth=df), component=comp, center=mp.Vector3(0, 0, z_src), size=mp.Vector3(P, P, 0))],
    boundary_layers=[mp.PML(A.dpml, direction=mp.Z)], k_point=mp.Vector3(), default_material=SiO2, Courant=A.courant,
    geometry=[mp.Block(size=mp.Vector3(mp.inf, mp.inf, A.dpml), center=mp.Vector3(0, 0, -sz / 2 + A.dpml / 2), material=SiO2_nd),
              mp.Block(size=mp.Vector3(mp.inf, mp.inf, A.dpml), center=mp.Vector3(0, 0, sz / 2 - A.dpml / 2), material=SiO2_nd)])
dref = ref.add_dft_fields([comp], freqs, where=mp.Volume(center=mp.Vector3(0, 0, z_mon), size=mp.Vector3(P, P, 0)))
fref = ref.add_flux(freqs, mp.FluxRegion(center=mp.Vector3(0, 0, z_mon), size=mp.Vector3(P, P, 0)))
ref.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-6, minimum_run_time=20, maximum_run_time=400))
Einc = np.array([np.mean(ref.get_dft_array(dref, comp, k)) for k in range(len(freqs))])
F_ref = np.array(mp.get_fluxes(fref))
t_ref = time.time() - t1

if mp.am_master():
    ny = Ex.shape[2] if Ex.ndim == 3 else 1
    exm = Ex.mean(axis=2) if Ex.ndim == 3 else Ex           # y-average = zero-order content along y
    eym = Ey.mean(axis=2) if Ey.ndim == 3 else Ey
    nn = np.array([n_in[float(l)] for l in lams])
    norm = 1.0 / (np.sqrt(nn) * Einc)
    tag = f"_{A.tag}" if A.tag else ""
    od = Path(A.out) if A.out else run; od.mkdir(parents=True, exist_ok=True)
    base = od / f"meep_strip_{A.pol}_res{A.resolution:g}{tag}"
    xs = np.asarray(xg) + xmid
    P_inc = F_ref * (wsrc / P)                               # incident power over the illuminated strip
    np.savez(str(base) + ".npz", x_um=xs, lam=lams, Ex=exm, Ey=eym, Einc=Einc, norm=norm, n_in=nn,
             t_co=(exm if A.pol == "x" else eym) * norm[:, None], t_cross=(eym if A.pol == "x" else exm) * norm[:, None],
             F_lens=F_lens, F_ref=F_ref, P_inc=P_inc, x_range=[x_lo, x_hi])
    info = dict(pol=A.pol, resolution=A.resolution, cells=int(len(cells)), cell_size=[sx, sy, sz], voxels=int(sx * sy * sz * A.resolution ** 3),
                atoms_by_family=used, fit_TiO2=fit_p, fit_SiO2=fit_s, wall_s_lens=t_run, wall_s_ref=t_ref, meep_time=t_meep,
                n_procs=mp.count_processors(), hit_tmax=bool(t_meep >= A.tmax), dft_change=dft_change, T_total=(F_lens / P_inc).tolist(), lams=lams.tolist(), x_range=[x_lo, x_hi])
    json.dump(info, open(str(base) + ".json", "w"), indent=1)
    print(f"[meep_strip] pol {A.pol} res {A.resolution:g}: {len(cells)} cells, {info['voxels'] / 1e6:.1f} M voxels, "
          f"{t_run / 60:.1f} min (+ref {t_ref / 60:.1f}), T = {np.round(F_lens / P_inc, 3).tolist()}", flush=True)
    print(f"  fits: TiO2 max dn {fit_p['max_dn']:.1e}, SiO2 max dn {fit_s['max_dn']:.1e}; wrote {base}.npz", flush=True)
