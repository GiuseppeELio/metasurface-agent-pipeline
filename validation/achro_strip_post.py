#!/usr/bin/env python3
"""Post-processing of validation/meep_achro_strip.py: rebuild the circular-lens aperture field from the two strip runs
and focus it with the same band-limited vector angular spectrum as the design model (tools/achro_tio2.py).

  x-pol strip  -> t_rad(r) (E along the radius),  y-pol strip -> t_az(r) (E along the rows)
  x-polarized illumination of the circular lens:  Ex = t_rad cos^2 phi + t_az sin^2 phi,  Ey = (t_rad - t_az) cos phi sin phi
The strip covers x in [-R, R]; t(r) is the mean of the two half-strips (x > 0 and x < 0), interpolated (complex, linear)
on the square cell grid. Outputs in runs/<lens>/: meep_focus.json, meep_focal_maps.npz, figures meep_*.png.

  python validation/achro_strip_post.py --run runs/<lens> --resolution 40
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "tools"))
from achro_tio2 import propagate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--resolution", type=float, default=40.0)
ap.add_argument("--tag", default="")
ap.add_argument("--segments", default="", help="glob tag of segmented runs, e.g. fullS (files ..._fullS0.npz, _fullS1.npz ...)")
A = ap.parse_args()
run = Path(A.run); L = json.load(open(run / "lens.json"))
tag = f"_{A.tag}" if A.tag else ""
def merge(pol):
    """Concatenate segment runs: every x takes the segment whose centre is nearest (overlaps are discarded)."""
    import glob
    fs = sorted(glob.glob(str(run / f"meep_strip_{pol}_res{A.resolution:g}_{A.segments}*.npz")))
    segs = [np.load(f) for f in fs]
    xs = np.unique(np.round(np.concatenate([d["x_um"] for d in segs]), 4))
    cen = np.array([0.5 * sum(d["x_range"]) for d in segs])
    own = np.argmin(np.abs(xs[:, None] - cen[None, :]), axis=1)
    K = len(segs[0]["lam"]); t = np.zeros((K, len(xs)), complex)
    for i, d in enumerate(segs):
        sel = own == i
        for k in range(K):
            t[k, sel] = np.interp(xs[sel], d["x_um"], d["t_co"][k].real) + 1j * np.interp(xs[sel], d["x_um"], d["t_co"][k].imag)
    P_inc = sum(d["P_inc"] for d in segs); F = sum(d["F_lens"] for d in segs)
    print(f"  merged {len(segs)} {pol}-pol segments: x {xs[0]:.1f}..{xs[-1]:.1f} um", flush=True)
    return dict(x_um=xs, lam=segs[0]["lam"], t_co=t, F_lens=F, P_inc=P_inc)


if A.segments:
    sx, sy = merge("x"), merge("y"); tag = f"_{A.segments}"
else:
    sx = np.load(run / f"meep_strip_x_res{A.resolution:g}{tag}.npz"); sy = np.load(run / f"meep_strip_y_res{A.resolution:g}{tag}.npz")
lut = np.load(run / "lut_used.npz", allow_pickle=True)
P = L["P_nm"] / 1000.0; R = L["R_um"]; f = L["f_um"]; NA = L["NA"]; band = L.get("band_na") or 2.5 * NA
lams = sx["lam"]; x = sx["x_um"]


def radial(tline):
    """mean of the two half-strips on a radial grid r = |x| (one value per cell centre)."""
    r = np.arange(0, R + P, P)
    pos = np.interp(r, x, tline.real) + 1j * np.interp(r, x, tline.imag)
    neg = np.interp(-r, x, tline.real) + 1j * np.interp(-r, x, tline.imag)
    return r, 0.5 * (pos + neg), np.abs(pos - neg).max() / max(np.abs(pos).max(), 1e-12)


ncell = int(round(2 * R * 1000 / L["P_nm"])) | 1
xc = (np.arange(ncell) - ncell // 2) * P
XC, YC = np.meshgrid(xc, xc, indexing="ij"); RC = np.hypot(XC, YC); inside = RC <= R
PHI = np.arctan2(YC, XC); c2, s2, cs = np.cos(PHI) ** 2, np.sin(PHI) ** 2, np.cos(PHI) * np.sin(PHI)
# LPA reference along the same row (what the design model assumed)
r_grid = np.array(L["radial_assign"]["r_nm"]) / 1000.0; assign = np.array(L["radial_assign"]["atom"])
ts = lut["t"] if float(lut["s"]) > 0 else np.conj(lut["t"]); lam_lut = lut["lam"]

zs = np.linspace(0.5 * f, 1.5 * f, 401)
res = {"lens": str(run), "resolution": A.resolution, "NA": NA, "f_um": f, "eval": [], "strip": {}}
maps = {}
for k, lam in enumerate(lams):
    r, trad, asym_x = radial(sx["t_co"][k]); _, taz, asym_y = radial(sy["t_co"][k])
    tr = np.interp(RC, r, trad.real) + 1j * np.interp(RC, r, trad.imag)
    ta = np.interp(RC, r, taz.real) + 1j * np.interp(RC, r, taz.imag)
    Ex = np.where(inside, tr * c2 + ta * s2, 0); Ey = np.where(inside, (tr - ta) * cs, 0)
    zr = L.get("z_ref_um")
    m, mp_ = propagate(Ex, P, lam / 1000, R, f, band, zs, ey=Ey, inside=inside, z_fixed=zr)
    if zr is not None:      # ideal hyperbolic lens at the same wavelength and plane (Strehl reference)
        Lr = np.sqrt(RC ** 2 + f ** 2) - f
        mi, _ = propagate(np.where(inside, np.exp(-1j * 2 * np.pi / (lam / 1000) * Lr), 0), P, lam / 1000, R, f, band, zs,
                          inside=inside, z_fixed=zr)
        m["strehl_fixed"] = m["peak_fixed"] / mi["peak_fixed"]; m["ideal_eff_fixed"] = mi["eff_fixed"]
    dl = 0.514 * lam / 1000 / NA
    row = dict(m); row.update(lam=float(lam), diffraction_fwhm_um=dl, asym_x=float(asym_x), asym_y=float(asym_y),
                              T_strip_x=float(sx["F_lens"][k] / sx["P_inc"][k]), T_strip_y=float(sy["F_lens"][k] / sy["P_inc"][k]))
    # comparison with the LUT coefficients along the radius (LPA model)
    kk = int(np.argmin(np.abs(lam_lut - lam)))
    a_r = assign[np.clip(np.searchsorted(r_grid, r), 0, len(r_grid) - 1)]
    t_lpa = ts[a_r, kk] if abs(lam_lut[kk] - lam) < 1e-6 else None
    tm = trad                       # Meep (exp(-i w t)): arg = retardation, same convention as ts
    if t_lpa is not None:
        g = np.vdot(t_lpa, tm) / np.vdot(t_lpa, t_lpa)                       # best global complex factor
        d = np.angle(tm / (g / abs(g)) / t_lpa)
        row.update(rms_phase_vs_lut_deg=float(np.degrees(np.sqrt(np.mean(d ** 2)))),
                   coherent_overlap_vs_lut=float(abs(np.vdot(t_lpa, tm)) ** 2 / (np.vdot(t_lpa, t_lpa).real * np.vdot(tm, tm).real)))
        res["strip"][f"{lam:.0f}"] = dict(r=r[::10].tolist(), phase_err_deg=np.degrees(d[::10]).tolist(),
                                          T_meep=(np.abs(trad[::10]) ** 2).tolist(), T_lut=(np.abs(t_lpa[::10]) ** 2).tolist())
    res["eval"].append(row); maps[f"{lam:.0f}"] = mp_
    print(f"  {lam:.0f} nm: z_f={m['z_focus_um'] / 1000:.3f} mm  FWHM {m['fwhm_x_um']:.2f}/{m['fwhm_y_um']:.2f} um (DL {dl:.2f})  "
          f"eff {m['focusing_efficiency']:.3f}  T {m['transmitted_fraction']:.3f}  "
          f"phase rms vs LUT {row.get('rms_phase_vs_lut_deg', float('nan')):.1f} deg | z_ref: eff {m.get('eff_fixed', float('nan')):.3f} "
          f"Strehl {m.get('strehl_fixed', float('nan')):.3f}", flush=True)
json.dump(res, open(run / f"meep_focus_res{A.resolution:g}{tag}.json", "w"), indent=1)
np.savez_compressed(run / f"meep_focal_maps_res{A.resolution:g}{tag}.npz",
                    **{f"{k}_{q}": v for k, mm in maps.items() for q, v in mm.items()}, zs_um=zs)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
ks = list(maps.keys())
fig, ax = plt.subplots(2, len(ks), figsize=(2.6 * len(ks), 5.4))
for i, k in enumerate(ks):
    m = maps[k]; xx = m["x"]; I = m["I"] / m["I"].max()
    ax[0, i].imshow(I.T, extent=[xx[0], xx[-1], xx[0], xx[-1]], origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax[0, i].set_title(f"{k} nm", fontsize=9); ax[0, i].set_xlabel("x (um)", fontsize=8)
    if i == 0: ax[0, i].set_ylabel("y (um)", fontsize=8)
    ax[0, i].tick_params(labelsize=7)
    xz = m["xz"] / m["xz"].max()
    ax[1, i].imshow(xz, extent=[m["xs"][0], m["xs"][-1], zs[-1] / 1000, zs[0] / 1000], aspect="auto", cmap="inferno")
    ax[1, i].axhline(f / 1000, color="w", lw=0.5, ls=":"); ax[1, i].set_xlabel("x (um)", fontsize=8); ax[1, i].tick_params(labelsize=7)
    if i == 0: ax[1, i].set_ylabel("z (mm)", fontsize=8)
fig.suptitle(f"Meep (full-diameter strip, axisymmetric reconstruction): focal-plane |E|^2 (top) and xz section (bottom)", fontsize=9)
fig.tight_layout(); fig.savefig(run / f"meep_focal_maps_res{A.resolution:g}{tag}.png", dpi=170); plt.close(fig)
print("[achro_strip_post] wrote", run)
