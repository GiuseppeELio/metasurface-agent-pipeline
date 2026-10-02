#!/usr/bin/env python3
"""Stage D: a circular metalens assembled from a meta-atom library (LUT), with a fast vector focusing model.

The required phase in the output medium is the hyperbolic profile
    phi(r) = -k0 n_out (sqrt(r^2 + f^2) - f),
sampled at the centre of every square cell (period P of the library). Each cell gets the library atom whose
complex transmission best projects onto exp(i phi): max sqrt(T_k) cos(phi_k - phi) ("merit"), or simply the
nearest phase ("nearest"). The lens is the union of the atoms' binary patterns; cells outside the aperture are
left empty (background material).

Fast model (local periodic approximation, LPA): just above the layer the field of cell k is sqrt(T_k) e^{i phi_k}
(unit-amplitude x-polarized plane wave from the input medium; T_k is the power transmission of atom k in its
periodic lattice). The field is propagated in the output medium with the vector angular-spectrum method
(E_z and H from transversality), giving the axial profile, the focal spot and
    focusing efficiency = power through a disk of radius 3 x FWHM in the focal plane / incident power on the aperture.
Two references isolate the loss channels: an ideal continuous-phase lens (|t| = 1), and the phases of the chosen
atoms with |t| = 1 ("ideal_sampled": phase sampling on the cell grid + level quantization). The Strehl ratio is the focal peak intensity relative to the ideal lens.

  python metalens.py --library lut_h800 --diameter-um 20 --na 0.25 --name lens_na025
  python metalens.py --library lut_h800 --diameter-um 20 --focal-um 17.6 --name lens_na050

Metrics: focus position (maximum of the on-axis intensity), depth of focus (DOF: full width at half maximum of
the on-axis intensity), lateral FWHM in x and y at the focus, focusing efficiency, Strehl ratio. Longitudinal
sections |E|^2(x, z) and |E|^2(y, z) show the propagation from the lens surface through the focus.

Outputs runs/<name>/: lens.json (parameters, atom map, metrics), atom_map.npy, rho_lens.npz (binary layout,
packed bits), lens_model.npz (axial profiles, focal cuts, longitudinal sections), lens_design.png, lens_focus.png. The full-wave check is validation/meep_lens.py.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ptopt.lensprop import VectorASM  # noqa: E402
if int(os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0"))) != 0:
    sys.exit(0)   # started under mpirun (e.g. as a line of a Meep list): only rank 0 builds the lens
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--library", required=True, help="runs/<lib>.json written by select_atoms.py")
ap.add_argument("--name", required=True)
ap.add_argument("--diameter-um", type=float, required=True, help="aperture diameter; rounded to an odd number of cells")
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--na", type=float, help="numerical aperture in the output medium (sets the focal length)")
g.add_argument("--focal-um", type=float)
ap.add_argument("--select", choices=["merit", "nearest"], default="merit")
ap.add_argument("--pol", choices=["x", "y"], default="x", help="library data used for t_k (d4 atoms: x = y)")
ap.add_argument("--px-per-cell", type=int, default=8, help="sampling of the LPA field (pixels per cell)")
ap.add_argument("--domain", type=float, default=2.2, help="ASM window, in units of the diameter")
ap.add_argument("--runs-dir", default=str(HERE / "runs"))
a = ap.parse_args()

runs = Path(a.runs_dir)
lib_file = Path(a.library) if Path(a.library).exists() else runs / f"{a.library}.json"
lib = json.loads(lib_file.read_text())
atoms = []
for name in lib["atoms"]:
    cfg = json.loads((runs / name / "config.json").read_text())
    det = json.loads((runs / name / "summary.json").read_text())["details_binary"][0]
    atoms.append({"dir": name, "T": det[f"T_{a.pol}"], "phase": det[f"phase_rel_deg_{a.pol}"] % 360,
                  "rho": np.load(runs / name / "rho_binary.npy").astype(np.uint8), "cfg": cfg})
cfg0 = atoms[0]["cfg"]
P = cfg0["period_nm"][0] / 1000.0
assert abs(cfg0["period_nm"][1] / 1000.0 - P) < 1e-9, "square cells expected"
lam = float(cfg0.get("wavelengths_nm", [cfg0.get("wavelength_nm")])[0]) / 1000.0
out_med = cfg0.get("superstrate", {"n": 1.0}) if cfg0.get("illumination", "from_substrate") == "from_substrate" \
    else cfg0.get("substrate", {"n": 1.0})
n_out = float(np.real(complex(out_med.get("n", 1.0))))
k = 2 * math.pi * n_out / lam

Nc = int(round(a.diameter_um / P)); Nc += (Nc + 1) % 2           # odd: one cell on the optical axis
D = Nc * P; R = D / 2
f = a.focal_um if a.focal_um else R * math.sqrt((n_out / a.na) ** 2 - 1)
NA = n_out * R / math.hypot(R, f)
c = (np.arange(Nc) - Nc // 2) * P
X, Y = np.meshgrid(c, c, indexing="ij")
r = np.hypot(X, Y)
inside = r <= R
phi_t = (-k * (np.sqrt(r ** 2 + f ** 2) - f)) % (2 * math.pi)

tk = np.array([math.sqrt(z["T"]) * np.exp(1j * math.radians(z["phase"])) for z in atoms])
ph_k = np.array([math.radians(z["phase"]) for z in atoms])
# a global phase offset is free: align the atom of the on-axis cell with the best overall projection
best_off, best_score = 0.0, -1
for off in np.radians(np.arange(0, 360, 1.0)):
    tgt = (phi_t + off)[inside]
    sc = np.max(np.real(tk[None, :] * np.exp(-1j * tgt[:, None])), axis=1).sum()
    if sc > best_score:
        best_off, best_score = off, sc
tgt = (phi_t + best_off) % (2 * math.pi)
if a.select == "merit":
    amap = np.argmax(np.real(tk[None, None, :] * np.exp(-1j * tgt[..., None])), axis=-1)
else:
    d = np.abs((ph_k[None, None, :] - tgt[..., None] + math.pi) % (2 * math.pi) - math.pi)
    amap = np.argmin(d, axis=-1)
amap = np.where(inside, amap, -1)

# binary layout (full resolution of the library atoms)
nper = atoms[0]["rho"].shape[0]
lay = np.zeros((Nc * nper, Nc * nper), np.uint8)
for i in range(Nc):
    for j in range(Nc):
        if amap[i, j] >= 0:
            lay[i * nper:(i + 1) * nper, j * nper:(j + 1) * nper] = atoms[amap[i, j]]["rho"]

# ---------------- fast vector angular-spectrum model -------------------------------------------------------
s = a.px_per_cell
dx = P / s
N = int(2 ** math.ceil(math.log2(a.domain * D / dx)))
L = N * dx
xs = (np.arange(N) - N // 2) * dx
XX, YY = np.meshgrid(xs, xs, indexing="ij")
icell = np.floor(XX / P + 0.5).astype(int) + Nc // 2
jcell = np.floor(YY / P + 0.5).astype(int) + Nc // 2
valid = (icell >= 0) & (icell < Nc) & (jcell >= 0) & (jcell < Nc)
ic, jc = np.clip(icell, 0, Nc - 1), np.clip(jcell, 0, Nc - 1)
ap_mask = valid & inside[ic, jc]


def near_field(kind):
    if kind == "lens":
        t_cell = np.where(amap >= 0, tk[np.clip(amap, 0, None)], 0)
    elif kind == "ideal_sampled":
        t_cell = np.where(amap >= 0, np.exp(1j * ph_k[np.clip(amap, 0, None)]), 0)
    else:  # ideal continuous phase, unit amplitude, same aperture, evaluated on the fine grid
        rr = np.hypot(XX, YY)
        return np.where(ap_mask, np.exp(-1j * k * (np.sqrt(rr ** 2 + f ** 2) - f)), 0)
    return np.where(ap_mask, t_cell[ic, jc], 0)


P_inc = 0.5 * n_out * ap_mask.sum() * dx * dx        # unit plane wave (power density 0.5 n |E|^2) on the aperture
zs = np.linspace(0.3 * f, 1.7 * f, 281)
zsec = np.linspace(0.0, 1.7 * f, 240)                 # longitudinal sections, from the lens surface
hw = 0.6 * D
res, secs = {}, {}
for kind in ("ideal", "ideal_sampled", "lens"):
    prop = VectorASM(near_field(kind), np.zeros((N, N), complex), dx, lam, n_out, pad=1.0)
    res[kind] = prop.metrics(zs, P_inc)
    E0 = near_field(kind)   # transmitted fraction of the LPA field (sum of the atoms' T over the aperture)
    res[kind]["transmitted_fraction"] = float(0.5 * n_out * (np.abs(E0) ** 2).sum() * dx * dx / P_inc)
    if kind != "ideal_sampled":
        xsec, secs[f"xz_{kind}"] = prop.section(zsec, "x", hw)
        _, secs[f"yz_{kind}"] = prop.section(zsec, "y", hw)
for kind in res:
    res[kind]["strehl_vs_ideal"] = res[kind]["peak_I"] / res["ideal"]["peak_I"]

out = runs / a.name; out.mkdir(parents=True, exist_ok=True)
np.save(out / "atom_map.npy", amap.astype(np.int16))
np.savez_compressed(out / "rho_lens.npz", packed=np.packbits(lay, axis=None), shape=np.array(lay.shape))
counts = np.bincount(amap[amap >= 0], minlength=len(atoms))
summary = {"name": a.name, "library": lib_file.stem, "atoms": [z["dir"] for z in atoms],
           "atom_T": [z["T"] for z in atoms], "atom_phase_deg": [z["phase"] for z in atoms], "atom_use_count": counts.tolist(),
           "wavelength_um": lam, "n_out": n_out, "period_um": P, "cells": Nc, "diameter_um": D, "focal_um": f, "NA": NA,
           "diffraction_limit_fwhm_um": 0.514 * lam / NA,
           "fresnel_number": R ** 2 / (lam / n_out * f),
           "diffraction_limit_fwhm_at_focus_um": 0.514 * lam / (n_out * R / math.hypot(R, res["ideal"]["z_focus_um"])), "select": a.select, "phase_offset_deg": math.degrees(best_off),
           "layer": cfg0["layer"], "substrate": cfg0.get("substrate"), "superstrate": cfg0.get("superstrate"),
           "illumination": cfg0.get("illumination", "from_substrate"), "atom_grid_px": int(nper),
           "lpa_model": {k_: {kk: v for kk, v in d.items() if not kk.startswith("_")} for k_, d in res.items()}}
(out / "lens.json").write_text(json.dumps(summary, indent=2))
sel_ = np.abs(xs) <= 4 * 0.514 * lam / NA
np.savez_compressed(out / "lens_model.npz", z_um=zs, x_um=xs[sel_], ideal_peak=res["ideal"]["peak_I"],
                    zsec_um=zsec, xsec_um=xsec, **secs,
                    **{f"axial_{k_}": d["_axial"] for k_, d in res.items()},
                    **{f"cutx_{k_}": d["_I"][sel_, N // 2] for k_, d in res.items()},
                    **{f"cuty_{k_}": d["_I"][N // 2, sel_] for k_, d in res.items()})
L_ = res["lens"]
print(f"[metalens] {a.name}: {Nc}x{Nc} cells, D={D:.2f} um, f={f:.2f} um, NA={NA:.3f}, "
      f"diffraction-limited FWHM {0.514 * lam / NA:.2f} um, Fresnel number {R ** 2 / (lam / n_out * f):.2f}"
      " (a low Fresnel number moves the intensity maximum towards the lens)")
for kind, d in res.items():
    print(f"  {kind:13s} focus z={d['z_focus_um']:.2f} um  FWHM x/y={d['fwhm_x_um']:.2f}/{d['fwhm_y_um']:.2f} um  "
          f"DOF={d['dof_um']:.2f} um  T={d['transmitted_fraction']:.3f}  focusing eff={d['focusing_efficiency']:.3f}  "
          f"Strehl={d['strehl_vs_ideal']:.3f}")

# ---------------- figures ----------------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    ph_show = np.where(inside, np.degrees(ph_k[np.clip(amap, 0, None)]), np.nan)
    im = axs[0].imshow(ph_show.T, origin="lower", cmap="twilight", vmin=0, vmax=360, extent=[-R, R, -R, R])
    fig.colorbar(im, ax=axs[0], label="atom phase (deg)")
    axs[0].set(title=f"phase map ({Nc}x{Nc} cells)", xlabel="x ($\\mu$m)", ylabel="y ($\\mu$m)")
    axs[1].imshow(lay.T, origin="lower", cmap="gray_r", extent=[-R, R, -R, R], interpolation="nearest")
    axs[1].set(title=f"layout, D = {D:.1f} $\\mu$m, NA = {NA:.2f}", xlabel="x ($\\mu$m)")
    zc = min(Nc // 2, 6)                                    # zoom: one quadrant edge to centre
    lo, hi = (Nc // 2) * nper, (Nc // 2 + zc + 1) * nper
    axs[2].imshow(lay[lo:hi, lo:hi].T, origin="lower", cmap="gray_r", interpolation="nearest",
                  extent=[-P / 2, (zc + 0.5) * P, -P / 2, (zc + 0.5) * P])
    axs[2].set(title="zoom from the optical axis", xlabel="x ($\\mu$m)", ylabel="y ($\\mu$m)")
    fig.tight_layout(); fig.savefig(out / "lens_design.png", dpi=170)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for kind, ls in (("ideal", ":"), ("ideal_sampled", "--"), ("lens", "-")):
        axs[0].plot(zs, res[kind]["_axial"] / res["ideal"]["peak_I"], ls, label=kind.replace("_", " "))
    axs[0].axvline(f, c="0.6", lw=0.6)
    axs[0].set(xlabel="z ($\\mu$m)", ylabel="on-axis |E|$^2$ / ideal peak", title="axial profile"); axs[0].legend(fontsize=7)
    w = 4 * 0.514 * lam / NA
    sel = np.abs(xs) <= w
    I = res["lens"]["_I"] / res["ideal"]["peak_I"]
    im = axs[1].imshow(I[np.ix_(sel, sel)].T, origin="lower", cmap="magma", extent=[-w, w, -w, w])
    fig.colorbar(im, ax=axs[1], label="|E|$^2$ / ideal peak")
    axs[1].set(title=f"focal plane, z = {L_['z_focus_um']:.1f} $\\mu$m", xlabel="x ($\\mu$m)", ylabel="y ($\\mu$m)")
    for kind, ls in (("ideal", ":"), ("lens", "-")):
        axs[2].plot(xs[sel], res[kind]["_I"][sel, N // 2] / res["ideal"]["peak_I"], ls, label=f"{kind}, x cut")
    axs[2].plot(xs[sel], res["lens"]["_I"][N // 2, sel] / res["ideal"]["peak_I"], "--", label="lens, y cut")
    axs[2].set(xlabel="position ($\\mu$m)", ylabel="|E|$^2$ / ideal peak",
               title=f"focusing eff. {100 * L_['focusing_efficiency']:.1f}%, Strehl {L_['strehl_vs_ideal']:.2f}")
    axs[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "lens_focus.png", dpi=170)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    pk = res["ideal"]["peak_I"]
    for ax_, key, ttl in ((axs[0], "xz_lens", "xz plane (y = 0)"), (axs[1], "yz_lens", "yz plane (x = 0)")):
        im = ax_.imshow(secs[key] / pk, origin="lower", aspect="auto", cmap="magma", vmin=0,
                        extent=[xsec[0], xsec[-1], zsec[0], zsec[-1]])
        ax_.axhline(L_["z_focus_um"], c="w", lw=0.5, ls=":")
        if L_["dof_range_um"]:
            for zz in L_["dof_range_um"]:
                ax_.axhline(zz, c="c", lw=0.6, ls="--")
        ax_.set(xlabel=ttl.split()[0][0] + " ($\\mu$m)", title=ttl)
    axs[0].set_ylabel("z above the lens ($\\mu$m)")
    fig.colorbar(im, ax=axs, label="|E|$^2$ / ideal peak")
    fig.suptitle(f"propagation: focus {L_['z_focus_um']:.1f} $\\mu$m, DOF {L_['dof_um']:.1f} $\\mu$m (dashed), "
                 f"FWHM {L_['fwhm_x_um']:.2f}/{L_['fwhm_y_um']:.2f} $\\mu$m", fontsize=9)
    fig.savefig(out / "lens_propagation.png", dpi=170, bbox_inches="tight")
    print("[metalens] wrote", out / "lens_design.png", out / "lens_focus.png", out / "lens_propagation.png")
except ImportError:
    pass
