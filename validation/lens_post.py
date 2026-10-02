#!/usr/bin/env python3
"""Propagation, depth of focus and FWHM of a metalens from the field recorded by Meep (validation/meep_lens.py).

The transverse field on the monitor plane just above the lens (meep_lens_plane_res<R>.npz) is propagated with the
same vector angular-spectrum code as the locally periodic model of metalens.py (ptopt/lensprop.py). Distances are
measured from the top of the lens. Normalization: the incident power is the propagating power on the plane divided
by Meep's total transmission T, so efficiencies are in fraction of the incident power, as in metalens.py.

  python validation/lens_post.py --run runs/<lens> [--res 30]

Writes runs/<lens>/lens_post_res<R>.json (Meep and LPA metrics side by side) and lens_post_res<R>.npz
(axial profiles and xz / yz sections), and lens_post_res<R>.png.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from ptopt.lensprop import VectorASM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--res", type=int, help="Meep resolution of the plane file (default: highest available)")
ap.add_argument("--pad", type=float, default=3.0, help="zero padding of the propagation window (x plane width)")
a = ap.parse_args()
run = Path(a.run)
L = json.loads((run / "lens.json").read_text())
files = sorted(glob.glob(str(run / "meep_lens_plane_res*.npz")), key=lambda p: int(p.split("res")[-1][:-4]))
if not files:
    sys.exit("[lens_post] no meep_lens_plane_res*.npz in " + str(run))
fp = next(p for p in files if p.endswith(f"res{a.res}.npz")) if a.res else files[-1]
R_ = int(fp.split("res")[-1][:-4])
P = np.load(fp)
Ex, Ey = P["Ex"], P["Ey"]
dx = float(P["width_um"]) / (Ex.shape[0] - 1)
z0 = float(P["z_above_layer_um"])
f = L["focal_um"]
prop = VectorASM(Ex, Ey, dx, float(P["wavelength_um"]), float(P["n_out"]), pad=a.pad)
P_inc = prop.power(0.0) / float(P["T_total"])
zs = np.linspace(0.3 * f, 1.7 * f, 281)
m = prop.metrics(zs - z0, P_inc)
m["z_focus_um"] += z0
if m["dof_range_um"]:
    m["dof_range_um"] = [z + z0 for z in m["dof_range_um"]]
m["transmitted_fraction"] = float(P["T_total"])
zsec = np.linspace(0.0, 1.7 * f, 240)
hw = 0.6 * L["diameter_um"]
xsec, xz = prop.section(np.maximum(zsec - z0, 0.0), "x", hw)
_, yz = prop.section(np.maximum(zsec - z0, 0.0), "y", hw)

lpa = L["lpa_model"]["lens"]
out = {"run": run.name, "meep_resolution": R_, "NA": L["NA"], "focal_um": f,
       "meep": {k: v for k, v in m.items() if not k.startswith("_")},
       "lpa": {k: lpa.get(k) for k in ("z_focus_um", "dof_um", "dof_range_um", "fwhm_x_um", "fwhm_y_um",
                                       "focusing_efficiency", "transmitted_fraction", "strehl_vs_ideal")},
       "ideal": {k: L["lpa_model"]["ideal"].get(k) for k in ("z_focus_um", "dof_um", "fwhm_x_um", "fwhm_y_um",
                                                             "focusing_efficiency")}}
(run / f"lens_post_res{R_}.json").write_text(json.dumps(out, indent=2))
np.savez_compressed(run / f"lens_post_res{R_}.npz", z_um=zs, axial=m["_axial"], zsec_um=zsec, xsec_um=xsec, xz=xz, yz=yz)
mm, ll = out["meep"], out["lpa"]
print(f"[lens_post] {run.name} (Meep res {R_}) vs LPA model:")
for k in ("z_focus_um", "dof_um", "fwhm_x_um", "fwhm_y_um", "focusing_efficiency", "transmitted_fraction"):
    print(f"  {k:22s} Meep {mm[k]:8.3f}   LPA {ll[k] if ll[k] is not None else float('nan'):8.3f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Mm = np.load(run / "lens_model.npz")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax_, data, ttl in ((axs[0], Mm["xz_lens"], "LPA model"), (axs[1], xz, f"Meep, res {R_}")):
        ax_.imshow(data / data.max(), origin="lower", aspect="auto", cmap="magma", vmin=0,
                   extent=[xsec[0], xsec[-1], zsec[0], zsec[-1]])
        ax_.set(xlabel="x ($\\mu$m)", ylabel="z above the lens ($\\mu$m)", title=f"{ttl}: |E|$^2$, xz plane")
    axs[2].plot(Mm["z_um"], Mm["axial_lens"] / Mm["axial_lens"].max(), label=f"LPA (DOF {ll['dof_um']:.1f} $\\mu$m)")
    axs[2].plot(zs, m["_axial"] / m["_axial"].max(), "--", label=f"Meep (DOF {mm['dof_um']:.1f} $\\mu$m)")
    axs[2].axhline(0.5, c="0.7", lw=0.6); axs[2].axvline(f, c="0.7", lw=0.6)
    axs[2].set(xlabel="z above the lens ($\\mu$m)", ylabel="on-axis |E|$^2$ (normalized)", title="axial profile")
    axs[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(run / f"lens_post_res{R_}.png", dpi=160)
except ImportError:
    pass
