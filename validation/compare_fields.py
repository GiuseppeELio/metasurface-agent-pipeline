#!/usr/bin/env python3
"""Side-by-side RCWA (TORCWA) vs FDTD (Meep) near fields and far field of a periodic design.

Inputs (same run directory): fields_rcwa.npz (validation/torcwa_fields.py) and
fields_meep_res<R>.npz (validation/meep_validate.py --fields).

Figure (per input polarization):
  (a,b) Re(E_pol) in the xz plane, RCWA and Meep, same colour scale
  (c,d) beam steering: a Gaussian-apodized aperture of K periods, propagated in the output medium with the
        angular-spectrum method from the near field just above the layer (all three E components, |E|^2)
  (e)   diffraction efficiencies in % of the incident power, computed from the near fields:
        eta_m = n_out cos(theta_m) sum_c |E_c,m|^2 / (n_in |E_inc|^2), with E_c,m the Fourier coefficients of
        the transmitted field (unit incident amplitude). Bars: RCWA and Meep; markers: the solvers' own
        S-parameter / eigenmode-coefficient results (summary.json, meep_res*.json) when available.
        The thin curves show the same efficiencies broadened by the finite aperture (peak height = eta_m).

  python validation/compare_fields.py --run runs/<dir> [--meep-res 50] [--periods 12] [--zprop 40]
"""
import argparse
import glob
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--meep-res", type=int, help="which fields_meep_res<R>.npz (default: highest)")
ap.add_argument("--periods", type=int, default=12, help="periods in the finite aperture")
ap.add_argument("--zprop", type=float, default=40.0, help="propagation distance shown (um)")
ap.add_argument("--zline", type=float, default=0.6, help="height above the layer where the near field is sampled (um)")
ap.add_argument("--pols", nargs="+", default=["x", "y"])
a = ap.parse_args()
run = Path(a.run)
R = np.load(run / "fields_rcwa.npz")
mfiles = sorted(glob.glob(str(run / "fields_meep_res*.npz")), key=lambda f: int(f.split("res")[-1].split(".")[0]))
M = (np.load(next(f for f in mfiles if f.endswith(f"res{a.meep_res}.npz")) if a.meep_res else mfiles[-1])
     if mfiles else None)
if M is None:
    print("[compare_fields] no fields_meep_res*.npz found: RCWA-only figure")
lam = float(R["wavelength_um"]); h = float(R["h_um"]); Lx = float(R["period_um"][0])
n_out = float(np.real(R["n_out"]))
k0 = 2 * math.pi / lam


def line_at(F, comp, pol, z_target):
    z = F["z_um"]; i = int(np.argmin(np.abs(z - z_target)))
    x = F["x_um"]; e = F[f"{comp}_{pol}"][:, i]
    if abs(x[-1] - x[0] - Lx) < 1e-6 * Lx + (x[1] - x[0]) * 0.5:   # drop duplicated end point of a closed grid
        x, e = x[:-1], e[:-1]
    xs = np.linspace(0, Lx, 256, endpoint=False)               # common periodic resampling
    xp = np.concatenate([x, x[:1] + Lx]); ep = np.concatenate([e, e[:1]])
    return xs, np.interp(xs, xp, ep.real) + 1j * np.interp(xs, xp, ep.imag)


def order_efficiencies(F, pol, z_target, mmax=6):
    """Power in each transmitted order from the Fourier coefficients of all E components on one period."""
    n_in = float(np.real(F["n_in"]))
    E2 = {}
    for comp in ("Ex", "Ey", "Ez"):
        xs, e = line_at(F, comp, pol, z_target)
        c = np.fft.fft(e) / e.size
        for m in range(-mmax, mmax + 1):
            E2[m] = E2.get(m, 0.0) + abs(c[m % e.size]) ** 2
    out = {}
    for m, v in E2.items():
        s_ = m * lam / (n_out * Lx)
        if abs(s_) < 1:
            ct = math.sqrt(1 - s_ ** 2)
            out[m] = (math.degrees(math.asin(s_)), n_out * ct * v / n_in)
    return out


def aperture_beam(xs, e, K, zmax, nz=300):
    dx = xs[1] - xs[0]
    E = np.tile(e, K)
    N = E.size
    X = np.arange(N) * dx
    w0 = 0.35 * K * Lx
    E = E * np.exp(-((X - X.mean()) / w0) ** 2)
    pad = 8 * N
    Ep = np.zeros(pad, complex); Ep[(pad - N) // 2:(pad + N) // 2] = E
    kx = 2 * math.pi * np.fft.fftfreq(pad, d=dx)
    A = np.fft.fft(Ep)
    kz = np.sqrt((n_out * k0) ** 2 - kx ** 2 + 0j)
    prop = np.where(np.abs(kx) < n_out * k0, 1.0, 0.0)
    zs = np.linspace(0, zmax, nz)
    beam = np.abs(np.fft.ifft(A[None, :] * prop * np.exp(1j * kz[None, :] * zs[:, None]), axis=1)) ** 2
    xg = (np.arange(pad) - pad / 2) * dx
    sin_t = kx / (n_out * k0); ok = np.abs(sin_t) < 1
    ang = np.degrees(np.arcsin(sin_t[ok])); spec = np.abs(A[ok]) ** 2
    o = np.argsort(ang)
    return xg, zs, beam, ang[o], spec[o]


for pol in a.pols:
    comp = "Ex" if pol == "x" else "Ey"
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.1, 0.9], hspace=0.55, wspace=0.3)
    vmax = np.percentile(np.abs(np.real(R[f"{comp}_{pol}"])), 99.5)
    solvers = [(R, "RCWA (TORCWA)")] + ([(M, f"FDTD (Meep, res {int(M['resolution'])} px/um)")] if M is not None else [])
    for j, (F, name) in enumerate(solvers):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(np.real(F[f"{comp}_{pol}"]).T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                  extent=[F["x_um"][0], F["x_um"][-1], F["z_um"][0], F["z_um"][-1]])
        ax.axhline(0, c="k", lw=0.6); ax.axhline(h, c="k", lw=0.6)
        ax.set_ylim(max(G["z_um"][0] for G, _ in solvers), min(G["z_um"][-1] for G, _ in solvers))  # common z window
        ax.set(title=f"{name}\nRe({comp}), {pol}-pol input", xlabel="x ($\\mu$m)", ylabel="z ($\\mu$m)")
    ax_f = fig.add_subplot(gs[2, :])
    beams, effs = [], []
    for j, (F, name, ls) in enumerate([(R, "RCWA", "-")] + ([(M, "Meep", "--")] if M is not None else [])):
        beam = 0
        for c in ("Ex", "Ey", "Ez"):
            xs, e = line_at(F, c, pol, h + a.zline)
            xg, zs, b, _, _ = aperture_beam(xs, e, a.periods, a.zprop)
            beam = beam + b
        beams.append(beam)
        effs.append(order_efficiencies(F, pol, h + a.zline))
    ms = sorted(set().union(*[set(e) for e in effs]))
    th = np.linspace(-89, 89, 2000)
    sig = lam / (n_out * a.periods * Lx * 0.35 * math.sqrt(2))       # angular width of the apodized aperture
    nsol = len(effs)
    for k, (eff, name, col) in enumerate(zip(effs, ("RCWA (from fields)", "Meep (from fields)"), ("C0", "C1"))):
        off = (k - 0.5) if nsol > 1 else 0.0
        angs = [eff[m][0] for m in ms if m in eff]; vals = [100 * eff[m][1] for m in ms if m in eff]
        ax_f.bar(np.array(angs) + off * 3.0, vals, width=2.8, color=col, alpha=0.85, label=name)
        curve = sum(100 * eff[m][1] * np.exp(-((np.sin(np.radians(th)) - math.sin(math.radians(eff[m][0]))) / sig) ** 2)
                    for m in eff)
        ax_f.plot(th, curve, color=col, lw=0.7)
        for ang_, v_ in zip(angs, vals):
            if v_ > 1:
                ax_f.text(ang_ + off * 7.0, v_ + 3, f"{v_:.1f}", ha="center", fontsize=6.5, color=col)
    print(f"[compare_fields] pol {pol}: eta from fields  RCWA " +
          " ".join(f"m={m}:{100 * effs[0][m][1]:.1f}%" for m in ms if m in effs[0]) +
          (" | Meep " + " ".join(f"m={m}:{100 * effs[1][m][1]:.1f}%" for m in ms if m in effs[1]) if len(effs) > 1 else ""),
          flush=True)
    vb = max(b.max() for b in beams)
    for j, (b, name) in enumerate(zip(beams, ("RCWA", "Meep")[:len(beams)])):
        ax_b = fig.add_subplot(gs[1, j])
        ax_b.imshow(b / vb, origin="lower", vmin=0, vmax=1, cmap="magma", aspect="auto", extent=[xg[0], xg[-1], 0, a.zprop])
        ax_b.set(xlim=(-0.8 * a.periods * Lx, 0.8 * a.periods * Lx + a.zprop * 1.3), xlabel="x ($\\mu$m)",
                 ylabel="z above layer ($\\mu$m)", title=f"{name}: beam |E|$^2$ ({a.periods} periods)")
    for m in range(-6, 7):
        s_ = m * lam / (n_out * Lx)
        if abs(s_) < 1:
            ax_f.axvline(math.degrees(math.asin(s_)), c="0.85", lw=0.6, zorder=0)
    ax_f.set(xlabel="angle in output medium (deg)", ylabel="$\\eta$ (% of incident power)", xlim=(-90, 90), ylim=(0, 105))
    ax_f.legend(fontsize=7)
    out = run / f"compare_fields_{pol}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("[compare_fields] wrote", out)
