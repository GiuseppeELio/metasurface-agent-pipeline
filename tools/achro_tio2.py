#!/usr/bin/env python3
"""Large-aperture achromatic metalens from a multi-wavelength LUT of fourfold atoms (tools/tio2_lut.py output).

Design rule as in tools/achro_lens.py (dispersion engineering, coherent overlap at every wavelength with a free global
phase C(lambda) per wavelength), with a fine radial grid suited to apertures of hundreds of microns.

Focusing model for large, low-NA lenses: locally periodic field (one LUT coefficient per square cell, dx = period),
vector angular spectrum restricted to |k_t| <= kt_max = --band-na * k (the light a low-NA lens sends towards the focus;
everything scattered at larger angles leaves the focal region and is counted only in the transmitted fraction).
The window is chosen so that this cone does not wrap around: half-width >= R + f * band_na.

  python tools/achro_tio2.py scan  --runs runs/tio2_lut_P450 runs/tio2_lut_P400 --diameter-um 400 --name achro850_scan
  python tools/achro_tio2.py build --run runs/tio2_lut_P450 --diameter-um 400 --na 0.02 --name achro850_lens
"""
import argparse, glob, json, math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "tools"))
import achro_lens as AL  # noqa: E402

NR = 4000                                       # radial samples of the design grid
_rg = AL.radial_grid
AL.radial_grid = lambda R, P, n=NR: _rg(R, P, NR)


def smooth_mask(T, lam, s, max_res=0.15, max_dT=0.08):
    """Atoms without sharp spectral features (guided-mode / lattice resonances need many identical neighbours and
    are not reproduced in a lens whose atoms change from cell to cell): phase residual from a quadratic fit in k
    below max_res (rad rms) and |T(l_k+1) - T(l_k)| below max_dT."""
    ts = T if s > 0 else np.conj(T); k = 2 * np.pi / lam
    ph = np.unwrap(np.angle(ts), axis=1)
    res = np.array([np.sqrt(np.mean((p - np.polyval(np.polyfit(k, p, 2), k)) ** 2)) for p in ph])
    dT = np.abs(np.diff(np.abs(T) ** 2, axis=1)).max(1)
    return (res <= max_res) & (dT <= max_dT)


def load(run, tmin=0.0, smooth=False):
    parts = sorted(glob.glob(str(Path(run) / "lut_*.npz")))
    ds = [np.load(p, allow_pickle=True) for p in parts]
    atoms = np.concatenate([d["atoms"] for d in ds]); T = np.concatenate([d["t"] for d in ds]).astype(complex)
    lam = ds[0]["lam"].astype(float); args = json.loads(str(ds[0]["args"]))
    args["period"] = float(args["period"])
    i_ref0 = int(np.argmin(np.where(atoms[:, 0] == 0, atoms[:, 1], 1e9)))
    s0 = -np.sign(np.polyfit(lam, np.unwrap(np.angle(T[i_ref0])), 1)[0])
    keep = (np.abs(T) ** 2).min(1) >= tmin
    if smooth:
        keep &= smooth_mask(T, lam, s0)
    s = s0                                                                    # sign from the smallest circular pillar
    return dict(lam=lam, atoms=atoms[keep], t=T[keep], s=s, args=args, path=str(run),
                families=[str(f) for f in ds[0]["families"]])


def f_of(R, na):
    return R * math.sqrt(1 - na ** 2) / na


def cmd_scan(A):
    out = HERE / "runs" / A.name; out.mkdir(parents=True, exist_ok=True)
    R = A.diameter_um * 500.0; res = {}
    for run in A.runs:
        L = load(run, A.tmin, A.smooth); key = Path(run).name; rows = []
        for na in A.nas:
            d = AL.design(L, R, f_of(R, na))
            rows.append(dict(na=float(na), f_mm=f_of(R, na) / 1e6, eta=d["eta"].tolist(), eta_phi=d["eta_phi"].tolist()))
            print(f"{key} NA {na:.4f} f {f_of(R, na) / 1e6:.2f} mm: min eta_phi {min(d['eta_phi']):.3f}  "
                  f"min eta {min(d['eta']):.3f}  mean eta {np.mean(d['eta']):.3f}", flush=True)
        ok = [r["na"] for r in rows if min(r["eta_phi"]) >= 0.8]
        res[key] = dict(args=L["args"], lam=L["lam"].tolist(), rows=rows, max_na=max(ok) if ok else None,
                        n_atoms=int(len(L["atoms"])), T_mean=float(np.mean(np.abs(L["t"]) ** 2)))
        print(f"== {key}: max NA (min eta_phi >= 0.8) = {res[key]['max_na']}", flush=True)
    json.dump(res, open(out / "scan.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, v in res.items():
        na = [r["na"] for r in v["rows"]]
        ax[0].plot(na, [min(r["eta_phi"]) for r in v["rows"]], marker="o", ms=3, label=key)
        ax[1].plot(na, [min(r["eta"]) for r in v["rows"]], marker="o", ms=3, label=key)
    ax[0].axhline(0.8, color="gray", lw=0.8, ls=":")
    ax[0].set(xlabel="NA", ylabel="min over band, phase-only efficiency", title=f"D = {A.diameter_um:.0f} um")
    ax[1].set(xlabel="NA", ylabel="min over band, coherent efficiency (T incl.)")
    for a in ax: a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "scan.png", dpi=170)
    print("[achro_tio2] wrote", out)


def propagate(tc, P_um, lam_um, R_um, f_um, band_na, zs, half_um=None, ey=None, inside=None, dx_target=None, z_fixed=None):
    """Focus the cell field (Ex = tc, Ey = ey; one complex value per cell of side P_um) with the vector angular
    spectrum, keeping only |k_t| <= band_na * k. The band-limited field is resampled exactly (spectrum crop) on a
    coarser grid dx' and placed in a window wide enough that the kept cone does not wrap. Returns metrics + maps."""
    from ptopt.lensprop import VectorASM, fwhm
    n0 = tc.shape[0]; NA = R_um / math.hypot(R_um, f_um)
    N0 = int(2 ** math.ceil(math.log2(n0 + 8)))
    def place(a, N):
        z = np.zeros((N, N), complex); o = (N - a.shape[0]) // 2; z[o:o + a.shape[0], o:o + a.shape[1]] = a; return z
    ex0 = place(tc, N0); ey0 = place(ey, N0) if ey is not None else np.zeros_like(ex0)
    full = VectorASM(ex0, ey0, P_um, lam_um, 1.0, pad=1.0)
    P_tr = full.power(0.0); del full
    k = 2 * math.pi / lam_um; dk0 = 2 * math.pi / (N0 * P_um)
    fwhm_dl = 0.514 * lam_um / NA
    dx_target = dx_target or fwhm_dl / 24.0
    Nc = int(2 * math.ceil(max(N0 * P_um / dx_target, 2.2 * band_na * k / dk0) / 2))
    Nc = min(Nc, N0); dx = N0 * P_um / Nc
    def resample(a):
        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))
        c0 = N0 // 2; h = Nc // 2
        Fc = F[c0 - h:c0 + h, c0 - h:c0 + h]
        KX, KY = np.meshgrid((np.arange(Nc) - h) * dk0, (np.arange(Nc) - h) * dk0, indexing="ij")
        Fc = np.where(np.hypot(KX, KY) <= band_na * k, Fc, 0)
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Fc))) * (Nc / N0) ** 2
    exc, eyc = resample(ex0), resample(ey0)
    half = R_um + f_um * band_na * 1.3 + 4 * fwhm_dl
    N = int(2 ** math.ceil(math.log2(max(2 * half / dx, Nc))))
    asm = VectorASM(place(exc, N), place(eyc, N), dx, lam_um, 1.0, pad=1.0)
    keep = np.hypot(asm.kx, asm.ky) <= band_na * asm.k
    for nm in ("ix", "iy", "kx", "ky", "kz", "Ax", "Ay", "Az"):
        setattr(asm, nm, getattr(asm, nm)[keep])
    Pinc = 0.5 * (np.count_nonzero(inside) if inside is not None else np.count_nonzero(np.abs(tc) > 0)) * P_um ** 2
    m = asm.metrics(zs, Pinc)
    I = m.pop("_I"); ax_I = m.pop("_axial")
    half_um = half_um or 3.0 * fwhm_dl
    c = N // 2; h = int(round(half_um / dx))
    m["transmitted_fraction"] = P_tr / Pinc
    m["band_fraction"] = asm.power(0.0) / Pinc
    m["grid"] = dict(dx_um=dx, N=N, Nc=Nc, N0=N0)
    if z_fixed is not None:                         # common focal plane (achromatic use): same disk radius 3 x FWHM_DL
        If, Szf = asm.plane(z_fixed); cc = N // 2
        XX, YY = np.meshgrid(asm.x, asm.x, indexing="ij")
        m["z_fixed_um"] = float(z_fixed); m["peak_fixed"] = float(If[cc, cc])
        m["eff_fixed"] = float(Szf[np.hypot(XX, YY) <= 3.0 * fwhm_dl].sum() * dx ** 2 / Pinc)
        If_c = If[cc - int(round(3 * fwhm_dl / dx)):cc + int(round(3 * fwhm_dl / dx)) + 1, cc]
        fw = fwhm(asm.x[cc - int(round(3 * fwhm_dl / dx)):cc + int(round(3 * fwhm_dl / dx)) + 1], If_c)
        m["fwhm_fixed_um"] = fw[0] if isinstance(fw, tuple) else fw
    xs, sec = asm.section(zs, "x", half_width=half_um)
    return m, dict(I=I[c - h:c + h + 1, c - h:c + h + 1].astype(np.float32), axial=ax_I.astype(np.float32),
                   x=asm.x[c - h:c + h + 1], xz=sec.astype(np.float32), xs=xs)


def cmd_build(A):
    out = HERE / "runs" / A.name; out.mkdir(parents=True, exist_ok=True)
    L = load(A.run, A.tmin, A.smooth); P = L["args"]["period"]; R = A.diameter_um * 500.0
    f = A.focal_um * 1000 if A.focal_um else f_of(R, A.na)
    lam = L["lam"]; NA = R / math.hypot(R, f)
    A.band_na = A.band_na or 2.5 * NA
    d = AL.design(L, R, f)
    ic = int(np.argmin(np.abs(lam - A.center))); dc = AL.design(L, R, f, sel_idx=[ic])
    ncell = int(round(2 * R / P)) | 1
    xc = (np.arange(ncell) - ncell // 2) * P
    XC, YC = np.meshgrid(xc, xc, indexing="ij"); RC = np.hypot(XC, YC); inside = RC <= R
    ridx = np.clip(np.searchsorted(d["r"], RC), 0, len(d["r"]) - 1)
    ts = L["t"] if L["s"] > 0 else np.conj(L["t"])
    zs = np.linspace(0.5 * f, 1.5 * f, 401) / 1000.0
    lams_eval = A.lams if A.lams else lam.tolist()
    result = {"P_nm": P, "R_um": R / 1000, "f_um": f / 1000, "NA": NA, "band_na": A.band_na, "lut": L["args"],
              "design_eta": d["eta"].tolist(), "design_eta_phi": d["eta_phi"].tolist(), "lam_design": lam.tolist(),
              "C": d["C"].tolist(), "eval": {}}
    maps = {}
    Lr = np.sqrt(RC ** 2 + f ** 2) - f
    ic_l = int(np.argmin(np.abs(lam - A.center)))
    m0, _ = propagate(np.where(inside, np.exp(-1j * 2 * np.pi / lam[ic_l] * Lr), 0), P / 1000, lam[ic_l] / 1000, R / 1000,
                      f / 1000, A.band_na, zs, inside=inside)
    z_ref = m0["z_focus_um"]; result["z_ref_um"] = z_ref; ideal_peak = {}
    for name, dd in (("ideal", None), ("achromatic", d), ("chromatic_ref", dc)):
        amap = np.where(inside, dd["assign"][ridx], -1) if dd is not None else None
        result["eval"][name] = []
        for lm in lams_eval:
            kk = int(np.argmin(np.abs(lam - lm)))
            tc = (np.where(inside, np.exp(-1j * 2 * np.pi / lam[kk] * Lr), 0) if dd is None
                  else np.where(inside, ts[np.maximum(amap, 0), kk], 0))
            m, mp_ = propagate(tc, P / 1000, lam[kk] / 1000, R / 1000, f / 1000, A.band_na, zs, inside=inside, z_fixed=z_ref)
            if dd is None: ideal_peak[kk] = m["peak_fixed"]
            m["strehl_fixed"] = m["peak_fixed"] / ideal_peak[kk]
            dl = 0.514 * lam[kk] / 1000 / NA
            row = {k_: v for k_, v in m.items()}; row.update(lam=float(lam[kk]), diffraction_fwhm_um=dl)
            result["eval"][name].append(row)
            if name == "achromatic":
                maps[f"{lam[kk]:.0f}"] = mp_
            print(f"  {name:13s} {lam[kk]:.0f} nm: z_f={m['z_focus_um'] / 1000:.3f} mm (design {f / 1e6:.3f})  "
                  f"FWHM={m['fwhm_x_um']:.2f}/{m['fwhm_y_um']:.2f} um (DL {dl:.2f})  eff={m['focusing_efficiency']:.3f}  "
                  f"T={m['transmitted_fraction']:.3f} | at z_ref {z_ref / 1000:.2f} mm: eff {m['eff_fixed']:.3f} "
                  f"Strehl {m['strehl_fixed']:.3f}", flush=True)
        if name == "achromatic":
            np.save(out / "atom_map.npy", amap.astype(np.int32))
    used = sorted(set(d["assign"].tolist()))
    result["atoms_used"] = {int(i): L["atoms"][i].tolist() for i in used}
    result["radial_assign"] = dict(r_nm=d["r"].tolist(), atom=d["assign"].tolist())
    json.dump(result, open(out / "lens.json", "w"), indent=1)
    np.savez_compressed(out / "focal_maps_lpa.npz", **{f"{k}_{q}": v for k, mm in maps.items() for q, v in mm.items()},
                        zs_um=zs * 1000.0)
    np.savez(out / "lut_used.npz", atoms=L["atoms"], t=L["t"], lam=lam, s=L["s"], families=np.array(L["families"]),
             args=json.dumps(L["args"]))
    plots(out, result, maps, L, d)
    print("[achro_tio2] wrote", out)


def plots(out, result, maps, L, d):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f_mm = result["f_um"] / 1000
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, ls in (("achromatic", "-"), ("chromatic_ref", "--"), ("ideal", ":")):
        ev = result["eval"][name]; l_ = [e["lam"] for e in ev]
        ax[0].plot(l_, [e["z_focus_um"] / 1000 for e in ev], ls, marker="o", ms=4, label=name)
        ax[1].plot(l_, [e["focusing_efficiency"] for e in ev], ls, marker="o", ms=4, label=name)
        ax[2].plot(l_, [e["fwhm_x_um"] / e["diffraction_fwhm_um"] for e in ev], ls, marker="o", ms=4, label=name)
    ax[0].axhline(f_mm, color="gray", lw=0.8, ls=":")
    ax[0].set(xlabel="wavelength (nm)", ylabel="focal distance (mm)",
              title=f"D = {2 * result['R_um']:.0f} um, NA = {result['NA']:.4f}")
    ax[1].set(xlabel="wavelength (nm)", ylabel="focusing efficiency")
    ax[2].set(xlabel="wavelength (nm)", ylabel="FWHM / diffraction limit")
    for a in ax: a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "achro_focus.png", dpi=170); plt.close(fig)
    ks = list(maps.keys()); sel = ks[:: max(1, len(ks) // 4)][:4] + ([ks[-1]] if ks[-1] not in ks[:: max(1, len(ks) // 4)][:4] else [])
    fig, ax = plt.subplots(1, len(sel), figsize=(3.2 * len(sel), 3.2))
    for a, k in zip(np.atleast_1d(ax), sel):
        m = maps[k]; x = m["x"]
        a.imshow(m["I"].T / m["I"].max(), extent=[x[0], x[-1], x[0], x[-1]], origin="lower", cmap="inferno")
        a.set(title=f"{k} nm (LPA)", xlabel="x (um)", ylabel="y (um)")
    fig.tight_layout(); fig.savefig(out / "focal_maps_lpa.png", dpi=170); plt.close(fig)
    rr = np.array(result["radial_assign"]["r_nm"]) / 1000; lam = L["lam"]
    dL = (np.sqrt((result["R_um"]) ** 2 + result["f_um"] ** 2) - result["f_um"]) - (np.sqrt(rr ** 2 + result["f_um"] ** 2) - result["f_um"])
    ts = L["t"] if L["s"] > 0 else np.conj(L["t"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    fam = L["atoms"][d["assign"], 0]
    ax[0].scatter(rr, fam, s=1); ax[0].set(xlabel="r (um)", ylabel="atom family", yticks=range(len(L["families"])),
                                           yticklabels=L["families"])
    for kk in range(0, len(lam), max(1, len(lam) // 5)):
        err = np.angle(ts[d["assign"], kk] * np.exp(-1j * (2 * np.pi / lam[kk] * dL * 1000 + d["C"][kk])))
        ax[1].plot(rr, np.degrees(err), lw=0.6, label=f"{lam[kk]:.0f} nm")
    ax[1].set(xlabel="r (um)", ylabel="phase error (deg)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "achro_design.png", dpi=170); plt.close(fig)


def main():
    # the Meep job script launches every line with mpirun: only rank 0 works
    if int(os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0"))) != 0:
        return
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("--runs", nargs="+", required=True); s.add_argument("--name", required=True)
    s.add_argument("--diameter-um", type=float, default=400.0); s.add_argument("--tmin", type=float, default=0.0)
    s.add_argument("--smooth", action="store_true", help="drop atoms with sharp spectral features (see smooth_mask)")
    s.add_argument("--nas", type=float, nargs="+", default=[0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225,
                                                           0.025, 0.03, 0.035, 0.04, 0.05])
    b = sub.add_parser("build"); b.add_argument("--run", required=True); b.add_argument("--name", required=True)
    b.add_argument("--diameter-um", type=float, default=400.0); g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--na", type=float); g.add_argument("--focal-um", type=float)
    b.add_argument("--lams", type=float, nargs="*"); b.add_argument("--center", type=float, default=850.0)
    b.add_argument("--band-na", type=float, default=0.0, help="angular band kept for the focal analysis (in NA); 0 = 2.5 x lens NA")
    b.add_argument("--tmin", type=float, default=0.0)
    b.add_argument("--smooth", action="store_true")
    A = ap.parse_args()
    cmd_scan(A) if A.cmd == "scan" else cmd_build(A)


if __name__ == "__main__":
    main()
