#!/usr/bin/env python3
"""Achromatic metalens from a multi-wavelength meta-atom LUT: design core (design()) shared by tools/achro_tio2.py.
The scan/build commands below expect the bilayer-stack LUT format (lut.npz with lam, radii, t[nb, nt, K]).

Design rule (dispersion engineering, e.g. Chen et al. 2018, Shrestha et al. 2018): at every wavelength the lens must
impose, up to a free *global* phase C(lambda),
    psi_req(r, lambda) = (2 pi / lambda) * [L(R) - L(r)] + C(lambda),    L(r) = sqrt(r^2 + f^2) - f   (output: air)
where psi = s * arg(t) is the retardation (s fixed so that a taller/fuller cell retards more).
For each radius the atom that maximizes  sum_k Re( t_a(l_k) e^{-i s psi_req} ) (coherent overlap) is selected;
C(lambda_k) and the assignment are optimized alternately from a grid of starts.
All atoms of one lens share the stack (P, hb, gap, ht): every LUT file is analysed on its own.

  python tools/achro_lens.py scan  --luts runs/vuv_A/lut.npz runs/vuv_B/lut.npz --diameter-um 10 --name achro_scan
  python tools/achro_lens.py build --lut runs/vuv_B/lut.npz --diameter-um 10 --na 0.08 --name achro_lens_B [--lams 120 140 160 180 200]

scan : coherent efficiency  eta_k = |<t e^{-i psi_req}>_area|^2  (T included) and phase-only eta_k^phi, vs NA, per stack.
       max NA = largest NA with min_k eta_k^phi >= 0.8 (Marechal).
build: square-cell layout, LPA field t_a(lambda) per cell, vector angular-spectrum focusing (ptopt.lensprop) at each
       wavelength; compares with a single-wavelength (chromatic) lens made of the same atoms at the centre wavelength.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def load(path):
    d = np.load(path, allow_pickle=True)
    lam, radii, t = d["lam"], d["radii"], d["t"]
    args = json.loads(str(d["args"]))
    nb, nt, K = t.shape
    atoms = [(radii[i], radii[j]) for i in range(nb) for j in range(nt)]
    T = t.reshape(nb * nt, K)
    # retardation sign from the empty cell: arg t0 ~ s * 2 pi H / lambda  ->  d arg / d lambda = -s * ...
    ph0 = np.unwrap(np.angle(t[0, 0]))
    s = -np.sign(np.polyfit(lam, ph0, 1)[0]) if len(lam) > 1 else 1.0
    return dict(lam=lam.astype(float), atoms=np.array(atoms), t=T, s=s, args=args, path=str(path))


def radial_grid(R, P, n=200):
    r = (np.arange(n) + 0.5) * R / n
    w = 2 * np.pi * r * (R / n)
    return r, w / w.sum()


def design(L, R, f, n_starts=12, iters=6, sel_idx=None, rng=None):
    """Return assignment (atom index per radial sample), C(lambda), eta (with T) and eta_phi (|t|=1)."""
    lam = L["lam"]; k = 2 * np.pi / lam
    r, w = radial_grid(R, None)
    dL = (np.sqrt(R ** 2 + f ** 2) - f) - (np.sqrt(r ** 2 + f ** 2) - f)  # >= 0, max at centre
    ts = L["t"] if L["s"] > 0 else np.conj(L["t"])                       # t with arg = retardation
    if sel_idx is not None:
        ts = ts[:, sel_idx]; lam = lam[sel_idx]; k = k[sel_idx]
    Ereq0 = np.exp(1j * np.outer(dL, k))                                  # [Nr, K]
    tnorm = ts / np.maximum(np.abs(ts), 1e-12)
    best = None
    dmax = dL.max() + 400.0
    for d0 in np.linspace(0, dmax, n_starts):
        for c0 in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
            C = c0 + k * d0
            for _ in range(iters):
                Ereq = Ereq0 * np.exp(1j * C)[None, :]
                merit = np.real(ts @ np.conj(Ereq).T)                    # [Na, Nr]
                a = np.argmax(merit, axis=0)
                ov = np.sum(w[:, None] * ts[a] * np.conj(Ereq0), axis=0)  # [K]
                C = np.angle(ov)
            ov = np.sum(w[:, None] * ts[a] * np.conj(Ereq0 * np.exp(1j * C)[None, :]), axis=0)
            ovp = np.sum(w[:, None] * tnorm[a] * np.conj(Ereq0 * np.exp(1j * C)[None, :]), axis=0)
            eta, etap = np.abs(ov) ** 2, np.abs(ovp) ** 2
            score = etap.min() + 0.2 * eta.mean()
            if best is None or score > best[0]:
                best = (score, a, C, eta, etap)
    _, a, C, eta, etap = best
    return dict(r=r, assign=a, C=C, eta=eta, eta_phi=etap)


def cmd_scan(A):
    out = HERE / "runs" / A.name; out.mkdir(parents=True, exist_ok=True)
    R = A.diameter_um * 500.0
    NAs = np.array(A.nas)
    res = {}
    for p in A.luts:
        L = load(p); key = Path(p).parent.name; rows = []
        for na in NAs:
            f = R * math.sqrt(1 - na ** 2) / na
            d = design(L, R, f)
            rows.append(dict(na=float(na), f_um=f / 1000, eta=d["eta"].tolist(), eta_phi=d["eta_phi"].tolist()))
        ok = [r["na"] for r in rows if min(r["eta_phi"]) >= 0.8]
        res[key] = dict(args=L["args"], lam=L["lam"].tolist(), rows=rows, max_na=max(ok) if ok else None,
                        T_mean=float(np.mean(np.abs(L["t"]) ** 2)))
        print(f"{key}: max NA (min eta_phi>=0.8) = {res[key]['max_na']}  | " +
              "  ".join(f"NA{r['na']:.2f}: phi {min(r['eta_phi']):.2f} / T {np.mean(r['eta']):.2f}" for r in rows), flush=True)
    json.dump(res, open(out / "scan.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, v in res.items():
        a = v["args"]; lab = f"P{a['period']:.0f} {a['hb']:.0f}/{a['gap']:.0f}/{a['ht']:.0f}"
        na = [r["na"] for r in v["rows"]]
        ax[0].plot(na, [min(r["eta_phi"]) for r in v["rows"]], marker="o", ms=3, label=lab)
        ax[1].plot(na, [np.mean(r["eta"]) for r in v["rows"]], marker="o", ms=3, label=lab)
    ax[0].axhline(0.8, color="gray", lw=0.8, ls=":")
    ax[0].set(xlabel="NA", ylabel="min over band of phase-only efficiency", title=f"D = {A.diameter_um} um, 120-200 nm")
    ax[1].set(xlabel="NA", ylabel="band-mean coherent efficiency (T incl.)")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "scan.png", dpi=170)
    print("[achro] wrote", out / "scan.json", out / "scan.png")


def cmd_build(A):
    from ptopt.lensprop import VectorASM
    out = HERE / "runs" / A.name; out.mkdir(parents=True, exist_ok=True)
    L = load(A.lut); P = L["args"]["period"]; R = A.diameter_um * 500.0
    f = A.focal_um * 1000 if A.focal_um else R * math.sqrt(1 - A.na ** 2) / A.na
    lam = L["lam"]
    d = design(L, R, f)
    # chromatic reference: same atoms, designed for the centre wavelength only
    ic = int(np.argmin(np.abs(lam - A.center)))
    dc = design(L, R, f, sel_idx=[ic])
    ncell = int(round(2 * R / P)) | 1
    xc = (np.arange(ncell) - ncell // 2) * P
    XC, YC = np.meshgrid(xc, xc, indexing="ij"); RC = np.hypot(XC, YC)
    inside = RC <= R
    ridx = np.clip(np.searchsorted(d["r"], RC), 0, len(d["r"]) - 1)
    ts = L["t"] if L["s"] > 0 else np.conj(L["t"])
    px = A.px_per_cell; dx = P / px
    lams_eval = A.lams if A.lams else lam.tolist()
    result = {"P_nm": P, "R_nm": R, "f_um": f / 1000, "NA": R / math.hypot(R, f), "stack": L["args"],
              "design_eta": d["eta"].tolist(), "design_eta_phi": d["eta_phi"].tolist(), "eval": {}}
    zs = np.linspace(0.3 * f, 1.7 * f, 400) / 1000.0
    Lr = np.sqrt(RC ** 2 + f ** 2) - f
    for name, dd in (("achromatic", d), ("chromatic_ref", dc), ("ideal", None)):
        amap = np.where(inside, dd["assign"][ridx], -1) if dd is not None else None
        result["eval"][name] = []
        for lm in lams_eval:
            kk = int(np.argmin(np.abs(lam - lm)))
            if dd is None:   # ideal continuous hyperbolic lens, |t| = 1 (reference for the low-Fresnel-number focal shift)
                tc = np.where(inside, np.exp(-1j * 2 * np.pi / lam[kk] * Lr), 0)
            else:
                tc = np.where(inside, ts[np.maximum(amap, 0), kk], 0)
            # pillar-only field normalisation: incident unit amplitude over aperture
            E = np.kron(tc, np.ones((px, px)))
            prop = VectorASM(E, np.zeros_like(E), dx / 1000.0, lam[kk] / 1000.0, 1.0, pad=A.pad)
            Pinc = 0.5 * np.count_nonzero(inside) * (P / 1000.0) ** 2
            m = prop.metrics(zs, Pinc)
            ideal_fwhm = 0.514 * lam[kk] / 1000 / (R / math.hypot(R, f))
            row = {k_: v for k_, v in m.items() if not k_.startswith("_")}
            row.update(lam=float(lam[kk]), diffraction_fwhm_um=ideal_fwhm)
            result["eval"][name].append(row)
            print(f"  {name:13s} lam={lam[kk]:.0f}: z_f={m['z_focus_um']:.2f} um (design {f/1000:.2f})  FWHM={m['fwhm_x_um']:.3f} "
                  f"(DL {ideal_fwhm:.3f})  eff={m['focusing_efficiency']:.3f}", flush=True)
        if name == "achromatic":
            np.save(out / "atom_map.npy", amap)
    result["atoms_used"] = [[float(L["atoms"][i][0]), float(L["atoms"][i][1])] for i in sorted(set(d["assign"].tolist()))]
    json.dump(result, open(out / "lens.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, ls in (("achromatic", "-"), ("chromatic_ref", "--"), ("ideal", ":")):
        ev = result["eval"][name]; l_ = [e["lam"] for e in ev]
        ax[0].plot(l_, [e["z_focus_um"] for e in ev], ls, marker="o", ms=4, label=name)
        ax[1].plot(l_, [e["focusing_efficiency"] for e in ev], ls, marker="o", ms=4, label=name)
        ax[2].plot(l_, [e["fwhm_x_um"] / e["diffraction_fwhm_um"] for e in ev], ls, marker="o", ms=4, label=name)
    ax[0].axhline(f / 1000, color="gray", lw=0.8, ls=":")
    ax[0].set(xlabel="wavelength (nm)", ylabel="focal distance (um)", title=f"D={2*R/1000:.0f} um, NA={result['NA']:.3f}")
    ax[1].set(xlabel="wavelength (nm)", ylabel="focusing efficiency")
    ax[2].set(xlabel="wavelength (nm)", ylabel="FWHM / diffraction limit")
    for a_ in ax: a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "achro_focus.png", dpi=170)
    rr = d["r"] / 1000
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    at = L["atoms"][d["assign"]]
    ax[0].plot(rr, at[:, 0], label="R bottom"); ax[0].plot(rr, at[:, 1], "--", label="R top")
    ax[0].set(xlabel="r (um)", ylabel="pillar radius (nm)"); ax[0].legend(fontsize=7)
    dL = (np.sqrt(R ** 2 + f ** 2) - f) - (np.sqrt(d["r"] ** 2 + f ** 2) - f)
    for kk in range(0, len(lam), max(1, len(lam) // 5)):
        err = np.angle(ts[d["assign"], kk] * np.exp(-1j * (2 * np.pi / lam[kk] * dL + d["C"][kk])))
        ax[1].plot(rr, np.degrees(err), lw=1, label=f"{lam[kk]:.0f} nm")
    ax[1].set(xlabel="r (um)", ylabel="phase error (deg)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "achro_design.png", dpi=170)
    print("[achro] wrote", out)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("--luts", nargs="+", required=True); s.add_argument("--name", required=True)
    s.add_argument("--diameter-um", type=float, default=10.0)
    s.add_argument("--nas", type=float, nargs="+", default=[0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3])
    b = sub.add_parser("build"); b.add_argument("--lut", required=True); b.add_argument("--name", required=True)
    b.add_argument("--diameter-um", type=float, default=10.0); g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--na", type=float); g.add_argument("--focal-um", type=float)
    b.add_argument("--lams", type=float, nargs="*"); b.add_argument("--center", type=float, default=160.0)
    b.add_argument("--px-per-cell", type=int, default=4); b.add_argument("--pad", type=float, default=1.5)
    A = ap.parse_args()
    cmd_scan(A) if A.cmd == "scan" else cmd_build(A)


if __name__ == "__main__":
    main()
