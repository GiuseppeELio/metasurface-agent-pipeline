#!/usr/bin/env python3
"""Multi-wavelength LUT of fourfold-symmetric dielectric pillars on a substrate (polarization-insensitive atoms).

Cell (period P, square lattice): substrate (input) | pillar layer of height H | air (output).
x-polarized plane wave at normal incidence from the substrate; zero-order transmission coefficient into air,
normalized so that |t|^2 is the transmitted power fraction (TORCWA S_parameters with power_norm=True;
checked against the Fresnel coefficient of a bare silica/air interface: |t| = 0.98299).

Atom families (all fourfold symmetric, so t_xx = t_yy and the lens is polarization insensitive):
  circ   (D)        circular pillar, diameter D
  square (W)        square pillar, side W
  ring   (D, d)     circular pillar of diameter D with a centred circular hole d
  sqring (W, w)     square pillar of side W with a centred square hole w
  cross  (L, w)     plus-shaped pillar, arm length L (tip to tip), arm width w
Fabrication rules: every solid feature, hole and gap >= --min-feature (default 80 nm).

  python tools/tio2_lut.py --name tio2_lut_P450 --period 450 --height 1000 --lams 700 1000 20 --order 9
  python tools/tio2_lut.py --name tio2_conv --period 450 --height 1000 --conv-orders 5 7 9 11
Output runs/<name>/lut_<i0>_<i1>.npz : lam, atoms [Na, 3] (family id, p1, p2 in nm), t [Na, K] complex, families, args
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "third_party" / "torcwa"))
import torcwa  # noqa: E402
from ptopt.materials import refractive_index  # noqa: E402

FAMILIES = ["circ", "square", "ring", "sqring", "cross"]


def atom_list(P, m, step, families, step2=None):
    step2 = step2 or step
    big = P - m                       # largest outer size (gap >= m)
    out = []
    s = lambda a, b: np.arange(a, b + 1e-9, step)
    s2 = lambda a, b: np.arange(a, b + 1e-9, step2)
    for fam in families:
        if fam == "circ":   out += [(0, D, 0) for D in s(m, big)]
        if fam == "square": out += [(1, W, 0) for W in s(m, big)]
        if fam == "ring":   out += [(2, D, d) for D in s2(3 * m, big) for d in s2(m, D - 2 * m)]
        if fam == "sqring": out += [(3, W, w) for W in s2(3 * m, big) for w in s2(m, W - 2 * m)]
        if fam == "cross":  out += [(4, L, w) for L in s2(2 * m, big) for w in s2(m, L - m)]
    return np.array(out, float)


def raster(X, Y, P, a):
    """Anti-aliased fill fraction (1 = pillar material) of atom a = (family, p1, p2); pixel size enters via px."""
    px = float(X[1, 0] - X[0, 0]); x = X - P / 2; y = Y - P / 2
    fam, p1, p2 = int(a[0]), a[1], a[2]
    cl = lambda sd: torch.clamp(0.5 - sd / px, 0.0, 1.0)            # sd: signed distance (<0 inside)
    r = torch.sqrt(x ** 2 + y ** 2); cheb = torch.maximum(x.abs(), y.abs())
    if fam == 0: return cl(r - p1 / 2)
    if fam == 1: return cl(cheb - p1 / 2)
    if fam == 2: return cl(r - p1 / 2) - cl(r - p2 / 2)
    if fam == 3: return cl(cheb - p1 / 2) - cl(cheb - p2 / 2)
    if fam == 4:
        armx = torch.maximum(x.abs() - p1 / 2, y.abs() - p2 / 2); army = torch.maximum(y.abs() - p1 / 2, x.abs() - p2 / 2)
        return cl(torch.minimum(armx, army))
    raise ValueError(fam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--period", type=float, default=450.0)
    ap.add_argument("--height", type=float, default=1000.0)
    ap.add_argument("--pillar", default='{"material": "TiO2_ALD_Siefke"}', help="material spec (JSON)")
    ap.add_argument("--substrate", default='{"material": "SiO2_Malitson"}')
    ap.add_argument("--lams", type=float, nargs=3, default=[700, 1000, 20])
    ap.add_argument("--min-feature", type=float, default=80.0)
    ap.add_argument("--step", type=float, default=10.0, help="geometry step of the one-parameter families (nm)")
    ap.add_argument("--step2", type=float, default=20.0, help="geometry step of ring, sqring and cross (nm)")
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--order", type=int, default=9)
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conv-orders", type=int, nargs="*", default=[])
    ap.add_argument("--atom-start", type=int, default=0); ap.add_argument("--atom-stop", type=int, default=10 ** 9)
    a = ap.parse_args()
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) or a.device == "cuda" else "cpu")
    if dev.type == "cuda": torch.backends.cuda.matmul.allow_tf32 = False
    out = HERE / "runs" / a.name; out.mkdir(parents=True, exist_ok=True)
    spec_p, spec_s = json.loads(a.pillar), json.loads(a.substrate)
    P, N = a.period, a.grid
    x = (torch.arange(N, device=dev, dtype=torch.float64) + 0.5) * P / N
    X, Y = torch.meshgrid(x, x, indexing="ij")
    lams = np.arange(a.lams[0], a.lams[1] + 1e-9, a.lams[2])
    atoms = atom_list(P, a.min_feature, a.step, [f for f in FAMILIES if f in a.families], a.step2)

    def t00(lam, order, fill):
        ns = refractive_index(spec_s, lam).real; ep = refractive_index(spec_p, lam) ** 2
        s = torcwa.rcwa(freq=1 / lam, order=[order, order], L=[P, P], dtype=torch.complex128, device=dev)
        s.add_input_layer(eps=ns ** 2); s.add_output_layer(eps=1.0); s.set_incident_angle(0., 0.)
        s.add_layer(thickness=a.height, eps=1.0 + (ep - 1.0) * fill.to(torch.complex128))
        s.solve_global_smatrix()
        tx = s.S_parameters(orders=[0, 0], direction="forward", port="transmission", polarization="xx", ref_order=[0, 0])
        return complex(tx.cpu())                      # power-normalized amplitude (power_norm=True default)

    if a.conv_orders:
        pick = [atoms[i] for i in np.linspace(0, len(atoms) - 1, 8).astype(int)]
        lc = [lams[0], lams[len(lams) // 2], lams[-1]]
        res = {"atoms": [p.tolist() for p in pick], "lams": [float(l) for l in lc], "t": {}, "sec": {}}
        for o in a.conv_orders:
            t0 = time.time()
            tt = np.array([[t00(l, o, raster(X, Y, P, g)) for l in lc] for g in pick])
            res["t"][o] = np.c_[tt.real.ravel(), tt.imag.ravel()].tolist(); res["sec"][o] = (time.time() - t0) / tt.size
            print(f"  order {o}: {res['sec'][o]:.2f} s/sim  T = {np.round(np.abs(tt) ** 2, 3).ravel().tolist()}", flush=True)
        ref = np.array(res["t"][a.conv_orders[-1]]); ref = ref[:, 0] + 1j * ref[:, 1]
        for o in a.conv_orders[:-1]:
            tt = np.array(res["t"][o]); tt = tt[:, 0] + 1j * tt[:, 1]
            print(f"  order {o} vs {a.conv_orders[-1]}: max dphase {np.degrees(np.abs(np.angle(tt / ref))).max():.2f} deg, "
                  f"max dT {np.abs(np.abs(tt) ** 2 - np.abs(ref) ** 2).max():.4f}", flush=True)
        json.dump(res, open(out / "conv.json", "w")); return

    i0, i1 = max(0, a.atom_start), min(len(atoms), a.atom_stop)
    t = np.zeros((i1 - i0, len(lams)), complex); T0 = time.time()
    for j, ia in enumerate(range(i0, i1)):
        fill = raster(X, Y, P, atoms[ia])
        for k, lam in enumerate(lams):
            t[j, k] = t00(lam, a.order, fill)
        if (j + 1) % 10 == 0 or j == i1 - i0 - 1:
            el = time.time() - T0
            print(f"  atom {j + 1}/{i1 - i0}  {el / 60:.1f} min, ETA {el / (j + 1) * (i1 - i0 - j - 1) / 60:.1f} min", flush=True)
    np.savez(out / f"lut_{i0:04d}_{i1:04d}.npz", lam=lams, atoms=atoms[i0:i1], t=t, families=np.array(FAMILIES),
             args=json.dumps(vars(a)))
    print(f"[tio2_lut] {dev} saved {out}/lut_{i0:04d}_{i1:04d}.npz ({i1 - i0} atoms x {len(lams)} lams) "
          f"in {(time.time() - T0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
