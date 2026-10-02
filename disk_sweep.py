#!/usr/bin/env python3
"""Stage A: parametric sweep of a Si disk (radius x height) in a square cell -> T and phase of t00.

  python disk_sweep.py --name disk_sio2_P700 --period 700 --heights 500 600 700 800 900 1000 \
      --radii 40 340 10 --n-sub 1.444 --order 10
Writes runs/<name>/sweep.json and sweep.png (T and unwrapped phase vs R for each h) + a coverage table.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "third_party" / "torcwa"))
import torcwa  # noqa: E402
from ptopt.device import describe, pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--period", type=float, default=700.0)
ap.add_argument("--heights", type=float, nargs="+", required=True)
ap.add_argument("--radii", type=float, nargs=3, default=[40, 340, 10], metavar=("START", "STOP", "STEP"))
ap.add_argument("--wavelength", type=float, default=1550.0)
ap.add_argument("--n-core", type=float, default=3.476)
ap.add_argument("--n-sub", type=float, default=1.444)
ap.add_argument("--n-sup", type=float, default=1.0)
ap.add_argument("--order", type=int, default=10)
ap.add_argument("--grid", type=int, default=140)
ap.add_argument("--device", default="auto")
ap.add_argument("--tmin", type=float, default=0.9)
a = ap.parse_args()
dev = pick_device(a.device)
if dev.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
out = HERE / "runs" / a.name; out.mkdir(parents=True, exist_ok=True)
P, N = a.period, a.grid
x = (torch.arange(N, device=dev) + 0.5) * P / N
X, Y = torch.meshgrid(x, x, indexing="ij")
radii = np.arange(a.radii[0], a.radii[1] + 1e-9, a.radii[2])
print(f"[disk_sweep] {describe(dev)} P={P} order={a.order} heights={a.heights} radii {radii[0]}..{radii[-1]}", flush=True)


def t00(rho, h):
    sim = torcwa.rcwa(freq=1 / a.wavelength, order=[a.order, a.order], L=[P, P], dtype=torch.complex64, device=dev)
    sim.add_input_layer(eps=a.n_sub ** 2); sim.add_output_layer(eps=a.n_sup ** 2)
    sim.set_incident_angle(0., 0.)
    sim.add_layer(thickness=h, eps=rho * a.n_core ** 2 + (1 - rho))
    sim.solve_global_smatrix()
    return complex(sim.S_parameters(orders=[0, 0], direction="forward", port="transmission",
                                    polarization="xx", ref_order=[0, 0]).cpu())


res = {"args": vars(a), "heights": {}}
t0 = time.time()
for h in a.heights:
    ref = t00(torch.zeros_like(X), h)  # empty cell: phase reference
    rows = []
    for R in radii:
        rho = (((X - P / 2) ** 2 + (Y - P / 2) ** 2) <= R ** 2).float()
        t = t00(rho, h) / (ref / abs(ref))
        rows.append((float(R), abs(t) ** 2, math.atan2(t.imag, t.real)))
    ph = np.unwrap([r[2] for r in rows])
    T = np.array([r[1] for r in rows])
    ok = T >= a.tmin
    cov = float(np.degrees(ph[ok].max() - ph[ok].min())) if ok.any() else 0.0
    res["heights"][str(h)] = {"R": radii.tolist(), "T": T.tolist(), "phase_deg": np.degrees(ph).tolist(),
                              "phase_span_deg": float(np.degrees(ph.max() - ph.min())),
                              f"phase_span_T>{a.tmin}_deg": cov, "T_min": float(T.min())}
    print(f"h={h:6.0f}  span={np.degrees(ph.max()-ph.min()):6.1f} deg  span(T>{a.tmin})={cov:6.1f} deg  Tmin={T.min():.2f}  ({time.time()-t0:.0f}s)", flush=True)
(out / "sweep.json").write_text(json.dumps(res, indent=1))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for h, d in res["heights"].items():
        ax[0].plot(d["R"], d["T"], label=f"h={float(h):.0f}"); ax[1].plot(d["R"], d["phase_deg"], label=f"h={float(h):.0f}")
    ax[0].set(xlabel="R (nm)", ylabel="T (0 order)", ylim=(0, 1.02)); ax[1].set(xlabel="R (nm)", ylabel="phase (deg, unwrapped)")
    ax[1].axhline(360, ls=":", c="k"); ax[0].legend(fontsize=8); [x_.grid(alpha=.3) for x_ in ax]
    fig.tight_layout(); fig.savefig(out / "sweep.png", dpi=140)
except ImportError:
    pass
print("[disk_sweep] DONE", out, flush=True)
