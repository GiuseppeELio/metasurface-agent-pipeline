#!/usr/bin/env python3
"""Topology optimization with TORCWA (generalizes TORCWA Example 6).

Usage:
  python run_topopt.py --config configs/example6_diffraction.json
  python run_topopt.py --config configs/x.json --iters 20 --device cpu --order 5 3   # quick test
  python run_topopt.py --resume runs/<run>/checkpoint.pt                            # resume

Outputs in runs/<name>_<timestamp>/ : config.json, history.csv, checkpoint.pt,
rho_final.npy, rho_history.npz, summary.json, design.png, fom.png
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "third_party" / "torcwa"))  # optional local copy; otherwise `pip install torcwa`

from ptopt.device import describe, pick_device  # noqa: E402
from ptopt.problem import Problem  # noqa: E402

DEFAULT_OPT = dict(iters=800, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8, beta_max=1000.0,
                   blur_radius_nm=20.0, symmetry="y", seed=333, checkpoint_every=50,
                   history_every=10, log_every=1)


def gaussian_filter_fft(nx, ny, L, radius, dtype, device):
    dx, dy = L[0] / nx, L[1] / ny
    xa = (torch.arange(nx, dtype=dtype, device=device) - (nx - 1) / 2) * dx
    ya = (torch.arange(ny, dtype=dtype, device=device) - (ny - 1) / 2) * dy
    X, Y = torch.meshgrid(xa, ya, indexing="ij")
    g = torch.exp(-(X ** 2 + Y ** 2) / radius ** 2)
    g = g / g.sum()
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(g)))


def blur(rho, g_fft):
    r = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(rho)))
    return torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(r * g_fft))))


def project(rho_bar, beta, eta=0.5):
    """Threshold projection (Wang, Lazarov & Sigmund 2011). eta<0.5 -> dilated, eta>0.5 -> eroded."""
    if eta == 0.5:
        return 0.5 + torch.tanh(2 * beta * rho_bar - beta) / (2 * math.tanh(beta))
    b = beta * 2  # same slope as the eta=0.5 formula
    return (math.tanh(b * eta) + torch.tanh(b * (rho_bar - eta))) / (math.tanh(b * eta) + math.tanh(b * (1 - eta)))


def robust_designs(rho_bar, beta, etas, geo=None):
    """Designs over which the FoM is soft-minimized.
    threshold mode: project the filtered field at several thresholds (edge shift depends on field slope).
    geometric mode (geo=(g2_fft, eta_dil, eta_ero)): re-filter the NOMINAL projected design with a second
    Gaussian and threshold it -> edge shift of exactly +/- d nm when the nominal design is binary."""
    if geo is None:
        return [project(rho_bar, beta, e) for e in etas]
    g2, eta_d, eta_e = geo
    nom = project(rho_bar, beta)
    nb = blur(nom, g2)
    return [project(nb, beta, eta_d), nom, project(nb, beta, eta_e)]


def robust_fom(prob, designs, k):
    """Soft-min of the FoM over eroded/nominal/dilated designs (fabrication robustness)."""
    foms = torch.stack([prob.fom(r) for r in designs])
    if len(designs) == 1:
        return foms[0], foms
    return -torch.logsumexp(-k * foms, 0) / k + math.log(len(designs)) / k, foms


def _rot(X, Y, deg):
    t = math.radians(deg)
    return X * math.cos(t) + Y * math.sin(t), -X * math.sin(t) + Y * math.cos(t)


# Initial geometries (centered in the cell). Sizes in nm. All return a 0/1 mask.
INIT_SHAPES = {
    "disk": lambda X, Y, s: X ** 2 + Y ** 2 <= s["radius_nm"] ** 2,
    "ellipse": lambda X, Y, s: (lambda u, v: (u / s["a_nm"]) ** 2 + (v / s["b_nm"]) ** 2 <= 1)(*_rot(X, Y, s.get("angle_deg", 0))),
    "square": lambda X, Y, s: (lambda u, v: (u.abs() <= s["side_nm"] / 2) & (v.abs() <= s["side_nm"] / 2))(*_rot(X, Y, s.get("angle_deg", 0))),
    "ring": lambda X, Y, s: (X ** 2 + Y ** 2 <= s["radius_nm"] ** 2) & (X ** 2 + Y ** 2 >= s["inner_nm"] ** 2),
    "cross": lambda X, Y, s: ((X.abs() <= s["arm_nm"] / 2) & (Y.abs() <= s["length_nm"] / 2)) | ((Y.abs() <= s["arm_nm"] / 2) & (X.abs() <= s["length_nm"] / 2)),
    # clover (four-leaf): 4 disks of radius r at distance d from the center (on the axes) + central disk
    "clover": lambda X, Y, s: sum(((X - dx) ** 2 + (Y - dy) ** 2 <= s["lobe_nm"] ** 2).float() for dx, dy in
                                  [(s["offset_nm"], 0), (-s["offset_nm"], 0), (0, s["offset_nm"]), (0, -s["offset_nm"]), (0, 0)]) > 0,
    # equilateral triangle with circumradius R, one vertex along +y
    "triangle": lambda X, Y, s: (lambda R: torch.stack([(X * math.cos(a) + Y * math.sin(a)) <= R / 2 for a in
                                                         (math.radians(-90), math.radians(30), math.radians(150))]).all(0))(s["radius_nm"]),
}


def symmetrize(rho, sym):
    if sym == "d4":  # square cell, polarization-independent at normal incidence: mirrors x, y and diagonal
        rho = (rho + torch.flip(rho, dims=[0])) / 2
        rho = (rho + torch.flip(rho, dims=[1])) / 2
        return (rho + rho.transpose(0, 1)) / 2
    if sym in ("y", "xy"):
        rho = (rho + torch.flip(rho, dims=[1])) / 2
    if sym in ("x", "xy"):
        rho = (rho + torch.flip(rho, dims=[0])) / 2
    return rho


def save_plots(out, rho_final, hist, cfg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    L = cfg["period_nm"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(rho_final.T, origin="lower", extent=[0, L[0], 0, L[1]], cmap="gray_r", vmin=0, vmax=1)
    ax.set(xlabel="x (nm)", ylabel="y (nm)", title=f"{cfg['name']} – design")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out / "design.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([h["iter"] for h in hist], [h["fom"] for h in hist])
    ax.set(xlabel="iteration", ylabel="FoM", title="Convergence"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fom.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--resume")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--order", type=int, nargs=2)
    ap.add_argument("--grid", type=int, nargs=2)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--outdir", default=str(HERE / "runs"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--init", help="warm start: checkpoint.pt (raw rho) or .npy density of a previous run")
    ap.add_argument("--set", action="append", default=[], metavar="key.sub=value",
                    help="override any config field, e.g. --set layer.thickness_nm=800 --set optimizer.lr=0.01")
    a = ap.parse_args()

    ckpt = None
    if a.resume:
        ckpt = torch.load(a.resume, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        out = Path(a.resume).parent
    else:
        cfg = json.loads(Path(a.config).read_text())
        for kv in a.set:
            k, v = kv.split("=", 1)
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            node = cfg
            *path, last = k.split(".")
            for pth in path:
                node = node.setdefault(pth, {})
            node[last] = v
        if a.order: cfg["order"] = a.order
        if a.grid: cfg["grid"] = a.grid
        opt_over = {"iters": a.iters, "seed": a.seed}
        cfg["optimizer"] = {**DEFAULT_OPT, **cfg.get("optimizer", {}), **{k: v for k, v in opt_over.items() if v is not None}}
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job = os.environ.get("SLURM_JOB_ID", "")
        name = "_".join(x for x in [cfg["name"], a.tag, stamp, job] if x)
        out = Path(a.outdir) / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
    o = cfg["optimizer"]

    device = pick_device(a.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False  # A100: keep full FP32 accuracy
        torch.backends.cudnn.allow_tf32 = False
    prob = Problem(cfg, device)
    gd = prob.geo_dtype
    for i, lam in enumerate(prob.wavelengths):
        po = prob.propagating_orders(i, "out")
        print(f"[ptopt] lambda={lam} nm: {len(po)} propagating T orders: " +
              ", ".join(f"{o}@{a:.1f}deg" for o, a in po), flush=True)
    print(f"[ptopt] run={out.name} device={describe(device)} order={cfg['order']} grid={cfg['grid']} "
          f"lambda={prob.wavelengths} iters={o['iters']}", flush=True)

    nx, ny = prob.nx, prob.ny
    etas = [float(e) for e in o.get("robust_eta", [0.5])]
    geo = None
    if o.get("robust_mode") == "geometric":
        d, r2 = float(o["edge_shift_nm"]), float(o.get("robust_blur_nm", 2.5 * float(o["edge_shift_nm"])))
        eta_d = 0.5 * (1 - math.erf(d / r2))  # blurred step exp(-r^2/R^2): profile 0.5*erfc(x/R)
        geo = (gaussian_filter_fft(prob.nx, prob.ny, prob.L, r2, prob.geo_dtype, device), eta_d, 1 - eta_d)
        etas = [eta_d, 0.5, 1 - eta_d]
        print(f"[ptopt] geometric robustness: +/-{d} nm edge shift (blur {r2} nm, eta {eta_d:.3f}/{1-eta_d:.3f})", flush=True)
    g_fft = gaussian_filter_fft(nx, ny, prob.L, o["blur_radius_nm"], gd, device)
    iters = o["iters"]
    b0 = float(o.get("beta_start", 1.0))
    betas = np.exp(np.log(b0) + np.arange(iters) * (np.log(o["beta_max"]) - np.log(b0)) / iters)
    lrs = o["lr"] * 0.5 * (1 + np.cos(np.arange(iters) * np.pi / iters))

    if ckpt:
        rho = ckpt["rho"].to(device); m = ckpt["m"].to(device); v = ckpt["v"].to(device)
        start, hist = ckpt["iter"] + 1, ckpt["history"]
        rho_hist = dict(np.load(out / "rho_history.npz")) if (out / "rho_history.npz").exists() else {}
    else:
        torch.manual_seed(o["seed"])
        if a.init:
            src = torch.load(a.init, map_location="cpu", weights_only=False)["rho"] if a.init.endswith(".pt") \
                else torch.as_tensor(np.load(a.init))
            rho = src.to(device=device, dtype=gd)
            assert rho.shape == (nx, ny), f"init shape {tuple(rho.shape)} != grid {(nx, ny)}"
            cfg["init_from"] = a.init
            (out / "config.json").write_text(json.dumps(cfg, indent=2))
            print(f"[ptopt] warm start from {a.init}", flush=True)
        elif cfg.get("init_shape", {}).get("type") in INIT_SHAPES:
            sh = cfg["init_shape"]
            xa = (torch.arange(nx, dtype=gd, device=device) + 0.5) * prob.L[0] / nx - prob.L[0] / 2
            ya = (torch.arange(ny, dtype=gd, device=device) + 0.5) * prob.L[1] / ny - prob.L[1] / 2
            X, Y = torch.meshgrid(xa, ya, indexing="ij")
            rho = INIT_SHAPES[sh["type"]](X, Y, sh).to(gd)
            noise = float(sh.get("noise", 0.1))
            rho = (1 - noise) * rho + noise * torch.rand_like(rho)
            rho = symmetrize(rho, o["symmetry"])
            print(f"[ptopt] init shape: {sh} (fill {float((rho > 0.5).float().mean()):.2f})", flush=True)
        elif False:
            # start from a solid disk (raw density 1 inside), optionally with a little noise to break the symmetry-trap
            R = float(cfg["init_shape"]["radius_nm"])
            xa = (torch.arange(nx, dtype=gd, device=device) + 0.5) * prob.L[0] / nx
            ya = (torch.arange(ny, dtype=gd, device=device) + 0.5) * prob.L[1] / ny
            X, Y = torch.meshgrid(xa, ya, indexing="ij")
            rho = (((X - prob.L[0] / 2) ** 2 + (Y - prob.L[1] / 2) ** 2) <= R ** 2).to(gd)
            noise = float(cfg["init_shape"].get("noise", 0.1))
            rho = (1 - noise) * rho + noise * torch.rand_like(rho)
            rho = symmetrize(rho, o["symmetry"])
            print(f"[ptopt] init: disk R={R} nm (+{noise} noise)", flush=True)
        else:
            rho = torch.rand((nx, ny), dtype=gd, device=device)
            rho = blur(symmetrize(rho, o["symmetry"]), g_fft)
        m = torch.zeros_like(rho); v = torch.zeros_like(rho)
        start, hist, rho_hist = 0, [], {}

    t0 = time.time()
    for it in range(start, iters):
        rho.requires_grad_(True)
        rho_b = blur(rho, g_fft)
        rho_t = project(rho_b, betas[it])
        # robust formulation only after a warm-up: at start rho_bar ~0.5 everywhere and eroded/dilated
        # designs are empty/full (zero FoM, no useful gradient)
        active = it >= int(o.get("robust_start_frac", 0.0) * iters)
        if geo is not None and active:
            designs = robust_designs(rho_b, betas[it], None, geo)
        else:
            designs = robust_designs(rho_b, betas[it], etas if active else [0.5])
        fom, foms = robust_fom(prob, designs, o.get("softmin_k", 10.0))
        fom.backward()
        with torch.no_grad():
            grad = rho.grad; rho.grad = None
            f = float(fom.detach().cpu())
            m = o["beta1"] * m + (1 - o["beta1"]) * grad
            v = o["beta2"] * v + (1 - o["beta2"]) * grad ** 2
            mh = m / (1 - o["beta1"] ** (it + 1)); vh = v / (1 - o["beta2"] ** (it + 1))
            rho = rho + lrs[it] * mh / torch.sqrt(vh + o["eps"])  # gradient ASCENT (maximize)
            rho = symmetrize(rho.clamp(0, 1), o["symmetry"])
            el = time.time() - t0
            row = {"iter": it, "fom": f, "beta": float(betas[it]), "lr": float(lrs[it]), "t_s": round(el, 2)}
            if len(foms) > 1:
                row.update({f"fom_eta{e}": float(v) for e, v in zip(etas, foms.detach().cpu())})
            hist.append(row)
            if it % o["history_every"] == 0 or it == iters - 1:
                rho_hist[f"it{it:05d}"] = rho_t.detach().cpu().numpy().astype(np.float16)
            if it % o["log_every"] == 0:
                extra = "  [" + " ".join(f"{float(v):.3f}" for v in foms.detach().cpu()) + "]" if len(foms) > 1 else ""
                print(f"it {it:5d}/{iters}  FoM {f:.4f}{extra}  beta {betas[it]:7.2f}  {el:7.1f}s", flush=True)
            if (it + 1) % o["checkpoint_every"] == 0 or it == iters - 1:
                torch.save({"config": cfg, "iter": it, "rho": rho.cpu(), "m": m.cpu(), "v": v.cpu(), "history": hist},
                           out / "checkpoint.pt")
                np.savez_compressed(out / "rho_history.npz", **rho_hist)
                with open(out / "history.csv", "w", newline="") as fh:
                    fields = list(dict.fromkeys(k for h in hist for k in h))
                    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(hist)

    # Final evaluation: continuous projected design and hard-binarized design
    with torch.no_grad():
        rho_final = project(blur(rho, g_fft), betas[-1])
        rho_bin = (rho_final > 0.5).to(gd)
        f_c, det_c = prob.fom(rho_final, return_details=True)
        f_b, det_b = prob.fom(rho_bin, return_details=True)
        analysis = prob.order_analysis(rho_bin)
        rb_final = blur(rho, g_fft)
        robust_eval = {}
        from ptopt.morph import feature_report, shift_edges
        bnp = rho_bin.cpu().numpy().astype(bool)
        ddx, ddy = prob.L[0] / nx, prob.L[1] / ny
        for sh in o.get("eval_shifts_nm", [-10, -5, 0, 5, 10]):
            bs = torch.as_tensor(shift_edges(bnp, sh, ddx, ddy), dtype=gd, device=device)
            robust_eval[f"{sh:+d}nm"] = {"fom_binary": float(prob.fom(bs)), "fill_fraction": float(bs.mean())}
        features = feature_report(bnp, ddx, ddy)
        vo = cfg.get("verify_order")
        if vo:  # convergence check of the binary design at higher Fourier truncation
            prob_v_order = [int(v) for v in vo]
            sims_fom = []
            for i in range(len(prob.wavelengths)):
                sim = prob.build_sim(rho_bin, i, order=prob_v_order)
                sims_fom.append(float(prob.objective(sim, prob.ocfg(i))[0]))
            verify = {"order": prob_v_order, "fom_binary": float(np.mean(sims_fom)),
                      "rel_diff": float(abs(np.mean(sims_fom) - float(f_b)) / max(1e-9, abs(float(f_b))))}
        else:
            verify = None
    rf = rho_final.cpu().numpy()
    np.save(out / "rho_final.npy", rf)
    np.save(out / "rho_binary.npy", rho_bin.cpu().numpy().astype(np.uint8))
    summary = {
        "run": out.name, "device": describe(device), "iters": iters,
        "fom_final": float(f_c), "fom_binary": float(f_b),
        "details_final": det_c, "details_binary": det_b,
        "order_analysis_binary": analysis,
        "convergence_check": verify,
        "robustness_binary": robust_eval,
        "feature_size": features,
        "fill_fraction": float(rho_bin.mean()),
        "grayness": float((4 * rho_final * (1 - rho_final)).mean()),
        "grayness_raw_rho": float((4 * rho * (1 - rho)).mean()),
        "wall_time_s": round(time.time() - t0, 1),
        "s_per_iter": round((time.time() - t0) / max(1, iters - start), 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    save_plots(out, rf, hist, cfg)
    print("[ptopt] DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
