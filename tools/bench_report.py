#!/usr/bin/env python3
"""Compare benchmark.json files from several machines (tools/benchmark.py): one table and three figures.

  python tools/bench_report.py runs/bench_*/benchmark.json other_machine/benchmark.json --out runs/bench_report
The first file is the reference for the agreement columns (max phase / efficiency difference of the same workloads).
Outputs: <out>/bench_table.md, bench_times.png (time per tier), bench_rcwa_orders.png (s/simulation vs Fourier order),
bench_meep_scaling.png (Meep wall time vs MPI processes), bench_summary.json.
"""
import argparse, json
from pathlib import Path
import numpy as np


def cplx(lst):
    a = np.array(lst, float)
    return a[:, 0] + 1j * a[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="runs/bench_report")
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                    help="relabel a file's machine, e.g. to merge a GPU-node and a CPU-node run of one cluster")
    A = ap.parse_args()
    ren = dict(r.split("=", 1) for r in A.rename)
    D, by = [], {}
    for f in A.files:                 # files with the same label are merged (e.g. tiers 0-3 and tier 4 run separately)
        d = json.loads(Path(f).read_text()); d["_file"] = f; d["label"] = ren.get(d["label"], d["label"])
        if d["label"] in by:
            e = by[d["label"]]
            for k, v in d.items():
                if k.startswith("t") and k[1:2].isdigit():
                    e[k] = v
            if d.get("machine", {}).get("meep"):
                e.setdefault("machine", {})["meep_node"] = f"{d['machine'].get('cpu')} ({d['machine'].get('logical_cpus')} cpus)"
        else:
            by[d["label"]] = d; D.append(d)
    out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
    ref = D[0]
    rows, summ = [], []
    for d in D:
        m = d.get("machine", {})
        hw = f"{m.get('cpu', '?')} ({m.get('torch_threads', '?')} thr)" + (f" + {m['gpu']}" if m.get("gpu") else "")
        r = {"label": d["label"], "hardware": hw}
        t0 = d.get("t0_selftest", {}); r["selftest"] = f"{t0.get('wall_s', float('nan')):.0f} s {'PASS' if t0.get('passed') else 'FAIL'}" if t0 else "-"
        for dev in ("cpu", "cuda"):
            k = d.get("t1_rcwa_kernel", {}).get(dev, {}).get("s_per_sim", {})
            r[f"rcwa_{dev}_o9"] = k.get("9"); r[f"rcwa_{dev}_o11"] = k.get("11")
            k2 = d.get("t2_topopt", {}).get(dev, {})
            r[f"topopt_{dev}_s_it"] = k2.get("s_per_iter")
        t3 = d.get("t3_mini_lens", {})
        r["lens_total_min"] = t3.get("total_wall_s", np.nan) / 60 if t3 else None
        r["lens_device"] = t3.get("device")
        pts = d.get("t4_meep", {}).get("points", [])
        if pts:
            best = min(pts, key=lambda p: p["wall_s"]); r["meep_best"] = f"{best['wall_s'] / 60:.1f} min @ np {best['np']}"
            one = [p for p in pts if p["np"] == 1]
            r["meep_np1_min"] = one[0]["wall_s"] / 60 if one else None
        # agreement with the reference machine
        dph = []
        for dev, k in d.get("t1_rcwa_kernel", {}).items():
            for o, tt in k.get("t", {}).items():
                for rdev, rk in ref.get("t1_rcwa_kernel", {}).items():
                    if o in rk.get("t", {}):
                        dph.append(np.degrees(np.abs(np.angle(cplx(tt) / cplx(rk["t"][o])))).max())
        r["rcwa_max_dphase_deg_vs_ref"] = float(max(dph)) if dph else None
        e, er = t3.get("eval"), ref.get("t3_mini_lens", {}).get("eval")
        r["lens_max_deff_vs_ref"] = float(max(abs(a["eff_fixed"] - b["eff_fixed"]) for a, b in zip(e, er))) if e and er else None
        rows.append(r); summ.append(dict(r))
    f = lambda v, fmt="{:.2f}": "-" if v is None or (isinstance(v, float) and np.isnan(v)) else (fmt.format(v) if not isinstance(v, str) else v)
    hdr = ["Machine", "Hardware", "T0 self-test", "T1 RCWA s/sim CPU (o9 / o11)", "T1 RCWA s/sim GPU (o9 / o11)",
           "T2 topopt s/iter CPU / GPU", "T3 mini lens (min, device)", "T4 Meep best", "T4 Meep np=1 (min)",
           "max Δφ vs ref (°)", "max Δη lens vs ref"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        lines.append("| " + " | ".join([r["label"], r["hardware"], r["selftest"],
                                        f"{f(r['rcwa_cpu_o9'])} / {f(r['rcwa_cpu_o11'])}", f"{f(r['rcwa_cuda_o9'])} / {f(r['rcwa_cuda_o11'])}",
                                        f"{f(r['topopt_cpu_s_it'])} / {f(r['topopt_cuda_s_it'])}",
                                        f"{f(r['lens_total_min'], '{:.1f}')} ({r['lens_device'] or '-'})", r.get("meep_best", "-"),
                                        f(r.get("meep_np1_min"), "{:.1f}"), f(r["rcwa_max_dphase_deg_vs_ref"], "{:.3f}"),
                                        f(r["lens_max_deff_vs_ref"], "{:.4f}")]) + " |")
    (out / "bench_table.md").write_text("\n".join(lines) + "\n")
    (out / "bench_summary.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(lines))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # RCWA cost vs order
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for d in D:
        for dev, k in d.get("t1_rcwa_kernel", {}).items():
            s = k.get("s_per_sim")
            if s:
                o = sorted(int(x) for x in s)
                ax.plot(o, [s[str(i)] for i in o], marker="o", ls="-" if dev == "cuda" else "--", label=f"{d['label']} {dev}")
    ax.set(yscale="log", xlabel="Fourier order N (harmonics (2N+1)^2)", ylabel="seconds per RCWA simulation",
           title="T1: RCWA kernel (TiO2 pillar, complex128)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both"); fig.tight_layout(); fig.savefig(out / "bench_rcwa_orders.png", dpi=170)
    plt.close(fig)
    # time per tier
    tiers = [("T0 self-test (s)", lambda d: d.get("t0_selftest", {}).get("wall_s")),
             ("T2 topopt s/iter, best device", lambda d: min([v["s_per_iter"] for v in d.get("t2_topopt", {}).values() if "s_per_iter" in v] or [np.nan])),
             ("T3 mini lens (min)", lambda d: d.get("t3_mini_lens", {}).get("total_wall_s", np.nan) / 60),
             ("T4 Meep best (min)", lambda d: min([p["wall_s"] for p in d.get("t4_meep", {}).get("points", [])] or [np.nan]) / 60)]
    fig, ax = plt.subplots(1, len(tiers), figsize=(3.2 * len(tiers), 3.6))
    for a, (name, fn) in zip(ax, tiers):
        vals = [fn(d) for d in D]; vals = [np.nan if v is None else v for v in vals]
        a.bar(range(len(D)), vals, color=[f"C{i}" for i in range(len(D))])
        a.set_xticks(range(len(D))); a.set_xticklabels([d["label"] for d in D], rotation=30, ha="right", fontsize=7)
        a.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "bench_times.png", dpi=170); plt.close(fig)
    # Meep scaling
    if any(d.get("t4_meep", {}).get("points") for d in D):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        for i, d in enumerate(D):
            pts = sorted(d.get("t4_meep", {}).get("points", []), key=lambda p: p["np"])
            if not pts:
                continue
            n = np.array([p["np"] for p in pts]); w = np.array([p["wall_s"] for p in pts]) / 60
            ax[0].plot(n, w, "o-", color=f"C{i}", label=d["label"])
            if n[0] == 1:
                ax[1].plot(n, w[0] / w / n, "o-", color=f"C{i}", label=d["label"])
        ax[0].set(xscale="log", yscale="log", xlabel="MPI processes", ylabel="wall time (min)", title="T4: Meep strip, 40 px/um")
        ax[1].set(xscale="log", xlabel="MPI processes", ylabel="parallel efficiency", ylim=(0, 1.1), title="strong scaling")
        for a in ax:
            if a.get_legend_handles_labels()[0]:
                a.legend(fontsize=7)
            a.grid(alpha=0.3, which="both")
        fig.tight_layout(); fig.savefig(out / "bench_meep_scaling.png", dpi=170); plt.close(fig)
    print("[bench_report] wrote", out)


if __name__ == "__main__":
    main()
