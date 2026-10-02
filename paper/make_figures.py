#!/usr/bin/env python3
"""Generate the paper figures from the finished runs (all labels in English).

  python paper/make_figures.py            # writes paper/figures/fig*.png and .pdf
"""
import glob
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
C = {"nom": "#3b6fb6", "worst": "#d9822b", "lib": "#8a8a8a", "x": "#3b6fb6", "y": "#c0392b", "ok": "#2e8b57"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def one(pattern):
    m = sorted(glob.glob(str(RUNS / pattern)))
    return Path(m[-1]) if m else None


def jload(p):
    return json.loads(Path(p).read_text())


# ---------------------------------------------------------------- Fig. 1: pipeline schematic
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")

    def box(x, y, w, h, title, body, fc="#eef3fa", ec="#3b6fb6"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top", fontsize=7.6, weight="bold")
        ax.text(x + w / 2, y + h - 0.47, body, ha="center", va="top", fontsize=6.3, linespacing=1.25)

    def arrow(x0, y0, x1, y1, c="k", rad=0.0):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.0, connectionstyle=f"arc3,rad={rad}"))

    # top row: the agent loop
    box(0.05, 3.3, 1.75, 1.9, "User (chat)", "target in plain words:\nwavelength, materials,\norders, polarization,\ntolerance, budget",
        fc="#f6f6f6", ec="#777777")
    box(2.15, 3.3, 2.05, 1.9, "AI agent", "AGENTS.md: rules,\nonboarding, checklist\nphysics checks\nconfig JSON / job lists\ncost estimate", fc="#fdf2e6", ec="#d9822b")
    box(4.55, 3.3, 1.85, 1.9, "Bridge", "queue/*.req (one line)\nwatcher: whitelist,\npush, submit, follow,\npull results\nsite.env (private)")
    box(6.75, 3.3, 3.2, 1.9, "Execution target", "local GPU / CPU, or SLURM cluster\n\nTORCWA (GPU): topology optimization,\nRCWA references, near fields\nMeep (CPU, MPI): independent FDTD")
    arrow(1.8, 4.25, 2.15, 4.25); arrow(4.2, 4.25, 4.55, 4.25); arrow(6.4, 4.25, 6.75, 4.25)
    # bottom row: analysis and validation
    box(6.75, 0.75, 3.2, 1.75, "Results (runs/)", "summary.json: all orders, T+R balance,\nconvergence check, exact edge shifts\n(+/-5, 10, 15 nm), feature size\nmeep_res*.json, rcwa_ref*.json")
    box(3.55, 0.75, 2.85, 1.75, "Analysis + validation", "checklist: energy, convergence,\nbinarization, worst case, loss channel\nTORCWA <-> Meep verdict:\nagree / marginal / inconclusive /\ndisagree", fc="#eaf5ee", ec="#2e8b57")
    box(0.05, 0.75, 3.15, 1.75, "Report + next step", "table: nominal vs worst case,\nper polarization, dominant loss\ninterpretation\none proposed iteration + cost", fc="#fdf2e6", ec="#d9822b")
    arrow(8.35, 3.3, 8.35, 2.5); arrow(6.75, 1.62, 6.4, 1.62); arrow(3.55, 1.62, 3.2, 1.62)
    arrow(1.1, 2.5, 2.6, 3.3, c=C["worst"], rad=-0.3)
    ax.text(2.35, 2.75, "next iteration", color=C["worst"], fontsize=6.5, ha="center")
    ax.text(5.0, 0.3, "workflows: A freeform grating   B meta-atom library -> supercell   C cross-solver validation   D metalens",
            ha="center", fontsize=6.5, style="italic")
    save(fig, "fig1_pipeline")


# ---------------------------------------------------------------- Fig. 2: case 1, nominal vs worst case
def fig_case1():
    rows = []
    for f in ("robustness_si1550_defl_v1_si1550_defl_v2_si1550_defl_v3.json", "robustness_si1550_defl_v4.json"):
        if (RUNS / f).exists():
            rows += jload(RUNS / f)
    if not rows:
        return
    import re as _re

    def _key(r):
        n = r["run"].replace("si1550_defl_", "").split("_2026")[0]
        m = _re.match(r"v(\d+)_?(h|s)?(\d+)?", n)
        return (int(m.group(1)), int(m.group(3) or 0)) if m else (99, 0)
    rows = sorted(rows, key=_key)
    lab, nom, w10 = [], [], []
    for r in rows:
        name = r["run"].replace("si1550_defl_", "").split("_2026")[0].replace("_", " ")
        s = r["shifts"]
        lab.append(name); nom.append(s["+0"]["fom"])
        w10.append(min(s[k]["fom"] for k in ("-10", "-5", "+0", "+5", "+10")))
    x = np.arange(len(lab))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [2.2, 1]})
    a0.bar(x - 0.2, nom, 0.4, label="nominal", color=C["nom"])
    a0.bar(x + 0.2, w10, 0.4, label="worst case, edge shift $\\pm$10 nm", color=C["worst"])
    a0.set_xticks(x); a0.set_xticklabels(lab, rotation=45, ha="right")
    a0.set_ylabel("+1-order efficiency (unpolarized)"); a0.set_ylim(0, 1.15); a0.legend(loc="upper left", ncol=2, frameon=False)
    a0.set_title("(a) Si metagrating, $\\lambda$=1550 nm, 50$^\\circ$ deflection")
    for r in rows:
        if r["run"].startswith("si1550_defl_v2_h900") or r["run"].startswith("si1550_defl_v4_s2"):
            sh = sorted(r["shifts"], key=float)
            a1.plot([float(k) for k in sh], [r["shifts"][k]["fom"] for k in sh], "o-",
                    label=r["run"].replace("si1550_defl_", "").split("_2026")[0])
    a1.set_xlabel("edge shift (nm)"); a1.set_ylabel("efficiency"); a1.set_ylim(0.4, 0.9); a1.legend()
    a1.set_title("(b) sensitivity to fabrication")
    fig.tight_layout(); save(fig, "fig2_case1_robustness")

    # designs
    ds = [("v2 h900 (best nominal)", one("si1550_defl_v2_h900_*")), ("v4 s2 (robust)", one("si1550_defl_v4_s2_*"))]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 1.3))
    for a, (t, d) in zip(ax, ds):
        if d is None:
            continue
        cfg = jload(d / "config.json"); b = np.load(d / "rho_binary.npy")
        L = cfg["period_nm"]
        a.imshow(b.T, origin="lower", cmap="gray_r", extent=[0, L[0], 0, L[1]]); a.set_title(t)
        a.set_xlabel("x (nm)"); a.set_ylabel("y (nm)")
    save(fig, "fig2b_case1_designs")


# ---------------------------------------------------------------- Fig. 3: disk sweep
def fig_disk():
    p = RUNS / "diskA_sio2_P700" / "sweep.json"
    if not p.exists():
        return
    d = jload(p)["heights"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for h, v in d.items():
        lw = 2.2 if float(h) == 800 else 0.9
        ax[0].plot(v["R"], v["T"], lw=lw, label=f"h = {float(h):.0f} nm")
        ax[1].plot(v["R"], v["phase_deg"], lw=lw)
    ax[0].set(xlabel="disk radius R (nm)", ylabel="$|t_{00}|^2$", ylim=(0, 1.02), title="(a) transmission")
    ax[1].set(xlabel="disk radius R (nm)", ylabel="phase of $t_{00}$ (deg, unwrapped)", title="(b) phase")
    ax[1].axhline(360, ls=":", c="k", lw=0.8); ax[0].legend(ncol=2)
    fig.suptitle("Si disk on SiO$_2$, square cell P = 700 nm, $\\lambda$ = 1550 nm", y=1.02)
    fig.tight_layout(); save(fig, "fig3_disk_sweep")


# ---------------------------------------------------------------- Fig. 4: LUT
def fig_lut():
    p = RUNS / "lut_h800.json"
    if not p.exists():
        return
    sel = jload(p)["selection"]
    tg = np.array([s["target"] for s in sel]); T = np.array([s["T"] for s in sel])
    nom = np.array([s["nominal"] for s in sel]); w = np.array([s["worst"] for s in sel])
    shapes = [s["shape"] for s in sel]
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0])
    a = fig.add_subplot(gs[0])
    a.plot(tg, T, "o-", color=C["nom"], label="$|t|^2$ (nominal)")
    a.plot(tg, w, "s--", color=C["worst"], label="worst-case FoM, $\\pm$10 nm")
    mk = {"disk": "o", "clover": "*", "cross": "P", "ring": "D"}
    for t_, w_, s_ in zip(tg, w, shapes):
        a.scatter([t_], [0.05], marker=mk.get(s_, "x"), color="k", s=25)
    a.set(xlabel="target phase (deg)", ylabel="value", ylim=(0, 1.05), xticks=range(0, 360, 45),
          title="24-level library (bottom markers: winning initial shape; o disk, * clover, + cross)")
    a.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), frameon=True)
    gs2 = gs[1].subgridspec(2, 12)
    for i, s in enumerate(sel):
        ax = fig.add_subplot(gs2[i // 12, i % 12])
        b = np.load(RUNS / s["dir"] / "rho_binary.npy")
        ax.imshow(b.T, origin="lower", cmap="gray_r"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{s['target']:.0f}$^\\circ$", fontsize=7, pad=1)
    save(fig, "fig4_lut")


# ---------------------------------------------------------------- Fig. 5: supercells
def fig_supercells():
    rows = []
    for N, n in ((8, "sc8_lut"), (12, "sc12_lut"), (24, "sc24_lut")):
        p = RUNS / n / "supercell.json"
        if p.exists():
            j = jload(p); rows.append((N, j, np.load(RUNS / n / "rho_supercell.npy")))
    if not rows:
        return
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(len(rows) + 1, 2, width_ratios=[2.6, 1], height_ratios=[1] * len(rows) + [0.1], hspace=0.9)
    for i, (N, j, r) in enumerate(rows):
        a = fig.add_subplot(gs[i, 0])
        Lx = 700 * N
        a.imshow(r.T, origin="lower", cmap="gray_r", extent=[0, Lx / 1000, 0, 0.7], aspect="equal")
        a.set_xlim(0, 16.8)
        e = j["shifts"]["+0nm"]
        a.set_title(f"N = {N}: {j['deflection_angle_deg'][0]:.1f}$^\\circ$, $\\eta_{{+1}}$ = {e['eta_plus1_mean']:.3f}", loc="left")
        a.set_yticks([]); a.set_xlabel("x ($\\mu$m)" if i == len(rows) - 1 else ""); a.spines["left"].set_visible(False)
    b = fig.add_subplot(gs[:len(rows), 1])
    Ns = [r[0] for r in rows]
    b.plot(Ns, [r[1]["library_prediction_eta_plus1"] for r in rows], "o--", color=C["lib"], label="library prediction")
    b.plot(Ns, [r[1]["shifts"]["+0nm"]["eta_plus1_mean"] for r in rows], "o-", color=C["nom"], label="full RCWA")
    b.plot(Ns, [r[1]["shifts"]["+0nm"]["per_pol"]["eff_in_x"] for r in rows], ".:", color=C["x"], label="x-pol")
    b.plot(Ns, [r[1]["shifts"]["+0nm"]["per_pol"]["eff_in_y"] for r in rows], ".:", color=C["y"], label="y-pol")
    b.set(xlabel="atoms per supercell N", ylabel="+1-order efficiency", ylim=(0.75, 1.0), xticks=Ns)
    b.legend()
    save(fig, "fig5_supercells")


# ---------------------------------------------------------------- Fig. 6: global refinement
def fig_refine():
    pairs = [("sc8_h800_best", "sc8_refine_2026*"), ("sc8_lut", "sc8_lut_refine_*"), ("sc12_lut", "sc12_lut_refine_*")]
    avail = []
    for lib, ref in pairs:
        r = one(ref)
        if r is not None and (RUNS / lib / "supercell.json").exists():
            avail.append((lib, r))
    if not avail:
        return
    fig, ax = plt.subplots(len(avail), 2, figsize=(7.2, 1.6 * len(avail) + 0.4), squeeze=False,
                           gridspec_kw={"width_ratios": [2.4, 1]})
    for i, (lib, r) in enumerate(avail):
        j = jload(RUNS / lib / "supercell.json"); s = jload(r / "summary.json")
        b = np.load(r / "rho_binary.npy"); L = jload(r / "config.json")["period_nm"]
        ax[i, 0].imshow(b.T, origin="lower", cmap="gray_r", extent=[0, L[0] / 1000, 0, L[1] / 1000], aspect="equal")
        ax[i, 0].set_yticks([]); ax[i, 0].set_title(f"refined from {lib}: $\\eta_{{+1}}$ = {s['fom_binary']:.3f}", loc="left")
        sh_l = {float(k.replace("nm", "")): v["eta_plus1_mean"] for k, v in j["shifts"].items()}
        sh_r = {float(k.replace("nm", "")): v["fom_binary"] for k, v in s["robustness_binary"].items()}
        ax[i, 1].plot(sorted(sh_l), [sh_l[k] for k in sorted(sh_l)], "o--", color=C["lib"], label="library")
        ax[i, 1].plot(sorted(sh_r), [sh_r[k] for k in sorted(sh_r)], "o-", color=C["nom"], label="refined")
        ax[i, 1].set(ylim=(0.6, 1.0), xlabel="edge shift (nm)" if i == len(avail) - 1 else "", ylabel="$\\eta_{+1}$")
        ax[i, 0].set_xlabel("x ($\\mu$m)" if i == len(avail) - 1 else "")
        if i == 0:
            ax[i, 1].legend()
    fig.tight_layout(); save(fig, "fig6_refinement")


# ---------------------------------------------------------------- Fig. 7: Meep vs RCWA, resolution study
def _target(run):
    cfgf = run / "config.json"
    if cfgf.exists():
        o = jload(cfgf)["objective"]
        if o.get("type") == "diffraction_efficiency":
            return [f"{a},{b}" for a, b in o["orders"]]
    return ["1,0"]


def fig_meep():
    runs = sorted({Path(f).parent for f in glob.glob(str(RUNS / "*" / "meep_res*.json"))})
    runs = [r for r in runs if len(glob.glob(str(r / "meep_res*.json"))) >= 2]
    if not runs:
        return
    fig, axs = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 2.8), squeeze=False)
    table = []
    for ax, r in zip(axs[0], runs):
        tg = _target(r)
        ms = sorted((jload(f) for f in glob.glob(str(r / "meep_res*.json"))), key=lambda j: j["resolution"])
        ms = [m for m in ms if m["resolution"] >= 30] or ms           # coarse smoke tests are not part of the study
        refs = sorted((jload(f) for f in glob.glob(str(r / "rcwa_ref*.json"))), key=lambda j: j["order"][0] * j["order"][1])
        refs = [q for q in refs if "+0" in q["shifts"]]
        ref = refs[-1] if refs else None
        for pol in ("x", "y"):
            res_ = [m["resolution"] for m in ms if pol in m["pols"]]
            eta = [sum(m["pols"][pol]["orders"].get(k, {}).get("eff", 0) for k in tg) for m in ms if pol in m["pols"]]
            ax.plot(res_, [100 * e for e in eta], "o-", color=C[pol], label=f"Meep, {pol}-pol")
            for q in refs[:-1]:                                   # lower truncations: RCWA convergence
                eq = sum(q["shifts"]["+0"][pol]["T"].get(k, 0) for k in tg)
                ax.axhline(100 * eq, color=C[pol], ls=":", lw=0.6, alpha=0.6)
            if ref:
                er = sum(ref["shifts"]["+0"][pol]["T"].get(k, 0) for k in tg); src = f"RCWA {ref['order']}"
            else:
                er = sum(ms[-1]["rcwa"][pol].get(k, 0) for k in tg); src = "RCWA (training order)"
            ax.axhline(100 * er, color=C[pol], ls="--", lw=0.9, label=f"{src}, {pol}-pol")
            table.append({"run": r.name, "pol": pol, "meep": dict(zip(res_, eta)), "rcwa": er, "rcwa_source": src})
        name = r.name.split("_2026")[0]
        ax.set(title=name, xlabel="Meep resolution (px/$\\mu$m)", ylabel=f"$\\eta_{{{tg[0]}}}$ (% of incident power)")
        ax.legend(fontsize=6)
    if any(len(glob.glob(str(r / "rcwa_ref*.json"))) > 1 for r in runs):
        fig.text(0.01, -0.02, "dotted: RCWA at lower Fourier truncations", fontsize=6)
    fig.tight_layout(); save(fig, "fig7_meep_validation")
    (OUT / "meep_table.json").write_text(json.dumps(table, indent=1))


# ---------------------------------------------------------------- Fig. 8: metalens from the library
def fig_lens():
    lenses = [Path(p).parent for p in sorted(glob.glob(str(RUNS / "lens_*" / "lens.json")))]
    lenses = [l for l in lenses if (l / "lens_model.npz").exists()]
    if not lenses:
        return
    fig, axs = plt.subplots(len(lenses), 3, figsize=(7.4, 2.4 * len(lenses)), squeeze=False)
    for row, l in zip(axs, lenses):
        L = jload(l / "lens.json"); M = np.load(l / "lens_model.npz")
        z_ = np.load(l / "rho_lens.npz"); lay = np.unpackbits(z_["packed"])[:int(np.prod(z_["shape"]))].reshape(z_["shape"])
        R, P, npx, Nc = L["diameter_um"] / 2, L["period_um"], L["atom_grid_px"], L["cells"]
        q = Nc // 2 * npx
        row[0].imshow(lay[q:, q:].T, origin="lower", cmap="gray_r", interpolation="nearest",
                      extent=[-P / 2, R, -P / 2, R])
        row[0].set(title=f"quadrant, D = {2 * R:.1f} $\\mu$m, NA = {L['NA']:.2f}", xlabel="x ($\\mu$m)", ylabel="y ($\\mu$m)")
        pk = float(M["ideal_peak"])
        row[1].plot(M["z_um"], M["axial_ideal"] / pk, ":", color=C["lib"], label="ideal lens")
        row[1].plot(M["z_um"], M["axial_lens"] / pk, "-", color=C["nom"], label="LUT lens (LPA)")
        meeps = sorted(glob.glob(str(l / "meep_lens_res*.npz")))
        lab = "LUT lens (Meep)"
        if meeps:
            E = np.load(meeps[-1]); mj = jload(meeps[-1].replace(".npz", ".json"))
            scale = (M["axial_lens"].max() / pk) / E["axial_I"].max()      # shape comparison: same peak as LPA
            row[1].plot(E["z_um"], E["axial_I"] * scale, "--", color=C["worst"], label=lab + f", res {mj['resolution']:g}")
        row[1].axvline(L["focal_um"], color="0.7", lw=0.6)
        row[1].set(xlabel="z above the lens ($\\mu$m)", ylabel="on-axis |E|$^2$ / ideal peak", title="axial profile")
        row[1].legend(fontsize=6)
        x = M["x_um"]
        row[2].plot(x, M["cutx_ideal"] / pk, ":", color=C["lib"], label="ideal")
        row[2].plot(x, M["cutx_lens"] / pk, "-", color=C["nom"], label="LPA, x cut")
        if meeps:
            xm = E["x_um"]; I = E["focal_I"]; c0 = I.shape[0] // 2
            row[2].plot(xm, I[:, c0] / I.max() * (M["cutx_lens"].max() / pk), "--", color=C["worst"], label="Meep, x cut")
        lp = L["lpa_model"]["lens"]
        ttl = f"focal plane; focusing eff. LPA {100 * lp['focusing_efficiency']:.1f}%"
        if meeps:
            ttl += f", Meep {100 * mj['focusing_efficiency']:.1f}%"
        row[2].set(xlabel="x ($\\mu$m)", ylabel="|E|$^2$ / ideal peak", xlim=(x[0], x[-1]))
        row[2].set_title(ttl, fontsize=7)
        row[2].legend(fontsize=6, loc="upper right", frameon=False)
    fig.tight_layout(); save(fig, "fig8_metalens")


# ---------------------------------------------------------------- Fig. 11: metalens propagation, DOF and FWHM
def fig_lens_prop():
    lenses = [Path(p).parent for p in sorted(glob.glob(str(RUNS / "lens_*" / "lens_model.npz")))]
    lenses = [l for l in lenses if "xz_lens" in np.load(l / "lens_model.npz").files]
    if not lenses:
        return
    fig, axs = plt.subplots(len(lenses), 3, figsize=(7.4, 2.9 * len(lenses)), squeeze=False,
                            gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for row, l in zip(axs, lenses):
        L = jload(l / "lens.json"); M = np.load(l / "lens_model.npz")
        posts = sorted(glob.glob(str(l / "lens_post_res*.npz")))
        E = np.load(posts[-1]) if posts else None
        J = jload(posts[-1].replace(".npz", ".json")) if posts else None
        lp = L["lpa_model"]["lens"]
        ext = [M["xsec_um"][0], M["xsec_um"][-1], M["zsec_um"][0], M["zsec_um"][-1]]
        panels = [(row[0], M["xz_lens"], "LPA model")] + ([(row[1], E["xz"], f"FDTD (Meep, {J['meep_resolution']} px/$\\mu$m)")] if E is not None else [])
        for ax, data, ttl in panels:
            ax.imshow(data / data.max(), origin="lower", aspect="auto", cmap="magma", vmin=0, extent=ext)
            ax.set(xlabel="x ($\\mu$m)", title=ttl)
            ax.set_xlim(-0.6 * L["diameter_um"], 0.6 * L["diameter_um"])
        row[0].set_ylabel(f"NA {L['NA']:.2f}\nz above the lens ($\\mu$m)")
        if E is None:
            row[1].axis("off")
        a = row[2]
        a.plot(M["z_um"], M["axial_lens"] / M["axial_lens"].max(), color=C["nom"], label=f"LPA, DOF {lp['dof_um']:.1f} $\\mu$m")
        if lp.get("dof_range_um"):
            a.axvspan(*lp["dof_range_um"], color=C["nom"], alpha=0.12)
        if E is not None:
            mm = J["meep"]
            a.plot(E["z_um"], E["axial"] / E["axial"].max(), "--", color=C["worst"], label=f"FDTD, DOF {mm['dof_um']:.1f} $\\mu$m")
        a.axhline(0.5, c="0.75", lw=0.6); a.axvline(L["focal_um"], c="0.75", lw=0.6)
        a.set(xlabel="z above the lens ($\\mu$m)", ylabel="on-axis |E|$^2$ (normalized)", title="axial profile")
        a.legend(fontsize=6.5, loc="upper right", frameon=False)
    fig.tight_layout(); save(fig, "fig11_metalens_propagation")


if __name__ == "__main__":
    for f in (fig_pipeline, fig_case1, fig_disk, fig_lut, fig_supercells, fig_refine, fig_meep, fig_lens, fig_lens_prop):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print("FAILED", f.__name__, e)
