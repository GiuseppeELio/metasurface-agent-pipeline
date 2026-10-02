#!/usr/bin/env python3
"""Cross-solver validation stage: TORCWA (the optimizer's solver) against Meep (independent FDTD).

Two sub-commands. Both read the policy in configs/validation_policy.json (levels, resolutions, tolerances).

  plan    write the job lists for one validation round, and optionally queue them:
            python validation/validate.py plan --runs runs/A runs/B --level quick|full --tag T [--enqueue]
          -> sweeps/val_T_gpu.txt   (bridge 'array': rcwa_reference.py, torcwa_fields.py)
             sweeps/val_T_meep.txt  (bridge 'meep':  meep_validate.py at each resolution / edge shift)
          prints a run-time estimate for the Meep part.

  report  after both jobs have finished (and the results are pulled), compare and give a verdict:
            python validation/validate.py report --runs runs/A runs/B [--figures]
          -> runs/<dir>/validation.json per run, a table on stdout, and the rank agreement between runs.
          The RCWA side is the rcwa_ref*.json with the highest truncation (add one with
          rcwa_reference.py --order Nx Ny --out rcwa_ref_<tag>.json when RCWA convergence is in doubt).
          --figures also calls validation/compare_fields.py where both field files exist.

Verdict per design and polarization (target order efficiency eta, total transmission T):
  agree         |eta_Meep - eta_RCWA| <= eta_abs and |T_Meep - T_RCWA| <= flux_abs
  inconclusive  not agree, and either only one Meep resolution exists, or Meep changes by more than half the gap
                between its two highest resolutions (not converged: raise the resolution before blaming a solver)
  marginal      not agree, gap <= 2 eta_abs
  disagree      otherwise -> diagnose (raise RCWA order, raise Meep resolution, check the geometry is identical)
The design verdict is the worst over polarizations. Edge-shifted variants get their own verdicts.
"""
import argparse
import glob
import json
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from ptopt.rundir import load_run, target_orders  # noqa: E402

RANK = {"agree": 0, "marginal": 1, "inconclusive": 2, "disagree": 3, "missing": 4}


def policy(path):
    return json.loads(Path(path).read_text())


def meep_cost_s(cfg, res):
    """Rough wall time on 48 MPI ranks, calibrated on a 2.0 x 0.4 x 5.4 um^3 cell: 26 s at 40 px/um (2 pols)."""
    sx, sy = cfg["period_nm"][0] / 1000, cfg["period_nm"][1] / 1000
    sz = 2 * 1.0 + 1.0 + cfg["layer"]["thickness_nm"] / 1000 + 1.5
    return 26.0 * (sx * sy * sz / 4.37) * (res / 40.0) ** 4


def cmd_plan(a):
    pol = policy(a.policy)
    lv = pol[a.level]
    gpu, meep, est = [], [], 0.0
    for r in a.runs:
        cfg, _, _ = load_run(r)
        shifts = lv.get("shifts_nm", [])
        gpu.append(f"validation/rcwa_reference.py --run {r} --shifts 0 " + " ".join(f"{s:g}" for s in shifts))
        if lv.get("fields"):
            gpu.append(f"validation/torcwa_fields.py --run {r}")
        pols = " ".join(lv.get("pols", ["x", "y"]))
        for i, R in enumerate(lv["resolutions"]):
            last = i == len(lv["resolutions"]) - 1
            meep.append(f"--run {r} --resolution {R:g} --pols {pols}" + (" --fields" if lv.get("fields") and last else ""))
            est += meep_cost_s(cfg, R)
        Rs = lv.get("shift_resolution", lv["resolutions"][0])
        for s in shifts:
            meep.append(f"--run {r} --resolution {Rs:g} --pols {pols} --shift-nm {s:g}")
            est += meep_cost_s(cfg, Rs)
    sw = HERE / "sweeps"; sw.mkdir(exist_ok=True)
    fg, fm = sw / f"val_{a.tag}_gpu.txt", sw / f"val_{a.tag}_meep.txt"
    fg.write_text("\n".join(gpu) + "\n"); fm.write_text("\n".join(meep) + "\n")
    print(f"[validate] wrote {fg.relative_to(HERE)} ({len(gpu)} GPU lines) and {fm.relative_to(HERE)} ({len(meep)} Meep runs)")
    print(f"[validate] Meep estimate ~{est / 60:.1f} min on 48 MPI ranks (rough; scales with cell volume x resolution^4)")
    if a.enqueue:
        q = HERE / "queue"; q.mkdir(exist_ok=True)
        (q / f"val_{a.tag}_gpu.req").write_text(f"array sweeps/{fg.name}\n")
        (q / f"val_{a.tag}_meep.req").write_text(f"meep sweeps/{fm.name}\n")
        print(f"[validate] queued val_{a.tag}_gpu.req and val_{a.tag}_meep.req")
    else:
        print(f"[validate] to queue:  echo 'array sweeps/{fg.name}' > queue/val_{a.tag}_gpu.req ;"
              f"  echo 'meep sweeps/{fm.name}' > queue/val_{a.tag}_meep.req")


def _rcwa_entry(run, shift_key, meep_json):
    """(per-pol dict with T orders and T_total, source label)."""
    refs = [json.loads(Path(f).read_text()) for f in glob.glob(str(run / "rcwa_ref*.json"))]
    refs = [r for r in refs if shift_key in r.get("shifts", {})]
    if refs:   # highest Fourier truncation available (rcwa_reference.py --order ... --out rcwa_ref_<tag>.json)
        ref = max(refs, key=lambda r: r["order"][0] * r["order"][1])
        return ref["shifts"][shift_key], f"rcwa_ref order {ref['order']}"
    if shift_key == "+0" and meep_json and "rcwa" in meep_json and "error" not in meep_json["rcwa"]:
        e = {p: {"T": {k: v for k, v in d.items() if k != "T_total"}, "T_total": d["T_total"]}
             for p, d in meep_json["rcwa"].items()}
        return e, "training order (run summary)"
    return None, "missing"


def _verdict(d_eta, d_T, spread, tol):
    if d_eta <= tol["eta_abs"] and d_T <= tol["flux_abs"]:
        return "agree"
    if spread is None or spread > 0.5 * d_eta:   # single resolution, or Meep still moving
        return "inconclusive"
    return "marginal" if d_eta <= 2 * tol["eta_abs"] else "disagree"


def report_run(run, tol):
    cfg, _, _ = load_run(run)
    targets = target_orders(cfg)
    groups = {}
    for f in glob.glob(str(run / "meep_*res*.json")):
        m = re.search(r"meep_(?:s([+-]\d+)_)?res(\d+)\.json$", f)
        if m:
            groups.setdefault(m.group(1) or "+0", []).append((int(m.group(2)), json.loads(Path(f).read_text())))
    out = {"run": run.name, "targets": targets, "tolerance": tol, "shifts": {}}
    for sk in sorted(groups, key=lambda s: (s != "+0", s)):
        ms = sorted(groups[sk], key=lambda t: t[0])
        R_hi, M_hi = ms[-1]
        ref, src = _rcwa_entry(run, sk, M_hi)
        entry = {"rcwa_source": src, "meep_resolutions": [r for r, _ in ms], "pols": {}}
        for p, mp_ in M_hi["pols"].items():
            if ref is None or p not in ref:
                entry["pols"][p] = {"verdict": "missing"}; continue
            tg = targets or [max(ref[p]["T"], key=ref[p]["T"].get)]
            eta_r = sum(ref[p]["T"].get(k, 0.0) for k in tg)
            eta_m = sum(mp_["orders"].get(k, {}).get("eff", 0.0) for k in tg)
            spread = None
            if len(ms) > 1 and p in ms[-2][1]["pols"]:
                eta_lo = sum(ms[-2][1]["pols"][p]["orders"].get(k, {}).get("eff", 0.0) for k in tg)
                spread = abs(eta_m - eta_lo)
            d_eta, d_T = abs(eta_m - eta_r), abs(mp_["T_total_flux"] - ref[p]["T_total"])
            entry["pols"][p] = {"orders": tg, "eta_rcwa": round(eta_r, 4), "eta_meep": round(eta_m, 4),
                                "T_rcwa": ref[p]["T_total"], "T_meep": round(mp_["T_total_flux"], 4),
                                "d_eta": round(d_eta, 4), "d_T": round(d_T, 4),
                                "meep_res_spread": None if spread is None else round(spread, 4),
                                "verdict": _verdict(d_eta, d_T, spread, tol)}
        vs = [v["verdict"] for v in entry["pols"].values()]
        entry["verdict"] = max(vs, key=RANK.get) if vs else "missing"
        pe = [v for v in entry["pols"].values() if "eta_rcwa" in v]
        if pe:
            entry["fom_rcwa"] = round(sum(v["eta_rcwa"] for v in pe) / len(pe), 4)
            entry["fom_meep"] = round(sum(v["eta_meep"] for v in pe) / len(pe), 4)
        out["shifts"][sk] = entry
    vs = [e["verdict"] for e in out["shifts"].values()]
    out["verdict"] = max(vs, key=RANK.get) if vs else "missing"
    (run / "validation.json").write_text(json.dumps(out, indent=2))
    return out


def cmd_report(a):
    pol = policy(a.policy)
    tol = pol["tolerance"]
    rows, reps = [], []
    for r in a.runs:
        rep = report_run(Path(r), tol); reps.append(rep)
        for sk, e in rep["shifts"].items():
            for p, v in e["pols"].items():
                if "eta_rcwa" in v:
                    sp = "-" if v["meep_res_spread"] is None else f"{v['meep_res_spread']:.3f}"
                    rows.append(f"| {rep['run'][:40]} | {sk} | {p} | {v['eta_rcwa']:.3f} | {v['eta_meep']:.3f} | "
                                f"{v['T_rcwa']:.3f} | {v['T_meep']:.3f} | {e['meep_resolutions']} | {sp} | {v['verdict']} |")
                else:
                    rows.append(f"| {rep['run'][:40]} | {sk} | {p} | - | - | - | - | - | - | {v['verdict']} |")
        if a.figures and (Path(r) / "fields_rcwa.npz").exists() and glob.glob(str(Path(r) / "fields_meep_res*.npz")):
            subprocess.run([sys.executable, str(HERE / "validation" / "compare_fields.py"), "--run", r], check=False)
    print("| run | shift nm | pol | eta RCWA | eta Meep | T RCWA | T Meep | Meep res | res spread | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    nom = [(rep["run"], rep["shifts"]["+0"]["fom_rcwa"], rep["shifts"]["+0"]["fom_meep"]) for rep in reps
           if "+0" in rep["shifts"] and "fom_rcwa" in rep["shifts"]["+0"]]
    if len(nom) > 1:
        pairs = [(i, j) for i in range(len(nom)) for j in range(i + 1, len(nom))]
        ok = sum((nom[i][1] - nom[j][1]) * (nom[i][2] - nom[j][2]) > 0 for i, j in pairs)
        print(f"\nrank agreement (nominal FoM): {ok}/{len(pairs)} pairs ordered the same by both solvers")
        for n, fr, fm in sorted(nom, key=lambda t: -t[1]):
            print(f"  {n[:48]:48s} RCWA {fr:.3f}  Meep {fm:.3f}")
    worst = max((rep["verdict"] for rep in reps), key=RANK.get)
    print(f"\noverall verdict: {worst}")
    if pol.get("on_fail") == "stop" and worst in ("disagree", "missing"):
        sys.exit(2)


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
p1 = sub.add_parser("plan"); p1.add_argument("--runs", nargs="+", required=True)
p1.add_argument("--level", choices=["quick", "full"], default="full"); p1.add_argument("--tag", required=True)
p1.add_argument("--enqueue", action="store_true")
p2 = sub.add_parser("report"); p2.add_argument("--runs", nargs="+", required=True)
p2.add_argument("--figures", action="store_true")
for p in (p1, p2):
    p.add_argument("--policy", default=str(HERE / "configs" / "validation_policy.json"))
a = ap.parse_args()
{"plan": cmd_plan, "report": cmd_report}[a.cmd](a)
