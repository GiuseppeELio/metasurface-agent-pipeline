#!/usr/bin/env python3
"""Compare the COMSOL results of a run with the RCWA reference and give a verdict.

  python validation/comsol/comsol_compare.py --run runs/<dir> [--policy configs/validation_policy.json]

Reads
  runs/<dir>/comsol/<case>_plan.json           written by comsol_export.py (one per edge shift)
  runs/<dir>/comsol/MS_<case>_L<k>_results.json written by the COMSOL class (one per mesh level)
  runs/<dir>/rcwa_ref.json                      validation/rcwa_reference.py (shift key "+0", "+10", ...);
                                                --rcwa picks another file, e.g. rcwa_ref_o70.json (higher order)
  runs/<dir>/validation.json                    optional: Meep values from validation/validate.py, shown alongside;
                                                if it is missing, the highest-resolution meep_res<R>.json is used
Writes
  runs/<dir>/comsol_validation.json

Per input polarization, eta = sum of the target transmitted orders (objective.orders), from the COMSOL Fourier
projection; T = total transmission from the COMSOL ports. Thresholds come from the validation policy
(tolerance.eta_abs, tolerance.flux_abs), the same as the Meep check:

  agree         |eta_COMSOL - eta_RCWA| <= eta_abs and |T_COMSOL - T_RCWA| <= flux_abs
  inconclusive  a COMSOL self-consistency gate fails, or not agree and only one mesh level exists,
                or the mesh levels differ by more than half the gap
  marginal      not agree, gap <= 2 eta_abs
  disagree      otherwise -> raise the RCWA order, refine the mesh, check the geometry (pixel agreement in the plan)

COMSOL gates (each must hold): |R_port + T_port - 1| < 0.01 (lossless stacks), |T_flux - T_port| < 0.01
(observation plane vs port), |sum of the Fourier orders - T_port| < 0.01.
Only the polarizations in objective.input_pols decide the run verdict; the others are reported.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RANK = {"agree": 0, "marginal": 1, "inconclusive": 2, "disagree": 3, "missing": 4}
GATE = 0.01


def target_orders(cfg):
    obj = cfg.get("objective", {})
    if obj.get("type") in ("diffraction_efficiency", "diffraction_selectivity") and obj.get("port", "transmission") == "transmission":
        return [f"{o[0]},{o[1]}" for o in obj.get("orders", [])]
    return None


def verdict(d_eta, d_T, spread, n_levels, gates_ok, tol):
    if not gates_ok:
        return "inconclusive"
    if d_eta <= tol["eta_abs"] and d_T <= tol["flux_abs"]:
        return "agree"
    if n_levels < 2 or (spread is not None and spread > 0.5 * d_eta):
        return "inconclusive"
    return "marginal" if d_eta <= 2 * tol["eta_abs"] else "disagree"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--rcwa", default="rcwa_ref.json", help="RCWA reference file in the run directory")
    ap.add_argument("--policy", default=str(ROOT / "configs" / "validation_policy.json"))
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    tol = json.loads(Path(a.policy).read_text())["tolerance"] if Path(a.policy).exists() else {"eta_abs": 0.03, "flux_abs": 0.02}
    targets = target_orders(cfg)
    if not targets:
        sys.exit("[comsol_compare] the objective is not a transmitted-order efficiency: compare the per-order tables by hand")
    design_pols = cfg.get("objective", {}).get("input_pols", ["x", "y"])
    rref_path = run / a.rcwa
    if not rref_path.exists():
        sys.exit(f"[comsol_compare] no {rref_path}: run  python validation/rcwa_reference.py --run {run} --shifts 0")
    rref = json.loads(rref_path.read_text())
    meep = json.loads((run / "validation.json").read_text()) if (run / "validation.json").exists() else {}
    meep_src = "validation.json" if meep else None
    if not meep:   # fall back on the raw Meep result at the highest resolution (nominal design only)
        raw = sorted(run.glob("meep_res*.json"), key=lambda f: float(f.stem.replace("meep_res", "") or 0))
        if raw:
            mr = json.loads(raw[-1].read_text())
            meep = {"shifts": {"+0": {"pols": {p: {"eta_meep": round(sum(v["orders"].get(o, {}).get("eff", 0.0) for o in targets), 4)}
                                               for p, v in mr.get("pols", {}).items()}}}}
            meep_src = raw[-1].name
    plans = sorted((run / "comsol").glob("*_plan.json"))
    if not plans:
        sys.exit(f"[comsol_compare] no plan in {run}/comsol: run comsol_export.py first")

    report = {"run": run.name, "targets": targets, "tolerance": tol, "rcwa_file": a.rcwa, "rcwa_order": rref.get("order"),
              "meep_source": meep_src, "cases": {}}
    worst_all = "agree"
    print(f"[comsol_compare] {run.name}: target orders {targets}, RCWA order {rref.get('order')}, "
          f"tolerance eta {tol['eta_abs']} / T {tol['flux_abs']}" + (f", Meep from {meep_src}" if meep_src else ""))
    for pf in plans:
        plan = json.loads(pf.read_text())
        case, shift = plan["case"], plan.get("edge_shift_nm", 0.0)
        skey = f"{shift:+.0f}"
        lossless = all(v[1] == 0 for v in plan["index"].values())
        levels = []
        for lv in plan["levels"]:
            f = run / "comsol" / f"{lv['cls']}_results.json"
            if f.exists():
                r = json.loads(f.read_text())
                levels.append((lv, r))
        entry = {"edge_shift_nm": shift, "levels": [lv["cls"] for lv, _ in levels], "pols": {}}
        ref = rref.get("shifts", {}).get(skey)
        if not levels or ref is None:
            why = "no COMSOL results" if not levels else f"no RCWA entry for shift {skey} in rcwa_ref.json"
            print(f"  {case}: missing ({why})")
            entry["verdict"] = "missing"
            report["cases"][case] = entry
            worst_all = max(worst_all, "missing", key=RANK.get)
            continue
        print(f"\n  {case} (edge shift {skey} nm), mesh levels: " +
              ", ".join(f"{lv['cls'].rsplit('_', 1)[1]} h={lv['h_pat_nm']} nm DOF={r.get('dof')}" for lv, r in levels))
        print("  pol  eta_RCWA  eta_COMSOL  eta_Meep  d_eta   T_RCWA  T_COMSOL  d_T     R+T    T_flux-T  sumFour-T  mesh_spread  verdict")
        worst = "agree"
        for p in plan["pols"]:
            vals = [(lv, r["pols"][p]) for lv, r in levels if p in r.get("pols", {})]
            if not vals or p not in ref:
                entry["pols"][p] = {"verdict": "missing"}
                continue
            fin = vals[-1][1]
            eta_c = sum(fin["T"].get(o, 0.0) for o in targets)
            eta_r = sum(ref[p]["T"].get(o, 0.0) for o in targets)
            T_c, T_r = fin["T_port"], ref[p]["T_total"]
            spread = abs(eta_c - sum(vals[-2][1]["T"].get(o, 0.0) for o in targets)) if len(vals) > 1 else None
            g_rt = abs(fin["T_port"] + fin["R_port"] - 1) if lossless else 0.0
            g_fl = abs(fin["T_flux"] - T_c)
            g_fo = abs(fin["sum_eta_T"] - T_c)
            gates_ok = g_rt < GATE and g_fl < GATE and g_fo < GATE
            d_eta, d_T = abs(eta_c - eta_r), abs(T_c - T_r)
            v = verdict(d_eta, d_T, spread, len(vals), gates_ok, tol)
            em = meep.get("shifts", {}).get(skey, {}).get("pols", {}).get(p, {}).get("eta_meep")
            entry["pols"][p] = {"eta_rcwa": round(eta_r, 4), "eta_comsol": round(eta_c, 4), "eta_meep": em,
                                "T_rcwa": round(T_r, 4), "T_comsol": round(T_c, 4), "R_comsol": round(fin["R_port"], 4),
                                "d_eta": round(d_eta, 4), "d_T": round(d_T, 4),
                                "mesh_spread": None if spread is None else round(spread, 4),
                                "gates": {"R+T-1": round(g_rt, 4), "T_flux-T_port": round(g_fl, 4), "sum_orders-T_port": round(g_fo, 4),
                                          "ok": gates_ok},
                                "orders_T_comsol": fin["T"], "orders_T_rcwa": ref[p]["T"],
                                "design_pol": p in design_pols, "verdict": v}
            if p in design_pols:
                worst = max(worst, v, key=RANK.get)
            ems = f"{em:8.4f}" if isinstance(em, (int, float)) else "     -  "
            sps = f"{spread:8.4f}   " if spread is not None else "    -      "
            print(f"  {p:3s}  {eta_r:8.4f}  {eta_c:10.4f}  {ems}  {d_eta:6.4f}  {T_r:6.4f}  {T_c:8.4f}  {d_T:6.4f}  "
                  f"{fin['T_port'] + fin['R_port']:6.4f}  {g_fl:8.4f}  {g_fo:9.4f}  {sps}  {v}"
                  + ("" if p in design_pols else "  (not a design polarization)"))
        entry["verdict"] = worst
        if levels[-1][1].get("warnings"):
            entry["warnings"] = levels[-1][1]["warnings"]
            print(f"  WARN lines in the COMSOL log: {len(entry['warnings'])} (see {levels[-1][0]['cls']}_log.txt)")
        print(f"  verdict ({', '.join(design_pols)}): {worst}")
        report["cases"][case] = entry
        worst_all = max(worst_all, worst, key=RANK.get)
    report["verdict"] = worst_all
    out = run / "comsol_validation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[comsol_compare] verdict {worst_all}; wrote {out}")


if __name__ == "__main__":
    main()
