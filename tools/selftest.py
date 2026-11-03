#!/usr/bin/env python3
"""Regression test: reproduce reference numbers of the worked example (silicon at 1550 nm) on the CPU.

  python tools/selftest.py          quick (~1 min on a laptop CPU): re-evaluation of the robust grating,
                                    8-level supercell from the library, NA 0.5 metalens
  python tools/selftest.py --full   adds a short robust optimization from scratch (~3-5 min on CPU)

Run it after installing the environment, after updating torch/torcwa/numpy, and before a release.
It seeds the worked example into runs/ if needed (tools/seed_examples.py, never overwrites), writes its outputs
under *_selftest names and removes them at the end (unless --keep). Your own runs are never touched.
References and tolerances: tools/selftest_reference.json. Exit code 0 = all checks pass.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REF = json.loads((HERE / "tools" / "selftest_reference.json").read_text())
LOGS = HERE / "logs"


def run(args, log):
    LOGS.mkdir(exist_ok=True)
    t = time.time()
    with open(LOGS / log, "w", encoding="utf-8") as f:
        rc = subprocess.run([sys.executable, "-u"] + args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - t


def compare(name, got, ref, tol):
    ok = all(abs(got[k] - ref[k]) <= tol for k in got)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: " + "  ".join(f"{k}={got[k]:.4f} (ref {ref[k]:.4f})" for k in got) + f"  tol ±{tol}")
    return ok


ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true")
ap.add_argument("--keep", action="store_true", help="keep the *_selftest outputs")
a = ap.parse_args()

runs = HERE / "runs"
created = []
if not (runs / "si1550_defl_v4_s2").exists() or not (runs / "lut_h800.json").exists():
    run(["tools/seed_examples.py"], "selftest_seed.out")
print(f"[selftest] {HERE} (CPU); logs in logs/selftest_*.out")
res = []

rc, dt = run(["validation/rcwa_reference.py", "--run", "runs/si1550_defl_v4_s2", "--order", "15", "3", "--device", "cpu",
              "--out", "rcwa_ref_selftest.json"], "selftest_rcwa_reference.out")
out = runs / "si1550_defl_v4_s2" / "rcwa_ref_selftest.json"; created.append(out)
try:
    d = json.loads(out.read_text())["shifts"]["+0"]
    got = {"x_T_+1": d["x"]["T"]["1,0"], "y_T_+1": d["y"]["T"]["1,0"], "x_T_total": d["x"]["T_total"], "y_T_total": d["y"]["T_total"]}
    r = REF["quick"]["rcwa_reference"]; res.append(compare(f"re-evaluation of the robust grating ({dt:.0f} s)", got, r, r["tol"]))
except Exception as e:
    print(f"  [FAIL] rcwa_reference: {e}"); res.append(False)

rc, dt = run(["supercell.py", "--name", "sc8_selftest", "--library", "lut_h800", "--levels", "8", "--order", "21", "5",
              "--shifts", "0", "--device", "cpu"], "selftest_supercell.out")
created.append(runs / "sc8_selftest")
try:
    d = json.loads((runs / "sc8_selftest" / "supercell.json").read_text())["shifts"]["+0nm"]
    got = {"eta_plus1": d["eta_plus1_mean"], "eff_x": d["per_pol"]["eff_in_x"], "eff_y": d["per_pol"]["eff_in_y"]}
    r = REF["quick"]["supercell"]; res.append(compare(f"8-level supercell from the library ({dt:.0f} s)", got, r, r["tol"]))
except Exception as e:
    print(f"  [FAIL] supercell: {e}"); res.append(False)

rc, dt = run(["metalens.py", "--library", "lut_h800", "--diameter-um", "20", "--na", "0.5", "--name", "lens_selftest"],
             "selftest_metalens.out")
created.append(runs / "lens_selftest")
try:
    d = json.loads((runs / "lens_selftest" / "lens.json").read_text())["lpa_model"]["lens"]
    got = {"focusing_efficiency": d["focusing_efficiency"], "strehl": d["strehl_vs_ideal"], "fwhm_x_um": d["fwhm_x_um"]}
    r = REF["quick"]["metalens"]; res.append(compare(f"metalens NA 0.5 ({dt:.0f} s)", got, r, r["tol"]))
except Exception as e:
    print(f"  [FAIL] metalens: {e}"); res.append(False)

if a.full:
    before = set(p.name for p in runs.iterdir())
    rc, dt = run(["run_topopt.py", "--config", "configs/metagrating_si1550_robust.json", "--iters", "150", "--order", "10", "3",
                  "--grid", "200", "40", "--set", "verify_order=[15,3]", "--device", "cpu", "--tag", "selftest"], "selftest_run_topopt.out")
    new = [runs / n for n in set(p.name for p in runs.iterdir()) - before if "selftest" in n]
    created += new
    try:
        s = json.loads((sorted(new, key=lambda p: p.stat().st_mtime)[-1] / "summary.json").read_text())
        r = REF["full"]["run_topopt"]; res.append(compare(f"robust grating from scratch ({dt:.0f} s)", {"fom_binary": s["fom_binary"]}, r, r["tol"]))
    except Exception as e:
        print(f"  [FAIL] run_topopt: {e}"); res.append(False)

ok = all(res)
print(f"[selftest] {'ALL PASSED' if ok else 'FAILED'} ({sum(res)}/{len(res)})")
if not a.keep:
    for p in created:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
sys.exit(0 if ok else 1)
