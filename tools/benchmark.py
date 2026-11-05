#!/usr/bin/env python3
"""Cross-machine benchmark of the pipeline: the same small workloads on laptops, workstations and HPC nodes.

Tiers (each one is timed and its physical result is stored, so machines can be compared for speed AND agreement):
  0  self-test            tools/selftest.py (CPU): installation and reproducibility of the worked example
  1  RCWA kernel          8 TiO2 meta-atoms x 3 wavelengths at Fourier orders 5..11 (tools/tio2_lut.py --conv-orders),
                          on the CPU and, if present, on the CUDA GPU: seconds per simulation vs order
  2  topology optimization 40 iterations of the robust Si grating (run_topopt.py), CPU and GPU: seconds per iteration
  3  mini achromatic lens  library of 30 TiO2 pillars x 16 wavelengths (700-1000 nm, order 9) on the faster device of tier 1,
                          then a D = 100 um, NA 0.04 achromatic lens (tools/achro_tio2.py): end-to-end time + efficiency
  4  Meep FDTD            10 um strip of the tier-3 lens, 40 px/um, broadband 700-1000 nm (validation/meep_achro_strip.py),
                          with 1, 2, 4, ... MPI processes (Linux/macOS with Meep; not available on native Windows)

  python tools/benchmark.py --label "RTX3080 + 16-core"                 tiers 0-3 (+4 if Meep is installed)
  python tools/benchmark.py --label "my laptop" --tiers 0 1             a subset
  python tools/benchmark.py --tiers 4 --np 1 2 4 8 16                   Meep scaling (mpirun on the PATH)
  (HPC) a Meep job line "tools/benchmark.py --tiers 4 --label HPC-48c" runs inside the job's mpirun: one point, np = ranks
Output: runs/bench_<label>_<date>/benchmark.json (+ logs/bench_*.out). Compare machines with tools/bench_report.py.
Nothing else in runs/ is touched; temporary runs are named *_bench* and removed unless --keep.
"""
import argparse, glob, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RUNS, LOGS = HERE / "runs", HERE / "logs"
MPI_SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
MPI_RANK = int(os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def machine_info():
    info = dict(host_os=f"{platform.system()} {platform.release()}", python=platform.python_version(),
                machine=platform.machine(), cpu=platform.processor() or "", logical_cpus=os.cpu_count())
    if platform.system() == "Linux":
        for line in open("/proc/cpuinfo", errors="ignore"):
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip(); break
        try:
            info["ram_gb"] = round(int(open("/proc/meminfo").readline().split()[1]) / 2 ** 20, 1)
        except Exception:
            pass
        info["slurm_cpus"] = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_NTASKS")
    elif platform.system() == "Darwin":
        info["cpu"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        try:
            info["ram_gb"] = round(int(sh(["sysctl", "-n", "hw.memsize"])) / 2 ** 30, 1)
        except Exception:
            pass
    elif platform.system() == "Windows":
        info["cpu"] = platform.processor()
    try:
        import torch
        info.update(torch=torch.__version__, torch_threads=torch.get_num_threads(), cuda=torch.version.cuda,
                    cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2 ** 30, 1), gpu_cc=f"{p.major}.{p.minor}")
    except Exception as e:
        info["torch_error"] = str(e)
    for mod in ("numpy", "torcwa", "meep"):
        try:
            m = __import__(mod); info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    info["nvidia_smi"] = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return info


def run(args, log, env=None):
    LOGS.mkdir(exist_ok=True)
    t = time.time()
    with open(LOGS / log, "w", encoding="utf-8") as f:
        rc = subprocess.run(args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return rc, round(time.time() - t, 2)


def py(*a):
    return [sys.executable, "-u", *a]


def tier0(R, A):
    rc, dt = run(py("tools/selftest.py"), "bench_t0_selftest.out")
    tail = (LOGS / "bench_t0_selftest.out").read_text(errors="ignore").strip().splitlines()[-1:]
    R["t0_selftest"] = dict(wall_s=dt, passed=rc == 0, summary=tail[0] if tail else "")


def devices():
    import torch
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tier1(R, A, tmp):
    out = {}
    for dev in devices():
        name = f"t1_{dev}_bench"; tmp.append(RUNS / name)
        rc, dt = run(py("tools/tio2_lut.py", "--name", name, "--period", "450", "--height", "1000",
                        "--conv-orders", *[str(o) for o in A.orders], "--device", dev), f"bench_t1_{dev}.out")
        try:
            c = json.loads((RUNS / name / "conv.json").read_text())
            out[dev] = dict(wall_s=dt, s_per_sim={k: round(v, 4) for k, v in c["sec"].items()}, t=c["t"])
        except Exception as e:
            out[dev] = dict(wall_s=dt, error=str(e))
    R["t1_rcwa_kernel"] = out


def tier2(R, A, tmp):
    out = {}
    for dev in devices():
        before = set(glob.glob(str(RUNS / "*")))
        rc, dt = run(py("run_topopt.py", "--config", "configs/metagrating_si1550_robust.json", "--iters", str(A.iters),
                        "--order", "10", "3", "--grid", "200", "40", "--set", "verify_order=[15,3]", "--device", dev,
                        "--seed", "1", "--tag", f"t2bench{dev}"), f"bench_t2_{dev}.out")
        new = [Path(p) for p in set(glob.glob(str(RUNS / "*"))) - before]; tmp += new
        try:
            s = json.loads((sorted(new, key=lambda p: p.stat().st_mtime)[-1] / "summary.json").read_text())
            out[dev] = dict(wall_s=dt, s_per_iter=s["s_per_iter"], iters=A.iters, fom_final=s["fom_final"])
        except Exception as e:
            out[dev] = dict(wall_s=dt, error=str(e))
    R["t2_topopt"] = out


def tier3(R, A, tmp):
    import torch
    dev = A.t3_device
    if dev is None:                       # the faster device of tier 1 at order 9 (consumer GPUs are slow in float64)
        k = {d: v.get("s_per_sim", {}).get("9") for d, v in R.get("t1_rcwa_kernel", {}).items()}
        k = {d: v for d, v in k.items() if v}
        dev = min(k, key=k.get) if k else ("cuda" if torch.cuda.is_available() else "cpu")
    lut, lens = "t3_lut_bench", "t3_lens_bench"; tmp += [RUNS / lut]
    t0 = time.time()
    rc1, dt1 = run(py("tools/tio2_lut.py", "--name", lut, "--period", "450", "--height", "1000", "--families", "circ", "square",
                      "--step", "20", "--lams", "700", "1000", "20", "--order", str(A.t3_order), "--device", dev), "bench_t3_lut.out")
    rc2, dt2 = run(py("tools/achro_tio2.py", "build", "--run", f"runs/{lut}", "--diameter-um", "100", "--na", "0.04",
                      "--name", lens, "--lams", "700", "850", "1000"), "bench_t3_lens.out")
    out = dict(device=dev, lut_wall_s=dt1, lens_wall_s=dt2, total_wall_s=round(time.time() - t0, 2))
    try:
        L = json.loads((RUNS / lens / "lens.json").read_text())
        out["eval"] = [dict(lam=e["lam"], eff_fixed=round(e["eff_fixed"], 4), strehl_fixed=round(e["strehl_fixed"], 4))
                       for e in L["eval"]["achromatic"]]
        out["n_atoms"] = len(L["atoms_used"])
    except Exception as e:
        out["error"] = str(e)
    R["t3_mini_lens"] = out
    if not A.keep_lens:
        tmp.append(RUNS / lens)


STRIP = ["validation/meep_achro_strip.py", "--run", "runs/t3_lens_bench", "--pol", "x", "--resolution", "40",
         "--x0-um", "25", "--width-um", "10", "--tmax", "300", "--out", "runs/t4_meep_bench"]


def tier4(R, A, tmp):
    if not (RUNS / "t3_lens_bench" / "lens.json").exists():
        R["t4_meep"] = dict(error="needs the tier-3 lens: run tier 3 first (or --keep-lens)"); return
    pts = []
    if MPI_SIZE > 1:                               # already inside mpirun (HPC job line): one point, all ranks run Meep
        import runpy
        t = time.time(); sys.argv = STRIP + ["--tag", f"np{MPI_SIZE}"]
        runpy.run_path(STRIP[0], run_name="__main__")
        pts.append(dict(np=MPI_SIZE, wall_s=round(time.time() - t, 2)))
    else:
        mpirun = shutil.which(A.mpirun) or A.mpirun
        for n in A.np:
            rc, dt = run([mpirun, "-np", str(n), sys.executable, "-u", *STRIP, "--tag", f"np{n}"], f"bench_t4_np{n}.out")
            pts.append(dict(np=n, wall_s=dt, rc=rc))
    for p in pts:
        try:
            j = json.loads((RUNS / "t4_meep_bench" / f"meep_strip_x_res40_np{p['np']}.json").read_text())
            p.update(voxels=j["voxels"], meep_time=j["meep_time"], T_total=[round(x, 4) for x in j["T_total"]])
        except Exception as e:
            p["error"] = str(e)
    R["t4_meep"] = dict(points=pts)
    tmp.append(RUNS / "t4_meep_bench")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default=platform.node() or "machine")
    ap.add_argument("--tiers", type=int, nargs="+", default=None, help="default: 0 1 2 3, plus 4 if Meep is importable")
    ap.add_argument("--orders", type=int, nargs="+", default=[5, 7, 9, 11])
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--t3-device", choices=["cpu", "cuda"])
    ap.add_argument("--t3-order", type=int, default=9, help=argparse.SUPPRESS)
    ap.add_argument("--np", type=int, nargs="+", default=None, help="MPI process counts for tier 4 (default 1 2 4 ... up to the cores)")
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--keep", action="store_true", help="keep the temporary *_bench runs")
    ap.add_argument("--keep-lens", action="store_true", help="keep runs/t3_lens_bench (needed for a later tier-4 run)")
    A = ap.parse_args()
    if A.tiers is None:
        A.tiers = [0, 1, 2, 3]
        try:
            import meep  # noqa: F401
            A.tiers.append(4)
        except Exception:
            pass
    if MPI_SIZE > 1:                      # inside mpirun only the Meep tier makes sense (all ranks take part)
        A.tiers = [4]
    os.chdir(HERE)
    if 4 in A.tiers and 3 not in A.tiers:
        A.keep_lens = True
    if 3 in A.tiers and 4 in A.tiers:
        A.keep_lens = True
    if A.np is None:
        A.np, n = [], 1
        while n <= (os.cpu_count() or 1):
            A.np.append(n); n *= 2
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in A.label)
    out = RUNS / f"bench_{safe}_{stamp}"
    R = dict(label=A.label, date=stamp, tiers=A.tiers, machine=machine_info() if MPI_RANK == 0 else {})
    tmp = []
    T0 = time.time()
    for t in sorted(A.tiers):
        if t == 4 and MPI_SIZE > 1:
            tier4(R, A, tmp); continue
        if MPI_RANK != 0:
            continue
        print(f"[benchmark] tier {t} ...", flush=True); ts = time.time()
        if t == 0:
            tier0(R, A)
        else:
            {1: tier1, 2: tier2, 3: tier3, 4: tier4}[t](R, A, tmp)
        print(f"[benchmark] tier {t} done in {time.time() - ts:.0f} s", flush=True)
    if MPI_RANK != 0:
        return
    R["total_wall_s"] = round(time.time() - T0, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(R, indent=1))
    if not A.keep:
        for p in tmp:
            if p.name == "t3_lens_bench" and A.keep_lens:
                continue
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else (p.unlink() if p.exists() else None)
    print(f"[benchmark] wrote {out / 'benchmark.json'} ({R['total_wall_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
