# Cross-machine benchmark

`tools/benchmark.py` runs the same small workloads on any machine and writes one `benchmark.json`. Each tier stores
its timing and its physical result, so machines can be compared for speed and for agreement.
`tools/bench_report.py` turns several files into a table and three figures.

| Tier | Workload | What it measures | Needs |
|---|---|---|---|
| 0 | `tools/selftest.py` | installation, reproducibility of the worked example | CPU |
| 1 | 8 TiO2 meta-atoms × 3 λ at Fourier orders 5, 7, 9, 11 | seconds per RCWA simulation vs order; CPU and GPU | CPU (+ CUDA GPU) |
| 2 | 40 iterations of the robust Si grating (`run_topopt.py`) | seconds per optimization iteration; CPU and GPU | CPU (+ CUDA GPU) |
| 3 | 30 TiO2 pillars × 16 λ (700–1000 nm, order 9), then a D = 100 µm, NA 0.04 achromatic lens | end-to-end time, focusing efficiency | the faster device of tier 1 |
| 4 | 10 µm strip of the tier-3 lens in Meep, 40 px/µm, 700–1000 nm | Meep wall time vs MPI processes | Meep (Linux, macOS, cluster) |

The RCWA kernel works in complex double precision: its cost is dominated by a dense eigenvalue problem of size
2(2N+1)². Consumer GPUs (GeForce RTX) are slow in double precision, so tier 1 also runs on the CPU. Tier 3 then uses
whichever device was faster.

## Getting the code, or updating an existing copy

- First time: `git clone https://codeberg.org/GiuseppeELio/metasurface-agent-pipeline.git`. A mirror is on GitHub
  under the same name.
- Already cloned: from the repository folder, run `git pull`. Your `runs/`, `logs/`, `queue/` and `hpc/site.env` are
  git-ignored and stay untouched.
- With an AI agent, ask it to update the repository with `git pull` and then run the benchmark following this file.

The benchmark is also meant for everyday use: run it once on each new machine to know what to expect. The agent uses
`benchmark.json` to estimate run times and to choose between CPU, GPU and cluster (`AGENTS.md`, onboarding step 6c).

## Linux or macOS workstation (with or without NVIDIA GPU, with Meep)

```bash
git clone https://codeberg.org/GiuseppeELio/metasurface-agent-pipeline.git && cd metasurface-agent-pipeline
bash hpc/setup_env.sh                  # .venv with torch (CUDA if an NVIDIA driver is found), torcwa, numpy, scipy, ...
bash hpc/setup_meep.sh                 # optional, tier 4: private Meep environment in .meep-env (2-3 GB)
.venv/bin/python tools/benchmark.py --label "WS RTX3080 16c" --tiers 0 1 2 3 --keep-lens
.meep-env/bin/python tools/benchmark.py --label "WS RTX3080 16c" --tiers 4 --np 1 2 4 8 16 --mpirun .meep-env/bin/mpirun
```

## Windows, without WSL (tiers 0–3)

In PowerShell, from the repository folder (Python 3.10–3.12 from python.org):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126   # NVIDIA GPU; omit the index for CPU only
.venv\Scripts\python -m pip install torcwa numpy scipy matplotlib pyyaml
.venv\Scripts\python tools\benchmark.py --label "WIN <gpu name>"
```

If `nvidia-smi` reports a driver older than CUDA 12, use `https://download.pytorch.org/whl/cu118`.
Meep has no native Windows build, so tier 4 is skipped automatically.

## HPC through the bridge (`hpc/bridge.sh watch` running)

One GPU job for tiers 0–3, then one CPU job for tier 4. Tier 4 runs inside the job's `mpirun`, so it gives one point
with np = the job's MPI ranks.

```bash
echo "tools/benchmark.py --label HPC-GPU --tiers 0 1 2 3 --keep-lens" > sweeps/bench_gpu.txt
echo "array sweeps/bench_gpu.txt" > queue/bench_gpu.req
# when it has finished:
echo "tools/benchmark.py --label HPC-CPU --tiers 4" > sweeps/bench_meep.txt
echo "meep sweeps/bench_meep.txt" > queue/bench_meep.req
```

## Collecting the results

Each machine writes `runs/bench_<label>_<date>/benchmark.json`, with no user names or host names. Copy the files
into one folder and run:

```bash
python tools/bench_report.py hpc/benchmark.json ws/benchmark.json win/benchmark.json --out runs/bench_report
```

The first file is the reference for the agreement columns: the maximum phase difference of the RCWA coefficients and
the efficiency difference of the mini lens. The outputs are:

- `bench_table.md`;
- `bench_times.png`;
- `bench_rcwa_orders.png`;
- `bench_meep_scaling.png`;
- `bench_summary.json`.

## Expected run times (indicative)

| Tier | 2-core cloud VM (measured) | V100 node (measured) | 16-core workstation |
|---|---|---|---|
| 0 | 24 s | | about 20 s |
| 1 | 0.20 / 1.0 / 4.5 s per simulation (orders 5 / 7 / 9) | 0.7 s (order 9), 1.5 s (order 11) on the GPU | |
| 2 | 1.7 s per iteration | | |
| 3 | about 35 min at order 9 | about 10 min | 10–20 min |
| 4 | – | | np = 1: about 15–20 min |

Fill the empty cells with the measured values: they are what the paper reports.
