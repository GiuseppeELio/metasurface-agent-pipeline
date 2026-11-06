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
echo "tools/benchmark.py --label HPC --tiers 0 1 2 3 --keep-lens" > sweeps/bench_gpu.txt
echo "array sweeps/bench_gpu.txt" > queue/bench_gpu.req
# when it has finished:
echo "tools/benchmark.py --label HPC --tiers 4" > sweeps/bench_meep.txt
echo "meep sweeps/bench_meep.txt" > queue/bench_meep.req
```

## Collecting the results

Each machine writes `runs/bench_<label>_<date>/benchmark.json`, with no user names or host names. Copy the files
into one folder and run:

```bash
python tools/bench_report.py hpc/benchmark.json ws/benchmark.json win/benchmark.json --out runs/bench_report
```

Files with the same label are merged into one row. This covers tiers 0–3 and tier 4 run separately on one machine;
`--rename OLD=NEW` relabels a file. The first file is the reference for the agreement columns: the maximum phase difference of the RCWA coefficients and
the efficiency difference of the mini lens. The outputs are:

- `bench_table.md`;
- `bench_times.png`;
- `bench_rcwa_orders.png`;
- `bench_meep_scaling.png`;
- `bench_summary.json`.

Agreement is checked on deterministic quantities only: the RCWA coefficients of tier 1 and the lens efficiencies of
tier 3. On the reference node, CPU and GPU give the same RCWA coefficients to 1e-12. The figure of merit after 40
optimization iterations (tier 2) depends on floating-point details: CPU and GPU follow different optimization paths.
Tier 2 is therefore a timing measure only.

## Reference timings

Measured with v0.3.0 (development snapshots) on 5–6 November 2026.

| | HPC node | Workstation A | Workstation B | Mac mini M4 |
|---|---|---|---|---|
| CPU (threads used by torch) | Xeon Platinum 8260, 8 of 48 cores | Core i7-14700KF (14) | 2× Xeon Silver 4310 (24) | Apple M4, 4P + 6E cores (4) |
| GPU | Tesla V100S 32 GB | GeForce RTX 3080 10 GB | NVIDIA L4 24 GB | not used (NA)ᵇ |
| OS | Linux (SLURM) | Windows + WSL2 | Windows Server 2022: tiers 0–3 native, Meep in WSL2 | macOS 26, 16 GB |
| T0 self-test | 20 s | 18 s | 56 s | 20 s |
| T1 RCWA, s/simulation, order 9 (CPU / GPU) | 1.38 / **0.72** | **1.18** / 1.45 | 2.09 / **1.69** | 2.16 / NA |
| T1 RCWA, s/simulation, order 11 (CPU / GPU) | 3.53 / **1.48** | **3.19** / 3.85 | **4.59** / 4.62 | 6.45 / NA |
| T2 topology optimization, s/iteration (CPU / GPU) | 0.54 / **0.33** | **0.44** / 0.51 | 0.68 / **0.49** | 0.56 / NA |
| T3 mini achromatic lens (device chosen) | 6.6 min (GPU) | 9.8 min (CPU) | 15.1 min (GPU) | 17.8 min (CPU) |
| T4 Meep strip, fastest run | 1.0 min, 48 processes | 3.8 min, 16 processes | 10.2 min, 8 processesᵃ (1 process: 14.0 min) | 2.8 min, 2 processes (1: 3.3 min; 4: 4.2; 8: 14.9) |

- **Agreement.** All machines reproduce the reference numbers:
  - self-test 3/3 PASS;
  - tier-1 RCWA coefficients equal to 1e-12, CPU and GPU alike;
  - tier-3 lens efficiencies equal to four decimals;
  - tier-4 Meep transmissions equal to four decimals.
- **Tier 2.** The figure of merit after 40 iterations differs between devices: it is a timing measure only.
- **Mac mini M4.**ᵇ The Apple GPU is not used: TORCWA needs a complex double-precision eigensolver, which the
  PyTorch MPS backend does not provide. PyTorch uses the 4 performance cores. Meep is fastest with 1–2 processes and
  slows down beyond the performance cores (4 processes: 4.2 min; 8 processes: 14.9 min).
- **Workstation B, tier 4.**ᵃ Measured before the benchmark set `OMP_NUM_THREADS=1`.
- **Meep scaling.** The tier-4 problem is small (2.0 M voxels, one cell thick in y), so beyond a few MPI processes
  communication dominates. The Mac shows the same flat or worsening curve with one OpenMP thread per process. For
  production runs, which are 10–30 times larger, the cluster scaled well up to 48 processes. Tier 4 compares single
  machines; it is not a scaling study of the full problem.

Reference files: `docs/benchmark/*.json`. They are the original files, with only the label changed so that the GPU and
Meep runs of one machine merge into one row. Compare your machine with
`python tools/bench_report.py docs/benchmark/*.json runs/bench_<yours>/benchmark.json`.
