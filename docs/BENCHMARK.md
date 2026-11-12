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

Measured with v0.3.0 (development snapshots) on 5–6 November 2026; Leonardo with v0.3.1 on 12 November 2026.

| | HPC node (Galileo100) | Leonardo Booster nodeᶜ | Workstation A | Workstation B | Mac mini M4 |
|---|---|---|---|---|---|
| CPU (threads used by torch) | Xeon Platinum 8260, 8 of 48 cores | Xeon Platinum 8358, 8 of 32 cores | Core i7-14700KF (14) | 2× Xeon Silver 4310 (24) | Apple M4, 4P + 6E cores (4) |
| GPU | Tesla V100S 32 GB | A100-SXM 64 GB | GeForce RTX 3080 10 GB | NVIDIA L4 24 GB | not used (NA)ᵇ |
| OS | Linux (SLURM) | Linux (SLURM) | Windows + WSL2 | Windows Server 2022: tiers 0–3 native, Meep in WSL2 | macOS 26, 16 GB |
| T0 self-test | 20 s | 17 s | 18 s | 56 s | 20 s |
| T1 RCWA, s/simulation, order 9 (CPU / GPU) | 1.38 / **0.72** | 1.05 / **0.49** (eig on GPU: 1.19) | **1.18** / 1.45 | 2.09 / **1.69** | 2.16 / NA |
| T1 RCWA, s/simulation, order 11 (CPU / GPU) | 3.53 / **1.48** | 2.71 / **1.00** (eig on GPU: 2.25) | **3.19** / 3.85 | **4.59** / 4.62 | 6.45 / NA |
| T2 topology optimization, s/iteration (CPU / GPU) | 0.54 / **0.33** | 0.36 / **0.24** (eig on GPU: 0.45) | **0.44** / 0.51 | 0.68 / **0.49** | 0.56 / NA |
| T3 mini achromatic lens (device chosen) | 6.6 min (GPU) | 4.5 min (GPU) | 9.8 min (CPU) | 15.1 min (GPU) | 17.8 min (CPU) |
| T4 Meep strip, fastest run | 1.0 min, 48 processes | 0.45 min, 112 processes (56: 0.63 min)ᶜ | 3.8 min, 16 processes | 10.2 min, 8 processesᵃ (1 process: 14.0 min) | 2.8 min, 2 processes (1: 3.3 min; 4: 4.2; 8: 14.9) |

- **Agreement.** All machines reproduce the reference numbers:
  - self-test 3/3 PASS;
  - tier-1 RCWA coefficients equal to 1e-12, CPU and GPU alike;
  - tier-3 lens efficiencies equal to four decimals;
  - tier-4 Meep transmissions equal to four decimals.
- **Tier 2.** The figure of merit after 40 iterations differs between devices: it is a timing measure only.
- **Mac mini M4.**ᵇ The Apple GPU is not used: TORCWA needs a complex double-precision eigensolver, which the
  PyTorch MPS backend does not provide. PyTorch uses the 4 performance cores. Meep is fastest with 1–2 processes and
  slows down beyond the performance cores (4 processes: 4.2 min; 8 processes: 14.9 min).
- **Leonardo.**ᶜ Tiers 0–3 on a Booster node with `METASURF_EIG_DEVICE=cpu` (see below; the values with PyTorch's
  default, eig on the GPU, are in parentheses). Tier 4 on a DCGP CPU node (Xeon Platinum 8480+, 112 cores).
- **Workstation B, tier 4.**ᵃ Measured before the benchmark set `OMP_NUM_THREADS=1`.
- **Meep scaling.** The tier-4 problem is small (2.0 M voxels, one cell thick in y), so beyond a few MPI processes
  communication dominates. The Mac shows the same flat or worsening curve with one OpenMP thread per process. For
  production runs, which are 10–30 times larger, the cluster scaled well up to 48 processes. Tier 4 compares single
  machines; it is not a scaling study of the full problem.

## Hybrid CPU/GPU placement of the eigensolver

Most of the cost of RCWA is `torch.linalg.eig` on dense complex matrices. On CUDA this is a hybrid routine: part of it
runs on the CPU cores of the job, and it is very sensitive to the number of OpenMP threads. On the Leonardo Booster
node it was slower than MKL alone on the 8 cores of the job:

| eig, n = 1054 (Leonardo, 8 OpenMP threads) | complex64 | complex128 |
|---|---|---|
| A100 (CUDA, hybrid) | 828 ms | 1498 ms |
| 8 Xeon 8358 cores (MKL) | **469 ms** | **949 ms** |

With 32 threads the CUDA eig became 5 times slower: 4421 ms for complex64. With PyTorch's default the A100 was
therefore slower than the V100S of Galileo100, and at order 9 slower than its own CPU (tier 3 then chose the CPU).

`METASURF_EIG_DEVICE=cpu` computes only the eigenvalue problem on the CPU and keeps everything else on the GPU. The
results are identical. On Leonardo it gave:

- a TORCWA forward + backward 1.4 times faster (847 ms against 1188 ms at [15,8]);
- an A100 1.36–1.47 times faster than the V100S in tiers 1–3.

Rules:

- Run the benchmark on a new GPU machine twice, with `--eig-device gpu` and `--eig-device cpu`, and keep the faster
  setting. On a cluster put it in `GPU_SBATCH`: `--export=ALL,METASURF_EIG_DEVICE=cpu`.
- Give each GPU job about 8 cores and never more OpenMP threads than cores.
- Scale out with independent jobs (SLURM arrays), not with one run over several GPUs. On Leonardo, 12 robust
  metagrating optimizations at [25,5] ran as 4 parallel jobs in 58 min (2.7–3.7 s per iteration, against 3.75 s on
  the V100S).

Reference files: `docs/benchmark/*.json`. They are the original files, with only the label changed so that the GPU and
Meep runs of one machine merge into one row. Compare your machine with
`python tools/bench_report.py docs/benchmark/*.json runs/bench_<yours>/benchmark.json`.
