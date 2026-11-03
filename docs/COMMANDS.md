# Command reference (Appendix)

This appendix lists every user- and agent-facing command, **when** to use it and its most important
options. All commands run from the repository root. Python commands assume the environment created by
`hpc/setup_env.sh` (`source .venv/bin/activate`), or go through the bridge (`hpc/bridge.sh`).

## A. Execution bridge — `hpc/bridge.sh`

| Command | When | What it does |
|---|---|---|
| `login` | Slurm mode; start of a session, or when the watcher reports `ssh_down` | Runs `SSH_LOGIN_CMD` from `hpc/site.env`, e.g. a 2FA or SSH-certificate helper. Credentials are typed by the user only. |
| `push` / `pull` | Slurm mode; the watcher calls them automatically | rsync of code to the cluster / of `runs/` and `logs/` back. Checkpoints stay remote. |
| `setup` | Once per target, and after changing the torch wheel | Creates `.venv` (torch, torcwa, numpy, scipy, matplotlib). Uses `TORCH_INDEX`/`TORCH_SPEC` (match the GPU driver) and `ENV_MODULES`. |
| `setup-meep` | Once, if FDTD validation is needed | Installs Meep (MPI build, conda-forge) in `.meep-env` with a private micromamba. Needs internet. |
| `check` | After `setup`, and on a new site | Short GPU job: is CUDA visible? Times the complex eigendecomposition. |
| `submit <config> [args]` | A single optimization | Runs `run_topopt.py --config <config> [args]` on one GPU (Slurm job or local). |
| `array <list.txt>` | Sweeps, multi-start, libraries, pipelines of dependent steps | One task per line: `run_topopt.py` arguments, or a line starting with `<script>.py`. At most `MAX_PARALLEL_GPU` tasks run at the same time. |
| `meep <list.txt>` | Independent validation | One `validation/meep_validate.py` call per line (or a line starting with `<script>.py`), with MPI on a CPU node (`CPU_SBATCH`) or locally (`LOCAL_MPI_NP`). |
| `status`, `log <id>`, `cancel <id>` | Monitoring | `squeue`, last log lines, `scancel`. |
| `usage [YYYY-MM-DD]` | Budget tracking during the day | Sums elapsed × billing units of your jobs from `sacct`. Site budget tools are often updated only nightly. |
| `watch` | Agent mode, keep it running | Executes `queue/*.req` requests (only `submit`, `array`, `meep`, `check`; shell metacharacters rejected). Reports state in `queue/.heartbeat`, `.watcher_state` and `.squeue`. |

## B. Optimization — `run_topopt.py`

```
python run_topopt.py --config <cfg.json> [--device auto|cuda|cpu] [--iters N] [--order Nx Ny] [--grid nx ny]
                     [--seed S] [--tag T] [--set key.sub=value ...] [--init file] [--resume ckpt] [--outdir runs]
```

- **When:** any topology optimization of one periodic cell. That covers a grating or deflector, a single meta-atom, or a whole supercell.
- **Key config fields:**

| Field | Meaning |
|---|---|
| `period_nm`, `grid` | Cell size and pixel grid |
| `order` | Fourier truncation [Nx, Ny] |
| `verify_order` | Higher truncation used for the final convergence check |
| `wavelengths_nm` | One or several wavelengths; the FoM is averaged |
| `illumination` | `from_substrate` or `from_superstrate` |
| `substrate`, `superstrate`, `layer.core`, `layer.background` | Materials, as `{"n":..}` or `{"material":"name"}` |
| `layer.thickness_nm` | Layer thickness |
| `objective` | `diffraction_efficiency` (orders, port, `input_pols`), `diffraction_selectivity` (+ `suppress`, `penalty`) or `zero_order_phase` (`target_phase_deg`) |
| `objective.wavelength_weights` | Optional, one weight per `wavelengths_nm` entry: weighted sum instead of the mean (divided by the sum of the positive weights). For narrow-band targets such as quasi-BIC resonances, e.g. `[1545, 1550, 1555]` with `[-0.5, 1, -0.5]`. |
| `init_shape` | `disk`, `clover`, `cross`, `ring`, `square`, `ellipse` or `triangle`, with sizes in nm and `noise` |
| `optimizer.iters`, `lr`, `beta_start`, `beta_max` | Adam settings and tanh-projection continuation |
| `optimizer.blur_radius_nm` | Density filter |
| `optimizer.symmetry` | `none`, `x`, `y`, `xy` or `d4` (`d4` = polarization-independent square cell) |
| `optimizer.robust_mode` | `geometric` (with `edge_shift_nm`, `robust_blur_nm`) gives exact ±d nm eroded/dilated designs; the threshold variant uses `robust_eta` |
| `optimizer.softmin_k` | Sharpness of the worst-case soft-min |
| `optimizer.robust_start_frac` | Keep at 0 |
| `optimizer.eval_shifts_nm` | Edge shifts used in the final report |

- **Options:**
  - `--set` overrides any field without a new file, e.g. `--set layer.thickness_nm=900` or `--set "init_shape={...}"`. Inside list files, write it without quotes or spaces.
  - `--init` warm-starts from a `checkpoint.pt` (latent density) or an `.npy` design, e.g. a library supercell. Combine it with `optimizer.beta_start` (≈30) and a small `lr`.
  - `--resume` continues an interrupted run from its checkpoint.
- **Output** (`runs/<name>_<tag>_<time>_<jobid>/`):
  - `summary.json`: FoM, per-order table, convergence, robustness, feature size;
  - `rho_binary.npy`, `rho_final.npy`, `history.csv`, `design.png`, `fom.png`, `checkpoint.pt`.

## C. Library and supercell tools

| Command | When | Notes |
|---|---|---|
| `python disk_sweep.py --name N --period P --heights h1 h2 .. --radii start stop step [--n-sub n] [--order k]` | Before building a library: choose the height and initial radii | Writes `sweep.json` and `sweep.png` (T and phase versus R per height) and reports the 2π coverage with T > `--tmin`. |
| `python tools/make_lut_sweep.py --sweep runs/<sweep>/sweep.json --height h --step 15 --name atomLUT --shapes disk clover cross --supercells 8 12 > sweeps/lut.txt` | To generate the library job list | One run per phase × shape, plus selection and supercell lines. Submit it with `bridge.sh array`. |
| `python select_atoms.py --prefixes <atom prefixes> --out <lib> [--shift-window 10]` | After the atom runs | Picks one atom per target phase by the worst-case FoM within ±window nm, x and y together. Writes `runs/<lib>.json`. |
| `python supercell.py --name <sc> --library <lib> --levels N --order Mx My [--shifts -10 0 10]` | To assemble and verify a supercell | Picks atoms closest to k·360°/N. Reports the full-RCWA +1 efficiency for x and y, the library (phase-sampling) prediction, all orders, and edge-shift robustness. Writes `rho_supercell.npy` for `--init`. |
| `python metalens.py --library <lib> --diameter-um D (--na NA \| --focal-um f) --name <lens>` | Metalens from a library | Hyperbolic phase, one atom per cell (best `sqrt(T)·cos Δφ`). The vector angular-spectrum model gives the axial profile, longitudinal sections, depth of focus, FWHM, focusing efficiency and Strehl ratio. Writes `lens.json`, `rho_lens.npz` (layout) and figures. |
| `python evaluate.py <run prefixes> [--shifts ...]` | Fair comparison of finished runs | Re-evaluates binary designs at `verify_order` with exact edge shifts. Writes `robustness_eval.json` per run plus a summary. |

## C2. Materials and self-test

| Command | When | Notes |
|---|---|---|
| `python tools/fetch_materials.py [--from <dir\|zip>]` | Once, before using tabulated materials | Downloads the refractiveindex.info database (CC0, about 30 MB) from GitHub into `materials/refractiveindex.info-database/` (git-ignored). `--from` installs it from a copy you already have. The agent tells the user before downloading. |
| `python tools/fetch_materials.py --search <text>` / `--show <shelf/book/page> --lam λ1 λ2` / `--where` | To choose a data set | Lists the pages; prints range, REFERENCES, COMMENTS and n, k. Use it in a config as `{"material": "rii:<shelf/book/page>"}`. Interpolation is PCHIP (k ≥ 0) and there is no extrapolation outside the data range. |
| `python tools/selftest.py [--full] [--keep]` | After `setup`, after updating torch/torcwa/numpy, before a release | Reproduces the reference numbers of the worked example (`tools/selftest_reference.json`) on the CPU: the robust grating re-evaluation, the 8-level supercell and the NA 0.5 metalens (about 1 min). `--full` adds a short optimization from scratch (3–5 min). Outputs are named `*_selftest` and removed at the end. The exit code is 0 when every check passes. |

## D. Validation and figures

| Command | When | Notes |
|---|---|---|
| `mpirun -np P python validation/meep_validate.py --run runs/<dir> --resolution R [--pols x y]` (or `bridge.sh meep list.txt`) | Final designs | 3-D FDTD with normal incidence from the input medium. Gives the efficiency of every propagating transmitted order (`DiffractedPlanewave`, s+p), the total transmitted flux, and the RCWA reference from the same directory. Run at least two resolutions; start with a low-resolution smoke test. |
| `mpirun -np P python validation/meep_lens.py --run runs/<lens> --resolution R [--crop-cells N]` (a `validation/meep_lens.py ...` line in a meep list) | Full-wave check of a metalens | 3-D FDTD of the whole aperture (or of the central N×N cells), focus via near-to-far. Reports T, focus position, FWHM and focusing efficiency next to the `metalens.py` model. |
| `python validation/lens_post.py --run runs/<lens> [--res R]` | After `meep_lens.py` | Propagates the Meep near field with the model's vector angular-spectrum code. Gives xz/yz sections, depth of focus (axial FWHM), lateral FWHM and focusing efficiency, Meep vs model (`lens_post_res<R>.json`, `.png`). |
| `python validation/rcwa_reference.py --run runs/<dir> [--order Nx Ny] [--shifts 0 -10 10]` | RCWA side of a validation round | Per-order T/R at the verification truncation, nominal and edge-shifted. Writes `rcwa_ref.json`. |
| `python validation/torcwa_fields.py --run runs/<dir>` | Field comparison | RCWA near fields in the xz plane, in `fields_rcwa.npz`. |
| `python validation/compare_fields.py --run runs/<dir>` | Field comparison | Near-field maps, beam propagation and per-order efficiencies (% of incident power) for both solvers. |
| `python validation/validate.py plan --runs ... --level quick\|full --tag T [--enqueue]` | Validation stage, step 1 | Writes the GPU list and the Meep list for one round, from `configs/validation_policy.json`, and estimates the Meep time. |
| `python validation/validate.py report --runs ... [--figures]` | Validation stage, step 2 | Verdict per design, edge shift and polarization (`agree`, `marginal`, `inconclusive`, `disagree`), plus the rank agreement. Writes `validation.json`. |
| `python paper/make_figures.py` | After the runs | Rebuilds the figures in `paper/figures/` from `runs/`. |

## E. Typical sessions

1. **New site:**
   1. agent onboarding, which creates `hpc/site.env`;
   2. `login`, `push`, `setup`;
   3. `watch`;
   4. agent request `check`.
2. **Grating design:**
   1. agent writes a config;
   2. `submit`;
   3. thickness/seed sweep with `array`;
   4. robust re-optimization;
   5. `evaluate.py`;
   6. validation stage: `validate.py plan`, then `array` + `meep`, then `validate.py report`.
3. **Metasurface:**
   1. `disk_sweep.py`;
   2. `tools/make_lut_sweep.py`, then `array`;
   3. `select_atoms.py` and `supercell.py` (included at the end of the list);
   4. supercell refinement with `--init`;
   5. validation stage;
   6. optionally `metalens.py`, then `meep_lens.py`.
4. **First contact:** the ten-minute tour in `examples/README.md`.
