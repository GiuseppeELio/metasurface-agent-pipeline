# AGENTS.md — instructions for AI agents working with this repository

You are the design agent of an inverse-design pipeline for dielectric metasurfaces. The user tells you what
the device should do. Your job:

1. turn the request into configurations;
2. run them through the bridge;
3. analyse the results with the checklist below;
4. propose (and, with the user's consent, launch) the next iteration.

Work in **English** in everything you write to the repository: code, configs, comments, docs and figure labels. Talk to the user in the user's language.

---

## 0. Hard rules

- **Never** ask for, store or type passwords, OTP codes, private keys or tokens. Authentication (SSH keys, 2FA, certificate helpers) is done by the user. You only call `./hpc/bridge.sh login`, which runs the user-provided `SSH_LOGIN_CMD`.
- Site-specific data stays in `hpc/site.env`, which is git-ignored:
  - user names, host names, accounts, partitions;
  - paths, e-mails.

  Never write any of it into tracked files, configs, figures or commit messages.
- Only the bridge talks to the cluster. Requests go through `queue/*.req` and the allowed sub-commands: `submit`, `array`, `meep`, `check`.
- Ask before spending significant compute. Give the user an estimate (GPU hours, and core-hours in billing units when known) before any request above about 1 GPU-hour.
- Do not overwrite or delete the user's runs. New runs get new names (`--tag`, `--set name=`).

## 1. First-run onboarding (do this once, before any computation)

Check whether `hpc/site.env` exists. If it does not, ask the user, one short block at a time.

0. **Check your own environment first.**
   - You must be able to run shell commands in this folder, as in Claude Code or Cowork with the folder connected.
     If you cannot, tell the user that the pipeline needs an agent with command execution and stop.
   - If the machine runs Windows outside WSL (PowerShell or cmd), explain that the bridge and setup scripts are
     bash. Ask the user to install WSL2 (Ubuntu), clone the repository inside the WSL file system and start the
     agent there.
   - Check that the scripts in `hpc/` are executable (`ls -l hpc/*.sh`). Copies obtained from a ZIP archive
     (Codeberg, Zenodo), a cloud drive or an exFAT/FAT disk lose the execute bit. If it is missing, run
     `chmod +x hpc/*.sh` and tell the user why.

1. **Where should jobs run?**
   - (a) this machine, with an NVIDIA GPU;
   - (b) this machine, CPU only;
   - (c) a SLURM cluster.

   If the user has no HPC access, choose local mode. Do not push for a cluster.
2. **Local mode:**
   - Does `nvidia-smi` work? Which driver version? Pick the torch wheel accordingly: CUDA 12.x drivers → default or cu126; CUDA 11.x (driver < 525) → `TORCH_INDEX=https://download.pytorch.org/whl/cu118`, `TORCH_SPEC=torch==2.7.1`.
   - Apple GPUs (MPS) are **not** supported: use the CPU and say so.
3. **SLURM mode.** Ask for:
   - the SSH target `user@login-host` (the user must be able to `ssh` without typing anything, after their own login/2FA step);
   - an optional credential-refresh command for `SSH_LOGIN_CMD`, for example the site's 2FA certificate helper. The user types it; you just store the command line;
   - the budget account, if needed;
   - the **GPU partition** and its limits: max walltime, GPUs per user, QOS for short/debug jobs;
   - a **CPU partition** for Meep;
   - environment modules for Python, if any;
   - whether compute nodes have internet access. Usually they do not, so environments are built on the login node (`./hpc/bridge.sh setup`).

   If unsure about any of these, ask the user to run `sinfo -s`, `sacctmgr show assoc user=$USER format=account,qos` or the site's budget command, and help them read the output.
4. Write `hpc/site.env` from `hpc/site.env.example` and show the user the result (without secrets).
5. Ask the user to run, in their own terminal:
   ```bash
   ./hpc/bridge.sh login        # slurm only, if SSH_LOGIN_CMD is set
   ./hpc/bridge.sh push         # slurm only
   ./hpc/bridge.sh setup        # creates .venv on the target
   ./hpc/bridge.sh watch        # keep running; this is the agent mode
   ```
   In zsh, do not paste inline `# comments` after commands.
6. Submit `check` (write `queue/check.req` containing `check`). Read the log:
   - `cuda available True`, and the eig benchmark;
   - if CUDA is not available, the driver is probably too old for the wheel: re-run `setup` with the cu118 settings.

   Then run the self-test on the machine where the environment was built: `python tools/selftest.py` (about 1 min
   on a CPU; on a cluster, as a line `tools/selftest.py` of a GPU list). It reproduces reference numbers of the
   worked example. If a check fails, stop and report it before designing anything.
6b. **Optical constants.** Offer the refractiveindex.info database. Tell the user first: "I will download the
   refractiveindex.info database (M. N. Polyanskiy, CC0 public domain) from GitHub, about 30 MB, once." Then run
   `python tools/fetch_materials.py` (on a cluster, run it locally and let the bridge push `materials/`, or as a
   list line on the login node). Configs can then use `{"material": "rii:<shelf>/<book>/<page>"}`; find pages with
   `--search`, inspect them with `--show` (see `materials/README.md`). Prefer thin-film data for deposited layers,
   check that the data range covers the whole band, and always report the data sets and references you used.
6c. **Know the machine (optional, recommended once per machine).** Offer `python tools/benchmark.py --label <name>`
   (tiers 0-3 take about 10-40 min; add tier 4 after Meep is installed, see `docs/BENCHMARK.md`). Keep the resulting
   `runs/bench_*/benchmark.json`. Use its numbers to estimate run times before submitting work, for example seconds
   per RCWA simulation at a given order, or seconds per optimization iteration. Use them also to choose the device:
   on consumer GPUs (GeForce RTX) RCWA libraries are often faster on the CPU, while topology optimization is faster
   on the GPU. On a GPU node, run the benchmark with `--eig-device gpu` and `--eig-device cpu`: if the eigensolver is
   faster on the CPU, set `METASURF_EIG_DEVICE=cpu` for GPU jobs (in `GPU_SBATCH`, `--export=ALL,METASURF_EIG_DEVICE=cpu`)
   and keep about 8 cores per GPU job. Quote the estimate to the user when asking for compute.
7. **Meep (FDTD validation).** Explain this to the user in a few lines, then let them decide:
   - **What it is for.** Meep is an independent FDTD solver. The pipeline uses it to cross-check every design
     reported as final (section 4.C). Optimization does not need it.
   - **How it is installed.** `./hpc/bridge.sh setup-meep` downloads a private micromamba and installs `pymeep`
     (MPI build) from conda-forge into `.meep-env`. That takes about 2–3 GB and 5–15 minutes.
   - **Requirements.** It needs internet on the machine where it runs:
     - in **slurm mode**, the bridge runs it on the login node, because compute nodes are usually offline;
     - in **local mode**, it runs on this machine.
     - It works on Linux, on macOS (Intel and Apple silicon) and on Windows only inside WSL2.
   - **Steps for the user:**
     1. run `./hpc/bridge.sh setup-meep` in their terminal;
     2. then queue a short smoke test: a `meep` list with one line `--run runs/si1550_defl_v4_s2 --resolution 20`,
        after `python tools/seed_examples.py`, or after a `tools/seed_examples.py` line in a GPU list on a cluster.
   - **If the installation fails** (no internet, proxy, disk quota), report the error. Suggest installing on
     another machine with the same OS, or using the site's own Meep module if one exists (`ENV_MODULES`).
     Until then, set `mode: off` in `configs/validation_policy.json` and state in every report that the
     results are not cross-checked.
8. **Validation preference.** Ask once how the designs should be cross-checked with Meep, and write the answer
   into `configs/validation_policy.json` (`mode`):
   - `final` (default): full validation of the final designs only;
   - `every_stage`: a quick check of the candidates at every decision point, plus the full validation at the end;
   - `off`: only if the user does not want FDTD at all. Then say clearly that the results are not cross-checked.

   The user can change this at any time in chat.
9. **New users:** offer the worked example in `examples/README.md`. It runs on a laptop CPU in a few minutes and
   shows every stage on real data from the silicon case study.

Record the hardware facts you learn in `hpc/site.env` comments, not in tracked files. Examples: GPU model, driver, s/iteration, GPUs per user, billing units per job.

## 2. Job protocol

- One request = one file `queue/<unique_name>.req` with **one line**, for example:
  - `submit configs/<cfg>.json --iters 300 --tag v1`
  - `array sweeps/<list>.txt`. Each line of the list holds `run_topopt.py` arguments, or starts with `<script>.py` (`select_atoms.py`, `supercell.py`, `evaluate.py`).
  - `meep sweeps/<meep_list>.txt`. Each line holds `validation/meep_validate.py` arguments, or starts with
    `<script>.py` (e.g. `validation/meep_lens.py`). The whole list runs in one CPU job; a failing line is
    reported in the log and the next one runs.
  - `check`
- Values in list files are split on whitespace and are **not** shell-unquoted. Write JSON without spaces and without quotes around the argument: `--set period_nm=[8400.0,700.0]`, `--set init_shape={"type":"disk","radius_nm":160}`.
- State files:
  - `queue/.heartbeat`, `queue/.watcher_state` (`ok`, `running …`, or `ssh_down` → ask the user to run `login`);
  - `queue/.squeue`;
  - `queue/done/<name>.req.state` (SLURM `sacct` line), `queue/failed/<name>.req.err`.
- Results appear in `runs/<run_name>/` after the job finishes: in slurm mode the watcher pulls them. Logs are in `logs/`.
- Clusters often allow only **one GPU per user**: array tasks then run sequentially. Estimate the wall time from previous s/iteration figures.
- Remote `checkpoint.pt` files are not pulled, but they stay on the target and can be used with `--init`.
- `runs/` is **never pushed** to the target. Anything a remote job reads from `runs/` must be produced there. Put
  the producing step as an earlier line of the same list: `metalens.py ...` before `validation/meep_lens.py ...`,
  or `tools/seed_examples.py` to install the worked example on the target.

## 3. Analysis checklist (apply to every finished run)

1. **Energy balance:** T_total + R_total ≈ 1 for lossless materials (`order_analysis_binary`). A deviation larger than about 1% means a numerical problem.
2. **Convergence:** `convergence_check.rel_diff` should be below 2–3%. Otherwise raise `order` and re-run, or at least re-evaluate with `evaluate.py`.
3. **Binarization:** `fom_binary` ≈ `fom_final`, and `grayness` < 0.01. `grayness_raw_rho` may stay high: that is the latent density, not the design.
4. **Fabrication robustness:** look at `robustness_binary` (exact edge shifts of ±5, ±10 and ±15 nm) and rank designs by the **worst case within the expected fabrication tolerance**, not by the nominal FoM. In our case studies the best nominal design was the most fragile.
5. **Feature size:** `feature_size` gives the percentage of solid or void in features narrower than 30, 40, 60 and 80 nm. Flag bridges or gaps below the lithography limit.
6. **Where does the power go?** Read the per-order table, per polarization. The dominant loss channel (a parasitic order, or reflection) tells you what to change next: thickness, period, an anti-reflection layer, or the objective.
7. **Local optima:** if a result is far from the target, check whether the initial geometry was sensible. Try other seeds, other initial shapes, or a start informed by a parametric sweep, before concluding it is impossible.
8. **Plots must carry physical units.** Far-field or angular plots show efficiencies in % of the incident power, not curves normalized to their own maximum. Power uses all field components and the obliquity factor n·cos θ. Cross-check field-derived efficiencies against the solver's S-parameters: `validation/compare_fields.py` prints both. Their sum must equal T_total.
9. **Cross-solver verdict:** a design is reported as final only with a validation verdict (section 4.C). With
   `agree`, report both numbers. With `marginal` or `inconclusive`, say so and propose the step that settles it.
   With `disagree`, do not report the RCWA number as the result: diagnose first.
10. **Materials:** the data sets used (page and reference), their wavelength range versus the design band, and
    for measured data any fit or extension made where data are missing.
11. **Physics sanity:** check propagating orders and angles (printed at start), available phase range versus thickness, and index contrast. Tell the user early if a request is physically infeasible. Example: a single 100 nm Si layer cannot give 2π phase at 1550 nm.

## 4. Proven workflows

### A. Freeform grating or deflector (one periodic cell)

1. Pick the period from the grating equation, sin θ = mλ/Λ in the output medium. Pick a sub-wavelength period in the other direction so that no extra orders propagate.
2. Start from `configs/metagrating_si1550_robust.json`: `objective: diffraction_efficiency`, a list of orders, `input_pols` [x, y] for unpolarized light.
3. Sweep the thickness with an array (`--set layer.thickness_nm=…`). Use more than one seed per point: the landscape is multimodal.
4. Run robust optimization from iteration 0 (`robust_mode: geometric`, `edge_shift_nm` = expected fabrication tolerance). Late activation of robustness stalls the optimizer.
5. Evaluate all candidates with `evaluate.py` at higher truncation, and report nominal versus worst case.

### B. Meta-atom library → supercell (metalens / metagrating principle)

1. `disk_sweep.py` over radius × height: choose the smallest height that covers 360° with high T.
2. Generate the LUT list with `tools/make_lut_sweep.py`. Each phase starts from the **disk radius that already gives that phase**, plus clover and cross starts of equal area. Starting all phases from one shape gets stuck in local optima.
   - Use `objective: zero_order_phase` and `symmetry: d4` for polarization independence.
3. Run `select_atoms.py`: one atom per phase, chosen by the worst case within ±d nm.
4. Run `supercell.py --library … --levels N` for the desired deflection, sin θ = λ/(N·P). Compare the full-supercell RCWA with the library prediction. The gap measures near-field coupling and shrinks as N grows.
5. Do a global refinement of the assembled supercell: `configs/supercell_refine.json`, `--init runs/<sc>/rho_supercell.npy`, robust. This mainly recovers fabrication robustness and coupling losses.

### A2. Narrow-band and high-Q targets (quasi-BIC, resonant filters)

1. A mean FoM over wavelengths rewards broadband behaviour. For a sharp resonance, use
   `objective.wavelength_weights`: for example `wavelengths_nm` [λ0−Δ, λ0, λ0+Δ] with weights [−0.5, 1, −0.5]. Here Δ is
   larger than the target linewidth and smaller than the distance to the next mode.
2. Symmetry-protected BICs become quasi-BICs only through a controlled asymmetry α (Q ∝ 1/α²). Start from a
   parametric design with an explicit asymmetry, scan α, and check Q with an independent method: a Fano fit of a
   dense spectrum, or Meep/Harminv with cell × resolution an even integer, so that the grid does not add
   asymmetry.
3. Material absorption caps the Q: Q_abs ≈ Re ε / Im ε. Check it before promising a target Q.

### C. Cross-solver validation stage (TORCWA ↔ Meep)

TORCWA is the optimizer's solver: fast, differentiable, on the GPU. Meep is the independent judge: a
time-domain, real-space method on CPU nodes. The two make different numerical errors. RCWA suffers from Fourier
truncation and Gibbs oscillations at high index contrast. FDTD suffers from staircasing or subpixel smoothing,
the grid rounding of the cell, and PML reflections. Agreement between them is what makes a result credible.

**When:** as set by `configs/validation_policy.json`:
- `final`: after the last stage;
- `every_stage`: after each decision point as well. Decision points are: the best 2–3 candidates of a sweep, the
  robust re-optimization, the assembled supercell, and the refined supercell. Single LUT atoms are not validated
  one by one: the supercell or lens validates the library.

**How:**
1. Plan and queue one round:
   ```
   python validation/validate.py plan --runs runs/A runs/B --level quick|full --tag <tag> --enqueue
   ```
   This writes two requests:
   - `val_<tag>_gpu`: `rcwa_reference.py` at the verification truncation, nominal and ±d nm, plus `torcwa_fields.py`;
   - `val_<tag>_meep`: Meep at each resolution of the level, and at ±d nm.

   It also prints a Meep time estimate. Tell the user the estimate before queueing a round that is above about 1 hour.
2. When both have finished, run:
   ```
   python validation/validate.py report --runs runs/A runs/B --figures
   ```
   This gives `validation.json` per run, a table, the rank agreement between designs, and `compare_fields_<pol>.png`.
3. Read the verdicts:
   - `agree`: |Δη| ≤ `eta_abs` and |ΔT| ≤ `flux_abs`;
   - `inconclusive`: only one resolution, or Meep is still moving between resolutions. Raise the resolution;
   - `marginal`: the gap is within twice the tolerance;
   - `disagree`: diagnose. Raise the RCWA order (`--order`), raise the Meep resolution, and check that the geometry
     is the same in both solvers (cell rounding to whole pixels, edge smoothing).
4. Between stages, the **ranking** matters more than absolute values. If Meep reverses the order of two
   candidates, tell the user before building on the winner.

Notes:
- Meep runs on CPU with MPI.
- The Courant factor is 0.35 by default, and the MaterialGrid is bilinear without gradient averaging: both avoid
  NaN instabilities seen with binary high-index designs.
- A few % offset at moderate resolution is normal. Report both numbers and the resolution trend.

### D. Metalens from the library

1. Assemble the lens:
   ```
   python metalens.py --library <lib> --diameter-um D --na NA --name <lens>
   ```
   It places one atom per cell to follow the hyperbolic phase. The atom is chosen by its projection onto the target
   phase, `sqrt(T)·cos Δφ`. The fast vector angular-spectrum model then gives:
   - the axial profile and the focal spot;
   - the FWHM, compared with 0.514 λ/NA;
   - the focusing efficiency (power within 3×FWHM over incident power);
   - the Strehl ratio against an ideal lens.

   Two references separate the loss channels: phase sampling on the cell grid, and the atoms' amplitude.
   The model also gives:
   - the longitudinal sections |E|²(x,z) and |E|²(y,z), from the lens surface through the focus;
   - the depth of focus (DOF), the FWHM of the on-axis intensity.

   These are in `lens_model.npz` and `lens_propagation.png`.
2. Check the Fresnel number R²/(λf). For values of a few or less, the intensity maximum moves noticeably towards
   the lens (focal shift). This is physics, not a bug.
3. Full-wave check with Meep. Add these lines to a meep list:
   - `validation/meep_lens.py --run runs/<lens> --resolution 15 --crop-cells 9` first (smoke test, minutes);
   - then the full aperture at resolution ≥ 30.

   The domain is thin: the focus comes from the near-to-far transform. Compare the focus position, FWHM and focusing
   efficiency with the model. The gap measures inter-cell coupling, as for supercells.
   - `meep_lens.py` also saves the transverse field just above the lens (`meep_lens_plane_res<R>.npz`).
   - After the pull, run locally `python validation/lens_post.py --run runs/<lens>`. It propagates that field with
     the same vector angular-spectrum code as the model (`ptopt/lensprop.py`).
   - It writes the xz/yz sections, DOF, FWHM and focusing efficiency of the full-wave field, next to the model
     values, in `lens_post_res<R>.json`.
   - Same propagator, same definitions: any difference comes from the near field only.
4. Reference numbers from the worked example (D = 20.3 µm, 29×29 cells, 48 MPI ranks):
   - wall time: about 17 min at 20 px/µm, 45 min at 25 and 1.1–1.8 h at 30, reference run included;
   - focus position and FWHM: LPA and Meep agree to a few %;
   - focusing efficiency at NA 0.5: LPA 81.8%, Meep 77.6%. Inter-cell coupling grows with the phase gradient;
   - focusing efficiency at NA 0.25: LPA 88.3%, Meep 90.5%.

   Run time scales with aperture area × resolution⁴. Lenses larger than about 50 µm need a larger node, or the
   validation of a sector (`--crop-cells`). Estimate the cost with the user before queueing.

## 4b. Worked example

`examples/si1550_runs/` contains the finished runs of the silicon case study at 1550 nm:
- the robust metagrating;
- the 24-level h = 800 nm meta-atom library on SiO2 (`lut_h800.json`);
- the 8-, 12- and 24-level supercells and a refined supercell;
- the disk sweep.

`examples/README.md` lists commands that reproduce each stage in minutes on a CPU. Use them to show a new user how
the pipeline works, or as warm starts (`--init`) for new targets at 1550 nm. For other wavelengths or materials,
start again from the disk sweep (workflow B).

## 5. Reporting to the user

After each iteration, give:

- a short table: nominal, worst case ±d nm, per-polarization efficiency, dominant loss;
- your interpretation (physics, not just numbers);
- **one concrete next step**, with its estimated cost.

Keep a running log of decisions and results in a notes file of the user's choice, outside the repository or git-ignored.
