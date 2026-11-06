# Changelog

All notable changes to this project are documented here. The pipeline is improved continuously; each entry
says what changed and why, so that results obtained with earlier versions can be traced.


## [0.3.0] - 2026-11-06

- **Accompanying paper** on arXiv: [arXiv:2611.08127](https://arxiv.org/abs/2611.08127). The reference is in the README,
  `CITATION.cff` and the Zenodo metadata.
- **Benchmark, tier 4:** local Meep runs now set `OMP_NUM_THREADS=1` (unless already set) and record it. Conda-forge
  Meep is built with OpenMP, and without this every MPI process could open one thread per core and oversubscribe
  the CPU.
- **Reference results** from four machines in `docs/benchmark/` and `docs/BENCHMARK.md`: HPC V100S node, two
  workstations (RTX 3080, L4) and a Mac mini M4 (CPU only). Summary table and figure in the README.
- **Cross-machine benchmark**: `tools/benchmark.py` and `tools/bench_report.py`, documented in `docs/BENCHMARK.md`.
  - Five tiers: self-test, RCWA kernel vs Fourier order (CPU and GPU), topology-optimization iteration time,
    mini achromatic lens end to end, and Meep MPI scaling.
  - Each tier stores its timing and its physical result, so laptops, workstations and HPC nodes can be compared for
    speed and for agreement.
  - Tiers 0–3 run on native Windows (no WSL).
  - `AGENTS.md` onboarding step 6c: run it once per machine; the agent uses the numbers to estimate run times and
    to choose the device.
- **Large-aperture achromatic metalenses** (TiO2 on SiO2, 700–1000 nm, D = 400 µm), new tools:
  - `tools/tio2_lut.py`: multi-wavelength library of fourfold-symmetric pillars (circle, square, ring, square ring,
    cross) with fabrication rules. |t|² is a power fraction (TORCWA `power_norm=True`, checked against the
    silica/air Fresnel coefficient).
  - `tools/achro_tio2.py`: dispersion-engineered design on a fine radial grid, and an optional filter that drops atoms
    with sharp lattice resonances. Focusing uses a band-limited vector angular spectrum, so apertures of hundreds of µm
    take seconds. Efficiency and Strehl are reported also at a common focal plane, which matters at low Fresnel number.
  - `tools/achro_lens.py`: shared design core.
- **Full-diameter Meep check** of rotationally symmetric lenses: `validation/meep_achro_strip.py` and
  `validation/achro_strip_post.py`.
  - A 3-D strip through the axis with the real pillars, run in segments and in two polarizations. The circular
    aperture field is rebuilt from the radial and azimuthal responses.
  - Lorentz fits of tabulated n,k (max Δn 3e-4).
  - The substrate is non-dispersive inside the PML layers: a Lorentzian medium in the PML made long runs diverge.
- **Removed `paper/`**. The figure script of the accompanying paper needs the authors' own run set and is not
  needed to use the pipeline. The figures belong to the paper; the worked example stays in `examples/`.
- **Optical constants shipped as tables** (refractiveindex.info, CC0): `materials/TiO2_ALD_Siefke.txt` and
  `materials/SiO2_Malitson.txt`.

## [0.2.1] - 2026-11-05

- First archived release: Zenodo, doi:10.5281/zenodo.23982524 (all versions: doi:10.5281/zenodo.23982523).
- Logo in the README and Zenodo metadata (`.zenodo.json`); no code changes since 0.2.0.

## [0.2.0] - 2026-11-03 (continuous improvement after the initial release)

- **Narrow-band objectives**: optional `objective.wavelength_weights` (`ptopt/problem.py`).
  - Use it for quasi-BIC and other high-Q designs.
  - The FoM becomes a weighted sum over `wavelengths_nm` (normalized by the positive weights) instead of the mean.
  - Example: weights [−0.5, 1, −0.5] on [λ0−Δ, λ0, λ0+Δ].
  - AGENTS.md gains workflow A2 for these targets: asymmetry scan, independent Q check, absorption limit.
- **Optical constants from the refractiveindex.info database.**
  - `tools/fetch_materials.py` downloads the database (CC0) from GitHub once, or installs it from a local copy.
    It also searches pages and prints n, k with the references.
  - Configs accept `{"material": "rii:<shelf>/<book>/<page>"}` (`ptopt/rii.py`): PCHIP interpolation, k ≥ 0,
    all dispersion formulas 1–9, and an error instead of extrapolation outside the data range.
  - `hpc/setup_env.sh` and `hpc/setup_meep.sh` now install `pyyaml`.
- **Self-test** `tools/selftest.py [--full]`: reproduces reference numbers of the worked example on the CPU.
  - Checks: re-evaluation of the robust grating, 8-level supercell, NA 0.5 metalens; with `--full`, a short
    optimization from scratch.
  - References and tolerances are in `tools/selftest_reference.json`.
- **Fix: tabulated materials in Meep and in the metalens model.**
  - `validation/meep_validate.py`, `validation/meep_lens.py` and `metalens.py` read only constant `"n"` values.
    A `{"material": ...}` spec silently became n = 1.
  - They now use the same material code as the optimizer (real part at the design wavelength).
- **Fix, `ptopt/lensprop.py`:**
  - When k/dk was an integer, a point of the angular-spectrum grid fell exactly on k. Its
    E_z = −(k·A)/k_z then diverged and corrupted the focus metrics.
  - Waves with k_z < 0.01 k are now dropped.

## [0.1.0] - initial release

Agent-driven inverse-design pipeline for dielectric metasurfaces (commit 4a1c804).
