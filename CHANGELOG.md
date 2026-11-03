# Changelog

All notable changes to this project are documented here. The pipeline is improved continuously; each entry
says what changed and why, so that results obtained with earlier versions can be traced.


## [0.2.0] - unreleased (continuous improvement after the initial release)

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
