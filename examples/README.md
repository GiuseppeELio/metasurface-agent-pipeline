# Worked example: silicon metasurfaces at 1550 nm

`si1550_runs/` holds the finished runs of the case studies in the paper. They are the "training set" of the
pipeline: optimized binary designs with their full reports. Use them in three ways:

- to see what every stage produces;
- to reproduce the main numbers in minutes on a laptop CPU;
- as warm starts (`--init`) for new targets at 1550 nm with the same materials.

| Directory | What it is |
|---|---|
| `si1550_defl_v4_s2` | Robust freeform metagrating: Si on Si, air above, Λ = 2023 × 400 nm, h = 900 nm, +1 order at 50°, unpolarized, geometric robustness ±10 nm. Includes the full cross-solver check: Meep at 40/50/60 px/µm, RCWA at the [35,7], [50,10] and [70,12] truncations. |
| `si1550_defl_v2_h900` | Same grating optimized without robustness: higher nominal FoM, much lower worst case. |
| `diskA_sio2_P700` | Si disks on SiO2 (P = 700 nm): T and phase versus radius and height. |
| `atom*_h800_p<phase>_<shape>` | The 24 meta-atoms of the library (h = 800 nm, square cell P = 700 nm, d4 symmetry, phase steps of 15°). Each is the best of the disk, clover and cross starts for its phase. |
| `lut_h800.json` | The library: one atom per phase, selected by the worst case within ±10 nm (`select_atoms.py`). |
| `sc8_lut`, `sc12_lut`, `sc24_lut` | Supercells assembled from the library: deflection at 16.1°, 10.6° and 5.3°, verified by full-supercell RCWA. |
| `sc8_lut_refine_r1`, `sc12_lut_refine_r1` | Global robust refinements. For `sc12`: nominal 0.889 vs 0.903 for the library supercell, worst case within ±10 nm 0.848 vs 0.749. |
| `lens_na050`, `lens_na025` | Metalenses (D = 20.3 µm, NA 0.5 and 0.25) built from the library by `metalens.py`. Each holds the angular-spectrum model (propagation sections, DOF, FWHM), the 3-D Meep results at 20/25/30 px/µm (`meep_lens_res*.json`) and the Meep field propagated with the same code (`lens_post_res30.*`). |

`sc12_lut` also carries Meep results at 40 and 50 px/µm and its RCWA reference (`rcwa_ref.json`).

The large files are not included: no latent densities, no checkpoints, no optimization histories. Every directory
has the binary design (`rho_binary.npy` or `rho_supercell.npy`) and the reports.

## Setup

```bash
bash hpc/setup_env.sh && source .venv/bin/activate
python tools/seed_examples.py          # copies examples/si1550_runs/* into runs/ (never overwrites)
```

On a cluster, also run it there once: `runs/` is not pushed. Use a GPU list containing the line
`tools/seed_examples.py`.

## Ten-minute tour (CPU)

Approximate times are for a two-core laptop. The low Fourier orders keep each step short. The `summary.json`
files hold the converged numbers.

1. **Inspect a finished design.** Read `runs/si1550_defl_v4_s2/summary.json`:
   - `order_analysis_binary`: every propagating order, per polarization;
   - `convergence_check`;
   - `robustness_binary`: exact ±5/10/15 nm edge shifts;
   - `feature_size`.

   Compare it with `si1550_defl_v2_h900`: the nominal winner is the fragile one.
2. **Re-evaluate it** (about 5 s):
   ```bash
   python validation/rcwa_reference.py --run runs/si1550_defl_v4_s2 --order 15 3 --device cpu --out rcwa_ref_demo.json
   ```
3. **Assemble a supercell from the library** (about 15 s). Expected η(+1) ≈ 0.87; the library prediction is 0.95:
   ```bash
   python supercell.py --name sc8_demo --library lut_h800 --levels 8 --order 21 5 --shifts 0 --device cpu
   ```
4. **Design a metalens from the same library** (about 15 s): D = 20 µm, NA = 0.5, focusing efficiency ≈ 0.82 in
   the angular-spectrum model:
   ```bash
   python metalens.py --library lut_h800 --diameter-um 20 --na 0.5 --name lens_demo
   ```
   The run also writes `lens_propagation.png`:
   - longitudinal xz/yz sections from the lens surface through the focus;
   - the depth of focus (about 9 µm here);
   - the lateral FWHM.

   The shipped lenses `lens_na050` and `lens_na025` also hold the Meep results propagated with the same code
   (`lens_post_res30.json`, `.png`). The plane-field files are not shipped (13 MB each): regenerate them with
   `validation/meep_lens.py`, then run `validation/lens_post.py`.
5. **A short robust optimization from scratch** (about 3 min). This uses a coarse grid and low order. Expect a FoM
   of about 0.67; the full settings on a GPU reach 0.74:
   ```bash
   python run_topopt.py --config configs/metagrating_si1550_robust.json --iters 150 --order 10 3 --grid 200 40 \
       --set verify_order=[15,3] --device cpu --tag demo
   ```
6. **Read the cross-solver verdicts** (instant, no Meep needed: the results are shipped):
   ```bash
   python validation/validate.py report --runs runs/si1550_defl_v4_s2 runs/sc12_lut
   ```
   Expected verdict: `agree` for both designs.
   - With only the [35,7] truncation, the y polarization of the grating would be `marginal`.
   - The higher RCWA truncations close the gap.
   - This is the diagnosis the validation stage asks for.
7. **Run Meep yourself.** First `bash hpc/setup_meep.sh`. Then:
   - `python validation/validate.py plan --runs runs/<dir> --level quick --tag demo`;
   - submit the two lists through the bridge, or run them locally with `mpirun`;
   - run `report` again.

   A metalens smoke test takes a few minutes on a workstation:
   ```bash
   mpirun -np 8 python validation/meep_lens.py --run runs/lens_na050 --resolution 15 --crop-cells 9
   ```

## Starting a new design from these data

- **New deflection angle at 1550 nm:**
  - Choose N from sin θ = λ/(N·P) and run `supercell.py --library lut_h800 --levels N`.
  - Optionally refine with `configs/supercell_refine.json` and `--init runs/<sc>/rho_supercell.npy`.
- **New lens:**
  - `metalens.py` with your D and NA;
  - then `validation/meep_lens.py`, first as a smoke test with `--crop-cells 9`.
- **Another wavelength, material or thickness:**
  - The library does not transfer.
  - Start from `disk_sweep.py`, then `tools/make_lut_sweep.py` (workflow B in `AGENTS.md`). These examples show what
    each step should produce.
