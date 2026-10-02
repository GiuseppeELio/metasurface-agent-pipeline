# Agent-Driven Inverse Design Pipeline for Dielectric Metasurfaces

`metasurface-agent-pipeline`: an agent-driven inverse-design pipeline for dielectric metasurfaces and metagratings.
The user describes the target in plain language. An AI coding agent turns it into a configuration, runs
GPU-accelerated, differentiable RCWA topology optimization on a local machine or on a SLURM cluster,
analyses the results, and proposes the next design iteration. Fabrication robustness and an independent
FDTD check are part of the loop.

Electromagnetic solver: [TORCWA](https://github.com/kch3782/torcwa) (PyTorch RCWA with automatic
differentiation; C. Kim and B. Lee, *Comput. Phys. Commun.* **282**, 108552, 2023).
Validation: [MIT Meep](https://meep.readthedocs.io) (FDTD).

> **Using an AI agent?** Point it to [`AGENTS.md`](AGENTS.md). It contains the onboarding procedure,
> the job protocol and the analysis checklist the agent is expected to follow.

## Requirements

- **An AI agent that can run commands on your machine.**
  - Use **Claude Code** (terminal or desktop) opened in this folder, or **Claude Cowork** with this folder connected.
  - Any coding agent that reads `AGENTS.md` and can use a shell should work.
  - A chat-only assistant (e.g. a claude.ai Project with the files uploaded) can read the code but cannot run
    anything, so it cannot drive the pipeline.
- **Python 3.10 or newer, and bash.**
- **Hardware:** an NVIDIA GPU (recommended) or a CPU, or access to a SLURM cluster.
- **Internet access** for the first installation: PyTorch and TORCWA from pip; Meep from conda-forge.

> **Windows users.** The bridge and the setup scripts are bash scripts, and Meep has no native Windows build.
> Use **WSL2** (Ubuntu): clone the repository inside the WSL file system and run the agent there.
> NVIDIA GPUs are available inside WSL2 with a recent Windows driver; check with `nvidia-smi` in the WSL shell.
> Running from PowerShell or `cmd` is not supported.

> **Meep (FDTD validation) is installed separately and is optional for optimization.**
> - `./hpc/bridge.sh setup-meep` downloads a private micromamba and installs `pymeep` (MPI build) from conda-forge
>   into `.meep-env` (about 2–3 GB).
> - It needs internet on the machine where it runs. On a cluster, run it on the login node: compute nodes are
>   usually offline.
> - Native on Linux and macOS (Intel and Apple silicon); on Windows, inside WSL2.
> - Without Meep the pipeline still optimizes and analyses designs, but the cross-solver validation stage cannot
>   run, and the agent will say so in its reports.
>
> The agent explains these steps at the first start (`AGENTS.md`, onboarding).

---

## What it does

| Stage | Tool | Output |
|---|---|---|
| Topology optimization of one periodic cell (any FoM in `ptopt/problem.py`) | `run_topopt.py` | `runs/<name>/summary.json`, binary design, plots |
| Parametric pre-screening (e.g. disk radius × height) | `disk_sweep.py` | T and phase maps |
| Meta-atom phase library (LUT) with several initial shapes | `run_topopt.py` + `tools/make_lut_sweep.py` | one run per phase/shape |
| Best atom per phase, worst case within ±d nm | `select_atoms.py` | `runs/<lib>.json` |
| Supercell assembly + rigorous RCWA of the whole supercell | `supercell.py` | `runs/<sc>/supercell.json` |
| Global refinement of an assembled supercell (warm start) | `run_topopt.py --init` | refined design |
| Post-hoc robustness at higher truncation | `evaluate.py` | `robustness_eval.json` |
| Metalens from the library + vector angular-spectrum focusing model | `metalens.py` | layout, focal spot, focusing efficiency |
| Cross-solver validation stage (TORCWA ↔ Meep): plan, run, verdict | `validation/validate.py` | `validation.json`, comparison figures |
| Independent FDTD runs (periodic designs / metalens) | `validation/meep_validate.py`, `validation/meep_lens.py` | `meep_res<R>.json`, `meep_lens_res<R>.json` |
| Figures of the accompanying paper (optional, for the authors' run set) | `paper/make_figures.py` | `paper/figures/` |

Every run reports:

- the FoM of the continuous design and of the binarized design;
- the efficiency and angle of **every propagating order** in transmission and reflection, for x and y polarization, with the energy balance;
- a **convergence check** at a higher Fourier truncation;
- the FoM under **exact edge shifts** (erosion and dilation, in nm);
- **minimum-feature statistics**.

Optimizer features:

- density filter and tanh projection with a β continuation;
- Adam;
- symmetry constraints (`y`, `x`, `xy`, `d4`);
- robust formulations: threshold-based, or geometric with an exact ±d nm edge shift;
- initial shapes (`disk`, `clover`, `cross`, `ring`, `square`, `ellipse`, `triangle`);
- multi-wavelength FoM averaging;
- warm start and resume;
- `--set key.sub=value` overrides for sweeps.

## Worked example

`examples/` contains the finished silicon case studies at 1550 nm: the robust metagrating, the 24-level
meta-atom library, the supercells and the disk sweep. [`examples/README.md`](examples/README.md) is a
ten-minute CPU tour of every stage on these data, and a starting point for new designs.

## Quick start (local)

```bash
git clone https://codeberg.org/<owner>/metasurface-agent-pipeline.git && cd metasurface-agent-pipeline
bash hpc/setup_env.sh                     # creates .venv with torch, torcwa, numpy, scipy, matplotlib
source .venv/bin/activate
python run_topopt.py --config configs/metaatom_template.json --iters 30 --order 5 5 --device auto
```

Hardware:

- **NVIDIA GPU (CUDA):** use `--device cuda`. If the driver is old, install a matching wheel, e.g.
  `TORCH_INDEX=https://download.pytorch.org/whl/cu118 TORCH_SPEC=torch==2.7.1 bash hpc/setup_env.sh`.
- **CPU:** works everywhere and is fine for small problems.
- **Apple-silicon GPUs (MPS):** not supported. PyTorch has no complex `linalg.eig` on MPS; use the CPU.

## Agent mode (local or HPC)

```bash
cp hpc/site.env.example hpc/site.env        # normally the agent fills this in with you (AGENTS.md)
./hpc/bridge.sh setup                        # environment on the execution target
./hpc/bridge.sh watch                        # keep this terminal open
```

The agent writes one-line requests into `queue/` (`submit`, `array`, `meep` or `check`). The watcher validates them, then:

1. in slurm mode, syncs the code and submits the job;
2. follows the job;
3. pulls the results;
4. moves the request to `queue/done/`.

The watcher only executes those four sub-commands and rejects shell metacharacters. Credentials never pass through the agent: login and 2FA stay with the user (`SSH_LOGIN_CMD`).

## Repository layout

```
ptopt/            problem definition, materials, objectives, morphology tools
configs/          example configurations (JSON)
tools/            sweep generators
validation/       Meep FDTD validation
hpc/              bridge.sh, environment setup, generic SLURM job templates, site.env.example
paper/            figures of the accompanying paper (optional; not needed to use the pipeline)
examples/         worked example: finished runs of the silicon case study at 1550 nm
AGENTS.md         instructions for AI agents working with this repository
```

## Citation

If you use this pipeline, please cite TORCWA (Kim & Lee 2023) and the accompanying paper (reference to be added).

## License

MIT License, see [`LICENSE`](LICENSE).

Third-party software is installed separately and keeps its own license:
- TORCWA: LGPL, installed with `pip install torcwa`;
- MIT Meep: GPL, installed from conda-forge by `hpc/setup_meep.sh`;
- PyTorch: BSD-style.
