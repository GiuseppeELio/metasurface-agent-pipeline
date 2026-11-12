"""Device selection.

TORCWA needs complex torch.linalg.eig / inv. On Apple Silicon the MPS backend does not
implement complex general eigendecomposition, so on macOS the only working backend is CPU.
On NVIDIA GPUs (local or HPC) CUDA is used.

Optional: METASURF_EIG_DEVICE=cpu computes torch.linalg.eig on the CPU (LAPACK/MKL) while everything else stays
on the GPU. The CUDA eig of a general complex matrix is a hybrid CPU/GPU routine; on nodes where the CPU part is
slow (e.g. CINECA Leonardo, A100 + Xeon 8358, 8 cores per GPU) MKL alone is faster: a TORCWA forward+backward ran
1.4x faster. Off by default; see docs/BENCHMARK.md. It is applied when ptopt is imported (ptopt/__init__.py) and
again by pick_device(), so every tool that uses ptopt follows it.
"""
import os

import torch

_EIG_ORIG = torch.linalg.eig


def _eig_on_cpu(A, *args, **kw):
    if A.is_cuda:
        w, v = _EIG_ORIG(A.cpu(), *args, **kw)
        return torch.return_types.linalg_eig((w.to(A.device), v.to(A.device)))
    return _EIG_ORIG(A, *args, **kw)


def eig_on_cpu_enabled():
    return os.environ.get("METASURF_EIG_DEVICE", "").lower() == "cpu"


def apply_eig_policy():
    """Route torch.linalg.eig of CUDA tensors through the CPU when METASURF_EIG_DEVICE=cpu."""
    torch.linalg.eig = _eig_on_cpu if eig_on_cpu_enabled() else _EIG_ORIG


def pick_device(requested="auto"):
    if requested == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "mps":
        raise ValueError("MPS is not supported by TORCWA (no complex linalg.eig on MPS): use 'cpu' or 'cuda'.")
    else:
        dev = torch.device(requested)
    apply_eig_policy()
    return dev


def describe(device):
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        eig = " | eig on cpu" if torch.linalg.eig is _eig_on_cpu else ""
        return f"cuda:{device.index or 0} {p.name} {p.total_memory/2**30:.0f} GiB{eig}"
    return f"cpu ({torch.get_num_threads()} threads)"
