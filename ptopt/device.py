"""Device selection.

TORCWA needs complex torch.linalg.eig / inv. On Apple Silicon the MPS backend does not
implement complex general eigendecomposition, so on macOS the only working backend is CPU.
On NVIDIA GPUs (local or HPC) CUDA is used.
"""
import torch


def pick_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps":
        raise ValueError("MPS is not supported by TORCWA (no complex linalg.eig on MPS): use 'cpu' or 'cuda'.")
    return torch.device(requested)


def describe(device):
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"cuda:{device.index or 0} {p.name} {p.total_memory/2**30:.0f} GiB"
    return f"cpu ({torch.get_num_threads()} threads)"
