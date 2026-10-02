"""Quick GPU sanity check + benchmark of the TORCWA bottleneck (complex eig of 2N x 2N)."""
import time

import torch

print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| cuda available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available: GPU driver too old for this torch build? Re-run setup with TORCH_INDEX=https://download.pytorch.org/whl/cu118 TORCH_SPEC=torch==2.7.1")
d = torch.device("cuda")
print("GPU", torch.cuda.get_device_name(0))
for dtype in (torch.complex64, torch.complex128):
    n = 1054  # order [15,8] -> 2*(31*17)
    A = torch.randn(n, n, dtype=dtype, device=d)
    torch.linalg.eig(A); torch.cuda.synchronize()
    t = time.time()
    for _ in range(3):
        torch.linalg.eig(A)
    torch.cuda.synchronize()
    print(f"eig {n}x{n} {dtype}: {(time.time()-t)/3*1000:.0f} ms")
