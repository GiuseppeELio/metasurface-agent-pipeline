"""ptopt: TORCWA-based topology optimization of metasurfaces.

Importing the package applies the optional eig policy (METASURF_EIG_DEVICE=cpu, see ptopt/device.py). The import
is guarded so that environments without torch (the Meep environment) can still use the torch-free modules.
"""
import os as _os

if _os.environ.get("METASURF_EIG_DEVICE", "").lower() == "cpu":
    try:
        from .device import apply_eig_policy as _apply

        _apply()
    except ImportError:
        pass
