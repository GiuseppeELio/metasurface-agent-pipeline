"""Material models: constant index or tabulated n,k (wavelength in nm)."""
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

MATERIALS_DIR = Path(__file__).resolve().parent.parent / "materials"


class TabulatedMaterial:
    def __init__(self, path):
        data = np.loadtxt(path)
        self.lam, self.n, self.k = data[:, 0], data[:, 1], data[:, 2]
        self._n = interp1d(self.lam, self.n, kind="cubic")
        self._k = interp1d(self.lam, self.k, kind="cubic")

    def nk(self, lam_nm):
        lam = float(np.clip(lam_nm, self.lam[0], self.lam[-1]))
        return complex(self._n(lam) + 1j * self._k(lam))


def refractive_index(spec, lam_nm):
    """spec: {"n": 1.46} | {"n": 3.5, "k": 0.01} | {"material": "aSiH"} (file materials/aSiH.txt)."""
    if "material" in spec:
        path = Path(spec["material"])
        if not path.suffix:
            path = MATERIALS_DIR / f"{spec['material']}.txt"
        return TabulatedMaterial(path).nk(lam_nm)
    return complex(spec.get("n", 1.0), spec.get("k", 0.0))


def permittivity(spec, lam_nm):
    return refractive_index(spec, lam_nm) ** 2
