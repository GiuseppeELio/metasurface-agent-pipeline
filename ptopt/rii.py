"""refractiveindex.info database: download, search and evaluate (wavelength in nm).

The database (https://github.com/polyanskiy/refractiveindex.info-database, CC0 public domain) is used from:
  1. $RII_DB (a folder that contains database/data), if set;
  2. materials/refractiveindex.info-database/ in this repository (created by `python tools/fetch_materials.py`);
  3. refractiveindex.info-database-main/ in the repository root (a manually unpacked GitHub ZIP).

  python tools/fetch_materials.py                    # download the GitHub archive (~30 MB), keep database/ only
  python tools/fetch_materials.py --from <dir|zip>   # use a copy you already have
  python tools/fetch_materials.py --search TiO2      # list matching pages (shelf/book/page)
  python tools/fetch_materials.py --show main/TiO2/nk/Siefke --lam 532 1550

In configs:  {"material": "rii:main/Si/nk/Schinke"}
Interpolation: PCHIP (shape preserving, never overshoots, k >= 0); no extrapolation outside the data range.
"""
import io
import os
import warnings
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator



REPO_URL = "https://github.com/polyanskiy/refractiveindex.info-database"
ZIP_URLS = [REPO_URL + "/archive/refs/heads/main.zip", REPO_URL + "/archive/refs/heads/master.zip"]
NOTICE = ("[materials] Downloading the refractiveindex.info database (M. N. Polyanskiy, CC0 public domain)\n"
          f"            from {REPO_URL}\n            into {{dest}} (about 30 MB, once).")


REPO = Path(__file__).resolve().parent.parent


def db_root():
    """Folder that contains database/data (the first existing candidate; else the download target)."""
    cands = ([Path(os.environ["RII_DB"]).expanduser()] if os.environ.get("RII_DB") else []) + [
        REPO / "materials" / "refractiveindex.info-database", REPO / "refractiveindex.info-database-main"]
    for c in cands:
        if (c / "database" / "data").is_dir():
            return c
    return REPO / "materials" / "refractiveindex.info-database"


def data_root():
    return db_root() / "database" / "data"


def available():
    return data_root().is_dir()


def _install_from_zip(zf, dest):
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    names = [n for n in zf.namelist() if "/database/" in n]
    if not names:
        raise SystemExit("[materials] archive does not contain a database/ folder")
    prefix = names[0].split("/database/")[0] + "/"
    for n in names:
        rel = n[len(prefix):]
        if not rel or n.endswith("/") or ".." in Path(rel).parts:
            continue
        out = tmp / rel; out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(n) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)


def fetch(source=None, force=False):
    dest = REPO / "materials" / "refractiveindex.info-database"
    if available() and not force:
        print(f"[materials] database already present: {dest}")
        return dest
    if source:
        src = Path(source).expanduser()
        if src.is_dir():
            base = src if (src / "database").is_dir() else src.parent
            if not (base / "database" / "data").is_dir():
                raise SystemExit(f"[materials] no database/data folder under {src}")
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(base / "database", dest / "database")
        else:
            with zipfile.ZipFile(src) as zf:
                _install_from_zip(zf, dest)
        print(f"[materials] installed from {src} into {dest}")
        return dest
    print(NOTICE.format(dest=dest), flush=True)
    last = None
    for url in ZIP_URLS:
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                _install_from_zip(zf, dest)
            print(f"[materials] done: {dest}")
            return dest
        except Exception as e:  # try the next branch name
            last = e
    raise SystemExit(f"[materials] download failed ({last}). Download the ZIP from {REPO_URL} yourself and run "
                     "`python tools/fetch_materials.py --from <file.zip>`.")


def _require():
    if not available():
        raise SystemExit("[materials] refractiveindex.info database not installed: run `python tools/fetch_materials.py`.")


def search(text):
    _require()
    t = text.lower()
    out = []
    for p in sorted(data_root().rglob("*.yml")):
        rel = p.relative_to(data_root()).with_suffix("").as_posix()
        if p.name == "about.yml":
            continue
        if t in rel.lower():
            out.append(rel)
    return out


def _yaml():
    try:
        import yaml
    except ImportError:  # pragma: no cover
        raise SystemExit("PyYAML is required: pip install pyyaml")
    return yaml


def _formula(ftype, c, lam_um):
    L = np.asarray(lam_um, float); L2 = L ** 2; c = list(c) + [0.0] * 20
    if ftype == 1:      # Sellmeier
        s = 1 + c[0] + sum(c[i] * L2 / (L2 - c[i + 1] ** 2) for i in range(1, 17, 2))
        return np.sqrt(s)
    if ftype == 2:      # Sellmeier-2
        s = 1 + c[0] + sum(c[i] * L2 / (L2 - c[i + 1]) for i in range(1, 17, 2))
        return np.sqrt(s)
    if ftype == 3:      # polynomial
        return np.sqrt(c[0] + sum(c[i] * L ** c[i + 1] for i in range(1, 17, 2)))
    if ftype == 4:      # RefractiveIndex.INFO
        s = c[0] + c[1] * L ** c[2] / (L2 - c[3] ** c[4]) + c[5] * L ** c[6] / (L2 - c[7] ** c[8])
        s = s + sum(c[i] * L ** c[i + 1] for i in range(9, 17, 2))
        return np.sqrt(s)
    if ftype == 5:      # Cauchy
        return c[0] + sum(c[i] * L ** c[i + 1] for i in range(1, 11, 2))
    if ftype == 6:      # gases
        return 1 + c[0] + sum(c[i] / (c[i + 1] - L ** -2) for i in range(1, 11, 2))
    if ftype == 7:      # Herzberger
        return c[0] + c[1] / (L2 - 0.028) + c[2] / (L2 - 0.028) ** 2 + c[3] * L2 + c[4] * L2 ** 2 + c[5] * L2 ** 3
    if ftype == 8:      # Retro
        A = c[0] + c[1] * L2 / (L2 - c[2]) + c[3] * L2
        return np.sqrt((1 + 2 * A) / (1 - A))
    if ftype == 9:      # Exotic
        return np.sqrt(c[0] + c[1] / (L2 - c[2]) + c[3] * (L - c[4]) / ((L - c[4]) ** 2 + c[5]))
    raise ValueError(f"formula {ftype} not supported")


class RIIMaterial:
    """One page of the database, e.g. RIIMaterial("main/Si/nk/Schinke"); nk(lam_nm) -> n + i k."""

    def __init__(self, page):
        _require()
        path = data_root() / (page if page.endswith(".yml") else page + ".yml")
        if not path.exists():
            raise SystemExit(f"[materials] no page {page} in {data_root()} (try `python tools/fetch_materials.py --search <text>`)")
        d = _yaml().safe_load(path.read_text(encoding="utf-8"))
        self.page, self.references, self.comments = page, str(d.get("REFERENCES", "")).strip(), str(d.get("COMMENTS", "")).strip()
        self._n = self._k = None; lo, hi = [], []
        for blk in d["DATA"]:
            t = blk["type"]
            if t.startswith("tabulated"):
                a = np.array([[float(x) for x in ln.split()] for ln in str(blk["data"]).strip().splitlines() if ln.strip()])
                lam = a[:, 0] * 1000.0
                lo.append(lam.min()); hi.append(lam.max())
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.interpolate")  # PCHIP on flat k = 0 segments
                if t == "tabulated nk":
                    self._n = PchipInterpolator(lam, a[:, 1]); self._k = PchipInterpolator(lam, a[:, 2])
                elif t == "tabulated n":
                    self._n = PchipInterpolator(lam, a[:, 1])
                elif t == "tabulated k":
                    self._k = PchipInterpolator(lam, a[:, 1])
            elif t.startswith("formula"):
                ft = int(t.split()[1]); coef = [float(x) for x in str(blk["coefficients"]).split()]
                r = [float(x) * 1000.0 for x in str(blk["wavelength_range"]).split()]
                lo.append(r[0]); hi.append(r[1])
                self._n = (lambda lam, ft=ft, coef=coef: _formula(ft, coef, np.asarray(lam) / 1000.0))
            else:
                raise ValueError(f"[materials] unsupported data type '{t}' in {page}")
        if self._n is None:
            raise ValueError(f"[materials] {page}: no refractive index data")
        self.range_nm = (max(lo), min(hi))

    def nk(self, lam_nm):
        lam = np.asarray(lam_nm, float)
        if lam.min() < self.range_nm[0] - 1e-6 or lam.max() > self.range_nm[1] + 1e-6:
            raise ValueError(f"[materials] {self.page}: {lam.min():.1f}-{lam.max():.1f} nm outside the data range "
                             f"{self.range_nm[0]:.1f}-{self.range_nm[1]:.1f} nm")
        n = np.real(self._n(lam)); k = np.maximum(self._k(lam), 0.0) if self._k is not None else 0.0 * lam
        v = n + 1j * k
        return complex(v) if np.ndim(lam_nm) == 0 else v


