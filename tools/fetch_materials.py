#!/usr/bin/env python3
"""Download, search and inspect the refractiveindex.info database (CC0) used by {"material": "rii:<shelf>/<book>/<page>"}.

  python tools/fetch_materials.py                       # download once (~30 MB) into materials/refractiveindex.info-database/
  python tools/fetch_materials.py --from <dir|zip>      # install from a copy you already have
  python tools/fetch_materials.py --where               # which copy is used
  python tools/fetch_materials.py --search TiO2         # list pages matching a text
  python tools/fetch_materials.py --show main/TiO2/nk/Siefke --lam 532 1550
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptopt import rii  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="source", help="local database folder or GitHub ZIP instead of downloading")
ap.add_argument("--force", action="store_true", help="download again even if a copy exists")
ap.add_argument("--where", action="store_true")
ap.add_argument("--search")
ap.add_argument("--show")
ap.add_argument("--lam", type=float, nargs="*", default=[])
a = ap.parse_args()
if a.where:
    print(rii.db_root(), "(installed)" if rii.available() else "(not installed)")
elif a.search:
    for r in rii.search(a.search):
        print(r)
elif a.show:
    m = rii.RIIMaterial(a.show)
    print(f"{m.page}: {m.range_nm[0]:.1f}-{m.range_nm[1]:.1f} nm\nREFERENCES: {m.references}\nCOMMENTS: {m.comments}")
    for lam in a.lam:
        v = m.nk(lam)
        print(f"  {lam:8.1f} nm  n = {v.real:.5f}  k = {v.imag:.3e}")
else:
    rii.fetch(a.source, a.force)
