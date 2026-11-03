# Materials

Three ways to give a material in a config (wavelength in nm everywhere):

| Spec | Meaning |
|---|---|
| `{"n": 3.476}` or `{"n": 2.0, "k": 0.01}` | constant (complex) index |
| `{"material": "<name>"}` | table `materials/<name>.txt` with three whitespace-separated columns `wavelength_nm  n  k` |
| `{"material": "rii:<shelf>/<book>/<page>"}` | entry of the [refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database), e.g. `rii:main/Si/nk/Schinke` |

The database (CC0, about 30 MB) is installed once with `python tools/fetch_materials.py` into
`materials/refractiveindex.info-database/` (git-ignored); `--from <dir|zip>` uses a copy you already have.
`python tools/fetch_materials.py --search TiO2` lists pages; `--show main/TiO2/nk/Siefke --lam 532 1550` prints the
range, the references and n, k. Interpolation is PCHIP (shape preserving, k ≥ 0) and wavelengths outside the data
range raise an error instead of extrapolating. All dispersion formulas (1–9) of the database are supported.

The same specs work in the optimizer (TORCWA), in `metalens.py` and in the Meep validation scripts (which use the
real part of the index at the design wavelength). Always report the data sets used (page and REFERENCES line) and
cite the database: M. N. Polyanskiy, *Sci. Data* 11, 94 (2024).
