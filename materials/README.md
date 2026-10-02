# Tabulated materials
Optional n,k tables, one file per material: `<name>.txt` with three whitespace-separated columns
`wavelength_nm  n  k`. Reference them in a config as `{"material": "<name>"}`.
Constant indices are given directly as `{"n": 3.476}` or `{"n": 2.0, "k": 0.01}`.
