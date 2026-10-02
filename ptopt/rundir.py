"""Load a finished run directory as (config, binary design), without importing torch.

Two kinds of run directories exist:
  * optimization runs (run_topopt.py): config.json + rho_binary.npy
  * library supercells (supercell.py):  supercell.json + rho_supercell.npy; the configuration is rebuilt from the
    first atom's config.json (period and grid multiplied by the number of atoms, +1-order objective).
"""
import json
from pathlib import Path

import numpy as np


def load_run(run, verify_per_atom=7):
    run = Path(run)
    if (run / "config.json").exists():
        cfg = json.loads((run / "config.json").read_text())
        rho = np.load(run / "rho_binary.npy").astype(np.float64)
        return cfg, rho, "topopt"
    sc = json.loads((run / "supercell.json").read_text())
    cfg = json.loads((run.parent / sc["atoms"][0]["dir"] / "config.json").read_text())
    n = len(sc["atoms"])
    rho = np.load(run / "rho_supercell.npy").astype(np.float64)
    cfg["name"] = run.name
    cfg["period_nm"] = [cfg["period_nm"][0] * n, cfg["period_nm"][1]]
    cfg["grid"] = list(rho.shape)
    cfg["order"] = [verify_per_atom * n, verify_per_atom]
    cfg["verify_order"] = [verify_per_atom * n, verify_per_atom]
    order_sign = sc.get("order_sign", 1)
    cfg["objective"] = {"type": "diffraction_efficiency", "port": "transmission", "orders": [[order_sign, 0]],
                        "input_pols": ["x", "y"]}
    return cfg, rho, "supercell"


def target_orders(cfg):
    """Orders the FoM is about, as 'm,n' keys (None for objectives that are not order efficiencies)."""
    obj = cfg.get("objective", {})
    if obj.get("type") in ("diffraction_efficiency", "diffraction_selectivity") and obj.get("port", "transmission") == "transmission":
        return [f"{o[0]},{o[1]}" for o in obj.get("orders", [])]
    return None


def shift_tag(d):
    return f"s{d:+.0f}" if d else ""
