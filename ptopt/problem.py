"""Physical problem: single patterned layer on a substrate, FoM from TORCWA S-parameters.

Extend OBJECTIVES with new figures of merit (register a function with the same signature).
"""
import math

import numpy as np
import torch
import torcwa

from .materials import permittivity

DTYPES = {"complex64": (torch.complex64, torch.float32), "complex128": (torch.complex128, torch.float64)}


class Problem:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.sim_dtype, self.geo_dtype = DTYPES[cfg.get("precision", "complex64")]
        self.L = [float(v) for v in cfg["period_nm"]]
        self.nx, self.ny = cfg["grid"]
        self.order = cfg["order"]
        lams = cfg.get("wavelengths_nm", [cfg.get("wavelength_nm")])
        self.wavelengths = [float(l) for l in lams]
        self.inc = math.radians(cfg.get("inc_ang_deg", 0.0))
        self.azi = math.radians(cfg.get("azi_ang_deg", 0.0))
        lay = cfg["layer"]
        self.thickness = float(lay["thickness_nm"])
        # precompute permittivities per wavelength
        # TORCWA 'forward' = from input layer to output layer. illumination decides which medium is the input.
        sub, sup = cfg.get("substrate", {"n": 1.0}), cfg.get("superstrate", {"n": 1.0})
        self.illumination = cfg.get("illumination", "from_substrate")
        if self.illumination == "from_superstrate":
            sub, sup = sup, sub
        self.eps_in = [permittivity(sub, l) for l in self.wavelengths]
        self.eps_out = [permittivity(sup, l) for l in self.wavelengths]
        self.eps_core = [permittivity(lay["core"], l) for l in self.wavelengths]
        self.eps_bg = [permittivity(lay.get("background", {"n": 1.0}), l) for l in self.wavelengths]
        obj = cfg["objective"]
        if obj["type"] not in OBJECTIVES:
            raise ValueError(f"unknown objective: {obj['type']} (available: {list(OBJECTIVES)})")
        self.objective = OBJECTIVES[obj["type"]]
        self.ref_t = None
        if obj["type"] in NEEDS_REFERENCE:  # t00 of the empty cell (same layer thickness, all background)
            with torch.no_grad():
                z = torch.zeros((self.nx, self.ny), dtype=self.geo_dtype, device=device)
                self.ref_t = []
                for i in range(len(self.wavelengths)):
                    sim = self.build_sim(z, i)
                    self.ref_t.append(complex(sim.S_parameters(orders=[0, 0], direction="forward", port="transmission",
                                                               polarization="xx", ref_order=[0, 0]).cpu()))

    def ocfg(self, i):
        o = dict(self.cfg["objective"])
        if self.ref_t is not None:
            o["_ref_phase"] = math.atan2(self.ref_t[i].imag, self.ref_t[i].real)
        return o

    def build_sim(self, rho, i, order=None):
        lam = self.wavelengths[i]
        sim = torcwa.rcwa(freq=1.0 / lam, order=order or self.order, L=self.L, dtype=self.sim_dtype, device=self.device)
        sim.add_input_layer(eps=self.eps_in[i])
        sim.add_output_layer(eps=self.eps_out[i])
        sim.set_incident_angle(inc_ang=self.inc, azi_ang=self.azi)
        eps = rho * self.eps_core[i] + (1.0 - rho) * self.eps_bg[i]
        sim.add_layer(thickness=self.thickness, eps=eps)
        sim.solve_global_smatrix()
        return sim

    def fom(self, rho, return_details=False):
        """FoM over wavelengths (differentiable): the mean, or a weighted sum if objective.wavelength_weights is set.

        wavelength_weights (one per entry of wavelengths_nm) builds narrow-band objectives, e.g. for quasi-BIC or
        other high-Q resonances: wavelengths_nm [1545, 1550, 1555] with weights [-0.5, 1, -0.5] rewards the response
        at the centre and penalizes it at the neighbours (the resonance must be narrower than the spacing).
        The weighted sum is divided by the sum of the positive weights, so 1 still means a perfect FoM.
        """
        vals, details = [], []
        for i in range(len(self.wavelengths)):
            sim = self.build_sim(rho, i)
            f, d = self.objective(sim, self.ocfg(i))
            vals.append(f)
            details.append({"wavelength_nm": self.wavelengths[i], **d})
        wts = self.cfg["objective"].get("wavelength_weights")
        if wts:
            assert len(wts) == len(vals), "objective.wavelength_weights must have one weight per wavelengths_nm entry"
            assert any(w > 0 for w in wts), "objective.wavelength_weights needs at least one positive weight"
            fom = sum(w * v for w, v in zip(wts, vals)) / sum(w for w in wts if w > 0)
        else:
            fom = torch.stack(vals).mean()
        return (fom, details) if return_details else fom


    def propagating_orders(self, i, medium="out"):
        """Diffraction orders propagating in the input/output medium, with angles (deg)."""
        lam = self.wavelengths[i]
        n = complex(self.eps_out[i] if medium == "out" else self.eps_in[i]) ** 0.5
        n_in = complex(self.eps_in[i]) ** 0.5
        kx0 = n_in.real * math.sin(self.inc) * math.cos(self.azi)
        ky0 = n_in.real * math.sin(self.inc) * math.sin(self.azi)
        res = []
        for mx in range(-self.order[0], self.order[0] + 1):
            for my in range(-self.order[1], self.order[1] + 1):
                kx = kx0 + mx * lam / self.L[0]
                ky = ky0 + my * lam / self.L[1]
                kt = math.hypot(kx, ky)
                if kt < n.real:
                    res.append(((mx, my), math.degrees(math.asin(kt / n.real))))
        return res

    @torch.no_grad()
    def order_analysis(self, rho, order=None):
        """Per-order T/R efficiencies of every propagating order, per input polarization, + energy balance."""
        out = []
        for i, lam in enumerate(self.wavelengths):
            sim = self.build_sim(rho, i, order=order)
            entry = {"wavelength_nm": lam, "fourier_order": list(order or self.order), "pols": {}}
            T_ord, R_ord = self.propagating_orders(i, "out"), self.propagating_orders(i, "in")
            for p_in in ("x", "y"):
                d = {"T": {}, "R": {}}
                for port, ords in (("transmission", T_ord), ("reflection", R_ord)):
                    key = "T" if port == "transmission" else "R"
                    for (o, ang) in ords:
                        e = sum(float(_eff(sim, list(o), port, po + p_in).cpu()) for po in ("x", "y"))
                        if e > 1e-4:
                            d[key][f"{o[0]},{o[1]}"] = {"eff": round(e, 4), "angle_deg": round(ang, 2)}
                d["T_total"] = round(sum(v["eff"] for v in d["T"].values()), 4)
                d["R_total"] = round(sum(v["eff"] for v in d["R"].values()), 4)
                d["absorbed_or_error"] = round(1 - d["T_total"] - d["R_total"], 4)
                entry["pols"][p_in] = d
            out.append(entry)
        return out


def _eff(sim, order, port, pol, direction="forward"):
    s = sim.S_parameters(orders=order, direction=direction, port=port, polarization=pol, ref_order=[0, 0])
    return (torch.abs(s) ** 2).sum()


def diffraction_efficiency(sim, ocfg):
    """Efficiency into target orders, averaged over input polarizations (value in [0,1]).

    ocfg: orders [[1,0],...], port transmission|reflection, input_pols ["x","y"] (default both).
    Example 6 (Kim & Lee 2023) == orders [[1,0]], port transmission, input_pols [x,y]  (their FoM = 2*this).
    """
    port = ocfg.get("port", "transmission")
    pols = ocfg.get("input_pols", ["x", "y"])
    per = {}
    for p_in in pols:
        tot = 0.0
        for o in ocfg["orders"]:
            for p_out in ("x", "y"):
                tot = tot + _eff(sim, o, port, p_out + p_in)  # TORCWA: 'yx' = out y, in x
        per[p_in] = tot
    fom = torch.stack(list(per.values())).mean()
    return fom, {f"eff_in_{k}": float(v.detach()) for k, v in per.items()}


def diffraction_selectivity(sim, ocfg):
    """Target efficiency minus penalty * efficiency into unwanted orders ("suppress")."""
    fom, d = diffraction_efficiency(sim, ocfg)
    sup = ocfg.get("suppress", [])
    if sup:
        bad, _ = diffraction_efficiency(sim, {**ocfg, "orders": sup})
        fom = fom - ocfg.get("penalty", 1.0) * bad
        d["eff_suppressed"] = float(bad.detach())
    return fom, d


def zero_order_phase(sim, ocfg):
    """Meta-atom objective: maximize Re(t00 * exp(-i*phi_target)) averaged over input pols (x->x, y->y).

    = |t| cos(phase error): rewards unit transmission AND the target phase at once.
    phi_target = target_phase_deg relative to the empty cell (set automatically as _ref_phase).
    """
    phi = math.radians(ocfg.get("target_phase_deg", 0.0)) + ocfg.get("_ref_phase", 0.0)
    rot = complex(math.cos(-phi), math.sin(-phi))
    vals, d = [], {}
    for p in ocfg.get("input_pols", ["x", "y"]):
        t = sim.S_parameters(orders=[0, 0], direction="forward", port="transmission", polarization=p + p, ref_order=[0, 0])
        t = t.reshape(())
        vals.append(torch.real(t * rot))
        tc = complex(t.detach().cpu())
        dph = math.degrees(math.atan2((tc * rot).imag, (tc * rot).real))
        d[f"T_{p}"] = abs(tc) ** 2
        d[f"phase_err_deg_{p}"] = dph
        d[f"phase_rel_deg_{p}"] = (math.degrees(math.atan2(tc.imag, tc.real) - ocfg.get("_ref_phase", 0.0)) + 360) % 360
    return torch.stack(vals).mean(), d


NEEDS_REFERENCE = {"zero_order_phase"}

OBJECTIVES = {
    "zero_order_phase": zero_order_phase,
    "diffraction_efficiency": diffraction_efficiency,
    "diffraction_selectivity": diffraction_selectivity,
}
