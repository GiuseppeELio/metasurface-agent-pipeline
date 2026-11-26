#!/usr/bin/env python3
"""Generate a COMSOL 6.4 model (Java source) of the 3D unit cell of a finished run, for the optional
COMSOL validation (finite elements, Wave Optics). One .java per run and mesh level; `run_comsol.sh` compiles
it into a .class that opens in the COMSOL GUI or runs in batch.

  python validation/comsol/comsol_export.py --run runs/<dir> [--shift-nm d] [--pols x y] [--ram-gb 16]
         [--levels 2] [--hpat-nm 60 45] [--wavelength-nm 1550] [--sigma-nm auto] [--compile]

Writes into runs/<dir>/comsol/:
  MS_<case>_L<k>.java   one class per mesh level (k = 1 coarse, 2 finer)
  <case>_plan.json      geometry check, stack, orders, mesh plan, DOF/RAM estimates

Geometry
  - The patterned layer is the binary design of the run (rho_binary.npy, or rho_supercell.npy), extruded over
    layer.thickness_nm, between the substrate (below, z < 0) and the superstrate (above).
  - Designs that do not vary along y (1D gratings tiled in y) become exact rectangles (pixel edges, as RCWA sees them).
  - Freeform designs become smoothed 0.5 iso-contours (Gaussian, default 1 px) (needs `pip install shapely scikit-image`). The pixel
    agreement with the binary design is checked (default threshold 99.9 %).
  - The cell is translated (a pure translation: efficiencies are unchanged) so that the x faces cut only background
    where an empty column exists, and so that faces that do cut the pattern cross its edges transversally.
    Vertices on the periodic faces are snapped pairwise.

Physics (names verified against a COMSOL 6.4.0.378 export): interface `ewfd`, Floquet periodic conditions,
periodic ports with diffraction orders (input port excited with 1 W), refractive-index materials.

Results (written by the class, no GUI needed): per order transmitted efficiency from a Fourier projection of the
field on an observation plane (independent of the port post-processing), reflected orders other than (0,0)
the same way on the input side, T_flux on the observation plane, ewfd.Ttotal / ewfd.Rtotal from the ports,
DOF and solve time. Compare with RCWA: validation/comsol/comsol_compare.py.

Exit codes: 0 ok, 2 geometry check failed, 3 nothing fits in the RAM budget.
"""
import argparse
import cmath
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

# DOF model of second-order vector elements in COMSOL 6.4: DOF ~ C_DOF * sum(V_i / h_i^3), calibrated on
# si1550_defl_v4_s2 (1.79 M DOF at L1, 2.39 M at L2; the estimate is conservative for finer levels, where the mesh is
# set by the geometry more than by h_pat). The class logs the real DOF.
C_DOF = 320.0
# Peak RAM (GB) = base + per-M-DOF slope. Iterative (GMRES), measured on a Mac mini (16 GB): 8.4 GB at 1.79 M DOF,
# 9.6 GB at 2.39 M DOF; 3.5 M DOF fitted in 16 GB. Direct (MUMPS): 3.3 M DOF did not fit in 16 GB.
RAM_MODEL = {"iterative": (4.8, 2.0), "direct": (2.0, 12.0)}
DIRECT_MAX_MDOF = 0.0   # the direct solver is never the default (RAM); it is only a fallback below 0.5 M DOF
DIRECT_FALLBACK_MAX_DOF = 500000


# ----------------------------------------------------------------------------------------------- run and stack
def load_run(run):
    cfg = json.loads((run / "config.json").read_text())
    for name in ("rho_binary.npy", "rho_supercell.npy"):
        if (run / name).exists():
            return cfg, np.load(run / name).astype(float), name
    sys.exit(f"[comsol_export] no rho_binary.npy / rho_supercell.npy in {run}")


def nk(spec, lam):
    if "material" in spec:
        from ptopt.materials import refractive_index
        z = complex(refractive_index(spec, lam))
    else:
        z = complex(spec.get("n", 1.0), spec.get("k", 0.0))
    return z.real, abs(z.imag)


def propagating(n_med, lam, L, kx0, ky0, nmax=60):
    """(m, n) orders propagating in a medium of index n_med (kx0, ky0 in units of k0)."""
    out = []
    for m in range(-nmax, nmax + 1):
        for n in range(-nmax, nmax + 1):
            kx, ky = kx0 + m * lam / L[0], ky0 + n * lam / L[1]
            if math.hypot(kx, ky) < n_med - 1e-9:
                out.append((m, n))
    return out


def decay_length(n_med, lam, L, kx0, ky0, nmax=60):
    """Decay length (same unit as lam) of the slowest evanescent order in the medium."""
    kmin = None
    for m in range(-nmax, nmax + 1):
        for n in range(-nmax, nmax + 1):
            kt = math.hypot(kx0 + m * lam / L[0], ky0 + n * lam / L[1])
            if kt > n_med + 1e-9:
                kap = math.sqrt(kt ** 2 - n_med ** 2)
                kmin = kap if kmin is None else min(kmin, kap)
    return lam / (2 * math.pi * kmin) if kmin else 0.0


# ------------------------------------------------------------------------------------------------- geometry
def empty_column_shift(b):
    """Index of the middle of the widest run of empty x-columns (periodic), or None."""
    empty = ~b.any(axis=1)
    if not empty.any():
        return None
    if empty.all():
        return 0
    nx = len(empty)
    start = int(np.argmin(empty))            # a solid column: runs do not wrap from here
    best, cur, cur0 = (0, None), 0, None
    for k in range(1, nx + 1):
        i = (start + k) % nx
        if empty[i]:
            if cur == 0:
                cur0 = k
            cur += 1
            if cur > best[0]:
                best = (cur, cur0)
        else:
            cur = 0
    width, k0 = best
    return (start + k0 + width // 2) % nx


def best_cut(f, axis):
    """Pixel boundary (0..n-1) where a periodic cut crosses the 0.5 level transversally: max over cuts of
    min |f - 0.5| on the cut line. Avoids slivers that touch one periodic face but not the other."""
    g = np.moveaxis(f, axis, 0)
    line = 0.5 * (g + np.roll(g, 1, axis=0))          # value on the boundary before index j
    score = np.abs(line - 0.5).min(axis=1)
    return int(np.argmax(score)), float(score.max())


def rings_1d(b, dx, Ly):
    """y-invariant design -> rectangles (x0, x1) of solid along x, inside [0, Lx]. Exact pixel edges."""
    col = b[:, 0]
    nx = len(col)
    rects, i = [], 0
    while i < nx:
        if col[i]:
            j = i
            while j < nx and col[j]:
                j += 1
            rects.append((i * dx, j * dx))
            i = j
        else:
            i += 1
    rings = []
    for x0, x1 in rects:
        rings.append(("ext", [(x0, 0.0), (x1, 0.0), (x1, Ly), (x0, Ly)]))
    return [[r] for r in rings]          # polygons: [exterior] (no holes)


def rings_2d(b, dx, dy, Lx, Ly, sigma_px, simplify_nm):
    try:
        import shapely
        from scipy.ndimage import gaussian_filter
        from shapely.geometry import Polygon, box
        from skimage import measure
    except ImportError:
        sys.exit("[comsol_export] freeform designs need: pip install shapely scikit-image")
    nx, ny = b.shape
    f = gaussian_filter(b.astype(float), sigma_px, mode="wrap") if sigma_px > 0 else b.astype(float)
    P = 10
    pad = np.pad(np.pad(f, P, mode="wrap"), 1, constant_values=0.0)
    region = Polygon()
    for c in measure.find_contours(pad, 0.5):
        if len(c) < 4:
            continue
        xy = np.c_[(c[:, 0] - P - 1 + 0.5) * dx, (c[:, 1] - P - 1 + 0.5) * dy]
        region = region.symmetric_difference(Polygon(xy).buffer(0))
    region = region.simplify(simplify_nm, preserve_topology=True)
    clip = region.intersection(box(0, 0, Lx, Ly))
    polys = [g for g in getattr(clip, "geoms", [clip]) if g.geom_type == "Polygon" and g.area > 1.0]
    xc, yc = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy, indexing="ij")
    inside = shapely.contains_xy(clip, xc.ravel(), yc.ravel()).reshape(nx, ny) if polys else np.zeros_like(b)
    agree = float((inside == b).mean())
    out = []
    for g in polys:
        out.append([("ext", list(g.exterior.coords)[:-1])] + [("hole", list(h.coords)[:-1]) for h in g.interiors])
    return out, agree


def snap_faces(polys, Lx, Ly, tol=1e-4):
    """Match vertices on the periodic faces pairwise (x=0 <-> x=Lx, y=0 <-> y=Ly). Coordinates in nm."""
    def collect(axis, val):
        return sorted({round(p[1 - axis], 6) for poly in polys for _, r in poly for p in r if abs(p[axis] - val) < tol})
    msgs = []
    for axis, L in ((0, Lx), (1, Ly)):
        a, bb = collect(axis, 0.0), collect(axis, L)
        if len(a) != len(bb):
            msgs.append(f"{'xy'[axis]} faces: {len(a)} vs {len(bb)} boundary vertices")
            continue
        if a:
            mism = max(abs(u - v) for u, v in zip(a, bb))
            if mism > 0.5:
                msgs.append(f"{'xy'[axis]} faces: vertex mismatch {mism:.2f} nm")
        mp = dict(zip(bb, a))
        for poly in polys:
            for _, r in poly:
                for k, p in enumerate(r):
                    q = list(p)
                    if abs(q[axis]) < tol:
                        q[axis] = 0.0
                    elif abs(q[axis] - L) < tol:
                        q[axis] = L
                        q[1 - axis] = mp.get(round(q[1 - axis], 6), q[1 - axis])
                    r[k] = (q[0], q[1])
    return msgs


# --------------------------------------------------------------------------------------------------- planning
def plan_mesh(lam, L, h_lay, n, pads, hpat_list, ram_gb, n_per_lam=8):
    """Element sizes and DOF/RAM estimates per mesh level. Outside the patterned layer: lambda/(n_per_lam*n), and
    lambda/(1.5*n_per_lam*n) on the observation planes (the Fourier analysis reads the field there)."""
    A = L[0] * L[1] * 1e-6                                    # um^2
    h_sub = lam / (n_per_lam * n["sub"])
    h_sup = lam / (n_per_lam * n["sup"])
    levels = []
    for hp in hpat_list:
        hp = min(hp, lam / (5 * max(n["core"], n["bg"])))
        s = (A * h_lay * 1e-3) / (hp * 1e-3) ** 3
        s += (A * pads["sub_total"] * 1e-3) / (h_sub * 1e-3) ** 3
        s += (A * pads["sup_total"] * 1e-3) / (h_sup * 1e-3) ** 3
        dof = C_DOF * s
        solver = "direct" if dof / 1e6 < DIRECT_MAX_MDOF else "iterative"
        base, slope = RAM_MODEL[solver]
        ram = base + slope * dof / 1e6
        levels.append({"h_pat_nm": round(hp, 1), "h_sub_nm": round(h_sub, 1), "h_sup_nm": round(h_sup, 1),
                       "h_obs_sub_nm": round(h_sub / 1.5, 1), "h_obs_sup_nm": round(h_sup / 1.5, 1), "n_per_lambda": n_per_lam,
                       "hmin_nm": round(min(20.0, hp / 3), 1), "dof_est": int(dof), "solver": solver,
                       "ram_est_gb": round(ram, 1), "fits": ram <= 0.85 * ram_gb})
    return levels


# ---------------------------------------------------------------------------------------------------- Java
def jstr(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def tag(m):
    return "0" if m == 0 else (f"p{m}" if m > 0 else f"m{-m}")


def write_java(path, cls, case, run, P, polys, plan_lvl, fallback_orders, out_orders, in_orders, pols, dump_props):
    C = 'model.component("comp1")'
    G = C + '.geom("geom1")'
    WP = G + '.feature("wp1").geom()'
    PH = C + '.physics("ewfd")'
    M = C + '.mesh("mesh1")'
    J = []
    w = J.append
    w(f"""/*
 * {cls}.java  --  COMSOL 6.4 (Wave Optics, ewfd), generated by validation/comsol/comsol_export.py
 * Run: {run}   case: {case}   mesh level: h_pat = {plan_lvl['h_pat_nm']} nm, solver {plan_lvl['solver']}
 * 3D unit cell {P['Lx']:.3f} x {P['Ly']:.3f} nm, patterned layer {P['h_lay']:.1f} nm, lambda {P['lam']:.2f} nm,
 * illumination {P['illum']}, cell translated by {P['shift_nm']:.2f} nm in x and {P['shift_y_nm']:.2f} nm in y
 * (pure translation of the periodic pattern: efficiencies are unchanged).
 *
 * Build:  bash validation/comsol/run_comsol.sh --build-only {run}     (compile only; open the .class in COMSOL)
 * Solve:  bash validation/comsol/run_comsol.sh {run}                  (compile + batch; results in this folder)
 *
 * What the class does depends on the file {cls}.mode next to it (written by run_comsol.sh):
 *   build (default, also when the file is missing): geometry, selections, materials, physics, mesh and study set up;
 *         nothing is meshed or solved, so opening the .class in the GUI is quick;
 *   mesh: also builds the mesh and logs the element count;
 *   solve: meshes, solves every polarization and writes {cls}_results.json.
 * Log: {cls}_log.txt (WARN lines = settings that could not be applied; fix them in the GUI).
 * DIR below is rewritten by run_comsol.sh to the folder of this file. COMSOL needs
 * Preferences > Security > File system access: All files.
 */
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;
import java.util.*;

public class {cls} {{

  static String DIR = {jstr(str(path.parent.resolve()))};
  static final String CASE = {jstr(cls)};
  static final String PFX = "ewfd";
  static PrintWriter log;
  static List<String> warns = new ArrayList<String>();

  static void L(String s) {{
    System.out.println("[" + CASE + "] " + s);
    if (log != null) log.println(s);
  }}

  static void W(String what, Throwable e) {{
    String m = "WARN " + what + ": " + (e == null ? "" : e.getMessage());
    warns.add(m);
    L(m);
  }}

  static String mode() {{
    try {{
      BufferedReader r = new BufferedReader(new FileReader(DIR + "/" + CASE + ".mode"));
      String s = r.readLine();
      r.close();
      return s == null ? "build" : s.trim();
    }} catch (Exception e) {{
      return "build";
    }}
  }}

  // methods called by reflection are not needed for the model; if missing in this COMSOL version, only a log line is lost
  static Object call(Object o, String m) throws Exception {{
    return o.getClass().getMethod(m).invoke(o);
  }}

  static double[][] cat(double[][]... parts) {{
    int n = 0;
    for (double[][] p : parts) n += p.length;
    double[][] o = new double[n][];
    int k = 0;
    for (double[][] p : parts) for (double[] v : p) o[k++] = v;
    return o;
  }}
""")
    # ring data, one static method per chunk (64 KB bytecode limit per method)
    ring_tags = []          # per polygon: list of (ring index, kind)
    ri = 0
    for pi, poly in enumerate(polys):
        tags = []
        for kind, pts in poly:
            chunks = [pts[i:i + 1000] for i in range(0, len(pts), 1000)]
            names = []
            for ci, ch in enumerate(chunks):
                rows = ",".join("{%.6f,%.6f}" % (x / 1000.0, y / 1000.0) for x, y in ch)
                w(f"  static double[][] r{ri}_{ci}() {{ return new double[][]{{{rows}}}; }}")
                names.append(f"r{ri}_{ci}()")
            w(f"  static double[][] ring{ri}() {{ return cat({', '.join(names)}); }}  // polygon {pi} {kind}")
            tags.append((ri, kind))
            ri += 1
        ring_tags.append(tags)

    # order lists for the results file
    def jarr(items):
        return "{" + ", ".join(items) + "}"
    w("")
    w(f"  static final String[] T_KEYS = {jarr(jstr(f'{m},{n}') for m, n in out_orders)};")
    w(f"  static final String[] T_VARS = {jarr(jstr(f'etaT_{tag(m)}_{tag(n)}') for m, n in out_orders)};")
    in_nz = [o for o in in_orders if o != (0, 0)]
    w(f"  static final String[] R_KEYS = {jarr(jstr(f'{m},{n}') for m, n in in_nz)};")
    w(f"  static final String[] R_VARS = {jarr(jstr(f'etaR_{tag(m)}_{tag(n)}') for m, n in in_nz)};")
    w(f"  static final String[] POLS = {jarr(jstr(p) for p in pols)};")
    w(f'  static final String SOLVER = "{plan_lvl["solver"]}";')
    w(f'  static final long DOF_EST = {plan_lvl["dof_est"]}L;   // planner estimate')
    w(f'  static final long DIRECT_FALLBACK_MAX_DOF = {DIRECT_FALLBACK_MAX_DOF}L;   // direct solver only as a fallback, below this')
    w(f"  static final int[][] FALLBACK_IN = {jarr('{%d,%d}' % o for o in fallback_orders['in'])};")
    w(f"  static final int[][] FALLBACK_OUT = {jarr('{%d,%d}' % o for o in fallback_orders['out'])};")

    w(f"""
  public static Model run() {{
    try {{
      return runImpl();
    }} catch (Exception e) {{
      L("ERROR " + e);
      throw new RuntimeException(e);
    }} finally {{
      if (log != null) log.flush();
    }}
  }}

  static Model runImpl() throws Exception {{
    String mode = mode();
    log = new PrintWriter(new FileWriter(DIR + "/" + CASE + "_log.txt", true), true);
    L("=== " + new Date() + "  mode = " + mode);
    try {{ L("COMSOL " + ModelUtil.class.getMethod("getComsolVersion").invoke(null)); }} catch (Throwable e) {{ }}

    Model model = ModelUtil.create("Model");
    model.modelPath(DIR);
    model.label(CASE + ".mph");
""")
    params = [
        ("lda0", f"{P['lam']:.6g}[nm]", "Wavelength"),
        ("f0", "c_const/lda0", "Frequency"),
        ("k0", "2*pi/lda0", "Free-space wavenumber"),
        ("Lx", f"{P['Lx']:.6g}[nm]", "Period x"),
        ("Ly", f"{P['Ly']:.6g}[nm]", "Period y"),
        ("A_cell", "Lx*Ly", "Cell area"),
        ("h_lay", f"{P['h_lay']:.6g}[nm]", "Patterned layer thickness"),
        ("d_obs_sub", f"{P['d_obs_sub']:.6g}[nm]", "Substrate: structure -> observation plane"),
        ("d_port_sub", f"{P['d_port_sub']:.6g}[nm]", "Substrate: observation plane -> port"),
        ("d_obs_sup", f"{P['d_obs_sup']:.6g}[nm]", "Superstrate: structure -> observation plane"),
        ("d_port_sup", f"{P['d_port_sup']:.6g}[nm]", "Superstrate: observation plane -> port"),
        ("z_obs_sub", "-d_obs_sub", "Observation plane in the substrate"),
        ("z_bot", "-d_obs_sub-d_port_sub", "Bottom port"),
        ("z_obs_sup", "h_lay+d_obs_sup", "Observation plane in the superstrate"),
        ("z_top", "z_obs_sup+d_port_sup", "Top port"),
        ("n_sub", f"{P['n']['sub']:.6g}", "Substrate index"), ("k_sub", f"{P['k']['sub']:.6g}", ""),
        ("n_sup", f"{P['n']['sup']:.6g}", "Superstrate index"), ("k_sup", f"{P['k']['sup']:.6g}", ""),
        ("n_core", f"{P['n']['core']:.6g}", "Pattern (solid) index"), ("k_core", f"{P['k']['core']:.6g}", ""),
        ("n_bg", f"{P['n']['bg']:.6g}", "Pattern (background) index"), ("k_bg", f"{P['k']['bg']:.6g}", ""),
        ("n_in", "n_sub" if P["illum"] == "from_substrate" else "n_sup", "Index of the incidence medium"),
        ("n_out", "n_sup" if P["illum"] == "from_substrate" else "n_sub", "Index of the transmission medium"),
        ("theta_in", f"{P['theta']:.6g}[deg]", "Incidence angle (in the incidence medium)"),
        ("phi_in", f"{P['phi']:.6g}[deg]", "Incidence azimuth"),
        ("kx_in", "k0*n_in*sin(theta_in)*cos(phi_in)", "Incident kx"),
        ("ky_in", "k0*n_in*sin(theta_in)*sin(phi_in)", "Incident ky"),
        ("pol", "0", "0 = input pol x, 1 = input pol y (pipeline convention)"),
        ("P_in", "1[W]", "Port input power"),
        ("Eax", P["Ea"][0], "E amplitude x"), ("Eay", P["Ea"][1], "E amplitude y"), ("Eaz", P["Ea"][2], "E amplitude z"),
    ]
    for k, v, d in params:
        w(f'    model.param().set("{k}", {jstr(v)}, {jstr(d)});')

    w(f"""
    // ------------------------------------------------------------------ geometry (um)
    model.component().create("comp1", true);
    {C}.geom().create("geom1", 3);
    {C}.mesh().create("mesh1");
    {G}.lengthUnit("\\u00b5m");
    {G}.geomRep("cadps");
    {G}.designBooleans(false);""")
    has_solid = len(polys) > 0
    if has_solid:
        w(f'    {G}.create("wp1", "WorkPlane");')
        w(f'    {G}.feature("wp1").set("unite", true);')
        finals = []
        for pi, tags in enumerate(ring_tags):
            for ri_, kind in tags:
                w(f'    {WP}.create("pol{ri_}", "Polygon");')
                w(f'    {WP}.feature("pol{ri_}").set("source", "table");')
                w(f'    {WP}.feature("pol{ri_}").set("table", ring{ri_}());')
            ext = [r for r, k in tags if k == "ext"][0]
            holes = [r for r, k in tags if k == "hole"]
            if holes:
                w(f'    {WP}.create("dif{pi}", "Difference");')
                w(f'    {WP}.feature("dif{pi}").selection("input").set("pol{ext}");')
                w(f'    {WP}.feature("dif{pi}").selection("input2").set({", ".join(jstr(f"pol{h}") for h in holes)});')
                finals.append(f"dif{pi}")
            else:
                finals.append(f"pol{ext}")
        if len(finals) > 1:
            w(f'    {WP}.create("uni1", "Union");')
            w(f'    {WP}.feature("uni1").selection("input").set({", ".join(jstr(t) for t in finals)});')
            w(f'    {WP}.feature("uni1").set("intbnd", false);')
        w(f'    {G}.create("ext1", "Extrude");')
        w(f'    {G}.feature("ext1").set("distance", new String[]{{"h_lay"}});')
        w(f'    {G}.feature("ext1").set("selresult", true);')
        w(f'    {G}.feature("ext1").selection("input").set("wp1");')
    for tg, z0, h in (("blk_sub2", "z_bot", "d_port_sub"), ("blk_sub1", "z_obs_sub", "d_obs_sub"),
                      ("blk_lay", "0", "h_lay"), ("blk_sup1", "h_lay", "d_obs_sup"), ("blk_sup2", "z_obs_sup", "d_port_sup")):
        w(f'    {G}.create("{tg}", "Block");')
        w(f'    {G}.feature("{tg}").set("size", new String[]{{"Lx", "Ly", "{h}"}});')
        w(f'    {G}.feature("{tg}").set("pos", new String[]{{"0", "0", "{z0}"}});')
    w(f'    {G}.run();')
    w(f'    {G}.run("fin");')

    E = "1[nm]"
    w("\n    // ------------------------------------------------------------------ selections (box: no entity numbers)")
    boxes = [
        ("s_sub", 3, "-1[m]", "1[m]", "-1[m]", "1[m]", f"z_bot-{E}", E),
        ("s_lay", 3, "-1[m]", "1[m]", "-1[m]", "1[m]", f"-{E}", f"h_lay+{E}"),
        ("s_sup", 3, "-1[m]", "1[m]", "-1[m]", "1[m]", f"h_lay-{E}", f"z_top+{E}"),
        ("b_x0", 2, f"-{E}", E, "-1[m]", "1[m]", "-1[m]", "1[m]"),
        ("b_xL", 2, f"Lx-{E}", f"Lx+{E}", "-1[m]", "1[m]", "-1[m]", "1[m]"),
        ("b_y0", 2, "-1[m]", "1[m]", f"-{E}", E, "-1[m]", "1[m]"),
        ("b_yL", 2, "-1[m]", "1[m]", f"Ly-{E}", f"Ly+{E}", "-1[m]", "1[m]"),
        ("b_bot", 2, "-1[m]", "1[m]", "-1[m]", "1[m]", f"z_bot-{E}", f"z_bot+{E}"),
        ("b_top", 2, "-1[m]", "1[m]", "-1[m]", "1[m]", f"z_top-{E}", f"z_top+{E}"),
        ("b_obs_sub", 2, "-1[m]", "1[m]", "-1[m]", "1[m]", f"z_obs_sub-{E}", f"z_obs_sub+{E}"),
        ("b_obs_sup", 2, "-1[m]", "1[m]", "-1[m]", "1[m]", f"z_obs_sup-{E}", f"z_obs_sup+{E}"),
        ("e_00", 1, f"-{E}", E, f"-{E}", E, "-1[m]", "1[m]"),
        ("e_0L", 1, f"-{E}", E, f"Ly-{E}", f"Ly+{E}", "-1[m]", "1[m]"),
    ]
    for t, dim, x0, x1, y0, y1, z0, z1 in boxes:
        S = f'{C}.selection("{t}")'
        w(f'    {C}.selection().create("{t}", "Box");')
        w(f'    {S}.set("entitydim", {dim});')
        for k, v in (("xmin", x0), ("xmax", x1), ("ymin", y0), ("ymax", y1), ("zmin", z0), ("zmax", z1)):
            w(f'    {S}.set("{k}", "{v}");')
        w(f'    {S}.set("condition", "inside");')
    for t, a, b_ in (("b_x", "b_x0", "b_xL"), ("b_y", "b_y0", "b_yL")):
        w(f'    {C}.selection().create("{t}", "Union");')
        w(f'    {C}.selection("{t}").set("entitydim", 2);')
        w(f'    {C}.selection("{t}").set("input", new String[]{{"{a}", "{b_}"}});')

    w("\n    // ------------------------------------------------------------------ materials (later ones override)")
    mats = [("mat_sub", "Substrate", "n_sub", "k_sub", "s_sub"), ("mat_sup", "Superstrate", "n_sup", "k_sup", "s_sup"),
            ("mat_bg", "Pattern background", "n_bg", "k_bg", "s_lay")]
    if has_solid:
        mats.append(("mat_core", "Pattern solid", "n_core", "k_core", "geom1_ext1_dom"))
    for t, lab, nn, kk, sel in mats:
        Mt = f'{C}.material("{t}")'
        w(f'    {C}.material().create("{t}", "Common");')
        w(f'    {Mt}.label("{lab} (n = {nn})");')
        w(f'    {Mt}.selection().named("{sel}");')
        w(f'    {Mt}.propertyGroup().create("RefractiveIndex", "RefractiveIndex", "Refractive index");')
        w(f'    {Mt}.propertyGroup("RefractiveIndex").set("n", new String[]{{"{nn}", "0", "0", "0", "{nn}", "0", "0", "0", "{nn}"}});')
        w(f'    {Mt}.propertyGroup("RefractiveIndex").set("ki", new String[]{{"{kk}", "0", "0", "0", "{kk}", "0", "0", "0", "{kk}"}});')

    w("\n    // ------------------------------------------------------------------ integration on the observation planes")
    for t, sel in (("intop_sub", "b_obs_sub"), ("intop_sup", "b_obs_sup")):
        w(f'    {C}.cpl().create("{t}", "Integration");')
        w(f'    try {{ {C}.cpl("{t}").selection().geom("geom1", 2); }} catch (Exception e) {{ W("{t} selection dim", e); }}')
        w(f'    {C}.cpl("{t}").selection().named("{sel}");')

    w("\n    // ------------------------------------------------------------------ Fourier analysis of the scattered orders")
    out_side, in_side = ("sup", "sub") if P["illum"] == "from_substrate" else ("sub", "sup")
    sgn = "1" if out_side == "sup" else "-1"
    V = f'{C}.variable("var1")'
    w(f'    {C}.variable().create("var1");')
    w(f'    {V}.set("T_flux", "{sgn}*intop_{out_side}(" + PFX + ".Poavz)/P_in", "Transmitted power / input (observation plane)");')
    for lbl, side, orders in (("T", out_side, out_orders), ("R", in_side, in_nz)):
        nmed = f"n_{side}"
        for m, n in orders:
            s = f"{lbl}_{tag(m)}_{tag(n)}"
            ph = f"exp(i*(kx{s}*x+ky{s}*y))"
            w(f'    {V}.set("kx{s}", "kx_in+2*pi*({m})/Lx");')
            w(f'    {V}.set("ky{s}", "ky_in+2*pi*({n})/Ly");')
            w(f'    {V}.set("kz{s}", "sqrt((k0*{nmed})^2-kx{s}^2-ky{s}^2)");')
            for comp in "xyz":
                w(f'    {V}.set("E{comp}{s}", "intop_{side}(" + PFX + ".E{comp}*{ph})/A_cell");')
            w(f'    {V}.set("eta{s}", "A_cell*(abs(Ex{s})^2+abs(Ey{s})^2+abs(Ez{s})^2)*real(kz{s})/(2*Z0_const*k0*P_in)", '
              f'"{"Transmitted" if lbl == "T" else "Reflected"} efficiency, order ({m},{n})");')

    port_in, port_out = ("b_bot", "b_top") if P["illum"] == "from_substrate" else ("b_top", "b_bot")
    w(f"""
    // ------------------------------------------------------------------ physics
    {C}.physics().create("ewfd", "ElectromagneticWavesFrequencyDomain", "geom1");
    try {{ {PH}.feature("wee1").set("DisplacementFieldModel", "RefractiveIndex"); }} catch (Exception e) {{ W("wee1 DisplacementFieldModel", e); }}
    for (String[] pc : new String[][]{{{{"pc_x", "b_x"}}, {{"pc_y", "b_y"}}}}) {{
      {PH}.create(pc[0], "PeriodicCondition", 2);
      {PH}.feature(pc[0]).selection().named(pc[1]);
      try {{ {PH}.feature(pc[0]).set("PeriodicType", "Floquet"); }} catch (Exception e) {{ W(pc[0] + " PeriodicType", e); }}
      try {{ {PH}.feature(pc[0]).set("kFloquet", new String[][]{{{{"kx_in"}}, {{"ky_in"}}, {{"0"}}}}); }} catch (Exception e) {{ W(pc[0] + " kFloquet", e); }}
    }}
    // port1 = incidence side (excited, default for the first port), port2 = transmission side
    {PH}.create("port1", "Port", 2);
    {PH}.feature("port1").selection().named("{port_in}");
    {PH}.create("port2", "Port", 2);
    {PH}.feature("port2").selection().named("{port_out}");
    for (String p : new String[]{{"port1", "port2"}}) {{
      try {{ {PH}.feature(p).set("PortType", "Periodic"); }} catch (Exception e) {{ W(p + " PortType", e); }}
    }}
    try {{ {PH}.feature("port1").set("PortExcitation", "on"); }} catch (Exception e) {{ W("port1 PortExcitation", e); }}
    try {{ {PH}.feature("port2").set("PortExcitation", "off"); }} catch (Exception e) {{ W("port2 PortExcitation", e); }}
    try {{ {PH}.feature("port1").set("Pin", "P_in"); }} catch (Exception e) {{ W("port1 Pin", e); }}
    try {{ {PH}.feature("port1").set("Eampl", new String[][]{{{{"Eax"}}, {{"Eay"}}, {{"Eaz"}}}}); }} catch (Exception e) {{ W("port1 Eampl", e); }}""")
    if abs(P["theta"]) > 1e-9:
        w(f"""    // oblique incidence: angle names as reported by ptpipe for 3D (alpha1_inc elevation, alpha2_inc azimuth) [CHECK]
    try {{ {PH}.feature("port1").set("alpha1_inc", "theta_in"); }} catch (Exception e) {{ W("port1 alpha1_inc", e); }}
    try {{ {PH}.feature("port1").set("alpha2_inc", "phi_in"); }} catch (Exception e) {{ W("port1 alpha2_inc", e); }}
    W("oblique incidence", new Exception("check the port angles and E amplitude in the GUI before trusting the results"));""")
    w(f"""    addOrders(model, "port1", FALLBACK_IN);
    addOrders(model, "port2", FALLBACK_OUT);""")
    if dump_props:
        w('    for (String t : new String[]{"port1", "port2", "pc_x"}) dumpProps(model, t);')

    lv = plan_lvl
    w(f"""
    // ------------------------------------------------------------------ mesh
    {M}.feature("size").set("custom", "on");
    {M}.feature("size").set("hmax", "{min(lv['h_sub_nm'], lv['h_sup_nm']):.1f}[nm]");
    {M}.feature("size").set("hmin", "{lv['hmin_nm']:.1f}[nm]");
    try {{ {M}.feature("size").set("hgrad", "1.5"); }} catch (Exception e) {{ W("mesh hgrad", e); }}""")
    for t, sel, h in (("size_sub", "s_sub", lv["h_sub_nm"]), ("size_sup", "s_sup", lv["h_sup_nm"]), ("size_lay", "s_lay", lv["h_pat_nm"])):
        S = f'{M}.feature("{t}")'
        w(f'    {M}.create("{t}", "Size");')
        w(f'    {S}.selection().geom("geom1", 3);')
        w(f'    {S}.selection().named("{sel}");')
        w(f'    {S}.set("custom", "on");')
        w(f'    {S}.set("hmaxactive", true);')
        w(f'    {S}.set("hmax", "{h:.1f}[nm]");')
    for t, sel, h in (("size_obs_sub", "b_obs_sub", lv["h_obs_sub_nm"]), ("size_obs_sup", "b_obs_sup", lv["h_obs_sup_nm"])):
        S = f'{M}.feature("{t}")'
        w(f'    {M}.create("{t}", "Size");')
        w(f'    {S}.selection().geom("geom1", 2);')
        w(f'    {S}.selection().named("{sel}");')
        w(f'    {S}.set("custom", "on");')
        w(f'    {S}.set("hmaxactive", true);')
        w(f'    {S}.set("hmax", "{h:.1f}[nm]");')
    w(f"""    {M}.create("edg1", "Edge");
    {M}.feature("edg1").selection().named("e_00");
    {M}.create("cpe1", "CopyEdge");
    {M}.feature("cpe1").selection("source").named("e_00");
    {M}.feature("cpe1").selection("destination").named("e_0L");
    {M}.create("ftri1", "FreeTri");
    {M}.feature("ftri1").selection().named("b_x0");
    {M}.create("cpf1", "CopyFace");
    {M}.feature("cpf1").selection("source").named("b_x0");
    {M}.feature("cpf1").selection("destination").named("b_xL");
    {M}.create("ftri2", "FreeTri");
    {M}.feature("ftri2").selection().named("b_y0");
    {M}.create("cpf2", "CopyFace");
    {M}.feature("cpf2").selection("source").named("b_y0");
    {M}.feature("cpf2").selection("destination").named("b_yL");
    {M}.create("ftet1", "FreeTet");

    // ------------------------------------------------------------------ study
    model.study().create("std1");
    model.study("std1").create("freq", "Frequency");
    model.study("std1").feature("freq").set("plist", "f0");

    if (warns.size() > 0) L(warns.size() + " WARN line(s) during setup: check them before solving");
    if (mode.equals("mesh") || mode.equals("solve")) {{
      long t0 = System.currentTimeMillis();
      {M}.run();
      try {{ L("mesh: " + call({M}, "getNumElem") + " elements, " + (System.currentTimeMillis() - t0) / 1000 + " s"); }}
      catch (Throwable e) {{ L("mesh built in " + (System.currentTimeMillis() - t0) / 1000 + " s"); }}
    }}
    // solver sequence in every mode, so that the model opened in the GUI already uses the iterative solver
    String used = "default";
    try {{
      model.sol().create("sol1");
      model.sol("sol1").study("std1");
      model.sol("sol1").createAutoSequence("std1");
      used = chooseSolver(model, SOLVER);
    }} catch (Exception e) {{ W("solver sequence (in the GUI: Study > Show Default Solver, then enable the iterative solver)", e); }}
    if (mode.equals("solve")) solveAll(model, used);
    L("done (mode " + mode + ")");
    return model;
  }}

  // diffraction orders: the GUI command first; if it is not available, explicit DiffractionOrder nodes for every
  // order that propagates in the port medium (both in-plane axis assignments, both polarizations), as in the export
  static void addOrders(Model model, String p, int[][] fallback) {{
    String[] cmds = {{"addDiffractionOrders", "adddiffractionorders", "AddDiffractionOrders"}};
    for (String c : cmds) {{
      try {{
        {PH}.feature(p).runCommand(c);
        int n = {PH}.feature(p).feature().tags().length;
        if (n > 0) {{ L(p + ": diffraction orders added by runCommand(" + c + "), " + n + " subnodes"); return; }}
      }} catch (Exception e) {{ }}
    }}
    try {{
      {PH}.feature(p).create("oport1", "OrthogonalPolarization", 2);
      int k = 0;
      for (int[] o : fallback) {{
        for (int pol = 0; pol < 2; pol++) {{
          k++;
          String t = "dport" + k;
          {PH}.feature(p).create(t, "DiffractionOrder", 2);
          {PH}.feature(p).feature(t).set("mOrder", o[0]);
          {PH}.feature(p).feature(t).set("nOrder", o[1]);
          if (pol == 1) {PH}.feature(p).feature(t).set("components", "outofplane");
        }}
      }}
      L(p + ": " + k + " DiffractionOrder nodes + OrthogonalPolarization created explicitly");
    }} catch (Exception e) {{ W(p + " diffraction orders (press 'Add Diffraction Orders' in the GUI)", e); }}
  }}

  static void dumpProps(Model model, String t) {{
    try {{
      String[] ps = (String[]) call({PH}.feature(t), "properties");
      StringBuilder sb = new StringBuilder("PROPS " + t + ":");
      for (String q : ps) {{
        String v;
        try {{ v = {PH}.feature(t).getString(q); }}
        catch (Exception e1) {{
          try {{ v = Arrays.toString({PH}.feature(t).getStringArray(q)); }}
          catch (Exception e2) {{
            try {{ v = Arrays.deepToString({PH}.feature(t).getStringMatrix(q)); }} catch (Exception e3) {{ v = "?"; }}
          }}
        }}
        sb.append(" ").append(q).append("=").append(v);
      }}
      L(sb.toString());
      L("SUBNODES " + t + ": " + Arrays.toString({PH}.feature(t).feature().tags()));
    }} catch (Exception e) {{ L("NOTE property dump of " + t + " not available: " + e); }}
  }}

  static String chooseSolver(Model model, String want) {{
    String used = "default";
    try {{
      String[] tags = model.sol("sol1").feature("s1").feature().tags();
      StringBuilder sb = new StringBuilder("solver s1 children:");
      String it = null, di = null;
      for (String t : tags) {{
        String ty = "";
        try {{ ty = String.valueOf(call(model.sol("sol1").feature("s1").feature(t), "getType")); }} catch (Throwable e) {{ }}
        sb.append(" ").append(t).append("(").append(ty).append(")");
        if (it == null && (ty.equals("Iterative") || (ty.isEmpty() && t.startsWith("i")))) it = t;
        if (di == null && (ty.equals("Direct") || (ty.isEmpty() && t.startsWith("d")))) di = t;
      }}
      L(sb.toString());
      String on = want.equals("direct") ? di : it, off = want.equals("direct") ? it : di;
      if (on != null) {{
        model.sol("sol1").feature("s1").feature(on).active(true);
        if (off != null) model.sol("sol1").feature("s1").feature(off).active(false);
        if (want.equals("iterative")) {{
          try {{ model.sol("sol1").feature("s1").feature(on).set("linsolver", "gmres"); }} catch (Exception e) {{ W("iterative solver: linsolver = gmres", e); }}
          used = "iterative GMRES (" + on + ")";
        }} else used = want + " (" + on + ")";
      }} else W("no " + want + " solver node in the auto sequence; using the default", null);
    }} catch (Exception e) {{ W("solver selection", e); }}
    L("linear solver: " + used);
    return used;
  }}

  static double[] evalGlobal(Model model, String[] exprs) {{
    double[] v = new double[exprs.length];
    if (exprs.length == 0) return v;
    try {{ model.result().numerical().remove("gev1"); }} catch (Exception e) {{ }}
    model.result().numerical().create("gev1", "EvalGlobal");
    model.result().numerical("gev1").set("data", "dset1");
    model.result().numerical("gev1").set("expr", exprs);
    double[][] r = model.result().numerical("gev1").getReal();
    if (r.length == exprs.length) for (int i = 0; i < exprs.length; i++) v[i] = r[i][r[i].length - 1];
    else for (int i = 0; i < exprs.length; i++) v[i] = r[r.length - 1][i];
    return v;
  }}

  static String num(double x) {{ return Double.isNaN(x) ? "null" : String.format(Locale.ROOT, "%.6f", x); }}

  static void solveAll(Model model, String used) throws Exception {{
    StringBuilder js = new StringBuilder();
    js.append("{{\\n  \\"case\\": \\"" + CASE + "\\",\\n  \\"linear_solver\\": \\"" + used + "\\",\\n  \\"pols\\": {{");
    long dof = -1;
    for (int ip = 0; ip < POLS.length; ip++) {{
      model.param().set("pol", "" + ip);
      long t0 = System.currentTimeMillis();
      try {{ model.sol("sol1").runAll(); }}
      catch (Exception e) {{
        if (used.startsWith("iterative") && DOF_EST < DIRECT_FALLBACK_MAX_DOF) {{
          W("iterative solver failed, retrying with the direct solver", e);
          used = chooseSolver(model, "direct");
          model.sol("sol1").runAll();
        }} else throw e;
      }}
      double ts = (System.currentTimeMillis() - t0) / 1000.0;
      try {{ dof = model.sol("sol1").getU().length; }} catch (Throwable e) {{ }}
      double[] t = evalGlobal(model, T_VARS);
      double[] r = evalGlobal(model, R_VARS);
      double[] g = evalGlobal(model, new String[]{{"T_flux", PFX + ".Ttotal", PFX + ".Rtotal"}});
      StringBuilder ts_ = new StringBuilder(), rs_ = new StringBuilder();
      double st = 0;
      for (int i = 0; i < t.length; i++) {{ ts_.append(i > 0 ? ", " : "").append("\\"" + T_KEYS[i] + "\\": " + num(t[i])); st += t[i]; }}
      for (int i = 0; i < r.length; i++) rs_.append(i > 0 ? ", " : "").append("\\"" + R_KEYS[i] + "\\": " + num(r[i]));
      L(String.format(Locale.ROOT, "pol %s: T_port %.4f  R_port %.4f  R+T %.4f  T_flux %.4f  sum eta_T %.4f  (%.0f s, DOF %d)",
                      POLS[ip], g[1], g[2], g[1] + g[2], g[0], st, ts, dof));
      js.append(ip > 0 ? "," : "").append("\\n    \\"" + POLS[ip] + "\\": {{\\"T\\": {{" + ts_ + "}}, \\"R_nonzero\\": {{" + rs_ + "}}, ");
      js.append("\\"T_flux\\": " + num(g[0]) + ", \\"T_port\\": " + num(g[1]) + ", \\"R_port\\": " + num(g[2]) + ", ");
      js.append("\\"sum_eta_T\\": " + num(st) + ", \\"solve_time_s\\": " + num(ts) + "}}");
      writeResults(js.toString() + "\\n  }},\\n  \\"dof\\": " + dof + ",\\n  \\"complete\\": " + (ip == POLS.length - 1) + warnJson() + "\\n}}\\n");
    }}
  }}

  static String warnJson() {{
    StringBuilder sb = new StringBuilder(",\\n  \\"warnings\\": [");
    for (int i = 0; i < warns.size(); i++) sb.append(i > 0 ? ", " : "").append("\\"" + warns.get(i).replace("\\\\", "/").replace("\\"", "'") + "\\"");
    return sb.append("]").toString();
  }}

  static void writeResults(String s) {{
    try {{
      PrintWriter o = new PrintWriter(new FileWriter(DIR + "/" + CASE + "_results.json"));
      o.print(s);
      o.close();
    }} catch (Exception e) {{ W("could not write the results file", e); }}
  }}

  public static void main(String[] args) {{
    run();
  }}
}}""")
    path.write_text("\n".join(J) + "\n")


# ------------------------------------------------------------------------------------------------------ main
def pol_vector(p, theta, phi):
    """Unit E transverse to k for the pipeline input polarization 'x' or 'y' (projection of x^ or y^)."""
    k = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
    e = np.array([1.0, 0.0, 0.0]) if p == "x" else np.array([0.0, 1.0, 0.0])
    e = e - k * (e @ k)
    return e / np.linalg.norm(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--shift-nm", type=float, default=0.0, help="exact edge shift before export (+ dilates solid)")
    ap.add_argument("--pols", nargs="+", default=["x", "y"], choices=["x", "y"])
    ap.add_argument("--wavelength-nm", type=float, help="default: first entry of wavelengths_nm")
    ap.add_argument("--ram-gb", type=float, default=16.0, help="RAM of the machine that runs COMSOL")
    ap.add_argument("--levels", type=int, default=2, help="mesh levels (2 gives a mesh-convergence check)")
    ap.add_argument("--hpat-nm", type=float, nargs="+", help="element size in the patterned layer per level (default 60 45)")
    ap.add_argument("--n-per-lambda", type=float, default=8.0,
                    help="elements per wavelength in the substrate and superstrate (lambda/(N n)); 5 was too coarse for the field analysis")
    ap.add_argument("--sigma-nm", type=float, help="smoothing of freeform contours (default 1 px)")
    ap.add_argument("--min-agreement", type=float, default=0.999)
    ap.add_argument("--no-dump", action="store_true", help="do not log the port/Floquet property dump")
    ap.add_argument("--compile", action="store_true", help="also compile the classes (runs run_comsol.sh --build-only)")
    a = ap.parse_args()

    run = Path(a.run)
    cfg, rho, rho_name = load_run(run)
    Lx, Ly = (float(v) for v in cfg["period_nm"])
    b = rho > 0.5
    nx, ny = b.shape
    dx, dy = Lx / nx, Ly / ny
    if a.shift_nm:
        from ptopt.morph import shift_edges
        b = shift_edges(b, a.shift_nm, dx, dy)
    lam = a.wavelength_nm or float((cfg.get("wavelengths_nm") or [cfg.get("wavelength_nm")])[0])
    lay = cfg["layer"]
    specs = {"sub": cfg.get("substrate", {"n": 1.0}), "sup": cfg.get("superstrate", {"n": 1.0}),
             "core": lay["core"], "bg": lay.get("background", {"n": 1.0})}
    n, k = {}, {}
    for key, sp in specs.items():
        n[key], k[key] = nk(sp, lam)
    illum = cfg.get("illumination", "from_substrate")
    theta, phi = math.radians(cfg.get("inc_ang_deg", 0.0)), math.radians(cfg.get("azi_ang_deg", 0.0))
    n_in, n_out = (n["sub"], n["sup"]) if illum == "from_substrate" else (n["sup"], n["sub"])
    kx0, ky0 = n_in * math.sin(theta) * math.cos(phi), n_in * math.sin(theta) * math.sin(phi)
    L = (Lx, Ly)

    # translation in x to an empty column, then geometry
    y_invariant = bool((b == b[:, :1]).all())
    sigma_px = (a.sigma_nm / min(dx, dy)) if a.sigma_nm is not None else 1.0
    notes = []
    sh = empty_column_shift(b)
    sy = 0
    if not y_invariant:
        from scipy.ndimage import gaussian_filter
        fs = gaussian_filter(b.astype(float), sigma_px, mode="wrap")
        if sh is None:
            sh, sc = best_cut(fs, 0)
            notes.append(f"no empty x-column: the x faces cut the pattern at the most transversal column (margin {sc:.2f})")
        sy, sc = best_cut(fs, 1)
        if sc < 0.1:
            notes.append(f"the y faces cut the pattern close to the 0.5 level (margin {sc:.2f})")
    elif sh is None:
        sh = 0
        notes.append("no empty x-column: the x faces cut the pattern")
    b = np.roll(np.roll(b, -sh, axis=0), -sy, axis=1)
    shift_x_nm, shift_y_nm = sh * dx, sy * dy
    if y_invariant:
        polys = rings_1d(b, dx, Ly)
        xc = (np.arange(nx) + 0.5) * dx
        raster = np.zeros(nx, bool)
        for poly in polys:
            x0, x1 = poly[0][1][0][0], poly[0][1][1][0]
            raster |= (xc > x0) & (xc < x1)
        agree, kind = float((raster == b[:, 0]).mean()), "1D (y-invariant): exact rectangles"
    else:
        polys, agree = rings_2d(b, dx, dy, Lx, Ly, sigma_px, simplify_nm=0.16 * min(dx, dy))
        kind = f"freeform: 0.5 contours, smoothing {sigma_px:.2f} px"
    msgs = snap_faces(polys, Lx, Ly)
    nverts = sum(len(r) for p in polys for _, r in p)
    print(f"[comsol_export] {run.name}: {rho_name} {nx}x{ny}, {kind}; cell shifted by {sh} px ({shift_x_nm:.1f} nm) in x, "
          f"{sy} px ({shift_y_nm:.1f} nm) in y")
    print(f"[comsol_export] {len(polys)} polygons, {nverts} vertices, pixel agreement {agree:.5f}, fill {b.mean():.4f}")
    for m in msgs + notes:
        print("[comsol_export] NOTE", m)
    if agree < a.min_agreement or any("vertices" in m for m in msgs):
        print("[comsol_export] geometry check failed (raise --sigma-nm smoothing down, or check the design)")
        sys.exit(2)

    # stack paddings from the slowest evanescent order on each side
    pads = {}
    for side in ("sub", "sup"):
        ld = decay_length(n[side], lam, L, kx0, ky0)
        d_obs = max(0.3 * lam / n[side], min(ld, lam))
        d_tot = max(0.6 * lam / n[side], min(3 * ld, 2 * lam))
        d_port = max(d_tot - d_obs, 0.2 * lam / n[side])
        pads[f"{side}_obs"], pads[f"{side}_port"], pads[f"{side}_total"], pads[f"{side}_decay"] = d_obs, d_port, d_obs + d_port, ld
    h_lay = float(lay["thickness_nm"])

    # orders
    out_orders = propagating(n_out, lam, L, kx0, ky0)
    in_orders = propagating(n_in, lam, L, kx0, ky0)

    def fallback(n_med):   # union of both axis assignments: the port's (m, n) axes are chosen by COMSOL
        a1 = set(propagating(n_med, lam, L, kx0, ky0))
        a2 = {(q, p) for p, q in a1}
        return sorted((a1 | a2) - {(0, 0)})
    fb = {"in": fallback(n_in), "out": fallback(n_out)}

    hpat = a.hpat_nm or [60.0, 45.0]
    levels = plan_mesh(lam, L, h_lay, n, pads, hpat if a.hpat_nm else hpat[: max(1, a.levels)], a.ram_gb, a.n_per_lambda)
    for i, lv in enumerate(levels, 1):
        print(f"[comsol_export] level L{i}: h_pat {lv['h_pat_nm']} nm, substrate {lv['h_sub_nm']} nm, superstrate "
              f"{lv['h_sup_nm']} nm -> ~{lv['dof_est'] / 1e6:.2f} M DOF, {lv['solver']}, ~{lv['ram_est_gb']} GB "
              f"{'ok' if lv['fits'] else 'DOES NOT FIT ' + str(a.ram_gb) + ' GB'}")
    if not any(lv["fits"] for lv in levels):
        print("[comsol_export] no mesh level fits the RAM budget: use a coarser --hpat-nm or a larger machine")
        sys.exit(3)

    out = run / "comsol"
    out.mkdir(exist_ok=True)
    case = re.sub(r"\W", "_", run.name) + (f"_s{a.shift_nm:+.0f}".replace("+", "p").replace("-", "m") if a.shift_nm else "")
    vx, vy = pol_vector("x", theta, phi), pol_vector("y", theta, phi)
    Ea = [f"(1-pol)*({vx[i]:.6g})+pol*({vy[i]:.6g})" for i in range(3)]
    P = {"Lx": Lx, "Ly": Ly, "h_lay": h_lay, "lam": lam, "illum": illum, "shift_nm": shift_x_nm, "n": n, "k": k,
         "theta": math.degrees(theta), "phi": math.degrees(phi), "Ea": Ea, "shift_y_nm": shift_y_nm,
         "d_obs_sub": pads["sub_obs"], "d_port_sub": pads["sub_port"], "d_obs_sup": pads["sup_obs"], "d_port_sup": pads["sup_port"]}
    classes = []
    for i, lv in enumerate(levels, 1):
        if not lv["fits"]:
            continue
        cls = f"MS_{case}_L{i}"
        write_java(out / f"{cls}.java", cls, case, str(run), P, polys, lv, fb, out_orders, in_orders, a.pols, not a.no_dump)
        classes.append(cls)
        print(f"[comsol_export] wrote {out / (cls + '.java')}")
    plan = {"run": str(run), "case": case, "design": rho_name, "edge_shift_nm": a.shift_nm, "wavelength_nm": lam,
            "illumination": illum, "inc_deg": math.degrees(theta), "azi_deg": math.degrees(phi), "pols": a.pols,
            "geometry": {"kind": kind, "x_shift_px": int(sh), "x_shift_nm": shift_x_nm, "y_shift_px": int(sy), "y_shift_nm": shift_y_nm, "polygons": len(polys),
                         "vertices": nverts, "pixel_agreement": agree, "notes": msgs + notes},
            "index": {kk: [n[kk], k[kk]] for kk in n}, "paddings_nm": {kk: round(v, 1) for kk, v in pads.items()},
            "orders_T": [f"{m},{q}" for m, q in out_orders], "orders_R": [f"{m},{q}" for m, q in in_orders],
            "levels": [dict(lv, cls=f"MS_{case}_L{i}") for i, lv in enumerate(levels, 1)], "ram_gb": a.ram_gb}
    (out / f"{case}_plan.json").write_text(json.dumps(plan, indent=2))
    if a.compile:
        subprocess.call(["bash", str(HERE / "run_comsol.sh"), "--build-only", str(run)])
    else:
        print(f"[comsol_export] next: bash validation/comsol/run_comsol.sh --build-only {run}   (compile, open in the GUI)")
        print(f"[comsol_export]   or: bash validation/comsol/run_comsol.sh {run}                (compile + solve)")


if __name__ == "__main__":
    main()
