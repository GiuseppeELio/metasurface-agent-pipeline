"""Vector angular-spectrum propagation of a transverse field in a homogeneous medium, and focusing metrics.

Shared by metalens.py (field from the locally periodic model) and validation/lens_post.py (field recorded by Meep
just above the lens), so that both are analysed with exactly the same propagator and definitions.

Conventions: E_z and H follow from transversality of every plane-wave component,
    E_z = -(k_x E_x + k_y E_y) / k_z,   H = n (k_hat x E)   (units of 1/eta0),
so that S_z = 0.5 Re(E x H*)_z = 0.5 n |E|^2 for a normally propagating plane wave. Evanescent components are dropped.
"""
import math

import numpy as np


def fwhm(x, y):
    """Full width at half maximum of the contiguous peak containing the global maximum (linear interpolation)."""
    y = np.asarray(y, float); i = int(np.argmax(y)); h = y[i] / 2
    l = i
    while l > 0 and y[l] >= h:
        l -= 1
    r = i
    while r < len(y) - 1 and y[r] >= h:
        r += 1
    if y[l] >= h or y[r] >= h:
        return float("nan")                         # peak not resolved inside the window
    xl = x[l] + (h - y[l]) * (x[l + 1] - x[l]) / (y[l + 1] - y[l])
    xr = x[r - 1] + (h - y[r - 1]) * (x[r] - x[r - 1]) / (y[r] - y[r - 1])
    return float(xr - xl), float(xl), float(xr)


class VectorASM:
    """Only propagating plane waves are kept, as compressed 1-D arrays: cheap even on fine, padded grids."""

    def __init__(self, Ex, Ey, dx, wavelength, n_out=1.0, pad=2.0):
        """Ex, Ey: complex transverse field on a square grid (indexed [x, y]) centred on the optical axis."""
        n0 = Ex.shape[0]
        N = int(2 ** math.ceil(math.log2(max(n0 * pad, n0))))
        self.N, self.dx, self.n = N, dx, n_out
        self.k = 2 * math.pi * n_out / wavelength
        o = (N - n0) // 2
        ex = np.zeros((N, N), complex); ey = np.zeros((N, N), complex)
        ex[o:o + n0, o:o + Ex.shape[1]] = Ex
        ey[o:o + n0, o:o + Ey.shape[1]] = Ey
        self.x = (np.arange(N) - N // 2) * dx
        kx = 2 * math.pi * np.fft.fftfreq(N, d=dx)
        KX, KY = np.meshgrid(kx, kx, indexing="ij")
        prop = (KX ** 2 + KY ** 2) < self.k ** 2
        self.ix, self.iy = np.nonzero(prop)
        self.kx, self.ky = KX[prop], KY[prop]
        self.kz = np.sqrt(self.k ** 2 - self.kx ** 2 - self.ky ** 2)
        self.Ax = np.fft.fft2(np.fft.ifftshift(ex))[prop]
        self.Ay = np.fft.fft2(np.fft.ifftshift(ey))[prop]
        self.Az = -(self.kx * self.Ax + self.ky * self.Ay) / self.kz

    def _spectra(self, z):
        ph = np.exp(1j * self.kz * z)
        return self.Ax * ph, self.Ay * ph, self.Az * ph

    def _full(self, a):
        F = np.zeros((self.N, self.N), complex)
        F[self.ix, self.iy] = a
        return np.fft.fftshift(np.fft.ifft2(F))

    def plane(self, z):
        """|E|^2 and S_z on the full plane at distance z."""
        ax, ay, az = self._spectra(z)
        k, n = self.k, self.n
        hx = n * (self.ky * az - self.kz * ay) / k
        hy = n * (self.kz * ax - self.kx * az) / k
        Ex, Ey, Ez, Hx, Hy = (self._full(c) for c in (ax, ay, az, hx, hy))
        I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
        Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
        return I, Sz

    def power(self, z=0.0):
        """Propagating power through a plane (Parseval; independent of z)."""
        a2 = np.abs(self.Ax) ** 2 + np.abs(self.Ay) ** 2 + np.abs(self.Az) ** 2
        # S_z of each plane wave = 0.5 n (kz/k) |E|^2 ; ifft2 normalization gives dx^2 / N^2 per component
        return float(0.5 * self.n * np.sum(a2 * self.kz / self.k) * self.dx ** 2 / self.N ** 2)

    def axial(self, zs):
        """|E|^2 on the optical axis versus z."""
        N2 = self.N ** 2
        ph = np.exp(1j * np.outer(zs, self.kz))
        return (np.abs(ph @ self.Ax) ** 2 + np.abs(ph @ self.Ay) ** 2 + np.abs(ph @ self.Az) ** 2) / N2 ** 2

    def section(self, zs, axis="x", half_width=None):
        """|E|^2 in the xz (axis='x', y = 0) or yz (axis='y', x = 0) plane: array [len(zs), n_points]."""
        sel = np.ones(self.N, bool) if half_width is None else np.abs(self.x) <= half_width
        idx = self.ix if axis == "x" else self.iy
        rows = []
        for z in zs:
            I = 0
            for A in self._spectra(z):
                line = np.bincount(idx, weights=A.real, minlength=self.N) + 1j * np.bincount(idx, weights=A.imag, minlength=self.N)
                I = I + np.abs(np.fft.fftshift(np.fft.ifft(line)) / self.N) ** 2
            rows.append(I[sel])
        return self.x[sel], np.array(rows)

    def metrics(self, zs, P_inc, disk_factor=3.0):
        """Focus position, depth of focus (axial FWHM), lateral FWHM x/y and focusing efficiency."""
        ax_I = self.axial(zs)
        i = int(np.argmax(ax_I)); zf = float(zs[i])
        dof = fwhm(zs, ax_I)
        I, Sz = self.plane(zf)
        c = self.N // 2
        fx, fy = fwhm(self.x, I[:, c]), fwhm(self.x, I[c, :])
        fxv = fx[0] if isinstance(fx, tuple) else fx
        fyv = fy[0] if isinstance(fy, tuple) else fy
        XX, YY = np.meshgrid(self.x, self.x, indexing="ij")
        disk = np.hypot(XX, YY) <= disk_factor * 0.5 * (fxv + fyv)
        return {"z_focus_um": zf, "peak_I": float(I[c, c]),
                "dof_um": dof[0] if isinstance(dof, tuple) else dof,
                "dof_range_um": list(dof[1:]) if isinstance(dof, tuple) else None,
                "fwhm_x_um": fxv, "fwhm_y_um": fyv,
                "focusing_efficiency": float(Sz[disk].sum() * self.dx ** 2 / P_inc),
                "transmitted_fraction": self.power(0.0) / P_inc,
                "_axial": ax_I, "_I": I}
