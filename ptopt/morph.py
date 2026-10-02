"""Morphology on periodic binary designs: exact edge shifts (erosion/dilation in nm) and feature-size metrics."""
import numpy as np
from scipy import ndimage as nd


def _tile(b):
    return np.tile(b, (3, 3))


def _crop(t, shape):
    nx, ny = shape
    return t[nx:2 * nx, ny:2 * ny]


def shift_edges(b, d_nm, dx, dy):
    """d_nm > 0 dilates solid by d_nm, d_nm < 0 erodes it (Euclidean, periodic)."""
    b = b.astype(bool)
    if d_nm == 0:
        return b
    t = _tile(b)
    if d_nm > 0:   # pixels of void within d of solid become solid
        dist = nd.distance_transform_edt(~t, sampling=(dx, dy))
        return _crop(t | (dist <= d_nm), b.shape)
    dist = nd.distance_transform_edt(t, sampling=(dx, dy))
    return _crop(t & (dist > -d_nm), b.shape)


def lost_fraction(b, width_nm, dx, dy):
    """Fraction of a phase removed by an opening with a disk of diameter width_nm (= features narrower than it)."""
    r = width_nm / 2
    rx, ry = int(np.ceil(r / dx)), int(np.ceil(r / dy))
    y, x = np.ogrid[-rx:rx + 1, -ry:ry + 1]
    k = (x * dy) ** 2 + (y * dx) ** 2 <= r * r
    t = _tile(b.astype(bool))
    o = _crop(nd.binary_opening(t, structure=k), b.shape)
    return float((b & ~o).sum() / max(1, b.sum()))


def feature_report(b, dx, dy, widths=(30, 40, 60, 80)):
    b = b.astype(bool)
    return {f"<{w}nm": {"solid_%": round(100 * lost_fraction(b, w, dx, dy), 2),
                        "void_%": round(100 * lost_fraction(~b, w, dx, dy), 2)} for w in widths}
