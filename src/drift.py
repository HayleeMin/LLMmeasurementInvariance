"""Vector preparation and drift measures (protocol §3)."""
import numpy as np


def prepare(vecs, center):
    """L2-normalize span vectors then subtract the centering mean. vecs: (n, dim); center: (dim,)."""
    v = vecs / (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-12)
    return v - center


def _cos_dist(a, b):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - an @ bn.T


def mean_within(u, groups=None):
    """Mean pairwise cosine distance within a set, excluding self-pairs (and same-group pairs if groups given)."""
    d = _cos_dist(u, u)
    mask = ~np.eye(len(u), dtype=bool)
    if groups is not None:
        g = np.asarray(groups)
        mask &= g[:, None] != g[None, :]
    return d[mask].mean() if mask.any() else np.nan


def apd(u0, u1):
    """Average pairwise cosine distance between two periods."""
    return _cos_dist(u0, u1).mean()


def usage_drift(u0, u1, groups0=None, groups1=None):
    """D_t = APD(U0, Ut) - 0.5 * [within(U0) + within(Ut)]."""
    return apd(u0, u1) - 0.5 * (mean_within(u0, groups0) + mean_within(u1, groups1))
