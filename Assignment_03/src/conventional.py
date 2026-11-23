"""Conventional channel estimators: LS + 2-D interpolation, and MMSE.

LS + interpolation
    scipy's linear 2-D interpolation (Delaunay, griddata) of the scattered
    pilot estimates is LINEAR in the pilot values, so it can be written as a
    fixed matrix W (NS*ND x Np):  H_interp = W h_LS.  We build W once by
    interpolating unit vectors and then apply it to all frames at once.
    Points outside the convex hull of the pilots use nearest-neighbour.
    The same interpolated image is the INPUT of the SRCNN stage (as in SRCNN).

MMSE (paper eqs. 3-5), with unit pilots (x x^H = I):
    A = R_hp (R_pp + sigma^2 I)^-1,   h_MMSE = A h_LS
    * "ideal" MMSE : R = E[h h^H] from the true channels (training set,
                     32000 frames -> essentially exact statistics)
    * "estimated" MMSE : R estimated from a small number of NOISY
                     LS-interpolated frames (what a receiver could do).
"""
import numpy as np
from scipy.interpolate import griddata

from . import config as C
from .pilots import pilot_positions, pilot_flat_index, noise_var


# ------------------------------------------------------------ interpolation
_W_CACHE = {}


def interp_matrix(n_pilots: int, method: str = "linear") -> np.ndarray:
    """Build W (NS*ND x Np) such that vec(H_interp) = W @ h_pilots."""
    key = (n_pilots, method)
    if key in _W_CACHE:
        return _W_CACHE[key]
    sc, sy = pilot_positions(n_pilots)
    pts = np.stack([sc, sy], 1).astype(float)
    gi, gj = np.meshgrid(np.arange(C.NS), np.arange(C.ND), indexing="ij")
    grid = np.stack([gi.ravel(), gj.ravel()], 1).astype(float)
    W = np.zeros((C.NS * C.ND, n_pilots))
    for k in range(n_pilots):
        e = np.zeros(n_pilots); e[k] = 1.0
        v = griddata(pts, e, grid, method=method)
        vn = griddata(pts, e, grid, method="nearest")
        W[:, k] = np.where(np.isnan(v), vn, v)
    _W_CACHE[key] = W
    return W


def ls_interp(h_ls: np.ndarray, n_pilots: int) -> np.ndarray:
    """LS + 2-D linear interpolation. h_ls (N, Np) -> (N, NS, ND)."""
    W = interp_matrix(n_pilots)
    return (h_ls @ W.T).reshape(len(h_ls), C.NS, C.ND).astype(np.complex64)


# ------------------------------------------------------------ MMSE
def correlation(H: np.ndarray) -> np.ndarray:
    """Sample channel correlation R = E[h h^H] over frames, h = vec(H)."""
    h = H.reshape(len(H), -1).astype(np.complex128)
    return (h.T @ h.conj()) / len(h)


def mmse_filter(R: np.ndarray, n_pilots: int, snr_db: float) -> np.ndarray:
    """A_MMSE = R_hp (R_pp + sigma^2 I)^-1   (paper eq. 5 with x = 1)."""
    p = pilot_flat_index(n_pilots)
    R_hp = R[:, p]
    R_pp = R[np.ix_(p, p)]
    s2 = noise_var(snr_db)
    # solve instead of inverting: A^T = (R_pp + s2 I)^-T R_hp^T
    return np.linalg.solve((R_pp + s2 * np.eye(len(p))).T, R_hp.T).T


def mmse_estimate(h_ls: np.ndarray, A: np.ndarray) -> np.ndarray:
    return (h_ls @ A.T).reshape(len(h_ls), C.NS, C.ND).astype(np.complex64)


def estimated_correlation(H_ls_interp: np.ndarray, snr_db: float, n_pilots: int) -> np.ndarray:
    """Correlation estimated from noisy LS-interpolated frames (practical MMSE).

    H_ls_interp = W (h_p + z)  ->  E[.] = W R_pp W^T + sigma^2 W W^T.
    We subtract the known noise term sigma^2 W W^T; what remains is the
    correlation of the *interpolated* channel, which is smoother than the
    true one (interpolation bias) and noisy (few frames) -> mismatch vs ideal.
    """
    R = correlation(H_ls_interp)
    W = interp_matrix(n_pilots)
    R -= noise_var(snr_db) * (W @ W.T)
    return R


# ------------------------------------------------------------ metric
def mse(H_hat: np.ndarray, H: np.ndarray) -> float:
    """Average per-element MSE  E|H_hat - H|^2  over all frames and REs."""
    return float(np.mean(np.abs(H_hat - H) ** 2))
