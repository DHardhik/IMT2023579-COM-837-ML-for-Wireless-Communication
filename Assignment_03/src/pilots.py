"""Lattice (diamond) pilot patterns, AWGN and LS estimation at the pilots.

System model on the 72 x 14 time-frequency grid (paper eq. 1):
        Y[i,k] = H[i,k] X[i,k] + Z[i,k],     Z ~ CN(0, sigma^2)
Pilots are unit-power (X = 1 at pilot positions), E|H|^2 = 1, hence
        SNR = 1 / sigma^2.
LS estimate at a pilot (paper eq. 2):  h_LS = y_p / x_p = h_p + z_p / x_p.
"""
import numpy as np

from . import config as C

# Lattice patterns: (OFDM symbols carrying pilots, frequency spacing, stagger).
# Pilots of consecutive pilot-symbols are shifted by `stagger` subcarriers,
# giving the diamond-shaped lattice used in LTE. 48 = LTE CRS antenna port 0
# (symbols 0, 4, 7, 11; every 6th subcarrier; shift of 3).
LATTICES = {
    24: ([0, 4, 7, 11], 12, 6),
    36: ([0, 4, 7, 11], 8, 4),
    48: ([0, 4, 7, 11], 6, 3),
    72: ([0, 2, 4, 7, 9, 11], 6, 3),
}


def pilot_positions(n_pilots: int):
    """Return (sub_idx, sym_idx) integer arrays of the pilot positions."""
    syms, spacing, stagger = LATTICES[n_pilots]
    sc, sy = [], []
    for j, s in enumerate(syms):
        offset = (j % 2) * stagger
        f = np.arange(offset, C.NS, spacing)
        sc.extend(f)
        sy.extend([s] * len(f))
    sc, sy = np.array(sc), np.array(sy)
    assert len(sc) == n_pilots, (n_pilots, len(sc))
    return sc, sy


def pilot_flat_index(n_pilots: int) -> np.ndarray:
    """Pilot positions as indices into the vectorised grid H.reshape(NS*ND)."""
    sc, sy = pilot_positions(n_pilots)
    return sc * C.ND + sy


def noise_var(snr_db: float) -> float:
    return 10.0 ** (-snr_db / 10.0)


def ls_at_pilots(H: np.ndarray, n_pilots: int, snr_db: float, seed: int = None) -> np.ndarray:
    """Simulate pilot transmission and return the noisy LS estimates.

    H : (N, NS, ND) perfect channels.  Returns (N, n_pilots) complex64.
    """
    rng = np.random.default_rng(C.NOISE_SEED + int(10 * snr_db) if seed is None else seed)
    p = pilot_flat_index(n_pilots)
    hp = H.reshape(len(H), -1)[:, p]
    s2 = noise_var(snr_db)
    z = np.sqrt(s2 / 2) * (rng.standard_normal(hp.shape) + 1j * rng.standard_normal(hp.shape))
    x = np.ones_like(hp)            # unit-power pilot symbols
    y = hp * x + z                  # received pilots
    return (y / x).astype(np.complex64)   # LS: h_LS = y / x
