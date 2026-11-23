"""Building network inputs / targets as images.

input  X : LS estimates at the pilots (noisy, at a given SNR), linearly
           interpolated to the full 72 x 14 grid  (the "low-resolution" image)
target Y : perfect channel H (the "high-resolution" image)

Complex grids are split into two real images (real part, imaginary part),
stacked along the batch axis:  (N, 72, 14) complex -> (2N, 1, 72, 14) float.
Both are normalised by one global scale (std of the training targets), so
the networks see unit-variance images; the scale is stored in the checkpoint
and undone before computing MSE, so reported MSEs are in channel units.

Noise seeds: train / val use their own seeds, while the TEST inputs use the
same seed as run_conventional.py, i.e. ChannelNet, LS and MMSE are compared
on exactly the same noise realisations.
"""
import numpy as np

from . import config as C
from .pilots import ls_at_pilots
from .conventional import ls_interp

SEED_OFFSET = {"train": 100000, "val": 200000}


def noisy_input(H, n_pilots, snr_db, split):
    """LS + interpolation input for a split ('train' / 'val' / 'test')."""
    seed = None if split == "test" else C.NOISE_SEED + SEED_OFFSET[split] + int(10 * snr_db)
    return ls_interp(ls_at_pilots(H, n_pilots, snr_db, seed=seed), n_pilots)


def to_images(Hc: np.ndarray) -> np.ndarray:
    """(N, NS, ND) complex -> (2N, 1, NS, ND) float32  [real ; imag]."""
    return np.concatenate([Hc.real, Hc.imag], 0)[:, None].astype(np.float32)


def from_images(I: np.ndarray) -> np.ndarray:
    """Inverse of to_images: (2N, 1, NS, ND) -> (N, NS, ND) complex."""
    n = len(I) // 2
    return (I[:n, 0] + 1j * I[n:, 0]).astype(np.complex64)
