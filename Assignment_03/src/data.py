"""Loading the MATLAB-generated ground-truth channels and splitting them.

gen_channels.m saves Hr, Hi as single [72 x 14 x N] in a -v7.3 .mat file
(HDF5). h5py reads MATLAB arrays with the axes reversed, i.e. (N, 14, 72),
so we transpose back to (N, 72, 14) = (frame, subcarrier, symbol).
"""
import os
import numpy as np

from . import config as C


def load_channels(model: str, data_dir: str = None) -> np.ndarray:
    """Return the perfect channels H as complex64 array of shape (N, NS, ND)."""
    path = os.path.join(data_dir or C.DATA_DIR, C.CHANNEL_FILES[model])
    try:
        import h5py  # MATLAB -v7.3 files are HDF5
        with h5py.File(path, "r") as f:
            hr = np.asarray(f["Hr"], dtype=np.float32).transpose(0, 2, 1)
            hi = np.asarray(f["Hi"], dtype=np.float32).transpose(0, 2, 1)
    except (OSError, ImportError):
        from scipy.io import loadmat  # fallback for non-v7.3 files
        m = loadmat(path)
        hr = np.moveaxis(m["Hr"].astype(np.float32), 2, 0)
        hi = np.moveaxis(m["Hi"].astype(np.float32), 2, 0)
    H = (hr + 1j * hi).astype(np.complex64)
    assert H.shape[1:] == (C.NS, C.ND), f"unexpected shape {H.shape}"
    return H


def split_indices(n_total: int, seed: int = C.SPLIT_SEED):
    """Fixed random train / val / test split (32000 / 4000 / 4000 in the paper)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_tr = min(C.N_TRAIN, int(0.8 * n_total))
    n_va = min(C.N_VAL, int(0.1 * n_total))
    return {"train": perm[:n_tr],
            "val": perm[n_tr:n_tr + n_va],
            "test": perm[n_tr + n_va:n_tr + n_va + C.N_TEST]}


def load_split(model: str, data_dir: str = None):
    """Convenience: returns dict {'train','val','test'} -> H arrays."""
    H = load_channels(model, data_dir)
    idx = split_indices(len(H))
    return {k: H[v] for k, v in idx.items()}
