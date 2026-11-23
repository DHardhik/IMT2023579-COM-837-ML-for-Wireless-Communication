"""Central configuration for COM837 Assignment 3 (ChannelNet).

Every script imports its constants from here so that the experiment is
fully described in one place (also copy these values into the report).
"""
import os

# ---------------------------------------------------------------- paths
# On Kaggle the data lives in /kaggle/input/<dataset>, outputs in /kaggle/working.
# Locally, override with the environment variables MLWC_DATA / MLWC_OUT.
DATA_DIR = os.environ.get("MLWC_DATA", "/kaggle/input/mlwc-a3-channels")
OUT_DIR = os.environ.get("MLWC_OUT", "/kaggle/working")
RESULTS_DIR = os.path.join(OUT_DIR, "results")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

CHANNEL_FILES = {"VehA": "H_VehA.mat", "SUI5": "H_SUI5.mat"}

# ---------------------------------------------------------------- OFDM grid
NS = 72          # subcarriers (frequency axis, image height)
ND = 14          # OFDM symbols (time axis, image width)

# ---------------------------------------------------------------- data split (paper)
N_TRAIN, N_VAL, N_TEST = 32000, 4000, 4000
SPLIT_SEED = 0   # permutation seed for train/val/test split

# ---------------------------------------------------------------- experiments
SNR_LIST_DB = [0, 5, 10, 15, 20, 25, 30]   # coarse SNR sweep for run_conventional.py
                                           # (Fig. 3/4 use the 2.5 dB grid SNR_FINE in test.py)
TRAIN_SNRS_DB = [12, 22]                   # "deep low-SNR" / "deep high-SNR"
MAIN_PILOTS = 48                           # main comparison
PILOT_LIST = [24, 36, 48, 72]              # pilot-density sweep (Fig. 5)
PILOT_SWEEP_SNR_DB = 12                    # fixed SNR for the pilot sweep
NOISE_SEED = 1234                          # base seed for AWGN generation


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
