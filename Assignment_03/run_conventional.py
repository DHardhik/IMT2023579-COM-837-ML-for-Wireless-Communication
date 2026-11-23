"""STEP 2 - Task 1 visualisations + Task 2 conventional channel estimation.

Produces (in results/):
  fig_groundtruth_<model>.png      real / imag channel images  (mandatory plot 1)
  fig_pilot_patterns.png           lattice pilot patterns for every density
  fig_ls_example_<model>.png       truth vs LS-interp vs MMSE at 12 dB
  conventional_<model>.json        MSE vs SNR (48 pilots) and MSE vs #pilots
  fig_conv_mse_snr.png             quick look at the baselines

Usage:  python run_conventional.py [--models VehA SUI5] [--est_frames 1000]
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.data import load_split
from src.pilots import ls_at_pilots, pilot_positions, LATTICES
from src.conventional import (ls_interp, correlation, mmse_filter, mmse_estimate,
                              estimated_correlation, mse)


def plot_groundtruth(H, model, path):
    """Normalised real / imaginary channel images of one frame (like paper Fig. 1)."""
    h = H[0]
    fig, ax = plt.subplots(1, 2, figsize=(7, 4.2))
    for a, part, name in zip(ax, [h.real, h.imag], ["Real", "Imag"]):
        img = (part - part.min()) / (part.max() - part.min() + 1e-12)   # [0,1]
        im = a.imshow(img, aspect="auto", origin="lower", cmap="viridis")
        a.set_title(f"{model}: {name}(H) (normalised)")
        a.set_xlabel("OFDM symbol"); a.set_ylabel("Subcarrier")
        fig.colorbar(im, ax=a)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_pilots(path):
    fig, ax = plt.subplots(1, len(LATTICES), figsize=(3 * len(LATTICES), 4.5))
    for a, n in zip(ax, LATTICES):
        grid = np.zeros((C.NS, C.ND))
        sc, sy = pilot_positions(n)
        grid[sc, sy] = 1
        a.imshow(grid, aspect="auto", origin="lower", cmap="Greys")
        a.set_title(f"{n} pilots"); a.set_xlabel("OFDM symbol"); a.set_ylabel("Subcarrier")
    fig.suptitle("Lattice (diamond) pilot patterns, 72 x 14 grid")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_example(H, H_ls, H_mmse, model, path):
    fig, ax = plt.subplots(1, 3, figsize=(9, 4.2))
    vmin, vmax = H[0].real.min(), H[0].real.max()
    for a, img, t in zip(ax, [H[0], H_ls[0], H_mmse[0]],
                         ["Ground truth", "LS + linear interp.", "Ideal MMSE"]):
        a.imshow(img.real, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        a.set_title(t); a.set_xlabel("OFDM symbol")
    ax[0].set_ylabel("Subcarrier")
    fig.suptitle(f"{model}, Real(H), 48 pilots, SNR = 12 dB")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate(model, est_frames):
    t0 = time.time()
    S = load_split(model)
    Htr, Hte = S["train"], S["test"]
    print(f"[{model}] train {Htr.shape}, test {Hte.shape}")

    plot_groundtruth(Hte, model, os.path.join(C.RESULTS_DIR, f"fig_groundtruth_{model}.png"))
    R_ideal = correlation(Htr)                 # channel statistics, 1008 x 1008
    res = {"snr_db": C.SNR_LIST_DB, "pilots": C.MAIN_PILOTS,
           "LS": [], "MMSE_ideal": [], "MMSE_est": []}

    # ---- MSE vs SNR, 48 pilots ------------------------------------------
    for snr in C.SNR_LIST_DB:
        h_ls = ls_at_pilots(Hte, C.MAIN_PILOTS, snr)
        H_ls = ls_interp(h_ls, C.MAIN_PILOTS)
        H_mi = mmse_estimate(h_ls, mmse_filter(R_ideal, C.MAIN_PILOTS, snr))
        # practical MMSE: statistics from a few noisy training frames
        h_tr = ls_at_pilots(Htr[:est_frames], C.MAIN_PILOTS, snr, seed=99 + snr)
        R_est = estimated_correlation(ls_interp(h_tr, C.MAIN_PILOTS), snr, C.MAIN_PILOTS)
        H_me = mmse_estimate(h_ls, mmse_filter(R_est, C.MAIN_PILOTS, snr))
        res["LS"].append(mse(H_ls, Hte))
        res["MMSE_ideal"].append(mse(H_mi, Hte))
        res["MMSE_est"].append(mse(H_me, Hte))
        print(f"  SNR {snr:2d} dB  LS {res['LS'][-1]:.2e}  MMSE-ideal {res['MMSE_ideal'][-1]:.2e}"
              f"  MMSE-est {res['MMSE_est'][-1]:.2e}")
    # example images at 12 dB
    h_ls = ls_at_pilots(Hte[:1], C.MAIN_PILOTS, 12)
    plot_example(Hte[:1], ls_interp(h_ls, C.MAIN_PILOTS),
                 mmse_estimate(h_ls, mmse_filter(R_ideal, C.MAIN_PILOTS, 12)),
                 model, os.path.join(C.RESULTS_DIR, f"fig_ls_example_{model}.png"))

    # ---- MSE vs number of pilots at fixed SNR ----------------------------
    snr = C.PILOT_SWEEP_SNR_DB
    res["pilot_sweep"] = {"snr_db": snr, "pilots": C.PILOT_LIST, "LS": [], "MMSE_ideal": []}
    for n in C.PILOT_LIST:
        h_ls = ls_at_pilots(Hte, n, snr)
        res["pilot_sweep"]["LS"].append(mse(ls_interp(h_ls, n), Hte))
        res["pilot_sweep"]["MMSE_ideal"].append(
            mse(mmse_estimate(h_ls, mmse_filter(R_ideal, n, snr)), Hte))
    print(f"  pilot sweep @ {snr} dB:", res["pilot_sweep"])

    with open(os.path.join(C.RESULTS_DIR, f"conventional_{model}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{model}] done in {time.time() - t0:.0f} s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["VehA", "SUI5"])
    ap.add_argument("--est_frames", type=int, default=1000,
                    help="noisy frames used to estimate R for 'estimated MMSE'")
    args = ap.parse_args()
    C.ensure_dirs()
    plot_pilots(os.path.join(C.RESULTS_DIR, "fig_pilot_patterns.png"))

    fig, ax = plt.subplots(1, len(args.models), figsize=(6 * len(args.models), 4.5), squeeze=False)
    for a, m in zip(ax[0], args.models):
        r = evaluate(m, args.est_frames)
        for k, mk in [("LS", "o-"), ("MMSE_est", "s-"), ("MMSE_ideal", "^-")]:
            a.semilogy(r["snr_db"], r[k], mk, label=k.replace("_", " "))
        a.set_title(f"{m}, {C.MAIN_PILOTS} pilots"); a.set_xlabel("SNR (dB)"); a.set_ylabel("MSE")
        a.grid(True, which="both", alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(C.RESULTS_DIR, "fig_conv_mse_snr.png"), dpi=150)


if __name__ == "__main__":
    main()
