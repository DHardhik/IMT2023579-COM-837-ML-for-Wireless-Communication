"""Testing / evaluation script (Task 6). Uses ONLY saved checkpoints.

For every channel model:
  * MSE vs SNR (48 pilots) of LS, estimated MMSE, ideal MMSE and ChannelNet
    trained at 12 dB ("deep low-SNR") and 22 dB ("deep high-SNR"),
    also the SR-only output (ablation) and the SNR-switched ChannelNet
        -> fig3_VehA_mse_snr.png, fig4_SUI5_mse_snr.png   (paper Fig. 3 / 4)
        -> fig_highsnr_<model>.png   (deep high-SNR case vs baselines)
  * MSE vs number of pilots at 12 dB (needs checkpoints for each pilot count)
        -> fig5_mse_pilots.png                            (paper Fig. 5)
  * example estimated channel images     -> fig_estimates_<model>.png
  * all numbers -> results/final_results.json and results/final_table.csv

All methods use the same test frames and the same noise realisations.

Usage: python test.py [--models VehA SUI5] [--switch_db 17]
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.data import load_split
from src.dataset import noisy_input, to_images, from_images
from src.pilots import ls_at_pilots
from src.conventional import (ls_interp, correlation, mmse_filter, mmse_estimate,
                              estimated_correlation, mse)
from src.models import SRCNN, DnCNN

SNR_FINE = list(np.arange(0, 30.1, 2.5))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_net(model, snr, pilots):
    """Returns (srcnn, dncnn_or_None, scale) or None if no checkpoint."""
    tag = f"{model}_snr{snr:g}_p{pilots}"
    sp, dp = [os.path.join(C.CKPT_DIR, f"{tag}_{s}.pt") for s in ("sr", "dn")]
    if not os.path.exists(sp):
        return None
    ck = torch.load(sp, map_location=DEV, weights_only=False)
    sr = SRCNN().to(DEV); sr.load_state_dict(ck["state_dict"]); sr.eval()
    dn = None
    if os.path.exists(dp):
        dn = DnCNN().to(DEV)
        dn.load_state_dict(torch.load(dp, map_location=DEV, weights_only=False)["state_dict"])
        dn.eval()
    return sr, dn, ck["scale"]


@torch.no_grad()
def run_net(nets, X_complex, bs=2048):
    """ChannelNet inference. Returns (H_sr, H_channelnet) complex arrays."""
    sr, dn, scale = nets
    X = torch.from_numpy(to_images(X_complex) / scale).float()
    Z, O = [], []
    for i in range(0, len(X), bs):
        z = sr(X[i:i + bs].to(DEV))
        Z.append(z.cpu())
        O.append((dn(z) if dn is not None else z).cpu())
    Z = torch.cat(Z).numpy() * scale
    O = torch.cat(O).numpy() * scale
    return from_images(Z), from_images(O)


def evaluate_model(model, switch_db, est_frames=1000):
    S = load_split(model)
    Htr, Hte = S["train"], S["test"]
    R = correlation(Htr)
    P = C.MAIN_PILOTS
    nets = {s: load_net(model, s, P) for s in C.TRAIN_SNRS_DB}
    r = {"snr_db": [float(s) for s in SNR_FINE], "LS": [], "MMSE_est": [], "MMSE_ideal": []}
    for s in C.TRAIN_SNRS_DB:
        r[f"SR_{s}dB"], r[f"ChannelNet_{s}dB"] = [], []
    r["ChannelNet_switched"] = []

    for snr in SNR_FINE:
        h_ls = ls_at_pilots(Hte, P, snr)                 # same noise as training script's test
        X = ls_interp(h_ls, P)
        r["LS"].append(mse(X, Hte))
        r["MMSE_ideal"].append(mse(mmse_estimate(h_ls, mmse_filter(R, P, snr)), Hte))
        h_tr = ls_at_pilots(Htr[:est_frames], P, snr, seed=99 + int(10 * snr))
        R_est = estimated_correlation(ls_interp(h_tr, P), snr, P)
        r["MMSE_est"].append(mse(mmse_estimate(h_ls, mmse_filter(R_est, P, snr)), Hte))
        for s, n in nets.items():
            if n is None:
                r[f"SR_{s}dB"].append(None); r[f"ChannelNet_{s}dB"].append(None); continue
            Hs, Hc = run_net(n, X)
            r[f"SR_{s}dB"].append(mse(Hs, Hte)); r[f"ChannelNet_{s}dB"].append(mse(Hc, Hte))
        lo, hi = r[f"ChannelNet_{C.TRAIN_SNRS_DB[0]}dB"][-1], r[f"ChannelNet_{C.TRAIN_SNRS_DB[1]}dB"][-1]
        r["ChannelNet_switched"].append(lo if snr < switch_db else hi)
        print(f"[{model}] SNR {snr:4.1f}: " + "  ".join(
            f"{k} {v[-1]:.2e}" for k, v in r.items() if k != "snr_db" and v[-1] is not None), flush=True)

    # example images at 12 dB
    if nets[C.TRAIN_SNRS_DB[0]] is not None:
        h_ls = ls_at_pilots(Hte[:1], P, 12, seed=7)
        X = ls_interp(h_ls, P)
        Hs, Hc = run_net(nets[C.TRAIN_SNRS_DB[0]], X)
        Hm = mmse_estimate(h_ls, mmse_filter(R, P, 12))
        imgs = [Hte[0], X[0], Hm[0], Hs[0], Hc[0]]
        names = ["Ground truth", "LS + interp.", "Ideal MMSE", "SRCNN only", "ChannelNet"]
        fig, ax = plt.subplots(2, 5, figsize=(13, 6.5))
        for row, part in enumerate(["real", "imag"]):
            ref = getattr(Hte[0], part)
            for a, im, t in zip(ax[row], imgs, names):
                a.imshow(getattr(im, part), aspect="auto", origin="lower",
                         vmin=ref.min(), vmax=ref.max())
                a.set_title(f"{t}\n{part}"); a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{model}, 48 pilots, SNR 12 dB (networks trained at 12 dB)")
        fig.tight_layout(); fig.savefig(os.path.join(C.RESULTS_DIR, f"fig_estimates_{model}.png"), dpi=150)
        plt.close(fig)

    # pilot sweep at 12 dB
    snr = C.PILOT_SWEEP_SNR_DB
    ps = {"snr_db": snr, "pilots": [], "LS": [], "MMSE_ideal": [], "ChannelNet": []}
    for n in C.PILOT_LIST:
        h_ls = ls_at_pilots(Hte, n, snr)
        X = ls_interp(h_ls, n)
        net = load_net(model, snr, n)
        ps["pilots"].append(n)
        ps["LS"].append(mse(X, Hte))
        ps["MMSE_ideal"].append(mse(mmse_estimate(h_ls, mmse_filter(R, n, snr)), Hte))
        ps["ChannelNet"].append(mse(run_net(net, X)[1], Hte) if net else None)
    print(f"[{model}] pilot sweep @ {snr} dB: {ps}")
    r["pilot_sweep"] = ps
    return r


def plot_snr(r, model, keys, title, path):
    style = {"LS": ("o-", "LS + interp."), "MMSE_est": ("s-", "Estimated MMSE"),
             "MMSE_ideal": ("^-", "Ideal MMSE"),
             "ChannelNet_12dB": ("D-", "ChannelNet (deep low-SNR, 12 dB)"),
             "ChannelNet_22dB": ("v-", "ChannelNet (deep high-SNR, 22 dB)"),
             "SR_12dB": ("D:", "SRCNN only (12 dB)"), "SR_22dB": ("v:", "SRCNN only (22 dB)"),
             "ChannelNet_switched": ("*-", "ChannelNet (SNR-switched)")}
    plt.figure(figsize=(6.5, 4.8))
    for k in keys:
        if k in r and all(v is not None for v in r[k]):
            plt.semilogy(r["snr_db"], r[k], style[k][0], label=style[k][1], ms=5)
    for s in C.TRAIN_SNRS_DB:
        plt.axvline(s, color="grey", lw=.6, ls="--")
    plt.xlabel("SNR (dB)"); plt.ylabel("MSE"); plt.title(title)
    plt.grid(True, which="both", alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["VehA", "SUI5"])
    ap.add_argument("--switch_db", type=float, default=17,
                    help="use the 12 dB network below this SNR, the 22 dB network above")
    args = ap.parse_args()
    C.ensure_dirs()
    allres = {}
    for m in args.models:
        allres[m] = r = evaluate_model(m, args.switch_db)
        fig_no = {"VehA": "fig3", "SUI5": "fig4"}.get(m, "fig")
        plot_snr(r, m, ["LS", "MMSE_est", "MMSE_ideal", "ChannelNet_12dB", "ChannelNet_22dB"],
                 f"{m}: channel estimation MSE, 48 pilots",
                 os.path.join(C.RESULTS_DIR, f"{fig_no}_{m}_mse_snr.png"))
        plot_snr(r, m, ["LS", "MMSE_est", "MMSE_ideal", "ChannelNet_22dB"],
                 f"{m}: deep high-SNR (22 dB) vs baselines",
                 os.path.join(C.RESULTS_DIR, f"fig_highsnr_{m}.png"))
        plot_snr(r, m, ["LS", "MMSE_ideal", "SR_12dB", "ChannelNet_12dB", "SR_22dB",
                        "ChannelNet_22dB", "ChannelNet_switched"],
                 f"{m}: SR-only vs full ChannelNet (ablation)",
                 os.path.join(C.RESULTS_DIR, f"fig_ablation_{m}.png"))

    # Fig. 5: MSE vs number of pilots
    fig, ax = plt.subplots(1, len(args.models), figsize=(6 * len(args.models), 4.5), squeeze=False)
    for a, m in zip(ax[0], args.models):
        ps = allres[m]["pilot_sweep"]
        for k, mk, lab in [("LS", "o-", "LS + interp."), ("MMSE_ideal", "^-", "Ideal MMSE"),
                           ("ChannelNet", "D-", "ChannelNet")]:
            pts = [(p, v) for p, v in zip(ps["pilots"], ps[k]) if v is not None]
            if pts:
                a.semilogy(*zip(*pts), mk, label=lab)
        a.set_title(f"{m}: MSE vs #pilots, SNR {ps['snr_db']} dB")
        a.set_xlabel("number of pilots"); a.set_ylabel("MSE"); a.set_xticks(ps["pilots"])
        a.grid(True, which="both", alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(C.RESULTS_DIR, "fig5_mse_pilots.png"), dpi=150)

    with open(os.path.join(C.RESULTS_DIR, "final_results.json"), "w") as f:
        json.dump(allres, f, indent=2)
    with open(os.path.join(C.RESULTS_DIR, "final_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for m, r in allres.items():
            keys = [k for k in r if k not in ("snr_db", "pilot_sweep")]
            w.writerow([m, "SNR_dB"] + keys)
            for i, s in enumerate(r["snr_db"]):
                w.writerow(["", s] + [f"{r[k][i]:.4e}" if r[k][i] is not None else "" for k in keys])
    print("saved:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(C.RESULTS_DIR, "*"))))


if __name__ == "__main__":
    main()
