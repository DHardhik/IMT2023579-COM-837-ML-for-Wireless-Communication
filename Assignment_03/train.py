"""Training script for ChannelNet (two-stage, paper Sec. III-C).

Stage 'sr' : train SRCNN,  loss C1 = mean ||Z - H||^2,  Z = f_S(X)
Stage 'dn' : FREEZE the trained SRCNN, train DnCNN on its output,
             loss C2 = mean ||H_hat - H||^2,  H_hat = f_R(Z)
Networks are trained separately for each SNR (paper: 12 dB = "deep
low-SNR", 22 dB = "deep high-SNR"), channel model and pilot count.

Examples
  python train.py --model VehA --snr 12 --stage sr
  python train.py --model VehA --snr 12 --stage dn      (needs the sr checkpoint)
  python train.py --model VehA --snr 12 --stage both
  quick smoke test:  add  --max_train 2000 --epochs_sr 2 --epochs_dn 1

Outputs
  checkpoints/<tag>_sr.pt / <tag>_dn.pt    weights + scale + settings + history
  results/loss_<tag>_<stage>.png / .json   training & validation loss curves
  with <tag> = <model>_snr<snr>_p<pilots>
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.data import load_split
from src.dataset import noisy_input, to_images
from src.models import SRCNN, DnCNN, count_params


def set_seed(seed):
    """Reproducibility: python / numpy / torch / cuDNN."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(net, X, bs=2048):
    net.eval()
    return torch.cat([net(X[i:i + bs]) for i in range(0, len(X), bs)])


@torch.no_grad()
def eval_loss(net, X, Y, bs=2048):
    net.eval()
    tot = 0.0
    for i in range(0, len(X), bs):
        tot += torch.sum((net(X[i:i + bs]) - Y[i:i + bs]) ** 2).item()
    return tot / Y.numel()


def fit(net, Xtr, Ytr, Xva, Yva, epochs, lr, bs, patience, use_amp, tag):
    """Adam + MSE, mini-batches drawn from GPU-resident tensors.
    Keeps the weights with the best validation loss (early stopping)."""
    dev = Xtr.device
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(2, patience // 3))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.MSELoss()
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    best, best_state, wait = float("inf"), None, 0
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        perm = torch.randperm(n, device=dev)
        run, cnt = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x, y = Xtr[idx], Ytr[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item() * len(idx); cnt += len(idx)
        tr, va = run / cnt, eval_loss(net, Xva, Yva)
        sched.step(va)
        hist["epoch"].append(ep); hist["train_loss"].append(tr)
        hist["val_loss"].append(va); hist["lr"].append(opt.param_groups[0]["lr"])
        flag = ""
        if va < best:
            best, wait, flag = va, 0, "*"
            best_state = copy.deepcopy(net.state_dict())
        else:
            wait += 1
        print(f"[{tag}] ep {ep:3d}  train {tr:.4e}  val {va:.4e}  "
              f"lr {hist['lr'][-1]:.1e}  {time.time() - t0:.1f}s {flag}", flush=True)
        if wait >= patience:
            print(f"[{tag}] early stop (no val improvement for {patience} epochs)")
            break
    net.load_state_dict(best_state)
    return hist, best


def plot_hist(hist, title, path):
    plt.figure(figsize=(5.5, 4))
    plt.semilogy(hist["epoch"], hist["train_loss"], label="train")
    plt.semilogy(hist["epoch"], hist["val_loss"], label="validation")
    plt.xlabel("epoch"); plt.ylabel("MSE loss (normalised)"); plt.title(title)
    plt.grid(True, which="both", alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="VehA", choices=list(C.CHANNEL_FILES))
    ap.add_argument("--snr", type=float, default=12)
    ap.add_argument("--pilots", type=int, default=C.MAIN_PILOTS)
    ap.add_argument("--stage", default="both", choices=["sr", "dn", "both"])
    ap.add_argument("--epochs_sr", type=int, default=100)
    ap.add_argument("--epochs_dn", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)       # paper
    ap.add_argument("--batch", type=int, default=128)        # paper
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max_train", type=int, default=None, help="subset for quick tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    args = ap.parse_args()

    C.ensure_dirs()
    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda" and not args.no_amp
    snr_tag = f"{args.snr:g}"
    tag = f"{args.model}_snr{snr_tag}_p{args.pilots}"
    print(f"device {dev}, GPUs {torch.cuda.device_count()}, AMP {use_amp}, tag {tag}")

    # ------------------------------------------------ data
    S = load_split(args.model)
    Htr, Hva = S["train"], S["val"]
    if args.max_train:
        Htr, Hva = Htr[:args.max_train], Hva[:max(200, args.max_train // 8)]
    Xtr_c = noisy_input(Htr, args.pilots, args.snr, "train")    # LS + interp (noisy)
    Xva_c = noisy_input(Hva, args.pilots, args.snr, "val")
    Ytr_np, Yva_np = to_images(Htr), to_images(Hva)
    scale = float(Ytr_np.std())                                 # global normalisation
    T = lambda a: torch.from_numpy(a / scale).float().to(dev)
    Xtr, Xva = T(to_images(Xtr_c)), T(to_images(Xva_c))
    Ytr, Yva = T(Ytr_np), T(Yva_np)
    print(f"train images {tuple(Xtr.shape)}, val {tuple(Xva.shape)}, scale {scale:.4f}")
    base = {"model": args.model, "snr_db": args.snr, "pilots": args.pilots,
            "scale": scale, "seed": args.seed, "lr": args.lr, "batch": args.batch,
            "n_train_frames": len(Htr), "n_val_frames": len(Hva)}
    sr_path = os.path.join(C.CKPT_DIR, f"{tag}_sr.pt")
    dn_path = os.path.join(C.CKPT_DIR, f"{tag}_dn.pt")

    # ------------------------------------------------ stage 1: SRCNN
    if args.stage in ("sr", "both"):
        sr = SRCNN().to(dev)
        print(f"SRCNN parameters: {count_params(sr)}")
        print(f"input (LS+interp) val loss before training: {eval_loss(nn.Identity(), Xva, Yva):.4e}")
        hist, best = fit(sr, Xtr, Ytr, Xva, Yva, args.epochs_sr, args.lr, args.batch,
                         args.patience, use_amp, tag + "_sr")
        torch.save({"state_dict": sr.state_dict(), "hist": hist, "best_val": best,
                    **base, "epochs": args.epochs_sr}, sr_path)
        plot_hist(hist, f"SRCNN - {args.model}, {snr_tag} dB, {args.pilots} pilots",
                  os.path.join(C.RESULTS_DIR, f"loss_{tag}_sr.png"))
        json.dump(hist, open(os.path.join(C.RESULTS_DIR, f"loss_{tag}_sr.json"), "w"))
        # with scale = std of the real/imag images (E|H|^2 = 2 scale^2), the
        # normalised per-pixel MSE equals the complex-channel MSE E|H_hat - H|^2
        print(f"SR best val MSE = {best * 2 * scale ** 2:.4e} (complex-channel MSE)")

    # ------------------------------------------------ stage 2: DnCNN on frozen SRCNN
    if args.stage in ("dn", "both"):
        sr = SRCNN().to(dev)
        sr.load_state_dict(torch.load(sr_path, map_location=dev, weights_only=False)["state_dict"])
        for p in sr.parameters():
            p.requires_grad = False                            # freeze SR weights
        Ztr, Zva = predict(sr, Xtr), predict(sr, Xva)          # SR outputs = DnCNN inputs
        del Xtr, Xva
        set_seed(args.seed)
        dn = DnCNN().to(dev)
        print(f"DnCNN parameters: {count_params(dn)}")
        net = nn.DataParallel(dn) if torch.cuda.device_count() > 1 else dn
        print(f"SR-only val loss: {eval_loss(nn.Identity(), Zva, Yva):.4e}")
        hist, best = fit(net, Ztr, Ytr, Zva, Yva, args.epochs_dn, args.lr, args.batch,
                         args.patience, use_amp, tag + "_dn")
        torch.save({"state_dict": dn.state_dict(), "hist": hist, "best_val": best,
                    **base, "epochs": args.epochs_dn}, dn_path)
        plot_hist(hist, f"DnCNN - {args.model}, {snr_tag} dB, {args.pilots} pilots",
                  os.path.join(C.RESULTS_DIR, f"loss_{tag}_dn.png"))
        json.dump(hist, open(os.path.join(C.RESULTS_DIR, f"loss_{tag}_dn.json"), "w"))
        print(f"ChannelNet best val MSE = {best * 2 * scale ** 2:.4e} (complex-channel MSE)")


if __name__ == "__main__":
    main()
