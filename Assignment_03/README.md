# COM837 Assignment 3 – Deep Learning-Based Channel Estimation (ChannelNet)

Hardhik Dhavala (IMT2023579)

Reproduction of ChannelNet (Soltani et al., IEEE Commun. Lett. 2019): SRCNN super-resolution followed by DnCNN denoising, compared with LS + 2-D interpolation and MMSE on VehA and SUI5 channels.

## Folder structure
```
matlab/gen_channels.m     Channel generation with the Vienna 5G LL Simulator (FastFading)
src/config.py             All settings: grid, split, SNRs, pilot counts, seeds, paths
src/data.py               Load .mat channels, fixed 32k/4k/4k train/val/test split
src/pilots.py             Lattice pilot patterns, AWGN, LS at pilots
src/conventional.py       LS + 2-D interpolation, ideal / estimated MMSE, MSE
src/dataset.py            Complex channel <-> real/imag images, network inputs
src/models.py             SRCNN and DnCNN
run_conventional.py       Task 1 plots + Task 2 baselines
train.py                  Training script (two-stage: --stage sr / dn / both)
test.py                   Testing script (loads checkpoints only) -> Fig. 3/4/5, tables
channelnet_full_run.ipynb Kaggle notebook that ran the whole pipeline (with its printed outputs)
requirements.txt          Python packages
checkpoints/              20 trained models: <model>_snr<SNR>_p<pilots>_{sr,dn}.pt
results/                  All plots, loss curves (png + json), final_results.json, final_table.csv
report/                   Technical report PDF + LaTeX source (main.tex, Images/)
```
Configuration: every experiment setting (grid size, split sizes, SNR lists, training SNRs, pilot
counts, seeds, paths) is in `src/config.py`. Training hyper-parameters are command-line arguments of
`train.py`, and their values are recorded inside each checkpoint.

Mandatory plots in `results/`:
| Required plot | File(s) |
|---|---|
| 1. Ground-truth real/imag channel images | `fig_groundtruth_VehA.png`, `fig_groundtruth_SUI5.png` |
| 2. Train/val loss vs epoch (SR and DN stages) | `loss_<model>_snr<SNR>_p<pilots>_{sr,dn}.png` |
| 3. MSE vs SNR (paper Fig. 3, 4) | `fig3_VehA_mse_snr.png`, `fig4_SUI5_mse_snr.png` |
| 4. VehA / SUI5, deep high-SNR case | `fig_highsnr_VehA.png`, `fig_highsnr_SUI5.png` |
| 5. MSE vs number of pilots (paper Fig. 5) | `fig5_mse_pilots.png` |
| Extra | `fig_pilot_patterns.png`, `fig_ls_example_*.png`, `fig_conv_mse_snr.png`, `fig_ablation_*.png`, `fig_estimates_*.png` |

## Environment
- Channel generation: MATLAB R2025b + Vienna 5G Link-Level Simulator (supplied `5GLLSimulator.zip`).
- Everything else: Python 3.10+, PyTorch 2.x (CUDA), NumPy, SciPy, h5py, Matplotlib.
  This was run on Kaggle (GPU T4 x2), where all packages are preinstalled. To run locally:
  `pip install -r requirements.txt`

## Exact execution steps
1. **Generate the channels (MATLAB).** Unzip the simulator into the project root, so that
   `5GLLSimulator/+Channel/FastFading.m` exists. `matlab/gen_channels.m` is already set to `N_FRAMES = 40000` (the full dataset). Run:
   ```matlab
   run('matlab/gen_channels.m')
   ```
   This writes `data/H_VehA.mat` and `data/H_SUI5.mat` (~290 MB each, seeds 1 and 2).
   The data files are not included in the submission because of their size; re-running the script
   regenerates them exactly.
2. **Point Python to the data and output folders:**
   ```bash
   export MLWC_DATA=./data      # folder with H_VehA.mat, H_SUI5.mat
   export MLWC_OUT=.            # results/ and checkpoints/ are created here
   ```
   On Kaggle, these variables are set by the first cell of `channelnet_full_run.ipynb`.
3. **Run the conventional baselines and Task 1 figures:**
   ```bash
   python run_conventional.py
   ```
4. **Train ChannelNet** (48 pilots; 12 dB = low-SNR network, 22 dB = high-SNR network):
   ```bash
   for m in VehA SUI5; do for s in 12 22; do
     python train.py --model $m --snr $s --pilots 48 --stage sr
     python train.py --model $m --snr $s --pilots 48 --stage dn --epochs_dn 40
   done; done
   # pilot-density sweep (Fig. 5)
   for m in VehA SUI5; do for p in 24 36 72; do
     python train.py --model $m --snr 12 --pilots $p --stage both --epochs_dn 20
   done; done
   ```
5. **Test using the saved checkpoints only:**
   ```bash
   python test.py
   ```
   To reproduce every figure and table from the provided checkpoints, only steps 1–3 and step 5 are needed.

## Reproducibility
- The split uses seed 0. Noise seeds are derived from base seed 1234 and the SNR, with separate seeds for train/val/test. Test noise is identical for all methods.
- Training sets the Python, NumPy and PyTorch seeds (`--seed 0`) and makes cuDNN deterministic.
- Training settings: Adam, learning rate 1e-3, batch 128, ReduceLROnPlateau, early stopping (patience 10).
  Max epochs: SRCNN 100; DnCNN 40 (48 pilots) and 20 (pilot sweep).
- Runtime on Kaggle (2 x T4): about 3 hours for the full pipeline.
