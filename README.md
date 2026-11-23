# IMT2023579 – COM 837: ML for Wireless Communication

Coursework for COM 837 (Machine Learning for Wireless Communication), Hardhik Dhavala (IMT2023579), IIIT Bangalore.

| Folder | Topic |
|---|---|
| [`Assignment_02/`](Assignment_02) | SVM-based LOS/NLOS identification; K-means clustering for 16-QAM demodulation |
| [`Assignment_03/`](Assignment_03) | Deep learning-based OFDM channel estimation (ChannelNet: SRCNN + DnCNN) |

Each assignment folder has its own README with the full setup and run instructions.

---

## Assignment 3 – Deep Learning-Based Channel Estimation (ChannelNet)

This assignment reproduces *Soltani et al., "Deep Learning-Based Channel Estimation", IEEE Commun. Lett., 2019*.
The 72 × 14 OFDM time-frequency channel is treated as an image. The pilot estimate goes through an
**SRCNN** (super-resolution) stage and then a **DnCNN** (denoising) stage, both trained in two stages.
The results are compared with **LS + 2-D interpolation** and with **ideal / estimated MMSE** on the **VehA** and **SUI5** channels.

**Pipeline**
1. **Channels (MATLAB):** `matlab/gen_channels.m` uses the Vienna 5G Link-Level Simulator's `FastFading` class.
   It generates 40,000 VehA and 40,000 SUI5 channels (2.1 GHz, 50 km/h; SUI5 as a custom delay profile).
2. **Baselines:** `run_conventional.py` computes LS + interpolation and ideal / estimated MMSE.
3. **Training:** `train.py` trains SRCNN, then DnCNN on the frozen SRCNN.
   Networks are trained at 12 dB and 22 dB, plus for 24/36/48/72 pilots (20 checkpoints in total).
4. **Testing:** `test.py` evaluates from the saved checkpoints and produces the MSE vs SNR and MSE vs pilots plots and tables.

All settings are in `src/config.py`. The full pipeline ran on Kaggle (2× T4) in about 3 hours.

**Key results (test MSE, 48 pilots)**

| | VehA @ 30 dB | SUI5 @ 25 dB |
|---|---|---|
| LS + interpolation | 2.3e-3 | 1.3e-1 |
| Estimated MMSE | 1.9e-3 | 1.3e-1 |
| **ChannelNet (22 dB network)** | **5.8e-4** | **1.7e-3** |
| Ideal MMSE (bound) | 2.4e-4 | 4.3e-4 |

- **VehA:** ChannelNet matches practical MMSE at low SNR and beats it at high SNR.
- **SUI5:** ChannelNet removes the ~0.13 interpolation error floor of LS and estimated MMSE, giving about 75× lower MSE at 25 dB.
- **Limitation:** each network only generalises near its training SNR. For example, the 22 dB network is worse than LS at 0 dB.
  The report discusses remedies: multi-SNR training, noise-conditioned networks, and SNR-switched model banks.

**Contents:** source code, `README.md` (exact steps), `checkpoints/`, `results/` (all plots + tables),
`report/` (6-page technical report, PDF + LaTeX), and the executed Kaggle notebook.
The `.mat` channel files (~290 MB each) are not included; `gen_channels.m` regenerates them exactly from fixed seeds.
