# COM 837 — Assignment 2
**ML for Wireless Communication** | Hardhik Dhavala (IMT2023579)

Full report: [`report.pdf`](./report.pdf) <!-- TODO: update path/filename if different -->

## Repository Structure

```
Assignment_02/
├── NLOS_LOS_Classification_main/
│   ├── .gitattributes
│   └── Dataset/
│       ├── generate_dataset.py            # provided (unmodified) dataset generator
│       └── los_nlos_dataset.csv           # provided (unmodified) dataset
│
├── Q1/
│   ├── A.py                     # Part (a): five-feature extraction (kurtosis, skewness, rising time, RMS delay, K-factor)
│   ├── B.py                     # Part (b): per-SNR SVM training (6 classifiers)
│   ├── C.py                     # Part (c): fixed-25dB vs matched-SNR generalization test
│   └── los_nlos_dataset_with_features.csv # output of A.py, used by B.py / C.py
│
├── Q2/
│   ├── generate_dataset.py      # 16-QAM AWGN dataset generator
│   ├── A.py                     # Part (a): amplitude (r) and phase (theta) features
│   ├── Q2_b_and_c.ipynb         # Parts (b) and (c): K-means clustering (run on Kaggle)
│   ├── qam16_dataset.csv                  # output of generate_dataset.py
│   └── qam16_dataset_with_features.csv    # output of A.py
│
└── README.md
```
<!-- TODO: adjust this tree to match your actual repo structure exactly -->

## Important: fix before submitting

- **`Q1/A.py` currently reads from a hardcoded personal Windows path**
  (`C:\Users\Dhava\Desktop\...`). Change this to a relative path
  (e.g. `../NLOS_LOS_Classification_main/Dataset/los_nlos_dataset.csv`) so the
  script runs on any machine, not just the original author's.
- **A leftover `Q2/B.py` exists in the repo** — this was the original local
  script that failed due to the Windows Application Control / scikit-learn DLL
  issue, superseded entirely by `Q2_b_and_c.ipynb` run on Kaggle. Either delete
  it or clearly mark it as superseded so it doesn't look like unexplained,
  unreferenced code.
## How to Run

### Question 1
```bash
cd Q1
python A.py   # generates los_nlos_dataset_with_features.csv
python B.py   # trains 6 SVMs per SNR, saves accuracy plot
python C.py   # generalization experiment, saves comparison plot
```

### Question 2
```bash
cd Q2
python generate_dataset.py   # generates qam16_dataset.csv
python A.py                  # generates qam16_dataset_with_features.csv
```
Parts (b) and (c) are in `Q2_b_and_c.ipynb`. This notebook was run on
**Kaggle** (not locally) due to a Windows Application Control policy blocking
a compiled scikit-learn extension (`_hierarchical_fast.pyd`) needed to import
`KMeans` on the local machine. To reproduce:
1. Upload `qam16_dataset_with_features.csv` as a Kaggle dataset input
2. Update the `pd.read_csv(...)` path in the notebook's first cell to match
3. Run all cells

## Dependencies
```
numpy
pandas
scikit-learn
matplotlib
```

## Reference
C. Huang, A. F. Molisch, R. He, R. Wang, P. Tang, B. Ai, and Z. Zhong,
"Machine Learning-Enabled LOS/NLOS Identification for MIMO Systems in Dynamic
Environments," *IEEE Transactions on Wireless Communications*, vol. 19, no. 6,
pp. 3643–3657, Jun. 2020. doi: 10.1109/TWC.2020.2967726.
