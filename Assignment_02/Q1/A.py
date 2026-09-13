import numpy as np
import pandas as pd

def extract_features(row):
    # build arrays of amplitude and delay for the 6 taps
    h_real = np.array([row[f'h_real_{i}'] for i in range(6)])
    h_imag = np.array([row[f'h_imag_{i}'] for i in range(6)])
    tau    = np.array([row[f'tau_{i}']    for i in range(6)])

    amp = np.abs(h_real + 1j * h_imag)  # |h_l|

    # mean and std of amplitude (used by kurtosis, skewness, K-factor)
    mu = np.mean(amp)
    sigma = np.sqrt(np.mean((amp - mu) ** 2))

    # Kurtosis (Eq. 2)
    # Note: the paper (and assignment) call this "kurtosis of received power,"
          # but Eq. (2) in the paper is explicitly defined on amplitude |h(t)|, not power.
          # I followed the paper's formula exactly, using amplitude.
    kurtosis = np.mean((amp - mu) ** 4) / (sigma ** 4)

    # Skewness (Eq. 5)
    skewness = np.mean((amp - mu) ** 3) / (sigma ** 3)

    # Rising time (Eq. 6)
    l_star = np.argmax(amp)                     
    rising_time = tau[l_star] - np.min(tau)      
    # RMS delay spread (Eq. 7-8)
    power = amp ** 2                              # |h_l|^2
    tau_m = np.sum(tau * power) / np.sum(power)   # power-weighted mean delay
    tau_rms = np.sqrt(np.sum(((tau - tau_m) ** 2) * power) / np.sum(power))

    # Rician K-factor (Eq. 9)
    k_factor = (np.max(amp) ** 2) / (2 * sigma ** 2)

    return pd.Series({
        'kurtosis': kurtosis,
        'skewness': skewness,
        'rising_time': rising_time,
        'tau_rms': tau_rms,
        'k_factor': k_factor
    })

# Apply row-wise
df = pd.read_csv(r'C:\Users\Dhava\Desktop\Coding\MLWL\Assignment_02\NLOS_LOS_Classification_main\Dataset\los_nlos_dataset.csv')
feature_cols = df.apply(extract_features, axis=1)
df = pd.concat([df, feature_cols], axis=1)
print(df.columns.tolist())
print(df.columns.duplicated().sum())
print("concat done successfully!!")
df.to_csv('los_nlos_dataset_with_features.csv', index=False)
print("Saved Successfully!!")