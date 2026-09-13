import numpy as np
import pandas as pd
import itertools

np.random.seed(67)  # reproducibility, as required

# ---- Step 1: Build raw 16-QAM constellation ----
levels = [-3, -1, 1, 3]

constellation_points = []   # will hold (I, Q) tuples, 16 total
symbol_ids = []              # 0..15, one per constellation point

# TODO 1: build all 16 (I, Q) combinations from `levels` using two nested loops
#         (or itertools.product), and assign each one a unique symbol_id (0-15)
for symbol_id, (I_val, Q_val) in enumerate(itertools.product(levels, levels)):
    constellation_points.append((I_val, Q_val))
    symbol_ids.append(symbol_id)

constellation_points = np.array(constellation_points)  # shape (16, 2)

# ---- Step 2: Normalize so average symbol energy = 1 ----
# Energy of a single point = I^2 + Q^2
raw_energies = constellation_points[:, 0]**2 + constellation_points[:, 1]**2

# TODO 2: compute the average energy across all 16 points
avg_energy = raw_energies.mean()

# TODO 3: scale every point by 1/sqrt(avg_energy) so new average energy = 1
scaled_constellation = constellation_points / np.sqrt(avg_energy)

# sanity check (should print ~1.0)
scaled_energies = scaled_constellation[:, 0]**2 + scaled_constellation[:, 1]**2
print("Average energy after scaling:", scaled_energies.mean())

# ---- Step 3: Generate noisy samples for every SNR value ----
SNR_DB_VALUES = [0, 5, 10, 15, 20, 25, 30]
N_SAMPLES_PER_POINT = 200

all_rows = []

for snr_db in SNR_DB_VALUES:

    # TODO 4: convert snr_db to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # TODO 5: derive noise variance PER COMPLEX DIMENSION (i.e., for I and Q separately).
    #         signal energy = 1 (whole symbol, I^2+Q^2 combined).
    #         SNR = signal_power / noise_power  ->  noise_power (total, I+Q) = 1/snr_linear
    #         That total noise power splits equally between I and Q dimensions.
    noise_var_per_dim = 1 / (2 * snr_linear)

    noise_std_per_dim = np.sqrt(noise_var_per_dim)

    for symbol_id, (I_clean, Q_clean) in zip(symbol_ids, scaled_constellation):

        # TODO 6: generate N_SAMPLES_PER_POINT noise draws for I and Q separately
        #         using np.random.randn(...) scaled by noise_std_per_dim
        noise_I = np.random.randn(N_SAMPLES_PER_POINT) * noise_std_per_dim
        noise_Q = np.random.randn(N_SAMPLES_PER_POINT) * noise_std_per_dim

        rx_I = I_clean + noise_I
        rx_Q = Q_clean + noise_Q

        for i in range(N_SAMPLES_PER_POINT):
            all_rows.append({
                'snr_db': snr_db,
                'symbol_id': symbol_id,
                'I_clean': I_clean,
                'Q_clean': Q_clean,
                'rx_I': rx_I[i],
                'rx_Q': rx_Q[i],
            })

df = pd.DataFrame(all_rows)
print(df.shape)   # sanity check: should be 7 SNRs * 16 points * 200 samples = 22400 rows
print(df.head())

df.to_csv('qam16_dataset.csv', index=False)