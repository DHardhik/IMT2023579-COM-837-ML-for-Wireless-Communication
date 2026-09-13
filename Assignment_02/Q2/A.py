import numpy as np
import pandas as pd

df = pd.read_csv('qam16_dataset.csv')

# Instantaneous Amplitude (r)
df['r'] = np.sqrt(df['rx_I']**2 + df['rx_Q']**2)

# Instantaneous Phase (theta)
df['theta'] = np.arctan2(df['rx_Q'], df['rx_I'])

df.to_csv('qam16_dataset_with_features.csv', index=False)