import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

df = pd.read_csv('los_nlos_dataset_with_features.csv')

feature_names = ['kurtosis', 'skewness', 'rising_time', 'tau_rms', 'k_factor']

classifier_configs = {
    'SVM-1 (Kurtosis)':    ['kurtosis'],
    'SVM-2 (Skewness)':    ['skewness'],
    'SVM-3 (Rising time)': ['rising_time'],
    'SVM-4 (RMS delay)':   ['tau_rms'],
    'SVM-5 (K-factor)':    ['k_factor'],
    'SVM-6 (All 5)':       feature_names,
}

df[feature_names] = df[feature_names].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_names).reset_index(drop=True)

snr_values = sorted(df['snr_db'].unique())

results = {name: [] for name in classifier_configs}

for snr_val in snr_values:

    subset = df[df['snr_db'] == snr_val]

    for clf_name, feat_cols in classifier_configs.items():
        X = subset[feat_cols].values
        y = subset['label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        clf = SVC(kernel='rbf', C=1)
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean() * 100
        results[clf_name].append(accuracy)

plt.figure(figsize=(10,6))
for clf_name, acc_list in results.items():
    plt.plot(snr_values, acc_list, marker='o', label=clf_name)
plt.xlabel('SNR (dB)')
plt.ylabel('Classification Accuracy (%)')
plt.title('SVM Accuracy vs SNR — single features vs combined')
plt.legend()
plt.grid(True)
plt.show()