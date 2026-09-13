import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

df = pd.read_csv('los_nlos_dataset_with_features.csv')
feature_names = ['kurtosis', 'skewness', 'rising_time', 'tau_rms', 'k_factor']
snr_values = sorted(df['snr_db'].unique())


train_eq_test_acc = []
saved_test_sets = {}   

for snr_val in snr_values:
    subset = df[df['snr_db'] == snr_val]
    X = subset[feature_names].values
    y = subset['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    saved_test_sets[snr_val] = (X_test, y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    clf = SVC(kernel='rbf', C=1)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = (y_pred == y_test).mean() * 100
    train_eq_test_acc.append(acc)

# Train ONE model at SNR=25dB only ----
train_25_subset = df[df['snr_db'] == 25]
X_25 = train_25_subset[feature_names].values
y_25 = train_25_subset['label'].values

X_25_train, X_25_test, y_25_train, y_25_test = train_test_split(
    X_25, y_25, test_size=0.2, random_state=42, stratify=y_25
)

scaler_25 = StandardScaler()
scaler_25.fit(X_25_train)  

clf_25 = SVC(kernel='rbf', C=1)
clf_25.fit(scaler_25.transform(X_25_train), y_25_train)

# Evaluate the fixed 25dB model against EVERY SNR's saved test set 
train_at_25_acc = []

for snr_val in snr_values:
    X_test_raw, y_test = saved_test_sets[snr_val]

    X_test_scaled = scaler_25.transform(X_test_raw)
    y_pred = clf_25.predict(X_test_scaled)
    acc = (y_pred == y_test).mean() * 100
    train_at_25_acc.append(acc)

# Plot both together
plt.figure(figsize=(10,6))
plt.plot(snr_values, train_eq_test_acc, marker='o', label='Train = Test SNR')
plt.plot(snr_values, train_at_25_acc, marker='s', label='Train at 25 dB')
plt.xlabel('SNR (dB)')
plt.ylabel('Classification Accuracy (%)')
plt.title('Generalization: Train=Test vs Fixed 25dB Training (SVM-6, all 5 features)')
plt.legend()
plt.grid(True)
plt.show()