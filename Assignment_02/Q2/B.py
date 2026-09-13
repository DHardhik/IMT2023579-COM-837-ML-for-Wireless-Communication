import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('qam16_dataset_with_features.csv')


subset_25 = df[df['snr_db'] == 25].reset_index(drop=True)


X_cartesian = subset_25[['rx_I', 'rx_Q']].values

k_values = range(2, 21)
inertia_list = []
silhouette_list = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_cartesian)
    inertia_list.append(kmeans.inertia_)
    sil_score = silhouette_score(X_cartesian, labels)
    silhouette_list.append(sil_score)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14,5))
axes[0].plot(list(k_values), inertia_list, marker='o')
axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia (WCSS)'); axes[0].set_title('Elbow Method')
axes[0].grid(True)

axes[1].plot(list(k_values), silhouette_list, marker='o', color='orange')
axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette Coefficient'); axes[1].set_title('Silhouette Method')
axes[1].grid(True)
plt.tight_layout()
plt.savefig('Q2b_elbow_silhouette.png')
plt.show()


kmeans_16 = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42)
cluster_labels_16 = kmeans_16.fit_predict(X_cartesian)
centroids_16 = kmeans_16.cluster_centers_   # shape (16, 2)

plt.figure(figsize=(8,8))
plt.scatter(X_cartesian[:,0], X_cartesian[:,1], c=cluster_labels_16, cmap='tab20', s=10, alpha=0.6)
plt.scatter(centroids_16[:,0], centroids_16[:,1], c='black', marker='X', s=200, label='Centroids')
plt.xlabel('rx_I'); plt.ylabel('rx_Q')
plt.title('K-means (K=16) on 16-QAM at SNR=25dB')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig('Q2b_kmeans16_scatter.png')
plt.show()


def cluster_purity(cluster_labels, true_labels):
    df_temp = pd.DataFrame({'cluster': cluster_labels, 'true_label': true_labels})
    majority_label_per_cluster = df_temp.groupby('cluster')['true_label'].agg(
        lambda x: x.mode().iloc[0]
    )

    predicted_via_majority = df_temp['cluster'].map(majority_label_per_cluster)
    correct = predicted_via_majority == df_temp['true_label']

    purity = correct.mean()
    return purity

true_labels = subset_25['symbol_id'].values

purity_set1 = cluster_purity(cluster_labels_16, true_labels)

X_polar = subset_25[['r', 'theta']].values

kmeans_polar = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42)
cluster_labels_polar = kmeans_polar.fit_predict(X_polar)
purity_set2 = cluster_purity(cluster_labels_polar, true_labels)

X_all = subset_25[['rx_I', 'rx_Q', 'r', 'theta']].values

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)
kmeans_all = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42)
cluster_labels_all = kmeans_all.fit_predict(X_all_scaled)
purity_set3 = cluster_purity(cluster_labels_all, true_labels)

print(f"Purity - Feature Set 1 (Cartesian):        {purity_set1:.4f}")
print(f"Purity - Feature Set 2 (Polar):             {purity_set2:.4f}")
print(f"Purity - Feature Set 3 (All 4, scaled):     {purity_set3:.4f}")