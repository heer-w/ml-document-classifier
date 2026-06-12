import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import MaxAbsScaler

BASE_DIR     = os.path.join(os.path.dirname(__file__), '..')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
OUTPUTS_DIR  = os.path.join(BASE_DIR, 'outputs')
SPLITS_DIR   = os.path.join(BASE_DIR, 'data', 'splits')

print("📂 Loading features...")
X_train = sp.load_npz(os.path.join(FEATURES_DIR, 'tfidf_train.npz'))
X_val   = sp.load_npz(os.path.join(FEATURES_DIR, 'tfidf_val.npz'))
X_test  = sp.load_npz(os.path.join(FEATURES_DIR, 'tfidf_test.npz'))
X_all   = sp.vstack([X_train, X_val, X_test])

y_train = pd.read_csv(os.path.join(FEATURES_DIR, 'y_train.csv'))['class_label'].values
y_val   = pd.read_csv(os.path.join(FEATURES_DIR, 'y_val.csv'))['class_label'].values
y_test  = pd.read_csv(os.path.join(FEATURES_DIR, 'y_test.csv'))['class_label'].values
y_all   = np.concatenate([y_train, y_val, y_test])

scaler  = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_all)

classes = sorted(list(set(y_all)))
colors  = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
class_to_color = dict(zip(classes, colors))

# ── 1. t-SNE Visualization ─────────────────────────────────────
print("\n🔵 Running t-SNE (this takes ~1 min)...")
tsne   = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d   = tsne.fit_transform(X_scaled.toarray())
print("   t-SNE done!")

fig, ax = plt.subplots(figsize=(12, 8))
for cls, color in zip(classes, colors):
    mask = y_all == cls
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=color, label=cls, alpha=0.8, s=100, edgecolors='black', linewidth=0.5)
ax.set_title('t-SNE Visualization of Document Classes', fontsize=16, fontweight='bold')
ax.set_xlabel('t-SNE Component 1', fontsize=12)
ax.set_ylabel('t-SNE Component 2', fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'tsne_visualization.png'))
plt.close()
print("   Saved tsne_visualization.png")

# ── 2. KMeans Clustering ───────────────────────────────────────
print("\n🟡 Running KMeans with 5 clusters...")
kmeans     = KMeans(n_clusters=5, random_state=42, n_init=10)
km_labels  = kmeans.fit_predict(X_scaled)

ari_kmeans = adjusted_rand_score(y_all, km_labels)
sil_kmeans = silhouette_score(X_scaled, km_labels)
print(f"   Adjusted Rand Index : {ari_kmeans:.4f}")
print(f"   Silhouette Score    : {sil_kmeans:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot true labels
for cls, color in zip(classes, colors):
    mask = y_all == cls
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=color, label=cls, alpha=0.8, s=80, edgecolors='black', linewidth=0.5)
axes[0].set_title('True Labels', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Plot KMeans clusters
km_colors = ['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6']
for i in range(5):
    mask = km_labels == i
    axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=km_colors[i], label=f'Cluster {i}',
                    alpha=0.8, s=80, edgecolors='black', linewidth=0.5)
axes[1].set_title(f'KMeans Clusters (ARI={ari_kmeans:.3f})', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.suptitle('True Labels vs KMeans Clusters', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'kmeans_clusters.png'))
plt.close()
print("   Saved kmeans_clusters.png")

# ── 3. Elbow Method ────────────────────────────────────────────
print("\n📊 Running Elbow Method...")
inertias = []
k_range  = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_range, inertias, marker='o', color='#FF6B6B', linewidth=2)
ax.axvline(x=5, color='blue', linestyle='--', label='k=5 (true classes)')
ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Clusters (k)', fontsize=12)
ax.set_ylabel('Inertia', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'elbow_method.png'))
plt.close()
print("   Saved elbow_method.png")

# ── 4. DBSCAN ──────────────────────────────────────────────────
print("\n🔴 Running DBSCAN...")
dbscan     = DBSCAN(eps=0.5, min_samples=3)
db_labels  = dbscan.fit_predict(X_2d)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise    = list(db_labels).count(-1)
print(f"   Clusters found : {n_clusters}")
print(f"   Noise points   : {n_noise}")

fig, ax = plt.subplots(figsize=(10, 7))
unique_labels = set(db_labels)
db_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
for label, color in zip(unique_labels, db_colors):
    if label == -1:
        color = 'black'
        marker = 'x'
        size = 50
    else:
        marker = 'o'
        size = 80
    mask = db_labels == label
    lbl  = f'Noise' if label == -1 else f'Cluster {label}'
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[color], label=lbl, marker=marker,
               alpha=0.8, s=size, edgecolors='black', linewidth=0.5)
ax.set_title(f'DBSCAN Clustering ({n_clusters} clusters, {n_noise} noise points)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Component 1', fontsize=12)
ax.set_ylabel('t-SNE Component 2', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'dbscan_clusters.png'))
plt.close()
print("   Saved dbscan_clusters.png")

# ── 5. Summary ─────────────────────────────────────────────────
clustering_results = {
    'kmeans': {
        'n_clusters':           5,
        'adjusted_rand_index':  round(ari_kmeans, 4),
        'silhouette_score':     round(sil_kmeans, 4),
    },
    'dbscan': {
        'n_clusters_found': n_clusters,
        'noise_points':     n_noise,
    }
}
with open(os.path.join(OUTPUTS_DIR, 'clustering_results.json'), 'w') as f:
    json.dump(clustering_results, f, indent=2)

print("\n" + "="*50)
print("        CLUSTERING SUMMARY")
print("="*50)
print(f"  KMeans ARI         : {ari_kmeans:.4f}")
print(f"  KMeans Silhouette  : {sil_kmeans:.4f}")
print(f"  DBSCAN Clusters    : {n_clusters}")
print(f"  DBSCAN Noise Pts   : {n_noise}")
print("="*50)
print("\n✅ All clustering outputs saved to outputs/")
