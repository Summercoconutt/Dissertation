"""
Voter Clustering Pipeline

Dependencies:
- pandas
- numpy
- scikit-learn
- matplotlib
- plotly (optional, for radar chart)

Run: python voter_clustering.py
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Optional: for radar chart
try:
    import plotly.express as px
    plotly_available = True
except ImportError:
    plotly_available = False

# 1. Load Data
csv_path = 'Framework/voter_base.csv'
df = pd.read_csv(csv_path)

# 2. Select features for clustering (exclude non-numeric and identifier columns)
features = [
    'total_votes', 'avg_voting_power', '%_for_votes', '%_against_votes',
    '%_abstain_votes', '%_aligned_with_majority', 'is_whale_ratio'
]
X = df[features].fillna(0)

# 3. Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Optimal K Selection
inertia = []
silhouette = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    if k > 1:
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette.append(score)
        print(f"Silhouette score for k={k}: {score:.3f}")

# Plot Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Curve')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.tight_layout()
plt.show()

# 5. Clustering (Set your optimal K here)
optimal_k = 3  # <-- Set this manually after inspecting the plots/scores
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 6. Cluster Interpretation
cluster_summary = df.groupby('cluster')[features].mean()
print("\nCluster feature means:\n", cluster_summary)

# Save cluster feature means as a CSV table
cluster_summary.to_csv('Framework/cluster_feature_means.csv')

# Save cluster feature means as a styled PNG table (if dataframe_image is available)
try:
    import dataframe_image as dfi
    styled = cluster_summary.style.format(precision=3).set_caption('Cluster Feature Means')
    dfi.export(styled, 'Framework/cluster_feature_means_table.png')
    print('Cluster feature means table saved as Framework/cluster_feature_means_table.png')
except ImportError:
    print('dataframe_image not installed: skipping PNG table export. Table saved as CSV.')

# Optional: Radar chart for each cluster
if plotly_available:
    for c in cluster_summary.index:
        fig = px.line_polar(
            r=cluster_summary.loc[c].values,
            theta=features,
            line_close=True,
            title=f'Cluster {c} Profile'
        )
        fig.show()
else:
    print("Plotly not installed: skipping radar charts.")

# 7. PCA Visualization (Optional)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]
pca_eigenvectors = pca.components_
# plot the pca eigenvectors with a bar plot for the 2 first components
plt.figure(figsize=(8, 6))
plt.bar(range(len(pca_eigenvectors[0])), pca_eigenvectors[0])
plt.title('PCA Eigenvector 1')
plt.xlabel('Feature')
plt.ylabel('Value')
# use features as x labels for bars
plt.xticks(range(len(features)), features)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(range(len(pca_eigenvectors[1])), pca_eigenvectors[1])
plt.title('PCA Eigenvector 2')
plt.xlabel('Feature')
plt.ylabel('Value')
plt.xticks(range(len(features)), features)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
for c in range(optimal_k):
    plt.scatter(df[df['cluster'] == c]['pca1'], df[df['cluster'] == c]['pca2'], label=f'Cluster {c}', alpha=0.6)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA Scatter Plot of Clusters')
plt.legend()
plt.tight_layout()
plt.show()

# 8. Export Results
df.to_csv('Framework/voter_base_with_clusters.csv', index=False)
print("Exported DataFrame with clusters and PCA coordinates to Framework/voter_base_with_clusters.csv") 