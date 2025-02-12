import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from pathlib import Path


# Define paths
file_path = Path("../../data/corpus/cleaned_corpus.csv")
output_plot = Path("../../outputs/plots/kmeans_tsne_visualization.png")
# Ensure the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: File not found at {file_path}")

# Load dataset
df = pd.read_csv(file_path)

# Ensure 'Cleaned_Summary' column exists
if 'Cleaned_Summary' not in df.columns:
    raise ValueError("Error: The column 'Cleaned_Summary' is missing from the dataset.")

# Ensure sufficient data for clustering
n_samples = df.shape[0]
if n_samples < 2:
    raise ValueError("Error: Dataset has too few rows for clustering.")

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')  # Limit feature size for efficiency
X = vectorizer.fit_transform(df['Cleaned_Summary'])

# Standardize features
scaler = StandardScaler(with_mean=False)  # Keep mean=False for sparse matrices
X_scaled = scaler.fit_transform(X)

# Determine number of clusters (KMeans)
n_clusters = min(5, n_samples)  # Avoid exceeding sample size
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# Apply PCA (reduce to min(10, n_samples, n_features))
pca_components = min(10, X_scaled.shape[0], X_scaled.shape[1])
pca = PCA(n_components=pca_components)
X_pca = pca.fit_transform(X_scaled.toarray())

# Adjust t-SNE perplexity dynamically
perplexity_value = min(30, max(2, n_samples - 1))  # Avoid error for small datasets

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster')

# Save plot
os.makedirs(os.path.dirname(output_plot), exist_ok=True)
plt.savefig(output_plot, dpi=300)
plt.show()

print(f"✅ t-SNE visualization saved at: {output_plot}")

# Calculate Silhouette Score (only if clusters > 1)
if n_clusters > 1:
    silhouette_avg = silhouette_score(X_scaled, y_kmeans)
    print(f"🔹 Silhouette Score: {silhouette_avg:.4f}")
else:
    print("🔹 Silhouette score not computed (only one cluster).")
