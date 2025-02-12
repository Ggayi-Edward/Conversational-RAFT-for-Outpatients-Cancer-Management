import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path


# Define file path
file_path = Path("../../data/corpus/cleaned_corpus.csv")

# Ensure the dataset exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: File not found at {file_path}")

# Load the dataset
df = pd.read_csv(file_path)

# Ensure 'Cleaned_Summary' column exists
if "Cleaned_Summary" not in df.columns:
    raise ValueError("Error: 'Cleaned_Summary' column is missing in the dataset.")

# Drop missing or empty summaries
df = df.dropna(subset=["Cleaned_Summary"])
df = df[df["Cleaned_Summary"].str.strip() != ""]

if df.empty:
    raise ValueError("Error: No valid summaries found after preprocessing!")

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["Cleaned_Summary"])

# Standardize the features
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

# Determine the number of PCA components
n_samples, n_features = X_scaled.shape
n_components = min(50, n_samples, n_features)  # Ensures PCA doesn't exceed data dimensions

# Apply PCA for dimensionality reduction
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled.toarray())  # Convert to dense only when necessary

# Dynamically adjust t-SNE perplexity
perplexity_value = min(30, max(5, n_samples // 5))  # Keeps within a reasonable range

# Apply t-SNE for further reduction to 2D
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, init="pca")
X_tsne = tsne.fit_transform(X_pca)

# Plot the t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.arange(len(X_tsne)), cmap="coolwarm", alpha=0.7, edgecolors="k")
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)
plt.colorbar(label="Document Index")
plt.grid(True)

# Save the plot
output_path = Path("../../outputs/plots/tsne_visualization.png")

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"✅ t-SNE visualization saved successfully at:\n📁 {output_path}")

# Optionally display the plot
plt.show()
